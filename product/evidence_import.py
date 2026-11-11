"""Preview/apply a reviewed evidence-pointer prototype through the native SDK."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from product import manage


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n"
        )


def checked_local(path: str, expected_hash: str) -> Path:
    target = (manage.OUTPUT / path).resolve(strict=True)
    if not target.is_relative_to(manage.OUTPUT.resolve()):
        raise ValueError("Prototype artifact escapes batch directory")
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Prototype catalog changed")
    return target


async def apply_batch(apply: bool) -> None:
    files = manage.checked_files()
    config = manage.settings()
    manifest = json.loads(manage.MANIFEST.read_text())
    if manifest["workspace"] != config["WORKSPACE"]:
        raise ValueError("Workspace does not match manifest")
    if config["EXTRACT_LLM_MODEL"] != "qwen3.8-flash" or config["MAX_GLEANING"] != "0":
        raise ValueError("Evidence prototype requires Flash initial extraction only")
    if Path(config["WORKING_DIR"]).resolve() != manage.ROOT / "rag_storage":
        raise ValueError("Unexpected native storage root")
    os.environ.update({k: v for k, v in config.items() if v is not None})

    import yaml

    from lightrag import LightRAG, QueryParam
    from lightrag.base import DocStatus
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.llm_roles import RoleLLMConfig
    from lightrag.prompt import (
        PROMPTS,
        resolve_entity_extraction_prompt_profile,
        validate_entity_extraction_prompt_profile_for_mode,
    )
    from lightrag.utils import EmbeddingFunc
    from product.evidence_extract import EvidenceChunk, EvidenceExtractor, EvidenceUnit
    from product.prepare_context import extract_source

    profile_path = (
        Path(config["PROMPT_DIR"]) / "entity_type" / config["ENTITY_TYPE_PROMPT_FILE"]
    )
    profile_file = yaml.safe_load(profile_path.read_text())
    profile = validate_entity_extraction_prompt_profile_for_mode(
        resolve_entity_extraction_prompt_profile(
            {"entity_type_prompt_file": config["ENTITY_TYPE_PROMPT_FILE"]}, True
        ),
        True,
    )
    context = {
        "entity_types_guidance": profile["entity_types_guidance"],
        "examples": "\n".join(profile["entity_extraction_json_examples"]),
        "language": config["SUMMARY_LANGUAGE"],
        "max_total_records": int(config["MAX_EXTRACTION_RECORDS"]),
        "max_entity_records": int(config["MAX_EXTRACTION_ENTITIES"]),
    }
    native_system = PROMPTS["entity_extraction_json_system_prompt"].format(**context)
    requests = {}
    documents = []
    catalog_hashes = []
    for source_path, entry in files:
        payload = json.loads((manage.OUTPUT / entry["prepared_path"]).read_text())
        catalog_path = checked_local(entry["catalog_path"], entry["catalog_sha256"])
        catalog_hashes.append(entry["catalog_sha256"])
        catalogs = json.loads(catalog_path.read_text())
        source, _ = extract_source(source_path)
        chunks = payload["text"].split(
            payload["chunking"]["params"]["split_by_character"]
        )
        if len(chunks) != len(catalogs):
            raise ValueError("Catalog does not cover prepared chunks")
        for content, item in zip(chunks, catalogs, strict=True):
            if hashlib.sha256(content.encode()).hexdigest() != item["content_sha256"]:
                raise ValueError("Catalog does not match prepared content")
            units = tuple(
                EvidenceUnit(
                    unit["id"],
                    source_path.name,
                    unit["start"],
                    unit["end"],
                    source,
                    tuple(unit.get("entity_types", [])),
                )
                for unit in item["units"]
            )
            chunk = EvidenceChunk(source_path.name, entry["product"], content, units)
            prompt = (
                PROMPTS["entity_extraction_json_user_prompt"]
                .format(**context, input_text=content, heading_context_block="")
                .strip()
            )
            if prompt in requests:
                raise ValueError("Duplicate extraction input")
            requests[prompt] = chunk
        documents.append((entry, payload))
    print(
        json.dumps(
            {
                "workspace": config["WORKSPACE"],
                "documents": [e["ingestion_label"] for e, _ in documents],
                "chunks": len(requests),
                "apply": apply,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not apply:
        return
    target = manage.ROOT / "rag_storage" / config["WORKSPACE"]
    if target.exists() or manage.RECEIPTS.exists():
        raise ValueError("Target or receipt exists; inspect it instead of replaying")
    contract_hash = hashlib.sha256(
        (
            profile_path.read_text()
            + "".join(catalog_hashes)
            + Path(__file__).with_name("evidence_extract.py").read_text()
        ).encode()
    ).hexdigest()

    def role_function(role: str):
        model = config[f"{role.upper()}_LLM_MODEL"]
        body = json.loads(config[f"{role.upper()}_OPENAI_LLM_EXTRA_BODY"])

        async def complete(prompt, system_prompt=None, **kwargs):
            class Usage:
                rows: list[dict[str, int]]

                def __init__(self):
                    self.rows = []

                def add_usage(self, value):
                    self.rows.append(value)

            usage = Usage()
            event = {"stage": role, "model": model, "state": "sending"}
            append_event(manage.OUTPUT / "provider-calls.jsonl", event)
            started = time.monotonic()
            try:
                result = await openai_complete_if_cache(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    base_url=config["LLM_BINDING_HOST"],
                    api_key=config["LLM_BINDING_API_KEY"],
                    temperature=0,
                    extra_body=body,
                    timeout=int(config["LLM_TIMEOUT"]),
                    token_tracker=usage,
                    **kwargs,
                )
            except Exception as exc:
                append_event(
                    manage.OUTPUT / "provider-calls.jsonl",
                    {
                        **event,
                        "state": "error_or_unknown",
                        "error_type": type(exc).__name__,
                        "usage": usage.rows,
                    },
                )
                raise
            append_event(
                manage.OUTPUT / "provider-calls.jsonl",
                {
                    **event,
                    "state": "completed",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "usage": usage.rows,
                },
            )
            return result

        return complete

    async def select_evidence(prompt, system_prompt):
        return await role_function("extract")(
            prompt,
            system_prompt=system_prompt,
            stream=False,
            response_format={"type": "json_object"},
        )

    adapter = EvidenceExtractor(
        requests=requests,
        native_system_prompt=native_system,
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance=profile["entity_types_guidance"],
        selection_guidance=profile_file["evidence_selection_guidance"],
        complete=select_evidence,
        record=lambda event: append_event(
            manage.OUTPUT / "evidence-events.jsonl", event
        ),
    )

    async def embed(texts, **kwargs):
        return await openai_embed.func(
            texts,
            model=config["EMBEDDING_MODEL"],
            base_url=config["EMBEDDING_BINDING_HOST"],
            api_key=config["EMBEDDING_BINDING_API_KEY"],
            **kwargs,
        )

    roles = {
        role: RoleLLMConfig(
            func=adapter if role == "extract" else role_function(role),
            max_async=int(
                config["EXTRACT_MAX_ASYNC_LLM"]
                if role == "extract"
                else config["MAX_ASYNC_LLM"]
            ),
            timeout=int(config["LLM_TIMEOUT"]),
            metadata={
                "binding": "openai",
                "model": config[f"{role.upper()}_LLM_MODEL"],
                "provider_options": {"evidence_contract_sha256": contract_hash}
                if role == "extract"
                else {},
            },
        )
        for role in ["extract", "keyword", "query"]
    }
    rag = LightRAG(
        working_dir=str(target.parent),
        workspace=config["WORKSPACE"],
        llm_model_func=role_function("extract"),
        llm_model_name=config["EXTRACT_LLM_MODEL"],
        role_llm_configs=roles,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(config["EMBEDDING_DIM"]),
            max_token_size=int(config["EMBEDDING_TOKEN_LIMIT"]),
            send_dimensions=True,
            model_name=config["EMBEDDING_MODEL"],
            func=embed,
        ),
        addon_params={
            "language": config["SUMMARY_LANGUAGE"],
            "entity_type_prompt_file": config["ENTITY_TYPE_PROMPT_FILE"],
        },
        chunk_token_size=8192,
        chunk_overlap_token_size=0,
        entity_extract_max_gleaning=0,
        entity_extract_max_entities=int(config["MAX_EXTRACTION_ENTITIES"]),
        entity_extract_max_records=int(config["MAX_EXTRACTION_RECORDS"]),
        force_llm_summary_on_merge=3,
        summary_max_tokens=64,
        summary_context_size=8192,
        max_parallel_insert=1,
    )
    manage.save_json(
        manage.OUTPUT / "contract.json",
        {
            "sha256": contract_hash,
            "workspace": config["WORKSPACE"],
            "model": config["EXTRACT_LLM_MODEL"],
            "merge_validation": "native threshold 64 tokens forces merge-path checks on real prototype evidence",
        },
    )
    receipts = []
    try:
        await rag.initialize_storages()
        for entry, payload in documents:
            receipt = {
                "file": entry["ingestion_label"],
                "state": "sending",
                "source_sha256": entry["sha256"],
                "started_time": time.time(),
            }
            receipts.append(receipt)
            manage.save_json(manage.RECEIPTS, receipts)
            track = await rag.ainsert(
                payload["text"],
                split_by_character=payload["chunking"]["params"]["split_by_character"],
                split_by_character_only=True,
                file_paths=entry["ingestion_label"],
            )
            statuses = await rag.doc_status.get_docs_by_track_id(track)
            states = {
                key: DocStatus(item.status).value for key, item in statuses.items()
            }
            receipt.update(
                state="processed"
                if states and all(state == "processed" for state in states.values())
                else "not_processed",
                track_id=track,
                documents=states,
            )
            manage.save_json(manage.RECEIPTS, receipts)
            if receipt["state"] != "processed":
                raise RuntimeError(
                    "Native document did not finish; inspect original status"
                )
        for case in manifest.get("retrieval_checks", []):
            response = await rag.aquery(
                case["question"],
                param=QueryParam(
                    mode="mix", only_need_prompt=True, enable_rerank=False
                ),
            )
            manage.save_json(
                manage.OUTPUT / f"context-{case['id']}.json",
                {
                    "question": case["question"],
                    "mode": "mix",
                    "only_need_prompt": True,
                    "execution": "native SDK",
                    "response": response,
                },
            )
    finally:
        await rag.finalize_storages()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manage.select_batch(args.batch)
    asyncio.run(apply_batch(args.apply))


if __name__ == "__main__":
    main()
