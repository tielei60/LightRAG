"""Rebuild A's unchanged business chunks in an isolated comparison workspace.

Only verified preparation artifacts are read from A. All credentials, model
calls, ledgers, native writes and query caches belong to this B application.
The old A graph and old model selections are never replayed as a cold run.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

import yaml

from .common import append_event, digest, inside, read_json, save_json
from .config import PROJECT_ROOT
from .contracts import contract_module
from .ingest import run_contract
from .models import initialize_rag, make_rag, provider
from .prepare import checked_manifest


def require_a(settings):
    if (
        settings.get("PREPARATION") != "business"
        or settings.get("CONTRACT_REVISION") != "current"
    ):
        raise ValueError(
            "A comparison requires the current frozen contract and business preparation arm"
        )


def snapshot_a_inputs(settings):
    require_a(settings)
    manifest = checked_manifest(settings)
    expected = {d["relative_path"]: d for d in manifest["documents"]}
    registry_path = PROJECT_ROOT / "output/workspaces/master/registry.json"
    registry = read_json(registry_path)
    documents, sources = [], []
    extract_source = contract_module(settings, "prepare_context").extract_source
    for batch in registry["batches"]:
        directory = PROJECT_ROOT / "output" / batch["batch"]
        path = directory / "manifest.json"
        if digest(path.read_bytes()) != batch["manifest_sha256"]:
            raise ValueError("A registered preparation manifest changed")
        original = read_json(path)
        sources.append({"path": str(path), "sha256": digest(path.read_bytes())})
        for entry in original["documents"]:
            target = expected.get(entry["relative_path"])
            if target is None or (target["sha256"], target["product"]) != (
                entry["sha256"],
                entry["product"],
            ):
                raise ValueError("A/B original source sets differ")
            artifacts = {}
            for label in ["prepared", "catalog", "source_map"]:
                p = inside(directory, directory / entry[label + "_path"])
                if digest(p.read_bytes()) != entry[label + "_sha256"]:
                    raise ValueError("A original preparation artifact changed")
                artifacts[label] = read_json(p)
            text, _ = extract_source(settings.originals / target["relative_path"])
            if artifacts["source_map"].get("extracted_text_sha256") not in {
                None,
                digest(text),
            }:
                raise ValueError(
                    "A source offsets no longer match the original extraction"
                )
            parts = artifacts["prepared"]["text"].split(
                artifacts["prepared"]["chunking"]["params"]["split_by_character"]
            )
            if len(parts) != len(artifacts["catalog"]):
                raise ValueError("A preparation has incomplete chunk coverage")
            for content, catalog in zip(parts, artifacts["catalog"], strict=True):
                if digest(content) != catalog["content_sha256"]:
                    raise ValueError("A prepared content and catalog differ")
            documents.append({"entry": target, "source_text": text, **artifacts})
    if (
        len(documents) != 21
        or {d["entry"]["relative_path"] for d in documents} != expected.keys()
    ):
        raise ValueError("A comparison must contain exactly the same 21 originals")
    result = {
        "method": "unchanged registered A business preparation, fresh model extraction",
        "registry_sha256": digest(registry_path.read_bytes()),
        "source_manifests": sources,
        "documents": documents,
    }
    path = settings.run_dir / "a-inputs.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("The frozen A input snapshot changed")
    save_json(path, result)
    return {
        "documents": 21,
        "prepared_chunks": sum(len(d["catalog"]) for d in documents),
        "snapshot": str(path),
    }


def adapter_options(settings):
    settings.activate()
    from lightrag.prompt import (
        PROMPTS,
        get_default_entity_extraction_prompt_profile,
        load_entity_extraction_prompt_profile,
        validate_entity_extraction_prompt_profile_for_mode,
    )

    module = contract_module(settings, "evidence_extract")
    raw = yaml.safe_load((settings.profiles / "entity_type/products.yml").read_text())
    taxonomy = contract_module(
        settings, "extraction_profile"
    ).ExtractionProfile.from_mapping(raw)
    profile = validate_entity_extraction_prompt_profile_for_mode(
        {
            **get_default_entity_extraction_prompt_profile(),
            **load_entity_extraction_prompt_profile(
                settings.profiles / "entity_type/products.yml"
            ),
        },
        True,
    )
    context = {
        "entity_types_guidance": profile["entity_types_guidance"],
        "examples": "\n".join(profile["entity_extraction_json_examples"]),
        "language": "Chinese",
        "max_total_records": settings.integer("MAX_EXTRACTION_RECORDS"),
        "max_entity_records": settings.integer("MAX_EXTRACTION_ENTITIES"),
    }
    requests = {}
    for document in read_json(settings.run_dir / "a-inputs.json")["documents"]:
        entry = document["entry"]
        parts = document["prepared"]["text"].split(
            document["prepared"]["chunking"]["params"]["split_by_character"]
        )
        for content, catalog in zip(parts, document["catalog"], strict=True):
            units = tuple(
                module.EvidenceUnit(
                    u["id"],
                    Path(entry["relative_path"]).name,
                    u["start"],
                    u["end"],
                    document["source_text"],
                    tuple(u.get("entity_types", [])),
                    u.get("role", "primary"),
                )
                for u in catalog["units"]
            )
            prompt = (
                PROMPTS["entity_extraction_json_user_prompt"]
                .format(**context, input_text=content, heading_context_block="")
                .strip()
            )
            if prompt in requests:
                raise ValueError("A native extraction prompts collide")
            requests[prompt] = module.EvidenceChunk(
                Path(entry["relative_path"]).name, entry["product"], content, units
            )
    return {
        "requests": requests,
        "native_system_prompt": PROMPTS["entity_extraction_json_system_prompt"]
        .format(**context)
        .strip(),
        "native_summary_template": PROMPTS["summarize_entity_descriptions"],
        "type_guidance": profile["entity_types_guidance"],
        "selection_guidance": raw["evidence_selection_guidance"],
        "complete": provider(settings, "extract"),
        "record": lambda event: append_event(
            settings.run_dir / "extraction-events.jsonl", event
        ),
        "ledger_path": settings.run_dir / "a-selection-ledger.json",
        "ledger_context": digest(
            json.dumps(run_contract(settings), ensure_ascii=False, sort_keys=True)
        ),
        "profile": taxonomy,
    }


class AReplay:
    def __init__(self, delegate):
        self.delegate = delegate
        self.source = None
        self.epoch = uuid.uuid4().hex

    def bind(self, rag):
        pass

    def cache_identity(self):
        return {"epoch": self.epoch, "source": self.source}

    def __deepcopy__(self, memo):
        return self

    async def __call__(self, *args, **kwargs):
        return await self.delegate(*args, **kwargs)


async def ingest_a(settings):
    require_a(settings)
    checked_manifest(settings)
    snapshot = run_contract(settings)
    path = settings.run_dir / "run-snapshot.json"
    if path.exists() and read_json(path) != snapshot:
        raise ValueError("A comparison contract changed; use a new run")
    save_json(path, snapshot)
    options = adapter_options(settings)
    module = contract_module(settings, "evidence_extract")
    approval = module.EvidenceExtractor(**options)
    slots = asyncio.Semaphore(4)

    async def approve(prompt):
        async with slots:
            for attempt in range(3):
                try:
                    await approval(
                        prompt, system_prompt=options["native_system_prompt"]
                    )
                    return
                except ValueError:
                    if attempt == 2:
                        raise

    # All model selections are validated before any native graph write, as in A.
    await asyncio.gather(*(approve(prompt) for prompt in options["requests"]))
    replay = AReplay(module.EvidenceExtractor(**options, replay_only=True))
    rag = make_rag(settings, extraction=replay)
    receipts = []
    try:
        await initialize_rag(rag, settings)
        from lightrag.base import DocStatus

        statuses = await rag.doc_status.get_docs_by_statuses(
            list(DocStatus), strict=True
        )
        if statuses:
            raise ValueError("A comparison import requires an empty workspace")
        for document in read_json(settings.run_dir / "a-inputs.json")["documents"]:
            entry = document["entry"]
            source = Path(entry["relative_path"]).name
            replay.source = source
            replay.delegate.activate_sources([source])
            payload = document["prepared"]
            started = time.time()
            track = await rag.ainsert(
                payload["text"],
                split_by_character=payload["chunking"]["params"]["split_by_character"],
                split_by_character_only=True,
                file_paths=source,
            )
            states = await rag.doc_status.get_docs_by_track_id(track)
            if len(states) != 1:
                raise ValueError("A native document identity is ambiguous")
            doc_id, status = next(iter(states.items()))
            if DocStatus(status.status) != DocStatus.PROCESSED:
                raise ValueError(f"A native import failed: {source}")
            originals = {
                digest(chunk.content): (prompt, chunk)
                for prompt, chunk in options["requests"].items()
                if chunk.source == source
            }
            chunks, selections = [], {}
            for chunk_id in status.chunks_list:
                value = await rag.text_chunks.get_by_id(chunk_id)
                prompt, _chunk = originals[digest(value["content"])]
                key = digest(prompt)
                selections[key] = approval.selections[key]
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "indexed_sha256": digest(value["content"]),
                        "indexed_content": value["content"],
                        "native_prompt": prompt,
                        "tokens": value.get("tokens"),
                    }
                )
            save_json(
                settings.run_dir / "catalogs" / (doc_id + ".json"),
                {
                    "document_id": doc_id,
                    "source": source,
                    "source_sha256": entry["sha256"],
                    "product": entry["product"],
                    "source_projection": document["source_text"],
                    "chunks": chunks,
                },
            )
            save_json(
                settings.run_dir / "selections" / (doc_id + ".json"),
                {"selections": selections},
            )
            receipts.append(
                {
                    "file": source,
                    "document_id": doc_id,
                    "state": "processed",
                    "chunks": len(chunks),
                    "seconds": time.time() - started,
                }
            )
            save_json(settings.run_dir / "import-receipts.json", receipts)
        report = {
            "expected_documents": 21,
            "processed": len(receipts),
            "chunks": sum(r["chunks"] for r in receipts),
            "cold_selection_requests": len(options["requests"]),
            "full_corpus_passed": len(receipts) == 21,
        }
        save_json(settings.run_dir / "import-report.json", report)
        return report
    finally:
        await rag.finalize_storages()
