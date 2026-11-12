"""Preflight evidence selections, then replay them through the native SDK."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
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


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def approve_extractions(adapter, requests: dict, concurrency: int) -> dict:
    """Validate every exact native request before allowing any storage mutation."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    approved = {}

    async def approve(prompt, chunk):
        async with semaphore:
            # Protocol/semantic validation gets one bounded local correction.
            # Provider technical retries remain the provider's responsibility.
            for attempt in range(2):
                try:
                    response = await adapter(
                        prompt, system_prompt=adapter.native_system_prompt
                    )
                    native = json.loads(response)
                    break
                except ValueError:
                    if attempt:
                        raise
            approved[digest(prompt)] = {
                "source": chunk.source,
                "content_sha256": digest(chunk.content),
                "prompt_sha256": digest(prompt),
                "native": native,
            }

    await asyncio.gather(
        *(approve(prompt, chunk) for prompt, chunk in requests.items())
    )
    return approved


def reuse_approved_selections(adapter, requests: dict, previous: Path) -> dict:
    """Revalidate old pointers against exact current source and evidence catalogs."""
    from product.evidence_extract import PointerResult, to_native

    previous = previous.resolve(strict=True)
    manifest = json.loads((previous / "manifest.json").read_text())
    approved = json.loads((previous / "approved-extractions.json").read_text())[
        "chunks"
    ]
    rows = json.loads((previous / "evidence-ledger.json").read_text())["selections"]
    source_root = Path(manifest["source_root"]).resolve(strict=True)
    catalogs = {}

    def artifact(name, expected):
        path = (previous / name).resolve(strict=True)
        if (
            not path.is_relative_to(previous)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Prior reviewed artifact changed or escapes its batch")
        return json.loads(path.read_text())

    for entry in manifest["documents"]:
        source = (source_root / entry["relative_path"]).resolve(strict=True)
        if (
            not source.is_relative_to(source_root)
            or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Prior approved source changed")
        payload = artifact(entry["prepared_path"], entry["prepared_sha256"])
        catalog = artifact(entry["catalog_path"], entry["catalog_sha256"])
        content = payload["text"].split(
            payload["chunking"]["params"]["split_by_character"]
        )
        if len(content) != len(catalog):
            raise ValueError("Prior approved catalog count changed")
        for text, item in zip(content, catalog, strict=True):
            if digest(text) != item["content_sha256"]:
                raise ValueError("Prior prepared chunk no longer matches catalog")
            key = (source.name, item["content_sha256"])
            if key in catalogs:
                raise ValueError("Prior approved chunk identity is ambiguous")
            catalogs[key] = (entry["product"], item["units"])

    checked = []
    for prompt, chunk in requests.items():
        key = digest(prompt)
        old = approved.get(key)
        if (
            old is None
            or old["prompt_sha256"] != key
            or old["content_sha256"] != digest(chunk.content)
            or old["source"] != chunk.source
        ):
            raise ValueError(
                "Prior approval does not match the exact current native input"
            )
        product, units = catalogs.get(
            (chunk.source, digest(chunk.content)), (None, None)
        )
        current_units = [
            {
                "id": unit.identifier,
                "start": unit.start,
                "end": unit.end,
                "entity_types": list(unit.entity_types),
                "role": unit.role,
            }
            for unit in chunk.units
        ]
        old_units = [
            {
                "id": unit["id"],
                "start": unit["start"],
                "end": unit["end"],
                "entity_types": unit.get("entity_types", []),
                "role": unit.get("role", "primary"),
            }
            for unit in units or []
        ]
        if product != chunk.product or old_units != current_units or key not in rows:
            raise ValueError(
                "Prior pointers do not match the current source/unit identities"
            )
        selection = PointerResult.model_validate(rows[key]["selection"])
        # Old native output and contract hashes confer no authority here.
        # The current adapter must validate and regenerate every native record.
        native = to_native(selection, chunk, profile=adapter.profile)
        checked.append((key, selection, native))
    if len(checked) != len(rows) or len(checked) != len(approved):
        raise ValueError("Prior approval scope differs from current batch")
    for key, selection, native in checked:
        adapter.persist_selection(key, selection, native)
    return {
        "source_batch": previous.name,
        "revalidated_chunks": len(checked),
        "provider_called": False,
        "method": "exact prompt/content/source/unit checks; current pointer validation and native regeneration",
    }


def reuse_approved_batches(
    adapter, requests: dict, root: Path, batch_names: list[str]
) -> dict:
    """Revalidate a complete union of historical batches for database replay."""
    from product.evidence_extract import PointerResult, to_native
    from product.workspace_registry import load_reviewed_batch

    if not batch_names or len(batch_names) != len(set(batch_names)):
        raise ValueError("Unique approved source batches are required")
    historical = {}
    for name in batch_names:
        previous = load_reviewed_batch(root, name)
        for prompt, chunk in previous.requests.items():
            if prompt in historical:
                raise ValueError(
                    "Approved source batches contain duplicate extraction inputs"
                )
            historical[prompt] = (chunk, previous.selections[digest(prompt)])
    if set(historical) != set(requests):
        raise ValueError(
            "Approved source batch union differs from the current import scope"
        )
    checked = []
    for prompt, chunk in requests.items():
        old, row = historical[prompt]
        if old != chunk:
            raise ValueError(
                "Historical source or evidence identity differs from the current input"
            )
        selection = PointerResult.model_validate(row["selection"])
        native = to_native(selection, chunk, profile=adapter.profile)
        checked.append((digest(prompt), selection, native))
    for key, selection, native in checked:
        adapter.persist_selection(key, selection, native)
    return {
        "source_batches": batch_names,
        "revalidated_chunks": len(checked),
        "provider_called": False,
        "method": "exact union of reviewed prompt/source/content/unit identities; current pointer validation",
    }


def validate_native_identities(approved: dict) -> None:
    """Refuse lossy identifier normalization or truncation before native writes."""
    from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH
    from lightrag.utils import normalize_entity_name

    seen = {}
    for item in approved.values():
        for entity in item["native"]["entities"]:
            name = entity["name"]
            normalized = normalize_entity_name(name)
            if not normalized or len(normalized) > DEFAULT_ENTITY_NAME_MAX_LENGTH:
                raise ValueError("Entity identifier would be empty or truncated")
            if normalized in seen and seen[normalized] != name:
                raise ValueError(
                    "Entity identifiers collide after native normalization"
                )
            seen[normalized] = name


def incremental_replay_adapter(
    options: dict, old_requests: dict, old_selections: dict, new_selections: dict
):
    """Restore old selected provenance before native shared-object merges."""
    from product.evidence_extract import EvidenceExtractor

    requests = {**old_requests, **options["requests"]}
    if len(requests) != len(old_requests) + len(options["requests"]):
        raise ValueError("New extraction input collides with registered evidence")
    adapter = EvidenceExtractor(
        **{**options, "requests": requests, "ledger_path": None}, replay_only=True
    )
    adapter.selections = {**old_selections, **new_selections}
    adapter.activate_sources([chunk.source for chunk in old_requests.values()])
    return adapter


def readback_files(target: Path, documents: list, approved: dict) -> dict:
    """Read the closed file-backed batch and verify persisted native evidence."""
    import networkx as nx

    from lightrag.utils import (
        normalize_entity_name,
        sanitize_and_normalize_extracted_text,
    )
    from lightrag.utils_pipeline import normalize_document_file_path
    from product.evidence_extract import canonical

    def normalized(text):
        return canonical(sanitize_and_normalize_extracted_text(text))

    def preserves_evidence(expected, stored):
        # Source-position merge ordering can differ from catalog order.
        # Verify complete original paragraphs independently; never reduce
        # this to a word/token presence check that could lose conditions.
        paragraphs = [
            normalized(part) for part in expected.split("\n\n") if part.strip()
        ]
        actual = normalized(stored)
        return bool(paragraphs) and all(part in actual for part in paragraphs)

    statuses = json.loads((target / "kv_store_doc_status.json").read_text())
    chunks = json.loads((target / "kv_store_text_chunks.json").read_text())
    graph = nx.read_graphml(target / "graph_chunk_entity_relation.graphml")
    vectors = {
        name: json.loads((target / f"vdb_{name}.json").read_text())
        for name in ("entities", "relationships", "chunks")
    }
    errors = []
    native_labels = [
        normalize_document_file_path(entry["ingestion_label"]) for entry, _ in documents
    ]
    expected_labels = set(native_labels)
    actual_labels = {row["file_path"] for row in statuses.values()}
    if len(expected_labels) != len(native_labels):
        errors.append("Manifest sources collide after native basename normalization")
    if len(statuses) != len(documents) or actual_labels != expected_labels:
        errors.append("Persisted document identities/count differ from manifest")
    if any(row["status"] != "processed" for row in statuses.values()):
        errors.append("One or more persisted documents are not processed")
    expected_content = Counter(item["content_sha256"] for item in approved.values())
    actual_content = Counter(digest(row["content"]) for row in chunks.values())
    if actual_content != expected_content:
        errors.append("Persisted chunk content differs from approved native input")
    vector_rows = {name: data.get("data", []) for name, data in vectors.items()}
    chunk_vector_ids = {row["__id__"] for row in vector_rows["chunks"]}
    if chunk_vector_ids != set(chunks):
        errors.append("Chunk vectors do not cover exactly persisted text chunks")
    entity_vectors = {row["entity_name"]: row for row in vector_rows["entities"]}
    relation_vectors = {
        frozenset((row["src_id"], row["tgt_id"])): row
        for row in vector_rows["relationships"]
    }
    expected_entities = {
        normalize_entity_name(entity["name"])
        for item in approved.values()
        for entity in item["native"]["entities"]
    }
    expected_relationships = {
        frozenset(normalize_entity_name(relation[key]) for key in ("source", "target"))
        for item in approved.values()
        for relation in item["native"]["relationships"]
    }
    if set(graph.nodes) != expected_entities:
        errors.append(
            "Persisted graph entities differ from the complete approved scope"
        )
    if {frozenset(edge) for edge in graph.edges} != expected_relationships:
        errors.append(
            "Persisted graph relations differ from the complete approved scope"
        )
    for item in approved.values():
        native = item["native"]
        for entity in native["entities"]:
            name = normalize_entity_name(entity["name"])
            node = graph.nodes.get(name)
            if node is None:
                errors.append(f"Missing graph entity: {name}")
                continue
            description = entity["description"]
            if not preserves_evidence(description, node["description"]):
                errors.append(f"Graph entity lost approved evidence: {name}")
            if name not in entity_vectors or not preserves_evidence(
                description, entity_vectors[name]["content"]
            ):
                errors.append(f"Entity vector lost approved evidence: {name}")
        for relation in native["relationships"]:
            source, destination = (
                normalize_entity_name(relation[k]) for k in ("source", "target")
            )
            edge = graph.get_edge_data(source, destination)
            if edge is None:
                errors.append(f"Missing graph relation: {source} / {destination}")
                continue
            description = relation["description"]
            if not preserves_evidence(description, edge["description"]):
                errors.append(
                    f"Graph relation lost approved evidence: {source} / {destination}"
                )
            vector = relation_vectors.get(frozenset((source, destination)))
            if vector is None or not preserves_evidence(description, vector["content"]):
                errors.append(
                    f"Relation vector lost approved evidence: {source} / {destination}"
                )
    for name, row in [
        *graph.nodes(data=True),
        *[(f"{u} / {v}", r) for u, v, r in graph.edges(data=True)],
    ]:
        source_ids = set(row.get("source_id", "").split("<SEP>"))
        if not source_ids or not source_ids <= chunks.keys():
            errors.append(f"Missing or unresolved graph chunk provenance: {name}")
    return {
        "workspace": target.name,
        "readback": "closed native files; fresh JSON, GraphML and vector reads",
        "documents": len(statuses),
        "processed_documents": sum(
            row["status"] == "processed" for row in statuses.values()
        ),
        "chunks": len(chunks),
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "vectors": {name: len(rows) for name, rows in vector_rows.items()},
        "errors": errors,
        "technical_passed": not errors,
        "business_acceptance": "pending native retrieval and answer review",
    }


async def apply_batch(
    apply: bool,
    approved_from: str | None = None,
    *,
    append: bool = False,
    base_batch: str | None = None,
    approved_batches: list[str] | None = None,
) -> None:
    """Import a new batch, or explicitly append it to a registered workspace."""
    if base_batch is not None and not append:
        raise ValueError("--base-batch requires explicit --append")
    from product.database_storage import is_database_backend

    if approved_from is not None and approved_batches:
        raise ValueError("Use one approval reuse option, not both")
    if apply and (append or is_database_backend(manage.settings())):
        from product.workspace_registry import read_json, workspace_writer, write_json

        config = manage.settings()
        with workspace_writer(manage.ROOT, config["WORKSPACE"]):
            try:
                await _apply_batch(
                    apply,
                    approved_from,
                    append=append,
                    base_batch=base_batch,
                    approved_batches=approved_batches,
                )
            except BaseException as exc:
                journal = manage.OUTPUT / (
                    "append-journal.json" if append else "database-import-journal.json"
                )
                if journal.exists():
                    state = read_json(journal)
                    if state.get("state") != "completed":
                        write_json(
                            journal,
                            {
                                **state,
                                "state": "failed_inspect_backup",
                                "error_type": type(exc).__name__,
                            },
                        )
                raise
    else:
        await _apply_batch(
            apply,
            approved_from,
            append=append,
            base_batch=base_batch,
            approved_batches=approved_batches,
        )


async def _apply_batch(
    apply: bool,
    approved_from: str | None,
    *,
    append: bool,
    base_batch: str | None,
    approved_batches: list[str] | None = None,
) -> None:
    files = manage.checked_files()
    config = manage.settings()
    from product.database_storage import (
        is_database_backend,
        storage_contract,
        storage_options,
    )

    database = is_database_backend(config)
    backend = storage_contract(config)
    manifest = json.loads(manage.MANIFEST.read_text())
    if manifest["workspace"] != config["WORKSPACE"]:
        raise ValueError("Workspace does not match manifest")
    if manifest.get("ingestion_mode") == "append" and not append:
        raise ValueError("This manifest requires explicit --append")
    if config["MAX_GLEANING"] != "0":
        raise ValueError(
            "Evidence replay requires initial extraction only (MAX_GLEANING=0)"
        )
    if Path(config["WORKING_DIR"]).resolve() != manage.ROOT / "rag_storage":
        raise ValueError("Unexpected native storage root")
    os.environ.update({k: v for k, v in config.items() if v is not None})

    import yaml

    from lightrag import LightRAG
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
    from product.extraction_profile import ExtractionProfile
    from product.prepare_context import extract_source

    profile_path = (
        Path(config["PROMPT_DIR"]) / "entity_type" / config["ENTITY_TYPE_PROMPT_FILE"]
    )
    profile_file = yaml.safe_load(profile_path.read_text())
    extraction_profile = ExtractionProfile.from_mapping(profile_file)
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
                    role=unit.get("role", "primary"),
                )
                for unit in item["units"]
            )
            chunk = EvidenceChunk(source_path.name, entry["product"], content, units)
            extraction_profile.validate_catalog(chunk.units)
            prompt = (
                PROMPTS["entity_extraction_json_user_prompt"]
                .format(**context, input_text=content, heading_context_block="")
                .strip()
            )
            if prompt in requests:
                raise ValueError("Duplicate extraction input")
            requests[prompt] = chunk
        documents.append((entry, payload))
    target = manage.ROOT / "rag_storage" / config["WORKSPACE"]
    database_before = None
    if database:
        from product.database_storage import snapshot_database_config
        from product.workspace_registry import read_json, registry_path

        database_before = await snapshot_database_config(
            config,
            manage.ROOT,
            manage.OUTPUT / "database-snapshots" / f"before-{time.time_ns()}",
        )
        counts = read_json(database_before / "database-export.json")["counts"]
        if not append and (
            any(counts.values())
            or registry_path(manage.ROOT, config["WORKSPACE"]).exists()
        ):
            raise ValueError(
                "Database workspace is not empty or already registered; use --append"
            )
    registry = None
    historical_batches = []
    old_documents, old_requests, old_approved, old_selections = [], {}, {}, {}
    if append:
        from product.verify_batch import verify_snapshot
        from product.workspace_registry import (
            embedding_contract,
            load_registered_batches,
            snapshot_for_batches,
            validate_additions,
        )

        if not database and not target.is_dir():
            raise ValueError("Incremental import requires an existing workspace")
        registry, historical_batches = load_registered_batches(
            manage.ROOT, config["WORKSPACE"], base_batch
        )
        if registry["embedding"] != embedding_contract(config):
            raise ValueError(
                "Embedding model, binding or dimension differs from the existing workspace"
            )
        from product.database_storage import STORAGE_DEFAULTS

        if registry.get("storage", STORAGE_DEFAULTS) != backend:
            raise ValueError(
                "The workspace uses different storage backends or database endpoints"
            )
        for previous in historical_batches:
            old_documents.extend(previous.documents)
            for current, extra in (
                (old_requests, previous.requests),
                (old_approved, previous.approved),
                (old_selections, previous.selections),
            ):
                if current.keys() & extra.keys():
                    raise ValueError("Registered historical extraction inputs collide")
                current.update(extra)
        validate_additions(old_documents, documents)
        persisted = database_before if database else target
        old_report = readback_files(persisted, old_documents, old_approved)
        snapshot = snapshot_for_batches(
            manage.ROOT, config["WORKSPACE"], historical_batches, registry
        )
        detailed = verify_snapshot(snapshot, persisted, int(config["EMBEDDING_DIM"]))
        if not old_report["technical_passed"] or not detailed["passed"]:
            raise ValueError(
                "Existing workspace does not match its registered sources; inspect readback"
            )
        manage.save_json(
            manage.OUTPUT / "before-append-readback.json",
            {"evidence": old_report, "indexes": detailed},
        )
    print(
        json.dumps(
            {
                "workspace": config["WORKSPACE"],
                "documents": [e["ingestion_label"] for e, _ in documents],
                "chunks": len(requests),
                "apply": apply,
                "append": append,
                "existing_documents": len(old_documents),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not apply:
        return
    if manage.PID_FILE.exists():
        try:
            os.kill(int(manage.PID_FILE.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Stop the batch server before the SDK writer starts")
    if (target.exists() and not append and not database) or manage.RECEIPTS.exists():
        raise ValueError("Target or receipt exists; inspect it instead of replaying")
    if append and (manage.OUTPUT / "append-journal.json").exists():
        raise ValueError(
            "Append journal exists; inspect its original result before retrying"
        )
    if database and (manage.OUTPUT / "database-import-journal.json").exists():
        raise ValueError(
            "Database import journal exists; inspect the original operation"
        )
    contract_hash = digest(
        json.dumps(
            {
                "profile": profile_path.read_text(),
                "catalogs": catalog_hashes,
                "adapter": Path(__file__).with_name("evidence_extract.py").read_text(),
                "manifest": manifest,
                "models": {
                    key: value
                    for key, value in config.items()
                    if any(
                        part in key
                        for part in (
                            "MODEL",
                            "EXTRA_BODY",
                            "EMBEDDING_DIM",
                            "BINDING_HOST",
                        )
                    )
                },
                "native_system": native_system,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

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

    adapter_options = {
        "requests": requests,
        "native_system_prompt": native_system,
        "native_summary_template": PROMPTS["summarize_entity_descriptions"],
        "type_guidance": profile["entity_types_guidance"],
        "selection_guidance": profile_file["evidence_selection_guidance"],
        "complete": select_evidence,
        "record": lambda event: append_event(
            manage.OUTPUT / "evidence-events.jsonl", event
        ),
        "ledger_path": manage.OUTPUT / "evidence-ledger.json",
        "ledger_context": contract_hash,
        "profile": extraction_profile,
    }
    approval_adapter = EvidenceExtractor(**adapter_options)
    if approved_from is not None:
        if not approved_from or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in approved_from
        ):
            raise ValueError("Invalid source approval batch")
        previous = manage.ROOT / "output" / approved_from
        if previous.resolve() == manage.OUTPUT.resolve():
            raise ValueError("Approval reuse requires a distinct new candidate batch")
        reused = reuse_approved_selections(approval_adapter, requests, previous)
        manage.save_json(manage.OUTPUT / "approval-reuse.json", reused)
    elif approved_batches:
        reused = reuse_approved_batches(
            approval_adapter, requests, manage.ROOT, approved_batches
        )
        manage.save_json(manage.OUTPUT / "approval-reuse.json", reused)
    approved = await approve_extractions(
        approval_adapter, requests, int(config["EXTRACT_MAX_ASYNC_LLM"])
    )
    validate_native_identities(approved)
    manage.save_json(
        manage.OUTPUT / "approved-extractions.json",
        {
            "contract_sha256": contract_hash,
            "state": "validated_before_native_writes",
            "chunks": approved,
            "semantic_acceptance": "requires source and native retrieval review",
        },
    )
    # A fresh adapter keeps future documents out of the current merge ledger.
    if append:
        all_approved = {**old_approved, **approved}
        validate_native_identities(all_approved)
        adapter = incremental_replay_adapter(
            adapter_options, old_requests, old_selections, approval_adapter.selections
        )
    else:
        all_approved = approved
        adapter = EvidenceExtractor(**adapter_options, replay_only=True)

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
        **storage_options(config),
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
        summary_max_tokens=int(config.get("SUMMARY_MAX_TOKENS", "1200")),
        summary_context_size=int(config.get("SUMMARY_CONTEXT_SIZE", "8192")),
        max_parallel_insert=1,
    )
    manage.save_json(
        manage.OUTPUT / "contract.json",
        {
            "sha256": contract_hash,
            "workspace": config["WORKSPACE"],
            "model": config["EXTRACT_LLM_MODEL"],
            "merge_validation": "position-based evidence replay; no provider call during merge",
            "approved_chunks": len(approved),
            "summary_max_tokens": rag.summary_max_tokens,
            "summary_context_size": rag.summary_context_size,
        },
    )
    backup = None
    if append or database:
        from product.workspace_registry import backup_workspace, write_json

        backup = (
            database_before if database else backup_workspace(target, manage.OUTPUT)
        )
        write_json(
            manage.OUTPUT
            / ("append-journal.json" if append else "database-import-journal.json"),
            {
                "state": "prepared",
                "workspace": config["WORKSPACE"],
                "batch": manage.OUTPUT.name,
                "backup": str(backup),
                "existing_batches": [batch.name for batch in historical_batches],
                "existing_documents": len(old_documents),
                "added_documents": len(documents),
                "storage": backend,
                "backup_type": "workspace-scoped native database verification export"
                if database
                else "native file copy",
                "failure_recovery": "Inspect the operation and restore from retained database backup; no automatic rollback is claimed"
                if database
                else "Inspect the retained native file backup",
                "started_at": time.time(),
            },
        )
    receipts = []
    database_after = None
    try:
        await rag.initialize_storages()
        for entry, payload in documents:
            adapter.activate_sources([Path(entry["relative_path"]).name])
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
        if database:
            from product.database_storage import capture_database_snapshot

            # Public callbacks flush the native vector buffers before reading
            # SQL. Reopening in a separate validation process remains possible.
            for storage in (rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb):
                await storage.index_done_callback()
            database_after = await capture_database_snapshot(
                rag, manage.OUTPUT / "database-snapshots" / "after-import"
            )

    finally:
        await rag.finalize_storages()
    persisted = database_after if database else target
    report = readback_files(persisted, [*old_documents, *documents], all_approved)
    report["workspace"] = config["WORKSPACE"]
    report["storage"] = backend
    manage.save_json(manage.OUTPUT / "final-code-readback.json", report)
    manage.save_json(manage.OUTPUT / "final-document-status.json", receipts)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["technical_passed"]:
        raise RuntimeError("Native readback failed; inspect final-code-readback.json")
    if append:
        from product.verify_batch import verify_snapshot
        from product.workspace_registry import (
            commit_append,
            read_json,
            verify_preserved,
            write_json,
        )

        preservation = verify_preserved(backup, persisted)
        manage.save_json(manage.OUTPUT / "preservation-readback.json", preservation)
        if not preservation["passed"]:
            raise RuntimeError(
                "Append changed prior document evidence; inspect backup and preservation report"
            )
        # Complete source/evidence checks above precede publishing membership.
        # Build the prospective full snapshot first, so an index failure cannot
        # publish a registry that falsely claims the append completed.
        from product.workspace_registry import batch_record, load_reviewed_batch

        current_batch = load_reviewed_batch(manage.ROOT, manage.OUTPUT.name)
        prospective = {
            **registry,
            "batches": [*registry["batches"], batch_record(current_batch)],
        }
        snapshot = snapshot_for_batches(
            manage.ROOT,
            config["WORKSPACE"],
            [*historical_batches, current_batch],
            prospective,
        )
        detailed = verify_snapshot(snapshot, persisted, int(config["EMBEDDING_DIM"]))
        manage.save_json(manage.OUTPUT / "workspace-readback.json", detailed)
        if not detailed["passed"]:
            raise RuntimeError(
                "Cumulative native indexes failed verification; inspect append backup"
            )
        commit_append(manage.ROOT, config["WORKSPACE"], registry, manage.OUTPUT.name)
        journal_path = manage.OUTPUT / "append-journal.json"
        write_json(
            journal_path,
            {
                **read_json(journal_path),
                "state": "completed",
                "completed_at": time.time(),
            },
        )
    elif database:
        from product.verify_batch import verify_snapshot
        from product.workspace_registry import (
            commit_initial_database,
            read_json,
            write_json,
        )

        detailed = verify_snapshot(
            manage.OUTPUT, persisted, int(config["EMBEDDING_DIM"])
        )
        manage.save_json(manage.OUTPUT / "workspace-readback.json", detailed)
        if not detailed["passed"]:
            raise RuntimeError("Initial database indexes failed verification")
        commit_initial_database(manage.ROOT, config["WORKSPACE"], manage.OUTPUT.name)
        journal_path = manage.OUTPUT / "database-import-journal.json"
        write_json(
            journal_path,
            {
                **read_json(journal_path),
                "state": "completed",
                "completed_at": time.time(),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add this batch to an existing registered workspace",
    )
    parser.add_argument(
        "--base-batch",
        help="Reviewed batch that owns an existing unregistered workspace",
    )
    parser.add_argument(
        "--approved-from", help="Revalidate pointer selections from an earlier batch"
    )
    parser.add_argument(
        "--approved-batches",
        nargs="+",
        help="Revalidate the exact union of historical batches for database replay",
    )
    args = parser.parse_args()
    manage.select_batch(args.batch)
    asyncio.run(
        apply_batch(
            args.apply,
            args.approved_from,
            append=args.append,
            base_batch=args.base_batch,
            approved_batches=args.approved_batches,
        )
    )


if __name__ == "__main__":
    main()
