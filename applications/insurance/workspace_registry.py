"""Track reviewed import batches that share one persistent native workspace.

This is an operator-side registry. Native storage remains owned by LightRAG;
imports use its public SDK rather than concatenating storage files.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checked_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid workspace or import batch name")
    return value


def registry_path(root: Path, workspace: str) -> Path:
    return root / "output/workspaces" / checked_name(workspace) / "registry.json"


def embedding_contract(config: dict) -> dict:
    keys = ("EMBEDDING_MODEL", "EMBEDDING_DIM", "EMBEDDING_BINDING")
    result = {key: str(config.get(key) or "") for key in keys}
    if not result["EMBEDDING_MODEL"] or int(result["EMBEDDING_DIM"]) <= 0:
        raise ValueError("The workspace embedding contract is incomplete")
    return result


def checked_artifact(batch: Path, name: str, expected_hash: str) -> Path:
    path = (batch / name).resolve(strict=True)
    if not path.is_relative_to(batch.resolve()) or sha256(path) != expected_hash:
        raise ValueError("Registered import artifact changed or escaped its batch")
    return path


@dataclass
class ReviewedBatch:
    name: str
    path: Path
    manifest: dict
    config: dict
    documents: list
    requests: dict
    selections: dict
    approved: dict


def load_reviewed_batch(root: Path, name: str) -> ReviewedBatch:
    """Revalidate historical pointers using that batch's own frozen profile."""
    import yaml

    from applications.insurance.evidence_extract import (
        EvidenceChunk,
        EvidenceUnit,
        PointerResult,
        to_native,
    )
    from applications.insurance.extraction_profile import ExtractionProfile
    from applications.insurance.prepare_context import extract_source
    from lightrag.prompt import (
        PROMPTS,
        get_default_entity_extraction_prompt_profile,
        load_entity_extraction_prompt_profile,
        validate_entity_extraction_prompt_profile_for_mode,
    )

    batch = root / "output" / checked_name(name)
    manifest = read_json(batch / "manifest.json")
    config = dict(dotenv_values(batch / "runtime.env", interpolate=False))
    profile_path = (
        Path(config["PROMPT_DIR"]) / "entity_type" / config["ENTITY_TYPE_PROMPT_FILE"]
    )
    raw_profile = yaml.safe_load(profile_path.read_text())
    profile = ExtractionProfile.from_mapping(raw_profile)
    native_profile = validate_entity_extraction_prompt_profile_for_mode(
        {
            **get_default_entity_extraction_prompt_profile(),
            **load_entity_extraction_prompt_profile(profile_path),
        },
        True,
    )
    context = {
        "entity_types_guidance": native_profile["entity_types_guidance"],
        "examples": "\n".join(native_profile["entity_extraction_json_examples"]),
        "language": config["SUMMARY_LANGUAGE"],
        "max_total_records": int(config["MAX_EXTRACTION_RECORDS"]),
        "max_entity_records": int(config["MAX_EXTRACTION_ENTITIES"]),
    }
    ledger = read_json(batch / "evidence-ledger.json")["selections"]
    approved = read_json(batch / "approved-extractions.json")["chunks"]
    requests, documents, selections = {}, [], {}
    source_root = Path(manifest["source_root"]).resolve(strict=True)
    for entry in manifest["documents"]:
        source = (source_root / entry["relative_path"]).resolve(strict=True)
        if not source.is_relative_to(source_root) or sha256(source) != entry["sha256"]:
            raise ValueError("Registered original source changed")
        payload = read_json(
            checked_artifact(batch, entry["prepared_path"], entry["prepared_sha256"])
        )
        catalogs = read_json(
            checked_artifact(batch, entry["catalog_path"], entry["catalog_sha256"])
        )
        if "source_map_path" in entry:
            checked_artifact(
                batch, entry["source_map_path"], entry["source_map_sha256"]
            )
        source_text, _ = extract_source(source)
        contents = payload["text"].split(
            payload["chunking"]["params"]["split_by_character"]
        )
        if len(contents) != len(catalogs):
            raise ValueError("Registered source catalog does not cover all chunks")
        for content, catalog in zip(contents, catalogs, strict=True):
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash != catalog["content_sha256"]:
                raise ValueError("Registered prepared chunk changed")
            units = tuple(
                EvidenceUnit(
                    unit["id"],
                    source.name,
                    unit["start"],
                    unit["end"],
                    source_text,
                    tuple(unit.get("entity_types", [])),
                    role=unit.get("role", "primary"),
                )
                for unit in catalog["units"]
            )
            chunk = EvidenceChunk(source.name, entry["product"], content, units)
            profile.validate_catalog(units)
            prompt = (
                PROMPTS["entity_extraction_json_user_prompt"]
                .format(**context, input_text=content, heading_context_block="")
                .strip()
            )
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt in requests or key not in approved or key not in ledger:
                raise ValueError("Registered evidence input is missing or ambiguous")
            row = approved[key]
            if (
                row.get("source") != source.name
                or row.get("content_sha256") != content_hash
            ):
                raise ValueError("Registered approval identity changed")
            if row.get("prompt_sha256") != key:
                raise ValueError("Registered approval prompt changed")
            selection = PointerResult.model_validate(ledger[key]["selection"])
            native = to_native(selection, chunk, profile=profile)
            if native != ledger[key]["native"] or native != row["native"]:
                raise ValueError(
                    "Registered native output no longer matches its pointers"
                )
            requests[prompt] = chunk
            selections[key] = {"selection": selection.model_dump(), "native": native}
        documents.append((entry, payload))
    if len(requests) != len(approved) or set(selections) != set(ledger):
        raise ValueError("Registered approval scope differs from its source manifest")
    return ReviewedBatch(
        name, batch, manifest, config, documents, requests, selections, approved
    )


def batch_record(batch: ReviewedBatch) -> dict:
    return {
        "batch": batch.name,
        **{
            name.replace(".json", "").replace("-", "_") + "_sha256": sha256(
                batch.path / name
            )
            for name in (
                "manifest.json",
                "approved-extractions.json",
                "evidence-ledger.json",
            )
        },
    }


def load_registered_batches(root: Path, workspace: str, base_batch: str | None = None):
    from applications.insurance.database_storage import (
        STORAGE_DEFAULTS,
        storage_contract,
    )

    location = registry_path(root, workspace)
    if location.exists():
        registry = read_json(location)
        if registry.get("version") != 1 or registry.get("workspace") != workspace:
            raise ValueError("Workspace registry identity is invalid")
        records = registry["batches"]
        if not records or len({record["batch"] for record in records}) != len(records):
            raise ValueError("Workspace registry batches are empty or duplicated")
        if base_batch and (not records or records[0]["batch"] != base_batch):
            raise ValueError(
                "Requested base batch differs from the registered workspace"
            )
        batches = []
        for record in records:
            batch = load_reviewed_batch(root, record["batch"])
            if batch_record(batch) != record:
                raise ValueError("A registered batch or evidence ledger was changed")
            if embedding_contract(batch.config) != registry["embedding"]:
                raise ValueError("A registered batch embedding contract was changed")
            if storage_contract(batch.config) != registry.get(
                "storage", STORAGE_DEFAULTS
            ):
                raise ValueError(
                    "A registered batch storage backend or database was changed"
                )
            batches.append(batch)
        return registry, batches
    if not base_batch:
        raise ValueError(
            "Existing workspace requires a registry or explicit --base-batch"
        )
    base = load_reviewed_batch(root, base_batch)
    if base.manifest["workspace"] != workspace or base.config["WORKSPACE"] != workspace:
        raise ValueError("The base batch does not own the requested workspace")
    return {
        "version": 1,
        "workspace": workspace,
        "embedding": embedding_contract(base.config),
        "storage": storage_contract(base.config),
        "batches": [batch_record(base)],
    }, [base]


def validate_additions(previous_documents: list, new_documents: list) -> None:
    from lightrag.utils_pipeline import normalize_document_file_path

    products = {entry["product"] for entry, _ in previous_documents}
    labels = {
        normalize_document_file_path(entry["ingestion_label"])
        for entry, _ in previous_documents
    }
    full_contents = {payload["text"].strip() for _, payload in previous_documents}
    contents = {
        content.strip()
        for _, payload in previous_documents
        for content in payload["text"].split(
            payload["chunking"]["params"]["split_by_character"]
        )
    }
    new_products = {entry["product"] for entry, _ in new_documents}
    if products & new_products:
        raise ValueError(
            "An added product is already registered; use a reviewed update workflow"
        )
    for entry, payload in new_documents:
        label = normalize_document_file_path(entry["ingestion_label"])
        if label in labels or payload["text"].strip() in full_contents:
            raise ValueError("Added document collides with an existing native document")
        labels.add(label)
        full_contents.add(payload["text"].strip())
        for content in payload["text"].split(
            payload["chunking"]["params"]["split_by_character"]
        ):
            normalized = content.strip()
            if normalized in contents:
                raise ValueError("Added chunk collides with an existing native chunk")
            contents.add(normalized)


def assert_workspace_stopped(root: Path, workspace: str) -> None:
    import psutil

    for runtime in (root / "output").glob("*/runtime.env"):
        config = dotenv_values(runtime, interpolate=False)
        if config.get("WORKSPACE") != workspace:
            continue
        pid_file = runtime.parent / "server.pid"
        if not pid_file.exists():
            continue
        try:
            process = psutil.Process(int(pid_file.read_text().strip()))
            command = process.cmdline()
            if (
                process.is_running()
                and Path(process.cwd()).resolve() == root.resolve()
                and (
                    any(Path(part).name == "lightrag-server" for part in command)
                    or any(
                        left == "-m" and right == "applications.insurance.server"
                        for left, right in itertools.pairwise(command)
                    )
                )
            ):
                raise ValueError(
                    "Stop every server for this workspace before appending"
                )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue


@contextmanager
def workspace_writer(root: Path, workspace: str):
    """Reserve one operator writer without altering native storage files."""
    path = registry_path(root, workspace).parent / "writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "A workspace writer reservation already exists; inspect it"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        assert_workspace_stopped(root, workspace)
        yield
    finally:
        path.unlink(missing_ok=True)


def file_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Workspace backup refuses symbolic links")
        if path.is_file():
            result[str(path.relative_to(directory))] = sha256(path)
    return result


def backup_workspace(target: Path, batch: Path) -> Path:
    backup = batch / "backups" / "before-append"
    if backup.exists():
        raise ValueError(
            "A prior append backup exists; inspect its journal before retrying"
        )
    before = file_hashes(target)
    if not before:
        raise ValueError("Existing native workspace is empty")
    shutil.copytree(target, backup)
    if file_hashes(backup) != before or file_hashes(target) != before:
        raise RuntimeError("Workspace changed while making the append backup")
    write_json(batch / "before-append-files.json", before)
    return backup


def verify_preserved(backup: Path, target: Path) -> dict:
    """Check durable old document carriers and graph attribution after append."""
    import networkx as nx

    issues = []
    for namespace in (
        "doc_status",
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
    ):
        name = f"kv_store_{namespace}.json"
        before, after = read_json(backup / name), read_json(target / name)
        for identifier, row in before.items():
            if identifier not in after or after[identifier] != row:
                issues.append(f"old_{namespace}_changed:{identifier}")
    old = nx.read_graphml(backup / "graph_chunk_entity_relation.graphml")
    new = nx.read_graphml(target / "graph_chunk_entity_relation.graphml")
    old_tracking, new_tracking = {}, {}
    for namespace in ("entity_chunks", "relation_chunks"):
        old_path = backup / f"kv_store_{namespace}.json"
        new_path = target / f"kv_store_{namespace}.json"
        old_tracking[namespace] = read_json(old_path) if old_path.exists() else {}
        new_tracking[namespace] = read_json(new_path) if new_path.exists() else {}
        for identifier, row in old_tracking[namespace].items():
            current = new_tracking[namespace].get(identifier)
            if current is None or not set(row.get("chunk_ids", [])) <= set(
                current.get("chunk_ids", [])
            ):
                issues.append(f"old_{namespace}_attribution_lost:{identifier}")
    for name, row in old.nodes(data=True):
        current = new.nodes.get(name)
        tracked = name in old_tracking["entity_chunks"]
        if current is None or (
            not tracked
            and not set(row.get("source_id", "").split("<SEP>"))
            <= set(current.get("source_id", "").split("<SEP>"))
        ):
            issues.append(f"old_entity_attribution_lost:{name}")
    for source, destination, row in old.edges(data=True):
        current = new.get_edge_data(source, destination)
        tracked = (
            "<SEP>".join(sorted((source, destination)))
            in old_tracking["relation_chunks"]
        )
        if current is None or (
            not tracked
            and not set(row.get("source_id", "").split("<SEP>"))
            <= set(current.get("source_id", "").split("<SEP>"))
        ):
            issues.append(f"old_relation_attribution_lost:{source}:{destination}")
    return {
        "passed": not issues,
        "issues": issues,
        "method": "Old documents, chunks and recovery anchors unchanged; old graph attribution retained",
    }


def workspace_snapshot(root: Path, workspace: str) -> Path:
    """Materialize a cumulative manifest for the existing strict readback tool."""
    registry, batches = load_registered_batches(root, workspace)
    return snapshot_for_batches(root, workspace, batches, registry)


def snapshot_for_batches(
    root: Path, workspace: str, batches: list, registry: dict
) -> Path:
    fingerprint = hashlib.sha256(
        json.dumps(registry, sort_keys=True).encode()
    ).hexdigest()[:16]
    snapshot = registry_path(root, workspace).parent / "snapshots" / fingerprint
    entries = []
    source_roots = {batch.manifest["source_root"] for batch in batches}
    if len(source_roots) != 1:
        raise ValueError("Registered imports use different source roots")
    for batch in batches:
        for original in batch.manifest["documents"]:
            entry = dict(original)
            for field in ("prepared_path", "catalog_path", "source_map_path"):
                if field not in entry:
                    continue
                source = batch.path / entry[field]
                relative = Path("batches") / batch.name / entry[field]
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha256(destination) != sha256(source):
                    raise ValueError("Cumulative readback snapshot was modified")
                if not destination.exists():
                    shutil.copy2(source, destination)
                entry[field] = str(relative)
            entry["import_batch"] = batch.name
            entries.append(entry)
    write_json(
        snapshot / "manifest.json",
        {
            "workspace": workspace,
            "batch": "workspace-cumulative-snapshot",
            "source_root": source_roots.pop(),
            "expected_product_count": len({entry["product"] for entry in entries}),
            "documents": entries,
            "batches": [batch.name for batch in batches],
        },
    )
    return snapshot


def commit_append(root: Path, workspace: str, registry: dict, batch_name: str) -> Path:
    from applications.insurance.database_storage import (
        STORAGE_DEFAULTS,
        storage_contract,
    )

    batch = load_reviewed_batch(root, batch_name)
    if storage_contract(batch.config) != registry.get("storage", STORAGE_DEFAULTS):
        raise ValueError("Cannot append across different storage backends or databases")
    if batch_name in {record["batch"] for record in registry["batches"]}:
        raise ValueError("Import batch is already registered")
    updated = {**registry, "batches": [*registry["batches"], batch_record(batch)]}
    location = registry_path(root, workspace)
    write_json(location, updated)
    return workspace_snapshot(root, workspace)


def commit_initial_database(root: Path, workspace: str, batch_name: str) -> Path:
    """Publish the first successful import into an empty database workspace."""
    from applications.insurance.database_storage import (
        is_database_backend,
        storage_contract,
    )

    location = registry_path(root, workspace)
    if location.exists():
        raise ValueError("A workspace registry already exists; use incremental import")
    batch = load_reviewed_batch(root, batch_name)
    if not is_database_backend(batch.config) or batch.config["WORKSPACE"] != workspace:
        raise ValueError(
            "Initial database registration has the wrong backend or workspace"
        )
    write_json(
        location,
        {
            "version": 1,
            "workspace": workspace,
            "embedding": embedding_contract(batch.config),
            "storage": storage_contract(batch.config),
            "batches": [batch_record(batch)],
        },
    )
    return workspace_snapshot(root, workspace)
