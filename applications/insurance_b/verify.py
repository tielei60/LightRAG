"""Closed-writer native readback of the independent B workspace."""

import math
import re
import unicodedata

import asyncpg

from .common import digest, read_json, save_json
from .config import Settings
from .ingest import run_contract, run_signature
from .models import initialize_rag, make_rag
from .prepare import checked_manifest


def foreign_product_entities(
    names: list[str], expected: str, products: set[str]
) -> list[str]:
    normalize = lambda value: unicodedata.normalize("NFKC", value)
    other = {
        normalize(product)
        for product in products
        if normalize(product) != normalize(expected)
    }
    return [
        name
        for name in names
        if any(
            normalize(name) == product or normalize(name).startswith(product + "|")
            for product in other
        )
    ]


def verified_chunk_strategy(settings, rag, metadata, prepared=None):
    if settings.get("PREPARATION") != "business":
        return metadata.get("chunk_method") == "paragraph_semantic"
    from lightrag.chunker import chunking_by_token_size
    from lightrag.pipeline import _format_chunking_params

    if (
        metadata.get("chunk_method") not in {"fixed_token", "legacy_chunking_func"}
        or rag.chunking_func is not chunking_by_token_size
        or prepared is None
    ):
        return False
    params = prepared["chunking"]["params"]
    expected = _format_chunking_params(
        8192,
        {
            "split_by_character": params["split_by_character"],
            "split_by_character_only": True,
            "overlap": 0,
        },
    )
    return metadata.get("chunk_opts") == expected


async def verify_vectors(
    settings: Settings, rag, chunk_ids: set, names: set, pairs: set
) -> tuple[dict, list]:
    """Compare actual native vector IDs with the persisted graph and chunk set."""
    from lightrag.utils import compute_mdhash_id

    errors = []
    connection = await asyncpg.connect(
        host=settings.get("POSTGRES_HOST"),
        port=settings.integer("POSTGRES_PORT"),
        database=settings.get("POSTGRES_DATABASE"),
        user=settings.get("POSTGRES_USER"),
        password=settings.get("POSTGRES_PASSWORD"),
    )
    vectors = {}
    try:
        async with connection.transaction(readonly=True):
            expected_ids = {
                "chunks": chunk_ids,
                "entities": {compute_mdhash_id(name, prefix="ent-") for name in names},
                "relationships": {
                    compute_mdhash_id(source + target, prefix="rel-")
                    for source, target in pairs
                },
            }
            for kind in ["chunks", "entities", "relationships"]:
                storage = getattr(rag, kind + "_vdb")
                table = storage.table_name
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                    raise ValueError("Invalid native vector table identifier")
                rows = await connection.fetch(
                    f"SELECT id, vector_dims(content_vector) AS dim, vector_norm(content_vector) AS norm FROM {table} WHERE workspace=$1",
                    settings.workspace,
                )
                vectors[kind] = len(rows)
                if {row["id"] for row in rows} != expected_ids[kind]:
                    errors.append(
                        f"Native {kind} vector IDs differ from the graph/chunks"
                    )
                if any(
                    row["dim"] != settings.integer("EMBEDDING_DIM")
                    or row["norm"] is None
                    or not math.isfinite(row["norm"])
                    or row["norm"] <= 0
                    for row in rows
                ):
                    errors.append(f"Invalid {kind} vector dimensions or values")
    finally:
        await connection.close()
    return vectors, errors


async def verify(settings: Settings) -> dict:
    manifest = checked_manifest(settings)
    if read_json(settings.run_dir / "run-snapshot.json") != run_contract(settings):
        raise ValueError("The frozen B import contract changed before readback")
    rag = make_rag(settings)
    errors, documents = [], []
    initialized = False
    try:
        await initialize_rag(rag, settings)
        initialized = True
        from lightrag.base import DocStatus

        statuses = await rag.doc_status.get_docs_by_statuses(
            [
                DocStatus.PROCESSED,
                DocStatus.PENDING,
                DocStatus.FAILED,
                DocStatus.PROCESSING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
            ],
            strict=True,
        )
        if len(statuses) != 21:
            errors.append(
                f"Full-corpus denominator is 21 documents; found {len(statuses)}"
            )
        expected_files = {
            entry["relative_path"].split("/")[-1]: entry
            for entry in manifest["documents"]
        }
        chunk_ids = set()
        seen_files = []
        a_prepared = (
            {
                row["entry"]["relative_path"].split("/")[-1]: row["prepared"]
                for row in read_json(settings.run_dir / "a-inputs.json")["documents"]
            }
            if settings.get("PREPARATION") == "business"
            else {}
        )
        for doc_id, status in statuses.items():
            name = status.file_path.split("/")[-1]
            seen_files.append(name)
            row = {"id": doc_id, "file": name, "status": DocStatus(status.status).value}
            documents.append(row)
            if name not in expected_files or row["status"] != "processed":
                errors.append(f"Unexpected or incomplete document: {name}")
                continue
            if not verified_chunk_strategy(
                settings, rag, status.metadata or {}, a_prepared.get(name)
            ):
                errors.append(
                    f"Document did not use its verified native chunk strategy: {name}"
                )
            catalog_path = settings.run_dir / "catalogs" / (doc_id + ".json")
            if not catalog_path.exists():
                errors.append(f"Final-chunk catalog missing: {name}")
                continue
            catalog = read_json(catalog_path)
            if settings.get("FINAL_CONTEXT") == "source-v1":
                from .input_adapters.source_context import unwrap

                proof_path = (
                    settings.run_dir / "final-chunk-adapter" / (doc_id + ".json")
                )
                if not proof_path.exists():
                    errors.append(
                        f"Native P/final source-context proof missing: {name}"
                    )
                else:
                    proof = read_json(proof_path)
                    if (
                        not proof.get("native_p_called")
                        or proof.get("requested_size") != settings.integer("P_SIZE")
                        or proof["origin"]["source_sha256"]
                        != expected_files[name]["sha256"]
                    ):
                        errors.append(
                            f"Native P or source identity proof differs: {name}"
                        )
                    if {r["sha256"] for r in proof["final_chunks"]} != {
                        r["indexed_sha256"] for r in catalog["chunks"]
                    }:
                        errors.append(
                            f"Final source context changed after native P: {name}"
                        )
                    for final in catalog["chunks"]:
                        try:
                            unwrap(
                                final["indexed_content"],
                                product=catalog["product"],
                                source=name,
                            )
                        except ValueError:
                            errors.append(
                                f"A final chunk lacks its verified identity envelope: {name}"
                            )
            ledger_path = settings.run_dir / "selections" / (doc_id + ".json")
            if not ledger_path.exists():
                errors.append(f"Validated extraction ledger missing: {name}")
            else:
                ledger = read_json(ledger_path)
                expected_prompts = {
                    digest(item["native_prompt"].strip()) for item in catalog["chunks"]
                }
                if (
                    len(expected_prompts) != len(catalog["chunks"])
                    or set(ledger["selections"]) != expected_prompts
                ):
                    errors.append(
                        f"Final chunks lack unambiguous per-document validated selections: {name}"
                    )
            if catalog["source_sha256"] != expected_files[name]["sha256"]:
                errors.append(f"Original hash mismatch: {name}")
            expected = {item["chunk_id"]: item for item in catalog["chunks"]}
            if set(status.chunks_list or []) != expected.keys():
                errors.append(f"Final chunk identities differ: {name}")
            for key, item in expected.items():
                value = await rag.text_chunks.get_by_id(key)
                if (
                    value is None
                    or value.get("full_doc_id") != doc_id
                    or digest(value.get("content", "")) != item["indexed_sha256"]
                ):
                    errors.append(f"Final native chunk missing or changed: {key}")
                chunk_ids.add(key)
            if await rag.full_docs.get_by_id(doc_id) is None:
                errors.append(f"Native full document missing: {name}")
            # Empty anchor ROWS are valid proof for a document owning no KG objects.
            for storage in [rag.full_entities, rag.full_relations]:
                anchor = await storage.get_by_id(doc_id)
                if anchor is None:
                    errors.append(f"Recovery anchor row missing: {name}")
                elif storage is rag.full_entities and foreign_product_entities(
                    anchor.get("entity_names", []),
                    catalog["product"],
                    {entry["product"] for entry in manifest["documents"]},
                ):
                    errors.append(
                        f"Document owns another corpus product's entity: {name}"
                    )
        if (
            len(seen_files) != len(set(seen_files))
            or set(seen_files) != expected_files.keys()
        ):
            errors.append(
                "The frozen source set is missing or has duplicate document identities"
            )
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        edges = await rag.chunk_entity_relation_graph.get_all_edges()
        names = {node.get("id") or node.get("entity_id") for node in nodes}
        pairs = {tuple(sorted([edge["source"], edge["target"]])) for edge in edges}
        for node in nodes:
            for key in str(node.get("source_id") or "").split("<SEP>"):
                if key and key not in chunk_ids:
                    errors.append("Graph entity references a non-corpus chunk")
        for edge in edges:
            ids = {
                key
                for key in str(edge.get("source_id") or "").split("<SEP>")
                if key and key not in {"UNKNOWN", "manual_creation"}
            }
            if not ids <= chunk_ids:
                errors.append("Graph relation references a non-corpus chunk")
            if float(edge.get("weight") or 0) < len(ids):
                errors.append("Graph relation weight is below the evidence floor")
        vectors, vector_errors = await verify_vectors(
            settings, rag, chunk_ids, names, pairs
        )
        errors.extend(vector_errors)
        report = {
            "passed": not errors,
            "run_contract_sha256": run_signature(settings),
            "workspace": settings.workspace,
            "expected_documents": 21,
            "documents": documents,
            "counts": {
                "documents": len(statuses),
                "chunks": len(chunk_ids),
                "entities": len(names),
                "relationships": len(pairs),
                "vectors": vectors,
            },
            "errors": list(dict.fromkeys(errors)),
            "business_acceptance": "not inferred from technical readback",
        }
        save_json(settings.run_dir / "readback-report.json", report)
        return report
    finally:
        if initialized:
            await rag.finalize_storages()
