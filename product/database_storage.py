"""Database configuration and strict, workspace-scoped verification exports.

Exports are audit/backup artifacts only. PostgreSQL and Neo4j remain the runtime
stores; no exported JSON or GraphML is consulted by retrieval or ingestion.
"""

from __future__ import annotations

import base64
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

STORAGE_DEFAULTS = {
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "JsonDocStatusStorage",
}
DATABASE_STORES = {
    "kv_storage": "PGKVStorage",
    "vector_storage": "PGVectorStorage",
    "graph_storage": "Neo4JStorage",
    "doc_status_storage": "PGDocStatusStorage",
}
KV_NAMES = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "llm_response_cache",
    "doc_status",
)


def storage_options(config: dict) -> dict[str, str]:
    options = {
        key: str(config.get("LIGHTRAG_" + key.upper()) or default)
        for key, default in STORAGE_DEFAULTS.items()
    }
    if options not in (STORAGE_DEFAULTS, DATABASE_STORES):
        raise ValueError(
            "Product imports support the complete file or PostgreSQL/Neo4j storage set"
        )
    return options


def is_database_backend(config: dict) -> bool:
    return storage_options(config) == DATABASE_STORES


def storage_contract(config: dict) -> dict:
    result: dict[str, Any] = dict(storage_options(config))
    if result == DATABASE_STORES:
        workspace = str(config.get("WORKSPACE") or "")
        for key in ("POSTGRES_WORKSPACE", "NEO4J_WORKSPACE"):
            if config.get(key) and str(config[key]).strip() != workspace:
                raise ValueError("Database workspace overrides must match WORKSPACE")
        uri = urlsplit(str(config.get("NEO4J_URI") or ""))
        result["postgres"] = {
            "host": str(config.get("POSTGRES_HOST") or "localhost"),
            "port": int(config.get("POSTGRES_PORT") or 5432),
            "database": str(config.get("POSTGRES_DATABASE") or ""),
        }
        result["neo4j"] = {
            "scheme": uri.scheme,
            "host": uri.hostname,
            "port": uri.port,
            "database": str(config.get("NEO4J_DATABASE") or ""),
        }
        if not result["postgres"]["database"] or not result["neo4j"]["database"]:
            raise ValueError("Both PostgreSQL and Neo4j databases must be explicit")
    return result


def _table_name(name: str) -> str:
    # Names originate in native storage metadata, not user SQL fragments.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Native database table name is invalid")
    return name


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Unsupported database export value: {type(value).__name__}")


def _write(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    )


async def _ids(storage) -> list[str]:
    table = getattr(storage, "table_name", None)
    if not table:
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table = namespace_to_table_name(storage.namespace)
    table = _table_name(table)
    rows = await storage.db.query(
        f"SELECT id FROM {table} WHERE workspace=$1 ORDER BY id",
        [storage.workspace],
        multirows=True,
    )
    ids = [str(row["id"]) for row in rows or []]
    if len(ids) != len(set(ids)):
        raise ValueError("Database snapshot found duplicate native IDs")
    return ids


async def capture_database_snapshot(rag, directory: Path) -> Path:
    """Export every native namespace, failing on missing or partial DB reads.

    Call only after the writer has flushed all pending vector operations. Native
    SQL enumeration uses workspace parameters; point reads use public storage
    methods for their canonical JSON normalization.
    """
    import networkx as nx
    import numpy as np

    directory.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name in KV_NAMES:
        storage = getattr(rag, name)
        ids = await _ids(storage)
        result = {}
        for offset in range(0, len(ids), 200):
            batch = ids[offset : offset + 200]
            # Point reads are strict in PGKV/PGDocStatus; bulk vector reads are
            # deliberately not used because they can swallow backend errors.
            for identifier in batch:
                row = await storage.get_by_id(identifier)
                if row is None:
                    raise ValueError("A database row disappeared during verification")
                result[identifier] = row
        filename = "llm_response_cache" if name == "llm_response_cache" else name
        _write(directory / f"kv_store_{filename}.json", result)
        counts[name] = len(result)
    graph = nx.Graph()
    for row in await rag.chunk_entity_relation_graph.get_all_nodes():
        values = dict(row)
        identifier = values.pop("id", None) or values.get("entity_id")
        if not identifier or identifier in graph:
            raise ValueError("Database graph has a missing or repeated entity identity")
        graph.add_node(
            identifier,
            **{
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list))
                else v
                for k, v in values.items()
                if v is not None
            },
        )
    for row in await rag.chunk_entity_relation_graph.get_all_edges():
        values = dict(row)
        source, target = values.pop("source"), values.pop("target")
        if source not in graph or target not in graph:
            raise ValueError("Database edge references a missing node")
        if graph.has_edge(source, target):
            # Neo4j's native undirected enumeration returns both orientations.
            if graph[source][target] != {
                k: v for k, v in values.items() if v is not None
            }:
                raise ValueError("Database graph has conflicting parallel relations")
            continue
        graph.add_edge(
            source, target, **{k: v for k, v in values.items() if v is not None}
        )
    nx.write_graphml(graph, directory / "graph_chunk_entity_relation.graphml")
    for name in ("entities", "relationships", "chunks"):
        storage = getattr(rag, name + "_vdb")
        table = _table_name(storage.table_name)
        rows = await storage.db.query(
            f"SELECT *, content_vector::text AS exported_vector FROM {table} WHERE workspace=$1 ORDER BY id",
            [storage.workspace],
            multirows=True,
        )
        payload, vectors = [], []
        for raw in rows or []:
            row = dict(raw)
            vector = np.asarray(
                json.loads(row.pop("exported_vector")), dtype=np.float32
            )
            row.pop("content_vector", None)
            if (
                vector.shape != (rag.embedding_func.embedding_dim,)
                or not np.isfinite(vector).all()
                or not np.linalg.norm(vector)
            ):
                raise ValueError(
                    "Database vector is missing, invalid or has the wrong dimension"
                )
            row["__id__"] = row["id"]
            if name == "relationships":
                row["src_id"], row["tgt_id"] = row["source_id"], row["target_id"]
            payload.append(row)
            vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float32).reshape(
            (-1, rag.embedding_func.embedding_dim)
        )
        _write(
            directory / f"vdb_{name}.json",
            {
                "embedding_dim": rag.embedding_func.embedding_dim,
                "data": payload,
                "matrix": base64.b64encode(matrix.tobytes()).decode(),
            },
        )
        counts["vector_" + name] = len(payload)
    counts.update(
        entities=graph.number_of_nodes(), relationships=graph.number_of_edges()
    )
    _write(
        directory / "database-export.json",
        {
            "workspace": rag.workspace,
            "purpose": "verification and backup artifact; not a runtime storage backend",
            "counts": counts,
        },
    )
    return directory


@asynccontextmanager
async def database_reader(config: dict, root: Path):
    """Open the configured native stores without any model/embedding requests."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    if not is_database_backend(config):
        raise ValueError("Database readback requires PostgreSQL and Neo4j backends")
    storage_contract(config)
    os.environ.update({k: str(v) for k, v in config.items() if v is not None})

    async def forbidden(*_args, **_kwargs):
        raise RuntimeError("Database verification must not invoke a model")

    rag = LightRAG(
        working_dir=str(root / "rag_storage"),
        workspace=config["WORKSPACE"],
        llm_model_func=forbidden,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(config["EMBEDDING_DIM"]),
            max_token_size=int(config.get("EMBEDDING_TOKEN_LIMIT") or 8192),
            model_name=config["EMBEDDING_MODEL"],
            func=forbidden,
        ),
        **storage_options(config),
    )
    try:
        await rag.initialize_storages()
        yield rag
    finally:
        await rag.finalize_storages()


async def snapshot_database_config(config: dict, root: Path, directory: Path) -> Path:
    async with database_reader(config, root) as rag:
        return await capture_database_snapshot(rag, directory)
