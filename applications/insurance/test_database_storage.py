"""Offline database export checks; no live databases or providers are used."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from applications.insurance.database_storage import (
    DATABASE_STORES,
    KV_NAMES,
    capture_database_snapshot,
    is_database_backend,
    storage_contract,
    storage_options,
)


def config():
    return {
        **{"LIGHTRAG_" + k.upper(): v for k, v in DATABASE_STORES.items()},
        "WORKSPACE": "master",
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DATABASE": "lightrag_products",
        "NEO4J_URI": "bolt://private:secret@localhost:7687",
        "NEO4J_DATABASE": "neo4j",
    }


def test_storage_selection_preserves_file_default_and_requires_complete_database_set():
    assert storage_options({})["kv_storage"] == "JsonKVStorage"
    assert is_database_backend(config())
    with pytest.raises(ValueError, match="complete file"):
        storage_options({"LIGHTRAG_KV_STORAGE": "PGKVStorage"})


def test_database_contract_tracks_database_location_without_credentials():
    current = config()
    contract = storage_contract(current)
    assert contract["postgres"]["port"] == 5433
    assert "secret" not in json.dumps(contract)
    assert "private" not in json.dumps(contract)
    assert contract != storage_contract({**current, "POSTGRES_DATABASE": "other"})
    assert contract != storage_contract({**current, "NEO4J_DATABASE": "other"})


@pytest.mark.parametrize("field", ["POSTGRES_WORKSPACE", "NEO4J_WORKSPACE"])
def test_database_contract_refuses_cross_workspace_override(field):
    with pytest.raises(ValueError, match="overrides"):
        storage_contract({**config(), field: "unexpected"})
    assert storage_contract({**config(), field: "master"})


def snapshot_fixture():
    queries = []
    rag = SimpleNamespace(
        workspace="master", embedding_func=SimpleNamespace(embedding_dim=2)
    )
    for name in KV_NAMES:
        row = {"content": "完整原文", "chunk_ids": ["chunk-1"]}
        if name == "doc_status":
            row = {
                "status": "processed",
                "metadata": {
                    "kg_write_state": "graph_mutation_started",
                    "kg_purge": {"phase": "completed"},
                },
            }
        if name == "llm_response_cache":
            row = {"return": "核准抽取", "cache_type": "extract", "chunk_id": "chunk-1"}

        async def query(sql, args, multirows, name=name):
            queries.append((name, sql, args))
            return [{"id": name + "-id"}]

        setattr(
            rag,
            name,
            SimpleNamespace(
                table_name="table_" + name,
                workspace="master",
                db=SimpleNamespace(query=query),
                get_by_id=AsyncMock(return_value=row),
            ),
        )
    graph = SimpleNamespace(
        get_all_nodes=AsyncMock(
            return_value=[
                {"id": "产品甲", "description": "原文", "source_id": "chunk-1"},
                {"id": "产品甲｜责任", "description": "原文", "source_id": "chunk-1"},
            ]
        ),
        get_all_edges=AsyncMock(
            return_value=[
                {
                    "source": "产品甲",
                    "target": "产品甲｜责任",
                    "description": "原文",
                    "source_id": "chunk-1",
                },
                {
                    "source": "产品甲｜责任",
                    "target": "产品甲",
                    "description": "原文",
                    "source_id": "chunk-1",
                },
            ]
        ),
    )
    rag.chunk_entity_relation_graph = graph
    for name in ["entities", "relationships", "chunks"]:
        row = {
            "id": name + "-vector",
            "content": "完整原文",
            "exported_vector": "[0.5, 0.8]",
            "content_vector": object(),
        }
        if name == "entities":
            row["entity_name"] = "产品甲"
        if name == "relationships":
            row.update(source_id="产品甲", target_id="产品甲｜责任")

        async def query(sql, args, multirows, name=name, row=row):
            queries.append((name, sql, args))
            return [row]

        setattr(
            rag,
            name + "_vdb",
            SimpleNamespace(
                table_name="vector_" + name,
                workspace="master",
                db=SimpleNamespace(query=query),
            ),
        )
    return rag, queries


@pytest.mark.asyncio
async def test_database_export_retains_cache_recovery_and_tracking_and_scopes_every_query(
    tmp_path,
):
    rag, queries = snapshot_fixture()
    exported = await capture_database_snapshot(rag, tmp_path / "export")
    assert len(queries) == 11
    assert all(args == ["master"] and "workspace=$1" in sql for _, sql, args in queries)
    status = json.loads((exported / "kv_store_doc_status.json").read_text())
    assert status["doc_status-id"]["metadata"]["kg_purge"]["phase"] == "completed"
    cache = json.loads((exported / "kv_store_llm_response_cache.json").read_text())
    assert cache["llm_response_cache-id"]["return"] == "核准抽取"
    tracking = json.loads((exported / "kv_store_relation_chunks.json").read_text())
    assert tracking["relation_chunks-id"]["chunk_ids"] == ["chunk-1"]
    graph = nx.read_graphml(exported / "graph_chunk_entity_relation.graphml")
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 1
    vectors = json.loads((exported / "vdb_relationships.json").read_text())
    assert vectors["data"][0]["src_id"] == "产品甲"
    assert vectors["data"][0]["tgt_id"] == "产品甲｜责任"
    assert vectors["data"][0]["__id__"] == "relationships-vector"
    assert "content_vector" not in vectors["data"][0]
    assert vectors["embedding_dim"] == 2


@pytest.mark.asyncio
async def test_database_export_fails_closed_on_missing_kv_row(tmp_path):
    rag, _ = snapshot_fixture()
    rag.full_entities.get_by_id.return_value = None
    with pytest.raises(ValueError, match="disappeared"):
        await capture_database_snapshot(rag, tmp_path / "export")
    assert not (tmp_path / "export/database-export.json").exists()


@pytest.mark.asyncio
async def test_database_export_propagates_database_failure_instead_of_empty_data(
    tmp_path,
):
    rag, _ = snapshot_fixture()
    rag.full_docs.db.query = AsyncMock(
        side_effect=ConnectionError("database unavailable")
    )
    with pytest.raises(ConnectionError):
        await capture_database_snapshot(rag, tmp_path / "export")


@pytest.mark.asyncio
async def test_database_export_rejects_wrong_vector_dimension(tmp_path):
    rag, _ = snapshot_fixture()
    rag.chunks_vdb.db.query = AsyncMock(
        return_value=[{"id": "bad", "exported_vector": "[1,2,3]"}]
    )
    with pytest.raises(ValueError, match="dimension"):
        await capture_database_snapshot(rag, tmp_path / "export")


@pytest.mark.asyncio
async def test_database_export_rejects_conflicting_undirected_duplicate_edges(tmp_path):
    rag, _ = snapshot_fixture()
    rag.chunk_entity_relation_graph.get_all_edges.return_value[1]["description"] = (
        "不同关系"
    )
    with pytest.raises(ValueError, match="parallel relations"):
        await capture_database_snapshot(rag, tmp_path / "export")
