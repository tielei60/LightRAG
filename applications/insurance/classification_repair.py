"""Prepare source-checked classification corrections for native graph stores.

The caller reserves the workspace writer, saves a complete backup and persists
the returned plan in a write-ahead journal before applying it. Historical import
approvals are evidence for this review; this module never rewrites them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from applications.insurance.evidence_extract import PointerResult, to_native
from lightrag.utils import make_relation_chunk_key


@dataclass(frozen=True)
class ClassificationCorrection:
    approval_key: str
    entity_name: str
    entity_type_before: str
    entity_type_after: str
    relation_source: str
    relation_target: str
    relation_keywords_before: str
    relation_keywords_after: str


def _one(rows, predicate, message):
    matches = [index for index, row in enumerate(rows) if predicate(row)]
    if len(matches) != 1:
        raise ValueError(message)
    return matches[0]


def _cache_identity(row):
    return {
        key: row.get(key)
        for key in (
            "id",
            "original_prompt",
            "return",
            "chunk_id",
            "cache_type",
            "queryparam",
            "create_time",
        )
    }


def _vector_identity(row):
    return {
        key: row[key]
        for key in ("id", "source_id", "target_id", "content", "chunk_ids", "file_path")
    }


def prepare_classification_repair(
    batch,
    profile,
    corrections: list[ClassificationCorrection],
    *,
    text_chunks: dict,
    cache_rows: dict,
    graph_nodes: dict,
    graph_edges: dict,
    relation_vectors: list[dict],
    entity_chunks: dict,
    relation_chunks: dict,
) -> dict[str, Any]:
    """Build a reviewable plan; only entity types and relation keywords may differ.

    Inputs must be strict reads made during the caller's stopped-writer window.
    ``graph_edges`` uses (source, target) keys; vector rows use PostgreSQL's raw
    source_id/target_id endpoint columns and chunk_ids provenance column.
    """
    if not corrections:
        raise ValueError("A reviewed classification correction is required")
    requests = {
        hashlib.sha256(prompt.encode()).hexdigest(): chunk
        for prompt, chunk in batch.requests.items()
    }
    plan = {
        key: {}
        for key in (
            "caches",
            "nodes",
            "edges",
            "vectors",
            "entity_tracking",
            "relation_tracking",
        )
    }
    seen = set()
    for correction in corrections:
        if correction.approval_key in seen or correction.entity_name in plan["nodes"]:
            raise ValueError("Duplicate reviewed classification correction")
        seen.add(correction.approval_key)
        key = correction.approval_key
        chunk = requests[key]
        old = copy.deepcopy(batch.approved[key]["native"])
        if old != batch.selections[key]["native"]:
            raise ValueError("Frozen approval and evidence ledger differ")
        entity_index = _one(
            old["entities"],
            lambda row, name=correction.entity_name: row["name"] == name,
            "Reviewed entity identity is missing or ambiguous",
        )
        pair = (correction.relation_source, correction.relation_target)
        relation_index = _one(
            old["relationships"],
            lambda row, pair=pair: (row["source"], row["target"]) == pair,
            "Reviewed relation identity is missing or ambiguous",
        )
        if (
            old["entities"][entity_index]["type"] != correction.entity_type_before
            or old["relationships"][relation_index]["keywords"]
            != correction.relation_keywords_before
            or correction.entity_type_before == correction.entity_type_after
            or correction.relation_keywords_before == correction.relation_keywords_after
        ):
            raise ValueError("Reviewed classification before-state differs")
        selection = copy.deepcopy(batch.selections[key]["selection"])
        selection["entities"][entity_index]["type"] = correction.entity_type_after
        selection["relationships"][relation_index]["keywords"] = (
            correction.relation_keywords_after
        )
        new = to_native(PointerResult.model_validate(selection), chunk, profile=profile)
        expected = copy.deepcopy(old)
        expected["entities"][entity_index]["type"] = correction.entity_type_after
        expected["relationships"][relation_index]["keywords"] = (
            correction.relation_keywords_after
        )
        if new != expected:
            raise ValueError(
                "A classification correction changed evidence or identities"
            )
        supporting_chunks = [
            (identifier, row)
            for identifier, row in text_chunks.items()
            if row.get("content") == chunk.content
            and row.get("file_path", "").endswith(chunk.source)
        ]
        if len(supporting_chunks) != 1:
            raise ValueError("Reviewed source chunk is missing or ambiguous")
        chunk_id, stored_chunk = supporting_chunks[0]
        tracking_key = make_relation_chunk_key(*pair)
        node_tracking = entity_chunks.get(correction.entity_name)
        edge_tracking = relation_chunks.get(tracking_key)
        if any(
            not row or row.get("chunk_ids") != [chunk_id] or row.get("count") != 1
            for row in (node_tracking, edge_tracking)
        ):
            raise ValueError(
                "Classification repair requires complete single-chunk tracking"
            )
        plan["entity_tracking"][correction.entity_name] = copy.deepcopy(node_tracking)
        plan["relation_tracking"][tracking_key] = copy.deepcopy(edge_tracking)
        extracted = 0
        for cache_id in stored_chunk.get("llm_cache_list", []):
            cached = cache_rows[cache_id]
            if cached.get("cache_type") != "extract":
                continue
            if (
                cached.get("chunk_id") != chunk_id
                or json.loads(cached["return"]) != old
                or cached.get("return_value", cached["return"]) != cached["return"]
            ):
                raise ValueError(
                    "Referenced extraction cache differs from reviewed approval"
                )
            updated = {
                **copy.deepcopy(cached),
                "return": json.dumps(new, ensure_ascii=False),
            }
            if "return_value" in updated:
                updated["return_value"] = updated["return"]
            plan["caches"][cache_id] = {
                "before": copy.deepcopy(cached),
                "after": updated,
            }
            extracted += 1
        if not extracted:
            raise ValueError("Reviewed source chunk has no durable extraction cache")
        node = graph_nodes[correction.entity_name]
        edge = graph_edges[pair]
        if (
            node["entity_type"] != correction.entity_type_before
            or edge["keywords"] != correction.relation_keywords_before
            or node["source_id"] != chunk_id
            or edge["source_id"] != chunk_id
        ):
            raise ValueError(
                "Native graph classification differs from reviewed before-state"
            )
        plan["nodes"][correction.entity_name] = {
            "before": copy.deepcopy(node),
            "after": {
                **copy.deepcopy(node),
                "entity_type": correction.entity_type_after,
            },
        }
        edge_key = json.dumps(pair, ensure_ascii=False)
        if edge_key in plan["edges"]:
            raise ValueError("Duplicate reviewed relation correction")
        plan["edges"][edge_key] = {
            "source": pair[0],
            "target": pair[1],
            "before": copy.deepcopy(edge),
            "after": {
                **copy.deepcopy(edge),
                "keywords": correction.relation_keywords_after,
            },
        }
        vector_index = _one(
            relation_vectors,
            lambda row, pair=pair: {row["source_id"], row["target_id"]} == set(pair),
            "Native relationship vector is missing or ambiguous",
        )
        vector = relation_vectors[vector_index]
        prefix = correction.relation_keywords_before + "\t"
        if (
            not vector["content"].startswith(prefix)
            or vector["chunk_ids"] != [chunk_id]
            or not vector["file_path"]
        ):
            raise ValueError("Relationship vector content or provenance differs")
        plan["vectors"][vector["id"]] = {
            "before": _vector_identity(vector),
            "after": {
                "src_id": vector["source_id"],
                "tgt_id": vector["target_id"],
                "source_id": "<SEP>".join(vector["chunk_ids"]),
                "file_path": vector["file_path"],
                "content": correction.relation_keywords_after
                + "\t"
                + vector["content"][len(prefix) :],
            },
        }
    return {"corrections": [asdict(item) for item in corrections], **plan}


async def apply_classification_repair(rag, plan: dict) -> None:
    """Check all before-states, then flush caches before graph/vector corrections.

    Failure leaves the caller's journal and backup for inspection. This function
    does not claim transactional rollback across PostgreSQL and Neo4j.
    """
    graph = rag.chunk_entity_relation_graph
    for namespace in ("entity_tracking", "relation_tracking"):
        storage = (
            rag.entity_chunks if namespace == "entity_tracking" else rag.relation_chunks
        )
        for identifier, expected in plan[namespace].items():
            if await storage.get_by_id(identifier) != expected:
                raise ValueError(
                    "Authoritative chunk tracking changed after repair review"
                )
    for identifier, change in plan["caches"].items():
        current = await rag.llm_response_cache.get_by_id(identifier)
        if current is None or _cache_identity(current) != _cache_identity(
            change["before"]
        ):
            raise ValueError("Extraction cache changed after repair review")
    for name, change in plan["nodes"].items():
        if await graph.get_node(name) != change["before"]:
            raise ValueError("Graph entity changed after repair review")
    for change in plan["edges"].values():
        if await graph.get_edge(change["source"], change["target"]) != change["before"]:
            raise ValueError("Graph relation changed after repair review")
    storage = rag.relationships_vdb
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", storage.table_name):
        raise ValueError("Native relationship table name is invalid")
    for identifier, change in plan["vectors"].items():
        rows = await storage.db.query(
            f"SELECT id,source_id,target_id,content,chunk_ids,file_path FROM {storage.table_name} WHERE workspace=$1 AND id=$2",
            [storage.workspace, identifier],
            multirows=True,
        )
        if len(rows or []) != 1 or _vector_identity(dict(rows[0])) != change["before"]:
            raise ValueError("Relationship vector changed after repair review")
    await rag.llm_response_cache.upsert(
        {key: change["after"] for key, change in plan["caches"].items()}
    )
    await rag.llm_response_cache.index_done_callback()
    for name, change in plan["nodes"].items():
        await graph.upsert_node(name, change["after"])
    for change in plan["edges"].values():
        await graph.upsert_edge(change["source"], change["target"], change["after"])
    await graph.index_done_callback()
    await storage.upsert(
        {key: change["after"] for key, change in plan["vectors"].items()}
    )
    await storage.index_done_callback()
