"""Reviewed label corrections must survive native rebuilds without losing provenance."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from applications.insurance.classification_repair import (
    ClassificationCorrection,
    apply_classification_repair,
    prepare_classification_repair,
)
from applications.insurance.evidence_extract import (
    EvidenceChunk,
    EvidenceUnit,
    PointerResult,
    to_native,
)
from applications.insurance.extraction_profile import ExtractionProfile
from applications.insurance.paths import PROMPT_PROFILE_DIR
from lightrag.operate import (
    _get_cached_extraction_results,
    _rebuild_from_extraction_result,
)
from lightrag.utils import make_relation_chunk_key

PRODUCT = "北京人寿京福盈嘉终身寿险"
CASES = (
    (
        "解除合同手续及风险",
        "解除合同所需材料",
        "条款.pdf",
        "   7.1 您解除合同\n的手续及风\n险   \n本合同成立后， 您可以解除本合同， 请填写解除合同通知书并向我们提供下列证\n明和资料： \n1.保险合同；  \n2.您的有效身份证件。 \n自我们收到解除合同通知书时起，本合同终止。您在犹豫期后解除本合同的，我\n们自收到解除合同通知书之日起30 日内向您退还本合同的现金价值。 \n您在犹豫期后解除合同可能会遭受一定损失。 \n \n8． 投保人其他权益                                                             \n",
    ),
    (
        "解除合同流程",
        "退保所需材料",
        "产品说明书.pdf",
        "（二）退保 \n本合同成立后， 您可以解除本合同， 请填写解除合同通知书并向\n我们提供下列证明和资料： \n1.保险合同； \n2.您的有效身份证件。 \n自我们收到解除合同通知书时起， 本合同终止。您在犹豫期后解\n除本合同的，我们自收到解除合同通知书之日起 30 日内向您退还本\n合同的现金价值。 \n  \n \n \n您在犹豫期后解除合同可能会遭受一定损失。 \n",
    ),
)


@pytest.fixture
def repair_inputs():
    profile = ExtractionProfile.from_mapping(
        yaml.safe_load((PROMPT_PROFILE_DIR / "entity_type/products.yml").read_text())
    )
    old_profile = replace(
        profile,
        semantic_roles={
            key: value
            for key, value in profile.semantic_roles.items()
            if key in {"plan", "coverage", "inclusion", "scope_option"}
        },
    )
    batch = SimpleNamespace(requests={}, approved={}, selections={})
    inputs = {
        key: {}
        for key in (
            "text_chunks",
            "cache_rows",
            "graph_nodes",
            "graph_edges",
            "entity_chunks",
            "relation_chunks",
        )
    }
    inputs["relation_vectors"] = []
    corrections = []
    for index, (source, target, filename, text) in enumerate(CASES):
        prompt = f"reviewed {filename} input"
        key = hashlib.sha256(prompt.encode()).hexdigest()
        chunk_id, cache_id = f"chunk-{index}", f"extract-{index}"
        selection = {
            "entities": [
                {"name": source, "type": "合同流程", "evidence_ids": ["unit"]},
                {"name": target, "type": "理赔材料", "evidence_ids": ["unit"]},
            ],
            "relationships": [
                {
                    "source": source,
                    "target": target,
                    "keywords": "需要理赔材料",
                    "evidence_ids": ["unit"],
                }
            ],
        }
        chunk = EvidenceChunk(
            filename,
            PRODUCT,
            text,
            (
                EvidenceUnit(
                    "unit",
                    filename,
                    len(PRODUCT) + 1,
                    len(PRODUCT) + 1 + len(text),
                    PRODUCT + "\n" + text,
                ),
            ),
        )
        native = to_native(
            PointerResult.model_validate(selection), chunk, profile=old_profile
        )
        batch.requests[prompt] = chunk
        batch.approved[key] = {"native": native}
        batch.selections[key] = {"selection": selection, "native": native}
        source, target = (
            native["relationships"][0][field] for field in ("source", "target")
        )
        corrections.append(
            ClassificationCorrection(
                key,
                target,
                "理赔材料",
                "保全材料",
                source,
                target,
                "需要理赔材料",
                "需要保全材料",
            )
        )
        inputs["text_chunks"][chunk_id] = {
            "content": text,
            "file_path": filename,
            "llm_cache_list": [cache_id],
        }
        result = json.dumps(native, ensure_ascii=False)
        inputs["cache_rows"][cache_id] = {
            "id": cache_id,
            "return": result,
            "return_value": result,
            "original_prompt": prompt,
            "cache_type": "extract",
            "chunk_id": chunk_id,
            "queryparam": {"model": "fixture-provider"},
            "create_time": 12,
            "update_time": 13,
        }
        node = {
            "entity_type": "理赔材料",
            "source_id": chunk_id,
            "file_path": filename,
            "description": text,
            "created_at": 12,
        }
        edge = {
            "keywords": "需要理赔材料",
            "source_id": chunk_id,
            "file_path": filename,
            "description": text,
            "weight": 1.0,
            "created_at": 12,
        }
        inputs["graph_nodes"][target] = node
        inputs["graph_edges"][(source, target)] = edge
        inputs["entity_chunks"][target] = {"chunk_ids": [chunk_id], "count": 1}
        inputs["relation_chunks"][make_relation_chunk_key(source, target)] = {
            "chunk_ids": [chunk_id],
            "count": 1,
        }
        inputs["relation_vectors"].append(
            {
                "id": f"relation-{index}",
                "source_id": source,
                "target_id": target,
                "chunk_ids": [chunk_id],
                "file_path": filename,
                "content": f"需要理赔材料\t{source}\n{target}\n{text}",
            }
        )
    return batch, profile, corrections, inputs


def prepare(fixture):
    batch, profile, corrections, inputs = fixture
    return prepare_classification_repair(batch, profile, corrections, **inputs)


def test_actual_surrender_evidence_changes_only_reviewed_labels(repair_inputs):
    batch, _, _, inputs = repair_inputs
    before = copy.deepcopy((batch.approved, batch.selections, inputs))
    plan = prepare(repair_inputs)
    assert (
        len(plan["caches"])
        == len(plan["nodes"])
        == len(plan["edges"])
        == len(plan["vectors"])
        == 2
    )
    assert (batch.approved, batch.selections, inputs) == before
    for change in plan["caches"].values():
        old, new = (
            json.loads(change["before"]["return"]),
            json.loads(change["after"]["return"]),
        )
        assert new["entities"][1]["type"] == "保全材料"
        assert new["relationships"][0]["keywords"] == "需要保全材料"
        expected = copy.deepcopy(old)
        expected["entities"][1]["type"] = "保全材料"
        expected["relationships"][0]["keywords"] = "需要保全材料"
        assert new == expected
        assert {
            k: v
            for k, v in change["after"].items()
            if k not in {"return", "return_value"}
        } == {
            k: v
            for k, v in change["before"].items()
            if k not in {"return", "return_value"}
        }
        assert change["after"]["return_value"] == change["after"]["return"]
    for change in plan["vectors"].values():
        assert change["after"]["source_id"] == "<SEP>".join(
            change["before"]["chunk_ids"]
        )
        assert change["after"]["src_id"] == change["before"]["source_id"]
        assert change["after"]["tgt_id"] == change["before"]["target_id"]
        assert change["after"]["file_path"] == change["before"]["file_path"]
        assert change["after"]["content"] == change["before"]["content"].replace(
            "需要理赔材料\t", "需要保全材料\t", 1
        )


@pytest.mark.parametrize(
    "damage", ["quote", "cache", "duplicate", "tracking", "vector"]
)
def test_drift_and_incomplete_tracking_are_rejected(repair_inputs, damage):
    batch, _, corrections, inputs = repair_inputs
    if damage == "quote":
        batch.approved[corrections[0].approval_key]["native"] = copy.deepcopy(
            batch.approved[corrections[0].approval_key]["native"]
        )
        batch.approved[corrections[0].approval_key]["native"]["entities"][0][
            "description"
        ] += "changed"
        batch.selections[corrections[0].approval_key]["native"] = copy.deepcopy(
            batch.approved[corrections[0].approval_key]["native"]
        )
    elif damage == "cache":
        inputs["cache_rows"]["extract-0"]["return"] = "{}"
    elif damage == "duplicate":
        corrections.append(corrections[0])
    elif damage == "tracking":
        inputs["entity_chunks"][corrections[0].entity_name]["chunk_ids"].append(
            "hidden-source"
        )
    else:
        inputs["relation_vectors"][0]["file_path"] = None
    with pytest.raises(ValueError):
        prepare(repair_inputs)


class MemoryKV:
    def __init__(self, rows, events, label):
        self.rows, self.events, self.label = copy.deepcopy(rows), events, label

    async def get_by_id(self, key):
        return copy.deepcopy(self.rows.get(key))

    async def get_by_ids(self, keys):
        return [await self.get_by_id(key) for key in keys]

    async def upsert(self, rows):
        self.events.append(self.label + ":upsert")
        self.rows.update(copy.deepcopy(rows))

    async def index_done_callback(self):
        self.events.append(self.label + ":flush")


def fake_rag(inputs):
    events = []
    graph = SimpleNamespace(
        get_node=AsyncMock(
            side_effect=lambda name: copy.deepcopy(inputs["graph_nodes"][name])
        ),
        get_edge=AsyncMock(
            side_effect=lambda source, target: copy.deepcopy(
                inputs["graph_edges"][(source, target)]
            )
        ),
        upsert_node=AsyncMock(side_effect=lambda *_args: events.append("graph:node")),
        upsert_edge=AsyncMock(side_effect=lambda *_args: events.append("graph:edge")),
        index_done_callback=AsyncMock(side_effect=lambda: events.append("graph:flush")),
    )
    vectors = {row["id"]: row for row in inputs["relation_vectors"]}
    vdb = SimpleNamespace(
        table_name="native_relationships",
        workspace="master",
        db=SimpleNamespace(
            query=AsyncMock(
                side_effect=lambda _sql, args, **_kw: [copy.deepcopy(vectors[args[1]])]
            )
        ),
        upsert=AsyncMock(side_effect=lambda *_args: events.append("vectors:upsert")),
        index_done_callback=AsyncMock(
            side_effect=lambda: events.append("vectors:flush")
        ),
    )
    rag = SimpleNamespace(
        llm_response_cache=MemoryKV(inputs["cache_rows"], events, "cache"),
        text_chunks=MemoryKV(inputs["text_chunks"], events, "chunks"),
        entity_chunks=MemoryKV(inputs["entity_chunks"], events, "entity_tracking"),
        relation_chunks=MemoryKV(
            inputs["relation_chunks"], events, "relation_tracking"
        ),
        chunk_entity_relation_graph=graph,
        relationships_vdb=vdb,
        entities_vdb=SimpleNamespace(upsert=AsyncMock()),
    )
    return rag, events


@pytest.mark.asyncio
async def test_cache_flush_precedes_graph_and_future_native_rebuild_is_correct(
    repair_inputs,
):
    plan = prepare(repair_inputs)
    rag, events = fake_rag(repair_inputs[-1])
    await apply_classification_repair(rag, plan)
    assert events == [
        "cache:upsert",
        "cache:flush",
        "graph:node",
        "graph:node",
        "graph:edge",
        "graph:edge",
        "graph:flush",
        "vectors:upsert",
        "vectors:flush",
    ]
    rag.entities_vdb.upsert.assert_not_called()
    assert rag.relationships_vdb.upsert.call_args.args[0] == {
        key: change["after"] for key, change in plan["vectors"].items()
    }
    cached, chunks = await _get_cached_extraction_results(
        rag.llm_response_cache, {"chunk-0", "chunk-1"}, rag.text_chunks
    )
    assert set(cached) == {"chunk-0", "chunk-1"}
    for chunk_id, results in cached.items():
        for result, timestamp in results:
            nodes, edges = await _rebuild_from_extraction_result(
                rag.text_chunks, result, chunk_id, timestamp, chunks[chunk_id]
            )
            material_nodes = [
                row
                for rows in nodes.values()
                for row in rows
                if row["entity_type"] in {"保全材料", "理赔材料"}
            ]
            assert (
                len(material_nodes) == 1
                and material_nodes[0]["entity_type"] == "保全材料"
            )
            assert all(
                row["keywords"] == "需要保全材料"
                for rows in edges.values()
                for row in rows
            )
            assert all(
                row["file_path"] == chunks[chunk_id]["file_path"]
                for rows in edges.values()
                for row in rows
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["cache", "node", "edge", "vector", "tracking"])
async def test_all_before_states_are_checked_before_any_mutation(repair_inputs, damage):
    plan = prepare(repair_inputs)
    rag, events = fake_rag(repair_inputs[-1])
    if damage == "cache":
        rag.llm_response_cache.rows["extract-0"]["return"] = "{}"
    elif damage == "node":
        rag.chunk_entity_relation_graph.get_node.side_effect = lambda _name: {}
    elif damage == "edge":
        rag.chunk_entity_relation_graph.get_edge.side_effect = lambda *_args: {}
    elif damage == "vector":
        rag.relationships_vdb.db.query.side_effect = lambda *_args, **_kw: []
    else:
        rag.entity_chunks.rows[next(iter(plan["entity_tracking"]))]["chunk_ids"].append(
            "hidden-source"
        )
    with pytest.raises(ValueError):
        await apply_classification_repair(rag, plan)
    assert events == []
