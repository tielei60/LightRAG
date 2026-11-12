"""Provider IDs remain compact while durable source identities survive replay."""

import json

import pytest

from applications.insurance.evidence_extract import (
    EvidenceChunk,
    EvidenceExtractor,
    EvidenceUnit,
    provider_example,
)
from applications.insurance.extraction_profile import ExtractionProfile
from lightrag.prompt import PROMPTS


def options(tmp_path, complete):
    text = "示例保险\n一项完整合同规则，不删除限制。"
    split = text.index("\n")
    chunk = EvidenceChunk(
        "fixture.txt",
        "示例保险",
        text,
        (
            EvidenceUnit(
                "e-aabbccddeeff-0-4",
                "fixture.txt",
                0,
                split,
                text,
                (),
                "product-identity",
            ),
            EvidenceUnit(
                "e-aabbccddeeff-5-22", "fixture.txt", split + 1, len(text), text
            ),
        ),
    )
    profile = ExtractionProfile.from_mapping(
        {
            "entity_types": ["保险产品", "合同规则"],
            "product_entity_type": "保险产品",
            "relationship_types": ["适用合同规则"],
        }
    )
    return {
        "requests": {"native input": chunk},
        "native_system_prompt": "native",
        "native_summary_template": PROMPTS["summarize_entity_descriptions"],
        "type_guidance": "Types",
        "selection_guidance": "Evidence",
        "complete": complete,
        "profile": profile,
        "ledger_path": tmp_path / "ledger.json",
    }


def reply(identifier="e2"):
    return json.dumps(
        {
            "entities": [
                {"name": "合同规则", "type": "合同规则", "evidence_ids": [identifier]}
            ],
            "relationships": [],
        }
    )


@pytest.mark.asyncio
async def test_long_hash_ids_are_absent_from_provider_input_but_preserved_in_ledger(
    tmp_path,
):
    calls = []

    async def complete(prompt, **kwargs):
        calls.append(json.loads(prompt))
        return reply()

    adapter = EvidenceExtractor(**options(tmp_path, complete))
    native = await adapter("native input", system_prompt="native")
    assert [row["id"] for row in calls[0]["evidence"]] == ["e1", "e2"]
    assert "aabbccddeeff" not in json.dumps(calls[0])
    persisted = json.loads((tmp_path / "ledger.json").read_text())
    selected = next(iter(persisted["selections"].values()))["selection"]
    assert selected["entities"][0]["evidence_ids"] == ["e-aabbccddeeff-5-22"]
    reloaded = EvidenceExtractor(**options(tmp_path, complete), replay_only=True)
    assert await reloaded("native input", system_prompt="native") == native
    assert len(calls) == 1


@pytest.mark.parametrize(
    "identifier", ["e99", "e-aabbccddeeff-aabbccddeeff-5-22", "e-aabbccddeeff-5-22"]
)
@pytest.mark.asyncio
async def test_unknown_or_durable_ids_from_provider_are_not_fuzzily_repaired(
    tmp_path, identifier
):
    async def complete(*args, **kwargs):
        return reply(identifier)

    adapter = EvidenceExtractor(**options(tmp_path, complete))
    with pytest.raises(ValueError, match="Unknown request-local evidence reference"):
        await adapter("native input", system_prompt="native")
    assert not adapter.selections
    assert not (tmp_path / "ledger.json").exists()


@pytest.mark.asyncio
async def test_duplicate_short_ids_are_rejected_before_approval(tmp_path):
    async def complete(*args, **kwargs):
        value = json.loads(reply())
        value["entities"][0]["evidence_ids"] = ["e2", "e2"]
        return json.dumps(value)

    adapter = EvidenceExtractor(**options(tmp_path, complete))
    with pytest.raises(ValueError, match="Duplicate evidence reference"):
        await adapter("native input", system_prompt="native")
    assert not adapter.selections


@pytest.mark.asyncio
async def test_unknown_short_id_feedback_uses_same_request_local_namespace(tmp_path):
    calls = []

    async def complete(prompt, **kwargs):
        calls.append(json.loads(prompt))
        return reply("e9" if len(calls) == 1 else "e2")

    adapter = EvidenceExtractor(**options(tmp_path, complete))
    with pytest.raises(ValueError, match="Unknown request-local"):
        await adapter("native input", system_prompt="native")
    await adapter("native input", system_prompt="native")
    assert calls[0]["evidence"] == calls[1]["evidence"]
    assert "e9" in calls[1]["previous_validation_error"]
    assert "e1..eN" in calls[1]["previous_validation_error"]


def test_each_instructional_example_has_its_own_short_id_bijection():
    example = {
        "input": {
            "product": "示例",
            "source_file": "示例.txt",
            "evidence": [
                {"id": "identity-long", "text": "示例"},
                {"id": "rule-long", "text": "规则"},
            ],
        },
        "output": {
            "entities": [
                {"name": "示例", "type": "保险产品", "evidence_ids": ["identity-long"]},
                {"name": "规则", "type": "合同规则", "evidence_ids": ["rule-long"]},
            ],
            "relationships": [
                {
                    "source": "示例",
                    "target": "规则",
                    "keywords": "适用合同规则",
                    "evidence_ids": ["identity-long", "rule-long"],
                }
            ],
        },
    }
    mapped = provider_example(example)
    assert [item["id"] for item in mapped["input"]["evidence"]] == ["e1", "e2"]
    assert mapped["output"]["relationships"][0]["evidence_ids"] == ["e1", "e2"]
    assert example["input"]["evidence"][0]["id"] == "identity-long"
