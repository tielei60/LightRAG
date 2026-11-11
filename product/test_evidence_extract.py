"""Integrity checks for source copying and the native callback boundary."""

import json

import pytest
from pydantic import ValidationError

from lightrag.prompt import PROMPTS
from lightrag.utils import TruncatedResponse
from product.evidence_extract import (
    EvidenceChunk,
    EvidenceExtractor,
    EvidenceUnit,
    PointerResult,
    to_native,
)


@pytest.fixture
def chunk():
    text = "Fixture product\nRule: A applies only with B; C is excluded.\nSecond complete rule."
    split = text.index("Second")
    return EvidenceChunk(
        source="fixture.txt",
        product="Fixture product",
        content=text,
        units=(
            EvidenceUnit("e1", "fixture.txt", 0, split, text),
            EvidenceUnit("e2", "fixture.txt", split, len(text), text),
        ),
    )


def selection():
    return {
        "entities": [
            {"name": "A", "type": "Rule", "evidence_ids": ["e1"]},
            {"name": "B", "type": "Concept", "evidence_ids": ["e2"]},
        ],
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "keywords": "related",
                "evidence_ids": ["e2", "e1"],
            }
        ],
    }


def test_native_descriptions_copy_whole_units_in_source_order(chunk):
    native = to_native(PointerResult.model_validate(selection()), chunk)
    assert native["entities"][0]["description"] == chunk.units[0].text
    assert "only with B; C is excluded" in native["entities"][0]["description"]
    assert native["relationships"][0]["description"] == "\n\n".join(
        unit.text for unit in chunk.units
    )
    assert set(native["relationships"][0]) == {
        "source",
        "target",
        "keywords",
        "description",
    }


def test_product_scope_is_applied_to_entities_and_relation_endpoints(chunk):
    native = to_native(PointerResult.model_validate(selection()), chunk)
    assert [entity["name"] for entity in native["entities"]] == [
        "Fixture product｜A",
        "Fixture product｜B",
    ]
    assert native["relationships"][0]["source"] == "Fixture product｜A"
    assert native["relationships"][0]["target"] == "Fixture product｜B"


def test_foreign_product_identity_is_rejected(chunk):
    raw = selection()
    raw["entities"][0].update(name="Another product", type="Product")
    with pytest.raises(ValueError, match="Product identity"):
        to_native(PointerResult.model_validate(raw), chunk)


def test_type_incompatible_with_reviewed_source_unit_is_rejected(chunk):
    typed = EvidenceChunk(
        chunk.source,
        chunk.product,
        chunk.content,
        tuple(
            EvidenceUnit(
                unit.identifier,
                unit.source,
                unit.start,
                unit.end,
                unit.source_text,
                ("Concept",),
            )
            for unit in chunk.units
        ),
    )
    with pytest.raises(ValueError, match="Entity type"):
        to_native(PointerResult.model_validate(selection()), typed)


@pytest.mark.parametrize("ids", [["missing"], ["e1", "e1"]])
def test_unknown_and_duplicate_evidence_is_rejected(chunk, ids):
    raw = selection()
    raw["entities"][0]["evidence_ids"] = ids
    with pytest.raises(ValueError):
        to_native(PointerResult.model_validate(raw), chunk)


def test_generated_description_cannot_enter_the_pointer_contract():
    raw = selection()
    raw["entities"][0]["description"] = "A applies without any conditions."
    with pytest.raises(ValidationError):
        PointerResult.model_validate(raw)


def test_unknown_relation_endpoint_is_rejected(chunk):
    raw = selection()
    raw["relationships"][0]["target"] = "Unselected entity"
    with pytest.raises(ValueError):
        to_native(PointerResult.model_validate(raw), chunk)


def test_source_span_outside_original_is_rejected():
    with pytest.raises(ValueError):
        EvidenceUnit("e1", "fixture.txt", 0, 100, "short original")


def test_evidence_from_another_chunk_cannot_be_selected(chunk):
    with pytest.raises(ValueError):
        EvidenceChunk(chunk.source, chunk.product, "unrelated chunk", chunk.units)


def test_product_namespace_requires_identity_in_original_source(chunk):
    with pytest.raises(ValueError):
        EvidenceChunk(chunk.source, "Unproven product", chunk.content, chunk.units)


@pytest.mark.asyncio
async def test_valid_json_flagged_as_truncated_is_rejected(chunk):
    async def complete(*args, **kwargs):
        return TruncatedResponse(json.dumps(selection()))

    adapter = EvidenceExtractor(
        requests={"native input": chunk},
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance="Evidence only",
        complete=complete,
    )
    with pytest.raises(ValueError, match="Truncated evidence"):
        await adapter("native input", system_prompt="native system")


@pytest.mark.asyncio
async def test_callback_returns_native_json_and_merge_does_not_call_model(chunk):
    calls = []
    events = []

    async def complete(prompt, *, system_prompt):
        calls.append((prompt, system_prompt))
        return json.dumps(selection())

    adapter = EvidenceExtractor(
        requests={"native input": chunk},
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance="Select original evidence only",
        complete=complete,
        record=events.append,
    )
    raw = await adapter("native input", system_prompt="native system")
    native = json.loads(raw)
    descriptions = [
        {"Description": native["entities"][0]["description"]},
        {"Description": native["relationships"][0]["description"]},
    ]
    prompt = PROMPTS["summarize_entity_descriptions"].format(
        description_type="Relation",
        description_name=f"({native['relationships'][0]['source']}, {native['relationships'][0]['target']})",
        description_list="\n".join(json.dumps(item) for item in descriptions),
        summary_length=10,
        language="English",
    )
    result = await adapter(prompt)
    assert result == "\n\n".join(unit.text for unit in chunk.units)
    assert len(calls) == 1
    assert events[-1]["provider_called"] is False


@pytest.mark.asyncio
async def test_merge_restores_evidence_removed_before_native_callback(chunk):
    replies = [
        {
            "entities": [{"name": "A", "type": "Rule", "evidence_ids": ["e1"]}],
            "relationships": [],
        },
        {
            "entities": [{"name": "A", "type": "Rule", "evidence_ids": ["e2"]}],
            "relationships": [],
        },
    ]
    events = []

    async def complete(*args, **kwargs):
        return json.dumps(replies.pop(0))

    second = EvidenceChunk(
        chunk.source, chunk.product, chunk.units[1].text, (chunk.units[1],)
    )
    adapter = EvidenceExtractor(
        requests={"first": chunk, "second": second},
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance="Evidence only",
        complete=complete,
        record=events.append,
    )
    await adapter("first", system_prompt="native system")
    await adapter("second", system_prompt="native system")
    prompt = PROMPTS["summarize_entity_descriptions"].format(
        description_type="Entity",
        description_name="Fixture product｜A",
        description_list=json.dumps({"Description": chunk.units[0].text}),
        summary_length=10,
        language="English",
    )
    assert await adapter(prompt) == "\n\n".join(unit.text for unit in chunk.units)
    assert events[-1]["restored_units_omitted_before_callback"] == 1
    assert not replies


@pytest.mark.asyncio
async def test_merge_rejects_abbreviation_and_unknown_callback(chunk):
    async def complete(*args, **kwargs):
        raise AssertionError("Invalid input must not reach a provider")

    adapter = EvidenceExtractor(
        requests={"native input": chunk},
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance="Evidence only",
        complete=complete,
    )
    prompt = PROMPTS["summarize_entity_descriptions"].format(
        description_type="Entity",
        description_name="A",
        description_list=json.dumps({"Description": "Rule: A applies..."}),
        summary_length=10,
        language="English",
    )
    with pytest.raises(ValueError, match="outside complete original"):
        await adapter(prompt)
    with pytest.raises(ValueError, match="Unsupported callback"):
        await adapter("unregistered native input", system_prompt="native system")
