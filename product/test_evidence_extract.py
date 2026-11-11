"""Integrity checks for source copying and the native callback boundary."""

import json

import pytest
from pydantic import ValidationError

from lightrag.prompt import PROMPTS
from lightrag.utils import TruncatedResponse, sanitize_and_normalize_extracted_text
from product.evidence_extract import (
    EvidenceChunk,
    EvidenceExtractor,
    EvidenceUnit,
    PointerResult,
    render_units,
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


def test_chinese_types_preserve_product_and_organization_identity(chunk):
    raw = {
        "entities": [
            {"name": chunk.product, "type": "产品", "evidence_ids": ["e1"]},
            {"name": "原文机构", "type": "机构", "evidence_ids": ["e2"]},
            {"name": "某规则", "type": "合同规则", "evidence_ids": ["e1"]},
        ],
        "relationships": [],
    }
    native = to_native(PointerResult.model_validate(raw), chunk)
    assert [row["name"] for row in native["entities"]] == [
        chunk.product,
        "原文机构",
        chunk.product + "｜某规则",
    ]
    assert [row["type"] for row in native["entities"]] == ["产品", "机构", "合同规则"]


def test_unknown_type_and_normalized_name_collisions_are_refused(chunk):
    raw = selection()
    raw["entities"][0]["type"] = "Unsupported"
    with pytest.raises(ValueError, match="Unsupported evidence entity type"):
        to_native(PointerResult.model_validate(raw), chunk)
    raw = selection()
    raw["entities"][0]["name"] = "A（限定）"
    raw["entities"][1]["name"] = "A(限定)"
    with pytest.raises(ValueError, match="aliases collide"):
        to_native(PointerResult.model_validate(raw), chunk)


def test_native_identifier_truncation_is_refused_before_ingestion(chunk):
    raw = selection()
    raw["entities"][0]["name"] = "长" * 200
    with pytest.raises(ValueError, match="truncated"):
        to_native(PointerResult.model_validate(raw), chunk)


def test_identical_text_at_distinct_positions_keeps_both_sources():
    source = "Repeated. Repeated."
    first = EvidenceUnit("a", "first.txt", 0, 9, source)
    second = EvidenceUnit("b", "first.txt", 10, 19, source)
    other = EvidenceUnit("c", "second.txt", 0, 9, source)
    assert (
        render_units([first, first, second, other])
        == "Repeated.\n\nRepeated.\n\nRepeated."
    )


def make_adapter(chunk, complete, **kwargs):
    return EvidenceExtractor(
        requests={"native input": chunk},
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance="Evidence only",
        complete=complete,
        **kwargs,
    )


def merge_prompt(name, description):
    return PROMPTS["summarize_entity_descriptions"].format(
        description_type="Entity",
        description_name=name,
        description_list=json.dumps({"Description": description}),
        summary_length=10,
        language="English",
    )


@pytest.mark.asyncio
async def test_durable_selection_replays_without_another_model_call(chunk, tmp_path):
    calls = []

    async def complete(*args, **kwargs):
        calls.append(True)
        return json.dumps(selection())

    ledger = tmp_path / "evidence-ledger.json"
    original = make_adapter(
        chunk, complete, ledger_path=ledger, ledger_context="model-v1"
    )
    native = await original("native input", system_prompt="native system")
    replay = make_adapter(
        chunk, complete, ledger_path=ledger, ledger_context="model-v1", replay_only=True
    )
    assert replay.expected == {}
    assert await replay("native input", system_prompt="native system") == native
    assert len(calls) == 1
    assert (
        await replay(merge_prompt("Fixture product｜A", chunk.units[0].text))
        == chunk.units[0].text
    )


@pytest.mark.asyncio
async def test_native_cache_hit_restores_only_explicitly_active_sources(
    chunk, tmp_path
):
    async def complete(*args, **kwargs):
        return json.dumps(selection())

    ledger = tmp_path / "evidence-ledger.json"
    original = make_adapter(chunk, complete, ledger_path=ledger)
    await original("native input", system_prompt="native system")
    replay = make_adapter(chunk, complete, ledger_path=ledger, replay_only=True)
    prompt = merge_prompt("Fixture product｜A", chunk.units[0].text)
    with pytest.raises(ValueError, match="no verified extraction ledger"):
        await replay(prompt)
    with pytest.raises(ValueError, match="unknown evidence source"):
        replay.activate_sources(["future.txt"])
    replay.activate_sources([chunk.source])
    assert await replay(prompt) == chunk.units[0].text


@pytest.mark.asyncio
async def test_ledger_rejects_config_changes_and_modified_native_output(
    chunk, tmp_path
):
    async def complete(*args, **kwargs):
        return json.dumps(selection())

    ledger = tmp_path / "evidence-ledger.json"
    adapter = make_adapter(
        chunk, complete, ledger_path=ledger, ledger_context="model-v1"
    )
    await adapter("native input", system_prompt="native system")
    with pytest.raises(ValueError, match="contract changed"):
        make_adapter(chunk, complete, ledger_path=ledger, ledger_context="model-v2")
    payload = json.loads(ledger.read_text())
    row = next(iter(payload["selections"].values()))
    row["native"]["entities"][0]["description"] = "Unverified replacement"
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="native output failed validation"):
        make_adapter(chunk, complete, ledger_path=ledger, ledger_context="model-v1")


@pytest.mark.asyncio
async def test_missing_approved_selection_never_falls_back_to_provider(chunk):
    async def complete(*args, **kwargs):
        raise AssertionError("Replay must not make a stochastic extraction call")

    adapter = make_adapter(chunk, complete, replay_only=True)
    with pytest.raises(ValueError, match="Approved evidence selection is missing"):
        await adapter("native input", system_prompt="native system")


@pytest.mark.asyncio
async def test_future_document_selection_is_not_restored_into_current_merge(
    chunk, tmp_path
):
    source = "Fixture product\nFuture document condition."
    future = EvidenceChunk(
        "future.txt",
        chunk.product,
        source,
        (EvidenceUnit("future", "future.txt", 0, len(source), source),),
    )
    replies = [
        {
            "entities": [{"name": "A", "type": "Rule", "evidence_ids": ["e1"]}],
            "relationships": [],
        },
        {
            "entities": [{"name": "A", "type": "Rule", "evidence_ids": ["future"]}],
            "relationships": [],
        },
    ]

    async def complete(*args, **kwargs):
        return json.dumps(replies.pop(0))

    settings = {
        "requests": {"current": chunk, "future": future},
        "native_system_prompt": "native system",
        "native_summary_template": PROMPTS["summarize_entity_descriptions"],
        "type_guidance": "Fixture types",
        "selection_guidance": "Evidence only",
        "complete": complete,
        "ledger_path": tmp_path / "ledger.json",
    }
    preparing = EvidenceExtractor(**settings)
    await preparing("current", system_prompt="native system")
    await preparing("future", system_prompt="native system")
    importing = EvidenceExtractor(**settings, replay_only=True)
    importing.activate_sources([chunk.source])
    prompt = merge_prompt("Fixture product｜A", chunk.units[0].text)
    assert await importing(prompt) == chunk.units[0].text
    importing.activate_sources([future.source])
    assert await importing(prompt) == chunk.units[0].text + "\n\n" + source


@pytest.mark.asyncio
async def test_merge_ledger_keeps_equal_quotations_at_distinct_source_positions(
    tmp_path,
):
    source = "Fixture product\nSame complete rule.\nSame complete rule."
    start = source.index("Same")
    second = source.index("Same", start + 1)
    chunk = EvidenceChunk(
        "repeat.txt",
        "Fixture product",
        source,
        (
            EvidenceUnit("e1", "repeat.txt", start, second - 1, source),
            EvidenceUnit("e2", "repeat.txt", second, len(source), source),
        ),
    )

    async def complete(*args, **kwargs):
        return json.dumps(
            {
                "entities": [
                    {"name": "A", "type": "Rule", "evidence_ids": ["e1", "e2"]}
                ],
                "relationships": [],
            }
        )

    adapter = make_adapter(chunk, complete, ledger_path=tmp_path / "ledger.json")
    await adapter("native input", system_prompt="native system")
    assert len(next(iter(adapter.expected.values()))) == 2
    assert await adapter(merge_prompt("Fixture product｜A", chunk.units[0].text)) == (
        "Same complete rule.\n\nSame complete rule."
    )


@pytest.mark.asyncio
async def test_validation_retry_receives_error_with_the_same_source_evidence(chunk):
    requests = []

    async def complete(prompt, **kwargs):
        requests.append(json.loads(prompt))
        raw = selection()
        if len(requests) == 1:
            raw["entities"][0]["evidence_ids"] = ["not-in-source"]
        return json.dumps(raw)

    adapter = make_adapter(chunk, complete)
    with pytest.raises(ValueError, match="out-of-scope evidence"):
        await adapter("native input", system_prompt="native system")
    await adapter("native input", system_prompt="native system")
    assert requests[0]["evidence"] == requests[1]["evidence"]
    assert "out-of-scope evidence" in requests[1]["previous_validation_error"]


@pytest.mark.asyncio
async def test_merge_accepts_exact_native_punctuation_normalization_only():
    source = "Fixture product\n评定标准为JR/T0083—2013（适用版本），但不含例外。"
    start = source.index("评定")
    chunk = EvidenceChunk(
        "standard.txt",
        "Fixture product",
        source,
        (EvidenceUnit("e1", "standard.txt", start, len(source), source),),
    )

    async def complete(*args, **kwargs):
        return json.dumps(
            {
                "entities": [{"name": "A", "type": "Rule", "evidence_ids": ["e1"]}],
                "relationships": [],
            }
        )

    adapter = make_adapter(chunk, complete)
    await adapter("native input", system_prompt="native system")
    original = chunk.units[0].text
    native = sanitize_and_normalize_extracted_text(original)
    assert "JR/T0083-2013" in native
    assert await adapter(merge_prompt("Fixture product｜A", native)) == original
    for altered in [native.replace("2013", "2025"), native.replace("但不含例外。", "")]:
        with pytest.raises(ValueError, match="outside complete original"):
            await adapter(merge_prompt("Fixture product｜A", altered))
