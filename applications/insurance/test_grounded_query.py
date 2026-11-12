"""Grounded source copying must reject fabricated claims and citation changes."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from applications.insurance.grounded_query import (
    GROUNDING_CONTRACT_VERSION,
    GroundedQuery,
    GroundingError,
    SourceSelection,
    parse_source_context,
    provider_limits,
    render_selection,
    validate_selection,
)


def selection_provider(selection):
    """Return an unchanged audit unless a test specifies a coverage repair."""

    async def complete(prompt, **kwargs):
        if "initial_selection" in json.loads(prompt):
            return json.dumps({"additional_unit_ids": []})
        return selection

    return AsyncMock(side_effect=complete)


def native_prompt(chunks, references=None, graph="incorrect graph-only claim"):
    if references is None:
        references = {"1": "原始条款.pdf"}
    return (
        "---Context---\nKnowledge Graph Data (Entity):\n```json\n"
        + json.dumps({"description": graph}, ensure_ascii=False)
        + "\n```\n\nDocument Chunks (Each entry has a reference_id):\n\n```json\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in chunks)
        + "\n```\n\nReference Document List (Each entry starts with a [reference_id]):\n\n```\n"
        + "\n".join(f"[{key}] {name}" for key, name in references.items())
        + "\n```\n"
    )


def wrapped(text, identity="测试产品"):
    return (
        "【来源标识：整理信息，不是合同新约定】\n产品："
        + identity
        + "\n【主原文开始】\n"
        + text
        + "\n【主原文结束】\n【同文件引用原文：product-identity；Unicode码点0:4】\n"
        + identity
        + "\n【引用原文结束】"
    )


def test_native_ndjson_assigns_distinct_units_to_repeated_file_reference():
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": "1", "content": wrapped("第一项责任。")},
                {"reference_id": "1", "content": wrapped("第二项责任。")},
            ]
        )
    )
    assert [unit.identifier for unit in context.units] == ["U0001", "U0002"]
    assert {unit.reference_id for unit in context.units} == {"1"}
    assert context.units[1].chunk_index == 1


def test_full_table_and_footnotes_are_one_original_unit():
    text = "计划表\n|计划|保额|\n|甲|0元|\n注：本项不含其他责任。"
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": wrapped(text)}])
    )
    assert context.units[0].text == text
    answer = render_selection(
        SourceSelection(status="supported", unit_ids=["U0001"]), context.units
    )
    assert "0元" in answer and "本项不含其他责任" in answer
    assert "来源标识" not in answer
    assert "### References\n\n- [1] 原始条款.pdf" in answer


def test_no_substring_quotes_can_strip_a_negative_condition():
    text = "发生保险事故但尚未支付保险金的，不退还现金价值。"
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": wrapped(text)}])
    )
    with pytest.raises(ValueError):
        SourceSelection.model_validate(
            {"status": "supported", "unit_ids": ["U0001"], "quote": "退还现金价值"}
        )
    selected = validate_selection(
        SourceSelection(status="supported", unit_ids=["U0001"]), context
    )
    assert selected[0].text == text


def test_source_cannot_be_substituted_by_graph_description():
    context = parse_source_context(
        native_prompt(
            [{"reference_id": "1", "content": "本项给付后仅本项责任终止。"}],
            graph="给付后全部合同终止。",
        )
    )
    assert len(context.units) == 1
    assert context.units[0].text == "本项给付后仅本项责任终止。"
    with pytest.raises(GroundingError, match="Unknown source unit"):
        validate_selection(
            SourceSelection(status="supported", unit_ids=["graph-1"]), context
        )


def test_unknown_native_file_reference_fails_closed():
    with pytest.raises(GroundingError, match="unknown document"):
        parse_source_context(
            native_prompt([{"reference_id": "9", "content": "条款。"}])
        )


def test_broken_source_wrapper_does_not_leak_source_metadata():
    with pytest.raises(GroundingError, match="no complete business"):
        parse_source_context(
            native_prompt(
                [{"reference_id": "1", "content": "【来源标识】未提供任何完整原文"}]
            )
        )


def test_plain_native_chunks_are_preserved_without_forcing_product_format():
    context = parse_source_context(
        native_prompt(
            [
                {
                    "reference_id": "1",
                    "content": "Original first paragraph.\n\nComplete exception.",
                }
            ]
        )
    )
    assert context.units[0].text == "Original first paragraph.\n\nComplete exception."


def test_primary_selection_automatically_includes_same_chunk_parent_and_shared_tail():
    primary = "住院津贴每次给付以30日为限。"
    parent = "可选责任须在基本责任基础上选择，且被保险人年龄不超过60周岁。"
    tail = "两项医疗责任累计给付之和达到基本保险金额时，两项责任终止。"
    definition = "费用定义的完整原文。"
    content = wrapped(primary) + (
        f"\n【同文件引用原文：structural-parent；Unicode码点100:180】\n{parent}\n【引用原文结束】"
        f"\n【同文件引用原文：structural-parent；Unicode码点250:320】\n{tail}\n【引用原文结束】"
        f"\n【同文件引用原文：explicit-definition-reference；Unicode码点400:450】\n{definition}\n【引用原文结束】"
    )
    unrelated = wrapped("其他责任的单独原文。")
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": "1", "content": content},
                {"reference_id": "1", "content": unrelated},
            ]
        )
    )
    selected = validate_selection(
        SourceSelection(status="supported", unit_ids=["U0001"]), context
    )
    assert [unit.text for unit in selected] == [primary, parent, tail]
    assert selected[1].original_source_span == (100, 180)
    assert context.units[0].required_unit_ids == ("U0002", "U0003")
    assert definition not in render_selection(
        SourceSelection(status="supported", unit_ids=["U0001"]), selected
    )
    for unit in selected:
        assert content[unit.source_start : unit.source_end] == unit.text


def test_dependency_closure_deduplicates_and_preserves_source_catalog_order():
    parent = "共同前提原文。"
    content = (
        wrapped("主责任规则。")
        + f"\n【同文件引用原文：structural-parent；Unicode码点10:20】\n{parent}\n【引用原文结束】"
    )
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": "1", "content": content},
                {"reference_id": "1", "content": content},
            ]
        )
    )
    assert len(context.units) == 2
    selected = validate_selection(
        SourceSelection(status="supported", unit_ids=["U0002", "U0001"]), context
    )
    assert [unit.identifier for unit in selected] == ["U0001", "U0002"]
    answer = render_selection(
        SourceSelection(status="supported", unit_ids=["U0002", "U0001"]), selected
    )
    assert answer.count(parent) == 1


def test_identical_benefit_text_keeps_each_plans_own_governing_context():
    body = "【生存保险金】\n按基本保险金额给付生存保险金。"
    parents = [
        "保障计划一：\n按以下约定承担责任。",
        "保障计划二：\n按以下约定承担责任。",
    ]
    contents = [
        wrapped(body)
        + f"\n【同文件引用原文：structural-parent；Unicode码点{start}:{start + 30}】\n{parent}\n【引用原文结束】"
        for start, parent in zip((100, 300), parents, strict=True)
    ]
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": value} for value in contents])
    )
    catalog = {unit.identifier: unit for unit in context.units}
    benefits = [unit for unit in context.units if unit.role == "primary"]
    assert len(benefits) == 2
    assert [unit.text for unit in benefits] == [body, body]
    for unit, parent, content in zip(benefits, parents, contents, strict=True):
        assert [catalog[key].text for key in unit.required_unit_ids] == [parent]
        assert content[unit.source_start : unit.source_end] == body


def test_same_named_parent_at_different_source_positions_keeps_scopes_distinct():
    body, parent = "在各自范围内给付保险金。", "本计划的共同条件。"
    contents = [
        wrapped(body)
        + f"\n【同文件引用原文：structural-parent；Unicode码点{start}:{start + 20}】\n{parent}\n【引用原文结束】"
        for start in (100, 300)
    ]
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": value} for value in contents])
    )
    catalog = {unit.identifier: unit for unit in context.units}
    benefits = [unit for unit in context.units if unit.role == "primary"]
    assert len(benefits) == 2
    assert [
        [catalog[key].original_source_span for key in unit.required_unit_ids]
        for unit in benefits
    ] == [[(100, 120)], [(300, 320)]]


def test_identical_text_in_primary_and_parent_keeps_both_source_roles():
    body = "该约定仅适用于已选择的责任。"
    content = wrapped("另一项责任的完整给付规则。") + (
        f"\n【同文件引用原文：structural-parent；Unicode码点100:130】\n{body}\n【引用原文结束】"
    )
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": "1", "content": wrapped(body)},
                {"reference_id": "1", "content": content},
            ]
        )
    )
    copies = [unit for unit in context.units if unit.text == body]
    assert [unit.role for unit in copies] == ["primary", "structural-parent"]
    other = next(unit for unit in context.units if unit.text.startswith("另一项"))
    assert other.required_unit_ids == (copies[1].identifier,)


def test_identical_body_with_same_parent_at_different_native_locations_is_separate():
    body, parent = "分别适用的完整原文。", "同一上级原文条件。"
    tail = f"\n【同文件引用原文：structural-parent；Unicode码点10:30】\n{parent}\n【引用原文结束】"
    contents = [
        wrapped(body).replace(
            "【主原文开始】",
            f"原文件Unicode码点（左闭右开）：{start}:{start + len(body)}\n【主原文开始】",
        )
        + tail
        for start in (100, 300)
    ]
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": value} for value in contents])
    )
    benefits = [unit for unit in context.units if unit.role == "primary"]
    parents = [unit for unit in context.units if unit.role == "structural-parent"]
    assert len(benefits) == 2
    assert [unit.chunk_index for unit in benefits] == [0, 1]
    assert len(parents) == 1
    assert all(unit.required_unit_ids == (parents[0].identifier,) for unit in benefits)


def test_same_text_at_unknown_origin_does_not_merge_across_file_references():
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": reference, "content": "完整的原文约定。"}
                for reference in ("1", "2", "1")
            ],
            references={"1": "甲条款.pdf", "2": "乙条款.pdf"},
        )
    )
    assert [(unit.reference_id, unit.chunk_index) for unit in context.units] == [
        ("1", 0),
        ("2", 1),
    ]


def test_known_origin_copy_deduplicates_across_different_native_chunks():
    parent = "两项责任均受本条完整条件约束。"
    tail = f"\n【同文件引用原文：structural-parent；Unicode码点100:140】\n{parent}\n【引用原文结束】"
    context = parse_source_context(
        native_prompt(
            [
                {"reference_id": "1", "content": wrapped(body) + tail}
                for body in ("甲责任的完整约定。", "乙责任的完整约定。")
            ]
        )
    )
    parents = [unit for unit in context.units if unit.role == "structural-parent"]
    assert len(parents) == 1
    assert parents[0].original_source_span == (100, 140)
    assert all(
        unit.required_unit_ids == (parents[0].identifier,)
        for unit in context.units
        if unit.role == "primary"
    )


def test_duplicate_selected_units_and_missing_supported_evidence_fail():
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": "原文。"}])
    )
    with pytest.raises(GroundingError, match="repeat"):
        validate_selection(
            SourceSelection(status="supported", unit_ids=["U0001", "U0001"]), context
        )
    with pytest.raises(GroundingError, match="requires source"):
        validate_selection(SourceSelection(status="supported", unit_ids=[]), context)


@pytest.mark.asyncio
async def test_callback_selects_and_audits_then_emits_exact_source_and_native_citations():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    events = []
    callback = GroundedQuery(provider, record=events.append)
    source = "给付比例为投保时约定给付比例的60%。"
    result = await callback(
        "问题",
        system_prompt=native_prompt(
            [{"reference_id": "1", "content": wrapped(source)}]
        ),
        history_messages=[],
        stream=True,
        hashing_kv="native-cache",
    )
    answer = "".join([part async for part in result])
    assert source in answer
    assert "100%" not in answer
    assert provider.await_count == 2
    options = provider.call_args.kwargs
    assert options["stream"] is False
    assert options["enable_cot"] is False
    assert options["response_format"] == {"type": "json_object"}
    assert options["hashing_kv"] == "native-cache"
    assert "incorrect graph-only claim" not in provider.call_args.args[0]
    assert callback.cache_identity["response_adapter"] == GROUNDING_CONTRACT_VERSION
    assert events[-1]["state"] == "validated"


@pytest.mark.asyncio
async def test_invalid_ids_get_one_reselection_without_any_ungrounded_fallback():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["not-real"]}',
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":[]}',
        ]
    )
    callback = GroundedQuery(provider)
    result = await callback(
        "问题",
        system_prompt=native_prompt([{"reference_id": "1", "content": "原文。"}]),
        stream=False,
    )
    assert "原文。" in result
    assert provider.await_count == 3
    assert "validation_error" in json.loads(provider.call_args_list[1].args[0])


@pytest.mark.asyncio
async def test_repeated_invalid_selection_raises_after_two_calls():
    provider = selection_provider('{"status":"supported","unit_ids":["not-real"]}')
    with pytest.raises(GroundingError, match="after one validation retry"):
        await GroundedQuery(provider)(
            "问题",
            system_prompt=native_prompt([{"reference_id": "1", "content": "原文。"}]),
        )
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_insufficient_sources_do_not_make_up_a_cash_value_or_refuse_its_rules():
    provider = selection_provider('{"status":"insufficient","unit_ids":["U0001"]}')
    source = "发生本合同约定事故但未支付时不退还现金价值。"
    answer = await GroundedQuery(provider)(
        "实际现金价值是多少",
        system_prompt=native_prompt(
            [{"reference_id": "1", "content": wrapped(source)}]
        ),
    )
    assert answer.startswith("现有资料不能据此确定完整结论。")
    assert source in answer


@pytest.mark.asyncio
async def test_unknown_prompt_protocol_does_not_call_provider():
    provider = AsyncMock()
    with pytest.raises(GroundingError, match="Expected one native"):
        await GroundedQuery(provider)("问题", system_prompt="No native source context")
    provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_repeat_uses_only_validated_memory_cache_and_preserves_stream():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    events = []
    callback = GroundedQuery(provider, record=events.append)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    first = await callback(
        "同一问题", system_prompt=system, history_messages=[], stream=False
    )
    repeated = await callback(
        "同一问题", system_prompt=system, history_messages=[], stream=True
    )
    assert first == "".join([part async for part in repeated])
    assert provider.await_count == 2
    assert events[-1]["state"] == "validated_cache_hit"


@pytest.mark.asyncio
async def test_same_question_new_context_or_reference_never_uses_stale_answer():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    callback = GroundedQuery(provider)
    original = native_prompt([{"reference_id": "1", "content": "旧合同不退款。"}])
    updated = native_prompt(
        [{"reference_id": "2", "content": "新合同按原文退款。"}],
        references={"2": "新版条款.pdf"},
    )
    await callback("是否退款", system_prompt=original)
    answer = await callback("是否退款", system_prompt=updated)
    assert provider.await_count == 4
    assert "旧合同" not in answer
    assert "新版条款.pdf" in answer
    assert "[2]" in answer


@pytest.mark.asyncio
async def test_failed_selections_are_not_cached():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["bad"]}',
            '{"status":"supported","unit_ids":["bad"]}',
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":[]}',
        ]
    )
    callback = GroundedQuery(provider)
    system = native_prompt([{"reference_id": "1", "content": "原文。"}])
    with pytest.raises(GroundingError):
        await callback("问题", system_prompt=system)
    answer = await callback("问题", system_prompt=system)
    assert "原文。" in answer
    assert provider.await_count == 4


@pytest.mark.asyncio
async def test_memory_cache_is_bounded_and_history_changes_key():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    callback = GroundedQuery(provider, cache_size=1)
    system = native_prompt([{"reference_id": "1", "content": "原文。"}])
    await callback("问题", system_prompt=system)
    await callback(
        "问题",
        system_prompt=system,
        history_messages=[{"role": "user", "content": "不同历史"}],
    )
    await callback("问题", system_prompt=system)
    assert provider.await_count == 6


@pytest.mark.asyncio
async def test_truncated_json_is_not_treated_as_approved_evidence():
    from lightrag.utils import TruncatedResponse

    provider = selection_provider(
        TruncatedResponse('{"status":"supported","unit_ids":["U0001"]}')
    )
    with pytest.raises(GroundingError, match="complete JSON"):
        await GroundedQuery(provider)(
            "问题",
            system_prompt=native_prompt([{"reference_id": "1", "content": "原文。"}]),
        )
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_partial_evidence_remains_insufficient_without_adding_unselected_rules():
    provider = selection_provider('{"status":"insufficient","unit_ids":["U0001"]}')
    callback = GroundedQuery(provider)
    system = native_prompt(
        [
            {"reference_id": "1", "content": "甲责任与乙责任共用一项约定。"},
            {"reference_id": "1", "content": "甲责任另有一项期限规则。"},
        ]
    )
    answer = await callback("共用约定和期限是否相同？", system_prompt=system)
    assert answer.startswith("现有资料不能据此确定完整结论。")
    assert "另有一项期限规则" not in answer
    request = provider.call_args
    # No second retrieval or generated facts are introduced by the coverage
    # instruction: the provider sees only the same request's source catalog.
    assert len(json.loads(request.args[0])["source_units"]) == 2
    assert request.kwargs["stream"] is False
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_one_complete_source_can_support_fact_without_forcing_duplicate_source():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    system = native_prompt(
        [
            {
                "reference_id": "1",
                "content": "完整约定：条件甲成立时承担责任，条件乙时除外。",
            },
            {
                "reference_id": "2",
                "content": "说明：条件甲成立时承担责任，条件乙时除外。",
            },
        ],
        references={"1": "原始条款.pdf", "2": "说明材料.pdf"},
    )
    answer = await GroundedQuery(provider)("责任条件是什么？", system_prompt=system)
    assert "原始条款.pdf" in answer
    assert "说明材料.pdf" not in answer
    assert answer.count("条件乙时除外") == 1


@pytest.mark.asyncio
async def test_supported_evidence_can_refute_a_false_premise_without_generated_claims():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001","U0002"]}')
    context = native_prompt(
        [
            {"reference_id": "1", "content": "甲责任限于事件发生后七日。"},
            {"reference_id": "1", "content": "乙责任限于事件发生后三十日。"},
        ]
    )
    answer = await GroundedQuery(provider)(
        "这两项责任期限完全一样吗？", system_prompt=context
    )
    assert not answer.startswith("现有资料不能据此确定完整结论")
    assert "甲责任限于事件发生后七日" in answer
    assert "乙责任限于事件发生后三十日" in answer
    assert "完全一样" not in answer


@pytest.mark.asyncio
async def test_scope_specific_prerequisites_are_not_removed_as_duplicate_evidence():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001","U0002"]}')
    context = native_prompt(
        [
            {
                "reference_id": "1",
                "content": "甲产品可选津贴责任须先投保基本责任且被保险人未满六十周岁。",
            },
            {"reference_id": "2", "content": "乙产品可选津贴责任须先投保医疗责任。"},
        ],
        references={"1": "甲产品原文.pdf", "2": "乙产品原文.md"},
    )
    answer = await GroundedQuery(provider)(
        "比较两个产品津贴责任的选择前提。", system_prompt=context
    )
    assert "未满六十周岁" in answer
    assert "须先投保医疗责任" in answer
    assert "甲产品原文.pdf" in answer and "乙产品原文.md" in answer


@pytest.mark.asyncio
async def test_formula_without_asked_numeric_result_is_insufficient_but_rule_is_preserved():
    provider = selection_provider('{"status":"insufficient","unit_ids":["U0001"]}')
    source = "退还金额＝原费用×（1－已过日数／总日数）。"
    answer = await GroundedQuery(provider)(
        "到指定日期实际退还多少元？",
        system_prompt=native_prompt([{"reference_id": "1", "content": source}]),
    )
    assert answer.startswith("现有资料不能据此确定完整结论。")
    assert source in answer
    assert "100元" not in answer
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_directly_stated_table_value_remains_answerable_without_calculation():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    source = "| 计划 | 专项保额 |\n| --- | --- |\n| 甲 | 0元 |\n| 乙 | 25万元 |\n注：本表仅限该项责任。"
    answer = await GroundedQuery(provider)(
        "计划乙这项专项保额是多少？",
        system_prompt=native_prompt([{"reference_id": "1", "content": source}]),
    )
    assert not answer.startswith("现有资料不能据此确定完整结论。")
    assert "25万元" in answer
    assert "本表仅限该项责任" in answer


@pytest.mark.asyncio
async def test_audit_adds_missing_operative_body_and_required_parent_without_rewriting():
    parent = "乙产品选择津贴须先投保基本责任，且年龄未满六十周岁。"
    body = "乙产品津贴属于可选责任，仅就约定事件住院给付。"
    context = native_prompt(
        [
            {"reference_id": "1", "content": "甲乙两个产品均列有津贴责任。"},
            {
                "reference_id": "2",
                "content": wrapped(body)
                + f"\n【同文件引用原文：structural-parent；Unicode码点10:40】\n{parent}\n【引用原文结束】",
            },
        ],
        references={"1": "计划表.md", "2": "有效条款.pdf"},
    )
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["U0002"]}',
        ]
    )
    events = []
    answer = await GroundedQuery(provider, record=events.append)(
        "乙产品的责任分类和选择前提？", system_prompt=context
    )
    assert "甲乙两个产品均列有津贴责任" in answer
    assert body in answer and parent in answer
    assert "[2] 有效条款.pdf" in answer
    audit_input = json.loads(provider.call_args.args[0])
    assert audit_input["initial_selection"]["unit_ids"] == ["U0001"]
    assert len(audit_input["source_units"]) == 3
    assert events[-1]["coverage_audit"]["additional_unit_ids"] == ["U0002"]
    assert events[-1]["selection"]["unit_ids"] == ["U0001", "U0002"]
    assert events[-1]["selected_units"][-1]["original_source_span"] == (10, 40)
    assert "selection_seconds" in events[-1] and "coverage_audit_seconds" in events[-1]


@pytest.mark.asyncio
async def test_audit_adds_available_evidence_without_overriding_availability_status():
    provider = AsyncMock(
        side_effect=[
            '{"status":"insufficient","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["U0002"]}',
        ]
    )
    answer = await GroundedQuery(provider)(
        "甲乙的实际条件相同吗？",
        system_prompt=native_prompt(
            [
                {"reference_id": "1", "content": "甲只适用条件一。"},
                {"reference_id": "1", "content": "乙只适用条件二。"},
            ]
        ),
    )
    assert answer.startswith("现有资料不能据此确定完整结论。")
    assert "条件一" in answer and "条件二" in answer
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_audit_preserves_formula_only_numeric_availability_without_calculating():
    provider = AsyncMock(
        side_effect=[
            '{"status":"insufficient","unit_ids":["U0001"]}',
            '{"additional_unit_ids":[]}',
        ]
    )
    source = "退还金额＝费用×剩余期间比例。"
    answer = await GroundedQuery(provider)(
        "实际能退多少元？",
        system_prompt=native_prompt([{"reference_id": "1", "content": source}]),
    )
    assert answer.startswith("现有资料不能据此确定完整结论。")
    assert source in answer and "100元" not in answer


@pytest.mark.asyncio
async def test_invalid_audit_marks_partial_output_incomplete_and_never_caches_it():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["unavailable"]}',
            '{"additional_unit_ids":["unavailable"]}',
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":[]}',
        ]
    )
    events = []
    callback = GroundedQuery(provider, record=events.append)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    stream = await callback("问题", system_prompt=system, stream=True)
    assert await anext(stream) == "> 完整原文。 [1]"
    with pytest.raises(GroundingError, match="after one validation retry"):
        await anext(stream)
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["emitted_characters"] > 0
    assert not any(event.get("state") == "validated" for event in events)
    answer = await callback("问题", system_prompt=system)
    assert "完整原文" in answer
    assert provider.await_count == 5


@pytest.mark.asyncio
async def test_protocol_repair_budget_is_shared_across_selection_and_audit():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["bad"]}',
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["bad"]}',
        ]
    )
    with pytest.raises(GroundingError):
        await GroundedQuery(provider)(
            "问题",
            system_prompt=native_prompt([{"reference_id": "1", "content": "原文。"}]),
        )
    assert provider.await_count == 3


@pytest.mark.asyncio
async def test_audit_cannot_exceed_active_unit_limit_or_drop_initial_evidence():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["U0002"]}',
            '{"additional_unit_ids":["U0002"]}',
        ]
    )
    with pytest.raises(GroundingError, match="Too many source units"):
        await GroundedQuery(provider, max_units=1)(
            "问题",
            system_prompt=native_prompt(
                [
                    {"reference_id": "1", "content": "初选原文。"},
                    {"reference_id": "1", "content": "追加原文。"},
                ]
            ),
        )
    assert provider.await_count == 3


@pytest.mark.asyncio
async def test_exact_context_cache_holds_only_completed_audit_with_added_sources():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"additional_unit_ids":["U0002"]}',
        ]
    )
    events = []
    callback = GroundedQuery(provider, record=events.append)
    system = native_prompt(
        [
            {"reference_id": "1", "content": "初选。"},
            {"reference_id": "1", "content": "审计补足。"},
        ]
    )
    first = await callback("问题", system_prompt=system)
    repeated = await callback("问题", system_prompt=system)
    assert first == repeated and "审计补足" in repeated
    assert provider.await_count == 2
    assert events[-1]["state"] == "validated_cache_hit"
    assert events[-1]["coverage_audited"] is True


@pytest.mark.asyncio
async def test_addition_only_audit_rejects_hidden_status_re_adjudication():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001"]}',
            '{"status":"insufficient","additional_unit_ids":[]}',
            '{"status":"insufficient","additional_unit_ids":[]}',
        ]
    )
    with pytest.raises(GroundingError, match="Extra inputs are not permitted"):
        await GroundedQuery(provider)(
            "问题",
            system_prompt=native_prompt(
                [{"reference_id": "1", "content": "完整原文。"}]
            ),
        )
    assert provider.await_count == 3


def six_source_context():
    return native_prompt(
        [
            {"reference_id": str(index), "content": f"资料{index}的完整约定。"}
            for index in range(1, 7)
        ],
        references={str(index): f"原始资料{index}.pdf" for index in range(1, 7)},
    )


@pytest.mark.asyncio
async def test_default_source_limit_accepts_six_unit_six_file_audited_union():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001","U0002","U0003","U0004"]}',
            '{"additional_unit_ids":["U0005","U0006"]}',
        ]
    )
    callback = GroundedQuery(provider)
    answer = await callback(
        "核对全部资料中的对应约定。", system_prompt=six_source_context()
    )
    assert callback.max_units == 6 and callback.max_sources == 6
    assert all(f"[{index}] 原始资料{index}.pdf" in answer for index in range(1, 7))
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_explicit_lower_source_limit_still_fails_closed_on_six_file_union():
    provider = AsyncMock(
        side_effect=[
            '{"status":"supported","unit_ids":["U0001","U0002","U0003","U0004"]}',
            '{"additional_unit_ids":["U0005","U0006"]}',
            '{"additional_unit_ids":["U0005","U0006"]}',
        ]
    )
    with pytest.raises(GroundingError, match="Too many reference documents"):
        await GroundedQuery(provider, max_sources=5)(
            "核对全部资料。", system_prompt=six_source_context()
        )
    assert provider.await_count == 3


@pytest.mark.asyncio
async def test_policy_version_change_cannot_reuse_a_previous_audited_memory_entry():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    callback = GroundedQuery(provider)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    await callback("问题", system_prompt=system)
    await callback("问题", system_prompt=system)
    assert provider.await_count == 2
    with patch(
        "applications.insurance.grounded_query.GROUNDING_CONTRACT_VERSION",
        "different-audit-contract",
    ):
        await callback("问题", system_prompt=system)
    assert provider.await_count == 4


def test_default_source_limit_tracks_a_smaller_active_unit_limit():
    callback = GroundedQuery(AsyncMock(), max_units=3)
    assert callback.max_sources == 3


def test_navigation_only_parents_are_hidden_without_changing_source_units():
    source = "被保险人未满60周岁时，方可选择本项责任。"
    titles = [
        "# 测试医疗保险核保规则",
        "##二、投保规则",
        "一、产品基本特征",
    ]
    content = wrapped(source, identity="测试医疗保险") + "".join(
        f"\n【同文件引用原文：structural-parent；Unicode码点{index}:{index + 10}】\n{title}\n【引用原文结束】"
        for index, title in enumerate(titles, 100)
    )
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": content}])
    )
    selection = SourceSelection(status="supported", unit_ids=["U0001"])
    selected = validate_selection(selection, context)
    answer = render_selection(selection, selected)
    assert source in answer
    assert all(title not in answer for title in titles)
    assert [unit.text for unit in selected] == [source, *titles]
    assert len(context.units[0].required_unit_ids) == len(titles)
    for unit in selected:
        assert content[unit.source_start : unit.source_end] == unit.text


@pytest.mark.parametrize(
    "parent",
    [
        "2.5.1 基本责任",
        "2.5.2 可选责任",
        "# 可选责任二",
        "## 不限次数",
        "## 年龄不超过60周岁",
        "## 责任免除",
        "## 特定疾病",
        "## 现金价值条款",
        "## 意外保险",
        "# 意外保险",
        "# 少儿保险",
        "# 医疗保险",
        "# 测试医疗保险（互联网）",
        "## 少儿投保规则",
        "## 女性投保规则",
        "# 少儿核保规则",
        "# 女性保险条款",
        "## 投保规则\n本项责任须先选择基本责任，且不承担约定外事故。",
        "未选择基本责任的，不得选择本项责任。",
    ],
)
def test_substantive_parent_headings_and_title_with_body_are_preserved(parent):
    content = wrapped("主责任完整约定。") + (
        f"\n【同文件引用原文：structural-parent；Unicode码点100:200】\n{parent}\n【引用原文结束】"
    )
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": content}])
    )
    selection = SourceSelection(status="supported", unit_ids=["U0001"])
    selected = validate_selection(selection, context)
    answer = render_selection(selection, selected)
    for line in parent.splitlines():
        assert "> " + line in answer
    assert selected[1].text == parent


def test_actual_document_title_and_enumerated_navigation_require_exact_identity():
    product = "北京人寿京安泰综合意外伤害保险"
    parents = [
        f"# {product}投保及核保规则",
        "## 二投保规则",
        "## 少儿投保规则",
        "## 女性投保规则",
        "## 二、可选责任",
        f"## 女性{product}投保及核保规则",
    ]
    content = wrapped("本责任仅限约定事故。", identity=product) + "".join(
        f"\n【同文件引用原文：structural-parent；Unicode码点{index}:{index + 10}】\n{title}\n【引用原文结束】"
        for index, title in enumerate(parents, 100)
    )
    context = parse_source_context(
        native_prompt([{"reference_id": "1", "content": content}])
    )
    selection = SourceSelection(status="supported", unit_ids=["U0001"])
    selected = validate_selection(selection, context)
    answer = render_selection(selection, selected)
    assert all(unit.product_name == product for unit in selected)
    assert parents[0] not in answer and parents[1] not in answer
    assert all(title in answer for title in parents[2:])
    assert [unit.text for unit in selected][1:] == parents


def test_document_title_can_match_exact_file_stem_but_not_arbitrary_prefix():
    parents = ["# 原始文档标题", "# 女性原始文档标题"]
    content = wrapped("完整原文。") + "".join(
        f"\n【同文件引用原文：structural-parent；Unicode码点{index}:{index + 10}】\n{title}\n【引用原文结束】"
        for index, title in enumerate(parents, 100)
    )
    context = parse_source_context(
        native_prompt(
            [{"reference_id": "1", "content": content}],
            references={"1": "目录/原始文档标题.md"},
        )
    )
    selection = SourceSelection(status="supported", unit_ids=["U0001"])
    answer = render_selection(selection, validate_selection(selection, context))
    assert "> # 原始文档标题" not in answer
    assert parents[1] in answer


def test_product_identity_comes_only_from_unique_program_prefix_before_original_body():
    fake_product = "正文中的少儿保险"
    original = f"产品：{fake_product}\n完整规则。"
    parent = f"# {fake_product}核保规则"
    attachment = f"\n【同文件引用原文：structural-parent；Unicode码点100:200】\n{parent}\n【引用原文结束】"
    for content, expected in [
        (wrapped(original, identity="真正产品") + attachment, "真正产品"),
        (
            f"产品：{fake_product}\n【主原文开始】\n完整规则。\n【主原文结束】"
            + attachment,
            None,
        ),
    ]:
        context = parse_source_context(
            native_prompt([{"reference_id": "1", "content": content}])
        )
        assert all(unit.product_name == expected for unit in context.units)
        selection = SourceSelection(status="supported", unit_ids=["U0001"])
        answer = render_selection(selection, validate_selection(selection, context))
        assert parent in answer


@pytest.mark.asyncio
async def test_real_stream_emits_initial_units_before_audit_then_only_adds_new_units():
    audit_started, release_audit = asyncio.Event(), asyncio.Event()
    parent = "可选责任须先投保基本责任。"
    initial_source = "初选完整原文，不包含其他责任。"
    extra_source = "补充完整原文，仅在约定期间给付。"
    system = native_prompt(
        [
            {"reference_id": "1", "content": extra_source},
            {
                "reference_id": "2",
                "content": wrapped(initial_source)
                + f"\n【同文件引用原文：structural-parent；Unicode码点10:40】\n{parent}\n【引用原文结束】",
            },
        ],
        references={"1": "补充条款.pdf", "2": "初选条款.pdf"},
    )

    async def provider(prompt, **kwargs):
        if "initial_selection" not in json.loads(prompt):
            return '{"status":"supported","unit_ids":["U0002"]}'
        audit_started.set()
        await release_audit.wait()
        return '{"additional_unit_ids":["U0001","U0002"]}'

    events = []
    provider = AsyncMock(side_effect=provider)
    callback = GroundedQuery(provider, record=events.append)
    stream = await callback("核对责任和期间", system_prompt=system, stream=True)
    provider.assert_not_awaited()
    first = await anext(stream)
    prerequisite = await anext(stream)
    assert first == f"> {initial_source} [2]"
    assert prerequisite == f"\n\n> {parent} [2]"
    assert not audit_started.is_set()
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(audit_started.wait(), 1)
    assert not pending.done()
    assert not any(event.get("state") == "validated" for event in events)
    release_audit.set()
    supplement = await pending
    tail = [part async for part in stream]
    answer = first + prerequisite + supplement + "".join(tail)
    assert supplement == f"\n\n> {extra_source} [1]"
    assert len(tail) == 1 and tail[0].startswith("\n\n### References")
    assert answer.count(initial_source) == answer.count(parent) == 1
    assert answer.count(extra_source) == answer.count("### References") == 1
    assert events[-1]["state"] == "validated"
    assert events[-1]["response_complete"] is True
    assert events[-1]["selection"]["unit_ids"] == ["U0002", "U0001"]
    assert provider.await_count == 2
    repeated = await callback("核对责任和期间", system_prompt=system)
    assert repeated == answer and provider.await_count == 2
    # Nonstream uses the identical initial-then-supplement sequence, even when
    # the added unit precedes the initial unit in the native source catalog.
    uncached = await GroundedQuery(provider, cache_size=0)(
        "核对责任和期间", system_prompt=system
    )
    assert uncached == answer


@pytest.mark.asyncio
async def test_stream_initial_failure_is_raised_during_iteration_for_native_error_frame():
    provider = AsyncMock()
    events = []
    stream = await GroundedQuery(provider, record=events.append)(
        "问题", system_prompt="Missing native context", stream=True
    )
    with pytest.raises(GroundingError, match="Expected one native"):
        await anext(stream)
    provider.assert_not_awaited()
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["emitted_characters"] == 0


@pytest.mark.asyncio
async def test_closing_after_source_prefix_never_starts_audit_or_caches_partial_text():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    events = []
    callback = GroundedQuery(provider, record=events.append)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    stream = await callback("问题", system_prompt=system, stream=True)
    assert await anext(stream) == "> 完整原文。 [1]"
    await stream.aclose()
    assert provider.await_count == 1
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["error_type"] == "GeneratorExit"
    answer = await callback("问题", system_prompt=system)
    assert "### References" in answer and provider.await_count == 3


@pytest.mark.asyncio
async def test_closing_at_final_reference_yield_cannot_record_completed_or_cache_answer():
    provider = selection_provider('{"status":"supported","unit_ids":["U0001"]}')
    events = []
    callback = GroundedQuery(provider, record=events.append)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    stream = await callback("问题", system_prompt=system, stream=True)
    await anext(stream)
    assert "### References" in await anext(stream)
    await stream.aclose()
    assert events[-1]["state"] == "incomplete"
    assert not any(event.get("state") == "validated" for event in events)
    answer = await callback("问题", system_prompt=system)
    assert "完整原文。" in answer and provider.await_count == 4


@pytest.mark.asyncio
async def test_stream_provider_failure_before_first_source_is_explicit_and_not_cached():
    events = []
    provider = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    stream = await GroundedQuery(provider, record=events.append)(
        "问题",
        system_prompt=native_prompt([{"reference_id": "1", "content": "原文。"}]),
        stream=True,
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await anext(stream)
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["error"] == "provider unavailable"
    assert events[-1]["emitted_characters"] == 0


@pytest.mark.asyncio
async def test_cancelled_audit_propagates_and_releases_provider_slot_without_caching():
    audit_started, provider_cancelled = asyncio.Event(), asyncio.Event()
    audit_attempts = 0

    async def provider(prompt, **kwargs):
        nonlocal audit_attempts
        if "initial_selection" not in json.loads(prompt):
            return '{"status":"supported","unit_ids":["U0001"]}'
        audit_attempts += 1
        if audit_attempts == 1:
            audit_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()
        return '{"additional_unit_ids":[]}'

    provider = AsyncMock(side_effect=provider)
    events = []
    callback = GroundedQuery(provider, provider_max_async=1, record=events.append)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    stream = await callback("问题", system_prompt=system, stream=True)
    await anext(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(audit_started.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert provider_cancelled.is_set()
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["error_type"] == "CancelledError"
    assert not any(event.get("state") == "validated" for event in events)
    retry = await asyncio.wait_for(callback("问题", system_prompt=system), 1)
    assert "### References" in retry and provider.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_stage", ["selection", "audit"])
async def test_provider_timeout_is_explicit_and_does_not_cache_or_hold_slot(
    timeout_stage,
):
    timed_out = False

    async def provider(prompt, **kwargs):
        nonlocal timed_out
        is_audit = "initial_selection" in json.loads(prompt)
        if not timed_out and (is_audit == (timeout_stage == "audit")):
            timed_out = True
            await asyncio.Event().wait()
        return (
            '{"additional_unit_ids":[]}'
            if is_audit
            else '{"status":"supported","unit_ids":["U0001"]}'
        )

    provider = AsyncMock(side_effect=provider)
    events = []
    callback = GroundedQuery(
        provider, record=events.append, provider_max_async=1, provider_timeout=0.05
    )
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    stream = await callback("问题", system_prompt=system, stream=True)
    prefix = []
    with pytest.raises(GroundingError, match="provider timed out"):
        async for part in stream:
            prefix.append(part)
    assert bool(prefix) == (timeout_stage == "audit")
    assert all("### References" not in part for part in prefix)
    assert events[-1]["state"] == "incomplete"
    retry = await asyncio.wait_for(callback("问题", system_prompt=system), 1)
    assert "### References" in retry
    assert provider.await_count == (4 if timeout_stage == "audit" else 3)


@pytest.mark.asyncio
async def test_stream_initial_and_audit_calls_share_one_provider_concurrency_limit():
    keys = [("甲", False), ("乙", False), ("甲", True)]
    entered = {key: asyncio.Event() for key in keys}
    release = {key: asyncio.Event() for key in keys}
    active, peak = 0, 0

    async def provider(prompt, **kwargs):
        nonlocal active, peak
        payload = json.loads(prompt)
        key = (payload["question"], "initial_selection" in payload)
        active += 1
        peak = max(peak, active)
        entered[key].set()
        try:
            await release[key].wait()
            return (
                '{"additional_unit_ids":[]}'
                if key[1]
                else '{"status":"supported","unit_ids":["U0001"]}'
            )
        finally:
            active -= 1

    async def advance(stream, attempted):
        attempted.set()
        return await anext(stream)

    callback = GroundedQuery(provider, provider_max_async=1)
    system = native_prompt([{"reference_id": "1", "content": "完整原文。"}])
    first = await callback("甲", system_prompt=system, stream=True)
    second = await callback("乙", system_prompt=system, stream=True)
    first_task = asyncio.create_task(anext(first))
    await asyncio.wait_for(entered[("甲", False)].wait(), 1)
    second_attempted = asyncio.Event()
    second_task = asyncio.create_task(advance(second, second_attempted))
    await asyncio.wait_for(second_attempted.wait(), 1)
    assert not entered[("乙", False)].is_set()
    release[("甲", False)].set()
    await first_task
    await asyncio.wait_for(entered[("乙", False)].wait(), 1)
    audit_attempted = asyncio.Event()
    audit_task = asyncio.create_task(advance(first, audit_attempted))
    await asyncio.wait_for(audit_attempted.wait(), 1)
    assert not entered[("甲", True)].is_set()
    release[("乙", False)].set()
    await second_task
    await asyncio.wait_for(entered[("甲", True)].wait(), 1)
    release[("甲", True)].set()
    assert "### References" in await audit_task
    assert [part async for part in first] == []
    await second.aclose()
    assert peak == 1 and active == 0


def test_provider_limits_use_explicit_query_then_general_settings_and_pure_defaults():
    assert provider_limits({}) == {
        "provider_max_async": 2,
        "provider_timeout": 240,
    }
    assert provider_limits({"MAX_ASYNC_LLM": "4", "LLM_TIMEOUT": "37"}) == {
        "provider_max_async": 4,
        "provider_timeout": 37,
    }
    assert provider_limits(
        {
            "QUERY_MAX_ASYNC_LLM": "3",
            "QUERY_LLM_TIMEOUT": "19.5",
            "MAX_ASYNC_LLM": "7",
            "LLM_TIMEOUT": "120",
        }
    ) == {"provider_max_async": 3, "provider_timeout": 19.5}


@pytest.mark.parametrize(
    "config",
    [
        {"QUERY_MAX_ASYNC_LLM": "0"},
        {"MAX_ASYNC_LLM": 0},
        {"QUERY_LLM_TIMEOUT": "-1"},
        {"LLM_TIMEOUT": "nan"},
        {"QUERY_LLM_TIMEOUT": "inf"},
    ],
)
def test_invalid_provider_limits_fail_before_starting_requests(config):
    with pytest.raises(ValueError, match="positive"):
        provider_limits(config)
