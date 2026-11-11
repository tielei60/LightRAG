"""Source-grounded plan/table checks reject graph facts without inventing answers."""

import json

import pytest

from lightrag.prompt import PROMPTS
from product.evidence_extract import (
    EvidenceChunk,
    EvidenceExtractor,
    EvidenceUnit,
    PointerResult,
    to_native,
)
from product.evidence_semantics import (
    EvidenceSemanticError,
    cell_kind,
    parse_markdown_tables,
    semantic_metadata,
    table_facts,
)
from product.extraction_profile import ExtractionProfile

ROLES = {
    "plan": ("保障计划",),
    "coverage": ("保险责任",),
    "inclusion": ("包含责任",),
    "scope_option": ("合同规则",),
}


def profile():
    return ExtractionProfile.from_mapping(
        {
            "entity_types": ["保险产品", "保障计划", "保险责任", "合同规则"],
            "product_entity_type": "保险产品",
            "unscoped_entity_types": [],
            "relationship_types": ["包含责任"],
            "semantic_roles": {name: list(values) for name, values in ROLES.items()},
        }
    )


def source_chunk(text, units=None):
    return EvidenceChunk(
        "fixture.md",
        "示例产品",
        text,
        tuple(units)
        if units
        else (EvidenceUnit("all", "fixture.md", 0, len(text), text),),
    )


def selection(pairs, plan_type="保障计划"):
    plans = dict.fromkeys(plan for plan, _ in pairs)
    coverages = dict.fromkeys(coverage for _, coverage in pairs)
    return PointerResult.model_validate(
        {
            "entities": [
                *(
                    {"name": name, "type": plan_type, "evidence_ids": ["all"]}
                    for name in plans
                ),
                *(
                    {"name": name, "type": "保险责任", "evidence_ids": ["all"]}
                    for name in coverages
                ),
            ],
            "relationships": [
                {
                    "source": plan,
                    "target": coverage,
                    "keywords": "包含责任",
                    "evidence_ids": ["all"],
                }
                for plan, coverage in pairs
            ],
        }
    )


def matrix():
    return "示例产品\n\n### 投保计划与保障\n\n| 责任 | 计划甲 | 计划乙 |\n| --- | --- | --- |\n| 医疗责任 | 0元 | 10000元 |\n| 航空责任 | 0元／天 | 200万元 |\n\n注：金额按本表列示，其他责任另按条款。\n"


def test_table_cells_keep_header_row_value_notes_and_absolute_source_spans():
    source = matrix()
    tables = parse_markdown_tables(source)
    assert len(tables) == 1
    table = tables[0]
    assert table.title == "### 投保计划与保障"
    assert source[table.start : table.end].endswith("其他责任另按条款。")
    facts = table_facts(source)
    assert len(facts) == 4
    zero = next(
        item
        for item in facts
        if item["column_header"] == "计划甲" and item["row_header"] == "医疗责任"
    )
    assert zero["raw_value"] == "0元"
    assert zero["cell_kind"] == "zero"
    assert source[zero["cell_start"] : zero["cell_end"]] == "0元"
    bound = semantic_metadata(source_chunk(source).units)["table_facts"]
    assert [
        {key: value for key, value in fact.items() if key != "evidence_ids"}
        for fact in bound
    ] == facts
    assert all(fact["evidence_ids"] == ["all"] for fact in bound)


@pytest.mark.parametrize(
    "value,kind",
    [
        ("0", "zero"),
        ("0.00元", "zero"),
        ("0元／天", "zero"),
        ("5000元", "positive"),
        ("5万元，扩展其他情形", "positive"),
        ("1,000元", "positive"),
        ("", "blank"),
        ("—", "dash"),
        ("未提供", "unknown"),
        ("按合同约定", "text"),
    ],
)
def test_only_the_actual_cell_value_is_classified(value, kind):
    assert cell_kind(value) == kind


def test_all_zero_cell_positive_edges_fail_together_without_invented_exclusions():
    selected = selection(
        [("计划甲", "医疗责任"), ("计划甲", "航空责任"), ("计划乙", "医疗责任")]
    )
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(selected, source_chunk(matrix()), profile=profile())
    assert len(caught.value.issues) == 2
    assert {item["value"] for item in caught.value.issues} == {"0元", "0元／天"}
    assert len(caught.value.repair_feedback) < 500
    assert "不反推整个合同不保" in caught.value.repair_feedback
    valid = selection([("计划乙", "医疗责任"), ("计划乙", "航空责任")])
    assert (
        len(
            to_native(valid, source_chunk(matrix()), profile=profile())["relationships"]
        )
        == 2
    )


def test_swapping_relation_endpoints_cannot_bypass_zero_cell_validation():
    selected = selection([("计划甲", "医疗责任")])
    edge = selected.relationships[0]
    edge.source, edge.target = edge.target, edge.source
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(selected, source_chunk(matrix()), profile=profile())
    assert caught.value.issues[0]["value"] == "0元"


def test_native_full_width_name_normalization_preserves_exact_table_identity():
    source = matrix().replace("计划甲", "计划甲（基础版）")
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(
            selection([("计划甲(基础版)", "医疗责任")]),
            source_chunk(source),
            profile=profile(),
        )
    assert caught.value.issues[0]["value"] == "0元"


@pytest.mark.parametrize("value", ["", "—", "未提供"])
def test_unknown_cells_cannot_support_positive_inclusion(value):
    source = matrix().replace("| 医疗责任 | 0元 |", f"| 医疗责任 | {value} |")
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(
            selection([("计划甲", "医疗责任")]), source_chunk(source), profile=profile()
        )
    assert caught.value.issues[0]["value"] == value


def test_other_zero_values_do_not_invalidate_a_positive_corresponding_cell():
    source = matrix() + "其他说明：首次咨询费用为0元。\n"
    native = to_native(
        selection([("计划乙", "医疗责任")]), source_chunk(source), profile=profile()
    )
    assert len(native["relationships"]) == 1
    assert "10000元" in native["relationships"][0]["description"]


def test_partial_table_without_required_header_or_note_cannot_bypass_validation():
    source = matrix()
    end = source.index("\n\n注：")
    whole = EvidenceUnit(
        "identity", "fixture.md", 0, source.index("\n"), source, (), "product-identity"
    )
    fragment = EvidenceUnit("all", "fixture.md", source.index("| 责任"), end, source)
    chunk = source_chunk(source, [whole, fragment])
    assert semantic_metadata(chunk.units)["table_facts"] == []
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(selection([("计划乙", "医疗责任")]), chunk, profile=profile())
    assert caught.value.issues[0]["kind"] == "incomplete_table"


def test_table_note_evidence_does_not_need_following_separator_newline():
    source = matrix() + "\n其他独立规则。\n"
    end = source.index("其他责任另按条款。") + len("其他责任另按条款。")
    unit = EvidenceUnit("all", "fixture.md", 0, end, source)
    assert len(semantic_metadata([unit])["table_facts"]) == 4
    assert (
        len(
            to_native(
                selection([("计划乙", "医疗责任")]),
                source_chunk(source, [unit]),
                profile=profile(),
            )["relationships"]
        )
        == 1
    )


def test_identity_only_relation_evidence_cannot_bypass_a_known_plan_table():
    source = matrix()
    primary = EvidenceUnit("all", "fixture.md", 0, len(source), source)
    identity = EvidenceUnit(
        "p", "fixture.md", 0, len("示例产品"), source, (), "product-identity"
    )
    selected = selection([("计划甲", "医疗责任")])
    selected.relationships[0].evidence_ids = ["p"]
    with pytest.raises(EvidenceSemanticError) as caught:
        to_native(
            selected, source_chunk(source, [primary, identity]), profile=profile()
        )
    assert caught.value.issues[0]["kind"] == "incomplete_table"


def test_row_plan_table_only_supports_listed_or_explicitly_inherited_responsibilities():
    source = "示例产品\n\n## 保障计划\n\n| 计划 | 保障责任 |\n| --- | --- |\n| 计划甲 | 医疗责任、航空责任 |\n| 计划乙 | 计划甲基本责任＋津贴责任 |\n"
    valid = selection(
        [("计划甲", "医疗责任"), ("计划乙", "航空责任"), ("计划乙", "津贴责任")]
    )
    assert (
        len(to_native(valid, source_chunk(source), profile=profile())["relationships"])
        == 3
    )
    with pytest.raises(EvidenceSemanticError):
        to_native(
            selection([("计划甲", "津贴责任")]), source_chunk(source), profile=profile()
        )


def test_explicit_scope_options_are_rules_and_not_invented_plan_entities():
    source = "示例产品\n可以在以下两种保障范围中选择一种：\n一、区域范围\n按约定区域承担责任。\n二、全国范围\n按全国范围承担责任。\n2.1 责任内容\n其他规则。\n"
    raw = {
        "entities": [{"name": "区域范围", "type": "保障计划", "evidence_ids": ["all"]}],
        "relationships": [],
    }
    with pytest.raises(EvidenceSemanticError, match="原文保障范围选项"):
        to_native(
            PointerResult.model_validate(raw), source_chunk(source), profile=profile()
        )
    raw["entities"][0]["type"] = "合同规则"
    assert (
        to_native(
            PointerResult.model_validate(raw), source_chunk(source), profile=profile()
        )["entities"][0]["type"]
        == "合同规则"
    )


def test_explicit_plan_table_can_support_a_plan_name_that_also_labels_scope():
    source = "示例产品\n可以在以下两种保障范围中选择一种：\n一、区域范围\n约定区域。\n二、全国范围\n全国。\n\n## 保障计划\n\n| 责任 | 区域范围 | 全国范围 |\n| --- | --- | --- |\n| 医疗责任 | 100元 | 200元 |\n"
    assert (
        len(
            to_native(
                selection([("区域范围", "医疗责任")]),
                source_chunk(source),
                profile=profile(),
            )["relationships"]
        )
        == 1
    )


def test_a_scope_to_responsibility_table_does_not_promote_its_scope_to_a_plan():
    source = "示例产品\n可以在以下两种保障范围中选择一种：\n一、区域范围\n约定区域。\n二、全国范围\n全国。\n\n## 保障范围\n\n| 保障范围 | 保障责任 |\n| --- | --- |\n| 区域范围 | 医疗责任 |\n| 全国范围 | 医疗责任 |\n"
    with pytest.raises(EvidenceSemanticError, match="原文保障范围选项"):
        to_native(
            selection([("区域范围", "医疗责任")]),
            source_chunk(source),
            profile=profile(),
        )


def test_malformed_table_row_is_not_silently_guessed():
    with pytest.raises(ValueError, match="row width"):
        table_facts("| 项目 | 计划甲 |\n| --- | --- |\n| 医疗责任 | 0元 | 多余列 |\n")


@pytest.mark.asyncio
async def test_semantic_feedback_repairs_the_model_output_before_any_approval(tmp_path):
    source = matrix()
    calls, events = [], []

    async def complete(prompt, **kwargs):
        calls.append(json.loads(prompt))
        reply = selection([("计划甲" if len(calls) == 1 else "计划乙", "医疗责任")])
        for row in [*reply.entities, *reply.relationships]:
            row.evidence_ids = ["e1"]
        return reply.model_dump_json()

    adapter = EvidenceExtractor(
        requests={"input": source_chunk(source)},
        native_system_prompt="native",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Types",
        selection_guidance="Evidence",
        complete=complete,
        profile=profile(),
        ledger_path=tmp_path / "ledger.json",
        record=events.append,
    )
    with pytest.raises(EvidenceSemanticError):
        await adapter("input", system_prompt="native")
    assert not adapter.selections
    assert not (tmp_path / "ledger.json").exists()
    native = json.loads(await adapter("input", system_prompt="native"))
    assert native["relationships"][0]["source"].endswith("计划乙")
    assert "单元='0元'" in calls[1]["previous_validation_error"]
    assert calls[0]["source_semantics"]["table_facts"][0]["raw_value"] == "0元"
    assert calls[0]["source_semantics"]["table_facts"][0]["evidence_ids"] == ["e1"]
    assert events[1]["validation_issues"][0]["kind"] == "unsupported_inclusion"
