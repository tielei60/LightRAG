"""Scope indexes preserve literal heading boundaries and never infer parents."""

import pytest

from product.source_scope import plan_scope_index


def documents(text, parents=()):
    return {"[1]": {"excerpts": [{"text": text, "governing_context": list(parents)}]}}


def test_plan_two_cannot_inherit_plan_one_death_formula():
    text = (
        "2.5 保险责任\n保障计划一\n生存保险金\n身故保险金：累计保费减生存金\n"
        "保障计划二\n生存保险金\n2.6 责任免除\n身故免责条款"
    )
    index = plan_scope_index(documents(text))
    assert [row["plan"] for row in index] == ["保障计划一", "保障计划二"]
    assert "身故保险金" in index[0]["text"]
    assert "身故" not in index[1]["text"]
    for row in index:
        assert row["text"] == text[row["start"] : row["end"]]


def test_unambiguous_parent_assigns_only_its_own_plan():
    index = plan_scope_index(
        documents("身故保险金公式。", ["产品标题", "保障计划一：\n适用条件"])
    )
    assert len(index) == 1
    assert index[0]["plan"] == "保障计划一"
    assert index[0]["text"] == "身故保险金公式。"


@pytest.mark.parametrize(
    "parents",
    [[], ["保障计划一\n正文\n保障计划二\n正文"], ["保障计划一", "保障计划二"]],
)
def test_absent_or_ambiguous_parent_never_creates_scope(parents):
    assert plan_scope_index(documents("身故保险金公式。", parents)) == []


def test_inline_plan_reference_is_not_a_heading():
    assert plan_scope_index(documents("与保障计划一相比，本项条件不同。")) == []
