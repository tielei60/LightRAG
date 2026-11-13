import json

import pytest

from ._current.source_scope import plan_scope_index
from .query_guard import (
    boundary_conflicts,
    guarded_provider,
    plan_conflicts,
    plan_heading_index,
    responsibility_index,
)


def documents(text):
    return {
        "[1]": {
            "source_file": "synthetic.md",
            "product_name": "示例保险",
            "excerpts": [{"text": text, "governing_context": []}],
        }
    }


@pytest.mark.parametrize(
    "boundary", ["2.6 责任免除", "## 2.6 责任免除", "## （二）责任免除"]
)
def test_last_plan_ends_before_general_exclusions_and_policy_benefits(boundary):
    text = (
        "保障计划一\n身故保险金\n计划一原文。\n保障计划二\n生存保险金\n计划二原文。\n"
        + boundary
        + "\n身故免责原文。\n## （六）保单利益\n合同总体包含身故保险金。"
    )
    plans = plan_scope_index(documents(text))
    assert len(plans) == 2
    assert "生存保险金" in plans[1]["text"]
    assert "身故" not in plans[1]["text"]
    assert "保单利益" not in plans[1]["text"]


def test_extension_clause_keeps_its_exclusive_original_responsibility_heading():
    text = "## 甲医疗保险金\n保险期间届满治疗未结束的，至出院止，最长连续20日。\n## 乙门诊保险金\n住院前3日和出院后10日。"
    index = responsibility_index(documents(text))
    assert len(index["entries"]) == 1
    row = index["entries"][0]
    assert row["responsibility_heading"] == "## 甲医疗保险金"
    assert (
        text[row["clause_span"][0] : row["clause_span"][1]].strip()
        == row["verbatim_clause"]
    )
    assert "乙门诊" not in row["verbatim_clause"]


def test_plan_heading_index_never_copies_a_sibling_or_exclusion_into_plan():
    docs = documents(
        "保障计划一\n## 【身故保险金】\n原文。\n保障计划二\n## 【生存保险金】\n原文。\n## （二）责任免除\n## 【身故保险金】"
    )
    rows = plan_heading_index(docs, plan_scope_index(docs))
    assert rows[1]["verbatim_headings_in_original_interval"] == ["## 【生存保险金】"]


def test_strict_boundary_conflict_does_not_reject_a_qualified_answer():
    docs = documents("累计保费超过20万元时需提供财务证明。")
    assert boundary_conflicts(
        "达到20万元就必须提供证明吗？", "是的，必须提供证明。", docs
    )
    assert not boundary_conflicts(
        "达到20万元就必须提供证明吗？",
        "刚好20万元未超过门槛，不能仅因金额断定必须提供。",
        docs,
    )
    assert not boundary_conflicts(
        "刚好30万元是否必须提供？", "是的，必须提供证明。", docs
    )


@pytest.mark.parametrize(
    "label",
    [
        "给付金额",
        "身故保险金",
        "身故保险金计算",
        "身故保险金的计算方式",
        "身故保险金给付标准",
    ],
)
def test_positive_formula_cannot_transfer_from_another_explicit_plan(label):
    docs = documents(
        "保障计划一\n身故保险金\n按原计划一约定给付。\n保障计划二\n生存保险金\n按月领取。\n## 2.6 责任免除\n身故免责条款。"
    )
    question = "示例保险计划二的身故保障是什么？"
    wrong = f"**1. 示例保险（保障计划二）**\n* **{label}：** 按计划一公式给付。"
    assert plan_conflicts(question, wrong, docs, plan_scope_index(docs))
    correct = "**1. 示例保险（保障计划二）**\n仅列生存保险金，未列明身故责任，不能套用计划一的公式。"
    assert not plan_conflicts(question, correct, docs, plan_scope_index(docs))


async def test_audit_self_debate_triggers_existing_protocol_recovery_not_a_positive(
    settings,
):
    async def provider(*args, **kwargs):
        return json.dumps(
            {
                "supported": False,
                "scope_consistent": True,
                "formula_consistent": True,
                "issues": ["等等，让我再想一下，为什么我要否定？"],
                "corrected_answer": None,
            }
        )

    guarded = guarded_provider(settings, provider, lambda e: None)
    request = {
        "question": "示例问题",
        "answer": "示例回答[1]",
        "source_documents": documents("示例原文。"),
    }
    result = json.loads(await guarded(json.dumps(request), system_prompt="audit"))
    assert "protocol_error" in result
    assert "supported" not in result


async def test_actual_augmented_audit_is_recorded_and_wrong_positive_verdict_is_rejected(
    settings,
):
    docs = documents("累计保费超过20万元时需提供财务证明。")
    request = {
        "question": "达到20万元就必须提供证明吗？",
        "answer": "是的，必须提供证明。[1]",
        "source_documents": docs,
    }
    calls, events = [], []
    positive = {
        "supported": True,
        "scope_consistent": True,
        "formula_consistent": True,
        "issues": [],
        "corrected_answer": None,
    }

    async def provider(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return json.dumps(positive)

    guarded = guarded_provider(settings, provider, events.append)
    result = json.loads(
        await guarded(json.dumps(request), system_prompt="original audit", stream=False)
    )
    assert not result["supported"]
    actual = json.loads(calls[0][0])
    assert actual["source_documents"] == docs
    assert actual["question"] == request["question"]
    assert actual["detected_shared_guard_conflicts"]
    assert events[0]["prompt"] == calls[0][0]
    assert events[0]["system_prompt"] == calls[0][1]["system_prompt"]
    assert events[-1]["state"] == "verdict_rejected"


async def test_malformed_audit_still_reaches_existing_bounded_protocol_recovery(
    settings,
):
    async def provider(*args, **kwargs):
        return "{malformed"

    guarded = guarded_provider(settings, provider, lambda e: None)
    request = {
        "question": "达到20万元就必须提供吗？",
        "answer": "是的，必须提供证明。",
        "source_documents": documents("超过20万元提供证明。"),
    }
    assert await guarded(json.dumps(request), system_prompt="audit") == "{malformed"
