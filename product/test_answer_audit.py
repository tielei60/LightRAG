"""Delivery regressions: semantic failures never escape before review."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from product.answer_query import AnswerError, AnswerQuery
from product.test_grounded_query import native_prompt, wrapped

PASS = {
    "supported": True,
    "scope_consistent": True,
    "formula_consistent": True,
    "issues": [],
    "corrected_answer": None,
}


def context(text):
    return native_prompt([{"reference_id": "1", "content": wrapped(text)}])


def provider_for(draft, verdicts):
    async def respond(*args, **kwargs):
        if kwargs.get("stream"):

            async def tokens():
                yield draft

            return tokens()
        return json.dumps(
            verdicts.pop(0) if len(verdicts) > 1 else verdicts[0], ensure_ascii=False
        )

    return AsyncMock(side_effect=respond)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,bad,good,field",
    [
        (
            "计划一：生存保险金、身故保险金。计划二：生存保险金。",
            "计划二同样给付计划一的身故保险金。[1]",
            "计划二只列生存保险金，不能套用计划一身故责任。[1]",
            "scope_consistent",
        ),
        (
            "部分领取费用＝部分领取金额×费率。退保费用＝申请退保当日账户价值×费率。",
            "部分领取费用＝部分领取金额×费率，退保费用＝申请退保当日账户价值×费率；基数相同，属于同一资金池。[1]",
            "两项费用基数不同，分别为部分领取金额与申请退保当日账户价值。[1]",
            "formula_consistent",
        ),
    ],
)
async def test_bad_conclusion_repaired_and_independently_rechecked(
    source, bad, good, field
):
    failure = {
        **PASS,
        field: False,
        "issues": ["结论与原文矛盾"],
        "corrected_answer": good,
    }
    provider = provider_for(bad, [failure, PASS.copy()])
    events = []
    adapter = AnswerQuery(provider, record=events.append)
    stream = await adapter("比较两项约定", system_prompt=context(source), stream=True)
    parts = [part async for part in stream]
    assert len(parts) == 1
    assert parts[0].startswith(good)
    assert bad not in parts[0]
    assert provider.await_count == 3
    requests = [json.loads(call.args[0]) for call in provider.call_args_list[1:]]
    assert requests[0]["answer"] == bad
    assert requests[1]["answer"] == good
    assert requests[0]["source_documents"] == requests[1]["source_documents"]
    assert source in str(requests[1]["source_documents"])
    assert events[-1]["semantic_audited"] is True
    assert events[-1]["audit_calls"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict",
    [
        {**PASS, "supported": False},
        {**PASS, "issues": ["仍有范围错误"]},
        {**PASS, "supported": "true"},
        {**PASS, "corrected_answer": "未审核的新答案。[1]"},
        {},
    ],
)
async def test_invalid_or_rejected_audit_emits_nothing(verdict):
    events = []
    adapter = AnswerQuery(provider_for("草稿。[1]", [verdict]), record=events.append)
    stream = await adapter("问题", system_prompt=context("原文"), stream=True)
    with pytest.raises(AnswerError):
        await anext(stream)
    assert events[-1]["state"] == "incomplete"
    assert events[-1]["answer_released"] is False
    assert adapter._provider_slots._value == 2


@pytest.mark.asyncio
async def test_repeated_rejection_has_no_third_repair_and_no_delivery():
    failure = {
        **PASS,
        "supported": False,
        "issues": ["错误"],
        "corrected_answer": "仍错。[1]",
    }
    provider = provider_for("错。[1]", [failure, failure])
    adapter = AnswerQuery(provider)
    stream = await adapter("问题", system_prompt=context("原文"), stream=True)
    with pytest.raises(AnswerError):
        await anext(stream)
    assert provider.await_count == 5


@pytest.mark.asyncio
async def test_unknown_citation_in_correction_is_rejected():
    provider = provider_for(
        "错。[1]",
        [
            {
                **PASS,
                "supported": False,
                "issues": ["错误"],
                "corrected_answer": "修正。[999]",
            }
        ],
    )
    with pytest.raises(AnswerError):
        await AnswerQuery(provider)("问题", system_prompt=context("原文"))
    assert provider.await_count == 4


@pytest.mark.asyncio
async def test_no_first_token_until_audit_finishes_and_cancellation_releases_slot():
    auditing = asyncio.Event()

    async def provider(*args, **kwargs):
        if kwargs.get("stream"):

            async def tokens():
                yield "草稿。[1]"

            return tokens()
        auditing.set()
        await asyncio.Event().wait()

    adapter = AnswerQuery(provider)
    stream = await adapter("问题", system_prompt=context("原文"), stream=True)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(auditing.wait(), 1)
    assert not pending.done()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert adapter._provider_slots._value == 2


@pytest.mark.asyncio
async def test_audit_timeout_emits_nothing():
    async def provider(*args, **kwargs):
        if kwargs.get("stream"):

            async def tokens():
                yield "草稿。[1]"

            return tokens()
        await asyncio.Event().wait()

    adapter = AnswerQuery(provider, provider_timeout=0.05)
    with pytest.raises(AnswerError, match="超时"):
        await adapter("问题", system_prompt=context("原文"))
    assert adapter._provider_slots._value == 2


@pytest.mark.asyncio
async def test_stream_and_nonstream_use_same_gate():
    adapter = AnswerQuery(provider_for("已核实。[1]", [PASS.copy(), PASS.copy()]))
    direct = await adapter("问题", system_prompt=context("原文"))
    stream = await adapter("问题", system_prompt=context("原文"), stream=True)
    assert direct == "".join([part async for part in stream])


def recorded_failures():
    from pathlib import Path

    return json.loads(
        (Path(__file__).parent / "fixtures/audit_failures_v5.json").read_text()
    )


@pytest.mark.asyncio
async def test_actual_schema_echo_is_retried_without_losing_correct_draft():
    provider = provider_for(
        "正确计划表。[1]", [recorded_failures()["schema_echo"], PASS.copy()]
    )
    events = []
    answer = await AnswerQuery(provider, record=events.append)(
        "计划表？", system_prompt=context("原文计划表")
    )
    assert answer.startswith("正确计划表。[1]")
    assert provider.await_count == 3
    requests = [json.loads(call.args[0]) for call in provider.call_args_list[1:]]
    assert requests[0]["answer"] == requests[1]["answer"]
    assert requests[1]["validation_feedback"]["kind"] == "invalid_audit_protocol"
    assert events[-1]["audit_calls"] == 2


@pytest.mark.asyncio
async def test_actual_self_contradictory_rejection_needs_fresh_positive_review():
    data = recorded_failures()
    good = "计划二只列生存金，不能套用计划一身故责任。[1]"
    repair = {**data["plan_correction"], "corrected_answer": good}
    provider = provider_for(
        "计划二给付身故金。[1]", [repair, data["contradictory_rejection"], PASS.copy()]
    )
    answer = await AnswerQuery(provider)(
        "计划二？", system_prompt=context("计划二：生存金")
    )
    assert answer.startswith(good)
    assert provider.await_count == 4
    last = json.loads(provider.call_args.args[0])
    assert last["answer"] == good
    assert last["validation_feedback"]["kind"] == "unresolved_negative_verdict"


@pytest.mark.asyncio
async def test_actual_missing_citations_are_repaired_then_reviewed_again():
    data = recorded_failures()["citationless_correction"]
    cited = {**data, "corrected_answer": data["corrected_answer"] + " [1]"}
    provider = provider_for("两种费用基数相同。[1]", [data, cited, PASS.copy()])
    answer = await AnswerQuery(provider)(
        "费用基数一样吗？", system_prompt=context("两个费用公式")
    )
    assert answer.startswith(cited["corrected_answer"])
    assert provider.await_count == 4
    requests = [json.loads(call.args[0]) for call in provider.call_args_list[1:]]
    assert requests[1]["answer"] == requests[0]["answer"]
    assert requests[2]["answer"] == cited["corrected_answer"]
    assert "引用" in requests[1]["validation_feedback"]["error"]


@pytest.mark.asyncio
async def test_all_recovery_budgets_share_five_call_limit():
    data = recorded_failures()
    uncited = data["citationless_correction"]
    cited = {**uncited, "corrected_answer": uncited["corrected_answer"] + " [1]"}
    provider = provider_for(
        "错误。[1]",
        [
            data["schema_echo"],
            uncited,
            cited,
            data["contradictory_rejection"],
            PASS.copy(),
        ],
    )
    answer = await AnswerQuery(provider)("问题", system_prompt=context("原文"))
    assert answer.startswith(cited["corrected_answer"])
    assert provider.await_count == 6  # one draft + five audit calls


@pytest.mark.asyncio
async def test_second_factual_correction_is_verified_before_delivery():
    first = {
        **PASS,
        "supported": False,
        "issues": ["范围错误"],
        "corrected_answer": "第一版修正。[1]",
    }
    second = {
        **PASS,
        "supported": False,
        "issues": ["仍有公式错误"],
        "corrected_answer": "第二版已核实修正。[1]",
    }
    provider = provider_for("错误草稿。[1]", [first, second, PASS.copy()])
    result = await AnswerQuery(provider)("问题", system_prompt=context("原文"))
    assert result.startswith(second["corrected_answer"])
    assert first["corrected_answer"] not in result
    assert (
        json.loads(provider.call_args.args[0])["answer"] == second["corrected_answer"]
    )
    assert provider.await_count == 4


@pytest.mark.asyncio
async def test_positive_model_verdict_cannot_override_formula_conflict():
    source = "部分领取费用 = 部分领取金额 × 部分领取费用率\n退保费用 = 申请退保之日的保单账户价值 × 退保费用率"
    bad = "部分领取费用和退保费用的计算基数在定义上是一致的。[1]"
    provider = provider_for(bad, [PASS.copy()])
    with pytest.raises(AnswerError, match="公式基数"):
        await AnswerQuery(provider)("基数一样吗？", system_prompt=context(source))
    assert provider.await_count == 4


@pytest.mark.asyncio
async def test_formula_conflict_can_be_repaired_after_false_positive_audit():
    source = "部分领取费用 = 部分领取金额 × 部分领取费用率\n退保费用 = 申请退保之日的保单账户价值 × 退保费用率"
    bad = "部分领取费用和退保费用基数相同。[1]"
    good = "部分领取费用和退保费用基数不同，分别为部分领取金额和退保日账户价值。[1]"
    corrected = {
        **PASS,
        "formula_consistent": False,
        "issues": ["基数不一致"],
        "corrected_answer": good,
    }
    provider = provider_for(bad, [PASS.copy(), corrected, PASS.copy()])
    answer = await AnswerQuery(provider)("基数一样吗？", system_prompt=context(source))
    assert answer.startswith(good)
    assert provider.await_count == 4
