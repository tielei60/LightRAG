"""No provider calls: unapproved extraction request and trace contracts."""

import copy
import json

import pytest

from product.import_proposals import ProposalError, build_requests, collect_proposals
from product.import_units import canonical_sha256


def prepared_fixture(count=1):
    units = [
        {
            "unit_id": f"unit-{i}",
            "product": {"id": "synthetic", "label": "合成资料"},
            "scope": {"combinations": []},
            "primary": [{"evidence_id": f"primary-{i}", "quote": "A且B，仅适用甲。"}],
            "context": [{"evidence_id": "definition", "quote": "甲不包括乙。"}],
        }
        for i in range(count)
    ]
    result = {"documents": [{"units": units}]}
    result["prepared_sha256"] = canonical_sha256(result)
    return result


def response(row, **overrides):
    proposal = {
        "status": "proposed",
        "entities": [
            {
                "id": "rule",
                "name": "条件",
                "type": "合同规则",
                "evidence_ids": row["evidence_ids"],
            }
        ],
        "relationships": [],
    }
    return {
        "content": json.dumps(proposal, ensure_ascii=False),
        "model": "fake-offline",
        "finish_reason": "stop",
        "usage": {"total_tokens": 5},
        **overrides,
    }


def bundle(count=1):
    return build_requests(prepared_fixture(count), model="fake-offline")


@pytest.mark.asyncio
async def test_requests_and_same_call_responses_remain_unapproved():
    requests, events, calls = bundle(), [], []

    async def complete(request):
        assert events[-1]["state"] == "sending"
        calls.append(copy.deepcopy(request))
        return response(requests["requests"][0])

    result = await collect_proposals(
        requests, complete, allowed_calls=1, record=events.append
    )
    assert result["status"] == "unapproved"
    assert len(calls) == 1 and calls[0] == events[0]["request"]
    assert (
        events[1]["response"]["content"] == response(requests["requests"][0])["content"]
    )
    selected = result["selections"]["unit-0"]["entities"][0]
    assert selected["scope"] == requests["requests"][0]["scope"]
    assert set(selected["evidence_ids"]) == {"primary-0", "definition"}
    assert "approved" not in selected


@pytest.mark.asyncio
async def test_zero_or_insufficient_budget_makes_zero_calls():
    async def forbidden(_):
        raise AssertionError("No callback authorized")

    for budget in (0, 1):
        with pytest.raises(ProposalError, match="allowance"):
            await collect_proposals(
                bundle(2), forbidden, allowed_calls=budget, record=lambda _: None
            )


@pytest.mark.asyncio
async def test_failed_durable_sending_record_prevents_call():
    async def forbidden(_):
        raise AssertionError("Record must precede callback")

    def fail(_):
        raise OSError("disk full")

    with pytest.raises(OSError):
        await collect_proposals(bundle(), forbidden, allowed_calls=1, record=fail)


@pytest.mark.asyncio
async def test_unknown_outcome_stops_without_retry_or_sensitive_error_body():
    calls, events = [], []

    async def fails(request):
        calls.append(request)
        raise RuntimeError("api_key=SECRET endpoint=PRIVATE")

    result = await collect_proposals(
        bundle(2), fails, allowed_calls=2, record=events.append
    )
    assert len(calls) == result["calls_started"] == 1
    assert result["status"] == "incomplete_unapproved"
    assert "SECRET" not in json.dumps(events)
    assert events[-1]["state"] == "error_or_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "missing_dependency",
        "duplicate_key",
        "model_drift",
        "unknown_field",
        "secret_usage",
        "secret_response_id",
        "secret_finish_reason",
    ],
)
async def test_invalid_response_never_produces_selection(change):
    requests, events = bundle(), []
    value = response(requests["requests"][0])
    proposal = json.loads(value["content"])
    if change == "truncated":
        value["finish_reason"] = "length"
    elif change == "missing_dependency":
        proposal["entities"][0]["evidence_ids"] = ["primary-0"]
    elif change == "duplicate_key":
        value["content"] = '{"status":"proposed","status":"needs_review"}'
    elif change == "model_drift":
        value["model"] = "different-model"
    elif change == "unknown_field":
        proposal["confidence"] = 100
    elif change == "secret_usage":
        value["usage"] = {"api_key": "SECRET"}
    elif change == "secret_response_id":
        value["response_id"] = {"api_key": "SECRET"}
    else:
        value["finish_reason"] = {"api_key": "SECRET"}
    if change not in {"duplicate_key"}:
        value["content"] = json.dumps(proposal)

    async def complete(_):
        return value

    result = await collect_proposals(
        requests, complete, allowed_calls=1, record=events.append
    )
    assert result["selections"] == {}
    assert result["status"] == "incomplete_unapproved"
    assert "SECRET" not in json.dumps(events)


@pytest.mark.asyncio
async def test_model_review_request_stops_and_does_not_create_empty_success():
    requests = bundle(2)

    async def complete(_):
        return response(
            requests["requests"][0],
            content=json.dumps({"status": "needs_review", "reason": "范围不明确"}),
        )

    result = await collect_proposals(
        requests, complete, allowed_calls=2, record=lambda _: None
    )
    assert result["calls_started"] == 1 and result["selections"] == {}
    assert result["stages"][0]["state"] == "needs_review"


@pytest.mark.asyncio
async def test_request_credentials_rejected_before_recorder_or_callback():
    requests = bundle()
    requests["requests"][0]["request"]["api_key"] = "SECRET"
    requests["requests_sha256"] = canonical_sha256(
        {k: v for k, v in requests.items() if k != "requests_sha256"}
    )

    async def forbidden(_):
        raise AssertionError("No callback")

    with pytest.raises(ProposalError, match="credentials"):
        await collect_proposals(
            requests,
            forbidden,
            allowed_calls=1,
            record=lambda _: pytest.fail("Must reject before tracing"),
        )


def test_prepared_drift_rejected_and_requests_copy_scope():
    source = prepared_fixture()
    requests = build_requests(source, model="fake-offline")
    source["documents"][0]["units"][0]["scope"]["new"] = "changed"
    assert "new" not in requests["requests"][0]["scope"]
    with pytest.raises(ProposalError, match="drift"):
        build_requests(source, model="fake-offline")


def test_whole_request_and_output_reserve_must_fit():
    with pytest.raises(ProposalError, match="budget"):
        build_requests(prepared_fixture(), model="fake-offline", input_token_limit=1)
    with pytest.raises(ProposalError, match="budget"):
        build_requests(
            prepared_fixture(), model="fake-offline", context_token_limit=8192
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["count", "input", "reserve"])
async def test_execution_remeasures_before_any_call(change):
    requests = bundle()
    if change == "count":
        requests["requests"][0]["input_tokens"] = 1
    elif change == "input":
        requests["limits"]["input_tokens"] = 1
    else:
        requests["limits"]["context_tokens"] = 8192
    requests["requests_sha256"] = canonical_sha256(
        {k: v for k, v in requests.items() if k != "requests_sha256"}
    )

    async def forbidden(_):
        raise AssertionError("Preflight before callback")

    with pytest.raises(ProposalError, match="budget"):
        await collect_proposals(
            requests,
            forbidden,
            allowed_calls=1,
            record=lambda _: pytest.fail("No trace on failed preflight"),
        )
