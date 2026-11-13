from copy import deepcopy
from pathlib import Path

import pytest

from .common import read_json, save_json
from .evaluate import compare, evaluate, questions, score


def results(settings):
    return {
        "question_set_sha256": "q" * 64,
        "source_set_sha256": "s" * 64,
        "query_contract_sha256": "c" * 64,
        "transport": "SDK",
        "results": [
            {
                "case_id": case["id"],
                "question": case["question"],
                "repeat": 1,
                "delivered": True,
                "answer": "synthetic answer",
                "grade": "pending_independent_review",
            }
            for case in questions(settings)
        ],
    }


def test_original_question_fixture_is_request_only(settings):
    rows = questions(settings)
    assert len(rows) == 29
    assert all(set(row) == {"id", "question"} for row in rows)


@pytest.mark.parametrize(
    "field", ["question_set_sha256", "source_set_sha256", "query_contract_sha256"]
)
def test_comparison_rejects_uncontrolled_variants(settings, field):
    left = results(settings)
    right = deepcopy(left)
    right[field] = "changed"
    save_json(settings.root / "a.json", left)
    save_json(settings.root / "b.json", right)
    with pytest.raises(ValueError, match="control"):
        compare(settings, settings.root / "a.json", settings.root / "b.json")


def test_pending_review_is_not_promoted_to_pass(settings):
    value = results(settings)
    save_json(settings.root / "a.json", value)
    save_json(settings.root / "b.json", value)
    report = compare(settings, settings.root / "a.json", settings.root / "b.json")
    assert report["counts"]["pending_review"] == 29
    assert report["counts"]["both_passed"] == 0


def test_comparison_rejects_different_extraction_contracts(settings):
    left, right = results(settings), results(settings)
    left["extraction_contract_sha256"] = "old"
    right["extraction_contract_sha256"] = "new"
    save_json(settings.root / "a.json", left)
    save_json(settings.root / "b.json", right)
    with pytest.raises(ValueError, match="extraction control"):
        compare(settings, settings.root / "a.json", settings.root / "b.json")


def test_unmatched_rounds_are_not_dropped(settings):
    left = results(settings)
    right = deepcopy(left)
    right["results"].pop()
    save_json(settings.root / "a.json", left)
    save_json(settings.root / "b.json", right)
    with pytest.raises(ValueError, match="complete|unmatched"):
        compare(settings, settings.root / "a.json", settings.root / "b.json")


@pytest.mark.parametrize("mode", ["uneven", "wrong_round", "duplicate"])
def test_both_variants_cannot_share_invalid_rounds(settings, mode):
    value = results(settings)
    if mode == "uneven":
        value["results"][0]["repeat"] = 3
    elif mode == "wrong_round":
        for row in value["results"]:
            row["repeat"] = 2
    else:
        value["results"].append(deepcopy(value["results"][0]))
    save_json(settings.root / "a.json", value)
    save_json(settings.root / "b.json", value)
    with pytest.raises(ValueError, match="round"):
        compare(settings, settings.root / "a.json", settings.root / "b.json")


async def test_evaluation_sends_only_original_questions_and_keeps_failures(
    settings, monkeypatch
):
    from . import evaluate as module

    calls = []

    class Rag:
        async def aquery_llm(self, question, *, param):
            calls.append((question, param))
            if len(calls) == 1:
                raise RuntimeError("synthetic delivery failure")
            return {
                "status": "success",
                "llm_response": {
                    "llm_generated": True,
                    "content": "synthetic response",
                },
            }

        async def finalize_storages(self):
            pass

    async def initialize(*args):
        pass

    save_json(
        settings.run_dir / "readback-report.json",
        {"passed": True, "workspace": settings.workspace},
    )
    monkeypatch.setattr(module, "make_rag", lambda *args, **kwargs: Rag())
    monkeypatch.setattr(module, "initialize_rag", initialize)
    monkeypatch.setattr(module, "accepted_readback", lambda *args: {})
    monkeypatch.setattr(
        module, "evaluation_contract", lambda *args: {"question_set_sha256": "test"}
    )
    report = await evaluate(settings, repeats=1)
    rows = read_json(Path(report["results"]))["results"]
    assert [question for question, _ in calls] == [
        case["question"] for case in questions(settings)
    ]
    assert all(param.conversation_history == [] for _, param in calls)
    assert report["requests"] == 29 and report["delivered"] == 28
    assert rows[0]["grade"] == "delivery_failed"
    assert all(row["grade"] == "pending_independent_review" for row in rows[1:])


def review(value):
    return {
        "question_set_sha256": value["question_set_sha256"],
        "method": "independent_original_source_review",
        "reviewed_by": "offline test reviewer",
        "rows": [
            {
                "case_id": row["case_id"],
                "repeat": row["repeat"],
                "full_answer_passed": True,
                "core_passed": True,
                "scope_passed": True,
                "conditions_passed": True,
                "citations_passed": True,
                "extra_business_error": False,
                "high_risk_error": False,
                "source_references": ["synthetic source locator"],
            }
            for row in value["results"]
        ],
    }


def test_scoring_preserves_raw_results_and_requires_support(settings):
    value = results(settings)
    independent = review(value)
    save_json(settings.root / "results.json", value)
    save_json(settings.root / "review.json", independent)
    report = score(
        settings, settings.root / "results.json", settings.root / "review.json"
    )
    assert report["passed"] == 29
    assert read_json(settings.root / "results.json") == value
    independent["rows"][0]["extra_business_error"] = True
    save_json(settings.root / "review.json", independent)
    with pytest.raises(ValueError, match="conflicts"):
        score(settings, settings.root / "results.json", settings.root / "review.json")


def test_model_self_audit_is_not_an_independent_review(settings):
    value = results(settings)
    save_json(settings.root / "results.json", value)
    save_json(
        settings.root / "review.json", {"supported": True, "scope_consistent": True}
    )
    with pytest.raises(ValueError, match="independent"):
        score(settings, settings.root / "results.json", settings.root / "review.json")
