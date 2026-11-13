import copy

import pytest

from .judge import validate_verdict


def valid():
    return {
        "rows": [
            {
                "id": "anonymous",
                "full_answer_passed": True,
                "core_passed": True,
                "scope_passed": True,
                "conditions_passed": True,
                "citations_passed": True,
                "extra_business_error": False,
                "high_risk_error": False,
                "source_references": ["synthetic.md:1 原文"],
                "notes": "核对通过。",
            }
        ]
    }


def test_verdict_requires_all_ids_and_actual_delivery():
    answers = {"anonymous": {"delivered": True}}
    validate_verdict(valid(), answers)
    with pytest.raises(ValueError, match="omitted or duplicated"):
        validate_verdict({"rows": []}, answers)
    with pytest.raises(ValueError, match="contradicts"):
        validate_verdict(valid(), {"anonymous": {"delivered": False}})


def test_review_cannot_override_private_arm_or_case_mapping():
    verdict = valid()
    verdict["rows"][0]["arm"] = "guessed-arm"
    with pytest.raises(ValueError, match="invalid verdict"):
        validate_verdict(verdict, {"anonymous": {"delivered": True}})


@pytest.mark.parametrize(
    "field,value",
    [("scope_passed", False), ("high_risk_error", True), ("source_references", [])],
)
def test_whole_answer_cannot_pass_with_missing_support_or_failed_dimension(
    field, value
):
    verdict = copy.deepcopy(valid())
    verdict["rows"][0][field] = value
    with pytest.raises(ValueError, match="contradicts"):
        validate_verdict(verdict, {"anonymous": {"delivered": True}})
