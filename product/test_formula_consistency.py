import pytest

from product.formula_consistency import fee_formula_index, formula_conflicts

DOCS = {
    "[1]": {
        "excerpts": [
            {
                "text": "部分领取费用 = 部分领取金额 × 部分领取费用率\n退保费用 = 申请退保之日的保单账户价值 × 退保费用率"
            }
        ]
    }
}


def test_formula_bases_are_taken_from_source_not_draft():
    rows = fee_formula_index(DOCS)
    assert [r["base"] for r in rows] == ["部分领取金额", "申请退保之日的保单账户价值"]
    assert all(r["reference"] == "[1]" for r in rows)


@pytest.mark.parametrize(
    "claim",
    [
        "两者基数相同。",
        "**是的，计算基数在定义上是一致的**。",
        "两项费用基数一样，属于同一资金池。",
    ],
)
def test_affirmative_equivalence_is_blocked_even_if_formula_table_is_correct(claim):
    answer = claim + "\n" + DOCS["[1]"]["excerpts"][0]["text"] + "\n两者基数对象不同。"
    assert formula_conflicts(answer, fee_formula_index(DOCS))


@pytest.mark.parametrize(
    "claim",
    [
        "两者基数不相同。",
        "基数并不一样。",
        "两者基数一样吗？不是。",
        "不能说基数相同。",
        "费率相同，但基数不同。",
    ],
)
def test_negative_question_and_rate_comparisons_are_not_flagged(claim):
    assert not formula_conflicts(
        claim + "\n" + DOCS["[1]"]["excerpts"][0]["text"], fee_formula_index(DOCS)
    )
