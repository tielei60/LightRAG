"""Recorded source/answer failures and conservative obligation matching."""

import json
from pathlib import Path

import pytest

from applications.insurance.answer_requirements import (
    answer_requirement_conflicts,
    source_answer_requirements,
)


def failures():
    return json.loads(
        (Path(__file__).parent / "fixtures/answer_failures_v7.json").read_text()
    )


@pytest.mark.parametrize("case", failures(), ids=lambda case: case["id"])
def test_recorded_final_defect_is_detected_from_same_request_sources(case):
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert answer_requirement_conflicts(case["bad_answer"], requirements)
    assert not answer_requirement_conflicts(case["corrected_answer"], requirements)
    for fact in requirements["source_facts"]:
        excerpt = case["source_documents"][fact["reference"]]["excerpts"][
            fact["excerpt"]
        ]
        assert fact["text"] in excerpt["text"]


def test_underwriting_index_retains_other_products_complete_addition():
    case = failures()[0]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert requirements["requested_products"] == ["北京人寿京福盈嘉终身寿险"]
    assert any(
        fact["reference"] == "[2]"
        and "所有增额终身寿险累计应交总保费＋本险种累计基本保额" in fact["text"]
        for fact in requirements["source_facts"]
    )


@pytest.mark.parametrize(
    "question",
    [
        "北京人寿的产品呢？",
        "尊享版是什么？",
        "终身寿险怎么样？",
        "尊享版终身寿险呢？",
        "它呢？",
    ],
)
def test_generic_or_unresolved_name_does_not_authorize_hard_scope_rejection(question):
    case = failures()[0]
    requirements = source_answer_requirements(question, case["source_documents"])
    assert requirements["requested_products"] == []
    assert not answer_requirement_conflicts(case["bad_answer"], requirements)


def test_disclaimer_is_not_an_extra_business_assertion():
    case = failures()[0]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = (
        case["corrected_answer"] + "本题未询问京富添利，不能据此断言它是否需要体检。[2]"
    )
    assert not answer_requirement_conflicts(answer, requirements)


def test_partial_name_in_explicit_comparison_is_left_for_semantic_resolution():
    case = failures()[1]
    requirements = source_answer_requirements(
        "京福盈嘉和宏裕的增长因子怎么比较？", case["source_documents"]
    )
    assert requirements["requested_products"] == ["北京人寿京福盈嘉终身寿险"]
    assert requirements["possible_short_name_products"] == [
        "北京人寿京福宏裕（尊享版）终身寿险（分红型）"
    ]
    assert not requirements["product_scope_resolved"]
    answer = "京福盈嘉增长因子是1.025；北京人寿京福宏裕（尊享版）终身寿险（分红型）增长因子是1.02，红利非保证。[1][3]"
    assert not answer_requirement_conflicts(answer, requirements)


def test_history_reference_does_not_get_mistaken_for_an_unasked_product():
    case = failures()[1]
    requirements = source_answer_requirements(
        "京福盈嘉和前面那款的增长因子怎么比较？",
        case["source_documents"],
        [{"role": "user", "content": "前面我们说的是京福宏裕（尊享版）。"}],
    )
    assert (
        "北京人寿京福宏裕（尊享版）终身寿险（分红型）"
        in requirements["possible_short_name_products"]
    )
    assert not answer_requirement_conflicts(
        "京福盈嘉增长因子1.025；京福宏裕增长因子1.02，红利非保证。[1][3]", requirements
    )


def test_pure_factor_comparison_does_not_require_unrelated_phase_details():
    case = failures()[1]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = "京福盈嘉的增长因子是1.025，京福宏裕是1.02；它们不是保费或现金价值的保证收益率，宏裕红利非保证。[1][3]"
    assert not answer_requirement_conflicts(answer, requirements)


def test_free_look_definition_accepts_chinese_duration_but_is_not_invented():
    case = failures()[2]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = case["corrected_answer"].replace("15日", "十五天")
    assert not answer_requirement_conflicts(answer, requirements)
    documents = {
        "[1]": {
            "product_name": "北京人寿京福盈嘉终身寿险",
            "excerpts": [{"text": "犹豫期内解除可退还保险费。"}],
        }
    }
    unknown_period = source_answer_requirements(case["question"], documents)
    assert not answer_requirement_conflicts(
        "犹豫期内解除可退还保险费。[1]", unknown_period
    )


def test_a_defined_compound_factor_is_not_mistaken_for_a_missing_power():
    case = failures()[1]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = "京福盈嘉的复合增长因子为1.025^(N-1)，N为身故时所在保单年度数，对应比较项可写基本保险金额×增长因子。[3]"
    assert not answer_requirement_conflicts(answer, requirements)


def modality_failure():
    return json.loads(
        (Path(__file__).parent / "fixtures/answer_modality_failure_v9.json").read_text()
    )


def test_actual_possible_loss_warning_does_not_support_typical_value_comparison():
    case = modality_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    warnings = [
        fact
        for fact in requirements["source_facts"]
        if fact["kind"] == "surrender_loss_warning"
    ]
    assert warnings and all(
        fact["product_name"] == "北京人寿京福盈嘉终身寿险" for fact in warnings
    )
    for fact in warnings:
        assert (
            fact["text"]
            in case["source_documents"][fact["reference"]]["excerpts"][fact["excerpt"]][
                "text"
            ]
        )
    conflicts = answer_requirement_conflicts(case["bad_answer"], requirements)
    assert any("模态" in conflict for conflict in conflicts)
    assert not answer_requirement_conflicts(case["corrected_answer"], requirements)


@pytest.mark.parametrize(
    "modal", ["通常", "一般", "往往", "必然", "一定", "必定", "总是", "肯定"]
)
def test_frequency_or_necessity_is_not_implied_by_possible_loss(modal):
    case = modality_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = case["corrected_answer"] + f"现金价值{modal}低于已交保费。[1]"
    assert any(
        "模态" in conflict
        for conflict in answer_requirement_conflicts(answer, requirements)
    )


@pytest.mark.parametrize(
    "assertion",
    [
        "现金价值通常体现为解除合同时按精算原理计算的退还金额。[1]",
        "无法断言现金价值通常低于已交保费。[1]",
        "现金价值不一定低于已交保费。[1]",
        "这不代表现金价值通常低于已交保费。[1]",
        "不能据此断言已交保费一般高于现金价值。[1]",
        "资料没有说明现金价值通常低于已交保费。[1]",
        "现金价值可能低于已交保费。[1]",
        "按条款[9.4]所述的终止约定判断。[1]",
    ],
)
def test_ordinary_definition_negation_uncertainty_and_local_section_are_allowed(
    assertion,
):
    case = modality_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert not answer_requirement_conflicts(
        case["corrected_answer"] + assertion, requirements
    )


@pytest.mark.parametrize(
    "source,answer,expected_conflict",
    [
        (
            "本产品退保可能遭受损失。现金价值通常低于已交保费。",
            "现金价值一般低于已交保费。",
            False,
        ),
        (
            "本产品退保可能遭受损失。已交保费通常高于现金价值。",
            "现金价值往往低于已交保费。",
            False,
        ),
        (
            "本产品退保可能遭受损失。现金价值通常低于已交保费。",
            "现金价值一定低于已交保费。",
            True,
        ),
        (
            "本产品退保可能遭受损失。现金价值通常高于已交保费。",
            "现金价值通常低于已交保费。",
            True,
        ),
        (
            "本产品退保可能遭受损失。不能断言现金价值通常低于已交保费。",
            "现金价值一般低于已交保费。",
            True,
        ),
    ],
)
def test_explicit_same_product_source_can_support_comparison(
    source, answer, expected_conflict
):
    product = "甲方安宁终身寿险"
    documents = {"[1]": {"product_name": product, "excerpts": [{"text": source}]}}
    requirements = source_answer_requirements(f"{product}退保会有损失吗？", documents)
    assert bool(answer_requirement_conflicts(answer, requirements)) is expected_conflict


def test_other_product_warning_or_typical_comparison_cannot_supply_requested_scope():
    first, other = "甲方安宁终身寿险", "乙方顺意终身寿险"
    documents = {
        "[1]": {"product_name": first, "excerpts": [{"text": "退保可能遭受损失。"}]},
        "[2]": {
            "product_name": other,
            "excerpts": [{"text": "退保可能遭受损失，现金价值通常低于已交保费。"}],
        },
    }
    requirements = source_answer_requirements(f"{first}退保会有损失吗？", documents)
    assert answer_requirement_conflicts("现金价值通常低于已交保费。[1]", requirements)
    documents["[1]"]["excerpts"] = [{"text": "退保退还现金价值。"}]
    requirements = source_answer_requirements(f"{first}退保会有损失吗？", documents)
    assert not answer_requirement_conflicts(
        "现金价值通常低于已交保费。[1]", requirements
    )


@pytest.mark.parametrize("prefix", ["不能断言具体金额，", "无法确认具体数额，"])
def test_unrelated_negation_does_not_hide_later_affirmative_comparison(prefix):
    case = modality_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    answer = case["corrected_answer"] + prefix + "现金价值通常低于已交保费。[1]"
    assert any(
        "模态" in conflict
        for conflict in answer_requirement_conflicts(answer, requirements)
    )


def phase_failure():
    return json.loads(
        (Path(__file__).parent / "fixtures/answer_phase_failure_v9.json").read_text()
    )


def test_recorded_three_way_maximum_requires_its_source_phase_boundaries():
    case = phase_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    conflicts = answer_requirement_conflicts(case["bad_answer"], requirements)
    assert any("年龄对应的保单周年日零时" in conflict for conflict in conflicts)
    assert any(
        "最后约定支付日后的首个保单周年日零时" in conflict for conflict in conflicts
    )
    assert not answer_requirement_conflicts(case["corrected_answer"], requirements)


@pytest.mark.parametrize(
    "wording", ["最终赔付取三者中的最大值", "两者取金额较大者", "三个比较项中取最大值"]
)
def test_equivalent_maximum_phrasings_need_phase_but_pure_factor_does_not(wording):
    case = phase_failure()
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert answer_requirement_conflicts(wording + "。[1]", requirements)
    assert not answer_requirement_conflicts(
        "盈嘉的基本保额增长因子为1.025，宏裕为1.02，不是保费或现金价值保证收益率。[1][3]",
        requirements,
    )


def new_product_failures():
    return json.loads(
        (Path(__file__).parent / "fixtures/new_product_failures_v9.json").read_text()
    )


def test_actual_formatted_n_definition_is_not_a_missing_definition():
    case = new_product_failures()[0]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert not answer_requirement_conflicts(
        case["answer"] + "本合同终止。", requirements
    )
    for definition in [
        "**N 的含义**：指身故时所在的保单年度数",
        "`N` 为身故时所在保单年度数",
        "N（身故时所在保单年度数）",
    ]:
        assert not answer_requirement_conflicts(
            "本合同终止。基本保险金额×1.02^(N-1)。" + definition, requirements
        )
    assert answer_requirement_conflicts(
        "基本保险金额×1.02^(N-1)，N为未知参数。", requirements
    )


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in new_product_failures()
        if c["id"] in {"qizhou-02", "qizhou-04", "suiyue-02"}
    ],
    ids=lambda c: c["id"],
)
def test_actual_missing_conditions_fail_and_source_supported_completion_passes(case):
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert answer_requirement_conflicts(case["answer"], requirements)
    supplements = {
        "qizhou-02": "须被保险人在每个保单周年日零时生存。",
        "qizhou-04": "自保险公司收到解除合同通知书时起，本合同终止。",
        "suiyue-02": "日常生活持续依赖他人监护和照顾；永久不可逆指自疾病确诊或意外伤害发生日起，经过积极治疗180天后，仍无法通过现有医疗手段恢复。",
    }
    assert not answer_requirement_conflicts(
        case["answer"] + supplements[case["id"]], requirements
    )
    for fact in requirements["source_facts"]:
        assert (
            fact["text"]
            in case["source_documents"][fact["reference"]]["excerpts"][fact["excerpt"]][
                "text"
            ]
        )


def test_precision_requirements_are_not_invented_when_source_absent():
    documents = {
        "[1]": {
            "product_name": "测试保险产品",
            "excerpts": [{"text": "生存保险金按合同规定给付。"}],
        }
    }
    requirements = source_answer_requirements(
        "测试保险产品生存保险金怎么计算？护理状态要求？退保呢？", documents
    )
    assert not requirements["source_facts"]
    assert not answer_requirement_conflicts("见合同。", requirements)


@pytest.mark.parametrize(
    "wording",
    [
        "自保险公司收到您解除合同的通知书时，本合同即被解除（终止）。",
        "合同自保险公司收到解除合同通知书时终止。",
        "自保险公司收到您的解除本合同的通知书时起，本合同终止。",
    ],
)
def test_termination_event_accepts_equivalent_natural_wording(wording):
    case = next(c for c in new_product_failures() if c["id"] == "qizhou-04")
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert not answer_requirement_conflicts(case["answer"] + wording, requirements)


def test_death_formula_question_does_not_require_unasked_survival_schedule():
    case = next(c for c in new_product_failures() if c["id"] == "qizhou-03")
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert "survival_condition" not in {f["kind"] for f in requirements["source_facts"]}


def test_before_after_nursing_status_question_does_not_force_disease_group_expansion():
    case = next(c for c in new_product_failures() if c["id"] == "suiyue-02")
    requirements = source_answer_requirements(
        "北京人寿京康岁悦长期护理保险已满足护理状态后，是否仍给付疾病身故保险金？",
        case["source_documents"],
    )
    assert not {"care_dependence", "irreversibility_definition"} & {
        f["kind"] for f in requirements["source_facts"]
    }


def test_source_explicit_death_termination_is_required_without_inventing_it():
    case = new_product_failures()[0]
    requirements = source_answer_requirements(
        case["question"], case["source_documents"]
    )
    assert any(
        "合同终止" in c
        for c in answer_requirement_conflicts(case["answer"], requirements)
    )
    assert not answer_requirement_conflicts(
        case["answer"] + "给付时本合同终止。[1]", requirements
    )
    requirements = source_answer_requirements(
        "测试保险身故保险金怎么算？",
        {
            "[1]": {
                "product_name": "测试保险",
                "excerpts": [{"text": "身故保险金按现金价值给付。"}],
            }
        },
    )
    assert not answer_requirement_conflicts("按现金价值给付。[1]", requirements)
