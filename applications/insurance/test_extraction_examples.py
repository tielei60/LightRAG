"""Validate instructional examples against the actual evidence-pointer contract."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from applications.insurance.evidence_extract import (
    EvidenceChunk,
    EvidenceUnit,
    PointerResult,
    to_native,
)
from applications.insurance.extraction_profile import ExtractionProfile

PROFILE_PATH = (
    Path(__file__).parent / "prompt_profiles" / "entity_type" / "products.yml"
)
CONFIG = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
EXAMPLES = CONFIG["evidence_selection_examples"]


@pytest.fixture(scope="module")
def profile():
    return ExtractionProfile.from_mapping(CONFIG)


def example_chunk(example):
    source = example["input"]
    text = "\n\n".join(row["text"] for row in source["evidence"])
    units = []
    cursor = 0
    for row in source["evidence"]:
        end = cursor + len(row["text"])
        units.append(
            EvidenceUnit(
                row["id"],
                source["source_file"],
                cursor,
                end,
                text,
                tuple(row.get("entity_types", [])),
                row.get("role", "primary"),
            )
        )
        cursor = end + 2
    return EvidenceChunk(source["source_file"], source["product"], text, tuple(units))


def converted_example(product, profile):
    example = next(ex for ex in EXAMPLES if ex["input"]["product"] == product)
    return to_native(
        PointerResult.model_validate(example["output"]),
        example_chunk(example),
        profile=profile,
    )


@pytest.mark.parametrize(
    "index,example",
    list(enumerate(EXAMPLES)),
    ids=[example["input"]["product"] for example in EXAMPLES],
)
def test_native_and_pointer_examples_have_the_same_source_faithful_output(
    index, example, profile
):
    chunk = example_chunk(example)
    profile.validate_catalog(chunk.units)
    pointer = PointerResult.model_validate(example["output"])
    native = to_native(pointer, chunk, profile=profile)
    native_example = CONFIG["entity_extraction_json_examples"][index]
    assert native_example.startswith("---输入原文---\n")
    original, output = native_example.removeprefix("---输入原文---\n").split(
        "\n---输出---\n", maxsplit=1
    )
    assert original == chunk.content
    documented_native = json.loads(output)

    # This compares two maintained protocols: changing a quote, reference,
    # endpoint or type in either example must not silently teach another output.
    assert native == documented_native
    for kind in ("entities", "relationships"):
        for selection, record in zip(
            example["output"][kind], native[kind], strict=True
        ):
            selected = set(selection["evidence_ids"])
            copied_originals = "\n\n".join(
                row["text"]
                for row in example["input"]["evidence"]
                if row["id"] in selected
            )
            assert record["description"] == copied_originals
    assert set(native) == {"entities", "relationships"}
    assert example["input"]["product"].startswith("示例")


def test_taxonomy_keeps_legacy_aliases_and_separates_plans_from_groups(profile):
    assert profile.entity_type("Product") == "保险产品"
    assert profile.entity_type("产品") == "保险产品"
    assert profile.entity_type("Coverage") == "保险责任"
    assert profile.entity_type("保障责任") == "保险责任"
    assert profile.entity_type("Organization") == "组织"
    assert profile.entity_type("机构") == "组织"
    assert profile.entity_type("保障计划") != profile.entity_type("责任分组")
    assert profile.entity_type("全残责任") == "全残责任"
    with pytest.raises(ValueError, match="Unsupported evidence entity type"):
        profile.entity_type("高度残疾")
    assert profile.unscoped_entity_types == ("组织",)


def test_branches_keep_shared_conditions_exceptions_and_termination_scope(profile):
    native = converted_example("示例给付保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    young = entities["示例给付保险｜计划甲｜身故保险金｜未满18周岁疾病身故"]
    adult = entities["示例给付保险｜计划甲｜身故保险金｜年满18周岁疾病身故"]
    for row in [young, adult]:
        assert "等待期自本合同生效之日起90日" in row["description"]
        assert "不含附加合同保险费" in row["description"]
        assert "投保人故意杀害被保险人导致身故，不予给付" in row["description"]
    assert "给付后本合同终止" in young["description"]
    assert "给付后本项身故责任终止，其他责任依约继续" in adult["description"]
    assert "给付后本合同终止" not in adult["description"]
    assert "160%两者中的较大者" in adult["description"]


def test_definitions_and_local_disease_exceptions_are_not_new_coverage(profile):
    fees = converted_example("示例费用保险", profile)
    term = next(row for row in fees["entities"] if row["name"].endswith("｜药费"))
    assert term["type"] == "术语定义"
    assert "不包括营养品费用" in term["description"]

    diseases = converted_example("示例疾病保险", profile)
    entries = {row["name"]: row for row in diseases["entities"]}
    first = entries["示例疾病保险｜计划甲｜严重哮喘"]
    second = entries["示例疾病保险｜计划乙｜严重哮喘"]
    assert "满6个月" in first["description"]
    assert "满12个月" in second["description"]
    assert "疑似病例不属于本定义" in second["description"]
    assert not any(row["type"] == "免责条款" for row in diseases["entities"])
    assert not any(row["keywords"] == "保障疾病" for row in diseases["relationships"])


def test_service_provider_and_listed_pharmacy_are_distinct_objects(profile):
    native = converted_example("示例服务保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    assert entities["示例健康服务公司"]["type"] == "组织"
    assert entities["示例甲药店"]["type"] == "组织"
    assert entities["示例服务保险｜就医预约服务"]["type"] == "医疗服务"
    providers = [
        row
        for row in native["relationships"]
        if frozenset((row["source"], row["target"]))
        == frozenset(("示例健康服务公司", "示例服务保险｜就医预约服务"))
    ]
    assert len(providers) == 1
    assert "由示例健康服务公司提供" in providers[0]["description"]
    pharmacy_relations = [
        row
        for row in native["relationships"]
        if "示例甲药店" in (row["source"], row["target"])
    ]
    assert [row["keywords"] for row in pharmacy_relations] == ["列入名录"]


def test_premium_calculation_is_an_underwriting_rule_not_a_benefit_payment(profile):
    native = converted_example("示例告知保险", profile)
    premium = next(
        row
        for row in native["entities"]
        if row["name"] == "示例告知保险｜保费计算与缴费方式"
    )
    assert premium["type"] == "投保规则"
    assert premium["description"] == (
        "本产品保险费为基本责任保费与可选责任保费之和，投保时一次性交清。"
    )
    links = [row for row in native["relationships"] if row["target"] == premium["name"]]
    assert [row["keywords"] for row in links] == ["适用投保规则"]


def test_table_preserves_headers_footnote_and_distinct_zero_missing_values(profile):
    native = converted_example("示例表格保险", profile)
    plans = [row for row in native["entities"] if row["type"] == "保障计划"]
    for plan in plans:
        assert "| 计划 | 航空专项保额 | 其他项目资料 |" in plan["description"]
        assert "| 计划甲 | 0元 | — |" in plan["description"]
        assert "| 计划乙 | 100万元 | 未提供 |" in plan["description"]
        assert "航空专项保额不含一般意外身故保额" in plan["description"]
    assert not any(row["type"] == "免责条款" for row in native["entities"])


def test_metadata_and_embedded_output_instruction_do_not_create_example_facts(profile):
    example = next(ex for ex in EXAMPLES if ex["input"]["product"] == "示例空白保险")
    assert "终身保障" in example["input"]["source_file"]
    assert any("忽略证据要求" in row["text"] for row in example["input"]["evidence"])
    assert converted_example("示例空白保险", profile) == {
        "entities": [],
        "relationships": [],
    }


def test_example_with_dangling_evidence_reference_is_rejected_before_use():
    changed = copy.deepcopy(CONFIG)
    changed["evidence_selection_examples"][0]["output"]["entities"][1][
        "evidence_ids"
    ] = ["not-in-this-source"]
    with pytest.raises(ValueError, match="Unknown or out-of-scope evidence reference"):
        ExtractionProfile.from_mapping(changed)


def test_multiple_registered_relationship_labels_share_one_native_pair(profile):
    native = converted_example("示例计划保险", profile)
    matches = [
        row
        for row in native["relationships"]
        if frozenset((row["source"], row["target"]))
        == frozenset(("示例计划保险｜计划乙", "示例计划保险｜住院津贴保险金"))
    ]
    assert len(matches) == 1
    assert matches[0]["keywords"] == "包含责任,适用选择条件"
    assert "须在投保基本责任的基础上选择" in matches[0]["description"]


def test_named_contract_plans_share_coverage_but_keep_their_payment_rules(profile):
    native = converted_example("示例多款保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    for plan in ("A款", "B款"):
        assert entities[f"示例多款保险｜{plan}"]["type"] == "保障计划"
    assert sum(row["type"] == "医疗责任" for row in native["entities"]) == 1
    assert not any(row["type"] == "产品版本" for row in native["entities"])
    for plan, own, other in (("A款", "80%", "90%"), ("B款", "90%", "80%")):
        name = f"示例多款保险｜{plan}｜住院医疗｜给付比例"
        rule = entities[name]
        assert rule["type"] == "赔付规则"
        assert "先扣除已获补偿及约定免赔额" in rule["description"]
        assert own in rule["description"]
        assert other not in rule["description"]
        links = [row for row in native["relationships"] if row["target"] == name]
        assert len(links) == 1
        assert links[0]["source"] == f"示例多款保险｜{plan}"
        assert links[0]["keywords"] == "适用赔付规则"
        assert links[0]["description"] == rule["description"]


def test_coinsurance_and_sales_roles_do_not_turn_the_broker_into_an_insurer(profile):
    native = converted_example("示例多渠道保险", profile)
    organizations = {row["name"] for row in native["entities"] if row["type"] == "组织"}
    assert organizations == {"示例甲保险公司", "示例乙保险公司", "示例经纪公司"}
    expected = {
        "示例甲保险公司": {"承保", "主承保", "共同承保", "销售"},
        "示例乙保险公司": {"承保", "共同承保"},
        "示例经纪公司": {"销售"},
    }
    for organization, labels in expected.items():
        links = [
            row
            for row in native["relationships"]
            if {row["source"], row["target"]} == {organization, "示例多渠道保险"}
        ]
        assert len(links) == 1
        assert set(links[0]["keywords"].split(",")) == labels
        if organization == "示例甲保险公司":
            assert "为主承方" in links[0]["description"]
            assert "通过直营网点渠道销售" in links[0]["description"]
        if organization == "示例乙保险公司":
            assert "为从共方" in links[0]["description"]


def test_channel_age_rule_does_not_apply_to_the_other_sales_channel(profile):
    native = converted_example("示例多渠道保险", profile)
    channels = {row["name"] for row in native["entities"] if row["type"] == "销售渠道"}
    assert channels == {
        "示例多渠道保险｜经纪互联网渠道",
        "示例多渠道保险｜直营网点渠道",
    }
    links = [
        row for row in native["relationships"] if row["keywords"] == "适用投保规则"
    ]
    assert len(links) == 1
    assert links[0]["source"] == "示例多渠道保险｜经纪互联网渠道"
    assert links[0]["target"] == "示例多渠道保险｜经纪互联网渠道｜新投保年龄"
    assert "18至55周岁（含两端）" in links[0]["description"]
    assert "不说明直营网点渠道的年龄" in links[0]["description"]


def test_age_only_revision_keeps_old_rule_and_proves_the_unchanged_coverage(profile):
    native = converted_example("示例年度保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    old = entities["示例年度保险｜2025基础版｜投保年龄"]
    new = entities["示例年度保险｜2026修订版｜经纪互联网渠道｜新投保年龄"]
    assert "18至55周岁（含两端）" in old["description"]
    assert "18至60周岁（含两端）" not in old["description"]
    for text in (
        "2026年1月1日",
        "仅适用经纪互联网渠道新投保",
        "18至60周岁（含两端）",
        "仅替换2025基础版的投保年龄条目",
        "其他条款沿用2025基础版",
        "2025既有合同及2026续保仍按2025基础版执行",
    ):
        assert text in new["description"]
    revisions = [
        row for row in native["relationships"] if row["keywords"] == "修订约定"
    ]
    assert len(revisions) == 1
    assert revisions[0]["source"] == new["name"]
    assert revisions[0]["target"] == old["name"]
    assert old["description"] in revisions[0]["description"]
    assert new["description"] in revisions[0]["description"]
    carried = [
        row
        for row in native["relationships"]
        if row["source"] == "示例年度保险｜2026修订版" and row["keywords"] == "包含责任"
    ]
    assert len(carried) == 1
    assert carried[0]["target"] == "示例年度保险｜住院医疗保险金"
    assert "疾病等待期为30日，意外伤害无等待期" in carried[0]["description"]
    assert "其他条款沿用2025基础版" in carried[0]["description"]


def test_new_age_without_baseline_does_not_teach_inheritance(profile):
    native = converted_example("示例缺证版本保险", profile)
    assert {row["type"] for row in native["entities"]} == {
        "保险产品",
        "产品版本",
        "投保规则",
    }
    assert {row["keywords"] for row in native["relationships"]} == {
        "具有产品版本",
        "适用投保规则",
    }
    assert all("2025" not in row["name"] for row in native["entities"])
    rule = next(row for row in native["entities"] if row["type"] == "投保规则")
    assert "未提供基准版本、其他条款、生效日期和适用渠道" in rule["description"]


def test_nursing_benefit_waiver_and_appointment_service_remain_distinct(profile):
    native = converted_example("示例护理保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    benefit = entities["示例护理保险｜特定疾病护理保险金"]
    waiver = entities["示例护理保险｜护理豁免保险费"]
    service = entities["示例护理保险｜护理预约服务"]
    assert benefit["type"] == "护理责任"
    assert waiver["type"] == "豁免责任"
    assert service["type"] == "医疗服务"
    for text in (
        "等待期后首次确诊",
        "且达到该疾病的护理状态要求",
        "最多24次",
        "本合同终止",
    ):
        assert text in benefit["description"]
    assert "该疾病的护理状态要求" in waiver["description"]
    assert "豁免本身不导致本合同终止" in waiver["description"]
    links = [row for row in native["relationships"] if row["target"] == service["name"]]
    assert [row["keywords"] for row in links] == ["提供服务"]
    assert "不是保险金给付责任" in links[0]["description"]


def test_account_interest_and_fees_are_not_dividends_or_benefit_payments(profile):
    native = converted_example("示例账户保险", profile)
    rules = [row for row in native["entities"] if row["name"] != "示例账户保险"]
    assert len(rules) == 2
    assert {row["type"] for row in rules} == {"合同规则"}
    assert {row["keywords"] for row in native["relationships"]} == {"适用合同规则"}
    settle = next(row for row in rules if row["name"].endswith("｜保单账户结算"))
    assert "只适用于保单账户价值" in settle["description"]
    assert "超过最低保证部分不确定" in settle["description"]
    assert "不是全部所交保费的固定收益率" in settle["description"]
    fee = next(row for row in rules if row["name"].endswith("｜初始费用"))
    assert "扣除2%的初始费用" in fee["description"]
    assert "不是保险金赔付比例" in fee["description"]


def test_annuity_and_non_guaranteed_dividends_keep_separate_evidence(profile):
    native = converted_example("示例红利年金保险", profile)
    annuity = next(row for row in native["entities"] if row["type"] == "年金责任")
    dividend = next(row for row in native["entities"] if row["type"] == "分红规则")
    assert (
        "年领按基本保险金额给付，月领按基本保险金额的8%给付" in annuity["description"]
    )
    assert "不同时领取两种金额" in annuity["description"]
    assert "不确定且非保证" in dividend["description"]
    assert "8%" not in dividend["description"]
    links = [
        row for row in native["relationships"] if row["target"] == dividend["name"]
    ]
    assert [row["keywords"] for row in links] == ["适用分红规则"]


def test_clause_cash_value_formula_and_contract_table_are_selected_in_full(profile):
    native = converted_example("示例计算表保险", profile)
    assert {row["type"] for row in native["entities"]} == {"保险产品", "现金价值"}
    rule = next(row for row in native["entities"] if row["type"] == "现金价值")
    assert "犹豫期后解除合同" in rule["description"]
    assert "按合同约定退还现金价值" in rule["description"]
    assert "现金价值指本合同所具有的价值" in rule["description"]
    assert (
        "本合同所交保险费×（1－35%）×（1－已过保险期间日数／保险期间日数）"
        in rule["description"]
    )
    assert "经过日数不足一日的按一日计算" in rule["description"]
    assert "不以利益演示金额代替" in rule["description"]
    assert "| 已发生保险金给付 | 不退还现金价值 |" in rule["description"]
    assert "| 已发生保险事故但尚未支付保险金 | 不退还现金价值 |" in rule["description"]
    example = next(ex for ex in EXAMPLES if ex["input"]["product"] == "示例计算表保险")
    assert example["input"]["source_file"].endswith("条款.txt")
    selected = next(
        row for row in example["output"]["entities"] if row["type"] == "现金价值"
    )
    assert selected["evidence_ids"] == ["r", "t"]
    complete_support = "\n\n".join(
        row["text"] for row in example["input"]["evidence"] if row["id"] in {"r", "t"}
    )
    assert rule["description"] == complete_support
    assert len(native["entities"]) == 2
    assert [row["keywords"] for row in native["relationships"]] == ["适用现金价值规则"]
    assert native["relationships"][0]["description"] == complete_support


def test_only_confirmed_standalone_calculation_table_example_is_empty(profile):
    standalone = next(
        ex for ex in EXAMPLES if ex["input"]["product"] == "示例纯计算表保险"
    )
    assert "独立费率与现金价值表" in standalone["input"]["source_file"]
    assert "整份独立表文件，无条款正文" in standalone["input"]["evidence"][1]["text"]
    assert converted_example("示例纯计算表保险", profile) == {
        "entities": [],
        "relationships": [],
    }


def test_named_main_contract_is_a_reference_and_basic_coverage_is_not_a_main_policy(
    profile,
):
    native = converted_example("示例附加住院意外保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    product = "示例附加住院意外保险"
    main = product + "｜主合同｜示例住院意外伤害保险合同"
    assert entities[main]["type"] == "保险合同"
    assert [row["name"] for row in native["entities"] if row["type"] == "保险产品"] == [
        product
    ]
    for group in ("基本责任组", "可选责任组"):
        assert entities[product + "｜" + group]["type"] == "责任分组"
    links = [row for row in native["relationships"] if row["keywords"] == "附加于"]
    assert len(links) == 1
    assert links[0]["source"] == product
    assert links[0]["target"] == main
    assert "附加于《示例住院意外伤害保险合同》" in links[0]["description"]
    assert "本合同为附加合同" in links[0]["description"]
    assert "最长为一年" in entities[product + "｜保险期间"]["description"]
    effect = entities[product + "｜主附合同效力联动"]["description"]
    for text in (
        "主合同终止时，本附加合同终止",
        "主合同效力中止时，本附加合同效力亦中止",
        "主合同无效时，本附加合同亦无效",
    ):
        assert text in effect
    assert "本附加合同终止时，主合同终止" not in effect
    conflict = entities[product + "｜主合同条款适用与冲突处理"]["description"]
    assert "未约定事项以主合同为准" in conflict
    assert "相抵触，以本附加合同的条款为准" in conflict
    assert entities[product + "｜麻醉意外伤残保险金"]["type"] == "意外责任"


def test_unnamed_main_contract_does_not_invent_a_product_or_expand_premium_waiver(
    profile,
):
    native = converted_example("示例附加豁免保险", profile)
    assert not any(row["type"] == "保险合同" for row in native["entities"])
    assert not any(row["keywords"] == "附加于" for row in native["relationships"])
    entities = {row["name"]: row for row in native["entities"]}
    product = "示例附加豁免保险"
    scope = entities[product + "｜主合同适用范围"]
    assert scope["type"] == "合同规则"
    assert "本公司认可的人身保险合同" in scope["description"]
    assert "未列明具体主险名称" in scope["description"]
    people = entities[product + "｜投被保险人关系"]
    assert "投保人和被保险人为同一人，且为被豁免主合同的投保人" in people["description"]
    assert "未说明其等同于主合同被保险人" in people["description"]
    assert "保险期间与主合同相同" in entities[product + "｜保险期间"]["description"]
    links = [row for row in native["relationships"] if row["keywords"] == "豁免保费"]
    assert len(links) == 1
    assert links[0]["source"] == product + "｜疾病豁免保险费"
    assert links[0]["target"] == product + "｜被豁免合同范围"
    for text in (
        "达到本附加合同约定疾病条件后",
        "主合同及保险单指定附加合同的续期保险费",
        "未指定附加合同不在豁免范围内",
        "本附加合同终止，被豁免主合同继续有效",
    ):
        assert text in links[0]["description"]


def test_main_contract_reference_cannot_bypass_the_verified_product_root(profile):
    example = next(
        ex for ex in EXAMPLES if ex["input"]["product"] == "示例附加住院意外保险"
    )
    selection = copy.deepcopy(example["output"])
    main = next(row for row in selection["entities"] if row["type"] == "保险合同")
    main["type"] = "保险产品"
    with pytest.raises(ValueError, match="Product identity differs"):
        to_native(
            PointerResult.model_validate(selection),
            example_chunk(example),
            profile=profile,
        )


def test_same_document_names_follow_claim_and_policy_service_purposes(profile):
    native = converted_example("示例材料用途保险", profile)
    entities = {row["name"]: row for row in native["entities"]}
    claim = entities["示例材料用途保险｜身故理赔所需材料"]
    assert claim["type"] == "理赔材料"
    assert "保险合同、有效身份证件" in claim["description"]
    assert "申请理赔" in claim["description"]
    for item in ("解除合同", "保单变更", "复效"):
        material = entities[f"示例材料用途保险｜{item}所需材料"]
        assert material["type"] == "保全材料"
        relations = [
            row for row in native["relationships"] if row["target"] == material["name"]
        ]
        assert len(relations) == 1
        assert relations[0]["keywords"] == "需要保全材料"
        assert relations[0]["description"] == material["description"]
        assert "申请理赔" not in material["description"]
    surrender = entities["示例材料用途保险｜解除合同所需材料"]
    assert "若已发生保险金给付" in surrender["description"]
    assert "不退还现金价值" in surrender["description"]
    assert "示例材料用途保险" in [
        row["input"]["product"] for row in profile.selected_evidence_examples
    ]


def test_example_bank_rejects_surrender_relabelled_as_claim_material():
    config = copy.deepcopy(CONFIG)
    example = next(
        row
        for row in config["evidence_selection_examples"]
        if row["input"]["product"] == "示例材料用途保险"
    )
    material = next(
        row
        for row in example["output"]["entities"]
        if row["name"].endswith("｜解除合同所需材料")
    )
    material["type"] = "理赔材料"
    relation = next(
        row
        for row in example["output"]["relationships"]
        if row["target"] == material["name"]
    )
    relation["keywords"] = "需要理赔材料"
    with pytest.raises(
        ValueError, match="Invalid evidence selection example.*policy_service"
    ):
        ExtractionProfile.from_mapping(config)
