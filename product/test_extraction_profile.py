"""Configuration-driven extraction types, relations, examples, and cache identity."""

import copy
import json

import pytest

from lightrag.prompt import PROMPTS
from product.evidence_extract import (
    EvidenceChunk,
    EvidenceExtractor,
    EvidenceUnit,
    PointerResult,
    provider_example,
    to_native,
)
from product.extraction_profile import ExtractionProfile


def mapping():
    return {
        "entity_types": ["保险产品", "合同规则", "组织", "医疗服务"],
        "entity_type_aliases": {
            "Product": "保险产品",
            "产品": "保险产品",
            "Rule": "合同规则",
            "Organization": "组织",
        },
        "product_entity_type": "保险产品",
        "unscoped_entity_types": ["组织"],
        "relationship_types": ["适用规则", "提供服务"],
        "relationship_type_aliases": {"applies": "适用规则"},
        "relationship_types_guidance": "仅从原文选择联系；同一对端点只输出一条关系。",
        "evidence_selection_examples": [],
    }


def example():
    return {
        "input": {
            "product": "示例产品",
            "source_file": "示例条款.txt",
            "evidence": [
                {
                    "id": "sample1",
                    "text": "示例产品约定仅在条件甲成立时适用规则乙。",
                    "entity_types": ["保险产品", "合同规则"],
                },
            ],
        },
        "output": {
            "entities": [
                {"name": "示例产品", "type": "保险产品", "evidence_ids": ["sample1"]},
                {"name": "规则乙", "type": "合同规则", "evidence_ids": ["sample1"]},
            ],
            "relationships": [
                {
                    "source": "示例产品",
                    "target": "规则乙",
                    "keywords": "适用规则",
                    "evidence_ids": ["sample1"],
                }
            ],
        },
    }


def chunk():
    text = (
        "Fixture product\nRule A applies only with B. The hospital provides a service."
    )
    return EvidenceChunk(
        "fixture.txt",
        "Fixture product",
        text,
        (EvidenceUnit("e1", "fixture.txt", 0, len(text), text),),
    )


def result():
    return PointerResult.model_validate(
        {
            "entities": [
                {"name": "Fixture product", "type": "Product", "evidence_ids": ["e1"]},
                {"name": "A", "type": "Rule", "evidence_ids": ["e1"]},
            ],
            "relationships": [
                {
                    "source": "Fixture product",
                    "target": "A",
                    "keywords": "applies,提供服务",
                    "evidence_ids": ["e1"],
                }
            ],
        }
    )


def test_profile_expands_types_without_adapter_source_changes():
    config = mapping()
    config["entity_types"].append("健康告知")
    profile = ExtractionProfile.from_mapping(config)
    selected = result()
    selected.entities[1].type = "健康告知"
    native = to_native(selected, chunk(), profile=profile)
    assert [item["type"] for item in native["entities"]] == ["保险产品", "健康告知"]
    assert native["entities"][0]["name"] == "Fixture product"
    assert native["relationships"][0]["keywords"] == "适用规则,提供服务"
    assert native["entities"][1]["description"] == chunk().content


def test_only_configured_unscoped_types_skip_the_product_namespace():
    profile = ExtractionProfile.from_mapping(mapping())
    selected = result()
    selected.entities[0].name = "原文医院"
    selected.entities[0].type = "Organization"
    selected.relationships = []
    assert (
        to_native(selected, chunk(), profile=profile)["entities"][0]["name"]
        == "原文医院"
    )
    selected.entities[0].type = "医疗服务"
    assert (
        to_native(selected, chunk(), profile=profile)["entities"][0]["name"]
        == "Fixture product｜原文医院"
    )


@pytest.mark.parametrize("keywords", ["unknown", "适用规则,", "适用规则,,提供服务"])
def test_unregistered_or_empty_relation_keywords_are_refused(keywords):
    selected = result()
    selected.relationships[0].keywords = keywords
    with pytest.raises(ValueError, match="relationship type|empty label"):
        to_native(selected, chunk(), profile=ExtractionProfile.from_mapping(mapping()))


def test_catalog_types_use_aliases_and_reject_unregistered_types():
    profile = ExtractionProfile.from_mapping(mapping())
    original = chunk()
    good = EvidenceUnit(
        "e1", original.source, 0, len(original.content), original.content, ("Rule",)
    )
    profile.validate_catalog([good])
    bad = EvidenceUnit(
        "e1", original.source, 0, len(original.content), original.content, ("Unknown",)
    )
    with pytest.raises(ValueError, match="Unsupported evidence entity type"):
        profile.validate_catalog([bad])


def test_context_type_does_not_reclassify_untyped_primary_business_evidence():
    source = "Fixture product\nRule A applies only with B."
    split = source.index("Rule")
    units = (
        EvidenceUnit(
            "identity", "source.txt", 0, split, source, ("Product",), "product-identity"
        ),
        EvidenceUnit("e1", "source.txt", split, len(source), source),
    )
    selected = PointerResult.model_validate(
        {
            "entities": [
                {"name": "A", "type": "Rule", "evidence_ids": ["identity", "e1"]}
            ],
            "relationships": [],
        }
    )
    profile = ExtractionProfile.from_mapping(mapping())
    source_chunk = EvidenceChunk("source.txt", "Fixture product", source, units)
    assert (
        to_native(selected, source_chunk, profile=profile)["entities"][0]["type"]
        == "合同规则"
    )
    selected.entities[0].evidence_ids = ["identity"]
    with pytest.raises(ValueError, match="not supported"):
        to_native(selected, source_chunk, profile=profile)
    typed_primary = EvidenceUnit(
        "e1", "source.txt", split, len(source), source, ("组织",)
    )
    typed_chunk = EvidenceChunk(
        "source.txt", "Fixture product", source, (units[0], typed_primary)
    )
    selected.entities[0].evidence_ids = ["identity", "e1"]
    with pytest.raises(ValueError, match="not supported"):
        to_native(selected, typed_chunk, profile=profile)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("entity_types", ["组织", "组织"], "duplicate"),
        ("entity_type_aliases", {"Old": "Unknown"}, "not a registered"),
        ("entity_type_aliases", {"合同规则": "组织"}, "shadows"),
        ("product_entity_type", "Unknown", "registered entity type"),
        ("unscoped_entity_types", ["保险产品"], "non-product"),
        ("unscoped_entity_types", ["Unknown"], "non-product"),
        ("relationship_type_aliases", {"old": "Unknown"}, "not a registered"),
        ("relationship_types", [], "nonempty"),
        ("misspelled_profile_key", [], "Unknown extraction profile"),
    ],
)
def test_invalid_schema_fails_before_model_use(field, value, match):
    config = mapping()
    config[field] = value
    with pytest.raises((ValueError, TypeError), match=match):
        ExtractionProfile.from_mapping(config)


@pytest.mark.parametrize(
    "field",
    [
        "entity_types_guidance",
        "evidence_selection_guidance",
        "relationship_types_guidance",
    ],
)
@pytest.mark.parametrize("value", [42, "", " \n "])
def test_invalid_guidance_is_refused_during_profile_preflight(field, value):
    config = mapping()
    config[field] = value
    with pytest.raises(
        (ValueError, TypeError), match=field + " must be a nonempty string"
    ):
        ExtractionProfile.from_mapping(config)


@pytest.mark.parametrize(
    "field", ["entity_extraction_examples", "entity_extraction_json_examples"]
)
@pytest.mark.parametrize("value", [42, "not-a-list", [{}], [""], [" \n "]])
def test_invalid_native_example_shapes_are_refused_during_profile_preflight(
    field, value
):
    config = mapping()
    config[field] = value
    with pytest.raises(
        (ValueError, TypeError), match=field + " must be a list of nonempty strings"
    ):
        ExtractionProfile.from_mapping(config)


def test_valid_multiline_native_examples_are_not_parsed_as_json():
    config = mapping()
    config["entity_types_guidance"] = "第一行指导\n第二行指导"
    config["evidence_selection_guidance"] = "第一行证据规则\n第二行证据规则"
    config["entity_extraction_examples"] = ["---输入---\n原文\n---输出---\n实体协议"]
    config["entity_extraction_json_examples"] = ["---输入原文---\n示例\n---输出---\n{}"]
    assert ExtractionProfile.from_mapping(config).entity_types == tuple(
        config["entity_types"]
    )


def test_pointer_examples_are_validated_with_real_adapter_contract():
    config = mapping()
    config["evidence_selection_examples"] = [example()]
    profile = ExtractionProfile.from_mapping(config)
    assert len(profile.evidence_selection_examples) == 1
    malformed = copy.deepcopy(config)
    malformed["evidence_selection_examples"][0]["output"]["entities"][0][
        "evidence_ids"
    ] = ["foreign"]
    with pytest.raises(ValueError, match="example 1.*out-of-scope"):
        ExtractionProfile.from_mapping(malformed)
    malformed = copy.deepcopy(config)
    malformed["evidence_selection_examples"][0]["output"]["entities"][0][
        "description"
    ] = "Generated"
    with pytest.raises(ValueError, match="example 1"):
        ExtractionProfile.from_mapping(malformed)


def semantic_mapping():
    config = mapping()
    config["entity_types"] += ["保障计划", "医疗责任"]
    config["relationship_types"] += ["包含责任"]
    config["semantic_roles"] = {
        "plan": ["保障计划"],
        "coverage": ["医疗责任"],
        "inclusion": ["包含责任"],
        "scope_option": ["合同规则"],
    }
    return config


def test_semantic_roles_are_registered_and_round_trip_in_the_contract():
    config = semantic_mapping()
    profile = ExtractionProfile.from_mapping(config)
    assert profile.semantic_roles["coverage"] == ("医疗责任",)
    assert ExtractionProfile.from_mapping(profile.contract).contract == profile.contract
    assert ExtractionProfile.from_mapping(mapping()).semantic_roles == {}


@pytest.mark.parametrize("mutation", ["missing", "unknown", "empty", "alias"])
def test_semantic_roles_reject_incomplete_or_unregistered_labels(mutation):
    config = semantic_mapping()
    if mutation == "missing":
        del config["semantic_roles"]["inclusion"]
    elif mutation == "unknown":
        config["semantic_roles"]["coverage"] = ["Unknown"]
    elif mutation == "empty":
        config["semantic_roles"]["plan"] = []
    else:
        config["semantic_roles"]["scope_option"] = ["Rule"]
    with pytest.raises(ValueError, match="semantic_roles"):
        ExtractionProfile.from_mapping(config)


def test_selected_examples_have_explicit_stable_order_and_validate_the_bank():
    config = mapping()
    first, second = example(), example()
    second = json.loads(
        json.dumps(second, ensure_ascii=False).replace("示例产品", "第二示例产品")
    )
    config["evidence_selection_examples"] = [first, second]
    config["evidence_selection_example_names"] = ["第二示例产品", "示例产品"]
    profile = ExtractionProfile.from_mapping(config)
    assert [row["input"]["product"] for row in profile.selected_evidence_examples] == [
        "第二示例产品",
        "示例产品",
    ]
    assert ExtractionProfile.from_mapping(profile.contract).contract == profile.contract
    config["evidence_selection_example_names"] = ["示例产品"]
    config["evidence_selection_examples"][1]["output"]["entities"][0][
        "evidence_ids"
    ] = ["foreign"]
    with pytest.raises(ValueError, match="example 2.*out-of-scope"):
        ExtractionProfile.from_mapping(config)


@pytest.mark.parametrize("names", [[], ["missing"], ["示例产品", "示例产品"]])
def test_selected_examples_reject_empty_unknown_or_duplicate_names(names):
    config = mapping()
    config["evidence_selection_examples"] = [example()]
    config["evidence_selection_example_names"] = names
    with pytest.raises(ValueError, match="example_names"):
        ExtractionProfile.from_mapping(config)


@pytest.mark.asyncio
async def test_real_pointer_prompt_receives_schema_relation_guidance_examples_and_roles():
    seen = []

    async def complete(prompt, *, system_prompt):
        seen.append((json.loads(prompt), system_prompt))
        return result().model_dump_json()

    config = mapping()
    config["evidence_selection_examples"] = [example()]
    original = chunk()
    role_unit = EvidenceUnit(
        "e1",
        original.source,
        0,
        len(original.content),
        original.content,
        (),
        "structural-parent",
    )
    source = EvidenceChunk(
        original.source, original.product, original.content, (role_unit,)
    )
    options = {
        "requests": {"input": source},
        "native_system_prompt": "native",
        "native_summary_template": PROMPTS["summarize_entity_descriptions"],
        "type_guidance": "Types guidance",
        "selection_guidance": "Evidence guidance",
        "complete": complete,
    }
    profile = ExtractionProfile.from_mapping(config)
    adapter = EvidenceExtractor(**options, profile=profile)
    await adapter("input", system_prompt="native")
    assert seen[0][0]["evidence"][0]["role"] == "structural-parent"
    assert (
        json.dumps(
            provider_example(profile.evidence_selection_examples[0]), ensure_ascii=False
        )
        in seen[0][1]
    )
    assert profile.relationship_types_guidance in seen[0][1]
    assert '"enum": ["保险产品", "合同规则", "组织", "医疗服务"]' in seen[0][1]
    modified = copy.deepcopy(config)
    modified["relationship_types_guidance"] += "新增指导"
    other = EvidenceExtractor(
        **options, profile=ExtractionProfile.from_mapping(modified)
    )
    assert adapter.contract_hash != other.contract_hash
    modified = copy.deepcopy(config)
    modified["evidence_selection_examples"][0]["input"]["evidence"][0]["text"] += (
        "条件保持。"
    )
    other = EvidenceExtractor(
        **options, profile=ExtractionProfile.from_mapping(modified)
    )
    assert adapter.contract_hash != other.contract_hash
