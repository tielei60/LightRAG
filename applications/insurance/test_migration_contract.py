"""Canonical extraction resumes exact contracts and rejects older source contracts."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from applications.insurance import evidence_extract
from applications.insurance.extraction_profile import ExtractionProfile
from lightrag.prompt import PROMPTS

CONTRACT_SOURCES = ("evidence_extract.py", "evidence_semantics.py")


@pytest.fixture
def current_adapter(tmp_path, monkeypatch):
    """Load current canonical code from an isolated contract fixture directory."""
    directory = tmp_path / "current_adapter"
    directory.mkdir()
    for name in CONTRACT_SOURCES:
        source = Path(evidence_extract.__file__).with_name(name)
        (directory / name).write_bytes(source.read_bytes())
    name = "insurance_contract_fixture"
    spec = importlib.util.spec_from_file_location(
        name, directory / "evidence_extract.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def profile():
    return ExtractionProfile.from_mapping(
        {
            "entity_types": ["保险产品", "保障计划", "保险责任", "合同规则"],
            "product_entity_type": "保险产品",
            "relationship_types": ["包含责任"],
            "semantic_roles": {
                "plan": ["保障计划"],
                "coverage": ["保险责任"],
                "inclusion": ["包含责任"],
                "scope_option": ["合同规则"],
            },
        }
    )


def requests(module, *, changed_evidence=False):
    result = {}
    sources = (
        ("policy input", "policy.md", "条款甲：十五日内可以撤销。"),
        ("underwriting input", "underwriting.md", "核保乙：投保年龄不超过六十周岁。"),
    )
    for prompt, filename, rule in sources:
        text = "示例产品\n" + rule
        if changed_evidence and filename == "underwriting.md":
            text += "另有一项条件。"
        unit = module.EvidenceUnit("durable-unit", filename, 0, len(text), text)
        result[prompt] = module.EvidenceChunk(filename, "示例产品", text, (unit,))
    return result


def outer_contract(module, *, adapter_source=None):
    # The importer hashes the adapter text together with the fixed batch inputs.
    return hashlib.sha256(
        json.dumps(
            {
                "adapter": (
                    Path(module.__file__).read_text()
                    if adapter_source is None
                    else adapter_source
                ),
                "profile": "frozen fixture profile",
                "catalogs": {"policy": "fixed", "underwriting": "fixed"},
                "manifest": {"batch": "migration-fixture", "workspace": "master"},
                "models": {"EXTRACTION_LLM_MODEL": "fixture-provider"},
                "native_system": "native system",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def adapter(module, ledger, complete, **changes):
    return module.EvidenceExtractor(
        requests=requests(
            module, changed_evidence=changes.get("changed_evidence", False)
        ),
        native_system_prompt="native system",
        native_summary_template=PROMPTS["summarize_entity_descriptions"],
        type_guidance="Fixture types",
        selection_guidance=changes.get("selection_guidance", "Evidence only"),
        complete=complete,
        ledger_path=ledger,
        ledger_context=changes.get("ledger_context", outer_contract(module)),
        profile=profile(),
    )


def selection(prompt):
    payload = json.loads(prompt)
    name = "条款甲" if payload["source_file"] == "policy.md" else "核保乙"
    return json.dumps(
        {
            "entities": [{"name": name, "type": "合同规则", "evidence_ids": ["e1"]}],
            "relationships": [],
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_same_contract_partial_ledger_resumes_without_reextracting(
    current_adapter, tmp_path
):
    async def complete(prompt, **kwargs):
        return selection(prompt)

    complete_ledger = tmp_path / "uninterrupted-ledger.json"
    uninterrupted = adapter(current_adapter, complete_ledger, complete)
    expected = [
        await uninterrupted(prompt, system_prompt="native system")
        for prompt in requests(current_adapter)
    ]

    async def interrupted(prompt, **kwargs):
        if json.loads(prompt)["source_file"] == "underwriting.md":
            raise RuntimeError("Simulated process interruption")
        return selection(prompt)

    partial_ledger = tmp_path / "partial-ledger.json"
    original = adapter(current_adapter, partial_ledger, interrupted)
    first = await original("policy input", system_prompt="native system")
    with pytest.raises(RuntimeError, match="Simulated process interruption"):
        await original("underwriting input", system_prompt="native system")
    partial = json.loads(partial_ledger.read_text())
    assert len(partial["selections"]) == 1

    resumed_calls = []

    async def resume_provider(prompt, **kwargs):
        filename = json.loads(prompt)["source_file"]
        assert filename != "policy.md", "The durable selection must be replayed"
        resumed_calls.append(filename)
        return selection(prompt)

    resumed = adapter(evidence_extract, partial_ledger, resume_provider)
    actual = [
        await resumed(prompt, system_prompt="native system")
        for prompt in requests(evidence_extract)
    ]
    assert actual[0] == first
    assert actual == expected
    assert resumed_calls == ["underwriting.md"]
    assert resumed.contract_hash == original.contract_hash
    assert resumed.contract_hash == uninterrupted.contract_hash
    assert json.loads(partial_ledger.read_text()) == json.loads(
        complete_ledger.read_text()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prompt", "evidence", *CONTRACT_SOURCES])
async def test_changed_contract_rejects_resume_without_modifying_ledger(
    current_adapter, tmp_path, change
):
    async def complete(prompt, **kwargs):
        return selection(prompt)

    ledger = tmp_path / "partial-ledger.json"
    original = adapter(current_adapter, ledger, complete)
    await original("policy input", system_prompt="native system")
    saved = ledger.read_bytes()
    provider_calls = []

    async def forbidden(*args, **kwargs):
        provider_calls.append(args)
        pytest.fail("An incompatible contract must not call the provider")

    changes = {}
    if change == "prompt":
        changes["selection_guidance"] = "Changed extraction instructions"
    elif change == "evidence":
        changes["changed_evidence"] = True
    else:
        source = Path(current_adapter.__file__).with_name(change)
        source.write_bytes(source.read_bytes() + b"\n# Changed contract input.\n")

    with pytest.raises(ValueError, match="Evidence ledger contract changed"):
        adapter(current_adapter, ledger, forbidden, **changes)
    assert ledger.read_bytes() == saved
    assert provider_calls == []


@pytest.mark.asyncio
async def test_old_import_source_contract_is_rejected_without_rewriting_ledger(
    tmp_path,
):
    source = Path(evidence_extract.__file__).read_text()
    previous_source = source
    for module in ("evidence_semantics", "extraction_profile"):
        canonical = f"from applications.insurance.{module} import"
        assert canonical in previous_source
        previous_source = previous_source.replace(
            canonical, f"from product.{module} import"
        )
    assert previous_source != source
    old_context = outer_contract(evidence_extract, adapter_source=previous_source)
    assert old_context != outer_contract(evidence_extract)

    async def complete(prompt, **kwargs):
        return selection(prompt)

    ledger = tmp_path / "previous-contract-ledger.json"
    original = adapter(evidence_extract, ledger, complete, ledger_context=old_context)
    await original("policy input", system_prompt="native system")
    saved = ledger.read_bytes()
    provider_calls = []

    async def forbidden(*args, **kwargs):
        provider_calls.append(args)
        pytest.fail("An old source contract must not call the provider")

    with pytest.raises(ValueError, match="Evidence ledger contract changed"):
        adapter(evidence_extract, ledger, forbidden)
    assert ledger.read_bytes() == saved
    assert provider_calls == []
