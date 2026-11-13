import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from .config import ROOT
from .contracts import contract_module
from .models import make_rag


def test_current_a_contract_is_frozen_and_only_import_prefixes_change():
    snapshot = json.loads((ROOT / "snapshots/current.json").read_text())
    for name, expected in snapshot["contract_modules"].items():
        frozen = (ROOT / "_current" / (name + ".py")).read_bytes()
        restored = frozen
        for patch in reversed(expected.get("patches", [])):
            restored = restored.replace(
                patch["after"].encode(), patch["before"].encode()
            )
        restored = restored.replace(
            b"applications.insurance_b._current.", b"applications.insurance."
        )
        assert hashlib.sha256(frozen).hexdigest() == expected["frozen_sha256"]
        assert hashlib.sha256(restored).hexdigest() == expected["source_sha256"]


def test_explicit_withdrawal_and_loan_applications_are_policy_service_not_claims():
    from ._current.evidence_semantics import material_purposes

    assert material_purposes(
        "您申请部分领取保单账户价值时，应填写申请书并提供保险合同及有效身份证件。"
    ) == {"policy_service"}
    assert material_purposes("申请保单贷款须提交保险合同和身份证件。") == {
        "policy_service"
    }
    assert material_purposes("申请身故保险金须提交保险合同和身份证件。") == {"claim"}
    assert material_purposes("保险合同和有效身份证件。") == set()


@pytest.mark.parametrize("wrong_type", ["理赔材料", "保全材料"])
@pytest.mark.parametrize(
    "text",
    [
        "示例保险。投保核保财务要求：超过原文规定门槛时须提供财务问卷及收入资产证明。",
        "示例保险。投保时申请抵交保费须提供原文规定的保费抵交申请书。",
    ],
)
def test_underwriting_evidence_cannot_be_relabelled_as_claim_or_policy_service(
    wrong_type,
    text,
):
    from ._current.evidence_extract import (
        EvidenceChunk,
        EvidenceUnit,
        PointerResult,
        to_native,
    )
    from ._current.evidence_semantics import EvidenceSemanticError
    from ._current.extraction_profile import ExtractionProfile

    profile = ExtractionProfile.from_mapping(
        yaml.safe_load((ROOT / "profiles_current/entity_type/products.yml").read_text())
    )
    unit = EvidenceUnit(
        "u",
        "synthetic.md",
        0,
        len(text),
        text,
        ("保险产品", "投保规则", "理赔材料", "保全材料"),
    )
    chunk = EvidenceChunk("synthetic.md", "示例保险", text, (unit,))

    def result(kind):
        return PointerResult.model_validate(
            {
                "entities": [
                    {"name": "示例保险", "type": "保险产品", "evidence_ids": ["u"]},
                    {"name": "财务要求", "type": kind, "evidence_ids": ["u"]},
                ],
                "relationships": [],
            }
        )

    with pytest.raises(EvidenceSemanticError):
        to_native(result(wrong_type), chunk, profile=profile)
    native = to_native(result("投保规则"), chunk, profile=profile)
    assert native["entities"][1]["type"] == "投保规则"


def test_candidate_uses_current_answer_contract_without_importing_live_a(
    settings, monkeypatch
):
    import lightrag

    shutil.copytree(ROOT / "profiles_current", settings.root / "profiles_current")
    shutil.copyfile(
        ROOT / "snapshots/current.json", settings.root / "snapshots/current.json"
    )
    candidate = replace(
        settings, values={**settings.values, "B_CONTRACT_REVISION": "current"}
    )
    monkeypatch.setattr(
        lightrag, "LightRAG", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    rag = make_rag(candidate)
    assert isinstance(
        rag.role_llm_configs["query"].func,
        contract_module(candidate, "answer_query").AnswerQuery,
    )
    assert rag.chunk_token_size == 8192
    assert candidate.profiles == settings.root / "profiles_current"
