import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from applications.insurance.evidence_extract import (
    EvidenceChunk,
    EvidenceUnit,
    PointerResult,
    to_native,
)
from applications.insurance.extraction_profile import ExtractionProfile
from applications.insurance.paths import INSURANCE_DIR
from applications.insurance.reviewed_repairs import apply_registered_repairs


def fixture(root):
    profile_path = INSURANCE_DIR / "prompt_profiles/entity_type/products.yml"
    profile = ExtractionProfile.from_mapping(yaml.safe_load(profile_path.read_text()))
    text = "示例保险｜理赔流程：申请人提供身份证件。"
    chunk = EvidenceChunk(
        "a.md", "示例保险", text, (EvidenceUnit("e1", "a.md", 0, len(text), text, ()),)
    )
    selection = {
        "entities": [
            {"name": "示例保险", "type": "保险产品", "evidence_ids": ["e1"]},
            {"name": "示例保险｜流程", "type": "理赔流程", "evidence_ids": ["e1"]},
        ],
        "relationships": [],
    }
    before = to_native(PointerResult.model_validate(selection), chunk, profile=profile)
    after_selection = copy.deepcopy(selection)
    after_selection["relationships"].append(
        {
            "source": "示例保险",
            "target": "示例保险｜流程",
            "keywords": "适用理赔流程",
            "evidence_ids": ["e1"],
        }
    )
    after = to_native(
        PointerResult.model_validate(after_selection), chunk, profile=profile
    )
    key = hashlib.sha256(b"prompt").hexdigest()
    batch = SimpleNamespace(
        name="batch",
        config={
            "PROMPT_DIR": str(profile_path.parent.parent),
            "ENTITY_TYPE_PROMPT_FILE": profile_path.name,
        },
        requests={"prompt": chunk},
        approved={key: {"native": before}},
        selections={key: {"selection": selection, "native": before}},
    )
    repair = {
        "workspace": "master",
        "state": "verified",
        "corrections": [
            {
                "batch": "batch",
                "approval_key": key,
                "before_native": before,
                "after_selection": after_selection,
                "after_native": after,
            }
        ],
    }
    path = root / "output/repair.json"
    path.parent.mkdir()
    path.write_text(json.dumps(repair))
    registry = {
        "workspace": "master",
        "repairs": [
            {
                "path": "output/repair.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    return batch, registry, path, key, after


def test_verified_overlay_replays_source_pointers_without_changing_original_file(
    tmp_path,
):
    batch, registry, path, key, after = fixture(tmp_path)
    original = path.read_bytes()
    apply_registered_repairs(tmp_path, registry, [batch])
    assert batch.approved[key]["native"] == after
    assert batch.selections[key]["native"] == after
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="before-state"):
        apply_registered_repairs(tmp_path, registry, [batch])


@pytest.mark.parametrize("damage", ["hash", "workspace", "state", "native", "evidence"])
def test_unverified_or_inconsistent_repairs_fail_closed(tmp_path, damage):
    batch, registry, path, _key, _after = fixture(tmp_path)
    before = copy.deepcopy(batch.approved)
    repair = json.loads(path.read_text())
    if damage in {"workspace", "state"}:
        repair[damage] = "wrong"
    if damage == "native":
        repair["corrections"][0]["after_native"]["relationships"] = []
    if damage == "evidence":
        repair["corrections"][0]["after_selection"]["relationships"][0][
            "evidence_ids"
        ] = ["missing"]
    path.write_text(json.dumps(repair, indent=2))
    if damage != "hash":
        registry["repairs"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        apply_registered_repairs(tmp_path, registry, [batch])
    assert batch.approved == before
