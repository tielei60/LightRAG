"""Replay registered, source-checked corrections without changing frozen imports."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from applications.insurance.evidence_extract import PointerResult, to_native
from applications.insurance.extraction_profile import ExtractionProfile


def apply_registered_repairs(root, registry, batches):
    """Overlay verified repairs in registry order, checking exact before states.

    This changes only the in-memory replay view. Original approvals, ledger,
    prepared chunks and their registered hashes remain immutable.
    """
    by_name = {batch.name: batch for batch in batches}
    for record in registry.get("repairs", []):
        path = (root / record["path"]).resolve(strict=True)
        if not path.is_relative_to((root / "output").resolve()):
            raise ValueError("Registered repair escapes output directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Registered repair hash changed")
        repair = json.loads(path.read_text())
        if (
            repair["workspace"] != registry["workspace"]
            or repair["state"] != "verified"
        ):
            raise ValueError("Registered repair is not verified for this workspace")
        for correction in repair["corrections"]:
            batch = by_name[correction["batch"]]
            key = correction["approval_key"]
            if (
                batch.approved[key]["native"] != correction["before_native"]
                or batch.selections[key]["native"] != correction["before_native"]
            ):
                raise ValueError("Registered repair before-state differs")
            chunks = {
                hashlib.sha256(prompt.encode()).hexdigest(): chunk
                for prompt, chunk in batch.requests.items()
            }
            profile_path = (
                Path(batch.config["PROMPT_DIR"])
                / "entity_type"
                / batch.config["ENTITY_TYPE_PROMPT_FILE"]
            )
            profile = ExtractionProfile.from_mapping(
                yaml.safe_load(profile_path.read_text())
            )
            selection = PointerResult.model_validate(correction["after_selection"])
            native = to_native(selection, chunks[key], profile=profile)
            if native != correction["after_native"]:
                raise ValueError(
                    "Registered repair no longer matches its source pointers"
                )
            batch.approved[key] = {
                **batch.approved[key],
                "native": copy.deepcopy(native),
            }
            batch.selections[key] = {
                "selection": selection.model_dump(),
                "native": native,
            }
