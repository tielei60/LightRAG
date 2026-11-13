"""Revalidate partial development selections after a declared shared guard fix.

This never copies native database rows or resets status. Existing successful
native work remains in the same developmental workspace; failed work is later
retried exclusively through the normal sticky native retry channel.
"""

from ..common import digest, read_json, save_json
from ..contracts import contract_module
from ..evidence_adapter import BExtractionAdapter
from ..ingest import run_contract


def revalidate_partial(previous, current):
    if previous.workspace != current.workspace or previous.run_id == current.run_id:
        raise ValueError(
            "Partial development adoption requires the same workspace and a new run id"
        )
    if read_json(previous.run_dir / "manifest.json") != read_json(
        current.run_dir / "manifest.json"
    ):
        raise ValueError("Source set changed during partial adoption")
    module = contract_module(current, "evidence_extract")
    validated, rejected = [], []
    for path in sorted((previous.run_dir / "catalogs").glob("*.json")):
        catalog = read_json(path)
        ledger_path = previous.run_dir / "selections" / path.name
        if not ledger_path.exists():
            continue
        adapter = BExtractionAdapter(current, None)
        adapter._hydrate(catalog)
        delegate = adapter.delegate
        old = read_json(ledger_path)["selections"]
        for key, row in old.items():
            try:
                prompt = delegate.request_keys[key]
                chunk = delegate.requests[prompt]
                selection = module.PointerResult.model_validate(row["selection"])
                native = module.to_native(selection, chunk, profile=delegate.profile)
                if native != row["native"] or digest(prompt) != key:
                    raise ValueError("Native interpretation or prompt identity changed")
                delegate.persist_selection(key, selection, native)
                validated.append({"document": catalog["source"], "prompt_sha256": key})
            except ValueError as exc:
                rejected.append(
                    {
                        "document": catalog["source"],
                        "prompt_sha256": key,
                        "error": str(exc),
                    }
                )
    if rejected:
        save_json(current.run_dir / "adoption-rejected.json", rejected)
        raise ValueError(
            "Prior selections do not all validate; no native retry authorized"
        )
    save_json(
        current.run_dir / "partial-adoption.json",
        {
            "previous_run": previous.run_id,
            "workspace": current.workspace,
            "validated_selections": len(validated),
            "validation": "Exact native prompt and evidence identities, current pointer/schema/semantic checks, byte-equivalent native interpretation",
            "native_rows_copied": 0,
            "native_status_reset": False,
            "cold_extraction_claim": False,
            "selections": validated,
        },
    )
    save_json(current.run_dir / "run-snapshot.json", run_contract(current))
    return {"validated": len(validated), "native_rows_changed": 0}
