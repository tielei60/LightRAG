"""Native pending-parse/P pipeline entry point; no manual storage row writes."""

import json
import time
from pathlib import Path

from .common import digest, read_json, save_json
from .config import Settings
from .evidence_adapter import BExtractionAdapter
from .models import initialize_rag, make_rag, provider
from .prepare import checked_manifest, stage_input


def run_contract(settings: Settings) -> dict:
    implementation = {
        path.name: digest(path.read_bytes())
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        if not path.name.startswith(("test_", "smoke")) and path.name != "conftest.py"
    }
    return {
        "implementation_sha256": implementation,
        "settings": settings.public(),
        "manifest_sha256": digest((settings.run_dir / "manifest.json").read_bytes()),
        "a_inputs_sha256": digest((settings.run_dir / "a-inputs.json").read_bytes())
        if settings.get("PREPARATION") == "business"
        else None,
        "profile_sha256": digest(
            (settings.profiles / "entity_type/products.yml").read_bytes()
        ),
        "query_prompt_sha256": digest(
            (settings.profiles / "user_prompt/products.md").read_bytes()
        ),
        "engine": read_json(settings.contract_snapshot),
        "final_input_adapter_sha256": digest(
            (settings.root / "input_adapters/source_context.py").read_bytes()
        )
        if settings.get("FINAL_CONTEXT") == "source-v1"
        else None,
        "native_entry": "A business preparation -> native F"
        if settings.get("PREPARATION") == "business"
        else "pending_parse -> native P -> frozen pointer extraction -> native merge and stores",
        "pdf_math_adapter_sha256": digest(
            (settings.root / "input_adapters/pdf_math.py").read_bytes()
        )
        if settings.get("STRUCTURE_REPAIR") == "headers-math-v2"
        else None,
    }


def run_signature(settings: Settings) -> str:
    return digest(
        json.dumps(run_contract(settings), ensure_ascii=False, sort_keys=True)
    )


def accepted_readback(settings: Settings) -> dict:
    report = read_json(settings.run_dir / "readback-report.json")
    if (
        not report.get("passed")
        or report.get("workspace") != settings.workspace
        or report.get("run_contract_sha256") != run_signature(settings)
    ):
        raise ValueError(
            "Complete native readback of this exact frozen B contract is required"
        )
    return report


async def ingest(settings: Settings, *, products: set[str] | None = None) -> dict:
    settings.require_models()
    manifest = checked_manifest(settings)
    snapshot = run_contract(settings)
    snapshot_path = settings.run_dir / "run-snapshot.json"
    if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
        raise ValueError("The run contract changed; create a new B run and workspace")
    save_json(snapshot_path, snapshot)
    adapter = BExtractionAdapter(settings, provider(settings, "extract"))
    rag = make_rag(settings, extraction=adapter)
    receipts = []
    initialized = False
    try:
        await initialize_rag(rag, settings)
        initialized = True
        from lightrag.base import DocStatus
        from lightrag.parser.routing import resolve_chunk_options

        existing = await rag.doc_status.get_docs_by_statuses(
            [
                DocStatus.PROCESSED,
                DocStatus.PENDING,
                DocStatus.FAILED,
                DocStatus.PROCESSING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
            ],
            strict=True,
        )
        expected_names = {
            Path(entry["relative_path"]).name for entry in manifest["documents"]
        }
        if any(
            Path(status.file_path).name not in expected_names
            for status in existing.values()
        ):
            raise ValueError(
                "The B workspace contains documents outside its frozen corpus"
            )
        if any(
            DocStatus(status.status) != DocStatus.PROCESSED
            for status in existing.values()
        ):
            raise ValueError(
                "An incomplete native document exists; inspect its status and use the explicit native retry command"
            )
        by_name = {Path(status.file_path).name: key for key, status in existing.items()}
        if len(by_name) != len(existing):
            raise ValueError("A source has ambiguous native document identities")
        # A partial invocation still merges into the whole workspace's existing KG.
        for entry in manifest["documents"]:
            previous_id = by_name.get(Path(entry["relative_path"]).name)
            if previous_id is not None:
                adapter.restore(previous_id)
        for entry in manifest["documents"]:
            if products and entry["product_directory"] not in products:
                continue
            doc_id = by_name.get(Path(entry["relative_path"]).name)
            if doc_id is not None:
                receipts.append(
                    {
                        "source_document_id": entry["document_id"],
                        "native_document_id": doc_id,
                        "state": "processed",
                        "reused_processed_document": True,
                    }
                )
                continue
            path = stage_input(settings, entry)
            receipt = {
                "source_document_id": entry["document_id"],
                "file": path.name,
                "source_sha256": entry["sha256"],
                "state": "sending",
                "started_at": time.time(),
            }
            receipts.append(receipt)
            save_json(settings.run_dir / "import-receipts.json", receipts)
            track = await rag.apipeline_enqueue_documents(
                [""],
                file_paths=[str(path)],
                docs_format="pending_parse",
                parse_engine=entry["parser"],
                process_options="P",
                chunk_options=resolve_chunk_options(
                    rag.addon_params, process_options="P"
                ),
            )
            adapter.begin_document(track, entry)
            await rag.apipeline_process_enqueue_documents()
            statuses = await rag.doc_status.get_docs_by_track_id(track)
            if len(statuses) != 1:
                raise ValueError(
                    "Native ingestion did not return one unambiguous document"
                )
            doc_id, status = next(iter(statuses.items()))
            receipt.update(
                native_document_id=doc_id,
                track_id=track,
                state=DocStatus(status.status).value,
                metadata=status.metadata,
                chunks=len(status.chunks_list or []),
                completed_at=time.time(),
            )
            save_json(settings.run_dir / "import-receipts.json", receipts)
            if (
                receipt["state"] != "processed"
                or (status.metadata or {}).get("chunk_method") != "paragraph_semantic"
            ):
                raise RuntimeError(
                    "The document did not complete verified native-P ingestion"
                )
        report = {
            "expected_documents": 21,
            "processed_this_invocation": len(receipts),
            "documents": receipts,
            "full_corpus_passed": len(receipts) == 21
            and all(row["state"] == "processed" for row in receipts),
        }
        save_json(settings.run_dir / "import-report.json", report)
        return report
    finally:
        if initialized:
            await rag.finalize_storages()
