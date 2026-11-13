"""Explicit one-document retry through the native sticky manual-retry channel."""

import time
import uuid
from pathlib import Path

from .common import read_json, save_json
from .config import Settings
from .evidence_adapter import BExtractionAdapter
from .ingest import run_contract
from .models import initialize_rag, make_rag, provider
from .prepare import checked_manifest, stage_input


async def run_native_retry(settings: Settings, rag, document_id: str) -> dict:
    """Publish one manual intent only after proving no other unfinished target exists."""
    from lightrag.base import DocStatus
    from lightrag.kg.pipeline_ingress import (
        ManualRetryPublishResult,
        PipelineIngressMessage,
    )
    from lightrag.kg.shared_storage import get_namespace_lock, get_pipeline_ingress

    unfinished = await rag.doc_status.get_docs_by_statuses(
        [
            DocStatus.PENDING,
            DocStatus.PROCESSING,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
            DocStatus.FAILED,
        ],
        strict=True,
    )
    if set(unfinished) != {document_id}:
        raise ValueError(
            "Retry requires exactly the selected unfinished document; no other document is scheduled implicitly"
        )
    request_id = "b-retry-" + uuid.uuid4().hex
    if DocStatus(unfinished[document_id].status) == DocStatus.FAILED:
        ingress = await get_pipeline_ingress(settings.workspace)
        async with get_namespace_lock("pipeline_status", workspace=settings.workspace):
            outcome = ingress.request_manual_retry(
                request_id,
                PipelineIngressMessage(
                    kind="rescan", retry_failed=True, request_id=request_id
                ),
            )
            if outcome != ManualRetryPublishResult.ACCEPTED:
                raise RuntimeError("The native manual retry request was refused")
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(document_id)
    report = {
        "request_id": request_id,
        "document_id": document_id,
        "status": row.get("status") if row else "missing",
        "completed_at": time.time(),
        "native_manual_retry": True,
    }
    save_json(settings.run_dir / "retries" / (request_id + ".json"), report)
    if report["status"] != "processed":
        raise RuntimeError(
            "B document retry failed; original status and recovery metadata were retained"
        )
    return report


async def retry(settings: Settings, document_id: str) -> dict:
    settings.require_models()
    manifest = checked_manifest(settings)
    snapshot = read_json(settings.run_dir / "run-snapshot.json")
    if snapshot != run_contract(settings):
        raise ValueError("The frozen B run contract changed")
    adapter = BExtractionAdapter(settings, provider(settings, "extract"))
    rag = make_rag(settings, extraction=adapter)
    initialized = False
    try:
        await initialize_rag(rag, settings)
        initialized = True
        from lightrag.base import DocStatus

        statuses = await rag.doc_status.get_docs_by_statuses(
            [
                DocStatus.PROCESSED,
                DocStatus.PENDING,
                DocStatus.PROCESSING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
                DocStatus.FAILED,
            ],
            strict=True,
        )
        incomplete = {
            key: row
            for key, row in statuses.items()
            if DocStatus(row.status) != DocStatus.PROCESSED
        }
        if set(incomplete) != {document_id}:
            raise ValueError(
                "Retry requires exactly the selected unfinished document; no other document is scheduled implicitly"
            )
        by_name = {
            Path(row.file_path).name: (key, row) for key, row in statuses.items()
        }
        target = None
        for entry in manifest["documents"]:
            match = by_name.get(Path(entry["relative_path"]).name)
            if match is None:
                continue
            key, row = match
            if key == document_id:
                target = entry
            elif DocStatus(row.status) == DocStatus.PROCESSED:
                adapter.restore(key)
        if target is None:
            raise ValueError("Retry target is outside the frozen B originals")
        stage_input(settings, target)
        adapter.begin_document(None, target, document_id=document_id)
        return await run_native_retry(settings, rag, document_id)
    finally:
        if initialized:
            await rag.finalize_storages()
