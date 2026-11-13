"""Synthetic native-storage integration with zero external model API calls."""

import asyncio
import json
import sys
import time
from unittest.mock import patch

import numpy as np

from .common import digest, read_json, save_json, workspace_lock
from .config import PROJECT_ROOT, Settings
from .evidence_adapter import BExtractionAdapter
from .models import initialize_rag, make_rag, query_parameters


async def smoke(base: Settings) -> dict:
    stamp = str(time.time_ns())
    settings = Settings(
        {
            **base.values,
            "B_RUN_ID": "smoke-" + stamp,
            "B_WORKSPACE": "insurance_b_smoke_" + stamp,
            "B_MODEL_API_KEY": "offline-smoke-not-an-api-key",
        },
        base.root,
    )
    settings.activate()
    calls, events = [], []
    answer = "合成验证保险的计划一住院医疗额度为10000，计划二该项为0。[1]"
    expected_final = answer + "\n\n### References\n\n- [1] synthetic.md"

    async def select(prompt, **kwargs):
        data = json.loads(prompt)
        calls.append("extract")
        identity = next(
            unit["id"]
            for unit in data["evidence"]
            if unit["role"] == "product-identity"
        )
        table = next(
            unit["id"]
            for unit in data["evidence"]
            if "| 住院医疗 | 10000 | 0 |" in unit["text"]
        )
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "合成验证保险",
                        "type": "保险产品",
                        "evidence_ids": [identity],
                    },
                    {"name": "计划一", "type": "保障计划", "evidence_ids": [table]},
                    {"name": "住院医疗", "type": "医疗责任", "evidence_ids": [table]},
                ],
                "relationships": [
                    {
                        "source": "合成验证保险",
                        "target": "计划一",
                        "keywords": "包含计划",
                        "evidence_ids": [table],
                    },
                    {
                        "source": "计划一",
                        "target": "住院医疗",
                        "keywords": "包含责任",
                        "evidence_ids": [table],
                    },
                ],
            },
            ensure_ascii=False,
        )

    def fake_provider(settings, role):
        async def complete(prompt, **kwargs):
            if role == "extract":
                return await select(prompt, **kwargs)
            if role == "keyword":
                calls.append("keyword")
                return json.dumps(
                    {
                        "high_level_keywords": ["住院医疗"],
                        "low_level_keywords": ["合成验证保险", "计划一"],
                    },
                    ensure_ascii=False,
                )
            if kwargs.get("stream"):
                calls.append("draft")

                async def stream():
                    yield answer

                return stream()
            calls.append("audit")
            return json.dumps(
                {
                    "supported": True,
                    "scope_consistent": True,
                    "formula_consistent": True,
                    "issues": [],
                    "corrected_answer": None,
                }
            )

        return complete

    async def embeddings(texts, **kwargs):
        return np.ones((len(texts), 1024), dtype=np.float32) / 32

    async def rerank(query, documents, top_n=None, **kwargs):
        return [
            {"index": index, "relevance_score": 1.0}
            for index in range(min(top_n or len(documents), len(documents)))
        ]

    import httpx
    from fastapi import FastAPI
    from lightrag.base import DocStatus
    from lightrag.llm.openai import openai_embed
    from lightrag.parser.routing import resolve_chunk_options

    from .server import native_routes

    content = "# 合成验证保险\n\n| 保障责任 | 计划一 | 计划二 |\n| --- | --- | --- |\n| 住院医疗 | 10000 | 0 |\n\n仅为程序合成验证，不是真实产品。\n"
    source = settings.input_dir / "synthetic.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content)
    entry = {
        "product": "合成验证保险",
        "relative_path": "synthetic.md",
        "sha256": digest(source.read_bytes()),
    }
    checks = {}
    with (
        workspace_lock(settings.root, settings.workspace, exclusive=True),
        patch("applications.insurance_b.models.provider", fake_provider),
        patch.object(openai_embed, "func", embeddings),
        patch("lightrag.rerank.ali_rerank", rerank),
    ):
        adapter = BExtractionAdapter(settings, select)
        rag = make_rag(settings, extraction=adapter, query_record=events.append)
        initialized = False
        try:
            await initialize_rag(rag, settings)
            initialized = True
            track = await rag.apipeline_enqueue_documents(
                [""],
                file_paths=[str(source)],
                docs_format="pending_parse",
                parse_engine="native",
                process_options="P",
                chunk_options=resolve_chunk_options(
                    rag.addon_params, process_options="P"
                ),
            )
            adapter.begin_document(track, entry)
            await rag.apipeline_process_enqueue_documents()
            statuses = await rag.doc_status.get_docs_by_track_id(track)
            checks["one_native_document"] = len(statuses) == 1
            if len(statuses) != 1:
                raise ValueError("Synthetic native document missing")
            doc_id, status = next(iter(statuses.items()))
            checks["processed"] = DocStatus(status.status) == DocStatus.PROCESSED
            checks["actual_native_p"] = (status.metadata or {}).get(
                "chunk_method"
            ) == "paragraph_semantic"
            checks["chunks_written"] = bool(status.chunks_list)
            checks["anchors_present"] = (
                await rag.full_entities.get_by_id(doc_id) is not None
                and await rag.full_relations.get_by_id(doc_id) is not None
            )
            checks["graph_entities"] = (
                len(await rag.chunk_entity_relation_graph.get_all_nodes()) == 3
            )
            relation = await rag.chunk_entity_relation_graph.get_edge(
                "合成验证保险｜计划一", "合成验证保险｜住院医疗"
            )
            checks["verified_relation_written"] = relation is not None
            from .verify import verify_vectors

            vector_counts, vector_errors = await verify_vectors(
                settings,
                rag,
                set(status.chunks_list),
                {
                    node["id"]
                    for node in await rag.chunk_entity_relation_graph.get_all_nodes()
                },
                {
                    tuple(sorted([edge["source"], edge["target"]]))
                    for edge in await rag.chunk_entity_relation_graph.get_all_edges()
                },
            )
            checks["native_vector_ids_dimensions_and_norms"] = (
                not vector_errors
                and vector_counts
                == {
                    "chunks": len(status.chunks_list),
                    "entities": 3,
                    "relationships": 2,
                }
            )
            result = await rag.aquery_llm(
                "合成验证保险计划一的住院医疗额度是多少？",
                param=query_parameters(settings, stream=True),
            )
            response = result.get("llm_response") or {}
            delivered = ""
            if response.get("is_streaming"):
                async for token in response["response_iterator"]:
                    delivered += token
            else:
                delivered = response.get("content") or ""
            checks["native_query_delivered"] = (
                result.get("status") == "success" and delivered == expected_final
            )
            checks["draft_and_audit_executed"] = "draft" in calls and "audit" in calls
            checks["same_request_sources_captured"] = any(
                event.get("source_documents") for event in events
            )
            app = FastAPI()
            app.include_router(native_routes(rag, top_k=30))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://b-synthetic"
            ) as client:
                response = await client.post(
                    "/query",
                    json={
                        "query": "合成验证保险计划一的住院医疗额度是多少？",
                        "mode": "mix",
                        "top_k": 30,
                        "chunk_top_k": 12,
                        "max_entity_tokens": 2000,
                        "max_relation_tokens": 2500,
                        "max_total_tokens": 18000,
                        "include_references": True,
                        "include_chunk_content": True,
                    },
                )
                checks["native_http_query_delivered"] = (
                    response.status_code == 200
                    and response.json().get("response") == expected_final
                )
            source.write_text(content)
            await rag.apipeline_enqueue_documents(
                [""],
                file_paths=[str(source)],
                docs_format="pending_parse",
                parse_engine="native",
                process_options="P",
            )
            await rag.apipeline_process_enqueue_documents()
            processed = await rag.doc_status.get_docs_by_statuses(
                [DocStatus.PROCESSED], strict=True
            )
            checks["duplicate_did_not_grow_corpus"] = len(processed) == 1
            checks["duplicate_did_not_call_extractor"] = calls.count("extract") == 1
        finally:
            if initialized:
                await rag.finalize_storages()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "applications.insurance_b.smoke_readback",
            settings.workspace,
            settings.run_id,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        process_output, _ = await asyncio.wait_for(process.communicate(), timeout=45)
        (settings.run_dir / "fresh-process-readback.log").write_bytes(process_output)
        if process.returncode:
            raise ValueError(
                "Synthetic readback process failed; inspect its dedicated log"
            )
        fresh = read_json(settings.run_dir / "fresh-process-readback.json")
        checks["fresh_process_readback"] = fresh == {
            "documents": 1,
            "entities": 3,
            "chunks": 1,
            "all_chunks_present": True,
        }
        reopened = make_rag(settings, audited_query=False)
        initialized = False
        try:
            await initialize_rag(reopened, settings)
            initialized = True
            checks["reopen_readback"] = (
                len(
                    await reopened.doc_status.get_docs_by_statuses(
                        [DocStatus.PROCESSED], strict=True
                    )
                )
                == 1
                and len(await reopened.chunk_entity_relation_graph.get_all_nodes()) == 3
            )
            duplicate_rows = await reopened.doc_status.get_docs_by_statuses(
                [DocStatus.FAILED], strict=True
            )
            for duplicate_id, duplicate_status in duplicate_rows.items():
                if not (duplicate_status.metadata or {}).get("is_duplicate"):
                    raise ValueError("An unexpected failed synthetic document exists")
                await reopened.adelete_by_doc_id(duplicate_id)
            checks[
                "synthetic_duplicate_receipt_deleted"
            ] = not await reopened.doc_status.get_docs_by_statuses(
                [DocStatus.FAILED], strict=True
            )
            await reopened.adelete_by_doc_id(doc_id)
            checks["synthetic_delete_completed"] = (
                await reopened.doc_status.get_by_id(doc_id) is None
                and not await reopened.chunk_entity_relation_graph.get_all_nodes()
            )
        finally:
            if initialized:
                await reopened.finalize_storages()
        # Force a real pre-merge failure, then exercise the same manual intent as CLI retry.
        from .retry import run_native_retry

        force_failure = True
        failed_calls = []

        async def retry_select(prompt, **kwargs):
            if force_failure:
                failed_calls.append("forced_failure")
                raise ValueError("Intentional synthetic extraction failure")
            return await select(prompt, **kwargs)

        retry_source = settings.input_dir / "retry.md"
        retry_source.write_text(content + "\n此文档仅验证失败后显式重试。\n")
        retry_entry = {
            **entry,
            "relative_path": "retry.md",
            "sha256": digest(retry_source.read_bytes()),
        }
        retry_adapter = BExtractionAdapter(settings, retry_select)
        retry_rag = make_rag(settings, extraction=retry_adapter, audited_query=False)
        initialized = False
        try:
            await initialize_rag(retry_rag, settings)
            initialized = True
            retry_track = await retry_rag.apipeline_enqueue_documents(
                [""],
                file_paths=[str(retry_source)],
                docs_format="pending_parse",
                parse_engine="native",
                process_options="P",
                chunk_options=resolve_chunk_options(
                    retry_rag.addon_params, process_options="P"
                ),
            )
            retry_adapter.begin_document(retry_track, retry_entry)
            await retry_rag.apipeline_process_enqueue_documents()
            retry_statuses = await retry_rag.doc_status.get_docs_by_track_id(
                retry_track
            )
            retry_doc_id, retry_status = next(iter(retry_statuses.items()))
            checks["forced_failure_is_retained"] = DocStatus(
                retry_status.status
            ) == DocStatus.FAILED and bool(failed_calls)
            failure_count = len(failed_calls)
            await retry_rag.apipeline_process_enqueue_documents()
            checks["failed_document_requires_manual_intent"] = (
                len(failed_calls) == failure_count
                and (await retry_rag.doc_status.get_by_id(retry_doc_id))["status"]
                == "failed"
            )
            force_failure = False
            retry_adapter.begin_document(None, retry_entry, document_id=retry_doc_id)
            retry_report = await run_native_retry(settings, retry_rag, retry_doc_id)
            checks["explicit_native_retry_succeeded"] = (
                retry_report["status"] == "processed"
            )
            checks["retry_recovery_anchors_present"] = (
                await retry_rag.full_entities.get_by_id(retry_doc_id) is not None
                and await retry_rag.full_relations.get_by_id(retry_doc_id) is not None
            )
            await retry_rag.adelete_by_doc_id(retry_doc_id)
            checks["retried_synthetic_document_deleted"] = (
                await retry_rag.doc_status.get_by_id(retry_doc_id) is None
                and not await retry_rag.chunk_entity_relation_graph.get_all_nodes()
            )
        finally:
            if initialized:
                await retry_rag.finalize_storages()
    report = {
        "passed": all(checks.values()),
        "mode": "synthetic integration; model providers and embeddings replaced",
        "external_model_api_calls": 0,
        "workspace": settings.workspace,
        "checks": checks,
        "stub_call_counts": {key: calls.count(key) for key in set(calls)},
        "same_request_events": events,
        "business_acceptance": "not evaluated by synthetic smoke",
    }
    save_json(settings.run_dir / "smoke-report.json", report)
    if not report["passed"]:
        raise ValueError(
            f"B synthetic integration failed; see {settings.run_dir / 'smoke-report.json'}"
        )
    return {
        "passed": True,
        "checks": len(checks),
        "external_model_api_calls": 0,
        "report": str(settings.run_dir / "smoke-report.json"),
    }
