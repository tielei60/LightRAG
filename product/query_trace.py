"""Capture the complete native model input and answer from one SDK query.

Run only after the import writer exits and before starting the HTTP server.
The diagnostic disables native query caches so the query wrapper is actually
called; it preserves stored caches and uses the batch's fixed query settings.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from product.verify_batch import (
    CASES,
    ROOT,
    QueryCase,
    fixed_query_settings,
    load_json,
    public_rerank_settings,
    rerank_enabled,
    save_json,
    summarize_timings,
)

logger = logging.getLogger(__name__)


def require_closed_writer(batch: Path, config: dict[str, Any]) -> None:
    """Refuse a running service or an incomplete import handoff."""
    pid_path = batch / "server.pid"
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Stop the recorded HTTP server before SDK tracing")
    with socket.socket() as connection:
        connection.settimeout(0.3)
        if connection.connect_ex((config["HOST"], int(config["PORT"]))) == 0:
            raise RuntimeError(
                "The configured HTTP port is active; stop it before SDK tracing"
            )
    if not (batch / "final-code-readback.json").is_file():
        raise RuntimeError("The SDK import has not finished its closed-writer readback")
    receipts = load_json(batch / "final-document-status.json")
    manifest = load_json(batch / "manifest.json")
    if len(receipts) != len(manifest["documents"]) or any(
        item.get("state") != "processed" for item in receipts
    ):
        raise RuntimeError("The complete batch must be processed before query tracing")


class QueryInputCapture:
    """Record the exact arguments handed to the native query model function."""

    def __init__(self, complete, model: str, options: dict[str, Any]):
        self.complete = complete
        self.model = model
        self.options = options
        self.calls: list[dict[str, Any]] = []
        self.native_inputs: list[dict[str, Any]] = []
        self.grounding_events: list[dict[str, Any]] = []
        self.grounding_contract: str | None = None
        self.path: Path | None = None

    def begin(self, path: Path) -> None:
        self.path = path
        self.calls = []
        self.native_inputs = []
        self.grounding_events = []

    def save(self) -> None:
        if self.path is None:
            raise RuntimeError("Query capture destination has not been selected")
        save_json(
            self.path,
            {
                "calls": self.calls,
                "native_inputs": self.native_inputs,
                "grounding_events": self.grounding_events,
                "grounding_contract": self.grounding_contract,
                "answer_contract": self.grounding_contract,
                "answer_events": self.grounding_events,
                "provider_output_kind": "actual consumed provider text, including stream chunks",
            },
        )

    def record_grounding(self, event: dict[str, Any]) -> None:
        self.grounding_events.append(event)
        self.save()

    async def __call__(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        if self.path is None:
            raise RuntimeError("Query capture destination has not been selected")
        started = time.monotonic()
        call = {
            "model": self.model,
            "provider_options": self.options,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "history_messages": history_messages or [],
            "stream": bool(kwargs.get("stream")),
            "state": "sending",
            "response_text": "",
            "response_chunks": [],
        }
        self.calls.append(call)
        self.save()
        try:
            response = await self.complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
        except asyncio.CancelledError:
            call.update(state="cancelled", completed_seconds=time.monotonic() - started)
            self.save()
            raise
        except Exception as exc:
            call.update(state="failed", error_type=type(exc).__name__)
            self.save()
            raise
        call["provider_response_received_seconds"] = time.monotonic() - started
        if hasattr(response, "__aiter__"):
            call["state"] = "streaming"
            self.save()

            async def captured_stream():
                try:
                    async for chunk in response:
                        if not isinstance(chunk, str):
                            raise TypeError("Query provider stream must yield text")
                        elapsed = time.monotonic() - started
                        call["response_chunks"].append(
                            {
                                "index": len(call["response_chunks"]),
                                "elapsed_seconds": elapsed,
                                "text": chunk,
                            }
                        )
                        call["response_text"] += chunk
                        if chunk and "first_response_seconds" not in call:
                            call["first_response_seconds"] = elapsed
                        self.save()
                        yield chunk
                    call.update(
                        state="completed", completed_seconds=time.monotonic() - started
                    )
                    self.save()
                except asyncio.CancelledError:
                    call.update(
                        state="cancelled", completed_seconds=time.monotonic() - started
                    )
                    self.save()
                    raise
                except GeneratorExit:
                    call.update(
                        state="closed", completed_seconds=time.monotonic() - started
                    )
                    self.save()
                    raise
                except Exception as exc:
                    call.update(
                        state="failed",
                        error_type=type(exc).__name__,
                        completed_seconds=time.monotonic() - started,
                    )
                    self.save()
                    raise
                finally:
                    close = getattr(response, "aclose", None)
                    if close is not None:
                        await close()

            return captured_stream()
        if not isinstance(response, str):
            call.update(state="failed", error_type="TypeError")
            self.save()
            raise TypeError(
                "Query provider response must be text or an async text stream"
            )
        call.update(
            state="completed",
            response_text=response,
            completed_seconds=time.monotonic() - started,
        )
        self.save()
        return response


def traced_query_role(capture: QueryInputCapture, config: dict[str, Any]):
    """Capture actual generated provider text and its native source envelope."""
    if str(config.get("PRODUCT_GROUNDED_QUERY") or "false").lower() != "true":
        return capture
    from lightrag.prompt import load_user_prompt_prefix_source
    from product.answer_query import ANSWER_CONTRACT_VERSION, AnswerQuery
    from product.grounded_query import provider_limits

    capture.grounding_contract = ANSWER_CONTRACT_VERSION
    adapter = AnswerQuery(
        capture,
        record=capture.record_grounding,
        answer_instructions=load_user_prompt_prefix_source(),
        **provider_limits(config),
    )

    async def query(prompt, system_prompt=None, history_messages=None, **kwargs):
        capture.native_inputs.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "history_messages": history_messages or [],
            }
        )
        capture.save()
        return await adapter(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return query


def build_native_reranker(config: dict[str, Any]):
    """Mirror the server's public provider hooks and call parameters."""
    if not rerank_enabled(config):
        return None
    from lightrag.rerank import (
        DEFAULT_RERANK_MAX_TOKENS_PER_DOC,
        ali_rerank,
        cohere_rerank,
        jina_rerank,
    )

    binding = str(config["RERANK_BINDING"]).strip().lower()
    functions = {"aliyun": ali_rerank, "cohere": cohere_rerank, "jina": jina_rerank}
    if binding not in functions:
        raise ValueError(f"Unsupported rerank binding: {binding}")
    provider = functions[binding]
    signature = inspect.signature(provider)
    model = config.get("RERANK_MODEL") or signature.parameters["model"].default
    host = config.get("RERANK_BINDING_HOST") or signature.parameters["base_url"].default
    chunking = str(config.get("RERANK_ENABLE_CHUNKING") or "false").lower() == "true"
    max_tokens = DEFAULT_RERANK_MAX_TOKENS_PER_DOC
    if binding == "cohere":
        max_tokens = int(config.get("RERANK_MAX_TOKENS_PER_DOC") or max_tokens)
        if max_tokens < 1:
            raise ValueError("RERANK_MAX_TOKENS_PER_DOC must be >= 1")

    async def rerank(
        query: str,
        documents: list,
        top_n: int | None = None,
        extra_body: dict | None = None,
    ):
        kwargs = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "api_key": config.get("RERANK_BINDING_API_KEY"),
            "model": model,
            "base_url": host,
        }
        if binding == "cohere":
            kwargs.update(enable_chunking=chunking, max_tokens_per_doc=max_tokens)
        return await provider(**kwargs, extra_body=extra_body)

    return rerank


def build_native_rag(config: dict[str, Any]):
    """Use public SDK role/embedding hooks without altering retrieval code."""
    os.environ.update(
        {key: value for key, value in config.items() if value is not None}
    )
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.llm_roles import RoleLLMConfig
    from lightrag.prompt import load_user_prompt_prefix_source
    from lightrag.utils import EmbeddingFunc

    def role_function(role):
        async def complete(prompt, system_prompt=None, **kwargs):
            return await openai_complete_if_cache(
                model=config[f"{role.upper()}_LLM_MODEL"],
                prompt=prompt,
                system_prompt=system_prompt,
                base_url=config["LLM_BINDING_HOST"],
                api_key=config["LLM_BINDING_API_KEY"],
                temperature=0,
                extra_body=json.loads(config[f"{role.upper()}_OPENAI_LLM_EXTRA_BODY"]),
                timeout=int(config["LLM_TIMEOUT"]),
                **kwargs,
            )

        return complete

    async def refuse_extract(*args, **kwargs):
        raise RuntimeError("This query diagnostic must not perform extraction")

    query = QueryInputCapture(
        role_function("query"),
        config["QUERY_LLM_MODEL"],
        json.loads(config["QUERY_OPENAI_LLM_EXTRA_BODY"]),
    )

    async def embed(texts, **kwargs):
        return await openai_embed.func(
            texts,
            model=config["EMBEDDING_MODEL"],
            base_url=config["EMBEDDING_BINDING_HOST"],
            api_key=config["EMBEDDING_BINDING_API_KEY"],
            **kwargs,
        )

    functions = {
        "extract": refuse_extract,
        "keyword": role_function("keyword"),
        "query": traced_query_role(query, config),
    }
    roles = {
        role: RoleLLMConfig(
            func=func,
            max_async=int(config["MAX_ASYNC_LLM"]),
            timeout=int(config["LLM_TIMEOUT"]),
            metadata={
                "binding": "openai",
                "model": config[f"{role.upper()}_LLM_MODEL"],
            },
        )
        for role, func in functions.items()
    }
    rag = LightRAG(
        working_dir=config["WORKING_DIR"],
        workspace=config["WORKSPACE"],
        llm_model_func=refuse_extract,
        llm_model_name=config["EXTRACT_LLM_MODEL"],
        role_llm_configs=roles,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(config["EMBEDDING_DIM"]),
            max_token_size=int(config["EMBEDDING_TOKEN_LIMIT"]),
            send_dimensions=True,
            model_name=config["EMBEDDING_MODEL"],
            func=embed,
        ),
        user_prompt_prefix=load_user_prompt_prefix_source(),
        addon_params={
            "language": config["SUMMARY_LANGUAGE"],
            "entity_type_prompt_file": config["ENTITY_TYPE_PROMPT_FILE"],
        },
        enable_llm_cache=False,
        rerank_model_func=build_native_reranker(config),
    )
    return rag, query


async def trace_queries(
    batch: Path, config: dict[str, Any], cases: tuple[QueryCase, ...], repeat: int
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("Repeat count must be positive")
    require_closed_writer(batch, config)
    rag, capture = build_native_rag(config)
    from lightrag import QueryParam

    run_dir = batch / "query-traces" / str(time.time_ns())
    settings = fixed_query_settings(config)
    rows = []
    report = {
        "workspace": config["WORKSPACE"],
        "settings": {**settings, "stream": True},
        "models": {
            key: config.get(key)
            for key in [
                "QUERY_LLM_MODEL",
                "KEYWORD_LLM_MODEL",
                "EMBEDDING_MODEL",
                "EMBEDDING_DIM",
            ]
        },
        "query_generation_options": json.loads(config["QUERY_OPENAI_LLM_EXTRA_BODY"]),
        "rerank": public_rerank_settings(config),
        "execution": "native SDK aquery_llm; query role wrapper captures actual model input for this answer",
        "grounded_query": bool(capture.grounding_contract),
        "grounding_contract": capture.grounding_contract,
        "answer_contract": capture.grounding_contract,
        "answer_mode": "LLM-generated natural answer from the same native retrieved context; no program-rendered selection output",
        "grounding_cache": "No generated-answer cache; every trace records actual consumed provider output.",
        "cache_condition": "Native keyword and answer caches disabled for tracing; no stored cache deleted. Repeat timing measures uncached provider requests, not hot answer-cache latency.",
        "run_directory": str(run_dir),
        "business_acceptance": "pending source-based review",
        "results": rows,
    }
    try:
        await rag.initialize_storages()
        for number in range(repeat):
            for case in cases:
                capture.begin(run_dir / f"{case.id}-{number + 1}-input.json")
                row = {
                    "case_id": case.id,
                    "question": case.question,
                    "review_focus": case.review_focus,
                    "repetition": number + 1,
                    "cache_condition": "first_request"
                    if number == 0
                    else "repeated_request",
                    "business_review": "pending",
                }
                started = time.perf_counter()
                first_output = None
                answer = []
                try:
                    response = await rag.aquery_llm(
                        case.question, param=QueryParam(**settings, stream=True)
                    )
                    llm = response.get("llm_response", {})
                    if llm.get("is_streaming"):
                        async for part in llm["response_iterator"]:
                            if part:
                                if first_output is None:
                                    first_output = time.perf_counter() - started
                                answer.append(part)
                    elif llm.get("content"):
                        first_output = time.perf_counter() - started
                        answer.append(llm["content"])
                    row.update(
                        answer="".join(answer),
                        retrieval_data=response.get("data"),
                        model_input_calls=capture.calls,
                        native_query_inputs=capture.native_inputs,
                        grounding_events=capture.grounding_events,
                        first_output_seconds=round(first_output, 3)
                        if first_output is not None
                        else None,
                        total_seconds=round(time.perf_counter() - started, 3),
                        full_input_captured=bool(capture.calls),
                        provider_output_captured=bool(capture.calls)
                        and all(call["state"] == "completed" for call in capture.calls),
                        llm_generated=llm.get("llm_generated", True),
                    )
                    if not capture.calls:
                        row["capture_limitation"] = (
                            "Query model was not called; this result has no generated-answer model-input trace"
                        )
                except Exception as exc:
                    logger.exception("Native diagnostic query failed for %s", case.id)
                    row.update(
                        error_type=type(exc).__name__,
                        error=str(exc),
                        model_input_calls=capture.calls,
                        native_query_inputs=capture.native_inputs,
                        grounding_events=capture.grounding_events,
                        total_seconds=round(time.perf_counter() - started, 3),
                        first_output_seconds=round(first_output, 3)
                        if first_output is not None
                        else None,
                    )
                rows.append(row)
                save_json(run_dir / f"{case.id}-{number + 1}.json", row)
                report["timings"] = summarize_timings(rows)
                save_json(run_dir / "report.json", report)
                print(
                    json.dumps(
                        {
                            "case": case.id,
                            "repeat": number + 1,
                            "full_input_captured": row.get("full_input_captured"),
                            "total_seconds": row["total_seconds"],
                            "error": row.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        await rag.finalize_storages()
    save_json(batch / "query-trace-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cases", nargs="+", choices=[case.id for case in CASES])
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.batch):
        raise ValueError("Invalid batch name")
    batch = ROOT / "output" / args.batch
    config = dict(dotenv_values(batch / "runtime.env", interpolate=False))
    if load_json(batch / "manifest.json")["workspace"] != config["WORKSPACE"]:
        raise ValueError("Manifest and runtime workspace differ")
    cases = tuple(case for case in CASES if not args.cases or case.id in args.cases)
    report = asyncio.run(trace_queries(batch, config, cases, args.repeat))
    if any(
        row.get("error") or not row.get("full_input_captured")
        for row in report["results"]
    ):
        raise SystemExit(
            "One or more queries have no complete generation trace; inspect the saved report"
        )


if __name__ == "__main__":
    main()
