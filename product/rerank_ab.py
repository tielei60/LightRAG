"""One-shot, same-candidate rerank trial. No network except explicit reviewed run.

All experiment data belongs in a private gitignored directory. This module never
uses the provider SDK, the retrying rerank adapter, or a running LightRAG server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
import contextvars
import copy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import socket
import time
from typing import Any

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
MODELS = {
    "keyword": "qwen3.8-flash",
    "embedding": "qwen3.7-text-embedding-flash",
    "rerank": "qwen3.7-text-rerank",
    "answer": "qwen3.7-plus",
}
RETURN_MODELS = {
    "keyword": [MODELS["keyword"]],
    "embedding": [MODELS["embedding"]],
    "rerank": [MODELS["rerank"]],
    "answer": [MODELS["answer"], "qwen3.7-plus-2026-05-26"],
}
QUERY = {
    "mode": "mix",
    "top_k": 30,
    "chunk_top_k": 12,
    "max_total_tokens": 18000,
    "max_entity_tokens": 4000,
    "max_relation_tokens": 5000,
    "conversation_history": [],
    "stream": False,
    "only_need_prompt": True,
    "enable_rerank": False,
}
CONFIG = {
    "LLM_BINDING_HOST": BASE,
    "EMBEDDING_BINDING_HOST": BASE,
    "LLM_MODEL": MODELS["keyword"],
    "KEYWORD_LLM_MODEL": MODELS["keyword"],
    "EMBEDDING_MODEL": MODELS["embedding"],
    "RERANK_BINDING": "aliyun",
    "RERANK_MODEL": MODELS["rerank"],
    "RERANK_BINDING_HOST": RERANK_URL,
}
STORAGE_FILES = (
    "graph_chunk_entity_relation.graphml",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "kv_store_text_chunks.json",
    "kv_store_full_docs.json",
    "kv_store_full_entities.json",
    "kv_store_full_relations.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
    "kv_store_doc_status.json",
)
CONTEXT_PARAMETERS = (
    "entities_context",
    "relations_context",
    "merged_chunks",
    "query",
    "query_param",
    "global_config",
    "chunk_tracking",
    "entity_id_to_original",
    "relation_id_to_original",
    "progress_callback",
    "system_prompt",
)
PUBLIC_GLOBAL_FIELDS = (
    "max_total_tokens",
    "min_rerank_score",
    "user_prompt_prefix",
    "enable_content_headings",
    "kg_chunk_pick_method",
    "related_chunk_number",
    "cosine_better_than_threshold",
    "enable_llm_cache",
    "tiktoken_model_name",
)
PUBLIC_FILES = (
    "product/rerank_ab.py",
    "product/test_rerank_ab.py",
    "product/rerank_ab.md",
)
PREFIX = "product/prompt_profiles/user_prompt/products_scoped.md"
NETWORK_ALLOWED = contextvars.ContextVar("rerank_ab_provider_network", default=False)


class Stop(Exception):
    """Messages must be hard-coded categories, never exception or provider text."""


def failure_category(error: BaseException) -> str:
    """Only return a fixed category, even if an unexpected exception holds a secret."""
    allowed = {
        "Non200Response",
        "ResponseModelMismatch",
        "NonStopFinish",
        "ExplicitRefusal",
        "UnexpectedReasoning",
        "InvalidChoices",
        "UnexpectedToolCall",
        "EmptyAnswer",
        "MissingUsage",
        "InvalidUsage",
        "MissingRequestId",
        "ProviderError",
        "CredentialEcho",
        "KeywordShape",
        "KeywordIncomplete",
        "EmbeddingCount",
        "EmbeddingIndices",
        "EmbeddingDimension",
        "EmbeddingNonFinite",
        "EmbeddingEncoding",
        "RerankCount",
        "RerankRow",
        "RerankIndex",
        "RerankScore",
        "RerankNotSorted",
        "UnexpectedReturnedDocument",
        "DuplicateJsonKey",
        "NonFiniteJson",
        "AttemptCap",
        "CaseStageReplayOrOrderMismatch",
        "PriorFailureNoFallback",
        "CodeChanged",
        "SourceChanged",
        "StorageChanged",
        "CredentialsChangedDuringRun",
        "IndependentReviewChanged",
    }
    if isinstance(error, Stop):
        category = error.args[0] if len(error.args) == 1 else None
        return (
            category
            if isinstance(category, str) and category in allowed
            else "ProtocolGuardRejected"
        )
    if isinstance(error, httpx.TimeoutException):
        return "Timeout"
    if isinstance(error, httpx.TransportError):
        return "TransportError"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "Interrupted"
    return "InvalidResponseOrLocalFailure"


def require(value: Any, category: str) -> None:
    if not value:
        raise Stop(category)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DuplicateJsonKey")
        result[key] = value
    return result


def parse(data: bytes) -> Any:
    return json.loads(
        data,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(Stop("NonFiniteJson")),
    )


def read(path: Path) -> bytes:
    require(path.is_file() and not path.is_symlink(), "UnsafeInput")
    return path.read_bytes()


def load(path: Path) -> Any:
    return parse(read(path))


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    sync_dir(path.parent)


def write_json(path: Path, value: Any) -> None:
    write_new(path, canonical(value) + b"\n")


def mkdir(path: Path) -> None:
    path.mkdir(mode=0o700)
    sync_dir(path.parent)


def private_tree(path: Path) -> None:
    require(path.is_dir() and not path.is_symlink(), "UnsafePrivateDirectory")
    for item in [path, *path.rglob("*")]:
        require(not item.is_symlink(), "PrivateSymlink")
        require(stat.S_IMODE(item.stat().st_mode) & 0o077 == 0, "PrivatePermissions")


def no_secrets(data: bytes, secrets: list[str]) -> None:
    for secret in secrets:
        require(secret.encode() not in data, "CredentialEcho")
    try:
        decoded = json.dumps(parse(data), ensure_ascii=False)
    except (ValueError, UnicodeError):
        return
    require(not any(secret in decoded for secret in secrets), "CredentialEcho")


def credentials(path: Path) -> tuple[dict, dict]:
    require(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "CredentialPermissions")
    read(path)
    values = dotenv_values(path, interpolate=False)
    public = {name: values.get(name) for name in CONFIG}
    require(public == CONFIG, "ProviderConfigurationChanged")
    keys = {
        "keyword": values.get("LLM_BINDING_API_KEY"),
        "answer": values.get("LLM_BINDING_API_KEY"),
        "embedding": values.get("EMBEDDING_BINDING_API_KEY"),
        "rerank": values.get("RERANK_BINDING_API_KEY"),
    }
    require(
        all(isinstance(key, str) and len(key.strip()) >= 8 for key in keys.values()),
        "CredentialMissing",
    )
    return public, keys


def inventory(directory: Path) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), "MissingStorage")
    names = {p.name for p in directory.iterdir()}
    require(
        set(STORAGE_FILES)
        <= names
        <= set(STORAGE_FILES) | {"kv_store_llm_response_cache.json"},
        "StorageFilesChanged",
    )
    return {name: digest(read(directory / name)) for name in STORAGE_FILES}


def source_fingerprints(paths: list[Path]) -> dict:
    return {str(path.resolve()): digest(read(path)) for path in paths}


def code_fingerprints() -> dict:
    files = [ROOT / name for name in PUBLIC_FILES]
    files += sorted((ROOT / "lightrag").rglob("*.py"))
    files += [ROOT / PREFIX, ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {str(p.relative_to(ROOT)): digest(read(p)) for p in files}


def dependencies() -> dict:
    names = ("httpx", "numpy", "tiktoken", "nano-vectordb", "networkx", "python-dotenv")
    return {name: importlib.metadata.version(name) for name in names}


def schedule(cases: list[dict]) -> list[dict]:
    rows = [
        {"case_id": c["id"], "stage": stage}
        for c in cases
        for stage in ("keyword", "embedding", "rerank")
    ]
    for i, case in enumerate(cases):
        for arm in ("off", "on") if i % 2 == 0 else ("on", "off"):
            rows.append({"case_id": case["id"], "stage": "answer_" + arm})
    require(
        len(rows) == len({(r["case_id"], r["stage"]) for r in rows}) == 15,
        "BudgetShape",
    )
    return rows


def blind_order(cases: list[dict]) -> list[dict]:
    # Predeclared shuffled IDs, not cryptographic blinding. Only the independent
    # scorer receives the artifact without this private assignment table.
    positions = [
        (2, "on", "R14"),
        (0, "off", "Q82"),
        (1, "on", "L39"),
        (2, "off", "M65"),
        (1, "off", "T27"),
        (0, "on", "V53"),
    ]
    return [
        {"case_id": cases[i]["id"], "arm": arm, "blind_id": code}
        for i, arm, code in positions
    ]


def clean_environment() -> None:
    # No ambient provider/profile/storage settings, proxy, or dotenv discovery.
    kept = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "TMPDIR", "TIKTOKEN_CACHE_DIR")
        if k in os.environ
    }
    os.environ.clear()
    os.environ.update(kept)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.disable(logging.CRITICAL)


@contextmanager
def network_guard():
    """Deny hidden downloads/calls, including tokenizer initialization, by default."""
    originals = [
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "getaddrinfo"),
        (socket, "create_connection"),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name in originals]

    def wrapped(original):
        def checked(*args, **kwargs):
            require(NETWORK_ALLOWED.get(), "UnguardedNetworkBlocked")
            return original(*args, **kwargs)

        return checked

    try:
        for obj, name, original in saved:
            setattr(obj, name, wrapped(original))
        yield
    finally:
        for obj, name, original in saved:
            setattr(obj, name, original)


def native_modules():
    sys.path.insert(0, str(ROOT))
    import lightrag
    from lightrag import LightRAG, QueryParam
    from lightrag import operate, utils
    from lightrag.prompt import PROMPTS

    for module in (lightrag, operate, utils):
        require(
            Path(module.__file__).resolve().is_relative_to(ROOT / "lightrag"),
            "ImportedWrongCheckout",
        )
    require(
        Path(__file__).resolve() == ROOT / "product/rerank_ab.py",
        "ImportedWrongHarness",
    )
    require(
        tuple(inspect.signature(operate._build_context_str).parameters)
        == CONTEXT_PARAMETERS,
        "ContextSignatureChanged",
    )
    return LightRAG, QueryParam, operate, utils, PROMPTS


def prepare(
    run: Path, source_root: Path, storage: Path, oracle_dir: Path, env: Path
) -> dict:
    """Offline only. The caller supplies independently frozen question/oracle files."""
    require(not run.exists(), "RunDirectoryAlreadyExists")
    require(run.is_relative_to(ROOT / "data"), "RunMustBePrivateGitignoredData")
    require(
        subprocess.run(["git", "check-ignore", "-q", str(run)], cwd=ROOT).returncode
        == 0,
        "RunNotIgnored",
    )
    public, keys = credentials(env)
    question_path, oracle_path = (
        oracle_dir / "question-plan.json",
        oracle_dir / "evaluation-oracle.json",
    )
    plan, oracle = load(question_path), load(oracle_path)
    cases = [{"id": c["id"], "question": c["question"]} for c in plan["cases"]]
    require(len(cases) == 3 and len({c["id"] for c in cases}) == 3, "CaseCount")
    require(
        all(isinstance(c["question"], str) and c["question"].strip() for c in cases),
        "EmptyQuestion",
    )
    require(oracle.get("do_not_send_to_answer_model") is True, "OracleBoundaryMissing")
    require(
        [{k: c[k] for k in ("id", "question")} for c in oracle["cases"]] == cases,
        "OracleQuestionsMismatch",
    )
    baseline = source_root / "data/answer-effect/jkyh-20261111-v3"
    original_plan, original_oracle = (
        load(baseline / "question-plan.json"),
        load(baseline / "evaluation-oracle.json"),
    )
    for cid in ("benefit_scope", "new_notice_exception"):
        old = next(c for c in original_plan["cases"] if c["id"] == cid)
        require(
            next(c for c in cases if c["id"] == cid)["question"] == old["question"],
            "OldQuestionChanged",
        )
        # Frozen scores and evidence may not be changed by the runner.
        old_o = next(c for c in original_oracle["cases"] if c["id"] == cid)
        new_o = next(c for c in oracle["cases"] if c["id"] == cid)
        require(
            all(new_o.get(k) == v for k, v in old_o.items() if k != "retrieval"),
            "OldOracleChanged",
        )
        historical = new_o.get("historical_retrieval_only", {})
        require(
            {k: v for k, v in historical.items() if k != "not_this_run"}
            == old_o.get("retrieval"),
            "HistoricalOracleChanged",
        )
    approved = load(baseline / "approved-sources.json")
    require(len(approved) == 3, "SourceCount")
    inputs = [
        question_path,
        oracle_path,
        baseline / "approved-sources.json",
        baseline / "question-plan.json",
        baseline / "evaluation-oracle.json",
        baseline / "source-review.json",
        source_root / PREFIX,
    ]
    inputs += sorted(p for p in oracle_dir.iterdir() if p.is_file() and p not in inputs)
    sources = {}
    for item in approved:
        path = (
            source_root / "docs/insurance_docs/京康优护" / Path(item["file_path"]).name
        )
        require(digest(read(path)) == item["source_sha256"], "SourceHashMismatch")
        require(
            digest(item["text"].encode()) == item["extracted_text_sha256"],
            "ReviewedTextMismatch",
        )
        inputs.append(path)
        sources[path.name] = {
            k: item[k]
            for k in ("document_ref", "source_sha256", "extracted_text_sha256")
        }
    require(
        digest(read(ROOT / PREFIX)) == digest(read(source_root / PREFIX)),
        "SystemPolicyChanged",
    )
    hashes = inventory(storage)
    chunks = load(storage / "kv_store_text_chunks.json")
    require(
        set(c["file_path"] for c in chunks.values()) == set(sources),
        "UnapprovedChunkSource",
    )
    for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
        require(
            load(storage / name)["embedding_dim"] == 1024, "EmbeddingDimensionChanged"
        )
    # Materialize only the selected workspace, never .env or quarantined data.
    run.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mkdir(run)
    for target in (run / "source-storage", run / "runtime-storage"):
        mkdir(target)
        for name in STORAGE_FILES:
            write_new(target / name, read(storage / name))
        require(inventory(target) == hashes, "CopyMismatch")
    for name in (
        "question-plan.json",
        "evaluation-oracle.json",
        "source-review.json",
        "freeze.md",
    ):
        write_new(run / name, read(oracle_dir / name))
    _, _, _, utils, prompts = native_modules()
    # With the CLI's network guard active, a missing tokenizer cache is an
    # offline blocker rather than an unbudgeted download during the paid run.
    tokenizer = utils.TiktokenTokenizer("gpt-4o-mini")
    tokenizer_probe = tokenizer.encode(
        "Offline tokenizer verification. 通用保险条款。\n12345"
    )
    manifest = {
        "schema": 1,
        "phase": "frozen_offline",
        "cases": cases,
        "schedule": schedule(cases),
        "blind_assignment": blind_order(cases),
        "blinding": "predeclared shuffled opaque IDs; not cryptographic",
        "identical_prompt_policy": "Generate each arm once even when prompt bytes match; answer differences then reflect generation noise, not evidence changes.",
        "provider_request_limit": 15,
        "query_parameters": QUERY,
        "min_rerank_score": 0.0,
        "models": MODELS,
        "response_model_allowlists": RETURN_MODELS,
        "public_configuration": public,
        "environment_path": str(env),
        "source_storage_path": str(storage),
        "storage_sha256": hashes,
        "cache_policy": "Existing LLM cache is excluded, not read or copied; all LLM and embedding caches disabled.",
        "source_files_sha256": source_fingerprints(inputs),
        "sources": sources,
        "code_sha256": code_fingerprints(),
        "dependencies": dependencies(),
        "tokenizer_probe": tokenizer_probe,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "templates": {
            "rag_response": prompts["rag_response"],
            "kg_query_context": prompts["kg_query_context"],
            "user_prompt_prefix": read(ROOT / PREFIX).decode(),
        },
        "oracle_sha256": digest(read(run / "evaluation-oracle.json")),
        "question_plan_sha256": digest(read(run / "question-plan.json")),
        "oracle_files_sha256": {
            name: digest(read(run / name))
            for name in (
                "question-plan.json",
                "evaluation-oracle.json",
                "source-review.json",
                "freeze.md",
            )
        },
        "limits": {
            "query_tokens": 4000,
            "document_tokens": 30000,
            "total_tokens": 120000,
            "documents": 500,
            "gate": "UTF-8 byte conservative ceiling; tiktoken is an estimate only",
            "sources": [
                "https://help.aliyun.com/zh/model-studio/rerank",
                "https://help.aliyun.com/zh/model-studio/text-rerank-api",
                "https://help.aliyun.com/zh/model-studio/qwen3-7-text-rerank",
            ],
        },
        "limitations": [
            "Three targeted cases, one generation per arm; no stability or generalization claim.",
            "References are compared by chunk ID and source/content hash, never number alone.",
            "The oracle is fingerprinted only, never sent or consulted for evidence selection.",
        ],
    }
    no_secrets(canonical(manifest), list(keys.values()))
    write_json(run / "manifest.json", manifest)
    preflight = {
        "schema": 1,
        "phase": "prepared_offline_not_authorized_to_send",
        "manifest_sha256": digest(read(run / "manifest.json")),
        "code_sha256": manifest["code_sha256"],
        "maximum_requests": 15,
        "review_required": "Independent review must approve this exact preflight SHA in a separate review file.",
    }
    write_json(run / "preflight.json", preflight)
    private_tree(run)
    return {
        "phase": preflight["phase"],
        "preflight_sha256": digest(read(run / "preflight.json")),
        "requests": 15,
    }


def verify(run: Path, reviewed_sha: str) -> tuple[dict, dict]:
    require(
        len(reviewed_sha) == 64
        and digest(read(run / "preflight.json")) == reviewed_sha,
        "PreflightPinMismatch",
    )
    preflight = load(run / "preflight.json")
    require(
        digest(read(run / "manifest.json")) == preflight["manifest_sha256"],
        "ManifestChanged",
    )
    m = load(run / "manifest.json")
    require(
        m["code_sha256"] == code_fingerprints() == preflight["code_sha256"],
        "CodeChanged",
    )
    require(m["dependencies"] == dependencies(), "DependenciesChanged")
    require(
        m["source_files_sha256"]
        == source_fingerprints([Path(p) for p in m["source_files_sha256"]]),
        "SourceChanged",
    )
    for path in (
        Path(m["source_storage_path"]),
        run / "source-storage",
        run / "runtime-storage",
    ):
        require(inventory(path) == m["storage_sha256"], "StorageChanged")
    require(
        digest(read(run / "evaluation-oracle.json")) == m["oracle_sha256"],
        "OracleChanged",
    )
    require(
        digest(read(run / "question-plan.json")) == m["question_plan_sha256"],
        "QuestionsChanged",
    )
    require(
        {name: digest(read(run / name)) for name in m["oracle_files_sha256"]}
        == m["oracle_files_sha256"],
        "FrozenOracleFilesChanged",
    )
    require(m["schedule"] == schedule(m["cases"]), "ScheduleChanged")
    require(m["blind_assignment"] == blind_order(m["cases"]), "BlindAssignmentChanged")
    public, keys = credentials(Path(m["environment_path"]))
    require(public == m["public_configuration"], "ConfigurationChanged")
    private_tree(run)
    return m, keys


@contextmanager
def run_lock(run: Path):
    fd = os.open(run / ".run.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop("RunLocked") from None
        yield
    finally:
        os.close(fd)


def check_reasoning(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("reasoning_tokens", "thinking_tokens"):
                require(
                    type(child) in (int, float) and child == 0, "UnexpectedReasoning"
                )
            if key in ("reasoning_content", "reasoning", "thinking"):
                require(not child, "UnexpectedReasoning")
            check_reasoning(child)
    elif isinstance(value, list):
        for child in value:
            check_reasoning(child)


def response_data(raw: bytes, stage: str, payload: dict, secrets: list[str]) -> dict:
    no_secrets(raw, secrets)
    data = parse(raw)
    require(
        isinstance(data, dict) and not data.get("error") and not data.get("code"),
        "ProviderError",
    )
    kind = "answer" if stage.startswith("answer_") else stage
    model = data.get("model")
    require(
        (kind == "rerank" and model is None) or model in RETURN_MODELS[kind],
        "ResponseModelMismatch",
    )
    require(isinstance(data.get("usage"), dict), "MissingUsage")
    require(
        isinstance(data.get("id") or data.get("request_id"), str), "MissingRequestId"
    )
    check_reasoning(data)
    if kind in ("keyword", "answer"):
        choices = data.get("choices")
        require(isinstance(choices, list) and len(choices) == 1, "InvalidChoices")
        require(choices[0].get("finish_reason") == "stop", "NonStopFinish")
        message = choices[0].get("message", {})
        require(not message.get("refusal"), "ExplicitRefusal")
        require(
            not message.get("tool_calls") and not message.get("function_call"),
            "UnexpectedToolCall",
        )
        content = message.get("content")
        require(isinstance(content, str) and bool(content.strip()), "EmptyAnswer")
        require(
            "<think>" not in content.lower() and "</think>" not in content.lower(),
            "UnexpectedReasoning",
        )
        require(
            all(
                type(data["usage"].get(k)) is int and data["usage"][k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ),
            "InvalidUsage",
        )
        if kind == "keyword":
            kw = parse(content.encode())
            require(
                isinstance(kw, dict)
                and set(kw) == {"high_level_keywords", "low_level_keywords"},
                "KeywordShape",
            )
            require(
                all(
                    isinstance(v, list)
                    and v
                    and all(isinstance(x, str) and x.strip() for x in v)
                    for v in kw.values()
                ),
                "KeywordIncomplete",
            )
    elif kind == "embedding":
        require(
            all(
                type(data["usage"].get(k)) is int and data["usage"][k] >= 0
                for k in ("prompt_tokens", "total_tokens")
            ),
            "InvalidUsage",
        )
        rows = data.get("data")
        require(
            isinstance(rows, list) and len(rows) == len(payload["input"]),
            "EmbeddingCount",
        )
        require(
            [r.get("index") for r in rows] == list(range(len(rows))), "EmbeddingIndices"
        )
        for row in rows:
            decode_vector(row.get("embedding"))
    else:
        require(
            type(data["usage"].get("total_tokens")) is int
            and data["usage"]["total_tokens"] >= 0,
            "InvalidUsage",
        )
        validate_ranking(
            data, len(payload["input"]["documents"]), payload["parameters"]["top_n"]
        )
    return data


def decode_vector(value):
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
            require(len(raw) == 4096, "EmbeddingDimension")
            value = list(struct.unpack("<1024f", raw))
        except (ValueError, struct.error):
            raise Stop("EmbeddingEncoding") from None
    require(isinstance(value, list) and len(value) == 1024, "EmbeddingDimension")
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in value),
        "EmbeddingNonFinite",
    )
    return value


def validate_ranking(data: dict, count: int, top_n: int) -> list[dict]:
    output = data.get("output", {})
    rows = output.get("results") if isinstance(output, dict) else None
    require(isinstance(rows, list) and len(rows) == min(top_n, count), "RerankCount")
    indices, previous = set(), math.inf
    for row in rows:
        require(isinstance(row, dict), "RerankRow")
        index, score = row.get("index"), row.get("relevance_score")
        require(
            type(index) is int and 0 <= index < count and index not in indices,
            "RerankIndex",
        )
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "RerankScore",
        )
        require(score <= previous, "RerankNotSorted")
        require("document" not in row, "UnexpectedReturnedDocument")
        previous = score
        indices.add(index)
    return rows


def rerank_size(query: str, documents: list[str]) -> dict:
    q, sizes = len(query.encode()), [len(d.encode()) for d in documents]
    require(
        0 < len(documents) <= 500 and q <= 4000 and all(0 < x <= 30000 for x in sizes),
        "RerankLengthLimit",
    )
    total = q * len(documents) + sum(sizes)
    require(total <= 120000, "RerankTotalLengthLimit")
    return {
        "query_utf8_bytes": q,
        "document_utf8_bytes": sizes,
        "query_repeated_total_utf8_bytes": total,
        "counting_method": "conservative UTF-8 byte ceiling",
    }


class Provider:
    """One durable attempt for each exact case/stage; failure poisons the run."""

    def __init__(self, run: Path, manifest: dict, keys: dict, client, verifier):
        self.run, self.manifest, self.keys, self.client = run, manifest, keys, client
        self.verifier = verifier
        self.attempted, self.completed, self.failed = 0, [], False
        self.case_id = None
        self.first_failure = None
        self.failure_record_saved = None

    def post(self, stage: str, payload: dict) -> dict:
        require(not self.failed, "PriorFailureNoFallback")
        response, folder, started = None, None, None
        send_reserved = False
        try:
            require(self.attempted < 15, "AttemptCap")
            row = {"case_id": self.case_id, "stage": stage}
            require(
                row == self.manifest["schedule"][self.attempted],
                "CaseStageReplayOrOrderMismatch",
            )
            self.verifier()
            raw = canonical(payload)
            no_secrets(raw, list(self.keys.values()))
            kind = "answer" if stage.startswith("answer_") else stage
            require(payload.get("model") == MODELS[kind], "RequestModelMismatch")
            url = (
                RERANK_URL
                if kind == "rerank"
                else BASE
                + ("/embeddings" if kind == "embedding" else "/chat/completions")
            )
            folder = self.run / "attempts" / f"{self.attempted + 1:04d}"
            mkdir(folder)
            write_new(folder / "request.json", raw)
            started = time.time_ns()
            write_json(
                folder / "attempt.json",
                {
                    **row,
                    "index": self.attempted + 1,
                    "phase": "reserved_sent_or_unknown_no_retry",
                    "started_ns": started,
                    "url": url,
                    "request_sha256": digest(raw),
                },
            )
            self.attempted += 1
            send_reserved = True
            token = NETWORK_ALLOWED.set(True)
            try:
                response = self.client.post(
                    url,
                    content=raw,
                    headers={
                        "Authorization": "Bearer " + self.keys[kind],
                        "Content-Type": "application/json",
                    },
                )
            finally:
                NETWORK_ALLOWED.reset(token)
            require(response.status_code == 200, "Non200Response")
            data = response_data(
                response.content, stage, payload, list(self.keys.values())
            )
            write_new(folder / "response.json", response.content)
            result = {
                **row,
                "index": self.attempted,
                "phase": "complete",
                "http_status": 200,
                "started_ns": started,
                "finished_ns": time.time_ns(),
                "request_sha256": digest(raw),
                "response_sha256": digest(response.content),
                "usage": data["usage"],
                "request_id": data.get("request_id") or data.get("id"),
                "returned_model": data.get("model"),
                "returned_model_present": "model" in data,
            }
            no_secrets(canonical(result), list(self.keys.values()))
            write_json(folder / "result.json", result)
            self.completed.append(result)
            return data
        except BaseException as error:
            self.failed = True
            if self.first_failure is None:
                status = response.status_code if response is not None else None
                self.first_failure = {
                    "error_type": failure_category(error),
                    "http_status": status
                    if type(status) is int and 100 <= status <= 599
                    else None,
                }
                if send_reserved:
                    # Only the already durable attempt gets an error record.
                    # Never serialize error text, response body, or headers.
                    record = {
                        **row,
                        **self.first_failure,
                        "index": self.attempted,
                        "phase": "halted_no_retry",
                        "started_ns": started,
                        "finished_ns": time.time_ns(),
                        "request_sha256": digest(raw),
                    }
                    self.failure_record_saved = False
                    try:
                        write_json(folder / "error.json", record)
                        self.failure_record_saved = True
                    except BaseException:
                        # Keep the original failure and durable reservation even
                        # if diagnostic persistence also fails. The halt record
                        # reports this failure; the run can never resume.
                        pass
            raise

    def raise_if_failed(self) -> None:
        if self.failed:
            raise Stop(
                (self.first_failure or {}).get("error_type", "PriorFailureNoFallback")
            )

    async def keyword(self, prompt, **kwargs):
        require(
            kwargs.get("response_format") == {"type": "json_object"},
            "UnexpectedLLMCall",
        )
        require(
            not kwargs.get("history_messages") and not kwargs.get("system_prompt"),
            "UnexpectedKeywordHistory",
        )
        payload = {
            "model": MODELS["keyword"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        return self.post("keyword", payload)["choices"][0]["message"]["content"]

    async def embedding(self, texts, **kwargs):
        import numpy as np

        require(1 <= len(texts) <= 3, "UnexpectedEmbeddingBatch")
        data = self.post(
            "embedding",
            {"model": MODELS["embedding"], "input": texts, "encoding_format": "base64"},
        )
        return np.asarray(
            [decode_vector(row["embedding"]) for row in data["data"]], dtype=np.float32
        )

    async def rerank(self, query, documents, top_n):
        require(top_n == 12, "RerankTopNChanged")
        rerank_size(query, documents)
        payload = {
            "model": MODELS["rerank"],
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": 12, "return_documents": False},
        }
        data = self.post("rerank", payload)
        return validate_ranking(data, len(documents), top_n)


def prompt_text(
    context: str, query: str, param, config: dict, template: str, utils
) -> str:
    effective = utils.resolve_user_prompt(
        param.user_prompt,
        config.get("user_prompt_prefix", ""),
        param.disable_user_prompt_prefix,
    )
    system = template.format(
        context_data=context,
        response_type=param.response_type or "Multiple Paragraphs",
        user_prompt=effective.slot,
    )
    return "\n\n".join([system, "---User Query---", query])


def lineage(chunks: list[dict], sources: dict, stored: dict) -> list[dict]:
    result = []
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        require(
            cid in stored and chunk.get("content") == stored[cid]["content"],
            "ChunkContentChanged",
        )
        source = chunk.get("file_path")
        require(
            source in sources and source == stored[cid]["file_path"],
            "UnapprovedChunkSource",
        )
        result.append(
            {
                "chunk_id": cid,
                "content_sha256": digest(chunk["content"].encode()),
                "file_path": source,
                "source_sha256": sources[source]["source_sha256"],
                "reference_id": chunk.get("reference_id"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
    return result


class Capture:
    def __init__(self, operate, utils):
        self.operate, self.utils = operate, utils
        self.original = operate._build_context_str
        self.ref_original = operate.generate_reference_list_from_chunks
        self.rerank_original = utils.apply_rerank_if_enabled
        self.args, self.final, self.calls = None, None, 0

    def references(self, chunks):
        refs, final = self.ref_original(chunks)
        self.final = copy.deepcopy(final)
        return refs, final

    async def wrapped(self, *args, **kwargs):
        self.calls += 1
        require(self.calls == 1, "RepeatedContextBuild")
        bound = inspect.signature(self.original).bind(*args, **kwargs)
        bound.apply_defaults()
        self.args = {
            k: (v if k == "global_config" else copy.deepcopy(v))
            for k, v in bound.arguments.items()
        }
        require(self.args["progress_callback"] is None, "UnexpectedProgressCallback")
        return await self.original(*args, **kwargs)

    def __enter__(self):
        self.operate._build_context_str = self.wrapped
        self.operate.generate_reference_list_from_chunks = self.references
        return self

    def __exit__(self, *_):
        self.operate._build_context_str = self.original
        self.operate.generate_reference_list_from_chunks = self.ref_original
        self.utils.apply_rerank_if_enabled = self.rerank_original

    async def build(self, rerank: bool, provider: Provider):
        args = {
            k: (v if k == "global_config" else copy.deepcopy(v))
            for k, v in self.args.items()
        }
        args["query_param"].enable_rerank = rerank
        expected = copy.deepcopy(args["merged_chunks"])
        invoked = 0

        async def strict_rerank(
            query, retrieved_docs, global_config, enable_rerank=True, top_n=None
        ):
            nonlocal invoked
            invoked += 1
            require(
                invoked == 1 and enable_rerank and retrieved_docs == expected,
                "RerankInputChanged",
            )
            require(global_config["min_rerank_score"] == 0.0, "RerankThresholdChanged")
            rows = await provider.rerank(
                query, [c["content"] for c in retrieved_docs], top_n
            )
            return [
                {**retrieved_docs[r["index"]], "rerank_score": r["relevance_score"]}
                for r in rows
            ]

        self.utils.apply_rerank_if_enabled = strict_rerank
        self.final = None
        context, data = await self.original(**args)
        require(invoked == int(rerank), "RerankMissingOrRepeated")
        require(self.final is not None, "FinalChunksMissing")
        return context, data, copy.deepcopy(self.final)


async def build_pairs(run: Path, m: dict, provider: Provider) -> list[dict]:
    LightRAG, QueryParam, operate, utils, prompts = native_modules()
    require(
        prompts["rag_response"] == m["templates"]["rag_response"]
        and prompts["kg_query_context"] == m["templates"]["kg_query_context"],
        "TemplateChanged",
    )
    prefix = m["templates"]["user_prompt_prefix"]

    # Functions deepcopy by identity. Bound methods would copy Provider and its
    # HTTP client's non-copyable lock through LightRAG's dataclass asdict.
    async def keyword(prompt, **kwargs):
        return await provider.keyword(prompt, **kwargs)

    async def embed(texts, **kwargs):
        return await provider.embedding(texts, **kwargs)

    embedding = utils.EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed)
    tokenizer = utils.TiktokenTokenizer("gpt-4o-mini")
    require(
        tokenizer.encode("Offline tokenizer verification. 通用保险条款。\n12345")
        == m["tokenizer_probe"],
        "TokenizerChanged",
    )
    rag = LightRAG(
        working_dir=str(run / "runtime-storage"),
        workspace="",
        llm_model_func=keyword,
        llm_model_name=MODELS["keyword"],
        embedding_func=embedding,
        user_prompt_prefix=prefix,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        embedding_cache_config={"enabled": False},
        addon_params={"language": "Chinese"},
        min_rerank_score=0.0,
        kg_chunk_pick_method="VECTOR",
        related_chunk_number=5,
        cosine_better_than_threshold=0.2,
        enable_content_headings=True,
        top_k=30,
        chunk_top_k=12,
        max_total_tokens=18000,
        max_entity_tokens=4000,
        max_relation_tokens=5000,
        llm_model_max_async=1,
        embedding_func_max_async=1,
        default_llm_timeout=240,
        default_embedding_timeout=240,
        tiktoken_model_name="gpt-4o-mini",
        tokenizer=tokenizer,
    )
    await rag.initialize_storages()
    stored = load(run / "source-storage/kv_store_text_chunks.json")
    pairs = []
    try:
        for case in m["cases"]:
            provider.case_id = case["id"]
            before = provider.attempted
            with Capture(operate, utils) as capture:
                native = await rag.aquery_llm(
                    case["question"],
                    param=QueryParam(**QUERY),
                    system_prompt=m["templates"]["rag_response"],
                )
                provider.raise_if_failed()
                require(
                    not provider.failed and provider.attempted == before + 2,
                    "RetrievalCallCount",
                )
                require(
                    capture.calls == 1 and capture.args is not None and capture.final,
                    "MissingCapture",
                )
                native_final = copy.deepcopy(capture.final)
                args = capture.args
                require(
                    args["query"] == case["question"]
                    and not args["query_param"].enable_rerank,
                    "CaptureQueryChanged",
                )
                require(args["merged_chunks"], "NoCandidateChunks")
                snapshot = {
                    k: copy.deepcopy(v)
                    for k, v in args.items()
                    if k not in ("global_config", "query_param", "progress_callback")
                }
                # Native relation lookup uses tuple keys. Keep that map untouched
                # in memory; persist explicit records instead of lossy string keys.
                snapshot["relation_id_to_original"] = [
                    {"key": list(k) if isinstance(k, tuple) else k, "value": v}
                    for k, v in (args["relation_id_to_original"] or {}).items()
                ]
                snapshot["query_param"] = asdict(args["query_param"])
                snapshot["public_global_config"] = {
                    k: args["global_config"].get(k) for k in PUBLIC_GLOBAL_FIELDS
                }
                snapshot["candidate_lineage"] = lineage(
                    args["merged_chunks"], m["sources"], stored
                )
                snapshot["rerank_length_preflight"] = rerank_size(
                    case["question"], [c["content"] for c in args["merged_chunks"]]
                )
                require(
                    len({c["chunk_id"] for c in args["merged_chunks"]})
                    == len(args["merged_chunks"]),
                    "DuplicateCandidate",
                )
                folder = run / "cases" / case["id"]
                mkdir(folder)
                no_secrets(canonical(snapshot), list(provider.keys.values()))
                write_json(folder / "candidate-snapshot.json", snapshot)
                context_a, data_a, final_a = await capture.build(False, provider)
                prompt_a = prompt_text(
                    context_a,
                    case["question"],
                    args["query_param"],
                    args["global_config"],
                    args["system_prompt"],
                    utils,
                )
                native_prompt = native.get("llm_response", {}).get("content")
                require(
                    native_prompt == prompt_a and native_final == final_a,
                    "NativeAByteMismatch",
                )
                context_b, data_b, final_b = await capture.build(True, provider)
                require(
                    provider.attempted == before + 3 and not provider.failed,
                    "RerankCallCount",
                )
                prompt_b = prompt_text(
                    context_b,
                    case["question"],
                    args["query_param"],
                    args["global_config"],
                    args["system_prompt"],
                    utils,
                )
                require(
                    args["entities_context"] == snapshot["entities_context"]
                    and args["relations_context"] == snapshot["relations_context"],
                    "GraphContextChanged",
                )
                for key in ("entities", "relationships"):
                    require(
                        data_a.get("data", {}).get(key)
                        == data_b.get("data", {}).get(key),
                        "GraphOutputChanged",
                    )
                ranking = load(
                    run / "attempts" / f"{provider.attempted:04d}" / "response.json"
                )["output"]["results"]
                expected_order = [
                    args["merged_chunks"][r["index"]]["chunk_id"] for r in ranking
                ]
                require(
                    [c["chunk_id"] for c in final_b] == expected_order[: len(final_b)],
                    "ActualRerankOrderMismatch",
                )
                for arm, prompt, context, data, final in (
                    ("off", prompt_a, context_a, data_a, final_a),
                    ("on", prompt_b, context_b, data_b, final_b),
                ):
                    require(final, "EmptyFinalContext")
                    no_secrets(prompt.encode(), list(provider.keys.values()))
                    write_new(folder / f"{arm}-prompt.txt", prompt.encode())
                    write_json(
                        folder / f"{arm}-context.json",
                        {
                            "context": context,
                            "raw_data": data,
                            "final_chunks": final,
                            "lineage": lineage(final, m["sources"], stored),
                        },
                    )
                write_json(
                    folder / "comparison.json",
                    {
                        "native_a_byte_identical": True,
                        "entity_relation_context_unchanged": True,
                        "merged_candidate_count": len(args["merged_chunks"]),
                        "candidate_sha256": digest(canonical(snapshot)),
                        "off_prompt_sha256": digest(prompt_a.encode()),
                        "on_prompt_sha256": digest(prompt_b.encode()),
                        "prompt_equal": prompt_a == prompt_b,
                        "off_chunks": len(final_a),
                        "on_chunks": len(final_b),
                    },
                )
                pairs.append(
                    {
                        "id": case["id"],
                        "question": case["question"],
                        "off": prompt_a,
                        "on": prompt_b,
                    }
                )
    finally:
        await rag.finalize_storages()
    return pairs


def real_client():
    return httpx.Client(
        transport=httpx.HTTPTransport(retries=0, verify=True, trust_env=False),
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(180.0),
    )


def blind_bundle(run: Path, m: dict, results: list[dict]) -> dict:
    require(len(results) == 15, "BlindBundleRequiresCompleteRun")
    answers = {
        (r["case_id"], r["stage"].removeprefix("answer_")): r
        for r in results
        if r["stage"].startswith("answer_")
    }
    cases = {c["id"]: c for c in m["cases"]}
    entries = []
    for row in m["blind_assignment"]:
        cid, arm = row["case_id"], row["arm"]
        result = answers[(cid, arm)]
        data = load(run / "attempts" / f"{result['index']:04d}" / "response.json")
        evidence = load(run / "cases" / cid / f"{arm}-context.json")
        entries.append(
            {
                "blind_id": row["blind_id"],
                "case_id": cid,
                "question": cases[cid]["question"],
                "answer": data["choices"][0]["message"]["content"],
                "actual_prompt": read(
                    run / "cases" / cid / f"{arm}-prompt.txt"
                ).decode(),
                "reference_sources": [
                    {k: v for k, v in item.items() if k != "rerank_score"}
                    for item in evidence["lineage"]
                ],
            }
        )
    return {
        "schema": 1,
        "instructions": "Score every answer against the frozen oracle and original sources before unblinding.",
        "entries": entries,
    }


def execute(
    run: Path,
    reviewed_sha: str,
    review: Path,
    review_sha: str,
    client_factory=real_client,
) -> dict:
    with run_lock(run):
        m, keys = verify(run, reviewed_sha)
        require(digest(read(review)) == review_sha, "IndependentReviewPinMismatch")
        approval = load(review)
        require(
            approval.get("approved") is True
            and approval.get("preflight_sha256") == reviewed_sha,
            "IndependentReviewMissing",
        )
        require(
            not any(
                (run / n).exists()
                for n in (
                    "run-reservation.json",
                    "attempts",
                    "halt.json",
                    "run-summary.json",
                )
            ),
            "PreviousRunNoResume",
        )
        write_json(
            run / "run-reservation.json",
            {
                "phase": "one_shot_no_resume",
                "started_ns": time.time_ns(),
                "maximum_requests": 15,
                "preflight_sha256": reviewed_sha,
                "independent_review_sha256": review_sha,
            },
        )
        mkdir(run / "attempts")
        mkdir(run / "cases")
        provider = None
        try:
            with client_factory() as client:

                def verify_current():
                    _, current_keys = verify(run, reviewed_sha)
                    require(current_keys == keys, "CredentialsChangedDuringRun")
                    require(
                        digest(read(review)) == review_sha, "IndependentReviewChanged"
                    )

                provider = Provider(run, m, keys, client, verify_current)
                pairs = asyncio.run(build_pairs(run, m, provider))
                require(
                    provider.attempted == len(provider.completed) == 9
                    and not provider.failed,
                    "RetrievalStageIncomplete",
                )
                by_id = {p["id"]: p for p in pairs}
                for row in m["schedule"][9:]:
                    provider.case_id = row["case_id"]
                    arm = row["stage"].removeprefix("answer_")
                    pair = by_id[row["case_id"]]
                    system, query = pair[arm].rsplit("\n\n---User Query---\n\n", 1)
                    require(query == pair["question"], "QuestionChanged")
                    provider.post(
                        row["stage"],
                        {
                            "model": MODELS["answer"],
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": query},
                            ],
                            "temperature": 0,
                            "max_tokens": 2048,
                            "stream": False,
                            "enable_thinking": False,
                        },
                    )
                require(
                    provider.attempted == len(provider.completed) == 15
                    and not provider.failed,
                    "IncompleteRun",
                )
                verify(run, reviewed_sha)
                bundle = blind_bundle(run, m, provider.completed)
                no_secrets(canonical(bundle), list(keys.values()))
                write_json(run / "blind-review-input.json", bundle)
                result = {
                    "phase": "complete",
                    "attempted": 15,
                    "completed": 15,
                    "preflight_sha256": reviewed_sha,
                    "results": provider.completed,
                    "pair_comparisons": [
                        {
                            "case_id": p["id"],
                            "prompt_equal": p["off"] == p["on"],
                            "off_prompt_sha256": digest(p["off"].encode()),
                            "on_prompt_sha256": digest(p["on"].encode()),
                        }
                        for p in pairs
                    ],
                    "quality_status": "unscored_requires_independent_oracle_review",
                }
                write_json(run / "run-summary.json", result)
                return {"phase": "complete", "attempted": 15, "completed": 15}
        except BaseException as error:
            first_failure = provider.first_failure if provider else None
            category = (
                first_failure["error_type"]
                if first_failure
                else failure_category(error)
            )
            write_json(
                run / "halt.json",
                {
                    "phase": "halted_no_retry",
                    "error_type": category,
                    "attempted": provider.attempted if provider else 0,
                    "completed": len(provider.completed) if provider else 0,
                    "first_provider_failure": first_failure,
                    "failure_record_saved": provider.failure_record_saved
                    if provider
                    else None,
                },
            )
            raise Stop(category) from None


def main():
    os.umask(0o077)
    clean_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("run-dir", "source-root", "storage", "oracle-dir", "env-file"):
        prep.add_argument("--" + name, required=True, type=Path)
    check = sub.add_parser("verify")
    run = sub.add_parser("run")
    for cmd in (check, run):
        cmd.add_argument("--run-dir", required=True, type=Path)
        cmd.add_argument("--preflight-sha", required=True)
    run.add_argument("--review-file", required=True, type=Path)
    run.add_argument("--review-sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(
                args.run_dir.resolve(),
                args.source_root.resolve(),
                args.storage.resolve(),
                args.oracle_dir.resolve(),
                args.env_file.resolve(),
            )
        elif args.command == "verify":
            verify(args.run_dir.resolve(), args.preflight_sha)
            result = {"phase": "verified_offline"}
        else:
            result = execute(
                args.run_dir.resolve(),
                args.preflight_sha,
                args.review_file.resolve(),
                args.review_sha,
            )
        print(json.dumps(result, ensure_ascii=False))
    except Stop as error:
        print(json.dumps({"phase": "halted", "category": str(error)}))
        return 1
    except BaseException:
        print(json.dumps({"phase": "halted", "category": "UnclassifiedLocalFailure"}))
        return 1
    return 0


if __name__ == "__main__":
    with network_guard():
        raise SystemExit(main())
