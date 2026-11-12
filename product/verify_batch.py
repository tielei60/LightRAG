"""Read persisted native indexes and record fixed native HTTP query checks.

Run readback after the SDK writer exits. Query reports retain the chunks returned
by the same answer request; they do not claim to capture its complete LLM prompt.
Business correctness remains pending source-based review of the saved answers.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import networkx as nx
import numpy as np
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SEPARATOR = "<SEP>"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def checked_child(directory: Path, name: str) -> Path:
    path = (directory / name).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes the batch directory")
    return path


def digest(text: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


def verify_snapshot(batch: Path, storage: Path, embedding_dim: int) -> dict[str, Any]:
    """Check published files without initializing or mutating native storages."""
    manifest = load_json(batch / "manifest.json")
    paths = {
        name: storage / f"kv_store_{name}.json"
        for name in ["doc_status", "full_docs", "text_chunks"]
    }
    paths.update(
        {
            name: storage / f"vdb_{name}.json"
            for name in ["entities", "relationships", "chunks"]
        }
    )
    paths["graph"] = storage / "graph_chunk_entity_relation.graphml"
    before = {
        name: (p.stat().st_mtime_ns, p.stat().st_size) for name, p in paths.items()
    }
    statuses, full_docs, chunks = (
        load_json(paths[name]) for name in ["doc_status", "full_docs", "text_chunks"]
    )
    graph = nx.read_graphml(paths["graph"])
    issues: list[str] = []
    documents = manifest["documents"]
    expected_documents: dict[str, dict[str, Any]] = {}
    expected_contents: Counter[str] = Counter()
    for entry in documents:
        payload_path = checked_child(batch, entry["prepared_path"])
        if (
            hashlib.sha256(payload_path.read_bytes()).hexdigest()
            != entry["prepared_sha256"]
        ):
            raise ValueError("Prepared input changed after manifest creation")
        payload = load_json(payload_path)
        label = Path(entry.get("ingestion_label", payload["file_source"])).name
        if label in expected_documents:
            issues.append(f"duplicate_document_source_label:{label}")
        delimiter = payload["chunking"]["params"]["split_by_character"]
        contents = [content.strip() for content in payload["text"].split(delimiter)]
        expected_documents[label] = {"product": entry["product"], "contents": contents}
        expected_contents.update(contents)

    if len(statuses) != len(documents):
        issues.append(f"document_count:{len(statuses)}!={len(documents)}")
    actual_files = Counter(
        Path(row.get("file_path", "")).name for row in statuses.values()
    )
    if actual_files != Counter(expected_documents.keys()):
        issues.append("document_source_labels_mismatch")
    if set(full_docs) != set(statuses):
        issues.append("full_document_set_mismatch")
    status_chunk_ids: Counter[str] = Counter()
    chunk_products: dict[str, str] = {}
    for doc_id, row in statuses.items():
        if row.get("status") != "processed":
            issues.append(f"document_not_processed:{doc_id}:{row.get('status')}")
        if doc_id not in full_docs:
            issues.append(f"full_document_missing:{doc_id}")
        if not row.get("chunks_list"):
            issues.append(f"document_chunks_missing:{doc_id}")
        status_chunk_ids.update(row.get("chunks_list", []))
        expected_document = expected_documents.get(Path(row.get("file_path", "")).name)
        actual_document_content = Counter()
        for chunk_id in row.get("chunks_list", []):
            if chunk_id not in chunks:
                issues.append(f"status_chunk_missing:{chunk_id}")
                continue
            chunk = chunks[chunk_id]
            if chunk.get("full_doc_id") != doc_id:
                issues.append(f"status_chunk_wrong_document:{chunk_id}")
            content = chunk.get("content", "")
            actual_document_content.update([content])
            if expected_document and content in expected_document["contents"]:
                chunk_products[chunk_id] = expected_document["product"]
        if expected_document and actual_document_content != Counter(
            expected_document["contents"]
        ):
            issues.append(f"document_chunk_content_mismatch:{doc_id}")
    if set(status_chunk_ids) != set(chunks):
        issues.append("document_chunk_membership_mismatch")
    if any(count != 1 for count in status_chunk_ids.values()):
        issues.append("document_chunk_membership_not_unique")
    actual_contents = Counter(row.get("content", "") for row in chunks.values())
    if actual_contents != expected_contents:
        issues.append(
            f"chunk_content_multiset_mismatch:missing={sum((expected_contents - actual_contents).values())}"
            f":extra={sum((actual_contents - expected_contents).values())}"
        )
    for chunk_id, row in chunks.items():
        if row.get("content", "") not in expected_contents:
            issues.append(f"chunk_text_mismatch:{chunk_id}")
        if row.get("full_doc_id") not in statuses:
            issues.append(f"chunk_document_missing:{chunk_id}")

    products = {entry["product"] for entry in documents}

    def product_of(name: str) -> str | None:
        return next(
            (p for p in products if name == p or name.startswith(p + "｜")), None
        )

    def check_sources(kind: str, name: str, row: dict, product: str | None) -> None:
        source_ids = [
            part
            for part in str(row.get("source_id", "")).split(SOURCE_SEPARATOR)
            if part
        ]
        if not source_ids:
            issues.append(f"{kind}_sources_missing:{name}")
        for chunk_id in source_ids:
            if chunk_id not in chunks:
                issues.append(f"{kind}_source_missing:{name}:{chunk_id}")
            elif (
                product
                and chunk_id in chunk_products
                and chunk_products[chunk_id] != product
            ):
                issues.append(f"{kind}_cross_product_source:{name}:{chunk_id}")
        if not str(row.get("description", "")).strip():
            issues.append(f"{kind}_description_missing:{name}")

    for name, row in graph.nodes(data=True):
        check_sources("entity", name, row, product_of(name))
    for source, target, row in graph.edges(data=True):
        source_product, target_product = product_of(source), product_of(target)
        if source_product and target_product and source_product != target_product:
            issues.append(f"relation_cross_product:{source}:{target}")
        check_sources(
            "relation", f"{source}::{target}", row, source_product or target_product
        )
    if not graph.number_of_nodes() or not graph.number_of_edges():
        issues.append("graph_empty")

    vector_counts = {}
    for namespace in ["entities", "relationships", "chunks"]:
        vector_store = load_json(paths[namespace])
        rows = vector_store.get("data", [])
        ids = [row.get("__id__") for row in rows]
        vector_counts[namespace] = len(rows)
        if len(ids) != len(set(ids)):
            issues.append(f"vector_ids_not_unique:{namespace}")
        if vector_store.get("embedding_dim") != embedding_dim:
            issues.append(f"vector_dimension_mismatch:{namespace}")
        raw = base64.b64decode(vector_store.get("matrix", ""), validate=True)
        if len(raw) != len(rows) * embedding_dim * 4:
            issues.append(f"vector_matrix_size_mismatch:{namespace}")
        elif rows:
            matrix = np.frombuffer(raw, dtype=np.float32).reshape(
                len(rows), embedding_dim
            )
            if (
                not np.isfinite(matrix).all()
                or (np.linalg.norm(matrix, axis=1) == 0).any()
            ):
                issues.append(f"vector_invalid:{namespace}")
        expected_ids = (
            set(chunks)
            if namespace == "chunks"
            else {digest(name, "ent-") for name in graph.nodes}
            if namespace == "entities"
            else {digest("".join(sorted((a, b))), "rel-") for a, b in graph.edges}
        )
        if set(ids) != expected_ids:
            issues.append(f"vector_index_set_mismatch:{namespace}")
        if any(not str(row.get("content", "")).strip() for row in rows):
            issues.append(f"vector_content_missing:{namespace}")
        if namespace == "chunks":
            for row in rows:
                if row["__id__"] in chunks and row.get("content") != chunks[
                    row["__id__"]
                ].get("content"):
                    issues.append(f"vector_chunk_content_mismatch:{row['__id__']}")
    after = {
        name: (p.stat().st_mtime_ns, p.stat().st_size) for name, p in paths.items()
    }
    if before != after:
        raise RuntimeError(
            "Native index files changed during readback; stop the writer first"
        )
    return {
        "workspace": manifest["workspace"],
        "method": "Read published native JSON/GraphML in a separate process; no storage mutation",
        "passed": not issues,
        "documents": len(statuses),
        "document_states": dict(
            Counter(row.get("status") for row in statuses.values())
        ),
        "chunks": len(chunks),
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "vectors": vector_counts,
        "issues": issues,
        "business_acceptance": "pending: exact persistence does not establish semantic correctness",
    }


def verification_source(root: Path, batch: Path, workspace: str) -> Path:
    """Verify the accumulated workspace when incremental batches are registered."""
    from product.workspace_registry import registry_path, workspace_snapshot

    if registry_path(root, workspace).exists():
        return workspace_snapshot(root, workspace)
    return batch


def verification_storage(root: Path, batch: Path, config: dict) -> Path:
    """Read database backends independently; file exports are verification only."""
    from product.database_storage import is_database_backend, snapshot_database_config

    if is_database_backend(config):
        return asyncio.run(
            snapshot_database_config(
                config, root, batch / "verification" / str(time.time_ns())
            )
        )
    return Path(config["WORKING_DIR"]) / config["WORKSPACE"]


@dataclass(frozen=True)
class QueryCase:
    id: str
    question: str
    evidence_terms: tuple[str, ...] = ()
    review_focus: str = ""


CASES = (
    QueryCase(
        "youhu-overview",
        "京康优护有哪些保障计划，各计划包含哪些基本和可选保障责任？",
        ("计划一", "计划二", "住院津贴"),
        "区分计划、基本与可选责任，不默认某计划",
    ),
    QueryCase(
        "youhu-plan-one",
        "京康优护计划一保哪些责任，包含住院津贴吗？",
        ("计划一", "计划二", "住院津贴"),
        "计划一与计划二差异、选择条件",
    ),
    QueryCase(
        "youhu-plan-two",
        "京康优护计划二与计划一相比多了什么，能否单独投保住院津贴？",
        ("计划一", "计划二", "住院津贴"),
        "责任选择前提必须有原文支持",
    ),
    QueryCase(
        "youhu-ratio",
        "京康优护是不是所有医疗费用都固定按60%赔付？什么情况下涉及60%，这个比例的基数是什么？",
        ("约定比例的60%", "结算"),
        "纠正错误前提，保留结算条件及约定比例基数",
    ),
    QueryCase(
        "youhu-deductible",
        "京康优护住院医疗和住院前后门急诊共用什么免赔额？这能否证明两项责任的等待期和延续保障规则也完全一样？",
        ("免赔额", "等待期"),
        "不能从一项共性推导所有规则共同适用",
    ),
    QueryCase(
        "youhu-underwriting",
        "京康优护被保险人的投保年龄和保险期间是什么，是否涉及加费？",
        ("28日", "65周岁", "不涉及加费"),
        "年龄上下界、期间与加费原文",
    ),
    QueryCase(
        "antai-plans",
        "京安泰有哪些投保计划，京安卡成人版和少儿版的航空意外保障有何差异？",
        ("成人版", "少儿版", "航空意外"),
        "表头列归属与0值不得串列",
    ),
    QueryCase(
        "antai-selection",
        "京安泰能否只选住院津贴、不选意外医疗？",
        ("意外医疗", "住院津贴"),
        "完整选择前提，否定错误组合",
    ),
    QueryCase(
        "antai-underwriting",
        "京安泰哪些公共核保规则不适用？这是否意味着产品自身职业和保额条件都不用满足？",
        ("不适用", "职业", "保额"),
        "区分公共规则例外与产品自身要求",
    ),
    QueryCase(
        "mixed-products",
        "对比京安泰和京康优护：住院津贴分别属于什么保障，选择前提是什么？",
        ("京安泰", "京康优护", "住院津贴"),
        "同库同名责任不能跨产品套用条件",
    ),
    QueryCase(
        "missing-evidence",
        "京康优护投保五年后的现金价值具体是多少？",
        (),
        "本批未导入现金价值表且不做金额查算，不得编造数值",
    ),
)


def fixed_query_settings(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "top_k": 30,
        "chunk_top_k": 12,
        "max_entity_tokens": 4000,
        "max_relation_tokens": 5000,
        "max_total_tokens": 18000,
    }
    return {
        "mode": "mix",
        "enable_rerank": rerank_enabled(config),
        "conversation_history": [],
        **{
            key: int(config.get(key.upper()) or default)
            for key, default in defaults.items()
        },
    }


def rerank_enabled(config: dict[str, Any]) -> bool:
    return str(config.get("RERANK_BINDING") or "null").strip().lower() not in {
        "",
        "null",
        "none",
    }


def public_rerank_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Include reproducible rerank settings without credentials."""
    return {
        key: config.get(key)
        for key in [
            "RERANK_BINDING",
            "RERANK_MODEL",
            "RERANK_BINDING_HOST",
            "RERANK_ENABLE_CHUNKING",
            "RERANK_MAX_TOKENS_PER_DOC",
        ]
    }


def run_query(
    client: httpx.Client, case: QueryCase, settings: dict[str, Any], stream: bool = True
) -> dict[str, Any]:
    request = {
        "query": case.question,
        **settings,
        "include_references": True,
        "include_chunk_content": True,
        "stream": stream,
    }
    started = time.perf_counter()
    references: list[dict] = []
    answer: list[str] = []
    first_output: float | None = None
    events: list[dict] = []
    if stream:
        request["include_progress"] = True
        with client.stream("POST", "/query/stream", json=request) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                events.append(event)
                if event.get("error"):
                    raise RuntimeError(f"Native query stream failed: {event['error']}")
                if "references" in event:
                    references = event["references"]
                if event.get("response"):
                    if first_output is None:
                        first_output = time.perf_counter() - started
                    answer.append(event["response"])
    else:
        response = client.post("/query", json=request)
        response.raise_for_status()
        result = response.json()
        events.append(result)
        answer.append(result["response"])
        references = result.get("references") or []
        first_output = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    if not answer:
        raise RuntimeError("Native query returned no answer content")
    context = re.sub(
        r"\s+",
        "",
        "\n".join(text for ref in references for text in ref.get("content", [])),
    )
    missing = [
        term for term in case.evidence_terms if re.sub(r"\s+", "", term) not in context
    ]
    return {
        "case": asdict(case),
        "request": request,
        "answer": "".join(answer),
        "references": references,
        "same_request_context_kind": "Selected native chunks returned with this answer; complete assembled LLM prompt is not captured",
        "evidence_term_checks": {
            "missing": missing,
            "passed": not missing,
            "scope": "mechanical text presence only; semantic review required",
        },
        "first_output_seconds": round(first_output, 3)
        if first_output is not None
        else None,
        "total_seconds": round(elapsed, 3),
        "stream": stream,
        "events": events,
        "business_review": "pending",
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower), 3)


def summarize_timings(rows: list[dict]) -> dict[str, Any]:
    result = {}
    for phase in ["first_request", "repeated_request"]:
        selected = [
            row
            for row in rows
            if row["cache_condition"] == phase and "total_seconds" in row
        ]
        elapsed = [row["total_seconds"] for row in selected]
        first = [
            row["first_output_seconds"]
            for row in selected
            if row["first_output_seconds"] is not None
        ]
        result[phase] = {
            "samples": len(selected),
            "total_p50_seconds": round(statistics.median(elapsed), 3)
            if elapsed
            else None,
            "total_p95_seconds": percentile(elapsed, 0.95),
            "first_output_p50_seconds": round(statistics.median(first), 3)
            if first
            else None,
            "first_output_p95_seconds": percentile(first, 0.95),
        }
    return result


def run_checks(
    batch: Path,
    url: str,
    config: dict[str, Any],
    cases: tuple[QueryCase, ...],
    repeat: int,
    stream: bool,
) -> dict[str, Any]:
    """Save every response before advancing; never erase earlier query evidence."""
    if repeat < 1:
        raise ValueError("Repeat count must be positive")
    settings = fixed_query_settings(config)
    rows: list[dict] = []
    run_dir = batch / "queries" / str(time.time_ns())
    report = {
        "workspace": config["WORKSPACE"],
        "url": url,
        "settings": settings,
        "models": {
            key: config.get(key)
            for key in [
                "QUERY_LLM_MODEL",
                "KEYWORD_LLM_MODEL",
                "EMBEDDING_MODEL",
                "EMBEDDING_DIM",
            ]
        },
        "query_generation_options": json.loads(
            config.get("QUERY_OPENAI_LLM_EXTRA_BODY") or "{}"
        ),
        "rerank": public_rerank_settings(config),
        "cache_measurement": "First versus repeated requests; cache hits are not inferred. Streaming skips native answer caching; keyword/embedding caches can still be warm. Existing cache is preserved.",
        "acceptance": "pending: saved answers and source conditions require business review; term presence is not answer correctness",
        "run_directory": str(run_dir),
        "results": rows,
    }
    with httpx.Client(
        base_url=url,
        timeout=httpx.Timeout(float(config.get("LLM_TIMEOUT") or 240) + 120),
        trust_env=False,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        if (
            health.json().get("configuration", {}).get("workspace")
            != config["WORKSPACE"]
        ):
            raise ValueError("Server workspace differs from selected batch")
        for number in range(repeat):
            for case in cases:
                try:
                    row = run_query(client, case, settings, stream)
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    row = {
                        "case": asdict(case),
                        "error": str(exc),
                        "business_review": "blocked",
                    }
                row.update(
                    repetition=number + 1,
                    cache_condition="first_request"
                    if number == 0
                    else "repeated_request",
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
                            "total_seconds": row.get("total_seconds"),
                            "missing_evidence_terms": row.get(
                                "evidence_term_checks", {}
                            ).get("missing"),
                            "error": row.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    save_json(batch / "query-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("readback")
    qa = subparsers.add_parser("qa")
    qa.add_argument("--url")
    qa.add_argument("--repeat", type=int, default=3)
    qa.add_argument("--cases", nargs="+", choices=[case.id for case in CASES])
    qa.add_argument(
        "--no-stream",
        action="store_true",
        help="Measure native answer-cache requests; first output then equals complete response time",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.batch):
        raise ValueError("Invalid batch name")
    batch = ROOT / "output" / args.batch
    config = dict(dotenv_values(batch / "runtime.env", interpolate=False))
    manifest = load_json(batch / "manifest.json")
    if manifest["workspace"] != config["WORKSPACE"]:
        raise ValueError("Manifest and runtime workspace differ")
    if args.action == "readback":
        source = verification_source(ROOT, batch, config["WORKSPACE"])
        report = verify_snapshot(
            source,
            verification_storage(ROOT, batch, config),
            int(config["EMBEDDING_DIM"]),
        )
        report_name = (
            "workspace-readback-report.json"
            if source != batch
            else "readback-report.json"
        )
        save_json(batch / report_name, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    else:
        cases = tuple(case for case in CASES if not args.cases or case.id in args.cases)
        run_checks(
            batch,
            args.url or f"http://{config['HOST']}:{config['PORT']}",
            config,
            cases,
            args.repeat,
            not args.no_stream,
        )


if __name__ == "__main__":
    main()
