"""Fixed-question, same-request evidence capture; no answers in model input."""

import time
from pathlib import Path

from .common import digest, inside, read_json, save_json
from .config import Settings
from .ingest import accepted_readback
from .models import initialize_rag, make_rag, query_parameters


def evaluation_contract(settings: Settings) -> dict:
    from .prepare import checked_manifest

    cases = questions(settings)
    manifest = checked_manifest(settings)
    frozen = read_json(settings.contract_snapshot)
    query_code = {
        name: {
            key: frozen["contract_modules"][name][key]
            for key in ["source_sha256", "frozen_sha256"]
        }
        for name in [
            "answer_query",
            "answer_audit",
            "grounded_query",
            "source_scope",
            "formula_consistency",
        ]
    }
    if "answer_requirements" in frozen["contract_modules"]:
        query_code["answer_requirements"] = frozen["contract_modules"][
            "answer_requirements"
        ]["source_sha256"]
    controls = {
        name: settings.get(name)
        for name in [
            "OPENAI_BASE_URL",
            "LLM_TEMPERATURE",
            "EXTRACT_MODEL",
            "KEYWORD_MODEL",
            "QUERY_MODEL",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIM",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_SEND_DIM",
            "EMBEDDING_USE_BASE64",
            "EMBEDDING_TOKEN_LIMIT",
            "RERANK_MODEL",
            "RERANK_URL",
            "TOP_K",
            "CHUNK_TOP_K",
            "MAX_ENTITY_TOKENS",
            "MAX_RELATION_TOKENS",
            "MAX_TOTAL_TOKENS",
        ]
    }
    controls.update(
        query_code=query_code,
        query_prompt_sha256=digest(
            (settings.profiles / "user_prompt/products.md").read_bytes()
        ),
        enable_thinking=False,
        mode="mix",
        enable_rerank=True,
        answer_cache=False,
        shared_answer_guard=settings.get("ANSWER_GUARD"),
        shared_answer_guard_sha256=digest(
            (settings.root / "query_guard.py").read_bytes()
        )
        if settings.get("ANSWER_GUARD") == "scope-v1"
        else None,
    )
    import json

    canonical = lambda value: digest(
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return {
        "question_set_sha256": canonical(sorted(cases, key=lambda row: row["id"])),
        "source_set_sha256": canonical(
            sorted(
                (row["product"], row["role"], row["sha256"], row["document_version"])
                for row in manifest["documents"]
            )
        ),
        "query_contract_sha256": canonical(controls),
        "extraction_contract_sha256": canonical(
            {
                "modules": {
                    name: {k: value[k] for k in ["source_sha256", "frozen_sha256"]}
                    for name, value in frozen["contract_modules"].items()
                    if name
                    in {"evidence_extract", "evidence_semantics", "extraction_profile"}
                },
                "profile": digest(
                    (settings.profiles / "entity_type/products.yml").read_bytes()
                ),
                "max_records": settings.integer("MAX_EXTRACTION_RECORDS"),
                "max_entities": settings.integer("MAX_EXTRACTION_ENTITIES"),
                "merge_summary_threshold": 3,
                "summary_max_tokens": 1200,
                "summary_context_size": 8192,
            }
        ),
        "query_controls": controls,
    }


def questions(settings: Settings) -> list[dict]:
    rows = read_json(settings.root / "fixtures/questions.json")["cases"]
    if (
        len(rows) != 29
        or len({row["id"] for row in rows}) != 29
        or len({row["question"] for row in rows}) != 29
    ):
        raise ValueError("The frozen corpus must contain 29 unique original questions")
    if any(set(row) != {"id", "question"} for row in rows):
        raise ValueError("Request-only questions cannot contain scoring facts")
    return rows


def complete_rounds(settings: Settings, rows: list[dict]) -> dict:
    expected = {row["id"]: row["question"] for row in questions(settings)}
    keyed = {}
    for row in rows:
        identifier = row["case_id"], row["repeat"]
        if type(identifier[1]) is not int or identifier[1] not in {1, 2, 3}:
            raise ValueError("Original question rounds must be numbered 1, 2, 3")
        if identifier in keyed:
            raise ValueError("Original question rounds are duplicated")
        if (
            row["case_id"] not in expected
            or row["question"] != expected[row["case_id"]]
        ):
            raise ValueError("Original question text differs from the frozen corpus")
        keyed[identifier] = row
    repeats = {row["repeat"] for row in rows}
    if repeats not in ({1}, {1, 2, 3}) or set(keyed) != {
        (case_id, repeat) for case_id in expected for repeat in repeats
    }:
        raise ValueError(
            "A complete 29-question round or three complete rounds are required"
        )
    return keyed


async def evaluate(settings: Settings, *, repeats=3) -> dict:
    settings.require_models()
    if repeats not in {1, 3}:
        raise ValueError("Use one diagnostic round or three frozen acceptance rounds")
    accepted_readback(settings)
    cases = questions(settings)
    contract = evaluation_contract(settings)
    destination = settings.run_dir / "evaluation" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    active = []
    rag = make_rag(settings, query_record=lambda event: active.append(event))
    rows = []
    initialized = False
    try:
        await initialize_rag(rag, settings)
        initialized = True
        for repeat in range(1, repeats + 1):
            for case in cases:
                active.clear()
                started = time.monotonic()
                item = {
                    "case_id": case["id"],
                    "question": case["question"],
                    "repeat": repeat,
                    "request": {
                        "query": case["question"],
                        "mode": "mix",
                        "top_k": settings.integer("TOP_K"),
                        "chunk_top_k": settings.integer("CHUNK_TOP_K"),
                        "max_entity_tokens": settings.integer("MAX_ENTITY_TOKENS"),
                        "max_relation_tokens": settings.integer("MAX_RELATION_TOKENS"),
                        "max_total_tokens": settings.integer("MAX_TOTAL_TOKENS"),
                        "enable_rerank": True,
                        "conversation_history": [],
                    },
                    "answer": "",
                    "delivered": False,
                    "grade": "pending_independent_review",
                }
                try:
                    result = await rag.aquery_llm(
                        case["question"], param=query_parameters(settings, stream=True)
                    )
                    response = result.get("llm_response") or {}
                    if result.get("status") != "success" or not response.get(
                        "llm_generated"
                    ):
                        raise ValueError(
                            "The native query did not produce a model-generated answer"
                        )
                    if response.get("is_streaming"):
                        iterator = response.get("response_iterator")
                        if iterator is None:
                            raise ValueError(
                                "The native query returned no answer iterator"
                            )
                        async for token in iterator:
                            if token and "first_final_output_seconds" not in item:
                                item["first_final_output_seconds"] = (
                                    time.monotonic() - started
                                )
                            item["answer"] += token
                    else:
                        item["answer"] = response.get("content") or ""
                    if not item["answer"].strip():
                        raise ValueError("The final answer was empty")
                    item.update(
                        delivered=True,
                        references=result.get("data", {}).get("references", []),
                        native_data=result.get("data", {}),
                        native_metadata=result.get("metadata", {}),
                    )
                except Exception as exc:  # noqa: BLE001 -- retain every failed request in the denominator
                    item.update(
                        error_type=type(exc).__name__,
                        error=str(exc),
                        grade="delivery_failed",
                    )
                item.update(
                    total_seconds=time.monotonic() - started,
                    same_request_events=list(active),
                    provider_token_usage="not exposed by the current provider wrapper",
                )
                rows.append(item)
                save_json(destination / f"{repeat}-{case['id']}.json", item)
                save_json(
                    destination / "results.json",
                    {
                        **contract,
                        "variant": settings.run_id,
                        "workspace": settings.workspace,
                        "transport": "native SDK with the frozen answer/audit callback",
                        "expected_requests": 29 * repeats,
                        "results": rows,
                    },
                )
        report = {
            **contract,
            "variant": settings.run_id,
            "workspace": settings.workspace,
            "transport": "native SDK with the frozen answer/audit callback",
            "expected_requests": 29 * repeats,
            "requests": len(rows),
            "delivered": sum(row["delivered"] for row in rows),
            "business_acceptance": "pending independent original-source review",
            "results": rows,
        }
        save_json(destination / "results.json", report)
        save_json(
            destination / "review-template.json",
            {
                "question_set_sha256": report["question_set_sha256"],
                "reviewed_by": "",
                "method": "independent_original_source_review",
                "rows": [
                    {
                        "case_id": row["case_id"],
                        "repeat": row["repeat"],
                        "full_answer_passed": None,
                        "core_passed": None,
                        "scope_passed": None,
                        "conditions_passed": None,
                        "citations_passed": None,
                        "extra_business_error": None,
                        "high_risk_error": None,
                        "source_references": [],
                        "notes": "",
                    }
                    for row in rows
                ],
            },
        )
        return {
            "results": str(destination / "results.json"),
            "requests": len(rows),
            "delivered": report["delivered"],
            "business_acceptance": report["business_acceptance"],
        }
    finally:
        if initialized:
            await rag.finalize_storages()


def compare(settings: Settings, a_file: Path, b_file: Path) -> dict:
    a, b = (read_json(inside(settings.root, path)) for path in [a_file, b_file])
    for field in ["question_set_sha256", "source_set_sha256", "query_contract_sha256"]:
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError(f"Frozen A/B control differs or is missing: {field}")
    if a.get("extraction_contract_sha256") != b.get("extraction_contract_sha256"):
        raise ValueError("Frozen A/B extraction control differs")
    a_rows, b_rows = (complete_rounds(settings, value["results"]) for value in [a, b])
    if a_rows.keys() != b_rows.keys():
        raise ValueError("The two variants have unmatched or duplicate question rounds")
    expected = {row["id"]: row["question"] for row in questions(settings)}
    rows = []
    for identifier in sorted(a_rows):
        left, right = a_rows[identifier], b_rows[identifier]
        if (
            left["question"] != right["question"]
            or left["question"] != expected[identifier[0]]
        ):
            raise ValueError("A question ID was reused for different original text")
        rows.append(
            {
                "case_id": identifier[0],
                "repeat": identifier[1],
                "a_grade": left["grade"],
                "b_grade": right["grade"],
                "a_delivered": left["delivered"],
                "b_delivered": right["delivered"],
            }
        )
    bins = {
        "both_passed": 0,
        "a_failed_b_passed": 0,
        "a_passed_b_failed": 0,
        "both_failed": 0,
        "pending_review": 0,
    }
    for row in rows:
        if row["a_grade"] not in {"passed", "failed", "delivery_failed"} or row[
            "b_grade"
        ] not in {"passed", "failed", "delivery_failed"}:
            bins["pending_review"] += 1
        else:
            left, right = row["a_grade"] == "passed", row["b_grade"] == "passed"
            bins[
                "both_passed"
                if left and right
                else "a_failed_b_passed"
                if right
                else "a_passed_b_failed"
                if left
                else "both_failed"
            ] += 1
    report = {
        "paired_requests": len(rows),
        "counts": bins,
        "rows": rows,
        "latency_comparable": a.get("transport") == b.get("transport"),
        "business_conclusion": "Answer comparisons require independent grades. KG evidence/scope acceptance remains separate; no automatic cutover.",
    }
    save_json(settings.run_dir / "ab-comparison.json", report)
    return report


def score(settings: Settings, results_file: Path, review_file: Path) -> dict:
    results_path = inside(settings.root, results_file)
    result = read_json(results_path)
    review = read_json(inside(settings.root, review_file))
    result_rows = complete_rounds(settings, result["results"])
    if (
        not review.get("reviewed_by")
        or review.get("method") != "independent_original_source_review"
        or review.get("question_set_sha256") != result.get("question_set_sha256")
    ):
        raise ValueError(
            "An identified independent original-source review of the same questions is required"
        )
    key = lambda row: (row["case_id"], row["repeat"])
    grades = {key(row): row for row in review["rows"]}
    if len(grades) != len(review["rows"]) or grades.keys() != result_rows.keys():
        raise ValueError("Independent review rounds are missing or duplicated")
    fields = [
        "full_answer_passed",
        "core_passed",
        "scope_passed",
        "conditions_passed",
        "citations_passed",
        "extra_business_error",
        "high_risk_error",
    ]
    for row in result["results"]:
        grade = grades[key(row)]
        if any(type(grade.get(field)) is not bool for field in fields):
            raise ValueError("Every independent grade must be an explicit boolean")
        if grade["full_answer_passed"] and (
            not row["delivered"]
            or not all(
                grade[field]
                for field in [
                    "core_passed",
                    "scope_passed",
                    "conditions_passed",
                    "citations_passed",
                ]
            )
            or grade["extra_business_error"]
            or grade["high_risk_error"]
            or not grade.get("source_references")
        ):
            raise ValueError(
                "A full-answer pass conflicts with delivery, supporting evidence or business errors"
            )
        row.update(
            grade="passed"
            if grade["full_answer_passed"]
            else "failed"
            if row["delivered"]
            else "delivery_failed",
            independent_review=grade,
        )
    count = len(result["results"])
    passed = sum(row["grade"] == "passed" for row in result["results"])
    result.update(
        business_acceptance="independently reviewed sample; KG acceptance remains separate",
        reviewed_by=review["reviewed_by"],
        review_summary={
            "requests": count,
            "passed": passed,
            "pass_rate": passed / count if count else 0,
            "high_risk_errors": sum(
                row["independent_review"]["high_risk_error"]
                for row in result["results"]
            ),
        },
    )
    destination = results_path.with_name("reviewed-results.json")
    save_json(destination, result)
    return {"reviewed_results": str(destination), **result["review_summary"]}
