"""Offline, extract-only answer gate over an independently pinned evidence catalog.

Hashes prove identity, not review authenticity or semantic entailment. The trusted
caller owns catalog review, source/index consistency, current scope and completion
metadata. No candidate text, review assertion or confidence can authorize prose.
"""

from __future__ import annotations

import json
from typing import Any

from product.evidence_trial import ID, assemble, digest, validate_catalog

CONTRACT_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 200_000
MAX_CLAIMS = 1000
EXTRACT_NOTICE = (
    "Source excerpts only. Semantic review was not performed. These excerpts "
    "are not a complete answer or a claim-payment decision."
)
BLOCKED_NOTICE = (
    "Publication blocked. The evidence or answer contract could not be verified. "
    "No answer or partial excerpts are released."
)


class ContractError(ValueError):
    """A fixed diagnostic code, never a copy of candidate-supplied content."""


def _fail(code: str) -> None:
    raise ContractError(code)


def _json_copy(value: Any) -> Any:
    """Accept bounded plain JSON values and detach caller-owned mutable objects."""
    nodes = 0
    minimum_bytes = 0

    def charge(size: int) -> None:
        nonlocal minimum_bytes
        minimum_bytes += size
        if minimum_bytes > MAX_JSON_BYTES:
            _fail("json_size_limit")

    def charge_text(value: str) -> None:
        # UTF-8 cannot use fewer bytes than the number of code points. Reject
        # oversized strings/keys before constructing their encoded copies.
        if len(value) > MAX_JSON_BYTES - minimum_bytes:
            _fail("json_size_limit")
        try:
            charge(len(value.encode("utf-8")) + 2)
        except UnicodeError:
            raise ContractError("invalid_json") from None

    def check(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _fail("json_structure_limit")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _fail("invalid_json_key")
            charge(2 + max(0, len(item) - 1) + len(item))
            for key, child in item.items():
                charge_text(key)
                check(child, depth + 1)
        elif type(item) is list:
            charge(2 + max(0, len(item) - 1))
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            charge_text(item)
        elif type(item) is int:
            try:
                charge(len(str(item)))
            except ValueError:
                raise ContractError("invalid_json") from None
        elif type(item) is bool:
            charge(4 if item else 5)
        elif item is None:
            charge(4)
        else:
            _fail("invalid_json_value")

    check(value, 0)
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > MAX_JSON_BYTES:
            _fail("json_size_limit")
        return json.loads(encoded)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid_json") from None


def strict_json_loads(raw: str) -> Any:
    """Parse one bounded JSON value, rejecting duplicate keys and trailing data."""
    if type(raw) is not str:
        _fail("invalid_json")
    try:
        if len(raw) > MAX_JSON_BYTES or len(raw.encode("utf-8")) > MAX_JSON_BYTES:
            _fail("json_size_limit")

        def pairs(items: list[tuple[str, Any]]) -> dict:
            result = {}
            for key, value in items:
                if key in result:
                    _fail("duplicate_json_key")
                result[key] = value
            return result

        def constant(_value: str) -> None:
            _fail("invalid_json_constant")

        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid_json") from None
    return _json_copy(value)


def _fields(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if (
        type(value) is not dict
        or not required <= value.keys()
        or not value.keys() <= required | (optional or set())
    ):
        _fail("invalid_fields")


def _list(value: Any, *, nonempty: bool = False) -> list:
    if type(value) is not list or (nonempty and not value):
        _fail("invalid_list")
    return value


def _strings(value: Any, *, nonempty: bool = True) -> list[str]:
    items = _list(value, nonempty=nonempty)
    if any(type(item) is not str or not item for item in items):
        _fail("invalid_identifiers")
    if len(items) != len(set(items)):
        _fail("duplicate_identifier")
    return items


def _hash(value: Any) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        _fail("invalid_sha256")


def _catalog(catalog: dict, expected_catalog_sha256: str) -> None:
    _hash(expected_catalog_sha256)
    _fields(
        catalog,
        {
            "schema_version",
            "snapshot_sha256",
            "manifest_sha256",
            "parser",
            "coverage",
            "documents",
        },
    )
    if type(catalog["schema_version"]) is not int:
        _fail("invalid_catalog_schema")
    if catalog["snapshot_sha256"] != expected_catalog_sha256:
        _fail("untrusted_catalog_snapshot")
    if catalog["coverage"] != "explicit_reviewed_units_only":
        _fail("invalid_catalog_coverage")
    _fields(catalog["parser"], {"entrypoint", "pypdf_version"})
    for doc in _list(catalog["documents"], nonempty=True):
        _fields(
            doc,
            {
                "document_ref",
                "product_ref",
                "file_path",
                "source_sha256",
                "extracted_text_sha256",
                "units",
            },
        )
        for unit in _list(doc["units"]):
            _fields(
                unit,
                {
                    "id",
                    "title",
                    "start",
                    "end",
                    "text",
                    "text_sha256",
                    "anchors",
                    "requires",
                    "physical_pages",
                    "scope",
                    "scope_status",
                },
            )
            scopes = _list(unit["scope"], nonempty=True)
            if len(scopes) != len({digest(row) for row in scopes}):
                _fail("duplicate_scope_tuple")
            _strings(unit["anchors"], nonempty=False)
    try:
        validate_catalog(catalog)
    except (ValueError, TypeError, KeyError):
        raise ContractError("invalid_catalog") from None


def _inputs(case: dict, retrieval: dict) -> None:
    _fields(
        case,
        {"question", "product_ref", "allowed_document_refs", "scope"},
        {"max_evidence_units", "max_depth", "max_context_chars", "exact_title_lookup"},
    )
    _strings(case["allowed_document_refs"])
    _fields(retrieval, {"status", "data"})
    if retrieval["status"] not in ("success", "error"):
        _fail("invalid_retrieval_status")
    _fields(retrieval["data"], {"chunks"})
    chunk_ids = []
    for chunk in _list(retrieval["data"]["chunks"]):
        _fields(chunk, {"chunk_id", "file_path", "content"})
        if any(type(chunk[key]) is not str or not chunk[key] for key in chunk):
            _fail("invalid_chunk")
        chunk_ids.append(chunk["chunk_id"])
    _strings(chunk_ids, nonempty=False)


def build_packet(
    catalog: dict,
    retrieval: dict,
    case: dict,
    *,
    expected_catalog_sha256: str,
) -> dict:
    """Rebuild a deterministic packet; never retain partial evidence when blocked.

    The expected catalog digest must come from an independent trusted approval
    record, not the supplied catalog or a generator. Inputs are already-loaded
    JSON objects; this function neither reads sources nor contacts a service.
    """
    catalog, retrieval, case = _json_copy([catalog, retrieval, case])
    _catalog(catalog, expected_catalog_sha256)
    _inputs(case, retrieval)
    try:
        result = assemble(catalog, retrieval, case)
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ContractError("invalid_assembly_inputs") from None
    # Even one rejected source/scope fails the entire strict packet. Do not
    # silently discard a bad branch and publish the remaining optimistic text.
    blockers = sorted({reason.split(":", 1)[0] for reason in result["blockers"]})
    if result["rejected"]:
        blockers.append("rejected_retrieval_evidence")
    selected = set(result["evidence_ids"])
    units = {
        unit["id"]: (doc, unit) for doc in catalog["documents"] for unit in doc["units"]
    }
    if any(
        dependency not in selected
        for identifier in selected
        for dependency in units[identifier][1]["requires"]
    ):
        blockers.append("incomplete_dependency_closure")
    evidence = []
    if not blockers:
        for identifier in result["evidence_ids"]:
            doc, unit = units[identifier]
            evidence.append(
                {
                    **unit,
                    **{key: value for key, value in doc.items() if key != "units"},
                }
            )
    packet = {
        "contract_version": CONTRACT_VERSION,
        "status": "blocked" if blockers else "ready",
        "catalog_snapshot_sha256": expected_catalog_sha256,
        "manifest_sha256": catalog["manifest_sha256"],
        "parser": catalog["parser"],
        "case": case,
        "case_sha256": digest(case),
        "retrieval_sha256": digest(retrieval),
        "seed_ids": [] if blockers else result["seed_ids"],
        "evidence_ids": [] if blockers else result["evidence_ids"],
        "evidence": evidence,
        "blockers": sorted(set(blockers)),
    }
    packet["packet_sha256"] = digest(packet)
    return _json_copy(packet)


def _claims(candidate: dict, packet: dict) -> list[dict]:
    _fields(candidate, {"contract_version", "packet_sha256", "claims"})
    if (
        type(candidate["contract_version"]) is not int
        or candidate["contract_version"] != CONTRACT_VERSION
        or candidate["packet_sha256"] != packet["packet_sha256"]
    ):
        _fail("stale_answer_contract")
    claims = _list(candidate["claims"], nonempty=True)
    if len(claims) > MAX_CLAIMS:
        _fail("claim_count_limit")
    evidence = {unit["id"]: unit for unit in packet["evidence"]}
    seen = set()
    checks = []
    for index, claim in enumerate(claims):
        _fields(claim, {"claim_id", "packet_sha256", "kind", "evidence_ids", "text"})
        identifier = claim["claim_id"]
        if type(identifier) is not str or not ID.fullmatch(identifier):
            _fail("invalid_claim_id")
        if identifier in seen:
            _fail("duplicate_claim_id")
        seen.add(identifier)
        if claim["packet_sha256"] != packet["packet_sha256"]:
            _fail("stale_claim_contract")
        if claim["kind"] not in ("quote", "paraphrase", "inference"):
            _fail("unsupported_claim_kind")
        ids = _strings(claim["evidence_ids"])
        if any(identifier not in evidence for identifier in ids):
            _fail("unknown_or_unselected_evidence")
        if type(claim["text"]) is not str or not claim["text"].strip():
            _fail("invalid_claim_text")
        if claim["kind"] == "quote":
            if len(ids) != 1 or claim["text"] != evidence[ids[0]]["text"]:
                _fail("quote_not_one_complete_unit")
            status = "exact_quote_verified"
        else:
            # Legal citations do not establish entailment. No keyword test,
            # model confidence or self-issued review can change this status.
            status = "pending_semantic_review"
        checks.append({"claim_index": index, "status": status})
    return checks


def gate_answer(
    packet: dict,
    candidate: dict | str | None = None,
    *,
    catalog: dict,
    retrieval: dict,
    case: dict,
    expected_catalog_sha256: str,
    completion_status: str = "unknown",
) -> dict:
    """Verify the whole answer before returning any public text.

    Rebuild against current trusted inputs instead of trusting packet self-hashes.
    With no candidate, report/extract only. For a candidate, the trusted caller
    must independently attest complete generation; unknown/truncated blocks.
    Raw candidate prose is deliberately absent even from the returned audit.
    """
    checks = []
    candidate_sha256 = None
    verified = None
    try:
        expected = build_packet(
            catalog,
            retrieval,
            case,
            expected_catalog_sha256=expected_catalog_sha256,
        )
        supplied = _json_copy(packet)
        # Canonical equality also distinguishes JSON booleans from integers.
        if digest(supplied) != digest(expected):
            _fail("stale_or_modified_packet")
        if expected["status"] != "ready":
            _fail("evidence_packet_blocked")
        if candidate is not None:
            if completion_status != "complete":
                _fail("generation_not_complete")
            candidate = (
                strict_json_loads(candidate)
                if type(candidate) is str
                else _json_copy(candidate)
            )
            candidate_sha256 = digest(candidate)
            checks = _claims(candidate, expected)
        verified = expected
        errors = []
    except ContractError as exc:
        errors = [str(exc)]
    except (ValueError, TypeError, KeyError, RecursionError):
        errors = ["invalid_contract"]
    # Never expose a partially validated claim, packet, error value or title.
    excerpts = []
    if verified is not None:
        for unit in verified["evidence"]:
            excerpts.append(
                {
                    "label": "Source excerpt",
                    "evidence_id": unit["id"],
                    "document_ref": unit["document_ref"],
                    "file_path": unit["file_path"],
                    "title": unit["title"],
                    "start": unit["start"],
                    "end": unit["end"],
                    "physical_pages": unit["physical_pages"],
                    "requires": unit["requires"],
                    "text_sha256": unit["text_sha256"],
                    "text": unit["text"],
                }
            )
    return {
        "status": "extracts_only" if verified is not None else "blocked",
        "public": {
            "notice": EXTRACT_NOTICE if verified is not None else BLOCKED_NOTICE,
            "excerpts": excerpts,
        },
        "audit": {
            "mechanical_errors": errors,
            "semantic_review": "not_performed",
            "claim_checks": checks if verified is not None else [],
            "candidate_sha256": candidate_sha256,
            "packet_sha256": verified["packet_sha256"] if verified else None,
            "free_text_released": False,
        },
    }
