"""Offline reviewed-reference compiler. No graph mutation, models, or network.

sources: [{id, product, text, sha256 (UTF-8 text)}]
units: [{id, source_id, kind: section|definition, entity_id, span,
         term_span (definitions), reference_spans (optional), scope_id (optional)}]
scopes: [{id, source_id, owner_entity_id, span, heading_span}]
All spans: {start,end,quote}, half-open Unicode offsets in original source text.
The caller reviews semantic entity bindings, complete reference terms, and scope
ownership. Integrity is verified here, not the truth of those reviewed claims.
Inventory maps exact existing graph IDs to reviewed product identities.
Malformed/stale catalogs fail atomically. Unresolved links remain diagnostics.
Output candidates use native extraction keys with separate audit provenance.
Provenance support_spans reproduce the description excerpts; evidence additionally
records exact reference spans or full structural containment proof. Keywords label
only definition citation or structural ownership, never insurance applicability.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping

MARKER = "(见释义)"


def _normalized(text):
    chars, offsets = [], []
    for index, char in enumerate(text):
        # Width forms only: compatibility symbols must not become aliases.
        value = (
            unicodedata.normalize("NFKC", char)
            if (0xFF01 <= ord(char) <= 0xFF5E or char == "\u3000")
            else char
        )
        for item in value:
            if not item.isspace():
                chars.append(item)
                offsets.append(index)
    return "".join(chars), offsets


def _index(rows, label):
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Missing or duplicate {label} identifier")
        result[key] = row
    return result


def _span(source, span):
    a, b, quote = span.get("start"), span.get("end"), span.get("quote")
    if (
        type(a) is not int
        or type(b) is not int
        or not 0 <= a < b <= len(source["text"])
        or not isinstance(quote, str)
        or not quote.strip()
        or source["text"][a:b] != quote
    ):
        raise ValueError("Stale or invalid exact source span")
    return {
        "source_id": source["id"],
        "sha256": source["sha256"],
        "start": a,
        "end": b,
        "quote": quote,
    }


def _contains(outer, inner):
    return outer["start"] <= inner["start"] < inner["end"] <= outer["end"]


def compile_catalog(catalog: dict, existing_entity_ids: Mapping[str, str]) -> dict:
    """Compile reviewed native-shaped relationship candidates without side effects."""
    if not isinstance(existing_entity_ids, Mapping):
        raise ValueError("Inventory must map exact entity IDs to products")
    sources = _index(catalog.get("sources", []), "source")
    units = _index(catalog.get("units", []), "unit")
    scopes = _index(catalog.get("scopes", []), "scope")
    for source in sources.values():
        if (
            not isinstance(source.get("text"), str)
            or not isinstance(source.get("product"), str)
            or not source["product"].strip()
            or hashlib.sha256(source["text"].encode()).hexdigest()
            != source.get("sha256")
        ):
            raise ValueError("Missing product identity or stale source hash")

    def source_for(row):
        if row.get("source_id") not in sources:
            raise ValueError("Unknown source")
        return sources[row["source_id"]]

    def entity_valid(key, source):
        return (
            isinstance(key, str)
            and bool(key)
            and existing_entity_ids.get(key) == source["product"]
        )

    for unit in units.values():
        source = source_for(unit)
        _span(source, unit["span"])
        if unit.get("kind") not in ("section", "definition"):
            raise ValueError("Unknown unit kind")
        if not entity_valid(unit.get("entity_id"), source):
            raise ValueError("Missing or cross-product graph entity")
        if unit["kind"] == "definition":
            _span(source, unit["term_span"])
            if not _contains(unit["span"], unit["term_span"]):
                raise ValueError("Definition title outside definition")
        refs = unit.get("reference_spans", [])
        for ref in refs:
            _span(source, ref)
            if not _contains(unit["span"], ref):
                raise ValueError("Reference outside unit")
        if len({(r["start"], r["end"]) for r in refs}) != len(refs):
            raise ValueError("Duplicate reference span")
    for scope in scopes.values():
        source = source_for(scope)
        _span(source, scope["span"])
        _span(source, scope["heading_span"])

    relationships, provenance, diagnostics = [], [], []

    def diagnostic(code, unit, **details):
        diagnostics.append({"code": code, "unit_id": unit["id"], **details})

    def emit(src, tgt, kind, evidence, unit, support_spans):
        if src == tgt:
            diagnostic("self_link_refused", unit, kind=kind)
            return
        relationships.append(
            {
                "source": src,
                "target": tgt,
                "keywords": {
                    "explicit_definition_reference": "引用释义",
                    "reviewed_scope_ownership": "原文范围归属",
                }[kind],
                "description": "\n\n".join(
                    dict.fromkeys(item["quote"] for item in support_spans)
                ),
            }
        )
        provenance.append(
            {
                "relationship_index": len(relationships) - 1,
                "unit_id": unit["id"],
                "kind": kind,
                "evidence": evidence,
                "support_spans": support_spans,
            }
        )

    for unit in units.values():
        source, body = source_for(unit), unit["span"]
        normalized, offsets = _normalized(body["quote"])
        markers = {
            (
                body["start"] + offsets[m.start()],
                body["start"] + offsets[m.end() - 1] + 1,
            )
            for m in re.finditer(re.escape(MARKER), normalized)
        }
        reviewed_markers = set()
        for ref in unit.get("reference_spans", []):
            text, positions = _normalized(ref["quote"])
            if (
                not text.endswith(MARKER)
                or text.count(MARKER) != 1
                or len(text) == len(MARKER)
            ):
                diagnostic(
                    "not_explicit_definition_reference",
                    unit,
                    evidence=_span(source, ref),
                )
                continue
            term = text[: -len(MARKER)]
            marker = (
                ref["start"] + positions[len(term)],
                ref["start"] + positions[-1] + 1,
            )
            if marker in reviewed_markers:
                raise ValueError("Multiple reviewed terms for one marker")
            reviewed_markers.add(marker)
            targets = [
                candidate
                for candidate in units.values()
                if candidate["kind"] == "definition"
                and candidate["source_id"] == unit["source_id"]
                and _normalized(candidate["term_span"]["quote"])[0] == term
            ]
            if len(targets) != 1:
                diagnostic(
                    "ambiguous_definition" if targets else "missing_definition",
                    unit,
                    term=term,
                    evidence=_span(source, ref),
                    candidate_unit_ids=[t["id"] for t in targets],
                )
                continue
            target = targets[0]
            emit(
                unit["entity_id"],
                target["entity_id"],
                "explicit_definition_reference",
                [
                    _span(source, ref),
                    _span(source, body),
                    _span(source, target["span"]),
                ],
                unit,
                [_span(source, body), _span(source, target["span"])],
            )
        for start, end in sorted(markers - reviewed_markers):
            diagnostic(
                "unreviewed_definition_reference",
                unit,
                evidence=_span(
                    source,
                    {"start": start, "end": end, "quote": source["text"][start:end]},
                ),
            )
        if unit.get("scope_id") is not None:
            scope = scopes.get(unit["scope_id"])
            if (
                scope is None
                or scope["source_id"] != unit["source_id"]
                or not entity_valid(scope.get("owner_entity_id"), source)
                or not _contains(scope["span"], body)
                or not _contains(scope["span"], scope["heading_span"])
                or scope["heading_span"]["end"] > body["start"]
            ):
                diagnostic("unproven_scope", unit, scope_id=unit["scope_id"])
            else:
                emit(
                    scope["owner_entity_id"],
                    unit["entity_id"],
                    "reviewed_scope_ownership",
                    [
                        _span(source, scope["heading_span"]),
                        _span(source, body),
                        _span(source, scope["span"]),
                    ],
                    unit,
                    [_span(source, scope["heading_span"]), _span(source, body)],
                )
    return {
        "entities": [],
        "relationships": relationships,
        "provenance": provenance,
        "diagnostics": diagnostics,
        "offset_unit": "Unicode code points; half-open; unmodified source text",
    }


def resolve_support_chunks(
    support_spans: list[dict],
    sources: Mapping[str, str],
    native_chunks: Mapping[str, dict],
) -> list[str]:
    """Resolve exact excerpts to a deterministic greedy native chunk cover.

    Source paths must match exactly, not by basename or suffix. Each support
    excerpt must occur literally in a chunk from its declared source. This is
    a narrow greedy cover, not a claim of global minimum. No fuzzy joins,
    invented chunks, or ungrounded fallback attribution are permitted.
    """
    if not support_spans:
        raise ValueError("At least one support span is required")
    coverage = {}
    for index, span in enumerate(support_spans):
        source_id, quote = span.get("source_id"), span.get("quote")
        path = sources.get(source_id)
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(quote, str)
            or not quote.strip()
        ):
            raise ValueError("Missing support source path or exact quote")
        for chunk_id, chunk in native_chunks.items():
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Native chunk IDs must be nonempty strings")
            if (
                chunk.get("file_path") == path
                and isinstance(chunk.get("content"), str)
                and quote in chunk["content"]
            ):
                coverage.setdefault(chunk_id, set()).add(index)
    uncovered = set(range(len(support_spans)))
    if set().union(*coverage.values()) != uncovered:
        raise ValueError("Support excerpt missing from its declared native source")
    selected = []
    while uncovered:
        winner = min(coverage, key=lambda key: (-len(coverage[key] & uncovered), key))
        newly_covered = coverage[winner] & uncovered
        if not newly_covered:
            raise ValueError("Cannot cover every support excerpt")
        selected.append(winner)
        uncovered -= newly_covered
    return selected
