"""Prepare reviewed, source-copied units for native LightRAG ingestion.

This is an offline compiler, not a second insurance rules database. It reuses
``prepare_context.extract_source`` and the native delimiter chunker. A manifest
is a human-reviewed partition of each extracted document. Hashes verify that
reviewed bytes, parser, offsets and metadata did not change; they do NOT prove
that a PDF was faithfully parsed or that a reviewer found every exception.

Manifest schema 1:
  {tokenizer_model, max_chunk_tokens, documents: [
    {source_id, document_version, path, original_sha256, extracted_text_sha256,
     parser_binding, product: {id, label, identity_span}, units: [
       {unit_id, span: {start,end,quote}, disposition:
          primary|context_only|excluded, reason: <required unless primary>,
        scope: {combinations: [{plan: {kind,value?},
          responsibility: {kind,value?}, evidence: [{start,end,quote}]}]},
        review: {complete_conditions: true, dependencies_complete: true},
        dependencies: [{unit_id, reason: definition|condition|exception|scope,
                        reference_span: {start,end,quote}}]}]}]}

Scope kinds are specific (nonempty value), universal, not_applicable, or
unknown. Unknown is recorded as a blocker, never interpreted as universal.
Every combination is explicit; dimension arrays are refused. Scope evidence
supports the caller's review, not an automatically inferred insurance claim.
Dependencies are explicit, same-source and same-document-version only and may
point forward. Their transitive closure is bounded, never silently truncated.
``inspect_source`` only supplies unapproved review inputs. ``prepare_import``
returns an in-memory artifact or raises ``PreparationError`` with a report;
it never writes, imports, approves extraction, or calls an inference provider.
For fully offline execution, preseed the native tiktoken encoding cache; an
uncached native tokenizer may otherwise download its public encoding data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from pypdf.errors import PyPdfError

from lightrag.chunker import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from product import prepare_context

SCHEMA_VERSION = 1
_SCOPE_KINDS = {"specific", "universal", "not_applicable"}
_DEPENDENCY_KINDS = {"definition", "condition", "exception", "scope"}


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON, without granting any semantic approval."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PreparationError(ValueError):
    """No importable output was produced; report retains review dispositions."""

    def __init__(self, code: str, message: str, report: dict):
        super().__init__(message)
        self.code = code
        self.report = {
            **report,
            "status": "blocked",
            "blocker": {
                "code": code,
                "message": message,
            },
        }


def parser_binding(path: Path) -> dict:
    """Pin the actual reused extractor implementation and relevant runtime."""
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise ValueError("Unsupported source format")
    binding = {
        "extractor": "product.prepare_context.extract_source",
        "implementation_sha256": _sha(Path(prepare_context.__file__).read_bytes()),
        "format": suffix,
        "python": platform.python_version(),
        "pypdf": importlib.metadata.version("pypdf") if suffix == ".pdf" else None,
        "offset_unit": "Unicode code points in unmodified extracted text",
    }
    return {**binding, "sha256": canonical_sha256(binding)}


def inspect_source(path: Path, source_id: str, document_version: str) -> dict:
    """Read a source for review. Returned pins and text are NOT an approval."""
    binding = parser_binding(path)
    original = _sha(path.read_bytes())
    text, pages = prepare_context.extract_source(path)
    if original != _sha(path.read_bytes()) or binding != parser_binding(path):
        raise ValueError("Source or parser changed during inspection")
    return {
        "source_id": source_id,
        "document_version": document_version,
        "original_sha256": original,
        "extracted_text_sha256": _sha(text.encode("utf-8")),
        "parser_binding": binding,
        "text": text,
        "pages": pages,
    }


def prepare_import(manifest: dict, root: Path, tokenizer=None) -> dict:
    """Validate all sources before returning complete, indivisible native units.

    An injected tokenizer must be the caller's actual native runtime tokenizer.
    Production callers must bind its configuration independently at approval and
    import time. The default uses LightRAG's ``TiktokenTokenizer``. The report
    records which path was used; an injected fake is only suitable for tests.
    """
    report: dict = {
        "status": "preparing",
        "coverage": [],
        "units": [],
        "semantic_completeness": (
            "Caller-reviewed assertions only. Hashes, exact spans and character "
            "coverage do not establish parsing fidelity or semantic completeness."
        ),
        "cross_file_composites": "unsupported; each source is still validated",
    }

    def fail(code: str, message: str) -> None:
        raise PreparationError(code, message, report)

    def nonempty(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            fail("invalid_identity", f"Missing or invalid {label}")
        if any(ord(char) < 32 for char in value):
            fail("invalid_identity", f"Control characters in {label}")
        return value

    def exact(source: dict, span: Any) -> dict:
        if not isinstance(span, dict):
            fail("invalid_span", "An exact source span is required")
        a, b, quote = span.get("start"), span.get("end"), span.get("quote")
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b <= len(source["text"])
            or not isinstance(quote, str)
            or not quote
            or source["text"][a:b] != quote
        ):
            fail("stale_span", "Stale or invalid exact source span")
        identity = {
            "source_id": source["source_id"],
            "document_version": source["document_version"],
            "original_sha256": source["original_sha256"],
            "extracted_text_sha256": source["extracted_text_sha256"],
            "parser_sha256": source["parser_binding"]["sha256"],
            "start": a,
            "end": b,
        }
        return {
            "evidence_id": "evidence-" + canonical_sha256(identity),
            **identity,
            "quote": quote,
            "physical_pages": [
                i + 1 for i, (x, y) in enumerate(source["pages"]) if x < b and y > a
            ]
            if source["parser_binding"]["format"] == ".pdf"
            else [],
        }

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        fail("invalid_manifest", "Unsupported manifest version")
    model = nonempty(manifest.get("tokenizer_model"), "tokenizer_model")
    limit = manifest.get("max_chunk_tokens")
    dep_limit = manifest.get("max_dependencies", 16)
    depth_limit = manifest.get("max_dependency_depth", 8)
    for value, maximum, label in (
        (limit, 8192, "chunk token budget"),
        (dep_limit, 64, "dependency count budget"),
        (depth_limit, 32, "dependency depth budget"),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            fail("invalid_budget", f"Invalid {label}")
    records = manifest.get("documents")
    if not isinstance(records, list) or not records:
        fail("invalid_manifest", "No documents specified")
    root = root.resolve()
    sources, units, paths, identities = [], {}, set(), set()
    product_labels: dict[str, str] = {}
    product_ids: dict[str, str] = {}

    # Check every source even if a later dependency is an unsupported composite.
    for record in records:
        if not isinstance(record, dict):
            fail("invalid_manifest", "Invalid document record")
        source_id = nonempty(record.get("source_id"), "source_id")
        version = nonempty(record.get("document_version"), "document_version")
        relative = nonempty(record.get("path"), "source path")
        path = (root / relative).resolve()
        identity = (source_id, version)
        if (
            Path(relative).is_absolute()
            or not path.is_relative_to(root)
            or path in paths
            or identity in identities
        ):
            fail(
                "ambiguous_source",
                "Source identity and path must be unique within root",
            )
        paths.add(path)
        identities.add(identity)
        try:
            source = inspect_source(path, source_id, version)
        except (ValueError, OSError, PyPdfError) as exc:
            fail("source_unreadable", str(exc))
        for key in ("original_sha256", "extracted_text_sha256", "parser_binding"):
            if source[key] != record.get(key):
                fail("stale_source", f"Source or parser changed: {source_id} ({key})")
        if not source["text"].strip():
            fail("empty_source", "Source has no readable text")
        if prepare_context.DELIMITER in source["text"]:
            fail("reserved_delimiter", "Reserved native delimiter occurs in source")
        product = record.get("product")
        if not isinstance(product, dict):
            fail("invalid_identity", "Reviewed product identity is required")
        product_id = nonempty(product.get("id"), "product id")
        label = nonempty(product.get("label"), "product label")
        identity_span = exact(source, product.get("identity_span"))
        if label not in identity_span["quote"]:
            fail(
                "invalid_identity",
                "Product label is absent from reviewed identity span",
            )
        if (
            product_labels.get(product_id, label) != label
            or product_ids.get(label, product_id) != product_id
        ):
            fail("ambiguous_product", "Conflicting reviewed product identity")
        product_labels[product_id], product_ids[label] = label, product_id
        source.update(
            {
                "product": {
                    "id": product_id,
                    "label": label,
                    "identity_span": identity_span,
                },
                "path": relative,
                "_path": path,
                "_unit_ids": [],
            }
        )
        rows = record.get("units")
        if not isinstance(rows, list) or not rows:
            fail("invalid_units", "Reviewed full-source partition is required")
        cursor = 0
        coverage = {
            "source_id": source_id,
            "document_version": version,
            "extracted_characters": len(source["text"]),
            "spans": [],
        }
        report["coverage"].append(coverage)
        for row in rows:
            if not isinstance(row, dict):
                fail("invalid_units", "Invalid unit record")
            unit_id = nonempty(row.get("unit_id"), "unit_id")
            if unit_id in units:
                fail("ambiguous_unit", "Duplicate reviewed unit identifier")
            span = exact(source, row.get("span"))
            if span["start"] != cursor:
                fail(
                    "incomplete_partition",
                    "Reviewed spans must partition source in order",
                )
            cursor = span["end"]
            disposition = row.get("disposition")
            if disposition not in {"primary", "context_only", "excluded"}:
                fail("invalid_disposition", "Unit disposition must be explicit")
            if disposition != "primary":
                nonempty(row.get("reason"), "context/exclusion reason")
            record_report = {
                "unit_id": unit_id,
                "evidence_id": span["evidence_id"],
                "disposition": disposition,
                "start": span["start"],
                "end": span["end"],
                "reason": row.get("reason"),
            }
            coverage["spans"].append(record_report)
            report["units"].append({**record_report, "status": "pending"})
            units[unit_id] = {
                **row,
                "_source": source,
                "_span": span,
                "_report": report["units"][-1],
            }
            source["_unit_ids"].append(unit_id)
        if cursor != len(source["text"]):
            fail(
                "incomplete_partition", "Reviewed partition omits trailing source text"
            )
        coverage["character_partition"] = "complete, non-overlapping"
        sources.append(source)

    def checked_scope(unit: dict) -> dict:
        scope = unit.get("scope")
        if not isinstance(scope, dict) or set(scope) != {"combinations"}:
            fail(
                "invalid_scope", "Use explicit scope combinations, not dimension arrays"
            )
        combos = scope["combinations"]
        if not isinstance(combos, list) or not combos:
            fail("unknown_scope", "Scope combinations are unknown or empty")
        checked, seen = [], set()
        for combo in combos:
            if not isinstance(combo, dict) or set(combo) != {
                "plan",
                "responsibility",
                "evidence",
            }:
                fail(
                    "invalid_scope",
                    "Each scope combination needs plan, responsibility and evidence",
                )
            pair = {}
            for dimension in ("plan", "responsibility"):
                value = combo[dimension]
                if not isinstance(value, dict) or value.get("kind") == "unknown":
                    fail(
                        "unknown_scope",
                        f"Unknown {dimension} scope in {unit['unit_id']}",
                    )
                kind = value.get("kind")
                keys = {"kind", "value"} if kind == "specific" else {"kind"}
                if kind not in _SCOPE_KINDS or set(value) != keys:
                    fail("invalid_scope", "Invalid explicit scope dimension")
                if kind == "specific":
                    nonempty(value["value"], "specific scope value")
                pair[dimension] = dict(value)
            pair_id = canonical_sha256(pair)
            if pair_id in seen:
                fail("ambiguous_scope", "Duplicate explicit scope combination")
            seen.add(pair_id)
            evidence = combo["evidence"]
            if not isinstance(evidence, list) or not evidence:
                fail(
                    "unsupported_scope", "Scope requires exact reviewed source evidence"
                )
            checked.append(
                {
                    **pair,
                    "evidence": [exact(unit["_source"], item) for item in evidence],
                }
            )
        return {"combinations": checked}

    # No graph assertion is manufactured for excluded/context-only source prose.
    for unit in units.values():
        unit["_dependencies"] = []
        if unit["disposition"] == "excluded":
            if unit.get("dependencies"):
                fail("excluded_dependency", "Excluded units cannot carry dependencies")
            unit["_report"]["status"] = "excluded_by_review"
            continue
        review = unit.get("review")
        if not isinstance(review, dict) or any(
            review.get(key) is not True
            for key in ("complete_conditions", "dependencies_complete")
        ):
            fail(
                "unreviewed_unit",
                "Complete conditions and dependencies need explicit review",
            )
        unit["_scope"] = checked_scope(unit)
        dependencies = unit.get("dependencies")
        if not isinstance(dependencies, list):
            fail(
                "unknown_dependencies", "Explicit reviewed dependency list is required"
            )
        seen = set()
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                fail("invalid_dependency", "Invalid dependency record")
            target_id = dependency.get("unit_id")
            if not isinstance(target_id, str) or target_id not in units:
                fail("missing_dependency", "Missing or unknown dependency unit")
            if target_id in seen:
                fail("ambiguous_dependency", "Repeated dependency identifier")
            seen.add(target_id)
            target = units[target_id]
            if target["disposition"] == "excluded":
                fail("excluded_dependency", "Required dependency is excluded")
            a, b = unit["_source"], target["_source"]
            if a["product"]["id"] != b["product"]["id"]:
                fail(
                    "cross_product_dependency", "Cross-product dependency is forbidden"
                )
            if a["source_id"] != b["source_id"]:
                fail("cross_source_dependency", "Cross-file composites are unsupported")
            if a["document_version"] != b["document_version"]:
                fail(
                    "cross_version_dependency",
                    "Cross-document-version dependency is forbidden",
                )
            if dependency.get("reason") not in _DEPENDENCY_KINDS:
                fail("invalid_dependency", "Dependency reason must be explicit")
            reference = exact(a, dependency.get("reference_span"))
            if not (
                unit["_span"]["start"]
                <= reference["start"]
                < reference["end"]
                <= unit["_span"]["end"]
            ):
                fail(
                    "invalid_dependency",
                    "Dependency reference is outside its referring unit",
                )
            unit["_dependencies"].append(
                {
                    "unit_id": target_id,
                    "reason": dependency["reason"],
                    "reference_span": reference,
                }
            )

    def supports(target: dict, origin: dict, reason: str) -> bool:
        # Definitions can be explicitly scope-neutral without pretending they
        # apply universally. Business conditions cannot borrow that exception.
        return all(
            target[key]["kind"] == "universal"
            or target[key] == origin[key]
            or (
                target[key]["kind"] == "not_applicable"
                and reason in {"definition", "scope"}
            )
            for key in ("plan", "responsibility")
        )

    for unit in units.values():
        for edge in unit["_dependencies"]:
            target = units[edge["unit_id"]]
            if not all(
                any(
                    supports(candidate, origin, edge["reason"])
                    for candidate in target["_scope"]["combinations"]
                )
                for origin in unit["_scope"]["combinations"]
            ):
                fail(
                    "cross_scope_dependency",
                    "Dependency does not cover the referring scope combinations",
                )

    memo: dict[str, tuple[list[str], list[dict], int]] = {}

    def closure(unit_id: str, active: frozenset[str] = frozenset()):
        if unit_id in active:
            fail("dependency_cycle", "Cyclic reviewed dependencies")
        if len(active) > depth_limit:
            fail("dependency_depth", "Necessary dependency depth exceeds budget")
        if unit_id in memo:
            return memo[unit_id]
        found, edges, height = set(), {}, 0
        for edge in units[unit_id]["_dependencies"]:
            target = edge["unit_id"]
            descendants, child_edges, child_height = closure(target, active | {unit_id})
            found.update([target, *descendants])
            if len(found) > dep_limit:
                fail("dependency_count", "Necessary dependencies exceed count budget")
            for item in [{"from_unit_id": unit_id, **edge}, *child_edges]:
                edges[canonical_sha256(item)] = item
            height = max(height, child_height + 1)
            if height > depth_limit:
                fail("dependency_depth", "Necessary dependency depth exceeds budget")
        result = (
            sorted(found, key=lambda key: units[key]["_span"]["start"]),
            list(edges.values()),
            height,
        )
        memo[unit_id] = result
        return result

    # Check unused context units too, so cycles/unknown scopes do not hide there.
    closures = {
        key: closure(key)
        for key, unit in units.items()
        if unit["disposition"] != "excluded"
    }
    injected = tokenizer is not None
    tokenizer = tokenizer if injected else TiktokenTokenizer(model_name=model)
    documents, context_uses = [], {key: [] for key in units}
    for source in sources:
        output_units = []
        for key in source["_unit_ids"]:
            unit = units[key]
            if unit["disposition"] != "primary":
                continue
            dependency_ids, dependency_edges, _ = closures[key]
            primary = [{**unit["_span"], "role": "primary"}]
            context = [
                {**units[dep]["_span"], "role": "context", "unit_id": dep}
                for dep in dependency_ids
            ]
            for dep in dependency_ids:
                context_uses[dep].append(key)
            header = {
                "source_id": source["source_id"],
                "document_version": source["document_version"],
                "unit_id": key,
                "evidence_id": unit["_span"]["evidence_id"],
                "file_path": source["path"],
                "product": {k: source["product"][k] for k in ("id", "label")},
                "reviewed_scope_combinations": [
                    {k: combo[k] for k in ("plan", "responsibility")}
                    for combo in unit["_scope"]["combinations"]
                ],
            }
            content = (
                "[Verified source metadata; not additional insurance terms]\n"
                + json.dumps(header, ensure_ascii=False, sort_keys=True)
                + "\n[Primary original source]\n"
                + primary[0]["quote"]
                + "\n[End primary original source]"
            )
            for span in context:
                content += (
                    "\n[Copied necessary context; not a new primary fact; evidence_id="
                    + span["evidence_id"]
                    + f"; offsets={span['start']}:{span['end']}]\n"
                    + span["quote"]
                    + "\n[End copied necessary context]"
                )
            # Scope support may sit outside dependencies. Keep that exact support
            # retrievable without silently declaring its entire unit necessary.
            scope_evidence = {
                source["product"]["identity_span"]["evidence_id"]: (
                    source["product"]["identity_span"],
                    "product_identity",
                )
            }
            for combo in unit["_scope"]["combinations"]:
                for span in combo["evidence"]:
                    scope_evidence[span["evidence_id"]] = (span, "scope")
            for span, reason in sorted(
                scope_evidence.values(),
                key=lambda item: (item[0]["start"], -item[0]["end"]),
            ):
                if any(
                    other["start"] <= span["start"] < span["end"] <= other["end"]
                    for other in primary + context
                ):
                    continue
                owners = [
                    other
                    for other in source["_unit_ids"]
                    if units[other]["_span"]["start"]
                    <= span["start"]
                    < span["end"]
                    <= units[other]["_span"]["end"]
                ]
                if not owners or units[owners[0]]["disposition"] == "excluded":
                    fail(
                        "excluded_scope_evidence",
                        "Scope/identity support must be within one non-excluded reviewed unit",
                    )
                owner = owners[0]
                if key not in context_uses[owner]:
                    context_uses[owner].append(key)
                context.append(
                    {**span, "role": "context", "reason": reason, "unit_id": owner}
                )
                content += (
                    "\n[Copied reviewed identity/scope evidence; not a new primary fact; evidence_id="
                    + span["evidence_id"]
                    + "]\n"
                    + span["quote"]
                    + "\n[End copied reviewed identity/scope evidence]"
                )
            tokens = len(tokenizer.encode(content))
            unit["_report"].update(
                {
                    "tokens": tokens,
                    "max_chunk_tokens": limit,
                    "dependency_unit_ids": dependency_ids,
                    "context_evidence_ids": [span["evidence_id"] for span in context],
                }
            )
            if tokens > limit:
                unit["_report"]["status"] = "blocked_oversized"
                fail(
                    "oversized_unit",
                    f"Complete unit {key} exceeds native token budget; no splitting",
                )
            output_units.append(
                {
                    "unit_id": key,
                    "evidence_id": unit["_span"]["evidence_id"],
                    "source_id": source["source_id"],
                    "document_version": source["document_version"],
                    "product": source["product"],
                    "scope": unit["_scope"],
                    "primary": primary,
                    "context": context,
                    "dependency_edges": dependency_edges,
                    "content": content,
                    "content_sha256": _sha(content.encode("utf-8")),
                    "tokens": tokens,
                }
            )
            unit["_report"]["status"] = "prepared_primary"
        if not output_units:
            continue
        payload = {
            "text": prepare_context.DELIMITER.join(
                row["content"] for row in output_units
            ),
            "file_source": source["path"],
            "chunking": {
                "strategy": "fixed_token",
                "params": {
                    "split_by_character": prepare_context.DELIMITER,
                    "split_by_character_only": True,
                    "chunk_token_size": limit,
                    "chunk_overlap_token_size": 0,
                },
            },
        }
        chunks = chunking_by_token_size(
            tokenizer, payload["text"], **payload["chunking"]["params"]
        )
        if len(chunks) != len(output_units) or any(
            chunk["content"] != unit["content"] or chunk["tokens"] != unit["tokens"]
            for chunk, unit in zip(chunks, output_units, strict=True)
        ):
            fail(
                "native_chunk_mismatch", "Native chunker changed reviewed unit content"
            )
        documents.append(
            {
                "source_id": source["source_id"],
                "document_version": source["document_version"],
                "file_path": source["path"],
                "payload": payload,
                "units": output_units,
            }
        )
    for key, unit in units.items():
        unit["_report"]["copied_into_unit_ids"] = context_uses[key]
        if unit["disposition"] == "context_only":
            unit["_report"]["status"] = (
                "copied_context_only"
                if context_uses[key]
                else "reviewed_context_not_selected"
            )
    if not documents:
        fail("no_primary_units", "No reviewed primary units selected for native import")
    # Recheck source/parser bytes after all preparation before releasing output.
    for source in sources:
        if (
            _sha(source["_path"].read_bytes()) != source["original_sha256"]
            or parser_binding(source["_path"]) != source["parser_binding"]
        ):
            fail("source_changed", "Source or parser changed during preparation")
    report["status"] = "prepared_not_semantically_approved"
    artifact = {
        "schema_version": 1,
        "manifest_sha256": canonical_sha256(manifest),
        "tokenizer": {"model": model, "injected": injected},
        "sources": [
            {
                k: source[k]
                for k in (
                    "source_id",
                    "document_version",
                    "path",
                    "product",
                    "original_sha256",
                    "extracted_text_sha256",
                    "parser_binding",
                )
            }
            for source in sources
        ],
        "documents": documents,
        "report": report,
    }
    return {**artifact, "prepared_sha256": canonical_sha256(artifact)}


def main() -> None:
    """Inspect unapproved inputs or prepare a caller-reviewed manifest offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Write unapproved source review inputs"
    )
    inspect.add_argument("source", type=Path)
    inspect.add_argument("output", type=Path)
    inspect.add_argument("--source-id", required=True)
    inspect.add_argument("--document-version", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a reviewed source manifest")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "inspect":
        result = {
            "review_status": "unapproved_source_inspection",
            **inspect_source(args.source, args.source_id, args.document_version),
        }
    else:
        from product.answer_evidence import strict_json_loads

        result = prepare_import(
            strict_json_loads(args.manifest.read_text(encoding="utf-8")), args.root
        )
    # Exclusive creation protects an existing review/approval artifact. No file
    # is opened until complete validation has returned successfully.
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
