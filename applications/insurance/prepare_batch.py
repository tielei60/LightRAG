"""Prepare source-faithful native import batches without product-specific rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from applications.insurance.evidence_semantics import scope_options, table_facts
from applications.insurance.paths import PROMPT_PROFILE_DIR, REPO_ROOT
from applications.insurance.prepare_context import DELIMITER, extract_source

ROOT = REPO_ROOT
CLAUSE = re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+)+)[ \t]+[^\n]+")


def digest(value: bytes | str) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    heading: str
    parent_labels: tuple[str, ...] = ()
    references: tuple[tuple[int, int, str], ...] = ()
    kind: str = "clause"
    responsibility_heading: str | None = None


def clause_spans(source: str) -> list[Span]:
    """Use final clause occurrences, preserving a preceding table of contents."""
    matches = {match.group(1): match for match in CLAUSE.finditer(source)}
    headings = sorted(matches.items(), key=lambda item: item[1].start())
    if not headings:
        return [Span(0, len(source), "document", source.splitlines()[0])]
    starts = [0] + [match.start() for _, match in headings if match.start() > 0]
    by_start = {
        match.start(): (number, match.group().strip()) for number, match in headings
    }
    return [
        Span(start, end, *by_start.get(start, ("front-matter", "原文件前置内容")))
        for start, end in zip(starts, starts[1:] + [len(source)], strict=True)
    ]


def _headings_to_spans(
    source: str, headings: list[tuple[int, str, int, str]]
) -> list[Span]:
    """Keep primary coverage exact while recording actual heading ancestry."""
    starts = sorted({0, *(start for start, _, _, _ in headings)})
    labels = {start: (heading, level, kind) for start, heading, level, kind in headings}
    stack: list[tuple[int, str]] = []
    spans = []
    for start, end in zip(starts, starts[1:] + [len(source)], strict=True):
        heading, level, kind = labels.get(start, ("原文件前置内容", 0, "front-matter"))
        while stack and stack[-1][0] >= level:
            stack.pop()
        label = f"section-{start}"
        parents = tuple(value for _, value in stack)
        spans.append(
            Span(
                start,
                end,
                label,
                heading,
                parents,
                kind=kind,
                responsibility_heading=heading if kind == "responsibility" else None,
            )
        )
        stack.append((level, label))
    return _shared_responsibility_rules(source, spans)


def text_structure_spans(source: str, role: str = "underwriting") -> list[Span]:
    """Keep whole Markdown tables and complete disclosure responsibility blocks."""
    if role == "underwriting":
        headings = [
            (
                match.start(),
                match.group().strip(),
                len(match.group(1)),
                "document-section",
            )
            for match in re.finditer(r"(?m)^(#{1,3})[ \t]+[^\n]+$", source)
        ]
        return _headings_to_spans(source, headings)
    plan_heading = re.compile(r"(?:保障)?计划[一二三四五六七八九十百0-9]+[：:]?")
    pattern = re.compile(
        r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、[^\n]+|[（(][一二三四五六七八九十]+[）)][ \t]*[^\n]+|[1-9]\d?\.[ \t]*(?:基本责任|可选责任[一二三四五]?)[ \t]*|(?:保障)?计划[一二三四五六七八九十百0-9]+[：:]?[ \t]*|【[^\n】]+】[ \t]*)$"
    )
    matches = list(pattern.finditer(source))
    # The limited/unlimited coverage range blocks are one shared introductory
    # rule. They must not become separate parents of all later responsibilities.
    responsibility_regions = []
    for index, match in enumerate(matches):
        text = re.sub(r"\s+", "", match.group())
        if re.fullmatch(r"[（(][一二三四五六七八九十]+[）)]保险责任", text):
            later = [
                other.start()
                for other in matches[index + 1 :]
                if re.match(r"[（(][一二三四五六七八九十]+[）)]", other.group().strip())
            ]
            responsibility_regions.append(
                (match.end(), later[0] if later else len(source))
            )
    headings = []
    for match in matches:
        heading = match.group().strip()
        if heading.startswith("【"):
            level, kind = 5, "responsibility"
        elif plan_heading.fullmatch(heading):
            # A plan is a scope parent, not part of the preceding benefit body.
            # Its basic/optional groups and repeated benefit names stay local.
            level, kind = 3, "coverage-plan"
        elif re.match(r"[1-9]", heading):
            level, kind = 4, "responsibility-group"
        elif heading.startswith(("（", "(")):
            level, kind = 2, "document-section"
        else:
            if any(a <= match.start() < b for a, b in responsibility_regions):
                continue
            level, kind = 1, "document-section"
        headings.append((match.start(), heading, level, kind))
    return _headings_to_spans(source, headings)


def _responsibility_heading_matches(source: str, span: Span) -> list[tuple[int, str]]:
    """Find standalone responsibility names printed before their full body."""
    lines = list(re.finditer(r"(?m)^[^\n]*(?:\n|$)", source[span.start : span.end]))
    found = []
    for index, line in enumerate(lines):
        if not line.group().lstrip().startswith("被保险人"):
            continue
        preceding = []
        for previous in reversed(lines[max(0, index - 4) : index]):
            text = previous.group().strip()
            if (
                not text
                or len(re.sub(r"\s+", "", text)) > 18
                or not re.fullmatch(r"[\u4e00-\u9fff（）()\s]+", text)
            ):
                break
            preceding.insert(0, previous)
        if preceding:
            title = "".join(item.group() for item in preceding).strip()
            compact = re.sub(r"\s+", "", title)
            if compact.endswith("保险金"):
                found.append((span.start + preceding[0].start(), title))
    return list(dict.fromkeys(found))


def _shared_responsibility_rules(source: str, spans: list[Span]) -> list[Span]:
    """Copy an explicit joint rule to siblings while keeping its primary owner."""
    from dataclasses import replace

    families: dict[tuple[str, ...], list[int]] = {}
    for index, span in enumerate(spans):
        if span.kind == "responsibility":
            families.setdefault(span.parent_labels, []).append(index)
    output = list(spans)
    for siblings in families.values():
        if len(siblings) < 2:
            continue
        names = {
            re.sub(r"[\s【】]", "", spans[index].responsibility_heading or "")
            for index in siblings
        } - {""}
        if len(names) < 2:
            continue
        last = spans[siblings[-1]]
        shared = []
        for sentence in re.finditer(
            r"[^。！？]+[。！？]?", source[last.start : last.end]
        ):
            text = re.sub(r"\s+", "", sentence.group())
            # Mentioning another benefit (for example, deducting its payouts)
            # does not make the current benefit's terms common to both.
            joint_rule = re.search(r"之和|合计|共享|共用|二者|两项保险责任|均", text)
            if joint_rule and sum(name in text for name in names) >= 2:
                shared.append(last.start + sentence.start())
        if shared:
            reference = (min(shared), last.end, "shared-sibling-rule")
            for index in siblings[:-1]:
                output[index] = replace(
                    output[index], references=output[index].references + (reference,)
                )
    return output


def policy_structure_spans(source: str) -> list[Span]:
    """Split only actual responsibility headings; keep each full definition."""
    base = clause_spans(source)
    definitions = definition_terms(source, base)
    definition_positions = {(span.start, span.end) for span in definitions.values()}
    result = []
    for span in base:
        levels = span.label.split(".")
        parents = tuple(".".join(levels[:depth]) for depth in range(2, len(levels)))
        matches = _responsibility_heading_matches(source, span)
        if not matches:
            kind = (
                "definition"
                if (span.start, span.end) in definition_positions
                else "front-matter"
                if span.label == "front-matter"
                else "clause"
            )
            result.append(
                Span(span.start, span.end, span.label, span.heading, parents, kind=kind)
            )
            continue
        # A true heading-only introduction remains an independent source span,
        # and is copied to each child together with its outer common premise.
        first_start = matches[0][0]
        if first_start > span.start:
            result.append(
                Span(
                    span.start,
                    first_start,
                    span.label,
                    span.heading,
                    parents,
                    kind="responsibility-group",
                )
            )
        for index, ((start, heading), end) in enumerate(
            zip(matches, [point for point, _ in matches[1:]] + [span.end], strict=True)
        ):
            result.append(
                Span(
                    start,
                    end,
                    f"{span.label}/responsibility-{index + 1}",
                    heading,
                    parents + ((span.label,) if first_start > span.start else ()),
                    kind="responsibility",
                    responsibility_heading=heading,
                )
            )
    return _shared_responsibility_rules(source, result)


def document_title_span(source: str, product: str, role: str) -> Span | None:
    """Allow a document node only when its full title occurs in source text."""
    suffixes = {
        "policy": r"条\s*款",
        "disclosure": r"产\s*品\s*说\s*明(?:\s*书)?",
        "underwriting": r"投\s*保(?:\s*及)?\s*核\s*保\s*规\s*则",
    }
    suffix = suffixes.get(role)
    if suffix is None:
        return None
    name = r"\s*".join(map(re.escape, product))
    match = re.search(name + r"\s*" + suffix, source)
    if match is None:
        return None
    return Span(match.start(), match.end(), "document-title", match.group())


def definition_terms(source: str, spans: list[Span]) -> dict[str, Span]:
    """Resolve only explicit definition sections; never guess a business scope."""
    sections = list(re.finditer(r"(?m)^[ \t]*(\d+)[．.、][ \t]*释义[^\n]*", source))
    if not sections:
        return {}
    prefix = sections[-1].group(1) + "."
    result = {}
    for span in spans:
        if not span.label.startswith(prefix):
            continue
        body = re.sub(r"\s+", "", source[span.start : span.end])
        body = body.removeprefix(span.label)
        term = re.match(r"(.{1,25}?)(?:指|是)", body)
        if term:
            result[term.group(1)] = span
    return result


def prepare_document(
    path: Path,
    product: str,
    role: str,
    count: Callable[[str], int],
    *,
    token_limit: int = 6000,
    target_tokens: int = 1800,
) -> tuple[dict, list[dict], dict]:
    """Preserve whole rules, structural parents and explicitly cited definitions."""
    source, pages = extract_source(path)
    if not source.strip() or DELIMITER in source:
        raise ValueError("Empty source or reserved chunk delimiter")
    identity = re.search(r"\s*".join(map(re.escape, product)), source)
    if not identity:
        raise ValueError(f"Product identity is not supported by {path.name}")
    identity_span = Span(identity.start(), identity.end(), "product-identity", product)
    document_title = document_title_span(source, product, role)
    spans = (
        policy_structure_spans(source)
        if role == "policy"
        else text_structure_spans(source, role)
    )
    by_label = {span.label: span for span in spans}
    definitions = definition_terms(source, spans) if role == "policy" else {}
    groups = [[span] for span in spans]
    source_tables = table_facts(source)
    source_scope_options = scope_options(source)

    def scope_frame(span: Span) -> dict:
        return {
            "document_role": role,
            "unit_kind": span.kind,
            "heading_path": [
                by_label[label].heading
                for label in span.parent_labels
                if label in by_label
            ]
            + [span.heading],
            "responsibility_heading": span.responsibility_heading,
        }

    source_hash = digest(path.read_bytes())
    chunks, catalogs, reports = [], [], []
    for group in groups:
        primary = {(span.start, span.end) for span in group}
        copies = {
            (identity_span.start, identity_span.end): (
                identity_span,
                "product-identity",
            )
        }
        for span in group:
            for parent_label in span.parent_labels:
                parent = by_label.get(parent_label)
                if parent and (parent.start, parent.end) not in primary:
                    copies[(parent.start, parent.end)] = (parent, "structural-parent")
            for a, b, purpose in span.references:
                copies[(a, b)] = (
                    Span(
                        a,
                        b,
                        purpose,
                        source[a:b].strip().splitlines()[0],
                        kind="shared-rule",
                    ),
                    "structural-parent",
                )
        body = source[group[0].start : group[-1].end]
        canonical_body = re.sub(r"\s+", "", body)
        for term, definition in definitions.items():
            if (
                term + "（见释义）" in canonical_body
                or (
                    group[0].kind == "responsibility"
                    and len(term) >= 2
                    and term in canonical_body
                )
            ) and (definition.start, definition.end) not in primary:
                copies[(definition.start, definition.end)] = (
                    definition,
                    "explicit-definition-reference",
                )
        start, end = group[0].start, group[-1].end
        page_ids = [
            index + 1 for index, (a, b) in enumerate(pages) if a < end and b > start
        ]
        location = (
            "PDF物理页（从1计数）：" + ",".join(map(str, page_ids))
            if path.suffix.lower() == ".pdf"
            else f"原文件Unicode码点（左闭右开）：{start}:{end}"
        )
        content = (
            "【来源标识：整理信息，不是合同新约定】\n"
            f"产品：{product}\n原件：{path.name}\n资料角色：{role}\n文档版本：未标明；以SHA256绑定本次原件\n"
            f"原件SHA256：{source_hash}\n{location}\n"
        )
        scope = scope_frame(group[0])
        if group[0].kind != "front-matter":
            content += "【本单元原文标题】\n" + group[0].heading + "\n"
        units = []
        for span in group:
            types = ["Product", "Organization"] if span.label == "front-matter" else []
            units.append(
                {
                    "id": f"e-{source_hash[:16]}-{span.start}-{span.end}",
                    "start": span.start,
                    "end": span.end,
                    "entity_types": types,
                    "role": "primary",
                    "scope": scope_frame(span),
                }
            )
        content += "【主原文开始】\n" + body + "\n【主原文结束】\n"
        for (a, b), (span, reason) in sorted(copies.items()):
            if any(main.start <= a and b <= main.end for main in group):
                continue
            content += f"【同文件引用原文：{reason}；Unicode码点{a}:{b}】\n{source[a:b]}\n【引用原文结束】\n"
            units.append(
                {
                    "id": f"e-{source_hash[:16]}-{a}-{b}",
                    "start": a,
                    "end": b,
                    "entity_types": ["Product"] if reason == "product-identity" else [],
                    "role": reason,
                    "scope": scope_frame(span),
                    "evidence_purpose": span.label
                    if span.kind == "shared-rule"
                    else reason,
                }
            )
        # The stable catalog includes identity separately even when already inside a
        # primary span, so selection can cite the name without citing the full TOC.
        if (identity_span.start, identity_span.end) in primary or any(
            s.start <= identity_span.start and identity_span.end <= s.end for s in group
        ):
            a, b = identity_span.start, identity_span.end
            units.append(
                {
                    "id": f"e-{source_hash[:16]}-{a}-{b}",
                    "start": a,
                    "end": b,
                    "entity_types": ["Product"],
                    "role": "product-identity",
                }
            )
        if (
            document_title is not None
            and start <= document_title.start < document_title.end <= end
        ):
            a, b = document_title.start, document_title.end
            units.append(
                {
                    "id": f"e-{source_hash[:16]}-{a}-{b}",
                    "start": a,
                    "end": b,
                    "entity_types": ["产品文档"],
                    "role": "primary",
                    "evidence_purpose": "verified-original-document-title",
                }
            )
        tokens = count(content)
        if tokens > token_limit:
            raise ValueError(
                f"Complete source unit exceeds token limit ({tokens}>{token_limit}): {path.name} {group[0].label}; review semantic boundaries"
            )
        chunks.append(content.strip())
        catalogs.append(
            {
                "content_sha256": digest(content.strip()),
                "scope": scope,
                "scope_options": [
                    option
                    for option in source_scope_options
                    if any(
                        a <= option["intro_start"]
                        and option["intro_end"] <= b
                        and a <= option["start"]
                        and option["end"] <= b
                        for a, b in primary | set(copies)
                    )
                ],
                "table_facts": [
                    fact
                    for fact in source_tables
                    if start <= fact["table_start"] and fact["table_end"] <= end
                ],
                "units": units,
            }
        )
        reports.append(
            {
                "start": start,
                "end": end,
                "clause_ids": [s.label for s in group],
                "headings": [s.heading for s in group],
                "scope": scope,
                "physical_pages": page_ids if path.suffix.lower() == ".pdf" else [],
                "tokens": tokens,
                "copies": [
                    {"start": a, "end": b, "reason": reason}
                    for (a, b), (_, reason) in sorted(copies.items())
                ],
                "content_sha256": digest(content.strip()),
            }
        )
    if "".join(source[item["start"] : item["end"]] for item in reports) != source:
        raise ValueError("Primary source coverage is incomplete")
    return (
        {
            "text": DELIMITER.join(chunks),
            "file_source": path.name,
            "chunking": {
                "strategy": "fixed_token",
                "params": {
                    "split_by_character": DELIMITER,
                    "split_by_character_only": True,
                    "chunk_token_size": token_limit,
                    "chunk_overlap_token_size": 0,
                },
            },
        },
        catalogs,
        {
            "source_sha256": source_hash,
            "document_role": role,
            "verified_document_title": {
                "start": document_title.start,
                "end": document_title.end,
                "text": document_title.heading,
            }
            if document_title is not None
            else None,
            "extracted_text_sha256": digest(source),
            "parser": "pypdf page.extract_text"
            if path.suffix.lower() == ".pdf"
            else "UTF-8 text",
            "offset_unit": "Unicode code points in unmodified extracted text",
            "source_span_coverage": "complete; primary spans reproduce source exactly once",
            "segmentation": "one complete source clause, responsibility or definition per unit; true parents and explicit dependencies copied separately",
            "document_version": "not stated; original SHA256 binds this batch",
            "layout_review": "not automatically certified; compare key tables and clauses to original",
            "original_extracted_text": source,
            "chunks": reports,
        },
    )


def discover(source_root: Path, folders: list[str]) -> tuple[list[dict], list[dict]]:
    """Require terms, disclosure and underwriting; inventory every other file."""
    documents, excluded = [], []
    for folder in folders:
        directory = (source_root / folder).resolve(strict=True)
        if not directory.is_relative_to(source_root) or not directory.is_dir():
            raise ValueError("Product directory escapes the source root")
        files = sorted(p for p in directory.iterdir() if p.is_file())
        policies = [p for p in files if p.name.endswith("条款.pdf")]
        if len(policies) != 1:
            raise ValueError(f"Exactly one policy PDF is required: {folder}")
        product = policies[0].stem.removesuffix("条款")
        selected = {
            "policy": policies,
            "disclosure": [p for p in files if p.name.endswith("产品说明书.pdf")],
            "underwriting": [
                p
                for p in files
                if p.suffix.lower() in (".md", ".pdf")
                and any(term in p.stem for term in ("投保及核保规则", "投核保规则"))
            ],
        }
        if any(len(matches) != 1 for matches in selected.values()):
            raise ValueError(
                f"Exactly one source per required document role is required: {folder}"
            )
        chosen = {p for matches in selected.values() for p in matches}
        for role, matches in selected.items():
            documents.append({"product": product, "role": role, "path": matches[0]})
        for path in files:
            if path not in chosen:
                excluded.append(
                    {
                        "relative_path": str(path.relative_to(source_root)),
                        "sha256": digest(path.read_bytes()),
                        "status": "excluded",
                        "reason": "Rate/cash-value calculation is outside this batch"
                        if any(term in path.stem for term in ("费率", "现金价值"))
                        else "Not one of the selected authoritative representations; requires explicit review",
                    }
                )
    return documents, excluded


def build_batch(
    batch: str,
    folders: list[str],
    workspace: str,
    *,
    source_root: Path | None = None,
    config_path: Path | None = None,
    token_limit: int = 6000,
    target_tokens: int = 1800,
    append: bool = False,
) -> dict:
    if not re.fullmatch(r"[a-z0-9_-]+", batch) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", workspace
    ):
        raise ValueError("Invalid batch or workspace name")
    if not folders or len(folders) != len(set(folders)):
        raise ValueError("Unique product directories are required")
    source_root = (source_root or ROOT / "docs/insurance_docs").resolve(strict=True)
    config_path = config_path or ROOT / ".env"
    output = ROOT / "output" / batch
    if output.exists():
        raise ValueError(
            "Batch already exists; inspect it instead of overwriting reviewed artifacts"
        )
    config = dict(dotenv_values(config_path, interpolate=False))
    if not config:
        raise ValueError("Base configuration is empty")
    from applications.insurance.database_storage import is_database_backend

    database = is_database_backend(config)
    target_exists = (ROOT / "rag_storage" / workspace).is_dir()
    if target_exists and not append and not database:
        raise ValueError("Target workspace exists; use --append for incremental import")
    if append and not target_exists and not database:
        raise ValueError("Incremental preparation requires an existing workspace")
    import tiktoken

    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    count = lambda value: len(encoder.encode(value))
    documents, excluded = discover(source_root, folders)
    if append and config.get("WORKSPACE") != workspace:
        raise ValueError("Incremental configuration must name the target workspace")
    embedding_limit = int(config.get("EMBEDDING_TOKEN_LIMIT") or 8192)
    if not 0 < target_tokens <= token_limit <= embedding_limit:
        raise ValueError("Prepared-unit limit must fit the embedding input limit")
    # Prepare everything before creating a batch; failures leave no partial batch.
    prepared = [
        (
            item,
            prepare_document(
                item["path"],
                item["product"],
                item["role"],
                count,
                token_limit=token_limit,
                target_tokens=target_tokens,
            ),
        )
        for item in documents
    ]
    output.mkdir(parents=True)
    (output / "prepared").mkdir()
    entries, reports = [], []
    for index, (item, (payload, catalogs, report)) in enumerate(prepared, 1):
        path = item["path"]
        stem = f"prepared/{index:02d}-{item['role']}"
        payload_path, catalog_path, report_path = [
            output / f"{stem}-{suffix}.json"
            for suffix in ("text", "catalog", "source-map")
        ]
        extracted_path = output / f"{stem}-extracted.txt"
        extracted_path.write_text(report.pop("original_extracted_text"))
        report["extracted_text_path"] = str(extracted_path.relative_to(output))
        save_json(payload_path, payload)
        save_json(catalog_path, catalogs)
        save_json(report_path, report)
        entries.append(
            {
                "product": item["product"],
                "document_role": item["role"],
                "relative_path": str(path.relative_to(source_root)),
                "sha256": digest(path.read_bytes()),
                "document_version": "not stated",
                "ingestion_label": str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else path.name,
                "prepared_path": str(payload_path.relative_to(output)),
                "prepared_sha256": digest(payload_path.read_bytes()),
                "catalog_path": str(catalog_path.relative_to(output)),
                "catalog_sha256": digest(catalog_path.read_bytes()),
                "source_map_path": str(report_path.relative_to(output)),
                "source_map_sha256": digest(report_path.read_bytes()),
            }
        )
        reports.append(
            {
                "file": path.name,
                "chunks": len(report["chunks"]),
                "max_tokens": max(c["tokens"] for c in report["chunks"]),
                "primary_source_coverage": report["source_span_coverage"],
            }
        )
    config.update(
        {
            "WORKSPACE": workspace,
            "HOST": "127.0.0.1",
            "PORT": (config.get("PORT") or "9621") if append else "9621",
            "WORKING_DIR": str(ROOT / "rag_storage"),
            "INPUT_DIR": str(ROOT / "inputs"),
            "PROMPT_DIR": (config.get("PROMPT_DIR") or str(PROMPT_PROFILE_DIR))
            if append
            else str(PROMPT_PROFILE_DIR),
            "LOG_DIR": str(ROOT / "output/logs"),
            **{
                key: (config.get(key) or value) if append else value
                for key, value in {
                    "TOP_K": "30",
                    "CHUNK_TOP_K": "12",
                    "MAX_ENTITY_TOKENS": "4000",
                    "MAX_RELATION_TOKENS": "5000",
                    "MAX_TOTAL_TOKENS": "18000",
                }.items()
            },
        }
    )
    runtime = output / "runtime.env"
    runtime.write_text(
        "\n".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in config.items()
            if value is not None
        )
        + "\n"
    )
    runtime.chmod(0o600)
    manifest = {
        "batch": batch,
        "workspace": workspace,
        "ingestion_mode": "append" if append else "new",
        "source_root": str(source_root),
        "expected_product_count": len(folders),
        "documents": entries,
        "excluded_documents": excluded,
        "retrieval_checks": [],
        "tokenizer": "o200k_base (gpt-4o-mini)",
        "unit_token_limit": token_limit,
        "target_tokens": target_tokens,
        "segmentation_revision": "source-semantic-v5",
        "target_tokens_policy": "retained for command compatibility; source business boundaries take precedence and are never split to meet a size target",
    }
    save_json(output / "manifest.json", manifest)
    save_json(
        output / "preparation-report.json",
        {
            "documents": reports,
            "total_chunks": sum(item["chunks"] for item in reports),
            "originals_modified": False,
            "inventory_coverage": "all files in selected product folders included or explicitly excluded",
            "semantic_review": "structural candidates; key table layout and scope require source review",
            "manifest_sha256": digest((output / "manifest.json").read_bytes()),
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument(
        "--workspace",
        default="master",
        help="Default master workspace; override only for an isolated test",
    )
    parser.add_argument(
        "--products", nargs="+", required=True, help="Product folders under source root"
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Prepare a new import batch for an existing workspace",
    )
    parser.add_argument("--token-limit", type=int, default=6000)
    parser.add_argument("--target-tokens", type=int, default=1800)
    args = parser.parse_args()
    manifest = build_batch(
        args.batch,
        args.products,
        args.workspace,
        source_root=args.source_root,
        config_path=args.config,
        token_limit=args.token_limit,
        target_tokens=args.target_tokens,
        append=args.append,
    )
    print(
        json.dumps(
            {
                "batch": args.batch,
                "workspace": args.workspace,
                "documents": len(manifest["documents"]),
                "manifest": str(ROOT / "output" / args.batch / "manifest.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
