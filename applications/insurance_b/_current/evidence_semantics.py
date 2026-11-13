"""Check explicit source table cells and scope labels before native graph writes."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from typing import Any


def normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


@dataclass(frozen=True)
class SourceCell:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class MarkdownTable:
    start: int
    end: int
    title: str
    headers: tuple[SourceCell, ...]
    rows: tuple[tuple[SourceCell, ...], ...]


def _cells(line: str, offset: int) -> tuple[SourceCell, ...] | None:
    body = line.rstrip("\r\n")
    positions = [match.start() for match in re.finditer(r"(?<!\\)\|", body)]
    if (
        len(positions) < 2
        or body[: positions[0]].strip()
        or body[positions[-1] + 1 :].strip()
    ):
        return None
    cells = []
    for left, right in pairwise(positions):
        raw = body[left + 1 : right]
        start = offset + left + 1 + len(raw) - len(raw.lstrip())
        text = raw.strip()
        cells.append(SourceCell(text, start, start + len(text)))
    return tuple(cells)


def parse_markdown_tables(source_text: str) -> tuple[MarkdownTable, ...]:
    """Read complete pipe tables, keeping raw cells and exact source offsets.

    Escaped pipes remain in cells. Merged/malformed rows are rejected rather
    than guessed. Adjacent explicit notes belong to the table's source span.
    This parser does not flatten PDF layout into invented table coordinates.
    """
    lines = source_text.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    result = []
    index = 0
    while index + 1 < len(lines):
        headers = _cells(lines[index], offsets[index])
        separator = _cells(lines[index + 1], offsets[index + 1])
        if (
            not headers
            or len(headers) < 2
            or not separator
            or len(headers) != len(separator)
            or not all(re.fullmatch(r":?-{3,}:?", cell.text) for cell in separator)
        ):
            index += 1
            continue
        start, title = offsets[index], ""
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and index - previous <= 3:
            candidate = lines[previous].strip()
            if candidate.startswith("#") or candidate.endswith(("表：", "表:")):
                title = candidate
                start = offsets[previous]
        rows = []
        end_index = index + 2
        while end_index < len(lines):
            row = _cells(lines[end_index], offsets[end_index])
            if row is None:
                break
            if len(row) != len(headers):
                raise ValueError(
                    f"Markdown table row width differs from its header at source offset {offsets[end_index]}"
                )
            rows.append(row)
            end_index += 1
        if not rows:
            index = end_index
            continue
        last_index = end_index - 1
        note_index = end_index
        while note_index < len(lines) and not lines[note_index].strip():
            note_index += 1
        if note_index < len(lines) and re.match(
            r"^(?:注(?:\d+|[一二三四五六七八九十])?|备注|说明)[:：]|^\[\^[^]]+\]:",
            lines[note_index].strip(),
        ):
            last_index = note_index
            while (
                last_index + 1 < len(lines)
                and lines[last_index + 1].strip()
                and not lines[last_index + 1].lstrip().startswith("#")
            ):
                last_index += 1
        # Blank separators and the line terminator carry no table evidence;
        # an exact note span may legitimately end before its final newline.
        end = offsets[last_index] + len(lines[last_index].rstrip("\r\n"))
        result.append(MarkdownTable(start, end, title, headers, tuple(rows)))
        index = max(end_index, last_index + 1)
    return tuple(result)


def cell_kind(value: str) -> str:
    """Classify the matched cell, never other numbers in its source passage."""
    text = normalized(value)
    if not text:
        return "blank"
    if text in {"-", "—", "–", "--", "―"}:
        return "dash"
    if text in {"未提供", "未列明", "待确认", "不详"}:
        return "unknown"
    amount = re.fullmatch(
        r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:万?元(?:/[天日年份])?|%|‰|[天日次份])?(?:[，,；;].*)?",
        text,
    )
    if amount:
        number = Decimal(amount.group("amount").replace(",", ""))
        return "zero" if number == 0 else "positive"
    return "text"


def table_facts(source_text: str) -> list[dict[str, Any]]:
    """Expose checked header/row/cell bindings for preparation and model input."""
    return [
        {
            "table_start": table.start,
            "table_end": table.end,
            "table_title": table.title,
            "column_header": table.headers[column].text,
            "row_header": row[0].text,
            "raw_value": cell.text,
            "cell_start": cell.start,
            "cell_end": cell.end,
            "cell_kind": cell_kind(cell.text),
        }
        for table in parse_markdown_tables(source_text)
        for row in table.rows
        for column, cell in enumerate(row[1:], start=1)
    ]


def scope_options(source_text: str) -> list[dict[str, Any]]:
    """Recognize only explicitly introduced, numbered coverage-scope options."""
    result = []
    intro = re.compile(
        r"(?:以下|下列)[^。；：:]{0,80}保障范围[^。；：:]{0,80}选择[^。；：:]{0,60}[:：]",
        re.DOTALL,
    )
    heading = re.compile(
        r"(?m)^[ \t]*[一二三四五六七八九十]+[、．.][ \t]*(?P<label>[^\s，。；:：]{1,30})[ \t]*(?:\n|$)"
    )
    for match in intro.finditer(source_text):
        tail = source_text[match.end() :]
        boundary = re.search(r"(?m)^\s*(?:\d+\.\d+(?:\.\d+)*\s|#{1,6}\s)", tail)
        end = match.end() + boundary.start() if boundary else len(source_text)
        for item in heading.finditer(source_text, match.end(), end):
            result.append(
                {
                    "dimension": "保障范围",
                    "label": item["label"],
                    "intro_start": match.start(),
                    "intro_end": match.end(),
                    "start": item.start("label"),
                    "end": item.end("label"),
                }
            )
    return result


def _covered(start: int, end: int, units: Sequence[Any]) -> bool:
    cursor = start
    for unit in sorted(units, key=lambda unit: (unit.start, unit.end)):
        if unit.start > cursor:
            break
        cursor = max(cursor, unit.end)
        if cursor >= end:
            return True
    return False


def semantic_metadata(units: Sequence[Any]) -> dict[str, Any]:
    """Bind facts to a complete selected source span, including required notes."""
    if not units:
        return {"table_facts": [], "scope_options": []}
    source = units[0].source_text
    tables = [
        {
            **fact,
            "evidence_ids": [
                unit.identifier
                for unit in units
                if unit.start < fact["table_end"] and unit.end > fact["table_start"]
            ],
        }
        for fact in table_facts(source)
        if _covered(fact["table_start"], fact["table_end"], units)
    ]
    options = [
        {
            **option,
            "evidence_ids": [
                unit.identifier
                for unit in units
                if unit.start < option["end"] and unit.end > option["intro_start"]
            ],
        }
        for option in scope_options(source)
        if _covered(option["intro_start"], option["intro_end"], units)
        and _covered(option["start"], option["end"], units)
    ]
    return {"table_facts": tables, "scope_options": options}


class EvidenceSemanticError(ValueError):
    """The proposed positive graph fact exceeds its precise source support."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        summaries = []
        for issue in issues:
            if issue["kind"] == "scope_type":
                summaries.append(
                    f"{issue['name']}是原文保障范围选项，类型须用{'/'.join(issue['allowed_types'])}，不能按计划/分组处理"
                )
            elif issue["kind"] == "incomplete_table":
                summaries.append(
                    f"{issue['plan']}→{issue['coverage']}未选择完整表头/表格/注释证据（原文单元={issue['fact']['raw_value']!r}）"
                )
            elif issue["kind"] == "material_purpose":
                summaries.append(
                    f"{issue['name']}的所选原文用途={issue['observed_purposes']}，"
                    f"不支持{issue['requested_purpose']}材料分类；须选择该申请事项的完整用途证据，"
                    "不能因同样需要保险合同、身份证或出现保险金字样而归为理赔材料"
                )
            elif issue["kind"] == "material_relation":
                summaries.append(
                    f"{issue['source']}→{issue['target']}的{issue['keywords']}"
                    f"须连接{'/'.join(issue['allowed_material_types'])}且由本关系所选证据证明"
                    f"{issue['requested_purpose']}申请用途"
                )
            else:
                summaries.append(
                    f"{issue['plan']}→{issue['coverage']}对应表格单元={issue['value']!r}，不支持正向包含责任"
                )
        self.repair_feedback = (
            "；".join(summaries)
            + "。按原文修正类型/关系，保留0值及范围规则，不反推整个合同不保；"
            "退保、合同变更及复效等保全申请与保险事故理赔分别取证，不改写原文。"
        )
        super().__init__(self.repair_feedback)


def _local_name(name: str, product: str) -> str:
    text = normalized(name)
    prefix = normalized(product) + "|"
    text = text.removeprefix(prefix)
    return text.rsplit("|", 1)[-1]


def _listed_coverage(
    plan: str,
    coverage: str,
    facts: Sequence[dict[str, Any]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if plan in seen:
        return False
    rows = {
        normalized(fact["row_header"]): normalized(fact["raw_value"])
        for fact in facts
        if normalized(fact["column_header"]) in {"保障责任", "保险责任", "责任"}
    }
    value = rows.get(plan, "")
    items = re.split(r"[、,，+＋]", value)
    if coverage in items:
        return True
    # Resolve only an explicit named-plan reference, not an inferred subset.
    for referenced in rows:
        if (
            referenced != plan
            and any(
                item in {referenced + "基本责任", referenced + "全部责任"}
                for item in items
            )
            and _listed_coverage(referenced, coverage, facts, seen | {plan})
        ):
            return True
    return False


def material_purposes(text: str) -> frozenset[str]:
    """Identify explicit application purposes, not the names of shared documents.

    A surrender passage can mention earlier benefit payments or accidents as
    conditions. Those words alone are not evidence of a claim application.
    Mixed purposes are left ambiguous: a material node must select its own
    application passage instead of borrowing a different clause's heading.
    """
    value = normalized(text)
    purposes = set()
    if re.search(
        r"申请(?:理赔|(?:身故|高度残疾|全残|医疗|疾病)?保险金|赔偿|给付保险金)"
        r"|(?:保险金|理赔)(?:的)?申请"
        r"|理赔(?:须|应|需|时|材料|所需|提交|提供)",
        value,
    ):
        purposes.add("claim")
    if re.search(
        r"退保|解除(?:本)?合同|合同解除|保全(?:申请|办理|材料|所需)"
        r"|(?:申请|办理)(?:复效|合同恢复|保单变更|合同变更|减保|部分领取(?:保单账户价值)?|保单贷款)"
        r"|(?:复效|保单变更|合同变更)(?:申请|所需|材料|须|应|需|时)",
        value,
    ):
        purposes.add("policy_service")
    return frozenset(purposes)


def _validate_material_semantics(
    selection: Any,
    entities: Mapping[str, tuple[Any, Mapping[str, Any]]],
    relationships: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    roles: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Check material types and links only when the profile enables both roles."""
    material_types = {
        purpose: set(roles.get(role, ()))
        for purpose, role in (
            ("claim", "claim_material"),
            ("policy_service", "policy_service_material"),
        )
    }
    requirements = {
        purpose: set(roles.get(role, ()))
        for purpose, role in (
            ("claim", "claim_material_requirement"),
            ("policy_service", "policy_service_material_requirement"),
        )
    }
    if not any(material_types.values()):
        return []

    def purposes(ids: Sequence[str]) -> frozenset[str]:
        return material_purposes("\n\n".join(catalog[key].text for key in ids))

    issues = []
    for selected, converted in entities.values():
        for purpose, types in material_types.items():
            if converted["type"] not in types:
                continue
            observed = purposes(selected.evidence_ids)
            if observed != {purpose}:
                issues.append(
                    {
                        "kind": "material_purpose",
                        "name": converted["name"],
                        "requested_purpose": purpose,
                        "observed_purposes": sorted(observed),
                        "evidence_ids": list(selected.evidence_ids),
                    }
                )
    for selected, converted in zip(selection.relationships, relationships, strict=True):
        for purpose, labels in requirements.items():
            if not labels.intersection(converted["keywords"].split(",")):
                continue
            endpoint_types = (
                entities[selected.source][1]["type"],
                entities[selected.target][1]["type"],
            )
            endpoint_purposes = [
                purposes(entities[name][0].evidence_ids)
                for name in (selected.source, selected.target)
            ]
            if (
                sum(kind in material_types[purpose] for kind in endpoint_types) != 1
                or purposes(selected.evidence_ids) != {purpose}
                or any(
                    observed and observed != {purpose} for observed in endpoint_purposes
                )
            ):
                issues.append(
                    {
                        "kind": "material_relation",
                        "source": converted["source"],
                        "target": converted["target"],
                        "keywords": converted["keywords"],
                        "requested_purpose": purpose,
                        "allowed_material_types": sorted(material_types[purpose]),
                        "evidence_ids": list(selected.evidence_ids),
                    }
                )
    return issues


def validate_selection_semantics(
    selection: Any,
    chunk: Any,
    native: Mapping[str, Any],
    *,
    roles: Mapping[str, Sequence[str]],
) -> None:
    """Reject unsupported positive mappings; never silently remove graph facts."""
    if not roles:
        return
    catalog = {unit.identifier: unit for unit in chunk.units}
    entities = {
        row.name: (row, converted)
        for row, converted in zip(selection.entities, native["entities"], strict=True)
    }
    issues = _validate_material_semantics(
        selection, entities, native["relationships"], catalog, roles
    )
    plan_types = set(roles.get("plan", ()))
    inclusion = set(roles.get("inclusion", ()))
    coverage_types = set(roles.get("coverage", ()))
    scope_types = set(roles.get("scope_option", ()))
    plan_axes = {"计划", "保障计划", "投保计划", "保险计划", "每份计划"}
    row_axes = {
        table.start: normalized(table.headers[0].text)
        for table in parse_markdown_tables(chunk.units[0].source_text)
    }
    available_metadata = semantic_metadata(chunk.units)
    available_tables = {
        (fact["table_start"], fact["table_end"])
        for fact in available_metadata["table_facts"]
    }

    def plan_row(fact: Mapping[str, Any]) -> bool:
        return row_axes[fact["table_start"]] in plan_axes and normalized(
            fact["column_header"]
        ) in {"保障责任", "保险责任", "责任"}

    for selected, converted in entities.values():
        if converted["type"] in scope_types:
            continue
        units = [catalog[key] for key in selected.evidence_ids]
        metadata = semantic_metadata(units)
        name = _local_name(converted["name"], chunk.product)
        matches = [
            item
            for item in available_metadata["scope_options"]
            if normalized(item["label"]) == name
        ]
        if matches:
            # A separately documented plan may reuse a scope label, but its
            # explicit plan table must be among this entity's own evidence.
            plan_support = any(
                normalized(fact["column_header"]) == name
                and "计划" in fact["table_title"]
                or normalized(fact["row_header"]) == name
                and plan_row(fact)
                for fact in metadata["table_facts"]
            )
            if not plan_support:
                issues.append(
                    {
                        "kind": "scope_type",
                        "name": name,
                        "source_options": matches,
                        "allowed_types": sorted(scope_types),
                    }
                )
    for selected, converted in zip(
        selection.relationships, native["relationships"], strict=True
    ):
        if not inclusion.intersection(converted["keywords"].split(",")):
            continue
        endpoints = [entities[selected.source], entities[selected.target]]
        plans = [item for item in endpoints if item[1]["type"] in plan_types]
        coverages = [item for item in endpoints if item[1]["type"] in coverage_types]
        if len(plans) != 1 or len(coverages) != 1:
            continue
        plan = _local_name(plans[0][1]["name"], chunk.product)
        coverage = _local_name(coverages[0][1]["name"], chunk.product)
        units = [catalog[key] for key in selected.evidence_ids]
        facts = semantic_metadata(units)["table_facts"]
        matches = [
            fact
            for fact in facts
            if normalized(fact["column_header"]) == plan
            and normalized(fact["row_header"]) == coverage
        ]
        for fact in matches:
            if fact["cell_kind"] != "positive":
                issues.append(
                    {
                        "kind": "unsupported_inclusion",
                        "plan": plan,
                        "coverage": coverage,
                        "value": fact["raw_value"],
                        "fact": fact,
                    }
                )
        listed = [
            fact
            for fact in facts
            if normalized(fact["row_header"]) == plan and plan_row(fact)
        ]
        if listed and not _listed_coverage(
            plan, coverage, [fact for fact in facts if plan_row(fact)]
        ):
            issues.append(
                {
                    "kind": "unsupported_inclusion",
                    "plan": plan,
                    "coverage": coverage,
                    "value": listed[0]["raw_value"],
                    "fact": listed[0],
                }
            )
        if not matches and not listed:
            # A row fragment cannot bypass the complete table/footnote fence.
            for fact in table_facts(chunk.units[0].source_text):
                related = (
                    normalized(fact["column_header"]) == plan
                    and normalized(fact["row_header"]) == coverage
                    or normalized(fact["row_header"]) == plan
                    and plan_row(fact)
                )
                if related and (
                    (fact["table_start"], fact["table_end"]) in available_tables
                    or any(
                        unit.start < fact["table_end"]
                        and unit.end > fact["table_start"]
                        for unit in units
                    )
                ):
                    issues.append(
                        {
                            "kind": "incomplete_table",
                            "plan": plan,
                            "coverage": coverage,
                            "fact": fact,
                        }
                    )
                    break
    if issues:
        raise EvidenceSemanticError(issues)
