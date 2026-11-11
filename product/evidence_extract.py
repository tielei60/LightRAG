"""Evidence selection through the public LightRAG extraction-role callback."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import cache
from string import Formatter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.utils import is_truncated_response


def canonical(text: str) -> str:
    """Account only for native whitespace/full-width character normalization."""
    return "".join(unicodedata.normalize("NFKC", text).split())


@dataclass(frozen=True)
class EvidenceUnit:
    identifier: str
    source: str
    start: int
    end: int
    source_text: str
    entity_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.identifier
            or not self.source
            or type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end <= len(self.source_text)
            or not self.text.strip()
            or "<SEP>" in self.text
        ):
            raise ValueError("Invalid or ambiguous original-source evidence span")

    @property
    def text(self) -> str:
        return self.source_text[self.start : self.end]


@dataclass(frozen=True)
class EvidenceChunk:
    source: str
    product: str
    content: str
    units: tuple[EvidenceUnit, ...]

    def __post_init__(self) -> None:
        identifiers = [unit.identifier for unit in self.units]
        if (
            not self.source
            or not self.product
            or not self.content.strip()
            or not identifiers
            or len(identifiers) != len(set(identifiers))
            or any(unit.source != self.source for unit in self.units)
            or any(unit.source_text != self.units[0].source_text for unit in self.units)
            or canonical(self.product) not in canonical(self.units[0].source_text)
            or any(unit.text not in self.content for unit in self.units)
        ):
            raise ValueError("Evidence chunk must have a unique same-source catalog")


class PointerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("name", "type")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("Labels must be single-line non-empty strings")
        return value


class PointerRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    keywords: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class PointerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    entities: list[PointerEntity]
    relationships: list[PointerRelationship]


def render_units(units: Sequence[EvidenceUnit]) -> str:
    """Copy complete source units, deduplicating identical original strings."""
    return "\n\n".join(dict.fromkeys(unit.text for unit in units))


def to_native(selection: PointerResult, chunk: EvidenceChunk) -> dict[str, Any]:
    """Validate all references before emitting the native JSON contract."""
    catalog = {unit.identifier: unit for unit in chunk.units}

    def description(identifiers: list[str]) -> str:
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate evidence reference")
        unknown = set(identifiers) - catalog.keys()
        if unknown:
            raise ValueError("Unknown or out-of-scope evidence reference")
        selected = set(identifiers)
        return render_units(
            [unit for unit in chunk.units if unit.identifier in selected]
        )

    names = [entity.name for entity in selection.entities]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate entity name")
    known = set(names)
    scoped_names = {}
    prefix = chunk.product + "｜"
    for entity in selection.entities:
        if entity.type.casefold() == "product":
            if canonical(entity.name) != canonical(chunk.product):
                raise ValueError("Product identity differs from the verified source")
            scoped_names[entity.name] = chunk.product
        elif entity.type.casefold() == "organization":
            scoped_names[entity.name] = entity.name
        else:
            scoped_names[entity.name] = (
                entity.name if entity.name.startswith(prefix) else prefix + entity.name
            )
    if len(set(scoped_names.values())) != len(scoped_names):
        raise ValueError("Entity aliases collide in the verified product scope")
    entities = [
        {
            "name": scoped_names[entity.name],
            "type": entity.type,
            "description": description(entity.evidence_ids),
        }
        for entity in selection.entities
    ]
    for entity in selection.entities:
        typed_units = [
            catalog[key] for key in entity.evidence_ids if catalog[key].entity_types
        ]
        if typed_units and not any(
            entity.type.casefold() in {kind.casefold() for kind in unit.entity_types}
            for unit in typed_units
        ):
            raise ValueError("Entity type is not supported by its reviewed evidence")
    pairs: set[frozenset[str]] = set()
    relationships = []
    for relation in selection.relationships:
        pair = frozenset((relation.source, relation.target))
        if len(pair) != 2 or not pair <= known or pair in pairs:
            raise ValueError(
                "Missing, self-referencing or duplicate relation endpoints"
            )
        pairs.add(pair)
        relationships.append(
            {
                "source": scoped_names[relation.source],
                "target": scoped_names[relation.target],
                "keywords": relation.keywords,
                "description": description(relation.evidence_ids),
            }
        )
    return {"entities": entities, "relationships": relationships}


def template_pattern(template: str) -> re.Pattern[str]:
    """Parse a fixed native protocol template, never arbitrary business prose."""
    pieces = []
    seen = set()
    for literal, name, _, _ in Formatter().parse(template.strip()):
        pieces.append(re.escape(literal))
        if name:
            pieces.append(f"(?P={name})" if name in seen else f"(?P<{name}>.*?)")
            seen.add(name)
    return re.compile("".join(pieces), re.DOTALL)


class EvidenceExtractor:
    """One Flash call for selection; exact source copying for output and merges."""

    def __init__(
        self,
        *,
        requests: dict[str, EvidenceChunk],
        native_system_prompt: str,
        native_summary_template: str,
        type_guidance: str,
        selection_guidance: str,
        complete: Callable[..., Awaitable[str]],
        record: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not requests:
            raise ValueError("No reviewed extraction inputs")
        self.requests = {prompt.strip(): chunk for prompt, chunk in requests.items()}
        self.native_system_prompt = native_system_prompt.strip()
        self.summary_pattern = template_pattern(native_summary_template)
        self.type_guidance = type_guidance
        self.selection_guidance = selection_guidance
        self.complete = complete
        self.record = record or (lambda _: None)
        self.units = tuple(unit for chunk in requests.values() for unit in chunk.units)
        self.source_order = {
            source: index
            for index, source in enumerate(
                dict.fromkeys(chunk.source for chunk in requests.values())
            )
        }
        self.expected: dict[tuple[str, str], dict[str, EvidenceUnit]] = {}
        self.system_prompt = (
            "你从带编号的完整原文证据中识别实体和直接关系。只返回JSON，不撰写description。\n"
            "每个实体和关系用evidence_ids选择直接支持它的完整证据。"
            "证据上的entity_types是已核原文结构支持的实体类型，实体须选有对应类型支持的证据。"
            "不得输出不存在的编号，不从来源标签或示例抽取保险事实。"
            "关系两端必须在本次entities中。实体名称保持产品及责任范围，组织名称可用原称。\n"
            + type_guidance
            + "\n"
            + selection_guidance
            + "\n唯一输出结构：\n"
            + json.dumps(PointerResult.model_json_schema(), ensure_ascii=False)
        )

    def remember(
        self, selection: PointerResult, chunk: EvidenceChunk, native: dict[str, Any]
    ) -> None:
        def add(kind: str, label: str, identifiers: list[str]) -> None:
            target = self.expected.setdefault((kind.casefold(), canonical(label)), {})
            selected = set(identifiers)
            target.update(
                (canonical(unit.text), unit)
                for unit in chunk.units
                if unit.identifier in selected
            )

        for entity, converted in zip(
            selection.entities, native["entities"], strict=True
        ):
            add("Entity", converted["name"], entity.evidence_ids)
        for relation, converted in zip(
            selection.relationships, native["relationships"], strict=True
        ):
            source, target = converted["source"], converted["target"]
            add("Relation", f"({source}, {target})", relation.evidence_ids)
            add("Relation", f"({target}, {source})", relation.evidence_ids)

    def evidence_for_description(self, description: str) -> tuple[EvidenceUnit, ...]:
        """Accept only concatenations of complete catalog units, including cache hits."""
        available = {canonical(unit.text): unit for unit in self.units}
        candidates = sorted(available, key=len, reverse=True)
        result = []
        for part in description.split("<SEP>"):
            text = canonical(part)

            @cache
            def match(
                position: int, text: str = text
            ) -> tuple[EvidenceUnit, ...] | None:
                if position == len(text):
                    return ()
                for candidate in candidates:
                    if text.startswith(candidate, position):
                        tail = match(position + len(candidate))
                        if tail is not None:
                            return (available[candidate], *tail)
                return None

            found = match(0)
            if not text or found is None:
                raise ValueError(
                    "Merge contains text outside complete original evidence units"
                )
            result.extend(found)
        return tuple(result)

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_: Any,
    ) -> str:
        if history_messages:
            raise ValueError("The evidence prototype uses initial extraction only")
        chunk = self.requests.get(prompt.strip())
        if chunk is not None:
            if (system_prompt or "").strip() != self.native_system_prompt:
                raise ValueError("Native extraction template changed")
            user = json.dumps(
                {
                    "product": chunk.product,
                    "source_file": chunk.source,
                    "evidence": [
                        {
                            "id": unit.identifier,
                            "text": unit.text,
                            "entity_types": list(unit.entity_types),
                        }
                        for unit in chunk.units
                    ],
                },
                ensure_ascii=False,
            )
            event: dict[str, Any] = {
                "stage": "extract",
                "state": "sending",
                "source": chunk.source,
                "input": user,
            }
            self.record(event.copy())
            started = time.monotonic()
            raw: str | None = None
            try:
                raw = await self.complete(user, system_prompt=self.system_prompt)
                if is_truncated_response(raw):
                    raise ValueError("Truncated evidence selection")
                selection = PointerResult.model_validate_json(raw)
                native = to_native(selection, chunk)
                self.remember(selection, chunk, native)
            except Exception as exc:
                self.record(
                    {
                        **event,
                        "state": "invalid_or_unknown",
                        "response": raw,
                        "error_type": type(exc).__name__,
                    }
                )
                raise
            self.record(
                {
                    **event,
                    "state": "completed",
                    "response": raw,
                    "native": native,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            return json.dumps(native, ensure_ascii=False)

        match = self.summary_pattern.fullmatch(prompt.strip())
        if match is None or system_prompt:
            raise ValueError("Unsupported callback; no unverified fallback model call")
        descriptions = [
            json.loads(line) for line in match["description_list"].splitlines()
        ]
        if any(set(item) != {"Description"} for item in descriptions):
            raise ValueError("Native summary input changed")
        supplied = [
            unit
            for item in descriptions
            for unit in self.evidence_for_description(item["Description"])
        ]
        expected = self.expected.get(
            (match["description_type"].casefold(), canonical(match["description_name"]))
        )
        if not expected:
            raise ValueError("Merge has no verified extraction ledger")
        supplied_keys = {canonical(unit.text) for unit in supplied}
        if not supplied_keys <= expected.keys():
            raise ValueError("Merge evidence was not selected for this graph object")
        units = sorted(
            expected.values(),
            key=lambda unit: (self.source_order[unit.source], unit.start, unit.end),
        )
        result = render_units(units)
        self.record(
            {
                "stage": "merge",
                "state": "completed",
                "name": match["description_name"],
                "input_descriptions": descriptions,
                "result": result,
                "provider_called": False,
                "restored_units_omitted_before_callback": len(
                    expected.keys() - supplied_keys
                ),
            }
        )
        return result
