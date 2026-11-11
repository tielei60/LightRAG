"""Evidence selection through the public LightRAG extraction-role callback."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Formatter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.constants import (
    DEFAULT_ENTITY_NAME_MAX_BYTES,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)
from lightrag.utils import (
    is_truncated_response,
    normalize_entity_name,
    sanitize_and_normalize_extracted_text,
)
from product.evidence_semantics import semantic_metadata, validate_selection_semantics
from product.extraction_profile import ExtractionProfile

ENTITY_TYPES = {
    "product": "产品",
    "plan": "保障计划",
    "coverage": "保障责任",
    "underwritingrule": "责任选择条件",
    "rule": "合同规则",
    "concept": "术语定义",
    "organization": "机构",
    "other": "其他",
}
ENTITY_TYPES.update({kind: kind for kind in tuple(ENTITY_TYPES.values())})


def entity_type(value: str) -> str:
    """Use Chinese business types while accepting the prototype's legacy aliases."""
    try:
        return ENTITY_TYPES[value.casefold()]
    except KeyError as exc:
        raise ValueError("Unsupported evidence entity type") from exc


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
    role: str = "primary"

    def __post_init__(self) -> None:
        if (
            not self.identifier
            or not self.source
            or type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end <= len(self.source_text)
            or not self.text.strip()
            or "<SEP>" in self.text
            or self.role
            not in {
                "primary",
                "product-identity",
                "structural-parent",
                "explicit-definition-reference",
            }
        ):
            raise ValueError("Invalid or ambiguous original-source evidence span")

    @property
    def text(self) -> str:
        return self.source_text[self.start : self.end]

    @property
    def identity(self) -> tuple[str, str, int, int]:
        """Keep identical quotations from distinct source versions or positions."""
        return (
            self.source,
            hashlib.sha256(self.source_text.encode()).hexdigest(),
            self.start,
            self.end,
        )


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
    """Copy whole units, deduplicating source positions rather than quotations."""
    unique = {unit.identity: unit for unit in units}
    return "\n\n".join(unit.text for unit in unique.values())


def _map_references(value: Any, identifiers: dict[str, str]) -> Any:
    """Map only declared evidence references through an exact request bijection."""
    if isinstance(value, list):
        return [_map_references(item, identifiers) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "evidence_ids":
            unknown = [
                identifier for identifier in item if identifier not in identifiers
            ]
            if unknown:
                raise ValueError(
                    f"Unknown request-local evidence reference: {unknown[0]}; use only the provided e1..eN IDs"
                )
            result[key] = [identifiers[identifier] for identifier in item]
        else:
            result[key] = _map_references(item, identifiers)
    return result


def provider_example(example: dict[str, Any]) -> dict[str, Any]:
    """Keep each instructional example in its own short evidence-ID namespace."""
    result = copy.deepcopy(example)
    evidence = result["input"]["evidence"]
    identifiers = {unit["id"]: f"e{index}" for index, unit in enumerate(evidence, 1)}
    for unit in evidence:
        unit["id"] = identifiers[unit["id"]]
    result["output"] = _map_references(result["output"], identifiers)
    return result


def to_native(
    selection: PointerResult,
    chunk: EvidenceChunk,
    *,
    profile: ExtractionProfile | None = None,
) -> dict[str, Any]:
    """Validate all references before emitting the native JSON contract."""
    catalog = {unit.identifier: unit for unit in chunk.units}
    normalize_type = profile.entity_type if profile is not None else entity_type
    product_type = profile.product_entity_type if profile is not None else "产品"
    unscoped_types = profile.unscoped_entity_types if profile is not None else ("机构",)
    if profile is not None:
        profile.validate_catalog(chunk.units)

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
        kind = normalize_type(entity.type)
        if kind == product_type:
            if canonical(entity.name) != canonical(chunk.product):
                raise ValueError("Product identity differs from the verified source")
            scoped_names[entity.name] = chunk.product
        elif kind in unscoped_types:
            scoped_names[entity.name] = entity.name
        else:
            scoped_names[entity.name] = (
                entity.name if entity.name.startswith(prefix) else prefix + entity.name
            )
    scoped_names = {
        name: normalize_entity_name(scoped) for name, scoped in scoped_names.items()
    }
    if any(
        not name
        or len(name) > DEFAULT_ENTITY_NAME_MAX_LENGTH
        or len(name.encode()) > DEFAULT_ENTITY_NAME_MAX_BYTES
        for name in scoped_names.values()
    ):
        raise ValueError(
            "Entity identity would be empty or truncated by native ingestion"
        )
    if len({canonical(name) for name in scoped_names.values()}) != len(scoped_names):
        raise ValueError("Entity aliases collide in the verified product scope")
    entities = [
        {
            "name": scoped_names[entity.name],
            "type": normalize_type(entity.type),
            "description": description(entity.evidence_ids),
        }
        for entity in selection.entities
    ]
    for entity in selection.entities:
        selected_units = [catalog[key] for key in entity.evidence_ids]
        primary = [unit for unit in selected_units if unit.role == "primary"]
        # Context labels constrain their own use, not an untyped main passage.
        # A copied product-identity heading cannot turn a business rule into a
        # product entity; explicitly typed main evidence still constrains it.
        support = primary or selected_units
        typed_units = [unit for unit in support if unit.entity_types]
        if primary and any(not unit.entity_types for unit in primary):
            continue
        if typed_units and not any(
            normalize_type(entity.type)
            in {normalize_type(kind) for kind in unit.entity_types}
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
                "keywords": profile.relationship_keywords(relation.keywords)
                if profile is not None
                else relation.keywords,
                "description": description(relation.evidence_ids),
            }
        )
    native = {"entities": entities, "relationships": relationships}
    semantic_roles = (
        getattr(profile, "semantic_roles", {}) if profile is not None else {}
    )
    if semantic_roles:
        validate_selection_semantics(selection, chunk, native, roles=semantic_roles)
    return native


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
        ledger_path: Path | None = None,
        ledger_context: str = "",
        replay_only: bool = False,
        profile: ExtractionProfile | None = None,
    ) -> None:
        if not requests:
            raise ValueError("No reviewed extraction inputs")
        self.requests = {prompt.strip(): chunk for prompt, chunk in requests.items()}
        if len(self.requests) != len(requests):
            raise ValueError("Extraction inputs collide after native normalization")
        self.native_system_prompt = native_system_prompt.strip()
        self.summary_pattern = template_pattern(native_summary_template)
        self.type_guidance = type_guidance
        self.selection_guidance = selection_guidance
        self.complete = complete
        self.record = record or (lambda _: None)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.replay_only = replay_only
        self.profile = profile
        self.units = tuple(unit for chunk in requests.values() for unit in chunk.units)
        if profile is not None:
            profile.validate_catalog(self.units)
        self.semantic_roles = (
            getattr(profile, "semantic_roles", {}) if profile is not None else {}
        )
        self.semantic_inputs = (
            {
                prompt: semantic_metadata(chunk.units)
                for prompt, chunk in self.requests.items()
            }
            if self.semantic_roles
            else {}
        )
        self.source_order = {
            source: index
            for index, source in enumerate(
                dict.fromkeys(chunk.source for chunk in requests.values())
            )
        }
        self.expected: dict[
            tuple[str, str], dict[tuple[str, str, int, int], EvidenceUnit]
        ] = {}
        type_instruction = (
            "实体type只用中文：产品、保障计划、保障责任、责任选择条件、合同规则、术语定义、机构、其他。\n"
            if profile is None
            else "实体type只能使用配置类型：" + "、".join(profile.entity_types) + "。\n"
        )
        pointer_schema = PointerResult.model_json_schema()
        if profile is not None:
            pointer_schema["$defs"]["PointerEntity"]["properties"]["type"]["enum"] = (
                list(profile.entity_types)
            )
        self.system_prompt = (
            "你从带编号的完整原文证据中识别实体和直接关系。只返回JSON，不撰写description。\n"
            "每个实体和关系用evidence_ids选择直接支持它的完整证据。"
            "证据上的entity_types是已核原文结构支持的实体类型，实体须选有对应类型支持的证据。"
            "不得输出不存在的编号，不从来源标签或示例抽取保险事实。"
            "关系两端必须在本次entities中。实体名称保持产品及责任范围，组织名称可用原称。\n"
            + type_instruction
            + type_guidance
            + "\n"
            + selection_guidance
            + "\n唯一输出结构：\n"
            + json.dumps(pointer_schema, ensure_ascii=False)
        )
        if profile is not None:
            self.system_prompt += (
                "\n产品根实体类型："
                + profile.product_entity_type
                + "；可使用无产品前缀名称的类型："
                + "、".join(profile.unscoped_entity_types)
                + "；其他业务实体须保留产品及原文明确作用域。"
                "证据role=primary为主原文；其余role仅提供身份、上级结构或明确引用上下文，"
                "不因上下文副本重复出现而重复制造独立事实。"
                "当前请求的证据编号仅为e1、e2等局部编号，必须逐字选用本次输入提供的编号，"
                "不得输出摘要哈希、拼接编号或引用示例编号。"
                "\n关系keywords使用以下关系类型；多个类型用英文逗号分隔："
                + "、".join(profile.relationship_types)
                + "\n"
                + profile.relationship_types_guidance
                + "\n以下仅为抽取方法示例，不是当前产品事实；"
                "证据编号仅在各自输入内有效，不得引用示例编号：\n"
                + json.dumps(
                    [
                        provider_example(example)
                        for example in profile.selected_evidence_examples
                    ],
                    ensure_ascii=False,
                )
            )
        self.request_keys = {
            hashlib.sha256(prompt.encode()).hexdigest(): prompt
            for prompt in self.requests
        }
        fingerprint = {
            "context": ledger_context,
            "system": self.system_prompt,
            "native_system": self.native_system_prompt,
            "summary_template": native_summary_template,
            "requests": [
                {
                    "prompt": key,
                    "product": self.requests[prompt].product,
                    "units": [
                        (unit.identifier, unit.identity, unit.entity_types)
                        + ((unit.role,) if unit.role != "primary" else ())
                        for unit in self.requests[prompt].units
                    ],
                }
                for key, prompt in self.request_keys.items()
            ],
        }
        if profile is not None:
            fingerprint["profile"] = profile.contract
            fingerprint["provider_evidence_id_protocol"] = "local-sequential-v1"
        if self.semantic_roles:
            fingerprint["source_semantics"] = self.semantic_inputs
            fingerprint["source_semantics_implementation_sha256"] = hashlib.sha256(
                Path(__file__).with_name("evidence_semantics.py").read_bytes()
            ).hexdigest()
        self.contract_hash = hashlib.sha256(
            json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        self.selections: dict[str, dict[str, Any]] = {}
        self.validation_errors: dict[str, str] = {}
        if self.ledger_path is not None and self.ledger_path.exists():
            ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if (
                not isinstance(ledger, dict)
                or set(ledger) != {"version", "contract_sha256", "selections"}
                or ledger["version"] != 1
                or ledger["contract_sha256"] != self.contract_hash
                or not isinstance(ledger["selections"], dict)
            ):
                raise ValueError("Evidence ledger contract changed or is invalid")
            for key, row in ledger["selections"].items():
                if key not in self.request_keys or set(row) != {"selection", "native"}:
                    raise ValueError("Evidence ledger contains an unknown selection")
                selected = PointerResult.model_validate(row["selection"])
                native = to_native(
                    selected,
                    self.requests[self.request_keys[key]],
                    profile=self.profile,
                )
                if native != row["native"]:
                    raise ValueError("Evidence ledger native output failed validation")
                self.selections[key] = row

    def persist_selection(
        self, key: str, selection: PointerResult, native: dict[str, Any]
    ) -> None:
        """Publish validated selections before returning native cacheable output."""
        rows = {
            **self.selections,
            key: {"selection": selection.model_dump(), "native": native},
        }
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.ledger_path.parent,
                    prefix=self.ledger_path.name + ".",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    json.dump(
                        {
                            "version": 1,
                            "contract_sha256": self.contract_hash,
                            "selections": rows,
                        },
                        stream,
                        ensure_ascii=False,
                        indent=2,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.ledger_path)
                directory = os.open(self.ledger_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.selections = rows

    def activate_sources(self, sources: Sequence[str]) -> None:
        """Restore current/prior sequential-document contributions on cache hits.

        Call immediately before inserting each document. Never activate future
        documents: the native summary callback does not expose source IDs, so
        only current/prior contributions may participate in restoration.
        """
        known = {chunk.source for chunk in self.requests.values()}
        if not set(sources) <= known:
            raise ValueError("Cannot activate an unknown evidence source")
        for key, row in self.selections.items():
            chunk = self.requests[self.request_keys[key]]
            if chunk.source in sources:
                self.remember(
                    PointerResult.model_validate(row["selection"]), chunk, row["native"]
                )

    def remember(
        self, selection: PointerResult, chunk: EvidenceChunk, native: dict[str, Any]
    ) -> None:
        def add(kind: str, label: str, identifiers: list[str]) -> None:
            target = self.expected.setdefault((kind.casefold(), canonical(label)), {})
            selected = set(identifiers)
            target.update(
                (unit.identity, unit)
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

    def evidence_for_description(
        self, description: str, units: Sequence[EvidenceUnit] | None = None
    ) -> tuple[EvidenceUnit, ...]:
        """Accept only concatenations of complete catalog units, including cache hits."""
        available: dict[str, dict[tuple[str, str, int, int], EvidenceUnit]] = {}
        for unit in self.units if units is None else units:
            # Native JSON extraction normalizes punctuation (including em dashes)
            # before merging. Match only whole original units or their exact
            # native-normalized forms; always return the original source span.
            for variant in (
                unit.text,
                sanitize_and_normalize_extracted_text(unit.text),
            ):
                normalized = canonical(variant)
                if normalized:
                    available.setdefault(normalized, {})[unit.identity] = unit
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
                            return (*available[candidate].values(), *tail)
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
            key = hashlib.sha256(prompt.strip().encode()).hexdigest()
            cached = self.selections.get(key)
            if cached is not None:
                self.remember(
                    PointerResult.model_validate(cached["selection"]),
                    chunk,
                    cached["native"],
                )
                self.record(
                    {
                        "stage": "extract",
                        "state": "replayed",
                        "source": chunk.source,
                        "native": cached["native"],
                        "provider_called": False,
                    }
                )
                return json.dumps(cached["native"], ensure_ascii=False)
            if self.replay_only:
                raise ValueError(
                    "Approved evidence selection is missing; no replay is possible"
                )
            user_payload = {
                "product": chunk.product,
                "source_file": chunk.source,
                "evidence": [
                    {
                        "id": unit.identifier,
                        "text": unit.text,
                        "entity_types": list(unit.entity_types),
                        **(
                            {"role": unit.role}
                            if self.profile is not None or unit.role != "primary"
                            else {}
                        ),
                    }
                    for unit in chunk.units
                ],
            }
            local_to_durable = {}
            if self.profile is not None:
                durable_to_local = {
                    unit.identifier: f"e{index}"
                    for index, unit in enumerate(chunk.units, 1)
                }
                local_to_durable = {
                    local: durable for durable, local in durable_to_local.items()
                }
                for evidence in user_payload["evidence"]:
                    evidence["id"] = durable_to_local[evidence["id"]]
            if self.semantic_roles:
                user_payload["source_semantics"] = _map_references(
                    self.semantic_inputs[prompt.strip()], durable_to_local
                )
            if key in self.validation_errors:
                user_payload["previous_validation_error"] = self.validation_errors[key]
                user_payload["correction_instruction"] = (
                    "根据同一原文证据修正上述结构或证据引用错误；不增加无依据事实。"
                )
            user = json.dumps(user_payload, ensure_ascii=False)
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
                if self.profile is not None:
                    selection = PointerResult.model_validate(
                        _map_references(selection.model_dump(), local_to_durable)
                    )
                native = to_native(selection, chunk, profile=self.profile)
                self.persist_selection(key, selection, native)
                self.remember(selection, chunk, native)
                self.validation_errors.pop(key, None)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    self.validation_errors[key] = (
                        f"{type(exc).__name__}: {str(exc)[:500]}"
                    )
                self.record(
                    {
                        **event,
                        "state": "invalid_or_unknown",
                        "response": raw,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "validation_issues": getattr(exc, "issues", []),
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
        try:
            supplied = [
                unit
                for item in descriptions
                for unit in self.evidence_for_description(
                    item["Description"], tuple(expected.values())
                )
            ]
        except ValueError as exc:
            raise ValueError(
                "Merge evidence was not selected for this graph object"
            ) from exc
        supplied_keys = {unit.identity for unit in supplied}
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
