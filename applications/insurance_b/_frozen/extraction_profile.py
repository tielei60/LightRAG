"""Validated configuration for evidence extraction taxonomies and examples."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any


def _label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
        or "<SEP>" in value
        or "," in value
        or "，" in value
    ):
        raise ValueError(f"{field} must contain nonempty single-line labels")
    return value


def _labels(value: Any, field: str, *, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (required and not value):
        raise ValueError(f"{field} must be {'a nonempty' if required else 'a'} list")
    result = tuple(_label(item, field) for item in value)
    if len({item.casefold() for item in result}) != len(result):
        raise ValueError(f"{field} contains duplicate labels")
    return result


def _aliases(value: Any, values: tuple[str, ...], field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    registered = {kind.casefold(): kind for kind in values}
    aliases: dict[str, str] = {}
    for alias, target in value.items():
        alias = _label(alias, field)
        target = _label(target, field)
        if target not in values:
            raise ValueError(f"{field} target is not a registered type: {target}")
        key = alias.casefold()
        if key in registered and registered[key] != target:
            raise ValueError(f"{field} shadows a canonical type: {alias}")
        if key in aliases and aliases[key] != target:
            raise ValueError(f"{field} contains conflicting aliases: {alias}")
        aliases[key] = target
    return aliases


@dataclass(frozen=True)
class ExtractionProfile:
    """Keep configurable vocabularies outside the native LightRAG protocol.

    Relationship labels use the existing comma-separated ``keywords`` field.
    Examples demonstrate the pointer contract and are validated before a model
    call. Changing any profile value changes the evidence-ledger fingerprint.
    """

    entity_types: tuple[str, ...]
    entity_type_aliases: dict[str, str]
    product_entity_type: str
    unscoped_entity_types: tuple[str, ...]
    relationship_types: tuple[str, ...]
    relationship_type_aliases: dict[str, str]
    relationship_types_guidance: str
    evidence_selection_examples: tuple[dict[str, Any], ...]
    semantic_roles: dict[str, tuple[str, ...]] = dataclass_field(default_factory=dict)
    evidence_selection_example_names: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExtractionProfile:
        """Reject unknown profile keys, dangling aliases, and invalid examples."""
        if not isinstance(value, Mapping):
            raise TypeError("Extraction profile must be a mapping")
        allowed = {
            "entity_types",
            "entity_type_aliases",
            "product_entity_type",
            "unscoped_entity_types",
            "relationship_types",
            "relationship_type_aliases",
            "relationship_types_guidance",
            "evidence_selection_examples",
            "entity_types_guidance",
            "evidence_selection_guidance",
            "entity_extraction_examples",
            "entity_extraction_json_examples",
            "semantic_roles",
            "evidence_selection_example_names",
        }
        if unknown := set(value) - allowed:
            raise ValueError(f"Unknown extraction profile keys: {sorted(unknown)}")
        for field in (
            "entity_types_guidance",
            "evidence_selection_guidance",
            "relationship_types_guidance",
        ):
            if field not in value:
                continue
            if not isinstance(value[field], str):
                raise TypeError(f"{field} must be a nonempty string")
            if not value[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        for field in ("entity_extraction_examples", "entity_extraction_json_examples"):
            if field not in value:
                continue
            if not isinstance(value[field], list):
                raise TypeError(f"{field} must be a list of nonempty strings")
            if any(
                not isinstance(item, str) or not item.strip() for item in value[field]
            ):
                raise ValueError(f"{field} must be a list of nonempty strings")
        types = _labels(value.get("entity_types"), "entity_types", required=True)
        aliases = _aliases(
            value.get("entity_type_aliases", {}), types, "entity_type_aliases"
        )
        product = _label(value.get("product_entity_type"), "product_entity_type")
        if product not in types:
            raise ValueError("product_entity_type must be a registered entity type")
        unscoped = _labels(
            value.get("unscoped_entity_types", []), "unscoped_entity_types"
        )
        if not set(unscoped) <= set(types) or product in unscoped:
            raise ValueError(
                "unscoped_entity_types must be registered non-product types"
            )
        relations = _labels(
            value.get("relationship_types"), "relationship_types", required=True
        )
        relation_aliases = _aliases(
            value.get("relationship_type_aliases", {}),
            relations,
            "relationship_type_aliases",
        )
        guidance = value.get("relationship_types_guidance", "")
        if not isinstance(guidance, str):
            raise TypeError("relationship_types_guidance must be a string")
        examples = value.get("evidence_selection_examples", [])
        if not isinstance(examples, list):
            raise TypeError("evidence_selection_examples must be a list")
        roles_value = value.get("semantic_roles", {})
        if not isinstance(roles_value, Mapping):
            raise TypeError("semantic_roles must be a mapping")
        roles = {}
        if roles_value:
            if set(roles_value) != {"plan", "coverage", "inclusion", "scope_option"}:
                raise ValueError(
                    "semantic_roles requires plan, coverage, inclusion, scope_option"
                )
            for name, labels in roles_value.items():
                labels = _labels(labels, f"semantic_roles.{name}", required=True)
                registered = relations if name == "inclusion" else types
                if not set(labels) <= set(registered):
                    raise ValueError(
                        f"semantic_roles.{name} references an unregistered label"
                    )
                roles[name] = labels
        selected_names = None
        if value.get("evidence_selection_example_names") is not None:
            selected_names = _labels(
                value["evidence_selection_example_names"],
                "evidence_selection_example_names",
                required=True,
            )
        profile = cls(
            types,
            aliases,
            product,
            unscoped,
            relations,
            relation_aliases,
            guidance,
            tuple(copy.deepcopy(examples)),
            roles,
            selected_names,
        )
        profile.validate_examples()
        _ = profile.selected_evidence_examples
        return profile

    @property
    def selected_evidence_examples(self) -> tuple[dict[str, Any], ...]:
        """Use an explicit, stable example subset while validating the whole bank."""
        if self.evidence_selection_example_names is None:
            return self.evidence_selection_examples
        examples = {}
        for example in self.evidence_selection_examples:
            name = example["input"]["product"]
            if name in examples:
                raise ValueError(
                    "Named evidence examples require unique product labels"
                )
            examples[name] = example
        if not set(self.evidence_selection_example_names) <= examples.keys():
            raise ValueError(
                "evidence_selection_example_names references an unknown example"
            )
        return tuple(examples[name] for name in self.evidence_selection_example_names)

    @property
    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "entity_types": list(self.entity_types),
                "entity_type_aliases": self.entity_type_aliases,
                "product_entity_type": self.product_entity_type,
                "unscoped_entity_types": list(self.unscoped_entity_types),
                "relationship_types": list(self.relationship_types),
                "relationship_type_aliases": self.relationship_type_aliases,
                "relationship_types_guidance": self.relationship_types_guidance,
                "evidence_selection_examples": list(self.evidence_selection_examples),
                "semantic_roles": {
                    name: list(values) for name, values in self.semantic_roles.items()
                },
                "evidence_selection_example_names": list(
                    self.evidence_selection_example_names
                )
                if self.evidence_selection_example_names is not None
                else None,
            }
        )

    def entity_type(self, value: str) -> str:
        registered = {kind.casefold(): kind for kind in self.entity_types}
        key = value.casefold()
        if key in registered:
            return registered[key]
        if key in self.entity_type_aliases:
            return self.entity_type_aliases[key]
        raise ValueError(f"Unsupported evidence entity type: {value}")

    def relationship_keywords(self, value: str) -> str:
        """Validate each keyword, allowing several labels on one native edge."""
        registered = {kind.casefold(): kind for kind in self.relationship_types}
        result = []
        for keyword in re.split("[,，]", value):
            key = keyword.strip().casefold()
            if not key:
                raise ValueError("Relationship keywords contain an empty label")
            label = registered.get(key) or self.relationship_type_aliases.get(key)
            if label is None:
                raise ValueError(
                    f"Unsupported evidence relationship type: {keyword.strip()}"
                )
            if label not in result:
                result.append(label)
        return ",".join(result)

    def validate_catalog(self, units: Sequence[Any]) -> None:
        for unit in units:
            for kind in unit.entity_types:
                self.entity_type(kind)

    def validate_examples(self) -> None:
        # Import lazily: the adapter consumes this profile while examples reuse
        # its existing source, endpoint, naming, and evidence-reference checks.
        from applications.insurance_b._frozen.evidence_extract import (
            EvidenceChunk,
            EvidenceUnit,
            PointerResult,
            to_native,
        )

        for index, example in enumerate(self.evidence_selection_examples, start=1):
            try:
                if not isinstance(example, dict) or set(example) != {"input", "output"}:
                    raise ValueError("Example requires exactly input and output")
                source = example["input"]
                if not isinstance(source, dict) or set(source) != {
                    "product",
                    "source_file",
                    "evidence",
                }:
                    raise ValueError(
                        "Example input requires product, source_file, evidence"
                    )
                product = _label(source["product"], "example product")
                filename = _label(source["source_file"], "example source_file")
                rows = source["evidence"]
                if not isinstance(rows, list) or not rows:
                    raise ValueError("Example evidence must be a nonempty list")
                original = []
                for row in rows:
                    if not isinstance(row, dict) or not {"id", "text"} <= set(row) <= {
                        "id",
                        "text",
                        "entity_types",
                        "role",
                    }:
                        raise ValueError("Example evidence requires id and text")
                    _label(row["id"], "example evidence id")
                    if not isinstance(row["text"], str) or not row["text"].strip():
                        raise ValueError("Example evidence text must be nonempty")
                    _labels(
                        row.get("entity_types", []), "example evidence entity_types"
                    )
                    original.append(row["text"])
                text = "\n\n".join(original)
                units, cursor = [], 0
                for row in rows:
                    end = cursor + len(row["text"])
                    units.append(
                        EvidenceUnit(
                            row["id"],
                            filename,
                            cursor,
                            end,
                            text,
                            tuple(row.get("entity_types", [])),
                            row.get("role", "primary"),
                        )
                    )
                    cursor = end + 2
                chunk = EvidenceChunk(filename, product, text, tuple(units))
                to_native(
                    PointerResult.model_validate(example["output"]), chunk, profile=self
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid evidence selection example {index}: {exc}"
                ) from exc
