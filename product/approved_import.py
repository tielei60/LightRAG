"""Offline, independently pinned extraction replay into a fresh native candidate.

This is a bounded local-storage rebuild, not an update or a production approval
service. A reviewer supplies selections and a separate approval pin. The adapter
never calls a model. Exact quotations live in the prepared ledger and chunks;
native graph descriptions undergo the pinned native normalization.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import platform
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.chunker import chunking_by_token_size
from lightrag.llm_roles import RoleLLMConfig
from lightrag.prompt import PROMPTS, resolve_entity_extraction_prompt_profile
from lightrag.utils import (
    EmbeddingFunc,
    TiktokenTokenizer,
    compute_mdhash_id,
    make_relation_chunk_key,
    normalize_entity_name,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
)
from lightrag.utils_pipeline import (
    build_chunks_dict_from_chunking_result,
    normalize_document_file_path,
)
from product.evidence_extract import template_pattern


class ApprovalError(ValueError):
    """The approved local replay contract cannot be proven."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


@dataclass(frozen=True)
class ImportConfig:
    """Every supported runtime choice is pinned; no provider configuration."""

    embedding_model: str = "synthetic-offline-v1"
    embedding_dim: int = 32
    embedding_token_limit: int = 131072
    tokenizer_model: str = "gpt-4o-mini"
    chunk_token_size: int = 8192
    summary_context_size: int = 131072
    max_sources: int = 10000
    max_records: int = 10000
    language: str = "Chinese"

    def __post_init__(self):
        require(
            all(
                type(value) is int and value > 0
                for key, value in asdict(self).items()
                if key not in {"embedding_model", "tokenizer_model", "language"}
            ),
            "Runtime limits must be positive integers",
        )
        require(bool(self.embedding_model), "Embedding identity is required")


def runtime_contract(config: ImportConfig) -> dict:
    root = Path(__file__).resolve().parents[1]
    names = sorted(
        str(path.relative_to(root)) for path in (root / "lightrag").rglob("*.py")
    )
    names += [
        "product/approved_import.py",
        "product/import_units.py",
        "product/prepare_context.py",
        "product/evidence_extract.py",
        "product/requirements.lock",
        "pyproject.toml",
    ]
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
        if (root / name).is_file()
    }
    profile = resolve_entity_extraction_prompt_profile({}, True)
    context = dict(
        entity_types_guidance=profile["entity_types_guidance"],
        examples="\n".join(profile["entity_extraction_json_examples"]),
        language=config.language,
        max_total_records=config.max_records,
        max_entity_records=config.max_records,
    )
    return {
        "config": asdict(config),
        "files": files,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "networkx",
                "nano-vectordb",
                "tiktoken",
                "pydantic",
                "json-repair",
            )
        },
        "profile": profile,
        "system_prompt": sanitize_text_for_encoding(
            PROMPTS["entity_extraction_json_system_prompt"].format(**context)
        ),
        "user_template": PROMPTS["entity_extraction_json_user_prompt"],
        "summary_template": PROMPTS["summarize_entity_descriptions"],
        "context": context,
        "policy": "fresh-local-only; native-cache; source-bound-complete-merge-v1",
    }


_TYPES = {
    alias: canonical
    for canonical, aliases in {
        "产品": ("Product", "产品", "保险产品"),
        "保障计划": ("Plan", "计划", "投保计划", "保障计划"),
        "保障责任": ("Coverage", "保险责任", "责任", "保障责任"),
        "责任选择条件": ("UnderwritingRule", "投保规则", "责任选择条件"),
        "合同规则": ("Rule", "规则", "合同规则"),
        "术语定义": ("Concept", "概念", "术语", "术语定义"),
        "机构": ("Organization", "机构", "组织"),
        "其他": ("Other", "其他"),
    }.items()
    for alias in aliases
}
_MARKER = "【原文位置:"


def _scope_identity(scope: dict) -> list:
    combinations = scope.get("combinations")
    require(
        isinstance(combinations, list) and bool(combinations), "Explicit scope required"
    )
    result = []
    for combination in combinations:
        item = {}
        for axis in ("plan", "responsibility"):
            value = combination.get(axis, {})
            require(
                value.get("kind") in {"specific", "universal", "not_applicable"},
                "Unknown scope is unsupported",
            )
            if value["kind"] == "specific":
                require(
                    isinstance(value.get("value"), str) and bool(value["value"]),
                    "Specific scope requires a label",
                )
            item[axis] = value
        result.append(item)
    return sorted(result, key=digest)


def _location(span: dict) -> tuple:
    return tuple(
        span.get(key)
        for key in (
            "source_id",
            "document_version",
            "original_sha256",
            "extracted_text_sha256",
            "parser_sha256",
            "start",
            "end",
        )
    )


def _render(ids: list[str] | set[str], evidence: dict, *, normalized=False) -> str:
    ordered = sorted(set(ids), key=lambda key: _location(evidence[key]))
    value = "\n\n".join(
        f"{_MARKER}{key}】\n{evidence[key]['quote']}\n【原文结束:{key}】"
        for key in ordered
    )
    return sanitize_and_normalize_extracted_text(value) if normalized else value


def _native_name(entity: dict, unit: dict, product: dict) -> str:
    require(entity.get("type") in _TYPES, "Unsupported entity type")
    require(
        entity.get("scope") == unit["scope"],
        "Selection scope differs from reviewed unit",
    )
    require(
        isinstance(entity.get("entity_id"), str) and bool(entity["entity_id"]),
        "Reviewer-supplied entity identity required",
    )
    label = entity.get("name", "")
    require(
        isinstance(label, str) and label.strip() == label and bool(label),
        "Invalid entity label",
    )
    require(
        normalize_entity_name(label) == label,
        "Entity label changes under native normalization",
    )
    identity = {
        "product_id": product["id"],
        "scope": _scope_identity(unit["scope"]),
        "entity_id": entity["entity_id"],
        "type": _TYPES[entity["type"]],
        "document_version": unit["document_version"],
    }
    name = f"{product['label']}｜{label}【{digest(identity)[:20]}】"
    # The native parser truncates identifiers. Never accept that silently.
    from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH

    require(
        len(name) <= DEFAULT_ENTITY_NAME_MAX_LENGTH
        and normalize_entity_name(name) == name,
        "Scoped graph identifier would normalize or truncate",
    )
    return name


def build_review_packet(
    prepared: dict, selections: dict[str, dict], config: ImportConfig | None = None
) -> dict:
    """Compile mechanical checks only. This function DOES NOT approve content."""
    config = config or ImportConfig()
    require(prepared.get("schema_version") == 1, "Unsupported prepared schema")
    require(
        prepared.get("prepared_sha256")
        == digest({k: v for k, v in prepared.items() if k != "prepared_sha256"}),
        "Prepared input drift",
    )
    require(
        prepared.get("tokenizer", {}).get("model") == config.tokenizer_model,
        "Prepared/native tokenizer configuration differs",
    )
    runtime = runtime_contract(config)
    tokenizer = TiktokenTokenizer(model_name=config.tokenizer_model)
    evidence: dict[str, dict] = {}
    locations: dict[tuple, str] = {}
    objects: dict[str, dict] = {}
    entity_identities, relation_identities = {}, {}
    source_catalog = {
        (source["source_id"], source["document_version"]): source
        for source in prepared["sources"]
    }
    require(
        len(source_catalog) == len(prepared["sources"]),
        "Duplicate prepared source identity",
    )
    chunks, documents, requests = {}, [], {}
    units = [u for doc in prepared["documents"] for u in doc["units"]]
    require(
        set(selections) == {u["unit_id"] for u in units},
        "Incomplete extraction selections",
    )
    require(len(units) <= config.max_sources, "Candidate exceeds provenance cap")
    labels = [
        normalize_document_file_path(d["payload"]["file_source"])
        for d in prepared["documents"]
    ]
    require(len(labels) == len(set(labels)), "Native source label collision")
    for document in prepared["documents"]:
        source = source_catalog.get(
            (document["source_id"], document["document_version"])
        )
        require(source is not None, "Prepared document lacks a source binding")
        payload = document["payload"]
        text = payload["text"]
        require(
            sanitize_text_for_encoding(text) == text and text.strip() == text,
            "Prepared document changes during native text sanitization",
        )
        doc_id = (
            "doc-" + digest([document["source_id"], document["document_version"]])[:32]
        )
        params = payload["chunking"]["params"]
        require(
            params.get("split_by_character_only") is True
            and params.get("split_by_character"),
            "Only explicit prepared fixed-token boundaries are supported",
        )
        split = chunking_by_token_size(
            tokenizer,
            text,
            split_by_character=params["split_by_character"],
            split_by_character_only=True,
            chunk_overlap_token_size=0,
            chunk_token_size=config.chunk_token_size,
        )
        require(
            [c["content"] for c in split] == [u["content"] for u in document["units"]],
            "Native chunks differ from reviewed units",
        )
        require(
            all(
                c["tokens"] == u["tokens"]
                for c, u in zip(split, document["units"], strict=True)
            ),
            "Prepared/native tokenizer counts differ",
        )
        require(
            all(c["tokens"] <= config.chunk_token_size for c in split), "Oversize unit"
        )
        native_chunks = build_chunks_dict_from_chunking_result(
            split,
            doc_id=doc_id,
            file_path=normalize_document_file_path(payload["file_source"]),
        )
        documents.append(
            {
                "doc_id": doc_id,
                "file_path": normalize_document_file_path(payload["file_source"]),
                "text": text,
                "split_by_character": params["split_by_character"],
                "chunk_ids": list(native_chunks),
            }
        )
        for (chunk_id, chunk), unit in zip(
            native_chunks.items(), document["units"], strict=True
        ):
            require(
                hashlib.sha256(unit["content"].encode()).hexdigest()
                == unit["content_sha256"],
                "Prepared chunk hash drift",
            )
            require(
                unit["source_id"] == document["source_id"]
                and unit["document_version"] == document["document_version"]
                and unit["product"] == source["product"],
                "Unit source/product/version binding differs",
            )
            catalog = {}
            for span in unit["primary"] + unit["context"]:
                key = span["evidence_id"]
                require(
                    span["quote"] in unit["content"],
                    "Evidence missing from native chunk",
                )
                require(
                    _MARKER not in span["quote"] and "<SEP>" not in span["quote"],
                    "Reserved provenance delimiter in source",
                )
                require(
                    sanitize_text_for_encoding("[" + span["quote"] + "]")
                    == "[" + span["quote"] + "]",
                    "Unsupported source control characters",
                )
                identity = {
                    key: span[key]
                    for key in (
                        "source_id",
                        "document_version",
                        "original_sha256",
                        "extracted_text_sha256",
                        "parser_sha256",
                        "start",
                        "end",
                    )
                }
                require(
                    key == "evidence-" + digest(identity),
                    "Evidence location identity drift",
                )
                require(
                    span["source_id"] == document["source_id"]
                    and span["document_version"] == document["document_version"]
                    and span["original_sha256"] == source["original_sha256"]
                    and span["extracted_text_sha256"] == source["extracted_text_sha256"]
                    and span["parser_sha256"] == source["parser_binding"]["sha256"],
                    "Evidence source/parser/version binding differs",
                )
                location = _location(span)
                require(
                    all(v is not None for v in location),
                    "Incomplete original-source location",
                )
                stable = {
                    k: span[k]
                    for k in (
                        "evidence_id",
                        "source_id",
                        "document_version",
                        "original_sha256",
                        "extracted_text_sha256",
                        "parser_sha256",
                        "start",
                        "end",
                        "quote",
                        "physical_pages",
                    )
                }
                require(
                    key not in evidence or evidence[key] == stable,
                    "Evidence identity collision",
                )
                require(
                    location not in locations or locations[location] == key,
                    "Location alias collision",
                )
                evidence[key], locations[location], catalog[key] = stable, key, span
            selection = selections[unit["unit_id"]]
            require(
                set(selection)
                in (
                    {"entities", "relationships"},
                    {"entities", "relationships", "mode"},
                ),
                "Unknown selection fields",
            )
            text_only = selection.get("mode") == "text_only"
            require(
                selection.get("mode") in (None, "graph", "text_only"),
                "Unknown selection mode",
            )
            require(
                not text_only
                or not selection["entities"]
                and not selection["relationships"],
                "Explicit text-only approval cannot contain graph records",
            )
            native = {"entities": [], "relationships": []}
            by_id = {}
            product = unit["product"]
            for entity in selection["entities"]:
                name = _native_name(entity, unit, product)
                require(
                    entity["entity_id"] not in by_id,
                    "Duplicate entity identity in chunk",
                )
                by_id[entity["entity_id"]] = name
                kind = _TYPES[entity["type"]]
                identity_key = digest(
                    [
                        product["id"],
                        unit["document_version"],
                        _scope_identity(unit["scope"]),
                        entity["entity_id"],
                    ]
                )
                require(
                    identity_key not in entity_identities
                    or entity_identities[identity_key] == (name, kind),
                    "Stable entity identity has conflicting labels or types",
                )
                entity_identities[identity_key] = (name, kind)
                native["entities"].append(
                    {
                        "name": name,
                        "type": kind,
                        "description": _selected_description(
                            entity, catalog, unit, evidence
                        ),
                    }
                )
                key = "Entity:" + name
                definition = {
                    "kind": "Entity",
                    "name": name,
                    "entity_type": kind,
                    "label": entity["name"],
                    "identity": entity["entity_id"],
                    "scope": _scope_identity(unit["scope"]),
                }
                _contribute(objects, key, definition, chunk_id, entity["evidence_ids"])
            pairs = set()
            for relation in selection["relationships"]:
                require(
                    relation.get("scope") == unit["scope"], "Relation scope mismatch"
                )
                require(
                    relation.get("source") in by_id and relation.get("target") in by_id,
                    "Missing relationship endpoints",
                )
                pair = tuple(
                    sorted((by_id[relation["source"]], by_id[relation["target"]]))
                )
                require(
                    pair[0] != pair[1] and pair not in pairs,
                    "Duplicate or self relationship",
                )
                pairs.add(pair)
                require(
                    bool(relation.get("relation_id")),
                    "Explicit relationship identity required",
                )
                keywords = relation.get("keywords", "")
                require(
                    isinstance(keywords, str)
                    and bool(keywords)
                    and sanitize_and_normalize_extracted_text(keywords, True)
                    == keywords,
                    "Invalid or normalizing relation keywords",
                )
                native["relationships"].append(
                    {
                        "source": pair[0],
                        "target": pair[1],
                        "keywords": keywords,
                        "description": _selected_description(
                            relation, catalog, unit, evidence
                        ),
                    }
                )
                identity_key = digest(
                    [
                        product["id"],
                        unit["document_version"],
                        _scope_identity(unit["scope"]),
                        relation["relation_id"],
                    ]
                )
                require(
                    identity_key not in relation_identities
                    or relation_identities[identity_key] == pair,
                    "Stable relationship identity has conflicting endpoint pairs",
                )
                relation_identities[identity_key] = pair
                key = "Relation:" + make_relation_chunk_key(*pair)
                definition = {
                    "kind": "Relation",
                    "name": f"({pair[0]}, {pair[1]})",
                    "pair": pair,
                    "identity": relation["relation_id"],
                    "scope": _scope_identity(unit["scope"]),
                    "keywords": ",".join(
                        sorted(
                            {
                                x.strip()
                                for x in keywords.replace("，", ",").split(",")
                                if x.strip()
                            }
                        )
                    ),
                }
                _contribute(
                    objects, key, definition, chunk_id, relation["evidence_ids"]
                )
            require(
                (text_only or bool(native["entities"]))
                and len(native["entities"]) + len(native["relationships"])
                <= config.max_records,
                "Empty or oversized reviewed extraction",
            )
            prompt = runtime["user_template"].format(
                **runtime["context"],
                input_text=unit["content"],
                heading_context_block="",
            )
            prompt = sanitize_text_for_encoding(prompt)
            require(prompt not in requests, "Ambiguous native extraction prompt")
            requests[prompt] = {
                "chunk_id": chunk_id,
                "native": native,
                "unit_id": unit["unit_id"],
            }
            chunks[chunk_id] = chunk
    require(
        len({doc["doc_id"] for doc in documents}) == len(documents),
        "Duplicate document identity",
    )
    # Conservative JSONL budget including every raw contribution and a previous
    # merged prefix. This prevents native pre-callback truncation and map-reduce.
    for obj in objects.values():
        fragments = [
            _render(ids, evidence, normalized=True)
            for ids in obj["contributions"].values()
        ]
        combined = _render(
            {i for ids in obj["contributions"].values() for i in ids}, evidence
        )
        budget_text = "\n".join(
            json.dumps({"Description": x}, ensure_ascii=False)
            for x in fragments + [combined]
        )
        require(
            len(tokenizer.encode(budget_text)) < config.summary_context_size,
            "Merge exceeds complete serialized-description budget",
        )
        prefix = (
            obj["name"] + "\n"
            if obj["kind"] == "Entity"
            else obj["keywords"] + "\t" + "\n".join(obj["pair"]) + "\n"
        )
        require(
            len(tokenizer.encode(prefix + combined)) < config.embedding_token_limit,
            "Complete vector payload exceeds embedding budget",
        )
    require(
        all(
            len(tokenizer.encode(c["content"])) < config.embedding_token_limit
            for c in chunks.values()
        ),
        "Chunk vector payload exceeds embedding budget",
    )
    packet = {
        "schema_version": 1,
        "prepared": prepared,
        "selections": selections,
        "runtime": runtime,
        "documents": documents,
        "chunks": chunks,
        "requests": requests,
        "evidence": evidence,
        "objects": objects,
        "semantic_status": "UNAPPROVED: mechanical checks are not semantic review",
    }
    return json.loads(json.dumps(packet, ensure_ascii=False))


def _selected_description(
    record: dict, catalog: dict, unit: dict, evidence: dict
) -> str:
    ids = record.get("evidence_ids", [])
    require(
        isinstance(ids, list)
        and bool(ids)
        and len(ids) == len(set(ids))
        and set(ids) <= catalog.keys(),
        "Unknown, duplicate or out-of-scope evidence",
    )
    # Bounded initial policy: every selected object carries the complete reviewed
    # primary unit and dependency closure. A reviewer can split units, not drop
    # a condition/exception or infer completeness from individual references.
    require(
        set(ids) == set(catalog),
        "Selection omits reviewed primary or dependency evidence",
    )
    return _render(ids, evidence)


def _contribute(objects, key, definition, chunk_id, ids):
    if key in objects:
        require(
            {k: v for k, v in objects[key].items() if k != "contributions"}
            == definition,
            "Distinct reviewed graph identities collapse to one native object",
        )
    else:
        objects[key] = {**definition, "contributions": {}}
    require(
        chunk_id not in objects[key]["contributions"], "Duplicate graph contribution"
    )
    objects[key]["contributions"][chunk_id] = list(ids)


def seal_review(packet: dict, approval: dict, *, expected_packet_sha256: str) -> dict:
    """Record an EXTERNAL approval, never manufacture semantic approval.

    Approval must carry an independently retained pin, a reviewer, every
    unit, and explicit scope/evidence review. Authentication of that reviewer
    belongs to the caller; a self-asserted JSON name is not an authenticated signoff.
    """
    require(digest(packet) == expected_packet_sha256, "Independent packet pin mismatch")
    require(
        approval.get("packet_sha256") == expected_packet_sha256
        and approval.get("decision") == "approved"
        and bool(approval.get("reviewer"))
        and bool(approval.get("selection_author"))
        and approval.get("scope_reviewed") is True
        and approval.get("evidence_reviewed") is True,
        "Explicit independent scope/evidence approval required",
    )
    require(
        set(approval.get("reviewed_unit_ids", [])) == set(packet["selections"]),
        "Incomplete independent review",
    )
    value = {
        "packet": packet,
        "approval": approval,
        "packet_sha256": expected_packet_sha256,
    }
    return {**value, "seal_sha256": digest(value)}


def validate_seal(sealed: dict, *, expected_seal_sha256: str) -> dict:
    require(
        sealed.get("seal_sha256")
        == expected_seal_sha256
        == digest({k: v for k, v in sealed.items() if k != "seal_sha256"}),
        "Seal pin mismatch",
    )
    packet = sealed["packet"]
    seal_review(
        packet, sealed["approval"], expected_packet_sha256=sealed["packet_sha256"]
    )
    config = ImportConfig(**packet["runtime"]["config"])
    require(
        runtime_contract(config) == packet["runtime"],
        "Native runtime/configuration drift",
    )
    rebuilt = build_review_packet(packet["prepared"], packet["selections"], config)
    require(rebuilt == packet, "Compiled replay differs from reviewed packet")
    return packet


class ApprovedReplay:
    """Exact extraction replay plus tracking-bound deterministic complete merge."""

    def __init__(self, packet: dict, record=lambda event: None):
        self.packet, self.record, self.rag = packet, record, None
        self.pattern = template_pattern(packet["runtime"]["summary_template"])
        self.summary_objects = {}
        for key, obj in packet["objects"].items():
            self.summary_objects[(obj["kind"], obj["name"])] = key
            if obj["kind"] == "Relation":
                a, b = obj["pair"]
                self.summary_objects[("Relation", f"({b}, {a})")] = key

    def description_ids(self, description: str) -> set[str]:
        # Parse only the fixed metadata protocol, never business prose.
        remaining, found = description, set()
        while remaining:
            matches = []
            for key in self.packet["evidence"]:
                for value in {
                    _render([key], self.packet["evidence"]),
                    _render([key], self.packet["evidence"], normalized=True),
                }:
                    if remaining == value or remaining.startswith(value + "\n\n"):
                        matches.append((len(value), key))
            require(bool(matches), "Unknown or partial merge evidence")
            length, key = max(matches)
            require(key not in found, "Repeated original location in merge fragment")
            found.add(key)
            remaining = remaining[length:]
            if remaining:
                require(remaining.startswith("\n\n"), "Unknown merge separator")
                remaining = remaining[2:]
        return found

    async def __call__(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        require(not history_messages, "Gleaning/history is not approved")
        require(
            set(kwargs) <= {"hashing_kv", "response_format"}, "Unknown callback options"
        )
        request = self.packet["requests"].get(prompt)
        if request is not None:
            require(
                system_prompt == self.packet["runtime"]["system_prompt"]
                and kwargs.get("response_format") == {"type": "json_object"},
                "Extraction prompt or response format drift",
            )
            self.record({"stage": "extract_replay", "chunk_id": request["chunk_id"]})
            return json.dumps(request["native"], ensure_ascii=False)
        match = self.pattern.fullmatch(prompt)
        require(
            match is not None
            and system_prompt is None
            and not kwargs.get("response_format"),
            "Unknown callback; provider fallback is forbidden",
        )
        require(
            match["language"] == self.packet["runtime"]["config"]["language"]
            and match["summary_length"] == "200",
            "Summary configuration drift",
        )
        key = self.summary_objects.get(
            (match["description_type"], match["description_name"])
        )
        require(
            key is not None and self.rag is not None,
            "Unknown summary object or unbound native instance",
        )
        obj = self.packet["objects"][key]
        storage = (
            self.rag.entity_chunks
            if obj["kind"] == "Entity"
            else self.rag.relation_chunks
        )
        storage_key = (
            obj["name"]
            if obj["kind"] == "Entity"
            else make_relation_chunk_key(*obj["pair"])
        )
        tracking = await storage.get_by_id(storage_key)
        require(
            isinstance(tracking, dict) and bool(tracking.get("chunk_ids")),
            "Missing native contribution tracking",
        )
        actual = tracking["chunk_ids"]
        require(
            len(actual) == len(set(actual))
            and set(actual) <= obj["contributions"].keys(),
            "Unapproved native contribution",
        )
        expected = {e for chunk in actual for e in obj["contributions"][chunk]}
        supplied = set()
        for line in match["description_list"].splitlines():
            value = json.loads(line)
            require(set(value) == {"Description"}, "Summary JSONL protocol drift")
            supplied.update(self.description_ids(value["Description"]))
        # Tracking alone is not proof: every currently participating location
        # must also survive the native prompt. Never restore omitted evidence.
        require(
            supplied == expected,
            "Summary omits or adds participating evidence; refusing restoration",
        )
        result = _render(supplied, self.packet["evidence"])
        self.record(
            {
                "stage": "merge",
                "object": key,
                "chunk_ids": actual,
                "evidence_ids": sorted(supplied),
                "provider_called": False,
            }
        )
        return result


async def no_model(*args, **kwargs):
    raise ApprovalError(
        "Live/default model calls are forbidden in approved offline replay"
    )


async def synthetic_embedding(texts, **kwargs):
    """Deterministic test-only vectors, not a semantic embedding model."""
    import numpy as np

    dim = int(kwargs.get("dimensions", 32))
    rows = []
    for text in texts:
        # Positive stable hash features yield nonzero similarities for smoke.
        raw = hashlib.shake_256(text.encode()).digest(dim)
        vector = np.asarray([1 + x / 255 for x in raw], dtype=np.float32)
        rows.append(vector / np.linalg.norm(vector))
    return np.asarray(rows)


def make_native_rag(
    packet: dict,
    target: Path,
    replay: ApprovedReplay,
    embedding_func: EmbeddingFunc,
    *,
    keyword_func=no_model,
) -> LightRAG:
    config = ImportConfig(**packet["runtime"]["config"])
    require(
        embedding_func.model_name == config.embedding_model
        and embedding_func.embedding_dim == config.embedding_dim
        and embedding_func.max_token_size == config.embedding_token_limit,
        "Embedding configuration drift",
    )
    workspace = (
        "verified_" + hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:24]
    )

    async def extract_callback(*args, **kwargs):
        return await replay(*args, **kwargs)

    rag = LightRAG(
        working_dir=str(target / "storage"),
        workspace=workspace,
        llm_model_func=no_model,
        llm_model_name="approved-replay",
        llm_model_kwargs={},
        role_llm_configs={
            "extract": RoleLLMConfig(
                func=extract_callback,
                kwargs={},
                max_async=1,
                metadata={"binding": "approved-local-replay", "model": digest(packet)},
            ),
            "query": RoleLLMConfig(func=no_model, kwargs={}),
            "keyword": RoleLLMConfig(func=keyword_func, kwargs={}),
            "vlm": RoleLLMConfig(func=no_model, kwargs={}),
        },
        embedding_func=embedding_func,
        kv_storage="JsonKVStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        doc_status_storage="JsonDocStatusStorage",
        entity_extraction_use_json=True,
        entity_extract_max_gleaning=0,
        entity_extract_max_entities=config.max_records,
        entity_extract_max_records=config.max_records,
        addon_params={
            "language": config.language,
            "chunker": {
                "chunk_token_size": config.chunk_token_size,
                "fixed_token": {
                    "chunk_overlap_token_size": 0,
                    "chunk_token_size": config.chunk_token_size,
                },
            },
        },
        tokenizer=TiktokenTokenizer(model_name=config.tokenizer_model),
        chunk_token_size=config.chunk_token_size,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        max_source_ids_per_entity=config.max_sources,
        max_source_ids_per_relation=config.max_sources,
        source_ids_limit_method="KEEP",
        max_file_paths=config.max_sources,
        summary_context_size=config.summary_context_size,
        summary_max_tokens=1,
        summary_length_recommended=200,
        force_llm_summary_on_merge=3,
        max_total_tokens=config.summary_context_size,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.0},
    )
    require(
        rag._entity_extraction_prompt_profile == packet["runtime"]["profile"],
        "Resolved profile drift",
    )
    replay.rag = rag
    return rag


def offline_embedding(config: ImportConfig | None = None) -> EmbeddingFunc:
    config = config or ImportConfig()
    require(
        config.embedding_model == "synthetic-offline-v1" and config.embedding_dim == 32,
        "CLI supports only synthetic offline embeddings",
    )
    return EmbeddingFunc(
        embedding_dim=32,
        max_token_size=config.embedding_token_limit,
        model_name=config.embedding_model,
        func=synthetic_embedding,
    )


def _write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _event(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def import_candidate(
    sealed: dict,
    target: Path,
    *,
    expected_seal_sha256: str,
    embedding_func: EmbeddingFunc,
) -> dict:
    """One native writer, fresh target only. Uncertain writes require new rebuild."""
    sealed = json.loads(json.dumps(sealed, ensure_ascii=False))
    packet = validate_seal(sealed, expected_seal_sha256=expected_seal_sha256)
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / (target.name + ".import.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ApprovalError("Candidate has an active writer") from exc
        require(
            not target.exists(),
            "Candidate exists; never replay an existing or uncertain import. Read back or rebuild elsewhere",
        )
        target.mkdir(mode=0o700)
        receipt = {
            "state": "prepared",
            "seal_sha256": expected_seal_sha256,
            "production_approved": False,
            "documents": [],
        }
        _write(target / "sealed.json", sealed)
        _write(target / "receipt.json", receipt)
        rag = None
        try:
            replay = ApprovedReplay(
                packet, lambda event: _event(target / "events.jsonl", event)
            )
            rag = make_native_rag(packet, target, replay, embedding_func)
            await rag.initialize_storages()
            require(
                await rag.llm_response_cache.is_empty(),
                "Fresh candidate cache is not empty",
            )
            receipt["state"] = "writing_or_uncertain"
            _write(target / "receipt.json", receipt)
            for doc in packet["documents"]:
                track_id = (
                    "verified-" + digest([expected_seal_sha256, doc["doc_id"]])[:24]
                )
                _event(
                    target / "events.jsonl",
                    {
                        "stage": "before_ainsert",
                        "doc_id": doc["doc_id"],
                        "track_id": track_id,
                    },
                )
                result = await rag.ainsert(
                    doc["text"],
                    ids=doc["doc_id"],
                    file_paths=doc["file_path"],
                    split_by_character=doc["split_by_character"],
                    split_by_character_only=True,
                    track_id=track_id,
                )
                require(result == track_id, "Native tracking ID changed")
                receipt["documents"].append(
                    {"doc_id": doc["doc_id"], "track_id": result}
                )
                _write(target / "receipt.json", receipt)
            report = await verify_native(rag, packet)
            await rag.finalize_storages()
            rag = None
            report = await _persisted_readback(target, expected_seal_sha256)
            receipt.update(state="completed", readback=report)
            _write(target / "receipt.json", receipt)
            return receipt
        except BaseException as exc:
            receipt.update(
                state="uncertain",
                error_type=type(exc).__name__,
                failure_stage=receipt["state"],
            )
            _write(target / "receipt.json", receipt)
            raise
        finally:
            if rag is not None:
                await rag.finalize_storages()


async def verify_native(rag: LightRAG, packet: dict) -> dict:
    """Read native stores, including attribution/recovery anchors and all vectors."""
    from lightrag.base import DocStatus

    replay = ApprovedReplay(packet)
    for doc in packet["documents"]:
        status = await rag.doc_status.get_by_id(doc["doc_id"])
        require(
            status is not None and status["status"] == DocStatus.PROCESSED,
            f"Native document failed or incomplete: {doc['doc_id']}: {status}",
        )
        require(
            set(status["chunks_list"]) == set(doc["chunk_ids"]),
            "Native chunk membership drift",
        )
        full = await rag.full_docs.get_by_id(doc["doc_id"])
        require(
            full and full["content"] == doc["text"],
            "Original prepared document changed",
        )
        entities = await rag.full_entities.get_by_id(doc["doc_id"])
        relations = await rag.full_relations.get_by_id(doc["doc_id"])
        require(
            entities is not None and relations is not None,
            "Native recovery anchors missing",
        )
        expected_entities = {
            o["name"]
            for o in packet["objects"].values()
            if o["kind"] == "Entity" and set(o["contributions"]) & set(doc["chunk_ids"])
        }
        expected_relations = {
            tuple(o["pair"])
            for o in packet["objects"].values()
            if o["kind"] == "Relation"
            and set(o["contributions"]) & set(doc["chunk_ids"])
        }
        require(
            set(entities["entity_names"]) == expected_entities,
            "Entity recovery anchors differ",
        )
        require(
            {tuple(sorted(pair)) for pair in relations["relation_pairs"]}
            == expected_relations,
            "Relation recovery anchors differ",
        )
        require(
            status.get("track_id")
            and status.get("metadata", {}).get("kg_write_state")
            == "graph_mutation_started",
            "Native tracking/recovery state missing",
        )
    for key, expected in packet["chunks"].items():
        value = await rag.text_chunks.get_by_id(key)
        require(
            value is not None
            and value["content"] == expected["content"]
            and value["full_doc_id"] == expected["full_doc_id"],
            "Native chunk originals differ",
        )
        require(
            bool(value.get("llm_cache_list")),
            "Native recovery extraction cache missing",
        )
        for cache_key in value["llm_cache_list"]:
            cache = await rag.llm_response_cache.get_by_id(cache_key)
            require(cache is not None, "Native cache anchor missing")
            request = next(
                x for x in packet["requests"].values() if x["chunk_id"] == key
            )
            require(
                cache.get("cache_type") == "extract"
                and cache.get("chunk_id") == key
                and json.loads(cache["return"]) == request["native"],
                "Native extraction cache changed",
            )
        vector = await rag.chunks_vdb.get_by_id(key)
        require(
            vector is not None
            and all(
                vector.get(field) == value.get(field)
                for field in ("content", "full_doc_id", "file_path")
            ),
            "Chunk vector differs",
        )
    for obj in packet["objects"].values():
        expected_ids = {e for ids in obj["contributions"].values() for e in ids}
        if obj["kind"] == "Entity":
            value = await rag.chunk_entity_relation_graph.get_node(obj["name"])
            tracking = await rag.entity_chunks.get_by_id(obj["name"])
            vector = await rag.entities_vdb.get_by_id(
                compute_mdhash_id(obj["name"], prefix="ent-")
            )
            require(
                value and value["entity_type"] == obj["entity_type"].lower(),
                "Entity missing/type drift",
            )
        else:
            a, b = obj["pair"]
            value = await rag.chunk_entity_relation_graph.get_edge(a, b)
            tracking = await rag.relation_chunks.get_by_id(
                make_relation_chunk_key(a, b)
            )
            vector = await rag.relationships_vdb.get_by_id(
                compute_mdhash_id(a + b, prefix="rel-")
            )
        require(
            value is not None and vector is not None,
            "Native graph/vector record missing",
        )
        expected_paths = {
            packet["chunks"][key]["file_path"] for key in obj["contributions"]
        }
        require(
            set(value["file_path"].split("<SEP>")) == expected_paths
            and vector["file_path"] == value["file_path"],
            "Graph/vector source label drift",
        )
        if obj["kind"] == "Entity":
            expected_content = obj["name"] + "\n" + value["description"]
            require(
                vector["entity_name"] == obj["name"], "Vector entity identity drift"
            )
        else:
            require(
                value["keywords"] == obj["keywords"]
                and value["weight"] == len(obj["contributions"]),
                "Relation keywords/weight drift",
            )
            require(
                [vector["src_id"], vector["tgt_id"]] == obj["pair"],
                "Vector relation endpoints drift",
            )
            expected_content = (
                obj["keywords"]
                + "\t"
                + "\n".join(obj["pair"])
                + "\n"
                + value["description"]
            )
        require(
            vector["content"] == expected_content,
            "Complete native vector content differs",
        )
        require(
            set(value["source_id"].split("<SEP>")) == set(obj["contributions"]),
            "Native source IDs lost or added",
        )
        require(
            tracking and set(tracking["chunk_ids"]) == set(obj["contributions"]),
            "Native authoritative tracking differs",
        )
        require(
            replay.description_ids(value["description"]) == expected_ids,
            "Stored graph evidence changed",
        )
        vector_description = vector["content"].split(
            "\n", 1 if obj["kind"] == "Entity" else 2
        )[-1]
        require(
            replay.description_ids(vector_description) == expected_ids,
            "Stored vector evidence changed",
        )
        require(vector["source_id"] == value["source_id"], "Vector source IDs differ")
    require(
        set(await rag.chunk_entity_relation_graph.get_all_labels())
        == {o["name"] for o in packet["objects"].values() if o["kind"] == "Entity"},
        "Unexpected graph entities",
    )
    require(
        {
            tuple(sorted((e["source"], e["target"])))
            for e in await rag.chunk_entity_relation_graph.get_all_edges()
        }
        == {
            tuple(o["pair"])
            for o in packet["objects"].values()
            if o["kind"] == "Relation"
        },
        "Unexpected graph relationships",
    )
    return {
        "documents": len(packet["documents"]),
        "chunks": len(packet["chunks"]),
        "entities": sum(o["kind"] == "Entity" for o in packet["objects"].values()),
        "relationships": sum(
            o["kind"] == "Relation" for o in packet["objects"].values()
        ),
        "native_vectors": ["chunks", "entities", "relationships"],
        "exact_originals": "prepared source ledger and native chunks",
        "graph_text": "complete evidence with pinned native normalization",
        "answer_correctness_tested": False,
        "production_approved": False,
    }


async def readback_candidate(
    sealed: dict,
    target: Path,
    *,
    expected_seal_sha256: str,
    embedding_func: EmbeddingFunc,
    smoke_query: str | None = None,
) -> dict:
    packet = validate_seal(sealed, expected_seal_sha256=expected_seal_sha256)
    require(target.is_dir(), "Candidate does not exist")
    require(
        (target / "sealed.json").is_file() and (target / "receipt.json").is_file(),
        "Candidate provenance/receipt missing; refusing to initialize unrelated directory",
    )
    require(
        json.loads((target / "sealed.json").read_text()) == sealed
        and json.loads((target / "receipt.json").read_text()).get("seal_sha256")
        == expected_seal_sha256,
        "Candidate identity/pin differs",
    )
    workspace = (
        "verified_" + hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:24]
    )
    require(
        (target / "storage" / workspace).is_dir(),
        "Persisted candidate storage is missing",
    )

    async def keywords(*args, **kwargs):
        return json.dumps(
            {
                "high_level_keywords": ["保险责任", "条件"],
                "low_level_keywords": ["保险", "责任"],
            },
            ensure_ascii=False,
        )

    replay = ApprovedReplay(packet)
    rag = make_native_rag(packet, target, replay, embedding_func, keyword_func=keywords)
    await rag.initialize_storages()
    try:
        report = await verify_native(rag, packet)
        if smoke_query:
            require(
                packet["runtime"]["config"]["embedding_model"]
                == "synthetic-offline-v1",
                "Synthetic keyword smoke is supported only for explicitly offline synthetic candidates",
            )
            report["smoke_configuration"] = {
                "mode": "mix",
                "top_k": 100,
                "chunk_top_k": 100,
                "max_total_tokens": 100000,
                "keywords": "deterministic synthetic stub",
                "purpose": "plumbing only; not production retrieval or answer correctness",
            }
            for flag in ("only_need_context", "only_need_prompt"):
                result = await rag.aquery(
                    smoke_query,
                    param=QueryParam(
                        mode="mix",
                        enable_rerank=False,
                        top_k=100,
                        chunk_top_k=100,
                        max_entity_tokens=30000,
                        max_relation_tokens=30000,
                        max_total_tokens=100000,
                        **{flag: True},
                    ),
                )
                require(
                    isinstance(result, str) and bool(result),
                    f"Native {flag} smoke failed",
                )
                # The smoke checks retrieval presence, not model answer correctness.
                require(
                    any(
                        span["quote"] in result
                        or json.dumps(span["quote"], ensure_ascii=False)[1:-1] in result
                        for span in packet["evidence"].values()
                    ),
                    f"Native {flag} lacks original evidence",
                )
                report[flag] = result
                report[flag + "_retrieved_evidence_ids"] = sorted(
                    key
                    for key, span in packet["evidence"].items()
                    if span["quote"] in result
                    or json.dumps(span["quote"], ensure_ascii=False)[1:-1] in result
                )
        return report
    finally:
        await rag.finalize_storages()


def synthetic_demo_seal(root: Path) -> dict:
    """Create only this fixed fictional fixture, never approve arbitrary inputs."""
    from product.import_units import inspect_source, prepare_import

    root.mkdir(parents=True, exist_ok=False)
    quote = (
        "合成测试保险。合成住院医疗责任。以下仅为离线程序测试条文，不是实际保险产品。"
        "因故意未履行如实告知义务，保险人对解除合同前发生的保险事故不承担赔偿责任且不退还保险费。"
        "因重大过失未履行如实告知义务且对保险事故发生有严重影响的，保险人不承担赔偿责任但应退还保险费。"
        "通知例外为保险人已经知道或者应当知道；通知例外不当然证明应予赔付。"
        "等待期内发生疾病不等同于等待期内住院。医疗费用扣除免赔额，日额津贴不适用医疗费用免赔额。"
    )
    documents = []
    for index in range(2):
        path = root / f"synthetic-{index}.txt"
        path.write_text(quote, encoding="utf-8")
        inspected = inspect_source(path, f"synthetic-source-{index}", "fixture-v1")
        span = {"start": 0, "end": len(quote), "quote": quote}
        documents.append(
            {
                **{
                    key: inspected[key]
                    for key in (
                        "source_id",
                        "document_version",
                        "original_sha256",
                        "extracted_text_sha256",
                        "parser_binding",
                    )
                },
                "path": path.name,
                "product": {
                    "id": "synthetic-only",
                    "label": "合成测试保险",
                    "identity_span": span,
                },
                "units": [
                    {
                        "unit_id": f"synthetic-rule-{index}",
                        "span": span,
                        "disposition": "primary",
                        "scope": {
                            "combinations": [
                                {
                                    "plan": {"kind": "not_applicable"},
                                    "responsibility": {
                                        "kind": "specific",
                                        "value": "合成住院医疗责任",
                                    },
                                    "evidence": [span],
                                }
                            ]
                        },
                        "review": {
                            "complete_conditions": True,
                            "dependencies_complete": True,
                        },
                        "dependencies": [],
                    }
                ],
            }
        )
    prepared = prepare_import(
        {
            "schema_version": 1,
            "tokenizer_model": "gpt-4o-mini",
            "max_chunk_tokens": 8192,
            "documents": documents,
        },
        root,
    )
    selections = {}
    for document in prepared["documents"]:
        unit = document["units"][0]
        common = {
            "scope": unit["scope"],
            "evidence_ids": [
                span["evidence_id"] for span in unit["primary"] + unit["context"]
            ],
        }
        selections[unit["unit_id"]] = {
            "entities": [
                {
                    "entity_id": "synthetic-coverage",
                    "name": "合成住院医疗责任",
                    "type": "保障责任",
                    **common,
                },
                {
                    "entity_id": "synthetic-rule",
                    "name": "合成完整约定",
                    "type": "规则",
                    **common,
                },
            ],
            "relationships": [
                {
                    "relation_id": "synthetic-relation",
                    "source": "synthetic-coverage",
                    "target": "synthetic-rule",
                    "keywords": "合成测试关系",
                    **common,
                }
            ],
        }
    packet = build_review_packet(prepared, selections)
    pin = digest(packet)
    return seal_review(
        packet,
        {
            "packet_sha256": pin,
            "decision": "approved",
            "reviewer": "fixed-fictional-fixture-review",
            "selection_author": "fixed-fictional-fixture-author",
            "scope_reviewed": True,
            "evidence_reviewed": True,
            "reviewed_unit_ids": list(selections),
            "purpose": "fixed synthetic demo only; not real-source or production approval",
        },
        expected_packet_sha256=pin,
    )


async def _persisted_readback(target: Path, pin: str) -> dict:
    """A new OS process bypasses native JSON shared-memory namespace caches."""
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1", PYTHONPATH=root)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "product.approved_import",
        "verify-persisted",
        str(target / "sealed.json"),
        str(target),
        "--pin",
        pin,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    require(
        process.returncode == 0,
        "Fresh-process persisted verification failed: "
        + stderr.decode(errors="replace")[-4000:],
    )
    report = json.loads((target / "persisted-readback.json").read_text())
    report["persisted_new_process_verified"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["demo", "import", "readback", "verify-persisted"]
    )
    parser.add_argument("sealed", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--pin", help="Independently retained seal SHA-256")
    args = parser.parse_args()
    if args.action == "demo":
        require(args.candidate is None, "Usage: demo NEW_CANDIDATE_DIRECTORY")
        target = args.sealed.resolve()
        require(not target.exists(), "Demo requires a wholly new candidate")
        sealed = synthetic_demo_seal(
            target.with_name(target.name + "-synthetic-inputs")
        )

        async def run_demo():
            await import_candidate(
                sealed,
                target,
                expected_seal_sha256=sealed["seal_sha256"],
                embedding_func=offline_embedding(),
            )
            return await readback_candidate(
                sealed,
                target,
                expected_seal_sha256=sealed["seal_sha256"],
                embedding_func=offline_embedding(),
                smoke_query="合成住院医疗责任有哪些完整条件？",
            )

        report = asyncio.run(run_demo())
        _write(target / "readback-report.json", report)
        print(
            json.dumps(
                {
                    "synthetic_only": True,
                    "candidate": str(target),
                    "seal_sha256": sealed["seal_sha256"],
                    "report": str(target / "readback-report.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    require(
        args.candidate is not None and bool(args.pin),
        "Candidate directory and independent --pin are required",
    )
    sealed = json.loads(args.sealed.read_text())
    config = ImportConfig(**sealed["packet"]["runtime"]["config"])
    func = import_candidate if args.action == "import" else readback_candidate
    embedding = (
        EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=config.embedding_token_limit,
            model_name=config.embedding_model,
            func=no_model,
        )
        if args.action == "verify-persisted"
        else offline_embedding(config)
    )
    kwargs = {"expected_seal_sha256": args.pin, "embedding_func": embedding}
    if args.action == "readback":
        kwargs["smoke_query"] = "保险责任有哪些完整条件？"
    result = asyncio.run(func(sealed, args.candidate, **kwargs))
    _write(
        args.candidate
        / (
            "persisted-readback.json"
            if args.action == "verify-persisted"
            else "readback-report.json"
        ),
        result,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"only_need_context", "only_need_prompt"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
