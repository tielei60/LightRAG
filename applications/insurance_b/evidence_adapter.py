"""Bind actual native-P chunks to the frozen, validated evidence contract."""

import asyncio
import json
import re
import unicodedata
import uuid
import weakref
from pathlib import Path

import yaml

from .common import append_event, digest, inside, read_json, save_json
from .config import Settings
from .contracts import contract_module

_TABLE = re.compile(
    r'<table\b[^>]*\bid="(?P<id>[^"]+)"[^>]*>(?P<body>.*?)</table>', re.DOTALL
)


def project_tables(content: str, tables: dict) -> tuple[str, list[dict]]:
    """Serialize only the table cells actually present in this final chunk."""
    mappings = []

    def replace(match):
        identifier = match["id"]
        if identifier not in tables:
            raise ValueError(
                "A chunk references a table absent from its structured sidecar"
            )
        rows = json.loads(match["body"])
        original = json.loads(tables[identifier]["content"])
        if not rows or not all(
            isinstance(row, list) and row and all(isinstance(cell, str) for cell in row)
            for row in rows
        ):
            raise ValueError(
                "A table slice is not a nonempty array of original text cells"
            )
        if len({len(row) for row in rows}) != 1 or any(
            row not in original for row in rows
        ):
            raise ValueError(
                "The P table slice cannot be verified against its original table"
            )
        escape = lambda text: (
            text.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        lines = ["| " + " | ".join(escape(cell) for cell in row) + " |" for row in rows]
        lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
        mappings.append(
            {
                "table_id": identifier,
                "native_span": list(match.span()),
                "rows": rows,
                "sidecar_ref": tables[identifier].get("self_ref"),
            }
        )
        return "\n\n" + "\n".join(lines) + "\n\n"

    return _TABLE.sub(replace, content), mappings


def identity_span(source: str, product: str) -> tuple[int, int]:
    letters, offsets = [], []
    for index, letter in enumerate(source):
        for normalized in unicodedata.normalize("NFKC", letter):
            if not normalized.isspace():
                letters.append(normalized)
                offsets.append(index)
    target = "".join(unicodedata.normalize("NFKC", product).split())
    start = "".join(letters).find(target)
    if start < 0:
        raise ValueError("The verified product name is absent from the actual P corpus")
    return offsets[start], offsets[start + len(target) - 1] + 1


def native_prompt(row: dict, context: dict, tokenizer) -> str:
    from lightrag.chunk_schema import (
        format_heading_context,
        strip_internal_multimodal_markup_for_extraction,
    )
    from lightrag.constants import DEFAULT_MAX_SECTION_CONTEXT_TOKENS
    from lightrag.operate import _truncate_section_context
    from lightrag.prompt import PROMPTS

    path = _truncate_section_context(
        format_heading_context(row), tokenizer, DEFAULT_MAX_SECTION_CONTEXT_TOKENS
    )
    heading = (
        PROMPTS["entity_extraction_section_context"].format(heading_path=path)
        if path
        else ""
    )
    return (
        PROMPTS["entity_extraction_json_user_prompt"]
        .format(
            **context,
            input_text=strip_internal_multimodal_markup_for_extraction(row["content"]),
            heading_context_block=heading,
        )
        .strip()
    )


def evidence_spans(text: str, *, complete_tables=False) -> list[tuple[int, int]]:
    spans = [
        (m.start(), m.end()) for m in re.finditer(r"\S[\s\S]*?(?:\n[ \t]*\n|\Z)", text)
    ]
    if complete_tables:
        from ._current.evidence_semantics import parse_markdown_tables

        for table in parse_markdown_tables(text):
            overlapping = [
                (a, b) for a, b in spans if a < table.end and b > table.start
            ]
            if not overlapping:
                raise ValueError("A source table has no evidence coverage")
            start = min(table.start, min(a for a, _ in overlapping))
            end = max(table.end, max(b for _, b in overlapping))
            spans = [(a, b) for a, b in spans if not (a < end and b > start)] + [
                (start, end)
            ]
    return sorted(spans)


def make_catalog(
    entry: dict,
    doc_id: str,
    chunks: dict,
    blocks_path: Path,
    context: dict,
    tokenizer,
    *,
    complete_table_units=False,
    source_context=False,
) -> dict:
    from lightrag.prompt import PROMPTS

    blocks = [
        json.loads(line)
        for line in blocks_path.read_text().splitlines()
        if line.strip()
    ]
    known = {
        block["blockid"]: block for block in blocks if block.get("type") == "content"
    }
    if not known:
        raise ValueError(
            "A P target has no structured content blocks; fallback is not accepted"
        )
    table_paths = list(blocks_path.parent.glob("*.tables.json"))
    tables = read_json(table_paths[0])["tables"] if table_paths else {}
    ordered = sorted(chunks.items(), key=lambda item: item[1]["chunk_order_index"])
    source, projected = "", []
    for key, row in ordered:
        refs = [
            ref["id"]
            for ref in (row.get("sidecar") or {}).get("refs", [])
            if ref.get("type") == "block"
        ]
        if not refs or not set(refs) <= known.keys():
            raise ValueError("An effective P chunk lacks verifiable block provenance")
        index_body, body_offset = row["content"], 0
        if source_context:
            from .input_adapters.source_context import unwrap

            index_body, body_offset = unwrap(
                index_body,
                product=entry["product"],
                source=Path(entry["relative_path"]).name,
            )
        text, table_mappings = project_tables(index_body, tables)
        for mapping in table_mappings:
            mapping["native_span"] = [p + body_offset for p in mapping["native_span"]]
        start = len(source)
        source += text + "\n\n"
        projected.append((key, row, text, start, refs, table_mappings))
    name_start, name_end = identity_span(source, entry["product"])
    identity = source[name_start:name_end]
    rows = []
    for key, original, text, base, refs, table_mappings in projected:
        units = []
        for start, end in evidence_spans(text, complete_tables=complete_table_units):
            a, b = base + start, base + end
            units.append(
                {
                    "id": "e-" + digest(f"{entry['sha256']}:{a}:{b}")[:24],
                    "start": a,
                    "end": b,
                    "entity_types": [],
                    "role": "primary",
                }
            )
        units.append(
            {
                "id": "identity-" + entry["sha256"][:24],
                "start": name_start,
                "end": name_end,
                "entity_types": [],
                "role": "product-identity",
            }
        )
        envelope = text if identity in text else text + "\n\n" + identity
        rows.append(
            {
                "chunk_id": key,
                "indexed_sha256": digest(original["content"]),
                "indexed_content": original["content"],
                "tokens": original.get("tokens"),
                "heading": original.get("heading"),
                "sidecar": original.get("sidecar"),
                "evidence_content": envelope,
                "units": units,
                "native_prompt": native_prompt(original, context, tokenizer),
                "block_ids": refs,
                "positions": [
                    position
                    for ref in refs
                    for position in known[ref].get("positions", [])
                ],
                "table_projections": table_mappings,
            }
        )
    return {
        "version": 1,
        "document_id": doc_id,
        "source": Path(entry["relative_path"]).name,
        "product": entry["product"],
        "source_sha256": entry["sha256"],
        "source_projection": source,
        "source_projection_kind": "native P output with verified table serialization; offsets refer to this projection",
        "blocks_path": str(blocks_path),
        "native_system_prompt": PROMPTS["entity_extraction_json_system_prompt"]
        .format(**context)
        .strip(),
        "chunks": rows,
    }


def restore_validation_feedback(delegate, catalog, event_path):
    """Resume the common extractor's bounded feedback after a process restart.

    Only an exact, unapproved evidence request from this document can recover
    its own semantic/format error. Provider failures and other sources cannot
    seed business feedback.
    """
    if not event_path.exists():
        return []
    restored = {}
    for line_number, line in enumerate(event_path.read_text().splitlines(), 1):
        event = json.loads(line)
        if (
            event.get("document_id") != catalog["document_id"]
            or event.get("stage") != "extract"
            or event.get("state") != "invalid_or_unknown"
            or event.get("error_type")
            not in {"ValueError", "ValidationError", "EvidenceSemanticError"}
        ):
            continue
        payload = json.loads(event["input"])
        matches = [
            prompt
            for prompt, chunk in delegate.requests.items()
            if digest(prompt.strip()) not in delegate.selections
            and payload.get("source_file") == chunk.source == catalog["source"]
            and payload.get("product") == chunk.product == catalog["product"]
            and [u.get("text") for u in payload["evidence"]]
            == [u.text for u in chunk.units]
        ]
        if len(matches) != 1:
            continue
        key = digest(matches[0].strip())
        delegate.validation_errors[key] = (
            f"{event['error_type']}: {event['error'][:500]}"
        )
        restored[key] = {"prompt_sha256": key, "event_line": line_number}
    return list(restored.values())


class BExtractionAdapter:
    """Read final chunks once, then reuse the frozen extractor and merge guard."""

    def __init__(self, settings: Settings, complete):
        self.settings = settings
        self.complete = complete
        self._rag = None
        self.entry = None
        self.track = None
        self.target_id = None
        self.delegate = None
        self.current_id = None
        self.units = {}
        self.expected = {}
        self.source_order = {}
        self.lock = asyncio.Lock()
        self.cache_epoch = uuid.uuid4().hex

    def __deepcopy__(self, memo):
        # The native config serializer copies callables; this live callback is shared.
        return self

    def bind(self, rag):
        self._rag = weakref.ref(rag)

    def cache_identity(self) -> dict:
        entry = self.entry or {}
        return {
            "epoch": self.cache_epoch,
            "product": entry.get("product"),
            "source_sha256": entry.get("sha256"),
            "source_path": entry.get("relative_path"),
        }

    def begin_document(
        self, track: str | None, entry: dict, *, document_id: str | None = None
    ):
        self.track, self.entry, self.current_id = track, entry, None
        self.target_id = document_id

    def _hydrate(self, catalog: dict) -> None:
        settings = self.settings
        module = contract_module(settings, "evidence_extract")
        EvidenceChunk, EvidenceUnit, EvidenceExtractor = (
            module.EvidenceChunk,
            module.EvidenceUnit,
            module.EvidenceExtractor,
        )
        ExtractionProfile = contract_module(
            settings, "extraction_profile"
        ).ExtractionProfile
        profile = yaml.safe_load(
            (settings.profiles / "entity_type/products.yml").read_text()
        )
        taxonomy = ExtractionProfile.from_mapping(profile)
        requests = {}
        for row in catalog["chunks"]:
            units = tuple(
                EvidenceUnit(
                    unit["id"],
                    catalog["source"],
                    unit["start"],
                    unit["end"],
                    catalog["source_projection"],
                    (taxonomy.product_entity_type,)
                    if unit["role"] == "product-identity"
                    else tuple(unit["entity_types"]),
                    unit["role"],
                )
                for unit in row["units"]
            )
            requests[row["native_prompt"]] = EvidenceChunk(
                catalog["source"], catalog["product"], row["evidence_content"], units
            )
        delegate = EvidenceExtractor(
            requests=requests,
            native_system_prompt=catalog["native_system_prompt"],
            native_summary_template=self.summary_template,
            type_guidance=profile["entity_types_guidance"],
            selection_guidance=profile["evidence_selection_guidance"],
            complete=self.complete,
            record=lambda event: append_event(
                settings.run_dir / "extraction-events.jsonl",
                {"document_id": catalog["document_id"], **event},
            ),
            ledger_path=settings.run_dir
            / "selections"
            / (catalog["document_id"] + ".json"),
            ledger_context=digest(
                json.dumps(
                    {
                        "source": catalog["source_sha256"],
                        "projection": digest(catalog["source_projection"]),
                        "snapshot": read_json(settings.contract_snapshot),
                    },
                    sort_keys=True,
                )
            ),
            profile=taxonomy,
        )
        recovered = restore_validation_feedback(
            delegate, catalog, settings.run_dir / "extraction-events.jsonl"
        )
        if recovered:
            append_event(
                settings.run_dir / "extraction-events.jsonl",
                {
                    "document_id": catalog["document_id"],
                    "stage": "extract_recovery",
                    "state": "validation_feedback_restored",
                    "requests": recovered,
                    "basis": "exact same-document unapproved evidence; existing common correction protocol",
                },
            )
        self.units.update({unit.identity: unit for unit in delegate.units})
        self.source_order.setdefault(catalog["source"], len(self.source_order))
        delegate.units = tuple(self.units.values())
        delegate.source_order = dict(self.source_order)
        delegate.expected = {key: dict(units) for key, units in self.expected.items()}
        delegate.activate_sources([catalog["source"]])
        self.delegate = delegate
        self.expected = delegate.expected
        self.current_id = catalog["document_id"]

    @property
    def summary_template(self):
        from lightrag.prompt import PROMPTS

        return PROMPTS["summarize_entity_descriptions"]

    def restore(self, doc_id: str):
        catalog = read_json(self.settings.run_dir / "catalogs" / (doc_id + ".json"))
        self._hydrate(catalog)

    async def _ensure_current(self):
        async with self.lock:
            if self.current_id is not None:
                return
            rag = self._rag() if self._rag else None
            if (
                rag is None
                or self.entry is None
                or (self.track is None and self.target_id is None)
            ):
                raise ValueError("The B extraction callback has no active document")
            if self.target_id is not None:
                from lightrag.base import DocStatus

                all_statuses = await rag.doc_status.get_docs_by_statuses(
                    [
                        DocStatus.PROCESSED,
                        DocStatus.PENDING,
                        DocStatus.PROCESSING,
                        DocStatus.PARSING,
                        DocStatus.ANALYZING,
                        DocStatus.FAILED,
                    ],
                    strict=True,
                )
                statuses = {
                    key: value
                    for key, value in all_statuses.items()
                    if key == self.target_id
                }
            else:
                statuses = await rag.doc_status.get_docs_by_track_id(self.track)
            if len(statuses) != 1:
                raise ValueError(
                    "The B callback requires one strictly identified document"
                )
            doc_id, status = next(iter(statuses.items()))
            metadata = status.metadata or {}
            if metadata.get("chunk_method") != "paragraph_semantic":
                raise ValueError("The native pipeline did not report actual P dispatch")
            ids = status.chunks_list
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("No complete final native chunk set is available")
            chunks = {}
            for key in ids:
                row = await rag.text_chunks.get_by_id(key)
                if row is None or row.get("full_doc_id") != doc_id:
                    raise ValueError(
                        "A final chunk is absent or belongs to another document"
                    )
                chunks[key] = row
            document = await rag.full_docs.get_by_id(doc_id)
            from lightrag.utils_pipeline import resolve_sidecar_uri

            directory = resolve_sidecar_uri(document.get("sidecar_location"))
            if directory is None:
                raise ValueError("A P target has no persisted structured sidecar URI")
            directory = inside(self.settings.root, directory)
            paths = list(directory.glob("*.blocks.jsonl"))
            if len(paths) != 1:
                raise ValueError("The source block file is missing or ambiguous")
            config = rag._build_global_config()
            profile = config["_entity_extraction_prompt_profile"]
            context = {
                "entity_types_guidance": profile["entity_types_guidance"],
                "examples": "\n".join(profile["entity_extraction_json_examples"]),
                "language": config["_resolved_summary_language"],
                "max_total_records": config["entity_extract_max_records"],
                "max_entity_records": config["entity_extract_max_entities"],
            }
            catalog = make_catalog(
                self.entry,
                doc_id,
                chunks,
                paths[0],
                context,
                rag.tokenizer,
                complete_table_units=self.settings.get("CONTRACT_REVISION")
                == "current",
                source_context=self.settings.get("FINAL_CONTEXT") == "source-v1",
            )
            path = self.settings.run_dir / "catalogs" / (doc_id + ".json")
            if path.exists() and read_json(path) != catalog:
                raise ValueError(
                    "Final P chunks or the frozen extraction input changed"
                )
            save_json(path, catalog)
            self._hydrate(catalog)

    async def __call__(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        if system_prompt:
            await self._ensure_current()
        if self.delegate is None:
            raise ValueError("No verified evidence exists for this callback")
        if prompt.strip() not in self.delegate.requests:
            return await self.delegate(
                prompt, system_prompt=system_prompt, history_messages=history_messages
            )
        for attempt in range(3):
            try:
                return await self.delegate(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                )
            except ValueError:
                if attempt == 2:
                    raise
