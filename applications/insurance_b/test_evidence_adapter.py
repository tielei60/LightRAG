import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import tiktoken

from .common import append_event, digest, save_json
from .evidence_adapter import (
    BExtractionAdapter,
    evidence_spans,
    identity_span,
    make_catalog,
    project_tables,
    restore_validation_feedback,
)


def test_restart_recovers_only_its_own_unapproved_evidence_error(tmp_path):
    path = tmp_path / "events.jsonl"
    chunk = SimpleNamespace(
        source="synthetic.md",
        product="示例保险",
        units=[SimpleNamespace(text="完整原文")],
    )
    delegate = SimpleNamespace(
        requests={"native-prompt": chunk}, selections={}, validation_errors={}
    )
    event = {
        "document_id": "doc-1",
        "stage": "extract",
        "state": "invalid_or_unknown",
        "error_type": "EvidenceSemanticError",
        "error": "真实类型错误",
        "input": json.dumps(
            {
                "source_file": "synthetic.md",
                "product": "示例保险",
                "evidence": [{"text": "完整原文"}],
            }
        ),
    }
    catalog = {"document_id": "doc-1", "source": "synthetic.md", "product": "示例保险"}
    append_event(path, {**event, "document_id": "other-doc"})
    append_event(path, {**event, "error_type": "ConnectionError"})
    assert not restore_validation_feedback(delegate, catalog, path)
    append_event(path, event)
    assert restore_validation_feedback(delegate, catalog, path)
    assert (
        delegate.validation_errors[digest("native-prompt")]
        == "EvidenceSemanticError: 真实类型错误"
    )
    delegate.selections[digest("native-prompt")] = {"approved": True}
    delegate.validation_errors.clear()
    assert not restore_validation_feedback(delegate, catalog, path)
    assert not delegate.validation_errors


def test_table_evidence_includes_original_title_blank_separators_and_notes():
    from ._current.evidence_extract import EvidenceUnit
    from ._current.evidence_semantics import semantic_metadata

    text = "## 原计划表\n\n\n| 项目 | 甲方案 | 乙方案 |\n| --- | --- | --- |\n| 航空 | 0 | 100万元 |\n\n注：仅适用于表内方案。\n\n## 其他\n无关原文。"
    units = [
        EvidenceUnit(str(i), "synthetic.md", a, b, text, ())
        for i, (a, b) in enumerate(evidence_spans(text, complete_tables=True))
    ]
    facts = semantic_metadata(units)["table_facts"]
    assert len(facts) == 2
    table_unit = next(u for u in units if "| 航空 |" in u.text)
    assert table_unit.text.startswith("## 原计划表")
    assert "注：仅适用于表内方案。" in table_unit.text
    assert len(semantic_metadata([table_unit])["table_facts"]) == 2
    assert "无关原文" not in table_unit.text


def test_table_projection_preserves_zero_dash_and_empty_cells():
    cells = [["计划", "保障"], ["计划一", "0"], ["计划二", "-"], ["计划三", ""]]
    source = (
        '<table id="t" format="json">'
        + json.dumps(cells, ensure_ascii=False)
        + "</table>"
    )
    text, mappings = project_tables(source, {"t": {"content": json.dumps(cells)}})
    assert "| 计划一 | 0 |" in text
    assert "| 计划二 | - |" in text
    assert "| 计划三 |  |" in text
    assert mappings[0]["rows"] == cells


def test_table_slice_must_be_supported_by_its_source():
    with pytest.raises(ValueError, match="verified"):
        project_tables(
            '<table id="t">[["fake"]]</table>', {"t": {"content": '[["real"]]'}}
        )


def test_identity_span_accepts_only_normalized_original_characters():
    source = "标题 北京人寿 示例（优选版）保险\n正文"
    a, b = identity_span(source, "北京人寿示例(优选版)保险")
    assert source[a:b] == "北京人寿 示例（优选版）保险"
    with pytest.raises(ValueError, match="absent"):
        identity_span(source, "另一份产品")


class Store:
    def __init__(self, rows):
        self.rows = rows

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def get_docs_by_track_id(self, track):
        return self.rows

    async def upsert(self, values):
        self.rows.update(values)


@pytest.fixture
def bridge(settings):
    settings.activate()
    from lightrag.prompt import PROMPTS, resolve_entity_extraction_prompt_profile
    from lightrag.utils import Tokenizer

    profile = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "products.yml"}, True
    )
    context = {
        "entity_types_guidance": profile["entity_types_guidance"],
        "examples": "\n".join(profile["entity_extraction_json_examples"]),
        "language": "Chinese",
        "max_total_records": 500,
        "max_entity_records": 200,
    }
    tokenizer = Tokenizer("gpt-4o-mini", tiktoken.encoding_for_model("gpt-4o-mini"))
    cells = [["保障责任", "计划一", "计划二"], ["住院医疗", "10000", "0"]]
    content = (
        '# 示例保险产品\n\n<table id="table-test" format="json">'
        + json.dumps(cells, ensure_ascii=False)
        + "</table>\n\n仅为合成测试。"
    )
    directory = settings.run_dir / "input/__parsed__/sample.md.parsed"
    directory.mkdir(parents=True)
    blocks_path = directory / "sample.blocks.jsonl"
    save_json(
        directory / "sample.tables.json",
        {"tables": {"table-test": {"content": json.dumps(cells)}}},
    )
    blocks_path.write_text(
        json.dumps({"type": "meta"})
        + "\n"
        + json.dumps(
            {
                "type": "content",
                "blockid": "block-test",
                "content": content,
                "positions": [{"anchor": "1"}],
            }
        )
        + "\n"
    )
    entry = {
        "product": "示例保险产品",
        "relative_path": "synthetic/sample.md",
        "sha256": "test-original",
    }
    chunks = {
        "chunk-test": {
            "content": content,
            "chunk_order_index": 0,
            "tokens": 100,
            "full_doc_id": "doc-test",
            "heading": {"level": 1, "heading": "示例保险产品", "parent_headings": []},
            "sidecar": {"refs": [{"type": "block", "id": "block-test"}]},
        }
    }
    config = {
        "_entity_extraction_prompt_profile": profile,
        "_resolved_summary_language": "Chinese",
        "entity_extract_max_records": 500,
        "entity_extract_max_entities": 200,
    }
    status = SimpleNamespace(
        metadata={"chunk_method": "paragraph_semantic"}, chunks_list=["chunk-test"]
    )
    rag = SimpleNamespace(
        doc_status=Store({"doc-test": status}),
        text_chunks=Store(chunks),
        full_docs=Store({"doc-test": {"sidecar_location": directory.as_uri() + "/"}}),
        tokenizer=tokenizer,
        _build_global_config=lambda: config,
    )
    return SimpleNamespace(
        settings=settings,
        entry=entry,
        chunks=chunks,
        blocks=blocks_path,
        context=context,
        config=config,
        profile=profile,
        tokenizer=tokenizer,
        rag=rag,
        system=PROMPTS["entity_extraction_json_system_prompt"]
        .format(**context)
        .strip(),
    )


def test_catalog_requires_actual_p_block_provenance(bridge):
    bad = deepcopy(bridge.chunks)
    bad["chunk-test"].pop("sidecar")
    with pytest.raises(ValueError, match="provenance"):
        make_catalog(
            bridge.entry,
            "doc-test",
            bad,
            bridge.blocks,
            bridge.context,
            bridge.tokenizer,
        )


def test_catalog_keeps_indexed_input_and_model_evidence_separate(bridge):
    result = make_catalog(
        bridge.entry,
        "doc-test",
        bridge.chunks,
        bridge.blocks,
        bridge.context,
        bridge.tokenizer,
    )
    row = result["chunks"][0]
    assert '<table id="table-test"' in row["indexed_content"]
    assert "| 住院医疗 | 10000 | 0 |" in row["evidence_content"]
    assert row["table_projections"][0]["table_id"] == "table-test"
    assert row["positions"] == [{"anchor": "1"}]


def test_final_identity_labels_are_excluded_from_evidence_and_table_offsets_stay_index_relative(
    bridge,
):
    from .input_adapters.source_context import finalize_chunks

    raw = list(bridge.chunks.values())
    final, _ = finalize_chunks(
        raw,
        entry=bridge.entry,
        parsed_text=raw[0]["content"],
        tokenizer=bridge.tokenizer,
    )
    result = make_catalog(
        bridge.entry,
        "doc-test",
        {"chunk-test": final[0]},
        bridge.blocks,
        bridge.context,
        bridge.tokenizer,
        source_context=True,
    )
    row = result["chunks"][0]
    assert "来源标识" in row["indexed_content"]
    assert "来源标识" not in row["evidence_content"]
    start, end = row["table_projections"][0]["native_span"]
    assert row["indexed_content"][start:end].startswith('<table id="table-test"')


def selection(payload, *, wrong_zero=False):
    identity = next(
        unit["id"] for unit in payload["evidence"] if unit["role"] == "product-identity"
    )
    table = next(
        unit["id"]
        for unit in payload["evidence"]
        if "| 住院医疗 | 10000 | 0 |" in unit["text"]
    )
    plan = "计划二" if wrong_zero else "计划一"
    return json.dumps(
        {
            "entities": [
                {
                    "name": "示例保险产品",
                    "type": "保险产品",
                    "evidence_ids": [identity],
                },
                {"name": plan, "type": "保障计划", "evidence_ids": [table]},
                {"name": "住院医疗", "type": "医疗责任", "evidence_ids": [table]},
            ],
            "relationships": [
                {
                    "source": "示例保险产品",
                    "target": plan,
                    "keywords": "包含计划",
                    "evidence_ids": [table],
                },
                {
                    "source": plan,
                    "target": "住院医疗",
                    "keywords": "包含责任",
                    "evidence_ids": [table],
                },
            ],
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_pointer_extraction_is_validated_and_durable_replay_skips_provider(
    bridge,
):
    calls = []

    async def complete(prompt, **kwargs):
        payload = json.loads(prompt)
        calls.append(payload)
        return selection(payload)

    # SimpleNamespace is not weak-referenceable; the real instance is.
    class Rag:
        pass

    rag = Rag()
    rag.__dict__.update(bridge.rag.__dict__)
    adapter = BExtractionAdapter(bridge.settings, complete)
    adapter.bind(rag)
    adapter.begin_document("track-test", bridge.entry)
    from .evidence_adapter import native_prompt

    prompt = native_prompt(
        bridge.chunks["chunk-test"], bridge.context, bridge.tokenizer
    )
    native = json.loads(await adapter(prompt, system_prompt=bridge.system))
    assert len(calls) == 1
    assert len(native["relationships"]) == 2
    assert (bridge.settings.run_dir / "selections/doc-test.json").is_file()
    replay = BExtractionAdapter(bridge.settings, complete)
    replay.bind(rag)
    replay.begin_document("track-test", bridge.entry)
    assert json.loads(await replay(prompt, system_prompt=bridge.system)) == native
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_zero_amount_never_becomes_positive_coverage(bridge):
    calls = []

    async def complete(prompt, **kwargs):
        calls.append(prompt)
        return selection(json.loads(prompt), wrong_zero=True)

    class Rag:
        pass

    rag = Rag()
    rag.__dict__.update(bridge.rag.__dict__)
    adapter = BExtractionAdapter(bridge.settings, complete)
    adapter.bind(rag)
    adapter.begin_document("track-test", bridge.entry)
    from .evidence_adapter import native_prompt

    with pytest.raises(ValueError):
        await adapter(
            native_prompt(
                bridge.chunks["chunk-test"], bridge.context, bridge.tokenizer
            ),
            system_prompt=bridge.system,
        )
    assert len(calls) == 3
    assert not (bridge.settings.run_dir / "selections/doc-test.json").exists()
