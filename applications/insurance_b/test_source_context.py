import copy
import json

import pytest
import tiktoken

from ._current.grounded_query import parse_source_context
from .input_adapters.source_context import finalize_chunks, unwrap


def test_wrapper_binds_source_from_native_metadata_not_sidecar_stem(
    settings, monkeypatch
):
    from types import SimpleNamespace

    import lightrag.chunker as chunkers

    from .input_adapters.source_context import install

    blocks = settings.root / "source.blocks.jsonl"
    blocks.write_text(json.dumps({"type": "meta", "document_name": "source.md"}) + "\n")
    entry = {"product": "示例保险", "relative_path": "source.md", "sha256": "original"}
    calls = []

    def native(tokenizer, content, size, **kwargs):
        calls.append(content)
        return [{"content": content, "chunk_order_index": 0}]

    monkeypatch.setattr(chunkers, "chunking_by_paragraph_semantic", native)
    install(settings, SimpleNamespace(entry=entry))
    chunks = chunkers.chunking_by_paragraph_semantic(
        tiktoken.get_encoding("o200k_base"),
        "示例保险原文。",
        blocks_path=str(blocks),
        doc_id="doc-1",
    )
    assert calls == ["示例保险原文。"]
    assert "产品：示例保险" in chunks[0]["content"]
    blocks.write_text(
        json.dumps({"type": "meta", "document_name": "another.md"}) + "\n"
    )
    with pytest.raises(ValueError, match="another source"):
        chunkers.chunking_by_paragraph_semantic(
            tiktoken.get_encoding("o200k_base"),
            "示例保险原文。",
            blocks_path=str(blocks),
            doc_id="doc-1",
        )


def test_each_final_chunk_exposes_verified_identity_without_treating_labels_as_business_text():
    tokenizer = tiktoken.get_encoding("o200k_base")
    parsed = "# 示例 保险\n\n甲条款原文。\n\n乙条款原文。"
    raw = [
        {"content": "# 示例 保险\n\n甲条款原文。", "chunk_order_index": 0},
        {"content": "乙条款原文。", "chunk_order_index": 1},
    ]
    before = copy.deepcopy(raw)
    final, origin = finalize_chunks(
        raw,
        entry={
            "product": "示例保险",
            "relative_path": "source.md",
            "sha256": "source-hash",
        },
        parsed_text=parsed,
        tokenizer=tokenizer,
    )
    assert raw == before
    assert (
        parsed[slice(*origin["parsed_text_span"])]
        == origin["verbatim_quote"]
        == "示例 保险"
    )
    ndjson = "\n".join(
        json.dumps({"reference_id": "1", "content": r["content"]}, ensure_ascii=False)
        for r in final
    )
    context = parse_source_context(
        "Document Chunks (DC):\n```json\n"
        + ndjson
        + "\n```\nReference Document List (R):\n```\n[1] source.md\n```\n"
    )
    assert {u.product_name for u in context.units} == {"示例保险"}
    assert all(
        "来源标识" not in u.text and "产品：" not in u.text for u in context.units
    )
    body, _ = unwrap(final[1]["content"], product="示例保险", source="source.md")
    assert body == "示例 保险\n\n乙条款原文。"


def test_identity_cannot_come_from_manifest_alone_and_cannot_overflow_embedding_budget():
    tokenizer = tiktoken.get_encoding("o200k_base")
    entry = {
        "product": "示例保险",
        "relative_path": "source.md",
        "sha256": "source-hash",
    }
    with pytest.raises(ValueError, match="absent"):
        finalize_chunks(
            [{"content": "其他原文", "chunk_order_index": 0}],
            entry=entry,
            parsed_text="另一产品",
            tokenizer=tokenizer,
        )
    with pytest.raises(ValueError, match="embedding budget"):
        finalize_chunks(
            [{"content": "示例保险", "chunk_order_index": 0}],
            entry=entry,
            parsed_text="示例保险",
            tokenizer=tokenizer,
            token_limit=1,
        )
