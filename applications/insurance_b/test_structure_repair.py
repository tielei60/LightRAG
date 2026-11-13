import json
from types import SimpleNamespace

import pytest
import tiktoken

from .common import digest, read_json, save_json
from .evidence_adapter import project_tables
from .structure_repair import B1NativeParser, restore_table_headers


async def test_native_p_keeps_real_plan_headers_in_index_and_extraction(settings):
    from lightrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.utils import Tokenizer

    class Context(ParseContext):
        async def archive_source(self, path):
            return None

    path = settings.root / "original.md"
    path.write_text(
        "# 示例保险\n\n## 原有计划\n\n| 项目 | 甲方案 | 乙方案 | 丙方案 |\n|---|---|---|---|\n| 航空 | 0 | — | |\n"
    )
    rag = build_debug_rag()
    rag._insurance_b_root = settings.root
    parsed = await B1NativeParser().parse(Context(rag, "synthetic", str(path), {}))
    tokenizer = Tokenizer("gpt-4o-mini", tiktoken.encoding_for_model("gpt-4o-mini"))
    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        parsed.content,
        blocks_path=parsed.blocks_path,
        chunk_token_size=2000,
        drop_references=False,
    )
    directory = __import__("pathlib").Path(parsed.blocks_path).parent
    tables = read_json(next(directory.glob("*.tables.json")))["tables"]
    indexed = "\n".join(c["content"] for c in chunks)
    projected, _ = project_tables(indexed, tables)
    assert all(
        name in indexed and name in projected for name in ["甲方案", "乙方案", "丙方案"]
    )
    assert "| 航空 | 0 | — |  |" in projected
    assert read_json(directory / "b1-structure-repair.json")["headers"]
    records = [
        json.loads(line)
        for line in __import__("pathlib")
        .Path(parsed.blocks_path)
        .read_text()
        .splitlines()
    ]
    assert next(r for r in records if r["type"] == "meta")[
        "document_hash"
    ] == "sha256:" + digest(parsed.content)
    before = parsed.content
    restore_table_headers(parsed, settings.root)
    assert parsed.content == before


def test_header_does_not_attach_to_an_unverified_table_body(tmp_path):
    blocks = tmp_path / "test.blocks.jsonl"
    content = '<table id="t">[["changed", "0"]]</table>'
    blocks.write_text(json.dumps({"type": "content", "content": content}) + "\n")
    save_json(
        tmp_path / "test.tables.json",
        {
            "tables": {
                "t": {
                    "format": "json",
                    "content": '[["original", "0"]]',
                    "table_header": '[["项目", "甲方案"]]',
                }
            }
        },
    )
    with pytest.raises(ValueError, match="changed table body"):
        restore_table_headers(
            SimpleNamespace(blocks_path=str(blocks), content=content), tmp_path
        )
    assert (
        read_json(tmp_path / "test.tables.json")["tables"]["t"]["content"]
        == '[["original", "0"]]'
    )
