import json
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from .common import digest, read_json, save_json
from .input_adapters.pdf_math import (
    replacements,
    restore_formula_layout,
    superscript_pairs,
)
from .structure_repair import B1DoclingParser


def make_pdf(path, *, raised=True):
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 1 0 0 1 40 200 Tm (1.08) Tj ET\n"
        + (
            b"BT /F1 6 Tf 1 0 0 1 70 205 Tm (N-1) Tj ET"
            if raised
            else b"BT /F1 12 Tf 1 0 0 1 70 200 Tm (N-1) Tj ET"
        )
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def parsed(tmp_path):
    text = "原文系数1.08 N-1，非保证收益。"
    blocks = tmp_path / "source.blocks.jsonl"
    blocks.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                {"type": "meta", "document_hash": "sha256:" + digest(text)},
                {"type": "content", "content": text},
            ]
        )
        + "\n"
    )
    save_json(
        tmp_path / "source.tables.json",
        {
            "tables": {
                "t": {
                    "content": '[["1.08 N-1"]]',
                    "table_header": '[["1.08 N-1"]]',
                }
            }
        },
    )
    return SimpleNamespace(content=text, blocks_path=str(blocks))


def test_real_pdf_glyphs_restore_exponent_and_all_indexed_representations(tmp_path):
    source = tmp_path / "source.pdf"
    make_pdf(source)
    pairs = superscript_pairs(source)
    assert [(p["physical_page"], p["base"], p["exponent"]) for p in pairs] == [
        (1, "1.08", "N-1")
    ]
    result = restore_formula_layout(
        parsed(tmp_path), tmp_path, pairs, digest(source.read_bytes())
    )
    assert result.content == "原文系数1.08^(N-1)，非保证收益。"
    records = [
        json.loads(line)
        for line in __import__("pathlib")
        .Path(result.blocks_path)
        .read_text()
        .splitlines()
    ]
    assert records[0]["document_hash"] == "sha256:" + digest(result.content)
    assert records[1]["content"] == result.content
    table = read_json(tmp_path / "source.tables.json")["tables"]["t"]
    assert table["content"] == table["table_header"] == '[["1.08^(N-1)"]]'
    proof = read_json(tmp_path / "b1-formula-layout-repair.json")
    assert proof["original_glyph_pairs"] == pairs
    assert proof["ocr_or_model_calls"] == 0
    assert replacements(result.content, pairs) == (result.content, [])


def test_same_line_variable_is_not_an_exponent_and_ambiguous_coverage_is_blocked(
    tmp_path,
):
    source = tmp_path / "flat.pdf"
    make_pdf(source, raised=False)
    assert superscript_pairs(source) == []
    with pytest.raises(ValueError, match="coverage differs"):
        replacements("1.08 N-1; 1.08 N-1", [{"base": "1.08", "exponent": "N-1"}])


async def test_docling_collects_layout_before_native_parser_archives_source(
    tmp_path, monkeypatch
):
    from lightrag.parser.external.docling.parser import DoclingParser

    source = tmp_path / "source.pdf"
    make_pdf(source)

    async def delegate(self, ctx):
        source.unlink()
        result = parsed(tmp_path)
        # This mock has no table headers to restore.
        (tmp_path / "source.tables.json").unlink()
        return result

    monkeypatch.setattr(DoclingParser, "parse", delegate)
    ctx = SimpleNamespace(
        rag=SimpleNamespace(
            _insurance_b_root=tmp_path, _insurance_b_structure_repair="headers-math-v2"
        ),
        source_path=lambda engine: source,
    )
    result = await B1DoclingParser().parse(ctx)
    assert "1.08^(N-1)" in result.content
