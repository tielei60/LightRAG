"""B1 source-preserving table repair before native P and embedding.

The parser stores Markdown header cells separately from table content. Small
tables reach P without those cells. Copy only the actual recorded header,
leaving zero, blank and dash cells unchanged, and retain an explicit audit.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

from .common import digest, inside, read_json, save_json

_TABLE = re.compile(r'<table\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</table>', re.DOTALL)


def restore_table_headers(result, root):
    blocks_path = inside(root, Path(result.blocks_path))
    table_paths = list(blocks_path.parent.glob("*.tables.json"))
    if not table_paths:
        return result
    if len(table_paths) != 1:
        raise ValueError("Ambiguous table sidecar during B1 structure repair")
    payload = read_json(table_paths[0])
    repairs, originals = {}, {}
    for key, table in payload["tables"].items():
        if table.get("format") != "json" or not table.get("table_header"):
            continue
        header = json.loads(table["table_header"])
        body = json.loads(table["content"])
        if not header or not body:
            continue
        if not all(
            isinstance(row, list) and len(row) == len(body[0]) for row in header + body
        ):
            raise ValueError("Original header and body have incompatible columns")
        if body[: len(header)] == header:
            continue
        originals[key] = body
        repairs[key] = header
        table["content"] = json.dumps(header + body, ensure_ascii=False)
        table["dimension"] = [len(header) + len(body), len(body[0])]
    if not repairs:
        return result

    def restore(text):
        def apply(match):
            key = match[1]
            if key not in repairs:
                return match[0]
            cells = json.loads(match[2])
            if cells[: len(repairs[key])] == repairs[key]:
                return match[0]
            if cells != originals[key]:
                raise ValueError(
                    "Cannot bind the original header to a changed table body"
                )
            return match[0].replace(
                match[2], json.dumps(repairs[key] + cells, ensure_ascii=False), 1
            )

        return _TABLE.sub(apply, text)

    before = result.content
    result.content = restore(result.content)
    records = [
        json.loads(line)
        for line in blocks_path.read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        if record.get("type") == "content":
            record["content"] = restore(record.get("content", ""))
        elif record.get("type") == "meta":
            record["document_hash"] = "sha256:" + digest(result.content)
    save_json(table_paths[0], payload)
    blocks_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    save_json(
        blocks_path.parent / "b1-structure-repair.json",
        {
            "revision": "headers-v1",
            "source_content_sha256": digest(before),
            "final_content_sha256": digest(result.content),
            "headers": repairs,
            "provenance": "Exact table_header cells from this parser's original-source sidecar; copied before native P",
        },
    )
    return result


class B1NativeParser:
    async def parse(self, ctx):
        from lightrag.parser.native_dispatch import NativeParser

        return restore_table_headers(
            await NativeParser().parse(ctx), ctx.rag._insurance_b_root
        )


class B1DoclingParser:
    async def parse(self, ctx):
        from lightrag.parser.external.docling.parser import DoclingParser

        from .input_adapters.pdf_math import restore_formula_layout, superscript_pairs

        repair_math = (
            getattr(ctx.rag, "_insurance_b_structure_repair", "") == "headers-math-v2"
        )
        source = ctx.source_path("docling") if repair_math else None
        source_sha256 = digest(source.read_bytes()) if source is not None else None
        pairs = superscript_pairs(source) if source is not None else []
        result = restore_table_headers(
            await DoclingParser().parse(ctx), ctx.rag._insurance_b_root
        )
        return (
            restore_formula_layout(
                result, ctx.rag._insurance_b_root, pairs, source_sha256
            )
            if repair_math
            else result
        )


def install_parsers():
    """Use the SDK registry extension inside this B process only."""
    from lightrag.parser.registry import parser_specs_snapshot, register_parser

    for name, cls in [("native", "B1NativeParser"), ("docling", "B1DoclingParser")]:
        register_parser(
            replace(parser_specs_snapshot()[name], impl=f"{__name__}:{cls}")
        )
