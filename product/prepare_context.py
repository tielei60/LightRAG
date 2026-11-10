"""Prepare source-preserving text inputs at explicitly reviewed boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

DELIMITER = "\n\n<<<LIGHTRAG_SOURCE_SECTION_BOUNDARY>>>\n\n"


def extract_source(path: Path) -> tuple[str, list[tuple[int, int]]]:
    if path.suffix.lower() == ".pdf":
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        if any(not page.strip() for page in pages):
            raise ValueError("A PDF page has no readable text; review the source")
    elif path.suffix.lower() in (".md", ".txt"):
        pages = [path.read_text(encoding="utf-8")]
    else:
        raise ValueError("Unsupported source format")
    spans = []
    cursor = 0
    for page in pages:
        spans.append((cursor, cursor + len(page)))
        cursor += len(page) + 1
    return "\n".join(pages), spans


def prepare(
    path: Path, product: str, reviewed_sections: list[dict], token_limit: int = 8192
) -> tuple[dict, dict]:
    """Build native /documents/text payloads without guessing clause boundaries."""
    source, pages = extract_source(path)
    if not product or "".join(product.split()) not in "".join(source.split()):
        raise ValueError("Product identity is not supported by the source")
    if DELIMITER in source or DELIMITER in product:
        raise ValueError("Reserved delimiter occurs in source content")
    starts = [section["start"] for section in reviewed_sections]
    if (
        not starts
        or starts[0] != 0
        or any(type(start) is not int for start in starts)
        or starts != sorted(set(starts))
        or starts[-1] >= len(source)
    ):
        raise ValueError("Reviewed boundaries must cover the full source in order")
    chunks, evidence, pieces = [], [], []
    for index, section in enumerate(reviewed_sections):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else len(source)
        body = source[start:end]
        if not section["anchor"] or not body.startswith(section["anchor"]):
            raise ValueError("Reviewed boundary no longer matches source text")
        page_ids = [i + 1 for i, (a, b) in enumerate(pages) if a < end and b > start]
        location = (
            "PDF物理页（从1计数）：" + ",".join(map(str, page_ids))
            if path.suffix.lower() == ".pdf"
            else "原文件文本范围（Unicode码点，左闭右开）：" + f"{start}:{end}"
        )
        context = (
            "【来源上下文：由原件身份核实，不是新增保险约定】\n"
            f"产品全称：{product}\n原文件：{path.name}\n{location}\n"
            "具体计划、责任及适用条件以以下原文为准；文件归属不等于所有计划均适用。\n"
            "【原文开始】\n"
        )
        chunks.append(context + body + "\n【原文结束】")
        pieces.append(body)
        evidence.append(
            {
                "order": index,
                "start": start,
                "end": end,
                "anchor": section["anchor"],
                "physical_pages": page_ids if path.suffix.lower() == ".pdf" else [],
                "source_text_sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        )
    if "".join(pieces) != source:
        raise ValueError("Prepared spans do not reproduce the extracted source")
    payload = {
        "text": DELIMITER.join(chunks),
        "file_source": path.name,
        "chunking": {
            "strategy": "fixed_token",
            "params": {
                "split_by_character": DELIMITER,
                "split_by_character_only": True,
                "chunk_token_size": token_limit,
                "chunk_overlap_token_size": 0,
            },
        },
    }
    return payload, {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "extracted_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "offset_unit": "Unicode code points in unmodified extracted text",
        "source_span_coverage": "complete; no omitted or overlapping source characters",
        "chunks": evidence,
    }
