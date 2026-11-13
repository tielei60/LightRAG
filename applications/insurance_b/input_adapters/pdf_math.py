"""Restore variable superscripts from real PDF text geometry, without OCR.

Only adjacent numeric factors and raised, smaller variable exponents are
accepted. No value is calculated and no product name or expected answer is
used. Every rewritten occurrence has original glyph and page evidence.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from ..common import digest, inside, read_json, save_json


def superscript_pairs(path):
    pairs = []
    for page_number, page in enumerate(PdfReader(path).pages, 1):
        spans = []

        def visit(text, cm, tm, font, font_size, spans=spans):
            if text.strip() and abs(tm[1]) < 0.000001 and abs(tm[2]) < 0.000001:
                spans.append(
                    {
                        "text": text,
                        "x": tm[4],
                        "y": tm[5],
                        "font_size": abs(font_size)
                        * math.hypot(tm[0], tm[1])
                        * math.hypot(cm[0], cm[1]),
                    }
                )

        page.extract_text(visitor_text=visit)
        for i, span in enumerate(spans):
            exponent = re.sub(r"\s+", "", span["text"]).replace("−", "-")
            if i == 0 or not re.fullmatch(r"[A-Za-z][-+]\d+", exponent):
                continue
            previous = spans[i - 1]
            if not (
                0 < span["font_size"] < previous["font_size"] * 0.85
                and span["y"] > previous["y"] + previous["font_size"] * 0.2
                and 0 <= span["x"] - previous["x"] <= float(page.mediabox.width)
            ):
                continue
            line = []
            for item in reversed(spans[max(0, i - 12) : i]):
                if (
                    abs(item["y"] - previous["y"]) > 1
                    or abs(item["font_size"] - previous["font_size"]) > 0.2
                ):
                    break
                line.insert(0, item["text"])
            text = re.sub(r"\s+", "", "".join(line))
            base = re.search(r"(?P<base>\d+(?:\.\d+)?)$", text)
            if base:
                pairs.append(
                    {
                        "physical_page": page_number,
                        "base": base["base"],
                        "exponent": exponent,
                        "base_glyphs": line,
                        "base_position": {
                            k: previous[k] for k in ["x", "y", "font_size"]
                        },
                        "superscript_glyph": span,
                    }
                )
    return pairs


def replacements(text, pairs):
    changes = []
    for (base, exponent), expected in Counter(
        (x["base"], x["exponent"]) for x in pairs
    ).items():
        exp = re.escape(exponent).replace(r"\-", "[-−]")
        gap = r"(?:\s|\\n)*"
        flat = re.compile(r"(?<![\d.])" + re.escape(base) + gap + exp + r"(?!\w)")
        canonical = re.compile(re.escape(base) + r"\s*\^\s*\(\s*" + exp + r"\s*\)")
        count, already = len(flat.findall(text)), len(canonical.findall(text))
        if count + already != expected:
            raise ValueError(
                f"PDF superscript occurrence coverage differs: factor={base}, expected={expected}, flat={count}, canonical={already}"
            )
        if count:
            value = base + "^(" + exponent + ")"
            changes.append(
                {
                    "base": base,
                    "exponent": exponent,
                    "pattern": flat.pattern,
                    "replacement": value,
                    "occurrences": count,
                }
            )
            text = flat.sub(lambda match, value=value: value, text)
    return text, changes


def restore_formula_layout(result, root, pairs, source_sha256):
    if not pairs:
        return result
    blocks_path = inside(root, Path(result.blocks_path))
    before = result.content
    result.content, changes = replacements(before, pairs)
    if changes:

        def apply(text):
            for change in changes:
                text = re.sub(
                    change["pattern"],
                    lambda match, change=change: change["replacement"],
                    text,
                )
            return text

        records = [
            json.loads(line)
            for line in blocks_path.read_text().splitlines()
            if line.strip()
        ]
        for record in records:
            if record.get("type") == "content":
                record["content"] = apply(record.get("content", ""))
            elif record.get("type") == "meta":
                record["document_hash"] = "sha256:" + digest(result.content)
        blocks_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
        )
        for path in blocks_path.parent.glob("*.tables.json"):
            payload = read_json(path)
            for table in payload["tables"].values():
                table["content"] = apply(table["content"])
                if isinstance(table.get("table_header"), str):
                    table["table_header"] = apply(table["table_header"])
            save_json(path, payload)
    save_json(
        blocks_path.parent / "b1-formula-layout-repair.json",
        {
            "revision": "pdf-variable-superscript-v1",
            "source_sha256": source_sha256,
            "parsed_before_sha256": digest(before),
            "parsed_after_sha256": digest(result.content),
            "original_glyph_pairs": pairs,
            "changes": changes,
            "ocr_or_model_calls": 0,
        },
    )
    return result
