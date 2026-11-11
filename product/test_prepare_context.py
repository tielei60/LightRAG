"""Offline checks for source coverage and the public delimiter chunking contract."""

import tempfile
import unittest
from pathlib import Path

from lightrag.chunker import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from product.prepare_context import prepare


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "source.md"
        self.text = "# Fixture Product\n\n1.1 First clause\n条件甲。\n\n1.2 Second clause\n条件乙。"
        self.path.write_text(self.text)
        self.sections = [
            {"start": 0, "anchor": "# Fixture Product"},
            {"start": self.text.index("1.2"), "anchor": "1.2 Second clause"},
        ]

    def test_native_chunks_preserve_context_and_complete_source(self):
        payload, report = prepare(self.path, "Fixture Product", self.sections)
        chunks = chunking_by_token_size(
            TiktokenTokenizer(model_name="gpt-4o-mini"),
            payload["text"],
            **payload["chunking"]["params"],
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("Fixture Product" in c["content"] for c in chunks))
        self.assertEqual(
            "".join(self.text[c["start"] : c["end"]] for c in report["chunks"]),
            self.text,
        )
        self.assertIn("1.2 Second clause\n条件乙。", chunks[1]["content"])

    def test_unproven_product_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.path, "Another Product", self.sections)

    def test_omitted_source_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.path, "Fixture Product", self.sections[1:])

    def test_stale_source_anchor_is_rejected(self):
        self.path.write_text(self.text.replace("Second clause", "Changed clause"))
        with self.assertRaises(ValueError):
            prepare(self.path, "Fixture Product", self.sections)

    def test_duplicate_or_reordered_boundaries_are_rejected(self):
        for sections in [self.sections[::-1], self.sections + [self.sections[-1]]]:
            with self.assertRaises(ValueError):
                prepare(self.path, "Fixture Product", sections)

    def test_verified_parent_context_preserves_separate_child_scope(self):
        parent_start = self.text.index("1.1")
        parent_end = self.text.index("\n条件甲")
        self.sections[1].update(
            parent_spans=[
                {"start": parent_start, "end": parent_end, "anchor": "1.1 First clause"}
            ],
            scope_quote="1.2 Second clause",
        )
        payload, report = prepare(self.path, "Fixture Product", self.sections)
        self.assertIn("【本段原文责任标题】\n1.2 Second clause", payload["text"])
        self.assertEqual(report["chunks"][1]["parent_spans"][0]["start"], parent_start)

    def test_scope_heading_cannot_be_borrowed_from_another_clause(self):
        self.sections[1]["scope_quote"] = "1.1 First clause"
        with self.assertRaises(ValueError):
            prepare(self.path, "Fixture Product", self.sections)

    def test_parent_context_cannot_include_future_or_changed_text(self):
        for parent in [
            {"start": 0, "end": len(self.text), "anchor": "# Fixture"},
            {"start": 0, "end": 5, "anchor": "changed"},
        ]:
            self.sections[1]["parent_spans"] = [parent]
            with self.assertRaises(ValueError):
                prepare(self.path, "Fixture Product", self.sections)


if __name__ == "__main__":
    unittest.main()
