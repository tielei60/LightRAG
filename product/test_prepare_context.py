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


if __name__ == "__main__":
    unittest.main()
