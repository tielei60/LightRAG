"""Offline source-reference compilation and fail-closed provenance checks."""

import copy
import hashlib
import unittest

from product.source_links import compile_catalog, resolve_support_chunks


def fixture(reference="术语（见释义）", title="术语"):
    text = f"产品甲\n责任甲\n适用{reference}。另有术语。\n释义\n{title}：定义正文。"

    def span(quote, last=False):
        start = text.rindex(quote) if last else text.index(quote)
        return {"start": start, "end": start + len(quote), "quote": quote}

    section = span(f"适用{reference}。另有术语。")
    catalog = {
        "sources": [
            {
                "id": "s",
                "product": "产品甲",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "units": [
            {
                "id": "u",
                "source_id": "s",
                "kind": "section",
                "entity_id": "exact-clause",
                "span": section,
                "reference_spans": [span(reference)],
                "scope_id": "scope",
            },
            {
                "id": "d",
                "source_id": "s",
                "kind": "definition",
                "entity_id": "exact-definition",
                "span": span(f"{title}：定义正文。"),
                "term_span": span(title, last=True),
            },
        ],
        "scopes": [
            {
                "id": "scope",
                "source_id": "s",
                "owner_entity_id": "exact-owner",
                "heading_span": span("责任甲"),
                "span": span(f"责任甲\n{section['quote']}"),
            }
        ],
    }
    inventory = {
        key: "产品甲" for key in ("exact-clause", "exact-definition", "exact-owner")
    }
    return catalog, inventory


class SourceLinksTests(unittest.TestCase):
    def test_native_candidates_have_exact_original_provenance(self):
        catalog, inventory = fixture()
        before = copy.deepcopy(catalog)
        result = compile_catalog(catalog, inventory)
        self.assertEqual(catalog, before)
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(len(result["relationships"]), 2)
        self.assertEqual(result["relationships"][0]["target"], "exact-definition")
        for entry in result["provenance"]:
            for proof in entry["evidence"]:
                source = catalog["sources"][0]
                self.assertEqual(
                    proof["quote"], source["text"][proof["start"] : proof["end"]]
                )
                self.assertEqual(proof["sha256"], source["sha256"])

    def test_width_and_whitespace_normalization_preserves_offsets(self):
        catalog, inventory = fixture("Ａ 术\n语（ 见 释义 ）", "A术语")
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 2)
        self.assertEqual(
            result["provenance"][0]["evidence"][0]["quote"], "Ａ 术\n语（ 见 释义 ）"
        )

    def test_stale_hash_rejected(self):
        catalog, inventory = fixture()
        catalog["sources"][0]["text"] += "changed"
        with self.assertRaises(ValueError):
            compile_catalog(catalog, inventory)

    def test_stale_quote_and_offsets_rejected(self):
        for key, value in [("quote", "wrong"), ("start", -1), ("end", True)]:
            catalog, inventory = fixture()
            catalog["units"][0]["span"][key] = value
            with self.assertRaises(ValueError):
                compile_catalog(catalog, inventory)

    def test_cross_product_inventory_rejected(self):
        catalog, inventory = fixture()
        inventory["exact-definition"] = "产品乙"
        with self.assertRaises(ValueError):
            compile_catalog(catalog, inventory)

    def test_definitions_from_other_sources_are_not_joined(self):
        catalog, inventory = fixture()
        other = copy.deepcopy(catalog["sources"][0])
        other.update(id="other", product="产品乙")
        catalog["sources"].append(other)
        catalog["units"][1]["source_id"] = "other"
        inventory["exact-definition"] = "产品乙"
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["diagnostics"][0]["code"], "missing_definition")

    def test_same_title_is_ambiguous_even_with_same_entity(self):
        catalog, inventory = fixture()
        duplicate = copy.deepcopy(catalog["units"][1])
        duplicate["id"] = "duplicate"
        catalog["units"].append(duplicate)
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["diagnostics"][0]["code"], "ambiguous_definition")

    def test_coincidental_mention_not_reference(self):
        catalog, inventory = fixture("术语")
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(
            result["diagnostics"][0]["code"], "not_explicit_definition_reference"
        )

    def test_unreviewed_marker_is_diagnostic_only(self):
        catalog, inventory = fixture()
        catalog["units"][0]["reference_spans"] = []
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(
            result["diagnostics"][0]["code"], "unreviewed_definition_reference"
        )

    def test_no_suffix_or_fuzzy_guess(self):
        catalog, inventory = fixture("非术语（见释义）")
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["diagnostics"][0]["code"], "missing_definition")

    def test_unknown_entity_is_not_created(self):
        catalog, inventory = fixture()
        del inventory["exact-clause"]
        with self.assertRaises(ValueError):
            compile_catalog(catalog, inventory)

    def test_out_of_scope_parent_does_not_link(self):
        catalog, inventory = fixture()
        catalog["scopes"][0]["span"] = catalog["scopes"][0]["heading_span"]
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["diagnostics"][0]["code"], "unproven_scope")

    def test_missing_scope_remains_unknown(self):
        catalog, inventory = fixture()
        catalog["scopes"] = []
        result = compile_catalog(catalog, inventory)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["diagnostics"][0]["code"], "unproven_scope")

    def test_no_zero_isolate_requirement(self):
        catalog, inventory = fixture()
        catalog["units"][0].pop("scope_id")
        catalog["units"][0]["reference_spans"] = []
        self.assertEqual(compile_catalog(catalog, inventory)["relationships"], [])

    def test_no_general_compatibility_aliases(self):
        catalog, inventory = fixture("①（见释义）", "1")
        result = compile_catalog(catalog, inventory)
        self.assertEqual(result["diagnostics"][0]["code"], "missing_definition")

    def test_exception_context_survives_definition_reference(self):
        catalog, inventory = fixture()
        source = catalog["sources"][0]
        old = "适用术语（见释义）。另有术语。"
        new = "除术语（见释义）外，均不予适用。"
        text = source["text"].replace(old, new)
        # Rebase all reviewed spans to the edited synthetic source.
        for row in catalog["units"] + catalog["scopes"]:
            for field in ("span", "term_span", "heading_span"):
                if field in row:
                    quote = row[field]["quote"].replace(old, new)
                    start = (
                        text.rindex(quote)
                        if field == "term_span"
                        else text.index(quote)
                    )
                    row[field] = {
                        "start": start,
                        "end": start + len(quote),
                        "quote": quote,
                    }
            for ref in row.get("reference_spans", []):
                start = text.index(ref["quote"])
                ref.update(start=start, end=start + len(ref["quote"]))
        source.update(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
        result = compile_catalog(catalog, inventory)
        edge, proof = result["relationships"][0], result["provenance"][0]
        self.assertEqual(edge["keywords"], "引用释义")
        self.assertIn(new, edge["description"])
        self.assertEqual(proof["support_spans"][0]["quote"], new)
        self.assertEqual(proof["evidence"][0]["quote"], "术语（见释义）")

    def test_broad_scope_is_proof_not_description(self):
        catalog, inventory = fixture()
        text = catalog["sources"][0]["text"]
        catalog["scopes"][0]["span"] = {"start": 0, "end": len(text), "quote": text}
        result = compile_catalog(catalog, inventory)
        edge, proof = result["relationships"][1], result["provenance"][1]
        self.assertEqual(edge["keywords"], "原文范围归属")
        self.assertNotIn("定义正文", edge["description"])
        self.assertEqual(proof["evidence"][-1]["quote"], text)
        self.assertEqual(len(proof["support_spans"]), 2)
        for edge, proof in zip(result["relationships"], result["provenance"]):
            self.assertEqual(
                edge["description"],
                "\n\n".join(
                    dict.fromkeys(span["quote"] for span in proof["support_spans"])
                ),
            )

    def test_duplicate_reference_refused(self):
        catalog, inventory = fixture()
        catalog["units"][0]["reference_spans"] *= 2
        with self.assertRaises(ValueError):
            compile_catalog(catalog, inventory)


class SupportChunkTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"s": "source-a.txt"}
        self.spans = [
            {"source_id": "s", "quote": "Header"},
            {"source_id": "s", "quote": "Specific body"},
        ]

    def test_repeated_header_uses_body_chunk_only(self):
        chunks = {
            "a": {"file_path": "source-a.txt", "content": "Header unrelated"},
            "b": {"file_path": "source-a.txt", "content": "Header Specific body"},
            "c": {"file_path": "source-a.txt", "content": "Header other"},
        }
        self.assertEqual(
            resolve_support_chunks(self.spans, self.sources, chunks), ["b"]
        )

    def test_two_genuinely_needed_chunks(self):
        chunks = {
            "a": {"file_path": "source-a.txt", "content": "Header"},
            "b": {"file_path": "source-a.txt", "content": "Specific body"},
        }
        self.assertEqual(
            resolve_support_chunks(self.spans, self.sources, chunks), ["a", "b"]
        )

    def test_wrong_source_same_quote_refused(self):
        for path in ("source-b.txt", "/other/source-a.txt"):
            chunks = {"a": {"file_path": path, "content": "Header Specific body"}}
            with self.assertRaises(ValueError):
                resolve_support_chunks(self.spans, self.sources, chunks)

    def test_missing_support_refused(self):
        chunks = {"a": {"file_path": "source-a.txt", "content": "Header"}}
        with self.assertRaises(ValueError):
            resolve_support_chunks(self.spans, self.sources, chunks)
        with self.assertRaises(ValueError):
            resolve_support_chunks(self.spans, {}, chunks)
        with self.assertRaises(ValueError):
            resolve_support_chunks([], self.sources, chunks)

    def test_deterministic_ties_independent_of_input_order(self):
        chunk = {"file_path": "source-a.txt", "content": "Header Specific body"}
        for chunks in ({"z": chunk, "a": chunk}, {"a": chunk, "z": chunk}):
            self.assertEqual(
                resolve_support_chunks(self.spans, self.sources, chunks), ["a"]
            )

    def test_no_whitespace_or_width_fuzzy_match(self):
        chunks = {
            "a": {"file_path": "source-a.txt", "content": "Header Specific  body"}
        }
        with self.assertRaises(ValueError):
            resolve_support_chunks(self.spans, self.sources, chunks)


if __name__ == "__main__":
    unittest.main()
