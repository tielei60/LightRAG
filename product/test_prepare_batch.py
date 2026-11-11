"""Offline safety checks for explicit clause preparation."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from product.prepare_batch import build
from lightrag.exceptions import ChunkTokenLimitExceededError


class Tokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


@pytest.fixture
def source():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        text = "# Product\n1. First\n完整条件。\n2. Second\n完整例外。"
        path = root / "source.md"
        path.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        manifest = {
            "schema_version": 1,
            "product": "Product",
            "max_chunk_tokens": 1000,
            "documents": [
                {
                    "path": "source.md",
                    "source_sha256": digest,
                    "extracted_text_sha256": digest,
                    "sections": [
                        {"start": 0, "anchor": "# Product"},
                        {"start": text.index("2. Second"), "anchor": "2. Second"},
                    ],
                }
            ],
        }
        yield root, path, manifest


def test_complete_source_and_scope(source):
    root, _, manifest = source
    result = build(manifest, root, Tokenizer())[0]
    assert len(result["chunk_tokens"]) == 2
    assert result["tokenizer_model"] is None
    assert result["payload"]["file_source"] == "source.md"
    assert "完整例外。" in result["payload"]["text"]


def test_source_changes_fail_closed(source):
    root, path, manifest = source
    path.write_text("changed")
    with pytest.raises(ValueError, match="Original file changed"):
        build(manifest, root, Tokenizer())


def test_extraction_changes_fail_closed(source):
    root, _, manifest = source
    manifest["documents"][0]["extracted_text_sha256"] = "changed"
    with pytest.raises(ValueError, match="Extracted text changed"):
        build(manifest, root, Tokenizer())


def test_duplicate_source_is_rejected(source):
    root, _, manifest = source
    manifest["documents"] *= 2
    with pytest.raises(ValueError, match="unique"):
        build(manifest, root, Tokenizer())


def test_path_escape_is_rejected(source):
    root, _, manifest = source
    manifest["documents"][0]["path"] = "../outside.md"
    with pytest.raises(ValueError, match="within"):
        build(manifest, root, Tokenizer())


def test_budget_cannot_split_clause(source):
    root, _, manifest = source
    manifest["max_chunk_tokens"] = 1
    with pytest.raises(ChunkTokenLimitExceededError):
        build(manifest, root, Tokenizer())


def test_empty_batch_is_rejected(source):
    root, _, manifest = source
    manifest["documents"] = []
    with pytest.raises(ValueError, match="No documents"):
        build(manifest, root, Tokenizer())


@pytest.mark.parametrize("limit", [True, 0, -1, 8193, "1000"])
def test_invalid_limits(source, limit):
    root, _, manifest = source
    manifest["max_chunk_tokens"] = limit
    with pytest.raises(ValueError, match="Invalid chunk token limit"):
        build(manifest, root, Tokenizer())


def test_symlink_escape(source):
    root, _, manifest = source
    (root / "outside-link.md").symlink_to(root.parent / "outside.md")
    manifest["documents"][0]["path"] = "outside-link.md"
    with pytest.raises(ValueError, match="within"):
        build(manifest, root, Tokenizer())


def test_source_mutation_during_prepare(source, monkeypatch):
    import product.prepare_batch as module

    root, _, manifest = source
    original = module.prepare

    def changed(*args, **kwargs):
        payload, evidence = original(*args, **kwargs)
        evidence["source_sha256"] = "changed after validation"
        return payload, evidence

    monkeypatch.setattr(module, "prepare", changed)
    with pytest.raises(ValueError, match="during preparation"):
        build(manifest, root, Tokenizer())


def test_failed_batch_does_not_create_output(source, monkeypatch):
    import json
    import sys
    import product.prepare_batch as module

    root, _, manifest = source
    manifest["documents"].append({**manifest["documents"][0], "path": "missing.md"})
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    out = root / "output"
    monkeypatch.setattr(module, "TiktokenTokenizer", lambda **_: Tokenizer())
    monkeypatch.setattr(
        sys, "argv", ["prepare_batch", str(path), str(out), "--root", str(root)]
    )
    with pytest.raises(FileNotFoundError):
        module.main()
    assert not out.exists()
