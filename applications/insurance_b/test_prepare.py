import json

import pytest

from .common import digest, save_json
from .prepare import PRODUCTS, check_structure, checked_manifest, copy_sources


def corpus(root):
    for short, product in PRODUCTS.items():
        directory = root / short
        directory.mkdir(parents=True)
        (directory / (product + "条款.pdf")).write_bytes(b"synthetic policy bytes")
        (directory / (product + "产品说明书.pdf")).write_bytes(
            b"synthetic disclosure bytes"
        )
        (directory / (short + "投保规则.md")).write_text(
            "# " + product + "\nSynthetic test only"
        )


def test_copy_is_complete_frozen_and_self_contained(settings, tmp_path):
    origin = tmp_path / "external-originals"
    corpus(origin)
    result = copy_sources(settings, origin)
    assert result["product_count"] == 7
    assert result["document_count"] == 21
    assert len(result["documents"]) == 21
    assert copy_sources(settings, origin) == result
    assert checked_manifest(settings) == result


def test_revised_source_does_not_overwrite_frozen_copy(settings, tmp_path):
    origin = tmp_path / "external-originals"
    corpus(origin)
    result = copy_sources(settings, origin)
    source = origin / result["documents"][0]["relative_path"]
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Frozen B original changed"):
        copy_sources(settings, origin)
    assert checked_manifest(settings) == result


def test_missing_file_keeps_full_corpus_requirement(settings, tmp_path):
    origin = tmp_path / "external-originals"
    corpus(origin)
    next(origin.rglob("*.md")).unlink()
    with pytest.raises(ValueError, match="Exactly one"):
        copy_sources(settings, origin)


@pytest.mark.parametrize("defect", ["hash", "parser", "page", "missing_table"])
def test_structure_checks_reject_untraceable_parse(tmp_path, defect):
    entry = {"sha256": "original", "parser": "docling"}
    meta = {
        "type": "meta",
        "document_hash": "sha256:" + digest("real text"),
        "parse_engine": "docling",
    }
    block = {
        "type": "content",
        "blockid": "one",
        "content": "real text",
        "positions": [{"anchor": "1"}],
    }
    if defect == "hash":
        meta["document_hash"] = "sha256:different"
    elif defect == "parser":
        meta["parse_engine"] = "legacy"
    elif defect == "page":
        block["positions"] = [{"anchor": "invented page"}]
    else:
        save_json(
            tmp_path / "source.tables.json",
            {"tables": {"table-missing": {"content": '[["real cell"]]'}}},
        )
    path = tmp_path / "source.blocks.jsonl"
    path.write_text(json.dumps(meta) + "\n" + json.dumps(block) + "\n")
    with pytest.raises(ValueError):
        check_structure(
            entry,
            {
                "parse_engine": "docling",
                "content": "real text",
                "source_sha256": "original",
            },
            path,
        )
