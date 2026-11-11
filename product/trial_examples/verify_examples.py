"""Verify sample byte hashes, extraction offsets and unique source anchors offline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from product.prepare_context import extract_source


def main() -> None:
    base = Path(__file__).resolve().parent
    count = 0
    for directory in ("synthetic", "jingkangyouhu"):
        path = base / directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        audit = json.loads(
            path.with_name("verification.json").read_text(encoding="utf-8")
        )
        root = (path.parent / manifest["source_root"]).resolve()
        ids = {u["id"] for d in manifest["documents"] for u in d["units"]}
        for document in manifest["documents"]:
            source = root / document["source"]
            assert source.name == document["file_path"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest()
                == document["source_sha256"]
            )
            text, pages = extract_source(source)
            assert (
                hashlib.sha256(text.encode()).hexdigest()
                == audit["extracted_text_sha256"]
            )
            for unit in document["units"]:
                assert 0 <= unit["start"] < unit["end"] <= len(text)
                quote = text[unit["start"] : unit["end"]]
                assert unit["title"] in quote
                assert all(a in quote and text.count(a) == 1 for a in unit["anchors"])
                assert set(unit["requires"]) <= ids
                expected = audit["units"][unit["id"]]
                assert (
                    hashlib.sha256(quote.encode()).hexdigest()
                    == expected["span_text_sha256"]
                )
                assert quote.startswith(expected["start_boundary"])
                assert text[unit["end"] :].startswith(expected["next_boundary"])
                actual_pages = (
                    [
                        i + 1
                        for i, (a, b) in enumerate(pages)
                        if a < unit["end"] and b > unit["start"]
                    ]
                    if source.suffix.lower() == ".pdf"
                    else []
                )
                assert actual_pages == expected["physical_pages"]
                count += 1
    print(f"Verified 2 manifests and {count} original-source spans offline.")


if __name__ == "__main__":
    main()
