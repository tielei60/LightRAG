"""Prepare hash-pinned, source-preserving HTTP inputs. Never imports or calls a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lightrag.chunker import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from product.prepare_context import extract_source, prepare


def build(
    manifest: dict, root: Path, tokenizer, *, tokenizer_model: str | None = None
) -> list[dict]:
    """Validate every input before producing any output; boundaries are explicit data."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest version")
    limit = manifest["max_chunk_tokens"]
    if type(limit) is not int or not 1 <= limit <= 8192:
        raise ValueError("Invalid chunk token limit")
    results = []
    seen = set()
    for record in manifest["documents"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or path in seen:
            raise ValueError("Source must be unique and within the chosen root")
        seen.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["source_sha256"]:
            raise ValueError("Original file changed; review boundaries again")
        text, _ = extract_source(path)
        if hashlib.sha256(text.encode()).hexdigest() != record["extracted_text_sha256"]:
            raise ValueError("Extracted text changed; review boundaries again")
        payload, evidence = prepare(
            path, manifest["product"], record["sections"], limit
        )
        if any(
            evidence[key] != record[key]
            for key in ("source_sha256", "extracted_text_sha256")
        ):
            raise ValueError("Source changed during preparation; no output produced")
        chunks = chunking_by_token_size(
            tokenizer, payload["text"], **payload["chunking"]["params"]
        )
        if len(chunks) != len(record["sections"]) or any(
            c["tokens"] > limit for c in chunks
        ):
            raise ValueError(
                "A source clause exceeds the budget; do not silently split it"
            )
        results.append(
            {
                "payload": payload,
                "evidence": evidence,
                "tokenizer_model": tokenizer_model,
                "chunk_tokens": [c["tokens"] for c in chunks],
            }
        )
    if not results:
        raise ValueError("No documents specified")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    results = build(
        json.loads(raw),
        args.root,
        TiktokenTokenizer(model_name="gpt-4o-mini"),
        tokenizer_model="gpt-4o-mini",
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for index, result in enumerate(results):
        (args.output / f"document-{index}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
    (args.output / "manifest.json").write_bytes(raw)
    print(
        f"Prepared {len(results)} documents / {sum(len(r['chunk_tokens']) for r in results)} chunks. No import or model call."
    )


if __name__ == "__main__":
    main()
