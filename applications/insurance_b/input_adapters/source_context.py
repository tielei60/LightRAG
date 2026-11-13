"""Carry verified source identity into final P chunks before native hashing.

The native P implementation remains the splitter. This B-process wrapper saves
its raw output, adds the same non-contractual identity envelope understood by
the common answer adapter, then lets the native pipeline hash/embed/store it.
"""

import copy
import json
from functools import wraps
from pathlib import Path

from ..common import digest, inside, read_json, save_json
from ..evidence_adapter import identity_span

VERSION = "source-context-v1"
PREFIX = "【来源标识：整理信息，不是合同新约定】\n"
START, END = "【主原文开始】\n", "\n【主原文结束】"


def unwrap(content, *, product, source):
    prefix = PREFIX + f"产品：{product}\n原文件：{source}\n" + START
    if not content.startswith(prefix) or not content.endswith(END):
        raise ValueError(
            "The indexed B identity envelope differs from its verified source"
        )
    return content[len(prefix) : -len(END)], len(prefix)


def finalize_chunks(chunks, *, entry, parsed_text, tokenizer, token_limit=8192):
    start, end = identity_span(parsed_text, entry["product"])
    quote = parsed_text[start:end]
    source = Path(entry["relative_path"]).name
    prefix = PREFIX + f"产品：{entry['product']}\n原文件：{source}\n" + START
    result = []
    for row in chunks:
        body = row["content"]
        if any(marker in body for marker in [PREFIX, START, END]):
            raise ValueError("A source contains reserved identity-envelope markers")
        if quote not in body:
            body = quote + "\n\n" + body
        value = prefix + body + END
        tokens = len(tokenizer.encode(value))
        if tokens > token_limit:
            raise ValueError(
                "Final P source identity would exceed the embedding budget"
            )
        result.append({**row, "content": value, "tokens": tokens})
    return result, {
        "parsed_text_span": [start, end],
        "verbatim_quote": quote,
        "parsed_text_sha256": digest(parsed_text),
        "source_sha256": entry["sha256"],
    }


def install(settings, extraction):
    import lightrag.chunker as chunkers

    current = chunkers.chunking_by_paragraph_semantic
    native = getattr(current, "_insurance_b_native_p", current)

    @wraps(native)
    def prepared_p(tokenizer, content, chunk_token_size=2000, **kwargs):
        entry = extraction.entry
        if not entry:
            raise ValueError("A final P adapter requires an active verified source")
        blocks = inside(settings.root, Path(kwargs["blocks_path"]))
        metadata = [
            row
            for line in blocks.read_text().splitlines()
            if line.strip()
            if (row := json.loads(line)).get("type") == "meta"
        ]
        if (
            len(metadata) != 1
            or metadata[0].get("document_name") != Path(entry["relative_path"]).name
        ):
            raise ValueError("The active P sidecar belongs to another source")
        raw = native(tokenizer, content, chunk_token_size, **kwargs)
        final, origin = finalize_chunks(
            raw,
            entry=entry,
            parsed_text=content,
            tokenizer=tokenizer,
            token_limit=settings.integer("EMBEDDING_TOKEN_LIMIT"),
        )
        report = {
            "version": VERSION,
            "native_splitter": native.__module__ + "." + native.__name__,
            "native_p_called": True,
            "source": entry["relative_path"],
            "requested_size": chunk_token_size,
            "effective_options": {
                key: value
                for key, value in kwargs.items()
                if key not in {"blocks_path", "doc_id"}
            },
            "origin": origin,
            "raw_p_chunks": copy.deepcopy(raw),
            "final_chunks": [
                {
                    "order": r["chunk_order_index"],
                    "sha256": digest(r["content"]),
                    "tokens": r["tokens"],
                }
                for r in final
            ],
        }
        path = settings.run_dir / "final-chunk-adapter" / (kwargs["doc_id"] + ".json")
        if path.exists() and read_json(path) != report:
            raise ValueError("Frozen native P or final source context changed")
        save_json(path, report)
        return final

    prepared_p._insurance_b_native_p = native
    # Process-local adaptation only. No SDK file, A process or splitter changes.
    chunkers.chunking_by_paragraph_semantic = prepared_p
