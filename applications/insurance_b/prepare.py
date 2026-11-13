"""Copy the seven-product corpus and run the native parser registry offline."""

import asyncio
import json
import re
import shutil
import time
from pathlib import Path

from .common import digest, inside, read_json, save_json
from .config import Settings

PRODUCTS = {
    "京康优护": "北京人寿京康优护医疗保险",
    "京安泰": "北京人寿京安泰综合意外伤害保险",
    "京富守护星": "北京人寿京富守护星年金保险",
    "京富添利": "北京人寿京富添利终身寿险（万能型）",
    "京安心": "北京人寿京安心老年人意外伤害保险",
    "京福宏裕（尊享版）": "北京人寿京福宏裕（尊享版）终身寿险（分红型）",
    "京福盈嘉": "北京人寿京福盈嘉终身寿险",
}


def check_structure(entry: dict, result: dict, blocks_path: Path) -> dict:
    """Check mechanical provenance; semantic layout remains an original-source review."""
    rows = [
        json.loads(line)
        for line in blocks_path.read_text().splitlines()
        if line.strip()
    ]
    metadata = [row for row in rows if row.get("type") == "meta"]
    blocks = [row for row in rows if row.get("type") == "content"]
    if (
        len(metadata) != 1
        or not blocks
        or len({row["blockid"] for row in blocks}) != len(blocks)
    ):
        raise ValueError("Structured blocks are empty or have ambiguous identities")
    meta = metadata[0]
    # The SDK document_hash hashes merged parsed text, rather than source-file bytes.
    if (
        result.get("source_sha256") != entry["sha256"]
        or meta.get("document_hash") != "sha256:" + digest(result.get("content", ""))
        or meta.get("parse_engine") != entry["parser"]
        or result.get("parse_engine") != entry["parser"]
    ):
        raise ValueError("Structured original/text hash or effective parser differs")
    tables_files = list(blocks_path.parent.glob("*.tables.json"))
    if len(tables_files) > 1:
        raise ValueError("Structured table sidecar is ambiguous")
    tables = read_json(tables_files[0])["tables"] if tables_files else {}
    from .evidence_adapter import project_tables

    seen_tables = set()
    pages = set()
    for block in blocks:
        if not block.get("content", "").strip() or not block.get("positions"):
            raise ValueError(
                "A structured block has no text or original-source position"
            )
        if not isinstance(block.get("parent_headings", []), list):
            raise TypeError("A structured heading path is invalid")
        _, mappings = project_tables(block["content"], tables)
        seen_tables.update(item["table_id"] for item in mappings)
        for position in block["positions"]:
            if entry["parser"] == "docling":
                if (
                    not str(position.get("anchor", "")).isdigit()
                    or int(position["anchor"]) < 1
                ):
                    raise ValueError("A PDF block has no real physical-page anchor")
                pages.add(int(position["anchor"]))
    if seen_tables != tables.keys():
        raise ValueError("A structured table is absent from the parsed text")
    return {
        "mechanical_checks_passed": True,
        "blocks": len(blocks),
        "headings": sum(bool(row.get("heading")) for row in blocks),
        "tables": len(tables),
        "physical_pages_with_positions": sorted(pages),
        "semantic_structure_review": "pending original-source review of reading order, headings, table layout, continuation and footnotes",
    }


def copy_sources(settings: Settings, source_root: Path) -> dict:
    """This explicit one-time copy never reads A configuration or task output."""
    source_root = source_root.resolve(strict=True)
    documents = []
    for short, product in PRODUCTS.items():
        directory = source_root / short
        selected = [
            ("policy", sorted(directory.glob("*条款.pdf"))),
            ("disclosure", sorted(directory.glob("*产品说明书.pdf"))),
            ("underwriting", sorted(directory.glob("*.md"))),
        ]
        for role, candidates in selected:
            if len(candidates) != 1:
                raise ValueError(f"Exactly one {role} original is required for {short}")
            source = candidates[0]
            relative = source.relative_to(source_root)
            destination = inside(settings.root, settings.originals / relative)
            payload = source.read_bytes()
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError(f"Frozen B original changed: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(payload)
            documents.append(
                {
                    "document_id": "source-" + digest(str(relative))[:24],
                    "product": product,
                    "product_directory": short,
                    "role": role,
                    "relative_path": str(relative),
                    "sha256": digest(payload),
                    "size_bytes": len(payload),
                    "document_version": "not stated",
                    "parser": "docling"
                    if source.suffix.lower() == ".pdf"
                    else "native",
                    "requested_chunker": "P",
                }
            )
    manifest = {
        "version": 1,
        "product_count": 7,
        "document_count": 21,
        "source_root": str(settings.originals),
        "documents": documents,
    }
    target = settings.run_dir / "manifest.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("The run already contains a different frozen manifest")
    save_json(target, manifest)
    return manifest


def checked_manifest(settings: Settings) -> dict:
    manifest = read_json(settings.run_dir / "manifest.json")
    if (
        manifest.get("source_root") != str(settings.originals)
        or manifest.get("document_count") != 21
    ):
        raise ValueError("The B manifest is incomplete or belongs to another project")
    if (
        len(manifest["documents"]) != 21
        or len({d["product"] for d in manifest["documents"]}) != 7
    ):
        raise ValueError("The full seven-product corpus must remain in the denominator")
    for entry in manifest["documents"]:
        source = inside(settings.root, settings.originals / entry["relative_path"])
        if digest(source.read_bytes()) != entry["sha256"]:
            raise ValueError(f"B original checksum mismatch: {entry['relative_path']}")
    return manifest


def stage_input(settings: Settings, entry: dict) -> Path:
    source = inside(settings.root, settings.originals / entry["relative_path"])
    destination = inside(settings.root, settings.input_dir / source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination.read_bytes()) != entry["sha256"]:
        raise ValueError("B input file identity changed")
    if not destination.exists():
        shutil.copyfile(source, destination)
    return destination


async def parse_all(settings: Settings, *, products: set[str] | None = None) -> dict:
    settings.activate()
    if settings.get("STRUCTURE_REPAIR") in {"headers-v1", "headers-math-v2"}:
        from .structure_repair import install_parsers

        install_parsers()
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.registry import get_parser

    class PreparationContext(ParseContext):
        async def archive_source(self, source_path: str):
            # The actual native ingestion pipeline owns source archiving.
            return None

    manifest = checked_manifest(settings)
    previous = settings.run_dir / "parse-report.json"
    if previous.exists():
        save_json(
            settings.run_dir / "parse-history" / (str(time.time_ns()) + ".json"),
            read_json(previous),
        )
    rows = []
    entries = sorted(
        manifest["documents"],
        key=lambda entry: (entry["parser"] == "docling", entry["size_bytes"]),
    )
    for entry in entries:
        if products and entry["product_directory"] not in products:
            continue
        path = stage_input(settings, entry)
        result_path = settings.run_dir / "parsed" / (entry["document_id"] + ".json")
        row = {
            "document_id": entry["document_id"],
            "file": path.name,
            "state": "parsing",
        }
        rows.append(row)
        save_json(
            settings.run_dir / "parse-report.json",
            {"expected_documents": 21, "documents": rows},
        )
        try:
            context = PreparationContext(
                build_debug_rag(),
                entry["document_id"],
                str(path),
                {
                    "parse_format": "pending_parse",
                    "content": "",
                    "process_options": "P",
                },
            )
            context.rag._insurance_b_root = settings.root
            context.rag._insurance_b_structure_repair = settings.get("STRUCTURE_REPAIR")
            result = (await get_parser(entry["parser"]).parse(context)).to_dict()
            blocks = inside(settings.root, Path(result.get("blocks_path") or ""))
            if not blocks.is_file() or not result.get("content", "").strip():
                raise ValueError(
                    "Structured parser did not emit usable text and blocks"
                )
            result["source_sha256"] = entry["sha256"]
            result["structure_checks"] = check_structure(entry, result, blocks)
            save_json(result_path, result)
            row.update(
                state="parsed",
                blocks_path=str(blocks),
                effective_parser=result.get("parse_engine"),
                structure_checks=result["structure_checks"],
            )
        except Exception as exc:  # noqa: BLE001 -- preserve a failed parse without dropping its source
            row.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        save_json(
            settings.run_dir / "parse-report.json",
            {"expected_documents": 21, "documents": rows},
        )
        await asyncio.sleep(0)
    report = {
        "expected_documents": 21,
        "selected_documents": len(rows),
        "parsed": sum(row["state"] == "parsed" for row in rows),
        "failed": sum(row["state"] == "failed" for row in rows),
        "documents": rows,
    }
    report["full_corpus_passed"] = len(rows) == 21 and report["parsed"] == 21
    save_json(settings.run_dir / "parse-report.json", report)
    return report


async def recover_docling_task(
    settings: Settings, document_id: str, task_id: str
) -> dict:
    """Finish a known timed-out submission without uploading the original again."""
    import httpx
    from lightrag.parser.external.docling.cache import (
        compute_options_signature,
        snapshot_tunable_env,
    )
    from lightrag.parser.external.docling.client import (
        FIXED_CONSTANTS,
        DoclingRawClient,
    )
    from lightrag.parser.external.docling.manifest import (
        build_and_write_docling_manifest,
        select_main_json,
    )

    settings.activate()
    manifest = checked_manifest(settings)
    entry = next(
        (
            entry
            for entry in manifest["documents"]
            if entry["document_id"] == document_id
        ),
        None,
    )
    if entry is None or entry["parser"] != "docling":
        raise ValueError("Recovery requires a Docling document in the B manifest")
    if not re.fullmatch(r"[0-9a-f-]{36}", task_id):
        raise ValueError("Invalid Docling task ID")
    reports = [
        settings.run_dir / "parse-report.json",
        *list((settings.run_dir / "parse-history").glob("*.json")),
    ]
    known = any(
        any(
            row.get("document_id") == document_id and task_id in row.get("error", "")
            for row in read_json(path).get("documents", [])
        )
        for path in reports
        if path.exists()
    )
    if not known:
        raise ValueError("Task ID does not match a recorded B parse failure")
    source = stage_input(settings, entry)
    directory = settings.input_dir / "__parsed__" / (source.name + ".docling_raw")
    directory.mkdir(parents=True, exist_ok=True)
    client = DoclingRawClient()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as http:
        await client._poll_until_done(http, task_id)
        await client._download_result_into(http, task_id, directory, source.name)
    select_main_json(directory, source)
    signature = compute_options_signature(
        tunable_env=snapshot_tunable_env({}), fixed_constants=FIXED_CONSTANTS
    )
    build_and_write_docling_manifest(
        directory,
        source_file_path=source,
        task_id=task_id,
        endpoint_signature=client.endpoint,
        engine_version=client.engine_version,
        options_signature=signature,
        fixed_constants=FIXED_CONSTANTS,
        recorded_filename=source.name,
    )
    report = {
        "document_id": document_id,
        "task_id": task_id,
        "reused_original_submission": True,
        "source_sha256": entry["sha256"],
        "raw_directory": str(directory),
        "state": "downloaded_and_verified",
    }
    save_json(settings.run_dir / "recoveries" / (document_id + ".json"), report)
    return report
