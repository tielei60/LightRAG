"""Command-line operations confined to the B application data."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .common import workspace_lock
from .config import PROJECT_ROOT, ROOT, load


def parser():
    result = argparse.ArgumentParser(prog="insurance-b")
    result.add_argument("--config", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("storage-start")
    commands.add_parser("storage-stop")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path)
    prepare.add_argument("--parse", action="store_true")
    prepare.add_argument("--products", nargs="*")
    recover = commands.add_parser("recover-parse")
    recover.add_argument("--document-id", required=True)
    recover.add_argument("--task-id", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--products", nargs="*")
    retry = commands.add_parser("retry")
    retry.add_argument("--document-id", required=True)
    commands.add_parser("verify")
    commands.add_parser("serve")
    commands.add_parser("prepare-a")
    commands.add_parser("ingest-a")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--repeats", type=int, choices=[1, 3], default=3)
    compare = commands.add_parser("compare")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--review", type=Path, required=True)
    commands.add_parser("smoke")
    return result


async def execute(args, settings):
    if args.command in {"prepare-a", "ingest-a"}:
        from .a_baseline import ingest_a, snapshot_a_inputs

        with workspace_lock(settings.root, settings.workspace, exclusive=True):
            return (
                await ingest_a(settings)
                if args.command == "ingest-a"
                else snapshot_a_inputs(settings)
            )
    if args.command in {"storage-start", "storage-stop", "status"}:
        from .storage import listening, start, stop

        if args.command == "storage-start":
            return await start(settings)
        if args.command == "storage-stop":
            stop(settings)
            return {"stopped": "B database processes only"}
        return {
            "project": str(PROJECT_ROOT),
            "application": str(settings.root),
            "workspace": settings.workspace,
            "run_id": settings.run_id,
            "model_credentials_configured": bool(settings.model_key()),
            "postgres_15433": listening(15433),
            "neo4j_17687": listening(17687),
            "http_9625": listening(9625),
        }
    if args.command == "prepare":
        from .prepare import checked_manifest, copy_sources, parse_all

        with workspace_lock(settings.root, settings.workspace, exclusive=True):
            manifest = (
                copy_sources(settings, args.source_root)
                if args.source_root
                else checked_manifest(settings)
            )
            if args.parse:
                return await parse_all(
                    settings, products=set(args.products) if args.products else None
                )
            return {
                "products": manifest["product_count"],
                "documents": manifest["document_count"],
                "manifest": str(settings.run_dir / "manifest.json"),
            }
    if args.command == "recover-parse":
        from .prepare import recover_docling_task

        with workspace_lock(settings.root, settings.workspace, exclusive=True):
            return await recover_docling_task(settings, args.document_id, args.task_id)
    if args.command == "ingest":
        from .ingest import ingest

        with workspace_lock(settings.root, settings.workspace, exclusive=True):
            return await ingest(
                settings, products=set(args.products) if args.products else None
            )
    if args.command == "retry":
        from .retry import retry

        with workspace_lock(settings.root, settings.workspace, exclusive=True):
            return await retry(settings, args.document_id)
    if args.command == "verify":
        from .verify import verify

        with workspace_lock(settings.root, settings.workspace):
            return await verify(settings)
    if args.command == "evaluate":
        from .evaluate import evaluate

        with workspace_lock(settings.root, settings.workspace):
            return await evaluate(settings, repeats=args.repeats)
    if args.command == "compare":
        from .evaluate import compare

        return compare(settings, args.a, args.b)
    if args.command == "score":
        from .evaluate import score

        return score(settings, args.results, args.review)
    if args.command == "smoke":
        from .smoke import smoke

        return await smoke(settings)
    raise ValueError("Unsupported command")


def resolve_cli_paths(args):
    """Resolve user paths before entering B's cwd for native SDK configuration."""
    for name in ["config", "source_root", "a", "b", "results", "review"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def main():
    args = resolve_cli_paths(parser().parse_args())
    # Native optional backends read config.ini relative to cwd. Use only B's cwd.
    os.chdir(ROOT)
    settings = load(args.config)
    settings.activate()
    if args.command == "serve":
        from .server import serve

        serve(settings)
        return
    try:
        value = asyncio.run(execute(args, settings))
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if isinstance(value, dict) and value.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
