"""Read synthetic persisted objects in a fresh process without model calls."""

import asyncio
import os
import sys
from dataclasses import replace

from .common import save_json
from .config import load
from .models import initialize_rag, make_rag


async def readback(workspace: str, run_id: str):
    if not workspace.startswith("insurance_b_smoke_") or not run_id.startswith(
        "smoke-"
    ):
        raise ValueError("This helper only reads synthetic smoke workspaces")
    base = load()
    settings = replace(
        base, values={**base.values, "B_WORKSPACE": workspace, "B_RUN_ID": run_id}
    )
    os.chdir(settings.root)
    rag = make_rag(settings, audited_query=False)
    initialized = False
    try:
        await initialize_rag(rag, settings)
        initialized = True
        from lightrag.base import DocStatus

        statuses = await rag.doc_status.get_docs_by_statuses(
            [DocStatus.PROCESSED], strict=True
        )
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        chunks = [
            await rag.text_chunks.get_by_id(key)
            for row in statuses.values()
            for key in row.chunks_list
        ]
        save_json(
            settings.run_dir / "fresh-process-readback.json",
            {
                "documents": len(statuses),
                "entities": len(nodes),
                "chunks": len(chunks),
                "all_chunks_present": all(row is not None for row in chunks),
            },
        )
    finally:
        if initialized:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(readback(*sys.argv[1:]))
