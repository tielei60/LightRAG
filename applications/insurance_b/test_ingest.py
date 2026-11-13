from dataclasses import replace
from types import SimpleNamespace

import pytest

from .common import save_json
from .ingest import accepted_readback, run_signature
from .prepare import copy_sources
from .test_prepare import corpus


def test_readback_cannot_authorize_a_different_run_contract(settings):
    save_json(settings.run_dir / "manifest.json", {"synthetic": True})
    save_json(
        settings.run_dir / "readback-report.json",
        {
            "passed": True,
            "workspace": settings.workspace,
            "run_contract_sha256": run_signature(settings),
        },
    )
    assert accepted_readback(settings)["passed"]
    changed = replace(settings, values={**settings.values, "B_TOP_K": "31"})
    with pytest.raises(ValueError, match="exact frozen"):
        accepted_readback(changed)


@pytest.mark.parametrize("other_incomplete", [False, True])
async def test_partial_ingest_restores_whole_workspace_and_refuses_other_pending(
    settings, monkeypatch, tmp_path, other_incomplete
):
    from . import ingest as module

    origin = tmp_path / "source-originals"
    corpus(origin)
    manifest = copy_sources(settings, origin)
    selected = manifest["documents"][0]
    existing = {
        row["document_id"]: SimpleNamespace(
            file_path=row["relative_path"], status="processed"
        )
        for row in manifest["documents"][1:]
    }
    if other_incomplete:
        existing[manifest["documents"][-1]["document_id"]].status = "pending"
    restored, enqueued = [], []

    class Adapter:
        def __init__(self, *args):
            pass

        def restore(self, document_id):
            restored.append(document_id)

        def begin_document(self, *args):
            pass

    class Status:
        async def get_docs_by_statuses(self, *args, **kwargs):
            return existing

        async def get_docs_by_track_id(self, *args):
            return {
                selected["document_id"]: SimpleNamespace(
                    status="processed",
                    metadata={"chunk_method": "paragraph_semantic"},
                    chunks_list=["synthetic-chunk"],
                )
            }

    class Rag:
        def __init__(self):
            self.doc_status = Status()
            self.addon_params = {}

        async def apipeline_enqueue_documents(self, *args, **kwargs):
            assert set(restored) == set(existing)
            enqueued.append(kwargs)
            return "synthetic-track"

        async def apipeline_process_enqueue_documents(self):
            pass

        async def finalize_storages(self):
            pass

    async def initialize(*args):
        pass

    monkeypatch.setattr(module, "make_rag", lambda *args, **kwargs: Rag())
    monkeypatch.setattr(module, "initialize_rag", initialize)
    monkeypatch.setattr(module, "provider", lambda *args: None)
    monkeypatch.setattr(module, "BExtractionAdapter", Adapter)
    if other_incomplete:
        with pytest.raises(ValueError, match="incomplete"):
            await module.ingest(settings, products={selected["product_directory"]})
        assert not enqueued
    else:
        result = await module.ingest(settings, products={selected["product_directory"]})
        assert len(restored) == 20
        assert len(enqueued) == 1
        assert result["processed_this_invocation"] == 3
        assert not result["full_corpus_passed"]
