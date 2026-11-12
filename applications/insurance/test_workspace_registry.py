"""Incremental workspace safety checks use local native-file fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from applications.insurance.workspace_registry import (
    assert_workspace_stopped,
    backup_workspace,
    batch_record,
    embedding_contract,
    load_registered_batches,
    registry_path,
    snapshot_for_batches,
    validate_additions,
    verify_preserved,
    workspace_writer,
    write_json,
)


def document(product="产品甲", label="甲.md", text="甲条款"):
    return (
        {"product": product, "ingestion_label": label},
        {"text": text, "chunking": {"params": {"split_by_character": "|BOUNDARY|"}}},
    )


def test_additions_accept_distinct_product_but_reject_duplicate_identity():
    old = [document()]
    validate_additions(old, [document("产品乙", "乙.md", "乙条款")])
    for added in [
        document("产品甲", "乙.md", "乙条款"),
        document("产品乙", "目录/甲.md", "乙条款"),
        document("产品乙", "乙.md", "甲条款"),
        document("产品乙", "乙.md", "新增|BOUNDARY|甲条款"),
    ]:
        with pytest.raises(ValueError, match="(already registered|collides)"):
            validate_additions(old, [added])


def test_additions_reject_collisions_within_the_new_batch():
    with pytest.raises(ValueError, match="document collides"):
        validate_additions([], [document(), document("产品乙", "甲.md", "乙条款")])


def test_embedding_contract_keeps_model_and_dimensions():
    original = {
        "EMBEDDING_MODEL": "model-a",
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_BINDING": "openai",
    }
    assert embedding_contract(original) != embedding_contract(
        {**original, "EMBEDDING_MODEL": "model-b"}
    )
    assert embedding_contract(original) != embedding_contract(
        {**original, "EMBEDDING_DIM": "2048"}
    )
    with pytest.raises(ValueError, match="incomplete"):
        embedding_contract({"EMBEDDING_MODEL": "", "EMBEDDING_DIM": "1024"})


def test_unknown_existing_workspace_requires_an_explicit_reviewed_base(tmp_path):
    with pytest.raises(ValueError, match="explicit --base-batch"):
        load_registered_batches(tmp_path, "existing")
    assert not registry_path(tmp_path, "existing").exists()


def batch_fixture(root: Path, name: str, product: str, text: str):
    directory = root / "output" / name
    directory.mkdir(parents=True)
    payload = {"text": text}
    prepared = directory / "prepared.json"
    write_json(prepared, payload)
    from applications.insurance.workspace_registry import sha256

    manifest = {
        "workspace": "existing",
        "source_root": str(root / "sources"),
        "documents": [
            {
                "product": product,
                "prepared_path": "prepared.json",
                "prepared_sha256": sha256(prepared),
                "ingestion_label": product + ".md",
            }
        ],
    }
    for filename, value in {
        "manifest.json": manifest,
        "approved-extractions.json": {"chunks": {}},
        "evidence-ledger.json": {"selections": {}},
    }.items():
        write_json(directory / filename, value)
    return SimpleNamespace(
        name=name,
        path=directory,
        manifest=manifest,
        config={
            "WORKSPACE": "existing",
            "EMBEDDING_MODEL": "model-a",
            "EMBEDDING_DIM": "1024",
            "EMBEDDING_BINDING": "openai",
        },
    )


def test_registry_detects_changed_prior_evidence_before_another_append(
    tmp_path, monkeypatch
):
    import applications.insurance.workspace_registry as module

    base = batch_fixture(tmp_path, "base", "甲", "原文")
    registry = {
        "version": 1,
        "workspace": "existing",
        "batches": [batch_record(base)],
        "embedding": embedding_contract(base.config),
    }
    write_json(registry_path(tmp_path, "existing"), registry)
    monkeypatch.setattr(module, "load_reviewed_batch", lambda *_: base)
    assert load_registered_batches(tmp_path, "existing")[1] == [base]
    write_json(base.path / "evidence-ledger.json", {"selections": {"changed": {}}})
    with pytest.raises(ValueError, match="was changed"):
        load_registered_batches(tmp_path, "existing")


def test_cumulative_snapshot_keeps_each_original_batch_and_all_products(tmp_path):
    first = batch_fixture(tmp_path, "base", "甲", "甲原文")
    second = batch_fixture(tmp_path, "addition", "乙", "乙原文")
    before = (first.path / "manifest.json").read_bytes()
    registry = {
        "workspace": "existing",
        "batches": [batch_record(first), batch_record(second)],
    }
    snapshot = snapshot_for_batches(tmp_path, "existing", [first, second], registry)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["expected_product_count"] == 2
    assert len(manifest["documents"]) == 2
    assert {
        json.loads((snapshot / entry["prepared_path"]).read_text())["text"]
        for entry in manifest["documents"]
    } == {"甲原文", "乙原文"}
    assert (first.path / "manifest.json").read_bytes() == before


def test_workspace_writer_rejects_overlapping_writer_and_releases_lock(tmp_path):
    with (
        workspace_writer(tmp_path, "existing"),
        pytest.raises(ValueError, match="reservation already exists"),
        workspace_writer(tmp_path, "existing"),
    ):
        pass
    assert not (registry_path(tmp_path, "existing").parent / "writer.lock").exists()


def running_server_fixture(
    tmp_path, monkeypatch, command, *, cwd=None, workspace="master"
):
    import psutil

    batch = tmp_path / "output" / "controller"
    batch.mkdir(parents=True)
    (batch / "runtime.env").write_text(f"WORKSPACE={workspace}\n")
    (batch / "server.pid").write_text("1234")
    process = SimpleNamespace(
        is_running=lambda: True,
        cwd=lambda: str(cwd or tmp_path),
        cmdline=lambda: command,
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "applications.insurance.server"],
        ["python", "/repo/.venv/bin/lightrag-server"],
    ],
)
def test_workspace_writer_refuses_current_and_native_services(
    tmp_path, monkeypatch, command
):
    running_server_fixture(tmp_path, monkeypatch, command)
    with (
        pytest.raises(ValueError, match="Stop every server"),
        workspace_writer(tmp_path, "master"),
    ):
        pytest.fail("An active workspace must not grant the writer reservation")
    assert not (registry_path(tmp_path, "master").parent / "writer.lock").exists()


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "product.server"],
        ["python", "-m", "product.server_extra"],
        ["python", "-m", "applications.insurance.server_extra"],
        ["python", "applications.insurance.server"],
        ["python", "some-lightrag-server"],
    ],
)
def test_workspace_guard_ignores_lookalike_process_arguments(
    tmp_path, monkeypatch, command
):
    running_server_fixture(tmp_path, monkeypatch, command)
    assert_workspace_stopped(tmp_path, "master")


@pytest.mark.parametrize("same_workspace", [True, False])
def test_workspace_guard_keeps_repository_and_workspace_scope(
    tmp_path, monkeypatch, same_workspace
):
    running_server_fixture(
        tmp_path,
        monkeypatch,
        ["python", "-m", "applications.insurance.server"],
        cwd=tmp_path / "another-repository" if same_workspace else tmp_path,
        workspace="master" if same_workspace else "isolated-test",
    )
    assert_workspace_stopped(tmp_path, "master")


def test_backup_copies_all_files_and_refuses_overwrite(tmp_path):
    target = tmp_path / "native"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "data.json").write_text('{"原文": 1}')
    batch = tmp_path / "batch"
    backup = backup_workspace(target, batch)
    assert (backup / "sub" / "data.json").read_bytes() == (
        target / "sub" / "data.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="prior append backup"):
        backup_workspace(target, batch)


def test_backup_refuses_symlink_in_native_storage(tmp_path):
    target = tmp_path / "native"
    target.mkdir()
    (target / "linked").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symbolic links"):
        backup_workspace(target, tmp_path / "batch")


def preserved_fixture(directory: Path):
    directory.mkdir()
    for namespace in (
        "doc_status",
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
    ):
        write_json(
            directory / f"kv_store_{namespace}.json", {"old": {"text": "原有完整证据"}}
        )
    write_json(
        directory / "kv_store_entity_chunks.json",
        {"共同机构": {"chunk_ids": ["old-chunk"]}},
    )
    write_json(directory / "kv_store_relation_chunks.json", {})
    graph = nx.Graph()
    graph.add_node("共同机构", source_id="old-chunk", description="旧机构证据")
    nx.write_graphml(graph, directory / "graph_chunk_entity_relation.graphml")


def test_preservation_allows_shared_organization_merge_without_losing_old_carriers(
    tmp_path,
):
    old, target = tmp_path / "old", tmp_path / "target"
    preserved_fixture(old)
    preserved_fixture(target)
    graph = nx.read_graphml(target / "graph_chunk_entity_relation.graphml")
    graph.nodes["共同机构"].update(
        description="旧机构证据\n新机构证据", source_id="new-chunk"
    )
    nx.write_graphml(graph, target / "graph_chunk_entity_relation.graphml")
    # Authoritative tracking can retain evidence beyond the capped graph view.
    write_json(
        target / "kv_store_entity_chunks.json",
        {"共同机构": {"chunk_ids": ["old-chunk", "new-chunk"]}},
    )
    assert verify_preserved(old, target)["passed"]
    write_json(target / "kv_store_full_entities.json", {})
    report = verify_preserved(old, target)
    assert not report["passed"]
    assert "old_full_entities_changed:old" in report["issues"]


def test_preservation_rejects_lost_authoritative_attribution(tmp_path):
    old, target = tmp_path / "old", tmp_path / "target"
    preserved_fixture(old)
    preserved_fixture(target)
    write_json(
        target / "kv_store_entity_chunks.json",
        {"共同机构": {"chunk_ids": ["new-chunk"]}},
    )
    assert (
        "old_entity_chunks_attribution_lost:共同机构"
        in verify_preserved(old, target)["issues"]
    )


def test_registered_workspace_refuses_a_changed_storage_backend(tmp_path, monkeypatch):
    import applications.insurance.workspace_registry as module
    from applications.insurance.database_storage import DATABASE_STORES

    base = batch_fixture(tmp_path, "base", "甲", "原文")
    registry = {
        "version": 1,
        "workspace": "existing",
        "batches": [batch_record(base)],
        "embedding": embedding_contract(base.config),
    }
    write_json(registry_path(tmp_path, "existing"), registry)
    base.config.update(
        {"LIGHTRAG_" + key.upper(): value for key, value in DATABASE_STORES.items()}
    )
    base.config.update(
        POSTGRES_DATABASE="products",
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_DATABASE="neo4j",
    )
    monkeypatch.setattr(module, "load_reviewed_batch", lambda *_: base)
    with pytest.raises(ValueError, match="storage backend"):
        load_registered_batches(tmp_path, "existing")


def test_initial_database_registry_publishes_backend_and_keeps_batch_immutable(
    tmp_path, monkeypatch
):
    import applications.insurance.workspace_registry as module
    from applications.insurance.database_storage import DATABASE_STORES

    base = batch_fixture(tmp_path, "database-base", "甲", "原文")
    before = (base.path / "manifest.json").read_bytes()
    base.config.update(
        {"LIGHTRAG_" + key.upper(): value for key, value in DATABASE_STORES.items()}
    )
    base.config.update(
        POSTGRES_DATABASE="products",
        POSTGRES_PORT="5433",
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_DATABASE="neo4j",
    )
    monkeypatch.setattr(module, "load_reviewed_batch", lambda *_: base)
    module.commit_initial_database(tmp_path, "existing", "database-base")
    saved = json.loads(registry_path(tmp_path, "existing").read_text())
    assert saved["storage"]["kv_storage"] == "PGKVStorage"
    assert saved["storage"]["postgres"]["port"] == 5433
    assert saved["batches"] == [batch_record(base)]
    assert (base.path / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="already exists"):
        module.commit_initial_database(tmp_path, "existing", "database-base")
