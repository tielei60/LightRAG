"""Regression checks for the preflight barrier and closed native readback."""

import json
from types import SimpleNamespace

import networkx as nx
import pytest

from product.evidence_import import (
    approve_extractions,
    digest,
    readback_files,
    validate_native_identities,
)


@pytest.mark.asyncio
async def test_preflight_validates_every_chunk_without_native_mutation(tmp_path):
    calls = []

    class Adapter:
        native_system_prompt = "native system"

        async def __call__(self, prompt, system_prompt):
            assert system_prompt == self.native_system_prompt
            calls.append(prompt)
            return json.dumps({"entities": [], "relationships": []})

    requests = {
        "first prompt": SimpleNamespace(source="first.md", content="first content"),
        "second prompt": SimpleNamespace(source="second.md", content="second content"),
    }
    result = await approve_extractions(Adapter(), requests, concurrency=2)
    assert set(calls) == set(requests)
    assert set(result) == {digest(prompt) for prompt in requests}
    assert result[digest("first prompt")]["content_sha256"] == digest("first content")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_invalid_preflight_has_one_correction_attempt_and_blocks():
    calls = 0

    class Adapter:
        native_system_prompt = "native system"

        async def __call__(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise ValueError("Unknown evidence id")

    with pytest.raises(ValueError, match="Unknown evidence"):
        await approve_extractions(
            Adapter(),
            {"prompt": SimpleNamespace(source="source.md", content="body")},
            concurrency=1,
        )
    assert calls == 2


def test_preflight_rejects_normalization_collision_and_truncation():
    def item(*names):
        return {"one": {"native": {"entities": [{"name": name} for name in names]}}}

    with pytest.raises(ValueError, match="collide"):
        validate_native_identities(item("产品（甲）", "产品(甲)"))
    with pytest.raises(ValueError, match="truncated"):
        validate_native_identities(item("甲" * 257))


def native_files(tmp_path):
    body = "产品甲\n完整条款：仅在条件乙成立时适用；丙除外。"
    entity_name = "产品甲｜规则"
    native = {
        "entities": [{"name": entity_name, "type": "合同规则", "description": body}],
        "relationships": [],
    }
    documents = [({"ingestion_label": "条款.md"}, {})]
    approved = {"prompt": {"content_sha256": digest(body), "native": native}}
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps({"doc-1": {"status": "processed", "file_path": "条款.md"}})
    )
    (tmp_path / "kv_store_text_chunks.json").write_text(
        json.dumps({"chunk-1": {"content": body}})
    )
    graph = nx.Graph()
    graph.add_node(entity_name, description=body, source_id="chunk-1")
    nx.write_graphml(graph, tmp_path / "graph_chunk_entity_relation.graphml")
    for name, rows in {
        "entities": [
            {"entity_name": entity_name, "content": entity_name + "\n" + body}
        ],
        "relationships": [],
        "chunks": [{"__id__": "chunk-1", "content": body}],
    }.items():
        (tmp_path / f"vdb_{name}.json").write_text(json.dumps({"data": rows}))
    return documents, approved


def test_closed_readback_verifies_native_chunks_graph_and_vectors(tmp_path):
    documents, approved = native_files(tmp_path)
    report = readback_files(tmp_path, documents, approved)
    assert report["technical_passed"]
    assert report["processed_documents"] == 1
    assert report["chunks"] == 1
    assert report["entities"] == 1
    assert report["business_acceptance"].startswith("pending")


def test_readback_refuses_missing_condition_in_entity_vector(tmp_path):
    documents, approved = native_files(tmp_path)
    vector_path = tmp_path / "vdb_entities.json"
    data = json.loads(vector_path.read_text())
    data["data"][0]["content"] = "产品甲｜规则\n仅在条件乙成立时适用。"
    vector_path.write_text(json.dumps(data))
    report = readback_files(tmp_path, documents, approved)
    assert not report["technical_passed"]
    assert any("Entity vector lost" in error for error in report["errors"])


def test_readback_refuses_unknown_graph_source_and_missing_chunk_vector(tmp_path):
    documents, approved = native_files(tmp_path)
    graph_path = tmp_path / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(graph_path)
    graph.nodes["产品甲｜规则"]["source_id"] = "lost-chunk"
    nx.write_graphml(graph, graph_path)
    (tmp_path / "vdb_chunks.json").write_text(json.dumps({"data": []}))
    report = readback_files(tmp_path, documents, approved)
    assert not report["technical_passed"]
    assert any("Chunk vectors" in error for error in report["errors"])
    assert any("provenance" in error for error in report["errors"])


def test_readback_accepts_evidence_reordered_by_original_source_position(tmp_path):
    documents, approved = native_files(tmp_path)
    primary = "赔付规则：仅在条件乙成立时适用；丙除外。"
    identity = "产品甲"
    parent = "共同前提：须选择对应责任。"
    approved["prompt"]["native"]["entities"][0]["description"] = (
        f"{primary}\n\n{identity}\n\n{parent}"
    )
    stored = f"{identity}\n\n{parent}\n\n{primary}"
    graph_path = tmp_path / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(graph_path)
    graph.nodes["产品甲｜规则"]["description"] = stored
    nx.write_graphml(graph, graph_path)
    vector_path = tmp_path / "vdb_entities.json"
    data = json.loads(vector_path.read_text())
    data["data"][0]["content"] = "产品甲｜规则\n" + stored
    vector_path.write_text(json.dumps(data))
    report = readback_files(tmp_path, documents, approved)
    assert report["technical_passed"], report["errors"]

    data["data"][0]["content"] = data["data"][0]["content"].replace("；丙除外", "")
    vector_path.write_text(json.dumps(data))
    report = readback_files(tmp_path, documents, approved)
    assert not report["technical_passed"]
    assert any("Entity vector lost" in error for error in report["errors"])


def approval_fixture(tmp_path):
    from product.evidence_extract import EvidenceChunk, EvidenceExtractor, EvidenceUnit
    from product.evidence_import import reuse_approved_selections

    previous = tmp_path / "previous"
    previous.mkdir()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    text = "产品甲\n完整条件：仅在乙成立时适用，丙除外。"
    source = source_root / "source.md"
    source.write_text(text)
    unit = EvidenceUnit("e1", source.name, 0, len(text), text)
    chunk = EvidenceChunk(source.name, "产品甲", text, (unit,))
    prompt = "exact native prompt"
    key = digest(prompt)
    payload = {"text": text, "chunking": {"params": {"split_by_character": "BOUNDARY"}}}
    catalog = [
        {
            "content_sha256": digest(text),
            "units": [{"id": "e1", "start": 0, "end": len(text)}],
        }
    ]
    for name, value in {"prepared.json": payload, "catalog.json": catalog}.items():
        (previous / name).write_text(json.dumps(value))
    (previous / "manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "documents": [
                    {
                        "product": chunk.product,
                        "relative_path": source.name,
                        "sha256": digest(text),
                        "prepared_path": "prepared.json",
                        "prepared_sha256": digest(
                            (previous / "prepared.json").read_text()
                        ),
                        "catalog_path": "catalog.json",
                        "catalog_sha256": digest(
                            (previous / "catalog.json").read_text()
                        ),
                    }
                ],
            }
        )
    )
    (previous / "approved-extractions.json").write_text(
        json.dumps(
            {
                "chunks": {
                    key: {
                        "source": source.name,
                        "content_sha256": digest(text),
                        "prompt_sha256": key,
                        "native": {"untrusted": True},
                    }
                }
            }
        )
    )
    selection = {
        "entities": [{"name": "规则", "type": "合同规则", "evidence_ids": ["e1"]}],
        "relationships": [],
    }
    (previous / "evidence-ledger.json").write_text(
        json.dumps(
            {
                "selections": {
                    key: {"selection": selection, "native": {"untrusted": True}}
                }
            }
        )
    )

    async def no_provider(*_args, **_kwargs):
        raise AssertionError("Approved reuse must not call the provider")

    options = {
        "requests": {prompt: chunk},
        "native_system_prompt": "native system",
        "native_summary_template": "{description_type}:{description_name}:{description_list}",
        "type_guidance": "",
        "selection_guidance": "",
        "complete": no_provider,
        "ledger_path": tmp_path / "new-ledger.json",
        "ledger_context": "new contract",
    }
    return previous, options, reuse_approved_selections, EvidenceExtractor


@pytest.mark.asyncio
async def test_checked_approval_reuse_regenerates_native_without_provider(tmp_path):
    previous, options, reuse, extractor = approval_fixture(tmp_path)
    adapter = extractor(**options)
    report = reuse(adapter, options["requests"], previous)
    assert report["revalidated_chunks"] == 1
    assert report["provider_called"] is False
    fresh = extractor(**options, replay_only=True)
    output = json.loads(
        await fresh("exact native prompt", system_prompt="native system")
    )
    assert output["entities"][0]["name"] == "产品甲｜规则"
    assert output["entities"][0]["description"].endswith("丙除外。")
    assert "untrusted" not in output


@pytest.mark.parametrize("change", ["content", "unit"])
def test_approval_reuse_refuses_changed_current_payload_before_persist(
    tmp_path, change
):
    from dataclasses import replace

    previous, options, reuse, extractor = approval_fixture(tmp_path)
    chunk = options["requests"]["exact native prompt"]
    if change == "content":
        chunk = replace(chunk, content=chunk.content + " changed")
    else:
        chunk = replace(
            chunk, units=(replace(chunk.units[0], identifier="different-id"),)
        )
    options["requests"] = {"exact native prompt": chunk}
    adapter = extractor(**options)
    with pytest.raises(ValueError, match="(exact current|unit identities)"):
        reuse(adapter, options["requests"], previous)
    assert adapter.selections == {}
    assert not options["ledger_path"].exists()


def test_readback_uses_native_basename_for_path_labels_and_refuses_wrong_name(tmp_path):
    documents, approved = native_files(tmp_path)
    documents[0][0]["ingestion_label"] = "docs/insurance_docs/产品甲/条款.md"
    report = readback_files(tmp_path, documents, approved)
    assert report["technical_passed"], report["errors"]

    statuses = tmp_path / "kv_store_doc_status.json"
    statuses.write_text(
        json.dumps({"doc-1": {"status": "processed", "file_path": "另一份条款.md"}})
    )
    report = readback_files(tmp_path, documents, approved)
    assert not report["technical_passed"]
    assert any("document identities" in error for error in report["errors"])


def test_readback_refuses_colliding_native_source_labels(tmp_path):
    documents, approved = native_files(tmp_path)
    documents[0][0]["ingestion_label"] = "产品甲/条款.md"
    documents.append(({"ingestion_label": "产品乙/条款.md"}, {}))
    report = readback_files(tmp_path, documents, approved)
    assert not report["technical_passed"]
    assert any("collide" in error for error in report["errors"])


def configured_profile():
    from product.extraction_profile import ExtractionProfile

    return ExtractionProfile.from_mapping(
        {
            "entity_types": ["保险产品", "合同规则"],
            "entity_type_aliases": {"Product": "保险产品", "Rule": "合同规则"},
            "product_entity_type": "保险产品",
            "unscoped_entity_types": [],
            "relationship_types": ["适用规则"],
            "relationship_type_aliases": {"applies": "适用规则"},
        }
    )


@pytest.mark.parametrize("invalid", ["entity_type", "relationship_keywords", "role"])
def test_approval_reuse_enforces_current_profile_before_persist(tmp_path, invalid):
    from dataclasses import replace

    previous, options, reuse, extractor = approval_fixture(tmp_path)
    options["profile"] = configured_profile()
    ledger_path = previous / "evidence-ledger.json"
    old = json.loads(ledger_path.read_text())
    row = next(iter(old["selections"].values()))
    if invalid == "entity_type":
        row["selection"]["entities"][0]["type"] = "不在当前配置的类型"
    elif invalid == "relationship_keywords":
        row["selection"]["entities"].append(
            {"name": "规则二", "type": "合同规则", "evidence_ids": ["e1"]}
        )
        row["selection"]["relationships"] = [
            {
                "source": "规则",
                "target": "规则二",
                "keywords": "不在当前配置的关系",
                "evidence_ids": ["e1"],
            }
        ]
    else:
        prompt, chunk = next(iter(options["requests"].items()))
        options["requests"] = {
            prompt: replace(
                chunk, units=(replace(chunk.units[0], role="structural-parent"),)
            )
        }
    ledger_path.write_text(json.dumps(old))
    adapter = extractor(**options)
    with pytest.raises(ValueError, match="(Unsupported evidence|unit identities)"):
        reuse(adapter, options["requests"], previous)
    assert adapter.selections == {}
    assert not options["ledger_path"].exists()


@pytest.mark.asyncio
async def test_profile_approval_reuse_maps_registered_aliases_and_ignores_old_native(
    tmp_path,
):
    previous, options, reuse, extractor = approval_fixture(tmp_path)
    options["profile"] = configured_profile()
    ledger_path = previous / "evidence-ledger.json"
    old = json.loads(ledger_path.read_text())
    row = next(iter(old["selections"].values()))
    row["selection"] = {
        "entities": [
            {"name": "规则", "type": "Rule", "evidence_ids": ["e1"]},
            {"name": "规则二", "type": "合同规则", "evidence_ids": ["e1"]},
        ],
        "relationships": [
            {
                "source": "规则",
                "target": "规则二",
                "keywords": "applies",
                "evidence_ids": ["e1"],
            }
        ],
    }
    ledger_path.write_text(json.dumps(old))
    adapter = extractor(**options)
    reuse(adapter, options["requests"], previous)
    fresh = extractor(**options, replay_only=True)
    output = json.loads(
        await fresh("exact native prompt", system_prompt="native system")
    )
    assert {item["type"] for item in output["entities"]} == {"合同规则"}
    assert output["relationships"][0]["keywords"] == "适用规则"
    assert "untrusted" not in output


@pytest.mark.asyncio
async def test_preview_rejects_invalid_profile_before_models_or_native_storage(
    tmp_path, monkeypatch
):
    import lightrag
    import lightrag.prompt
    from product import manage
    from product.evidence_import import apply_batch

    profile_dir = tmp_path / "profiles" / "entity_type"
    profile_dir.mkdir(parents=True)
    (profile_dir / "invalid.yml").write_text("entity_types: [123]\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"workspace": "preview"}))
    config = {
        "WORKSPACE": "preview",
        "MAX_GLEANING": "0",
        "WORKING_DIR": str(tmp_path / "rag_storage"),
        "PROMPT_DIR": str(profile_dir.parent),
        "ENTITY_TYPE_PROMPT_FILE": "invalid.yml",
        "SUMMARY_LANGUAGE": "Chinese",
        "MAX_EXTRACTION_RECORDS": "10",
        "MAX_EXTRACTION_ENTITIES": "5",
    }
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "MANIFEST", manifest)
    monkeypatch.setattr(manage, "settings", lambda: config)
    monkeypatch.setattr(manage, "checked_files", list)
    for key, value in config.items():
        monkeypatch.setenv(key, value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preview must not initialize native storage")

    monkeypatch.setattr(lightrag, "LightRAG", forbidden)
    native_profile = {
        "entity_types_guidance": "guidance",
        "entity_extraction_json_examples": ["{}"],
    }
    monkeypatch.setattr(
        lightrag.prompt,
        "resolve_entity_extraction_prompt_profile",
        lambda *_args: native_profile,
    )
    monkeypatch.setattr(
        lightrag.prompt,
        "validate_entity_extraction_prompt_profile_for_mode",
        lambda *_args: native_profile,
    )
    with pytest.raises(ValueError, match="entity_types"):
        await apply_batch(False)
    assert not (tmp_path / "rag_storage").exists()


@pytest.mark.asyncio
async def test_incremental_shared_organization_merge_restores_prior_selected_evidence():
    from product.evidence_extract import (
        EvidenceChunk,
        EvidenceUnit,
        PointerResult,
        to_native,
    )
    from product.evidence_import import incremental_replay_adapter

    def evidence(product, filename, text):
        unit = EvidenceUnit("e1", filename, 0, len(text), text)
        chunk = EvidenceChunk(filename, product, text, (unit,))
        selection = PointerResult.model_validate(
            {
                "entities": [
                    {"name": "共同保险机构", "type": "机构", "evidence_ids": ["e1"]}
                ],
                "relationships": [],
            }
        )
        return chunk, {
            "selection": selection.model_dump(),
            "native": to_native(selection, chunk),
        }

    old, old_row = evidence("产品甲", "旧原文.md", "产品甲由共同保险机构承保。")
    new, new_row = evidence("产品乙", "新原文.md", "产品乙由共同保险机构承保。")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Existing products and merge evidence must not call a model"
        )

    options = {
        "requests": {"new prompt": new},
        "native_system_prompt": "native system",
        "native_summary_template": "{description_type}:{description_name}:{description_list}",
        "type_guidance": "",
        "selection_guidance": "",
        "complete": forbidden,
    }
    adapter = incremental_replay_adapter(
        options,
        {"old prompt": old},
        {digest("old prompt"): old_row},
        {digest("new prompt"): new_row},
    )
    adapter.activate_sources([new.source])
    result = await adapter(
        "Entity:共同保险机构:"
        + "\n".join(
            json.dumps({"Description": chunk.units[0].text}, ensure_ascii=False)
            for chunk in [old, new]
        )
    )
    assert old.units[0].text in result
    assert new.units[0].text in result
    assert len(adapter.expected[("entity", "共同保险机构")]) == 2


@pytest.mark.asyncio
async def test_append_base_argument_cannot_silently_enable_overwriting():
    from product.evidence_import import apply_batch

    with pytest.raises(ValueError, match="explicit --append"):
        await apply_batch(False, base_batch="existing")


def test_multiple_approval_reuse_validates_exact_union_before_persisting(
    tmp_path, monkeypatch
):
    import product.workspace_registry as registry
    from product.evidence_extract import EvidenceChunk, EvidenceUnit
    from product.evidence_import import reuse_approved_batches

    requests, reviewed = {}, {}
    for name, product in [("first", "产品甲"), ("second", "产品乙")]:
        text = product + "：仅在完整条件下适用。"
        unit = EvidenceUnit("e1", name + ".md", 0, len(text), text)
        chunk = EvidenceChunk(name + ".md", product, text, (unit,))
        prompt = name + " prompt"
        requests[prompt] = chunk
        reviewed[name] = SimpleNamespace(
            requests={prompt: chunk},
            selections={
                digest(prompt): {
                    "selection": {
                        "entities": [
                            {"name": "规则", "type": "合同规则", "evidence_ids": ["e1"]}
                        ],
                        "relationships": [],
                    }
                }
            },
        )
    monkeypatch.setattr(
        registry, "load_reviewed_batch", lambda _root, name: reviewed[name]
    )
    persisted = []
    adapter = SimpleNamespace(
        profile=None, persist_selection=lambda *args: persisted.append(args)
    )
    with pytest.raises(ValueError, match="union differs"):
        reuse_approved_batches(adapter, requests, tmp_path, ["first"])
    assert persisted == []
    report = reuse_approved_batches(adapter, requests, tmp_path, ["first", "second"])
    assert report["revalidated_chunks"] == 2 and report["provider_called"] is False
    assert len(persisted) == 2
    assert all(
        "完整条件" in native["entities"][0]["description"] for _, _, native in persisted
    )


def test_multiple_approval_reuse_refuses_changed_source_identity(tmp_path, monkeypatch):
    import product.workspace_registry as registry
    from product.evidence_extract import EvidenceChunk, EvidenceUnit
    from product.evidence_import import reuse_approved_batches

    text = "产品甲：完整条件。"
    old = EvidenceChunk(
        "old.md", "产品甲", text, (EvidenceUnit("e1", "old.md", 0, len(text), text),)
    )
    new = EvidenceChunk(
        "new.md", "产品甲", text, (EvidenceUnit("e1", "new.md", 0, len(text), text),)
    )
    monkeypatch.setattr(
        registry,
        "load_reviewed_batch",
        lambda *_: SimpleNamespace(
            requests={"prompt": old}, selections={digest("prompt"): {}}
        ),
    )
    adapter = SimpleNamespace(
        profile=None, persist_selection=lambda *_: pytest.fail("must not persist")
    )
    with pytest.raises(ValueError, match="identity differs"):
        reuse_approved_batches(adapter, {"prompt": new}, tmp_path, ["base"])
