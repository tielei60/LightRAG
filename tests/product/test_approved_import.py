"""Real offline native-pipeline smoke; no provider calls or answer claims."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from product.approved_import import (
    ApprovalError,
    ApprovedReplay,
    ImportConfig,
    build_review_packet,
    digest,
    import_candidate,
    offline_embedding,
    readback_candidate,
    seal_review,
    validate_seal,
)
from product.import_units import inspect_source, prepare_import

pytestmark = pytest.mark.offline


def fixture_packet(tmp_path, *, count=2, document_versions=None):
    """Two original locations share text but retain distinct source identities."""
    quote = (
        "安心保险。住院医疗保险责任。因故意或者因重大过失未履行如实告知义务，"
        "足以影响保险人决定是否同意承保或者提高保险费率的，保险人有权解除合同。"
        "通知例外为已经知道或者应当知道；通知例外不当然证明应予赔付。"
        "等待期内发生疾病不等同于等待期内住院。医疗费用扣除免赔额，"
        "住院日额津贴不适用医疗费用免赔额。"
    )
    docs = []
    for index in range(count):
        path = tmp_path / f"terms-{index}.txt"
        path.write_text(quote)
        source = inspect_source(
            path,
            f"source-{index}",
            document_versions[index] if document_versions else "document-v1",
        )
        span = {"start": 0, "end": len(quote), "quote": quote}
        scope = {
            "combinations": [
                {
                    "plan": {"kind": "not_applicable"},
                    "responsibility": {"kind": "specific", "value": "住院医疗保险责任"},
                    "evidence": [{"start": 0, "end": 14, "quote": quote[:14]}],
                }
            ]
        }
        docs.append(
            {
                k: source[k]
                for k in (
                    "source_id",
                    "document_version",
                    "original_sha256",
                    "extracted_text_sha256",
                    "parser_binding",
                )
            }
            | {
                "path": path.name,
                "product": {"id": "p1", "label": "安心保险", "identity_span": span},
                "units": [
                    {
                        "unit_id": f"rule-{index}",
                        "span": span,
                        "disposition": "primary",
                        "scope": scope,
                        "review": {
                            "complete_conditions": True,
                            "dependencies_complete": True,
                        },
                        "dependencies": [],
                    }
                ],
            }
        )
    prepared = prepare_import(
        {
            "schema_version": 1,
            "tokenizer_model": "gpt-4o-mini",
            "max_chunk_tokens": 8192,
            "documents": docs,
        },
        tmp_path,
    )
    selections = {}
    for doc in prepared["documents"]:
        unit = doc["units"][0]
        common = {
            "scope": unit["scope"],
            "evidence_ids": [
                s["evidence_id"] for s in unit["primary"] + unit["context"]
            ],
        }
        selections[unit["unit_id"]] = {
            "entities": [
                {
                    "entity_id": "coverage-1",
                    "name": "住院医疗保险责任",
                    "type": "保险责任",
                    **common,
                },
                {
                    "entity_id": "rule-1",
                    "name": "告知与等待期条件",
                    "type": "规则",
                    **common,
                },
            ],
            "relationships": [
                {
                    "relation_id": "coverage-rule-1",
                    "source": "coverage-1",
                    "target": "rule-1",
                    "keywords": "适用条件",
                    **common,
                }
            ],
        }
    return build_review_packet(prepared, selections)


def fixture_sealed(tmp_path, **kwargs):
    packet = fixture_packet(tmp_path, **kwargs)
    pin = digest(packet)
    approval = {
        "packet_sha256": pin,
        "decision": "approved",
        "reviewer": "offline-test-reviewer",
        "selection_author": "offline-test-author",
        "scope_reviewed": True,
        "evidence_reviewed": True,
        "reviewed_unit_ids": list(packet["selections"]),
        "purpose": "synthetic fixture only; not production approval",
    }
    return seal_review(packet, approval, expected_packet_sha256=pin)


def test_approval_and_config_drift_fail_closed(tmp_path):
    sealed = fixture_sealed(tmp_path)
    validate_seal(sealed, expected_seal_sha256=sealed["seal_sha256"])
    bad = copy.deepcopy(sealed)
    bad["packet"]["runtime"]["config"]["summary_context_size"] += 1
    with pytest.raises(ApprovalError, match="Seal pin"):
        validate_seal(bad, expected_seal_sha256=sealed["seal_sha256"])
    bad = copy.deepcopy(sealed["approval"])
    bad["reviewed_unit_ids"] = []
    with pytest.raises(ApprovalError, match="Incomplete"):
        seal_review(
            sealed["packet"], bad, expected_packet_sha256=sealed["packet_sha256"]
        )
    bad["reviewed_unit_ids"] = list(sealed["packet"]["selections"])
    bad["scope_reviewed"] = False
    with pytest.raises(ApprovalError, match="independent"):
        seal_review(
            sealed["packet"], bad, expected_packet_sha256=sealed["packet_sha256"]
        )


def test_scope_and_normalization_rejected(tmp_path):
    packet = fixture_packet(tmp_path)
    for change, pattern in [("scope", "scope"), ("name", "normalization")]:
        selections = copy.deepcopy(packet["selections"])
        entity = next(iter(selections.values()))["entities"][0]
        entity[change] = {"combinations": []} if change == "scope" else "Ａ责任"
        with pytest.raises(ApprovalError, match=pattern):
            build_review_packet(packet["prepared"], selections)


def test_distinct_locations_same_quote_not_deduplicated(tmp_path):
    packet = fixture_packet(tmp_path)
    assert len(packet["evidence"]) == 2
    assert len({x["quote"] for x in packet["evidence"].values()}) == 1
    assert len(packet["objects"]) == 3
    assert all(len(obj["contributions"]) == 2 for obj in packet["objects"].values())


@pytest.mark.asyncio
async def test_replay_unknown_prompt_and_cache_options_refused(tmp_path):
    packet = fixture_packet(tmp_path)
    replay = ApprovedReplay(packet)
    prompt, request = next(iter(packet["requests"].items()))
    result = await replay(
        prompt,
        system_prompt=packet["runtime"]["system_prompt"],
        response_format={"type": "json_object"},
    )
    assert json.loads(result) == request["native"]
    with pytest.raises(ApprovalError, match="Unknown callback"):
        await replay("unapproved")
    with pytest.raises(ApprovalError, match="options"):
        await replay(prompt, temperature=0)
    with pytest.raises(ApprovalError, match="Gleaning"):
        await replay(prompt, history_messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_real_native_import_restart_readback_and_query(tmp_path):
    sealed = fixture_sealed(tmp_path)
    target = tmp_path / "candidate"
    receipt = await import_candidate(
        sealed,
        target,
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
    )
    assert receipt["state"] == "completed"
    assert receipt["readback"]["chunks"] == 2
    assert receipt["readback"]["entities"] == 2
    assert receipt["readback"]["relationships"] == 1
    events = [
        json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["stage"] == "merge" for e in events)
    with pytest.raises(ApprovalError, match="Candidate exists"):
        await import_candidate(
            sealed,
            target,
            expected_seal_sha256=sealed["seal_sha256"],
            embedding_func=offline_embedding(),
        )
    # A separate OS process proves persisted stores/caches, not shared RAM.
    env = dict(
        os.environ,
        PYTHON_DOTENV_DISABLED="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "product.approved_import",
            "readback",
            str(target / "sealed.json"),
            str(target),
            "--pin",
            sealed["seal_sha256"],
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((target / "readback-report.json").read_text())
    assert report["answer_correctness_tested"] is False
    assert report["production_approved"] is False
    for key in ("only_need_context", "only_need_prompt"):
        assert "足以影响保险人" in report[key]
        assert "不当然证明应予赔付" in report[key]
        assert "等待期内发生疾病不等同于等待期内住院" in report[key]
        assert "住院日额津贴不适用医疗费用免赔额" in report[key]


def approve_packet(packet):
    pin = digest(packet)
    return seal_review(
        packet,
        {
            "packet_sha256": pin,
            "decision": "approved",
            "reviewer": "fictional-fixture-reviewer",
            "selection_author": "fictional-fixture-author",
            "scope_reviewed": True,
            "evidence_reviewed": True,
            "reviewed_unit_ids": list(packet["selections"]),
        },
        expected_packet_sha256=pin,
    )


def test_budget_incomplete_selection_alias_and_source_label_refusal(tmp_path):
    packet = fixture_packet(tmp_path)
    selections = copy.deepcopy(packet["selections"])
    selections["rule-0"]["entities"][0]["evidence_ids"] = []
    with pytest.raises(ApprovalError, match="evidence"):
        build_review_packet(packet["prepared"], selections)
    selections = copy.deepcopy(packet["selections"])
    selections["rule-1"]["entities"][0]["name"] = "同一身份的不同名称"
    with pytest.raises(ApprovalError, match="conflicting labels"):
        build_review_packet(packet["prepared"], selections)
    with pytest.raises(ApprovalError, match="provenance cap"):
        build_review_packet(
            packet["prepared"], packet["selections"], ImportConfig(max_sources=1)
        )
    with pytest.raises(ApprovalError, match="serialized-description budget"):
        build_review_packet(
            packet["prepared"],
            packet["selections"],
            ImportConfig(summary_context_size=20),
        )
    changed = copy.deepcopy(packet["prepared"])
    changed["documents"][1]["payload"]["file_source"] = "other/terms-0.txt"
    changed["prepared_sha256"] = digest(
        {k: v for k, v in changed.items() if k != "prepared_sha256"}
    )
    with pytest.raises(ApprovalError, match="source label collision"):
        build_review_packet(changed, packet["selections"])


@pytest.mark.asyncio
async def test_full_dependency_fixture_and_explicit_text_only(tmp_path):
    from tests.product.test_import_units import synthetic_manifest

    prepared = prepare_import(synthetic_manifest(tmp_path), tmp_path)
    selections = {}
    for document in prepared["documents"]:
        for unit in document["units"]:
            selections[unit["unit_id"]] = {
                "entities": [
                    {
                        "entity_id": unit["unit_id"],
                        "name": "测试规则" + unit["unit_id"],
                        "type": "规则",
                        "scope": unit["scope"],
                        "evidence_ids": [
                            s["evidence_id"] for s in unit["primary"] + unit["context"]
                        ],
                    }
                ],
                "relationships": [],
            }
    selections["deadline"] = {"mode": "text_only", "entities": [], "relationships": []}
    packet = build_review_packet(prepared, selections)
    sealed = approve_packet(packet)
    target = tmp_path / "full-candidate"
    result = await import_candidate(
        sealed,
        target,
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
    )
    assert result["readback"]["persisted_new_process_verified"] is True
    report = await readback_candidate(
        sealed,
        target,
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
        smoke_query="测试保险完整责任条件",
    )
    for span in packet["evidence"].values():
        for flag in ("only_need_context", "only_need_prompt"):
            assert (
                span["quote"] in report[flag]
                or json.dumps(span["quote"], ensure_ascii=False)[1:-1] in report[flag]
            )
    assert report["relationships"] == 0
    assert report["entities"] == 6
    assert report["chunks"] == 7


@pytest.mark.asyncio
async def test_text_only_empty_anchors_and_unapproved_empty_refused(tmp_path):
    packet = fixture_packet(tmp_path, count=1)
    empty = {"rule-0": {"entities": [], "relationships": []}}
    with pytest.raises(ApprovalError, match="Empty"):
        build_review_packet(packet["prepared"], empty)
    empty["rule-0"]["mode"] = "text_only"
    sealed = approve_packet(build_review_packet(packet["prepared"], empty))
    result = await import_candidate(
        sealed,
        tmp_path / "text-only",
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
    )
    assert result["readback"]["entities"] == result["readback"]["relationships"] == 0


@pytest.mark.asyncio
async def test_native_cache_hit_has_preloaded_ledger_and_truncated_merge_refuses(
    tmp_path,
):
    from lightrag.operate import extract_entities
    from product.approved_import import make_native_rag

    sealed = fixture_sealed(tmp_path, count=3)
    target = tmp_path / "cache-candidate"
    await import_candidate(
        sealed,
        target,
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
    )
    packet = sealed["packet"]
    events = []
    replay = ApprovedReplay(packet, events.append)
    rag = make_native_rag(packet, target, replay, offline_embedding())
    await rag.initialize_storages()
    try:
        extracted = await extract_entities(
            packet["chunks"],
            rag._build_global_config(),
            llm_response_cache=rag.llm_response_cache,
            text_chunks_storage=rag.text_chunks,
        )
        assert (
            len(extracted) == 3 and events == []
        )  # True native cache hits bypass replay.
        obj = next(o for o in packet["objects"].values() if o["kind"] == "Entity")
        descriptions = [
            x["native"]["entities"][0]["description"]
            for x in packet["requests"].values()
        ]

        def prompt(values):
            return (
                packet["runtime"]["summary_template"]
                .format(
                    description_type="Entity",
                    description_name=obj["name"],
                    description_list="\n".join(
                        json.dumps({"Description": v}, ensure_ascii=False)
                        for v in values
                    ),
                    summary_length=200,
                    language=packet["runtime"]["config"]["language"],
                )
                .strip()
            )

        merged = await replay(prompt(descriptions))
        assert len(replay.description_ids(merged)) == 3
        with pytest.raises(ApprovalError, match="omits or adds"):
            await replay(prompt(descriptions[:2]))
        with pytest.raises(ApprovalError, match="partial merge evidence"):
            await replay(prompt([descriptions[0][:-3], *descriptions[1:]]))
    finally:
        await rag.finalize_storages()
    history = [
        json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()
    ]
    merges = [event for event in history if event["stage"] == "merge"]
    assert [len(e["evidence_ids"]) for e in merges] == [2, 2, 2, 3, 3, 3]


@pytest.mark.asyncio
async def test_uncertain_import_is_durable_and_not_replayed(tmp_path):
    from lightrag.utils import EmbeddingFunc

    sealed = fixture_sealed(tmp_path, count=1)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic failure before any provider")

    embedding = EmbeddingFunc(
        embedding_dim=32,
        max_token_size=131072,
        model_name="synthetic-offline-v1",
        func=unavailable,
    )
    target = tmp_path / "uncertain"
    with pytest.raises(Exception):
        await import_candidate(
            sealed,
            target,
            expected_seal_sha256=sealed["seal_sha256"],
            embedding_func=embedding,
        )
    assert json.loads((target / "receipt.json").read_text())["state"] == "uncertain"
    with pytest.raises(ApprovalError, match="Candidate exists"):
        await import_candidate(
            sealed,
            target,
            expected_seal_sha256=sealed["seal_sha256"],
            embedding_func=offline_embedding(),
        )


@pytest.mark.asyncio
async def test_readback_refuses_unrelated_directory_before_init(tmp_path):
    sealed = fixture_sealed(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ApprovalError, match="provenance/receipt"):
        await readback_candidate(
            sealed,
            unrelated,
            expected_seal_sha256=sealed["seal_sha256"],
            embedding_func=offline_embedding(),
        )
    assert list(unrelated.iterdir()) == []


@pytest.mark.parametrize(
    "native_type,english_alias",
    [
        ("产品", "Product"),
        ("保障计划", "Plan"),
        ("保障责任", "Coverage"),
        ("责任选择条件", "UnderwritingRule"),
        ("合同规则", "Rule"),
        ("术语定义", "Concept"),
        ("机构", "Organization"),
        ("其他", "Other"),
    ],
)
def test_all_advertised_native_types_and_english_aliases(
    tmp_path, native_type, english_alias
):
    packet = fixture_packet(tmp_path, count=1)
    for alias in (native_type, english_alias):
        selections = copy.deepcopy(packet["selections"])
        selections["rule-0"]["entities"][0]["type"] = alias
        rebuilt = build_review_packet(packet["prepared"], selections)
        request = next(iter(rebuilt["requests"].values()))
        assert request["native"]["entities"][0]["type"] == native_type


@pytest.mark.asyncio
async def test_all_eight_chinese_types_persist_in_real_native_graph(tmp_path):
    from product.approved_import import make_native_rag

    packet = fixture_packet(tmp_path, count=1)
    unit = packet["prepared"]["documents"][0]["units"][0]
    kinds = [
        "产品",
        "保障计划",
        "保障责任",
        "责任选择条件",
        "合同规则",
        "术语定义",
        "机构",
        "其他",
    ]
    selections = {
        "rule-0": {
            "entities": [
                {
                    "entity_id": f"kind-{index}",
                    "name": "测试类型" + kind,
                    "type": kind,
                    "scope": unit["scope"],
                    "evidence_ids": [
                        span["evidence_id"]
                        for span in unit["primary"] + unit["context"]
                    ],
                }
                for index, kind in enumerate(kinds)
            ],
            "relationships": [],
        }
    }
    sealed = approve_packet(build_review_packet(packet["prepared"], selections))
    target = tmp_path / "chinese-types"
    await import_candidate(
        sealed,
        target,
        expected_seal_sha256=sealed["seal_sha256"],
        embedding_func=offline_embedding(),
    )
    replay = ApprovedReplay(sealed["packet"])
    rag = make_native_rag(sealed["packet"], target, replay, offline_embedding())
    await rag.initialize_storages()
    try:
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        assert {node["entity_type"] for node in nodes} == set(kinds)
    finally:
        await rag.finalize_storages()


def test_distinct_document_versions_never_merge_by_name(tmp_path):
    packet = fixture_packet(
        tmp_path, document_versions=["source-revision-1", "source-revision-2"]
    )
    assert len([o for o in packet["objects"].values() if o["kind"] == "Entity"]) == 4
    assert len([o for o in packet["objects"].values() if o["kind"] == "Relation"]) == 2
    assert all(len(o["contributions"]) == 1 for o in packet["objects"].values())


def test_untrusted_self_rehash_cannot_replace_independent_pin(tmp_path):
    sealed = fixture_sealed(tmp_path)
    trusted_pin = sealed["seal_sha256"]
    changed = copy.deepcopy(sealed)
    changed["approval"]["reviewer"] = "untrusted-replacement"
    changed["seal_sha256"] = digest(
        {k: v for k, v in changed.items() if k != "seal_sha256"}
    )
    with pytest.raises(ApprovalError, match="Seal pin mismatch"):
        validate_seal(changed, expected_seal_sha256=trusted_pin)


@pytest.mark.asyncio
async def test_single_writer_refusal_before_candidate_creation(tmp_path):
    import fcntl

    sealed = fixture_sealed(tmp_path)
    target = tmp_path / "busy-candidate"
    with (tmp_path / "busy-candidate.import.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ApprovalError, match="active writer"):
            await import_candidate(
                sealed,
                target,
                expected_seal_sha256=sealed["seal_sha256"],
                embedding_func=offline_embedding(),
            )
    assert not target.exists()
