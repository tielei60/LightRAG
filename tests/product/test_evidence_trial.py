"""Offline contracts for the opt-in external evidence experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import stat

import pytest

from product import evidence_trial as trial

pytestmark = pytest.mark.offline


@pytest.fixture
def prepared(tmp_path):
    text = "Sample product\nParent condition: A is required.\nBenefit rule: B applies.\nOther plan: C applies."
    path = tmp_path / "source.md"
    path.write_text(text)
    parent_start = text.index("Parent")
    rule_start = text.index("Benefit")
    other_start = text.index("Other")
    manifest = {
        "schema_version": trial.SCHEMA_VERSION,
        "documents": [
            {
                "document_ref": "existing-doc",
                "product_ref": "existing-product",
                "file_path": "source.md",
                "source": "source.md",
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "units": [
                    {
                        "id": "parent",
                        "start": parent_start,
                        "end": rule_start,
                        "title": "Parent condition",
                        "anchors": ["Parent condition:"],
                        "scope_status": "reviewed",
                        "scope": [{}],
                        "requires": [],
                    },
                    {
                        "id": "benefit",
                        "start": rule_start,
                        "end": other_start,
                        "title": "Benefit rule",
                        "anchors": ["Benefit rule:"],
                        "scope_status": "reviewed",
                        "scope": [
                            {"plan": "one", "option": "base"},
                            {"plan": "two", "option": "plus"},
                        ],
                        "requires": ["parent"],
                    },
                    {
                        "id": "other",
                        "start": other_start,
                        "end": len(text),
                        "title": "Other plan",
                        "anchors": ["Other plan:"],
                        "scope_status": "reviewed",
                        "scope": [{"plan": "three"}],
                        "requires": [],
                    },
                ],
            }
        ],
    }
    manifest["documents"][0]["extracted_text_sha256"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    for unit in manifest["documents"][0]["units"]:
        unit["text_sha256"] = hashlib.sha256(
            text[unit["start"] : unit["end"]].encode()
        ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    catalog = trial.prepare(manifest_path)
    case = {
        "question": "What is the benefit?",
        "product_ref": "existing-product",
        "allowed_document_refs": ["existing-doc"],
        "scope": {"plan": "one", "option": "base"},
        "expected_evidence_ids": ["parent", "benefit"],
    }
    retrieval = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "chunk_id": "chunk1",
                    "file_path": "source.md",
                    "content": "Benefit rule: B applies.",
                }
            ],
            "entities": [],
            "relationships": [],
        },
    }
    return manifest_path, catalog, case, retrieval


def resign(catalog):
    catalog["snapshot_sha256"] = trial.digest(
        {k: v for k, v in catalog.items() if k != "snapshot_sha256"}
    )


def test_complete_dependency_is_added_without_rewriting(prepared):
    _, catalog, case, data = prepared
    result = trial.assemble(catalog, data, case)
    assert result["status"] == "ready"
    assert result["seed_ids"] == ["benefit"]
    assert result["evidence_ids"] == ["parent", "benefit"]
    assert result["metrics"]["seed_hits"] == 1
    assert result["metrics"]["all_expected_present"]
    assert "Parent condition: A is required." in result["context"]
    assert result["evidence"][1]["text"] == catalog["documents"][0]["units"][1]["text"]


@pytest.mark.parametrize(
    "scope,ready",
    [
        ({"plan": "one", "option": "base"}, True),
        ({"plan": "two", "option": "plus"}, True),
        ({"plan": "one", "option": "plus"}, False),
        ({"plan": "two", "option": "base"}, False),
        ({"plan": "one"}, False),
        ({}, False),
    ],
)
def test_scope_is_or_of_exact_tuples_not_cartesian(prepared, scope, ready):
    _, catalog, case, data = prepared
    case["scope"] = scope
    assert (trial.assemble(catalog, data, case)["status"] == "ready") == ready


def test_unreviewed_scope_blocks(prepared):
    _, catalog, case, data = prepared
    catalog["documents"][0]["units"][1]["scope_status"] = "unreviewed"
    resign(catalog)
    assert "unreviewed_scope:benefit" in trial.assemble(catalog, data, case)["blockers"]


def test_foreign_chunk_is_excluded_even_if_text_matches(prepared):
    _, catalog, case, data = prepared
    data["data"]["chunks"][0]["file_path"] = "different-product.md"
    result = trial.assemble(catalog, data, case)
    assert result["status"] == "blocked"
    assert result["rejected"][0]["reason"] == "unallowed_document"
    assert not result["evidence_ids"]


def test_unknown_allowlist_and_product_are_rejected(prepared):
    _, catalog, case, data = prepared
    for update in (
        {"allowed_document_refs": ["missing"]},
        {"product_ref": "another"},
        {"allowed_document_refs": []},
    ):
        with pytest.raises(ValueError):
            trial.assemble(catalog, data, {**case, **update})


def test_dependency_outside_allowed_documents_blocks(prepared):
    _, catalog, case, data = prepared
    original = catalog["documents"][0]
    parent = original["units"].pop(0)
    catalog["documents"].append(
        {
            **original,
            "document_ref": "second-doc",
            "file_path": "second.md",
            "units": [parent],
        }
    )
    resign(catalog)
    result = trial.assemble(catalog, data, case)
    assert "dependency_outside_allowlist:parent" in result["blockers"]


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"max_context_chars": 20}, "complete_evidence_context_budget"),
        ({"max_evidence_units": 1}, "evidence_unit_budget"),
        ({"max_depth": 0}, "dependency_depth:parent"),
    ],
)
def test_limits_block_instead_of_silently_cutting_conditions(
    prepared, change, expected
):
    _, catalog, case, data = prepared
    result = trial.assemble(catalog, data, {**case, **change})
    assert result["status"] == "blocked"
    assert expected in result["blockers"]
    if "max_context_chars" in change:
        assert "Parent condition: A is required." in result["context"]
        assert "Benefit rule: B applies." in result["context"]


def test_dependency_cycle_blocks(prepared):
    _, catalog, case, data = prepared
    catalog["documents"][0]["units"][0]["requires"] = ["benefit"]
    resign(catalog)
    assert any(
        x.startswith("dependency_cycle")
        for x in trial.assemble(catalog, data, case)["blockers"]
    )


def test_oracle_never_changes_selection_or_prompt(prepared):
    _, catalog, case, data = prepared
    first = trial.run_trial(catalog, data, case)
    case["expected_evidence_ids"] = ["other"]
    second = trial.run_trial(catalog, data, case)
    assert first["candidate"]["evidence_ids"] == second["candidate"]["evidence_ids"]
    assert first["prompts"] == second["prompts"]
    assert not second["candidate"]["metrics"]["all_expected_present"]


def test_exact_title_lookup_is_opt_in_and_auditable(prepared):
    _, catalog, case, _ = prepared
    case["question"] = "Explain Benefit rule"
    empty = {"status": "success", "data": {"chunks": []}}
    assert trial.assemble(catalog, empty, case)["status"] == "blocked"
    assert trial.assemble(catalog, empty, {**case, "exact_title_lookup": True})[
        "evidence_ids"
    ] == ["parent", "benefit"]


def test_graph_description_is_not_used_as_source_evidence(prepared):
    _, catalog, case, data = prepared
    data["data"]["entities"] = [{"description": "Benefit rule: B applies."}]
    data["data"]["chunks"] = []
    assert trial.assemble(catalog, data, case)["status"] == "blocked"


def test_same_text_different_source_keeps_distinct_identity(prepared):
    _, catalog, case, data = prepared
    duplicate = copy.deepcopy(catalog["documents"][0])
    duplicate.update(document_ref="copy-doc", file_path="copy.md")
    duplicate["units"] = [duplicate["units"][0]]
    duplicate["units"][0]["id"] = "copy-parent"
    catalog["documents"].append(duplicate)
    catalog["documents"][0]["units"][1]["requires"].append("copy-parent")
    case["allowed_document_refs"].append("copy-doc")
    resign(catalog)
    result = trial.assemble(catalog, data, case)
    assert result["evidence_ids"] == ["parent", "copy-parent", "benefit"]
    assert len({e["document_ref"] for e in result["evidence"]}) == 2


def test_catalog_tampering_and_restart_identity(prepared, tmp_path):
    _, catalog, case, data = prepared
    path = tmp_path / "catalog.json"
    trial.write_json(path, catalog)
    assert trial.assemble(trial.read_json(path), data, case)["evidence_ids"] == [
        "parent",
        "benefit",
    ]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    catalog["documents"][0]["units"][0]["text"] += "changed"
    with pytest.raises(ValueError, match="snapshot"):
        trial.validate_catalog(catalog)
    resign(catalog)
    with pytest.raises(ValueError, match="text changed"):
        trial.validate_catalog(catalog)


def test_original_hash_and_unique_anchors(prepared):
    manifest_path, _, _, _ = prepared
    manifest = trial.read_json(manifest_path)
    manifest["documents"][0]["units"][0]["anchors"] = ["applies."]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly once"):
        trial.prepare(manifest_path)
    manifest["documents"][0]["source_sha256"] = "stale"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source changed"):
        trial.prepare(manifest_path)


def test_symlink_escape_is_rejected(prepared, tmp_path):
    manifest_path, _, _, _ = prepared
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "alias.md").symlink_to(tmp_path / "source.md")
    manifest = trial.read_json(manifest_path)
    manifest["source_root"] = "nested"
    manifest["documents"][0]["source"] = "alias.md"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        trial.prepare(manifest_path)


def test_replay_makes_no_network_calls_and_preserves_full_trace(
    prepared, tmp_path, monkeypatch
):
    _, catalog, case, data = prepared
    monkeypatch.setattr(
        trial,
        "post_json",
        lambda *a, **k: pytest.fail("Offline replay made network call"),
    )
    paths = []
    for name, value in (("catalog", catalog), ("case", case), ("retrieval", data)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths.extend([f"--{name}", str(path)])
    output = tmp_path / "run.json"
    trial.main(["replay", *paths, "--output", str(output)])
    record = trial.read_json(output)
    assert record["retrieval"] == data
    assert record["answers"] == {}
    assert "Parent condition" in record["prompts"]["candidate"]
    with pytest.raises(SystemExit):
        trial.main(["replay", *paths, "--output", str(output)])


def test_live_requires_explicit_network_opt_in(prepared, tmp_path, monkeypatch):
    _, catalog, case, _ = prepared
    monkeypatch.setattr(
        trial, "post_json", lambda *a, **k: pytest.fail("Network before approval")
    )
    c, q = tmp_path / "catalog.json", tmp_path / "case.json"
    c.write_text(json.dumps(catalog))
    q.write_text(json.dumps(case))
    with pytest.raises(SystemExit):
        trial.main(
            [
                "live",
                "--catalog",
                str(c),
                "--case",
                str(q),
                "--output",
                str(tmp_path / "run.json"),
            ]
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///secret",
        "http://remote.example",
        "https://user:password@example.com",
        "https://example.com?key=secret",
        "https://example.com#secret",
    ],
)
def test_unsafe_endpoints_rejected_before_network(url):
    with pytest.raises(ValueError):
        trial.post_json(url, "/query/data", {}, "secret", 1)


def test_blocked_candidate_never_generates(prepared, tmp_path, monkeypatch):
    _, catalog, case, data = prepared
    case["max_context_chars"] = 1
    monkeypatch.setattr(trial, "post_json", lambda *a, **k: data)
    monkeypatch.setattr(
        trial, "generate", lambda *a, **k: pytest.fail("Blocked answer generated")
    )
    c, q = tmp_path / "catalog.json", tmp_path / "case.json"
    c.write_text(json.dumps(catalog))
    q.write_text(json.dumps(case))
    output = tmp_path / "run.json"
    trial.main(
        [
            "live",
            "--allow-network",
            "--generate",
            "both",
            "--catalog",
            str(c),
            "--case",
            str(q),
            "--output",
            str(output),
        ]
    )
    assert trial.read_json(output)["generation"] == "blocked"


def test_query_options_cannot_replace_messages(monkeypatch):
    monkeypatch.setenv("QUERY_LLM_MODEL", "test")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://example.com/v1")
    monkeypatch.setenv("QUERY_OPENAI_LLM_EXTRA_BODY", '{"messages": []}')
    with pytest.raises(ValueError, match="cannot replace"):
        trial.generate("prompt", "question", 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("anchors", None),
        ("requires", None),
        ("physical_pages", None),
        ("physical_pages", [True]),
        ("requires", [{}]),
    ],
)
def test_malformed_catalog_fields_fail_during_validation(prepared, field, value):
    _, catalog, _, _ = prepared
    catalog["documents"][0]["units"][0][field] = value
    resign(catalog)
    with pytest.raises(ValueError):
        trial.validate_catalog(catalog)


def test_invented_source_title_is_rejected(prepared):
    _, catalog, _, _ = prepared
    catalog["documents"][0]["units"][0]["title"] = "Invented benefit promise"
    resign(catalog)
    with pytest.raises(ValueError, match="original-text quote"):
        trial.validate_catalog(catalog)


def test_exact_title_ignores_proven_other_plan(prepared):
    _, catalog, case, _ = prepared
    original = catalog["documents"][0]["units"][1]
    duplicate = copy.deepcopy(original)
    duplicate.update(id="other-plan-benefit", scope=[{"plan": "other"}])
    catalog["documents"][0]["units"].append(duplicate)
    resign(catalog)
    case.update(question="Explain Benefit rule", exact_title_lookup=True)
    result = trial.assemble(
        catalog, {"status": "success", "data": {"chunks": []}}, case
    )
    assert result["status"] == "ready"
    assert result["evidence_ids"] == ["parent", "benefit"]


def test_baseline_does_not_send_disallowed_chunks_or_mixed_graph(prepared):
    _, catalog, case, data = prepared
    data["data"]["chunks"].append(
        {
            "file_path": "private-other-product.md",
            "content": "DO NOT TRANSMIT OTHER PRODUCT",
        }
    )
    data["data"]["entities"].append({"description": "MIXED GRAPH DESCRIPTION"})
    record = trial.run_trial(catalog, data, case)
    assert "DO NOT TRANSMIT OTHER PRODUCT" not in record["prompts"]["baseline"]
    assert "MIXED GRAPH DESCRIPTION" not in record["prompts"]["baseline"]
    assert "DO NOT TRANSMIT OTHER PRODUCT" in json.dumps(record["retrieval"])


def live_args(prepared, tmp_path):
    _, catalog, case, _ = prepared
    c, q, output = (
        tmp_path / "catalog.json",
        tmp_path / "case.json",
        tmp_path / "run.json",
    )
    c.write_text(json.dumps(catalog))
    q.write_text(json.dumps(case))
    return [
        "live",
        "--allow-network",
        "--catalog",
        str(c),
        "--case",
        str(q),
        "--output",
        str(output),
    ], output


def test_full_same_request_trace_never_saves_keys(prepared, tmp_path, monkeypatch):
    _, _, case, data = prepared
    args, output = live_args(prepared, tmp_path)
    monkeypatch.setenv("QUERY_LLM_MODEL", "fixture-query-model")
    monkeypatch.setenv("QUERY_LLM_BINDING_HOST", "https://provider.example/v1")
    monkeypatch.setenv("QUERY_LLM_BINDING_API_KEY", "FIXTURE-PROVIDER-SECRET")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "FIXTURE-RAG-SECRET")
    calls = []

    def transport(endpoint, path, payload, key, timeout, **kwargs):
        calls.append((path, payload))
        saved = trial.read_json(output)
        if path == "/query/data":
            assert saved["retrieval_stage"]["status"] == "sending"
            return data
        assert any(a["status"] == "sending" for a in saved["answers"].values())
        return {
            "id": "fixture-response",
            "choices": [
                {
                    "message": {"content": "Supported answer [evidence:benefit]"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 42},
        }

    monkeypatch.setattr(trial, "post_json", transport)
    trial.main([*args, "--generate", "both"])
    record = trial.read_json(output)
    serialized = output.read_text()
    assert "FIXTURE-PROVIDER-SECRET" not in serialized
    assert "FIXTURE-RAG-SECRET" not in serialized
    assert [path for path, _ in calls] == [
        "/query/data",
        "/chat/completions",
        "/chat/completions",
    ]
    assert record["case"] == case
    assert record["retrieval"] == data
    assert (
        record["answers"]["candidate"]["request"]["messages"][0]["content"]
        == record["prompts"]["candidate"]
    )
    assert record["answers"]["candidate"]["mechanical_checks_passed"]
    assert record["generation_stage"]["status"] == "completed"
    assert record["unavailable_stages"]


def test_retrieval_failure_keeps_pending_request_and_unknown_outcome(
    prepared, tmp_path, monkeypatch
):
    args, output = live_args(prepared, tmp_path)

    def fail(*a, **kw):
        raise TimeoutError("Sensitive server body must not enter trace")

    monkeypatch.setattr(trial, "post_json", fail)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        trial.main(args)
    record = trial.read_json(output)
    assert record["retrieval_stage"]["status"] == "error_or_unknown"
    assert record["retrieval_stage"]["error_type"] == "TimeoutError"
    assert "Sensitive server body" not in output.read_text()


def test_invalid_retrieval_preserves_received_data(prepared, tmp_path, monkeypatch):
    args, output = live_args(prepared, tmp_path)
    bad = {"status": "success", "data": {"chunks": "malformed"}}
    monkeypatch.setattr(trial, "post_json", lambda *a, **k: bad)
    with pytest.raises(ValueError):
        trial.main(args)
    record = trial.read_json(output)
    assert record["retrieval"] == bad
    assert record["assembly_stage"]["status"] == "failed"


def test_interrupted_generation_retains_completed_arm(prepared, tmp_path, monkeypatch):
    _, _, _, data = prepared
    args, output = live_args(prepared, tmp_path)
    monkeypatch.setenv("QUERY_LLM_MODEL", "fixture")
    monkeypatch.setenv("QUERY_LLM_BINDING_HOST", "https://provider.example/v1")
    monkeypatch.setattr(trial, "post_json", lambda *a, **k: data)
    replies = [{"answer": "baseline", "complete": True}, TimeoutError()]

    def generate(*a, **k):
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(trial, "generate", generate)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        trial.main([*args, "--generate", "both"])
    record = trial.read_json(output)
    assert record["answers"]["baseline"]["answer"] == "baseline"
    assert record["answers"]["candidate"]["status"] == "error_or_unknown"


@pytest.mark.parametrize(
    "finish,answer",
    [
        ("length", "Cut off [evidence:benefit]"),
        ("stop", "Wrong citation [evidence:missing]"),
    ],
)
def test_truncation_and_unknown_citations_not_marked_valid(
    prepared, tmp_path, monkeypatch, finish, answer
):
    _, _, _, data = prepared
    args, output = live_args(prepared, tmp_path)
    monkeypatch.setenv("QUERY_LLM_MODEL", "fixture")
    monkeypatch.setenv("QUERY_LLM_BINDING_HOST", "https://provider.example/v1")

    def transport(endpoint, path, payload, key, timeout, **kwargs):
        if path == "/query/data":
            return data
        return {"choices": [{"message": {"content": answer}, "finish_reason": finish}]}

    monkeypatch.setattr(trial, "post_json", transport)
    trial.main([*args, "--generate", "candidate"])
    assert not trial.read_json(output)["answers"]["candidate"][
        "mechanical_checks_passed"
    ]


def test_provider_body_secrets_rejected(monkeypatch):
    monkeypatch.setenv("QUERY_LLM_MODEL", "fixture")
    monkeypatch.setenv("QUERY_LLM_BINDING_HOST", "https://provider.example/v1")
    monkeypatch.setenv(
        "QUERY_OPENAI_LLM_EXTRA_BODY", '{"nested": {"api_key": "secret"}}'
    )
    with pytest.raises(ValueError, match="credentials"):
        trial.provider_configuration()


def test_exact_lookup_requires_boolean(prepared):
    _, catalog, case, data = prepared
    with pytest.raises(ValueError, match="must be boolean"):
        trial.assemble(catalog, data, {**case, "exact_title_lookup": "false"})


@pytest.mark.parametrize("field", ["manifest_sha256", "parser"])
def test_trace_identity_is_validated_before_network(
    prepared, tmp_path, monkeypatch, field
):
    _, catalog, _, _ = prepared
    catalog.pop(field)
    resign(catalog)
    args, _ = live_args(prepared, tmp_path)
    monkeypatch.setattr(
        trial,
        "post_json",
        lambda *a, **k: pytest.fail("Invalid catalog reached network"),
    )
    with pytest.raises(ValueError):
        trial.main(args)


def test_same_original_parser_drift_is_rejected(prepared, monkeypatch):
    from product import prepare_context

    manifest_path, _, _, _ = prepared
    original = prepare_context.extract_source
    source = manifest_path.parent / "source.md"
    unchanged_bytes = source.read_bytes()

    def shifted(path):
        text, _ = original(path)
        # Original bytes are unchanged; only the parser output shifts offsets.
        return " " + text, [(0, len(text) + 1)]

    monkeypatch.setattr(prepare_context, "extract_source", shifted)
    with pytest.raises(ValueError, match="Reviewed extracted text changed"):
        trial.prepare(manifest_path)
    assert source.read_bytes() == unchanged_bytes


def test_reviewed_offsets_cannot_shift_within_unchanged_text(prepared):
    manifest_path, _, _, _ = prepared
    manifest = trial.read_json(manifest_path)
    unit = manifest["documents"][0]["units"][1]
    unit["start"] -= 1
    unit["end"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reviewed evidence span changed"):
        trial.prepare(manifest_path)


@pytest.mark.parametrize("field", ["extracted_text_sha256", "text_sha256"])
@pytest.mark.parametrize("value", [None, "", "invalid", "0" * 64])
def test_review_digests_are_mandatory_and_never_recomputed(prepared, field, value):
    manifest_path, _, _, _ = prepared
    manifest = trial.read_json(manifest_path)
    target = manifest["documents"][0]
    if field == "text_sha256":
        target = target["units"][0]
    if value is None:
        target.pop(field)
    else:
        target[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reviewed|Reviewed"):
        trial.prepare(manifest_path)
    assert trial.read_json(manifest_path) == manifest


def test_schema_one_catalog_and_manifest_are_not_silently_reused(prepared):
    manifest_path, catalog, _, _ = prepared
    manifest = trial.read_json(manifest_path)
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported manifest"):
        trial.prepare(manifest_path)
    catalog["schema_version"] = 1
    resign(catalog)
    with pytest.raises(ValueError, match="Unsupported catalog"):
        trial.validate_catalog(catalog)


def test_existing_output_bytes_and_permissions_are_untouched(prepared, tmp_path):
    args, output = live_args(prepared, tmp_path)
    output.write_bytes(b"Existing unrelated audit; do not modify")
    output.chmod(0o640)
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        trial.main(args)
    assert output.read_bytes() == before
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert not output.with_name(f".{output.name}.lock").exists()


def test_parallel_live_reserves_before_catalog_read_or_provider_call(
    prepared, tmp_path, monkeypatch
):
    import threading

    _, _, _, data = prepared
    args, output = live_args(prepared, tmp_path)
    first_read = threading.Event()
    release = threading.Event()
    outcomes, calls = [], []
    original_read = trial.read_json
    catalog_path = tmp_path / "catalog.json"

    def delayed_read(path):
        if path == catalog_path and threading.current_thread().name == "first-run":
            first_read.set()
            assert release.wait(5)
        return original_read(path)

    monkeypatch.setattr(trial, "read_json", delayed_read)
    monkeypatch.setattr(
        trial, "post_json", lambda *a, **k: calls.append("provider") or data
    )

    def first_run():
        try:
            trial.main(args)
            outcomes.append("completed")
        except BaseException as exc:
            outcomes.append(exc)

    first = threading.Thread(target=first_run, name="first-run")
    first.start()
    try:
        assert first_read.wait(5)
        # This is the former exists-before-first-write race window.
        with pytest.raises(SystemExit):
            trial.main(args)
        assert calls == []
    finally:
        release.set()
        first.join(5)
    assert not first.is_alive()
    assert outcomes == ["completed"]
    assert calls == ["provider"]
    assert trial.read_json(output)["retrieval_stage"]["status"] == "completed"
    assert not output.with_name(f".{output.name}.lock").exists()


def test_two_processes_cannot_publish_same_catalog(prepared, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    manifest_path, _, _, _ = prepared
    output = tmp_path / "simultaneous.json"
    command = [
        sys.executable,
        "-m",
        "product.evidence_trial",
        "prepare",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    ]
    root = Path(__file__).resolve().parents[2]
    processes = [
        subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    for process in processes:
        process.communicate(timeout=20)
    assert sorted(process.returncode for process in processes) == [0, 2]
    trial.validate_catalog(trial.read_json(output))
    assert not output.with_name(f".{output.name}.lock").exists()


def test_output_run_id_change_blocks_replace_and_cleans_temp(tmp_path):
    output = tmp_path / "run.json"
    with trial.reserve_output(output) as run_id:
        changed = {"run_id": "different-owner", "audit": "must survive"}
        output.write_text(json.dumps(changed))
        with pytest.raises(RuntimeError, match="ownership changed"):
            trial.write_json(
                output, {"run_id": run_id, "audit": "old"}, expected_run_id=run_id
            )
        assert trial.read_json(output) == changed
        assert not list(tmp_path.glob(".evidence-*"))
    assert not output.with_name(f".{output.name}.lock").exists()


def test_failed_live_releases_lock_but_preserves_audit(prepared, tmp_path, monkeypatch):
    args, output = live_args(prepared, tmp_path)

    def fail(*a, **kw):
        raise TimeoutError()

    monkeypatch.setattr(trial, "post_json", fail)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        trial.main(args)
    record = trial.read_json(output)
    assert record["retrieval_stage"]["status"] == "error_or_unknown"
    assert not output.with_name(f".{output.name}.lock").exists()
    with pytest.raises(SystemExit):
        trial.main(args)
    assert trial.read_json(output) == record


def test_failure_before_request_preserves_reason_and_releases_lock(prepared, tmp_path):
    args, output = live_args(prepared, tmp_path)
    (tmp_path / "catalog.json").write_text("invalid json")
    with pytest.raises(ValueError):
        trial.main(args)
    record = trial.read_json(output)
    assert record["status"] == "failed_before_request"
    assert record["error_type"] == "JSONDecodeError"
    assert not output.with_name(f".{output.name}.lock").exists()
