"""Synthetic, offline contracts for atomic extract-only publication."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from product import answer_evidence as gate
from product.evidence_trial import digest


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resign(catalog: dict) -> None:
    catalog["snapshot_sha256"] = digest(
        {key: value for key, value in catalog.items() if key != "snapshot_sha256"}
    )


def make_document(ref: str, product: str, rows: list[tuple]) -> dict:
    units = []
    offset = 0
    for identifier, title, text, requires, scope in rows:
        units.append(
            {
                "id": identifier,
                "title": title,
                "start": offset,
                "end": offset + len(text),
                "text": text,
                "text_sha256": sha(text),
                "anchors": [title],
                "scope_status": "reviewed",
                "scope": scope,
                "requires": requires,
                "physical_pages": [1],
            }
        )
        offset += len(text)
    source = "".join(unit["text"] for unit in units)
    return {
        "document_ref": ref,
        "product_ref": product,
        "file_path": f"{ref}.md",
        "source_sha256": sha(source),
        "extracted_text_sha256": sha(source),
        "units": units,
    }


@pytest.fixture
def inputs() -> dict:
    selected_scope = [{"plan": "one", "option": "base"}]
    catalog = {
        "schema_version": 2,
        "manifest_sha256": "1" * 64,
        "parser": {"entrypoint": "synthetic.offline", "pypdf_version": "test-only"},
        "coverage": "explicit_reviewed_units_only",
        "documents": [
            make_document(
                "terms",
                "sample-product",
                [
                    (
                        "parent",
                        "General condition",
                        "General condition\nA prior request is necessary, but not sufficient.\n",
                        [],
                        [{}],
                    ),
                    (
                        "rule",
                        "Benefit condition",
                        "Benefit condition\nA or B is required. The general condition also applies.\n",
                        ["parent"],
                        selected_scope,
                    ),
                    (
                        "definition",
                        "Term definition",
                        "Term definition\nTerm Z means a scheduled event.\n",
                        [],
                        selected_scope,
                    ),
                    (
                        "other-plan",
                        "Different plan",
                        "Different plan\nOnly the second plan uses condition C.\n",
                        [],
                        [{"plan": "two", "option": "plus"}],
                    ),
                ],
            ),
            make_document(
                "other-terms",
                "sample-product",
                [
                    (
                        "other-doc",
                        "Other document",
                        "Other document\nD applies.\n",
                        [],
                        [{}],
                    )
                ],
            ),
            make_document(
                "foreign-terms",
                "foreign-product",
                [
                    (
                        "foreign",
                        "Foreign document",
                        "Foreign document\nE applies.\n",
                        [],
                        [{}],
                    )
                ],
            ),
        ],
    }
    resign(catalog)
    return {
        "catalog": catalog,
        "case": {
            "question": "What do the selected source units say?",
            "product_ref": "sample-product",
            "allowed_document_refs": ["terms"],
            "scope": {"plan": "one", "option": "base"},
        },
        "retrieval": {
            "status": "success",
            "data": {
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "file_path": "terms.md",
                        "content": "Benefit condition\nTerm definition",
                    }
                ]
            },
        },
        "expected_catalog_sha256": catalog["snapshot_sha256"],
    }


def packet_for(inputs: dict) -> dict:
    return gate.build_packet(**inputs)


def candidate_for(packet: dict, identifier: str = "rule", kind: str = "quote") -> dict:
    text = next(unit["text"] for unit in packet["evidence"] if unit["id"] == identifier)
    return {
        "contract_version": gate.CONTRACT_VERSION,
        "packet_sha256": packet["packet_sha256"],
        "claims": [
            {
                "claim_id": "claim-1",
                "packet_sha256": packet["packet_sha256"],
                "kind": kind,
                "evidence_ids": [identifier],
                "text": text,
            }
        ],
    }


def run(inputs: dict, packet: dict, candidate=None, **kwargs) -> dict:
    return gate.gate_answer(
        packet,
        candidate,
        **inputs,
        completion_status=kwargs.pop("completion_status", "complete"),
        **kwargs,
    )


def assert_blocked(result: dict) -> None:
    assert result["status"] == "blocked"
    assert result["public"] == {"notice": gate.BLOCKED_NOTICE, "excerpts": []}
    assert result["audit"]["mechanical_errors"]
    assert result["audit"]["claim_checks"] == []
    assert result["audit"]["free_text_released"] is False


def test_complete_quote_copies_all_evidence_and_dependencies(inputs):
    original = copy.deepcopy(inputs)
    packet = packet_for(inputs)
    assert packet == packet_for(inputs)
    assert packet["seed_ids"] == ["rule", "definition"]
    assert packet["evidence_ids"] == ["parent", "rule", "definition"]
    assert packet["packet_sha256"] == digest(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    result = run(inputs, packet, candidate_for(packet))
    assert result["status"] == "extracts_only"
    assert result["audit"]["claim_checks"] == [
        {"claim_index": 0, "status": "exact_quote_verified"}
    ]
    assert result["audit"]["semantic_review"] == "not_performed"
    assert result["audit"]["free_text_released"] is False
    assert result["audit"]["mechanical_errors"] == []
    excerpts = result["public"]["excerpts"]
    assert [unit["evidence_id"] for unit in excerpts] == packet["evidence_ids"]
    assert [unit["text"] for unit in excerpts] == [
        unit["text"] for unit in packet["evidence"]
    ]
    assert inputs == original
    packet["evidence"][0]["text"] = "mutated after publication"
    assert "mutated" not in excerpts[0]["text"]


def test_default_without_generator_only_reports_original_excerpts(inputs):
    result = gate.gate_answer(packet_for(inputs), **inputs)
    assert result["status"] == "extracts_only"
    assert result["audit"]["claim_checks"] == []
    assert "not a complete answer" in result["public"]["notice"]


@pytest.mark.parametrize("kind", ["paraphrase", "inference"])
@pytest.mark.parametrize(
    "identifier,text",
    [
        ("rule", "A and B are required."),
        ("parent", "A prior request is sufficient for payment."),
        ("definition", "Any event outside this definition must be denied."),
    ],
)
def test_valid_citations_cannot_authorize_reversed_conclusions(
    inputs, kind, identifier, text
):
    packet = packet_for(inputs)
    candidate = candidate_for(packet, identifier, kind)
    candidate["claims"][0]["text"] = text
    result = run(inputs, packet, candidate)
    assert result["status"] == "extracts_only"
    assert result["audit"]["claim_checks"][0]["status"] == "pending_semantic_review"
    assert text not in json.dumps(result, ensure_ascii=False)
    assert len(result["public"]["excerpts"]) == 3


@pytest.mark.parametrize("kind", ["quote", "paraphrase", "inference"])
@pytest.mark.parametrize(
    "identifier", ["missing", "other-doc", "foreign", "other-plan"]
)
def test_unknown_wrong_document_and_unselected_scope_citations_block(
    inputs, identifier, kind
):
    packet = packet_for(inputs)
    candidate = candidate_for(packet, kind=kind)
    candidate["claims"][0]["evidence_ids"] = [identifier]
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize("replacement", [[], ["rule", "rule"], ["rule", "parent"]])
def test_empty_duplicate_and_multiunit_quote_citations_block(inputs, replacement):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    candidate["claims"][0]["evidence_ids"] = replacement
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize("change", ["truncate", "join", "normalize", "invert"])
def test_quote_must_equal_one_full_unit_without_normalization(inputs, change):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    claim = candidate["claims"][0]
    if change == "truncate":
        claim["text"] = claim["text"].splitlines()[0]
    elif change == "join":
        claim["text"] += packet["evidence"][0]["text"]
    elif change == "normalize":
        claim["text"] = claim["text"].strip()
    else:
        claim["text"] = claim["text"].replace(" or ", " and ")
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize(
    "field,value",
    [("reviewed", True), ("confidence", 100), ("title", "Pay this claim")],
)
@pytest.mark.parametrize("level", ["answer", "claim"])
def test_self_review_confidence_and_extra_display_fields_block(
    inputs, field, value, level
):
    packet = packet_for(inputs)
    candidate = candidate_for(packet, kind="inference")
    target = candidate if level == "answer" else candidate["claims"][0]
    target[field] = value
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize(
    "field", ["claim_id", "packet_sha256", "kind", "evidence_ids", "text"]
)
def test_missing_claim_contract_fields_block(inputs, field):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    del candidate["claims"][0][field]
    assert_blocked(run(inputs, packet, candidate))


def test_duplicate_claim_ids_block_entire_answer(inputs):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    candidate["claims"].append(copy.deepcopy(candidate["claims"][0]))
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize("position", [0, 1])
def test_bad_claim_never_leaks_previous_or_later_valid_claim(inputs, position):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    bad = copy.deepcopy(candidate["claims"][0])
    bad.update(claim_id="bad-claim", text="Unconditionally payable.")
    candidate["claims"].insert(position, bad)
    result = run(inputs, packet, candidate)
    assert_blocked(result)
    assert "Unconditionally payable" not in json.dumps(result)


def test_mixed_quote_and_inference_emits_only_complete_source_bundle(inputs):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    inference = copy.deepcopy(candidate["claims"][0])
    inference.update(
        claim_id="claim-2", kind="inference", text="Payment is guaranteed."
    )
    candidate["claims"].append(inference)
    result = run(inputs, packet, candidate)
    assert result["status"] == "extracts_only"
    assert [item["status"] for item in result["audit"]["claim_checks"]] == [
        "exact_quote_verified",
        "pending_semantic_review",
    ]
    assert "Payment is guaranteed" not in json.dumps(result)
    assert len(result["public"]["excerpts"]) == 3


@pytest.mark.parametrize("where", ["answer", "claim"])
def test_stale_answer_or_claim_packet_hash_blocks(inputs, where):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    target = candidate if where == "answer" else candidate["claims"][0]
    target["packet_sha256"] = "0" * 64
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize(
    "change", ["text", "dependency", "id", "extra", "hash", "bool-version"]
)
def test_modified_even_self_rehashed_packet_is_not_trusted(inputs, change):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    if change == "text":
        packet["evidence"][1]["text"] = "Benefit condition\nPay everything."
        packet["evidence"][1]["text_sha256"] = sha(packet["evidence"][1]["text"])
    elif change == "dependency":
        packet["evidence"] = packet["evidence"][1:]
        packet["evidence_ids"].remove("parent")
    elif change == "id":
        packet["evidence_ids"].append("rule")
    elif change == "extra":
        packet["answer"] = "Pay everything."
    elif change == "bool-version":
        packet["contract_version"] = True
    else:
        packet["packet_sha256"] = "0" * 64
    if change != "hash":
        packet["packet_sha256"] = digest(
            {key: value for key, value in packet.items() if key != "packet_sha256"}
        )
    assert_blocked(run(inputs, packet, candidate))


@pytest.mark.parametrize(
    "changed", ["question", "scope", "allowlist", "budget", "retrieval", "catalog"]
)
def test_current_trusted_context_invalidates_old_packet(inputs, changed):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    if changed == "question":
        inputs["case"]["question"] = "A different question?"
    elif changed == "scope":
        inputs["case"]["scope"]["plan"] = "two"
    elif changed == "allowlist":
        inputs["case"]["allowed_document_refs"].append("other-terms")
    elif changed == "budget":
        inputs["case"]["max_context_chars"] = 50000
    elif changed == "retrieval":
        inputs["retrieval"]["data"]["chunks"][0]["chunk_id"] = "new-retrieval"
    else:
        inputs["catalog"]["manifest_sha256"] = "2" * 64
        resign(inputs["catalog"])
        inputs["expected_catalog_sha256"] = inputs["catalog"]["snapshot_sha256"]
    assert_blocked(run(inputs, packet, candidate))


def test_catalog_self_hash_does_not_replace_independent_pin(inputs):
    packet = packet_for(inputs)
    inputs["catalog"]["documents"][0]["units"][1]["scope"] = [{}]
    resign(inputs["catalog"])
    with pytest.raises(gate.ContractError, match="untrusted_catalog_snapshot"):
        packet_for(inputs)
    assert_blocked(run(inputs, packet))


@pytest.mark.parametrize("pin", [None, "", "0" * 64, "F" * 64])
def test_missing_invalid_or_wrong_expected_pin_blocks(inputs, pin):
    packet = packet_for(inputs)
    inputs["expected_catalog_sha256"] = pin
    assert_blocked(run(inputs, packet))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-scope",
        "unreviewed",
        "dependency",
        "cycle",
        "depth",
        "units",
        "chars",
        "failed",
        "unmapped",
        "foreign",
        "other-scope",
    ],
)
def test_known_scope_dependency_and_budget_deficits_release_no_partial_bundle(
    inputs, mutation
):
    catalog = inputs["catalog"]
    rule = catalog["documents"][0]["units"][1]
    if mutation == "missing-scope":
        inputs["case"]["scope"] = {"plan": "one"}
    elif mutation == "unreviewed":
        rule["scope_status"] = "unreviewed"
    elif mutation == "dependency":
        rule["requires"] = ["other-doc"]
    elif mutation == "cycle":
        catalog["documents"][0]["units"][0]["requires"] = ["rule"]
    elif mutation == "depth":
        inputs["case"]["max_depth"] = 0
    elif mutation == "units":
        inputs["case"]["max_evidence_units"] = 1
    elif mutation == "chars":
        inputs["case"]["max_context_chars"] = 1
    elif mutation == "failed":
        inputs["retrieval"]["status"] = "error"
    else:
        content = (
            "No known original anchor." if mutation == "unmapped" else "Different plan"
        )
        path = "foreign-terms.md" if mutation == "foreign" else "terms.md"
        inputs["retrieval"]["data"]["chunks"].append(
            {"chunk_id": "bad-chunk", "content": content, "file_path": path}
        )
    resign(catalog)
    inputs["expected_catalog_sha256"] = catalog["snapshot_sha256"]
    packet = packet_for(inputs)
    assert packet["status"] == "blocked"
    assert packet["evidence"] == packet["evidence_ids"] == packet["seed_ids"] == []
    assert_blocked(run(inputs, packet))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-dependency",
        "duplicate-unit",
        "duplicate-document",
        "duplicate-scope",
        "duplicate-chunk",
        "duplicate-allowlist",
        "foreign-allowlist",
        "damaged-hash",
    ],
)
def test_invalid_catalog_or_input_structure_is_fail_closed(inputs, mutation):
    previous = packet_for(inputs)
    doc = inputs["catalog"]["documents"][0]
    if mutation == "unknown-dependency":
        doc["units"][1]["requires"] = ["missing"]
    elif mutation == "duplicate-unit":
        doc["units"].append(copy.deepcopy(doc["units"][0]))
    elif mutation == "duplicate-document":
        inputs["catalog"]["documents"].append(copy.deepcopy(doc))
    elif mutation == "duplicate-scope":
        doc["units"][0]["scope"] = [{}, {}]
    elif mutation == "duplicate-chunk":
        chunks = inputs["retrieval"]["data"]["chunks"]
        chunks.append(copy.deepcopy(chunks[0]))
    elif mutation == "duplicate-allowlist":
        inputs["case"]["allowed_document_refs"] = ["terms", "terms"]
    elif mutation == "foreign-allowlist":
        inputs["case"]["allowed_document_refs"] = ["terms", "foreign-terms"]
    else:
        doc["units"][0]["text_sha256"] = "0" * 64
    resign(inputs["catalog"])
    inputs["expected_catalog_sha256"] = inputs["catalog"]["snapshot_sha256"]
    assert_blocked(run(inputs, previous))


@pytest.mark.parametrize(
    "level",
    ["catalog", "parser", "document", "unit", "case", "retrieval", "data", "chunk"],
)
def test_unknown_input_fields_cannot_smuggle_display_conclusions(inputs, level):
    packet = packet_for(inputs)
    catalog = inputs["catalog"]
    targets = {
        "catalog": catalog,
        "parser": catalog["parser"],
        "document": catalog["documents"][0],
        "unit": catalog["documents"][0]["units"][0],
        "case": inputs["case"],
        "retrieval": inputs["retrieval"],
        "data": inputs["retrieval"]["data"],
        "chunk": inputs["retrieval"]["data"]["chunks"][0],
    }
    targets[level]["summary"] = "Guaranteed payment."
    resign(catalog)
    inputs["expected_catalog_sha256"] = catalog["snapshot_sha256"]
    result = run(inputs, packet)
    assert_blocked(result)
    assert "Guaranteed payment" not in json.dumps(result)


@pytest.mark.parametrize(
    "completion", ["unknown", "truncated", "length", "", None, True]
)
def test_noncomplete_provider_state_is_independent_and_blocking(inputs, completion):
    packet = packet_for(inputs)
    assert_blocked(
        run(inputs, packet, candidate_for(packet), completion_status=completion)
    )


def test_candidate_requires_out_of_band_completion_even_if_json_is_complete(inputs):
    packet = packet_for(inputs)
    assert_blocked(gate.gate_answer(packet, candidate_for(packet), **inputs))


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"a":{"b":1,"b":2}}',
        "{} trailing",
        "{}{}",
        "{",
        "NaN",
        "[Infinity]",
        "1.5",
        '"\\ud800"',
        "[" * 70 + "]" * 70,
    ],
)
def test_strict_json_parser_rejects_ambiguous_or_unsupported_values(raw):
    with pytest.raises(gate.ContractError):
        gate.strict_json_loads(raw)


def test_json_size_node_and_cycle_limits(monkeypatch):
    monkeypatch.setattr(gate, "MAX_JSON_BYTES", 10)
    with pytest.raises(gate.ContractError, match="json_size_limit"):
        gate.strict_json_loads('"' + "x" * 11 + '"')
    monkeypatch.setattr(gate, "MAX_JSON_BYTES", 1000)
    monkeypatch.setattr(gate, "MAX_JSON_NODES", 5)
    with pytest.raises(gate.ContractError, match="json_structure_limit"):
        gate.strict_json_loads("[0,0,0,0,0,0]")
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(gate.ContractError, match="json_structure_limit"):
        gate._json_copy(cyclic)


def test_json_candidate_duplicate_key_blocks_and_clean_json_works(inputs):
    packet = packet_for(inputs)
    raw = json.dumps(candidate_for(packet))
    assert run(inputs, packet, raw)["status"] == "extracts_only"
    ambiguous = raw[:-1] + ',"claims":[]}'
    assert_blocked(run(inputs, packet, ambiguous))
    assert_blocked(run(inputs, packet, raw[:-2]))


def test_maximum_claim_count_and_invalid_kind_block(inputs, monkeypatch):
    packet = packet_for(inputs)
    candidate = candidate_for(packet)
    monkeypatch.setattr(gate, "MAX_CLAIMS", 0)
    assert_blocked(run(inputs, packet, candidate))
    monkeypatch.setattr(gate, "MAX_CLAIMS", 1000)
    candidate["claims"][0]["kind"] = "reviewed"
    assert_blocked(run(inputs, packet, candidate))


def test_complete_quotes_from_distinct_sources_keep_source_identity(inputs):
    other_doc = inputs["catalog"]["documents"][1]
    same_text = inputs["catalog"]["documents"][0]["units"][2]["text"]
    unit = other_doc["units"][0]
    unit.update(
        title="Term definition",
        text=same_text,
        end=len(same_text),
        text_sha256=sha(same_text),
        anchors=["Term definition"],
    )
    other_doc.update(source_sha256=sha(same_text), extracted_text_sha256=sha(same_text))
    inputs["case"]["allowed_document_refs"].append("other-terms")
    inputs["retrieval"]["data"]["chunks"].append(
        {"chunk_id": "chunk-2", "file_path": "other-terms.md", "content": same_text}
    )
    resign(inputs["catalog"])
    inputs["expected_catalog_sha256"] = inputs["catalog"]["snapshot_sha256"]
    packet = packet_for(inputs)
    result = run(inputs, packet, candidate_for(packet, "other-doc"))
    assert result["status"] == "extracts_only"
    repeats = [
        unit for unit in result["public"]["excerpts"] if unit["text"] == same_text
    ]
    assert {unit["document_ref"] for unit in repeats} == {"terms", "other-terms"}


def test_functions_do_not_call_file_network_or_trial_io(inputs, monkeypatch):
    import builtins
    import socket
    from product import evidence_trial

    def forbidden(*args, **kwargs):
        raise AssertionError("I/O is forbidden in the extract-only gate")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for name in ("read_json", "write_json", "prepare", "post_json", "prompts"):
        monkeypatch.setattr(evidence_trial, name, forbidden)
    packet = packet_for(inputs)
    assert run(inputs, packet, candidate_for(packet))["status"] == "extracts_only"


@pytest.mark.parametrize("value", ["x" * 100, {"x" * 100: 0}, ["abcd", "abcd", "abcd"]])
def test_object_size_precheck_runs_before_whole_serialization(monkeypatch, value):
    monkeypatch.setattr(gate, "MAX_JSON_BYTES", 10)

    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized input reached whole-object serialization")

    monkeypatch.setattr(gate.json, "dumps", forbidden)
    with pytest.raises(gate.ContractError, match="json_size_limit"):
        gate._json_copy(value)


def test_final_byte_limit_includes_json_escaping(monkeypatch):
    monkeypatch.setattr(gate, "MAX_JSON_BYTES", 10)
    with pytest.raises(gate.ContractError, match="json_size_limit"):
        gate._json_copy("\x00" * 8)
