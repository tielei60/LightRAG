"""Synthetic offline preparation contracts, not real-product semantic gold.

``synthetic_manifest`` is a small executable recipe for reviewers/CLI examples.
It assigns review assertions only to this explicitly fictional fixed fixture.
It must not be used to auto-approve arbitrary source files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from product import import_units as module

pytestmark = pytest.mark.offline
FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_reviewed_policy.md"


class CharacterTokenizer:
    """Deterministic injected test tokenizer, never a production claim."""

    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def exact(text, quote):
    start = text.index(quote)
    return {"start": start, "end": start + len(quote), "quote": quote}


def dimension(value):
    return (
        {"kind": value}
        if value in {"universal", "not_applicable", "unknown"}
        else {"kind": "specific", "value": value}
    )


def synthetic_manifest(root):
    """Explicit review recipe for the fixed fictional MD fixture only."""
    path = root / "synthetic_reviewed_policy.md"
    path.write_bytes(FIXTURE.read_bytes())
    inspected = module.inspect_source(path, "synthetic-policy", "fixture-v1")
    text = inspected["text"]
    starts = [0] + [
        text.index("## " + heading)
        for heading in (
            "告知义务",
            "通知例外",
            "等待期间",
            "医疗责任",
            "日额责任",
            "就诊地点",
            "申请期限",
            "后文释义",
            "编辑备注",
        )
    ]
    keys = [
        "identity",
        "disclosure",
        "notice",
        "waiting",
        "medical",
        "daily",
        "hospital",
        "deadline",
        "definition",
        "editor",
    ]
    rows = []
    header = {"start": 0, "end": starts[1], "quote": text[: starts[1]]}
    for key, start, end in zip(keys, starts, [*starts[1:], len(text)], strict=True):
        scope = [("universal", "universal")]
        if key in {"identity", "definition"}:
            scope = [("not_applicable", "not_applicable")]
        if key == "medical":
            scope = [("计划一", "医疗责任"), ("计划二", "附加医疗责任")]
        if key == "daily":
            scope = [("计划二", "日额责任")]
        span = {"start": start, "end": end, "quote": text[start:end]}
        rows.append(
            {
                "unit_id": key,
                "span": span,
                "disposition": "excluded"
                if key == "editor"
                else (
                    "context_only" if key in {"identity", "definition"} else "primary"
                ),
                "reason": "Fictional review fixture: identity, glossary, or editor note",
                "scope": {
                    "combinations": [
                        {
                            "plan": dimension(plan),
                            "responsibility": dimension(responsibility),
                            "evidence": [
                                span if key in {"medical", "daily"} else header
                            ],
                        }
                        for plan, responsibility in scope
                    ]
                },
                "review": {"complete_conditions": True, "dependencies_complete": True},
                "dependencies": [],
            }
        )
    rows[6]["dependencies"] = [
        {
            "unit_id": "definition",
            "reason": "definition",
            "reference_span": exact(text, "认可替代机构（见后文释义）"),
        }
    ]
    return {
        "schema_version": 1,
        "tokenizer_model": "gpt-4o-mini",
        "max_chunk_tokens": 8192,
        "documents": [
            {
                **{
                    key: value
                    for key, value in inspected.items()
                    if key not in {"text", "pages"}
                },
                "path": path.name,
                "product": {
                    "id": "synthetic-product",
                    "label": "合成测试产品",
                    "identity_span": exact(text, "合成测试产品"),
                },
                "units": rows,
            }
        ],
    }


@pytest.fixture
def prepared_input(tmp_path):
    return tmp_path, synthetic_manifest(tmp_path)


def build(inputs):
    root, manifest = inputs
    return module.prepare_import(manifest, root, CharacterTokenizer())


def row(manifest, key):
    return next(r for r in manifest["documents"][0]["units"] if r["unit_id"] == key)


def output(result, key):
    return next(
        r for d in result["documents"] for r in d["units"] if r["unit_id"] == key
    )


def rejects(inputs, code):
    with pytest.raises(module.PreparationError) as caught:
        build(inputs)
    assert caught.value.code == code
    assert caught.value.report["status"] == "blocked"
    return caught.value.report


def test_full_condition_branches_and_exceptions_are_source_copied(prepared_input):
    result = build(prepared_input)
    manifest = prepared_input[1]
    for key in (
        "disclosure",
        "notice",
        "waiting",
        "medical",
        "daily",
        "hospital",
        "deadline",
    ):
        actual = output(result, key)
        assert actual["primary"][0]["quote"] == row(manifest, key)["span"]["quote"]
        assert actual["primary"][0]["quote"] in actual["content"]
    assert "且对保险事故的发生有严重影响" in output(result, "disclosure")["content"]
    assert "该例外本身不构成给付条件" in output(result, "notice")["content"]
    assert "不是开始住院" in output(result, "waiting")["content"]
    assert "合同另有更长约定" in output(result, "deadline")["content"]
    assert "住院日额为每日二百元" not in output(result, "medical")["content"]
    assert "医疗责任免赔额为人民币一千元" not in output(result, "daily")["content"]


def test_later_definition_is_complete_context_not_new_primary(prepared_input):
    result = build(prepared_input)
    hospital = output(result, "hospital")
    definition = next(
        s for s in hospital["context"] if s.get("unit_id") == "definition"
    )
    assert definition["role"] == "context"
    assert definition["start"] > hospital["primary"][0]["end"]
    assert "不具备任一必要条件" in definition["quote"]
    assert definition["quote"] == row(prepared_input[1], "definition")["span"]["quote"]
    assert [u["unit_id"] for u in result["documents"][0]["units"]].count(
        "definition"
    ) == 0
    report = next(r for r in result["report"]["units"] if r["unit_id"] == "definition")
    assert report["status"] == "copied_context_only"
    assert report["copied_into_unit_ids"] == ["hospital"]


def test_scope_pairs_never_form_cartesian_product(prepared_input):
    medical = output(build(prepared_input), "medical")
    assert [
        (r["plan"]["value"], r["responsibility"]["value"])
        for r in medical["scope"]["combinations"]
    ] == [("计划一", "医疗责任"), ("计划二", "附加医疗责任")]
    assert len(medical["scope"]["combinations"]) == 2


def test_dimension_arrays_rejected(prepared_input):
    row(prepared_input[1], "medical")["scope"] = {
        "plans": ["one", "two"],
        "responsibilities": ["a", "b"],
    }
    rejects(prepared_input, "invalid_scope")


@pytest.mark.parametrize("kind", ["unknown", None])
def test_unknown_is_not_universal(prepared_input, kind):
    row(prepared_input[1], "medical")["scope"]["combinations"][0]["plan"] = (
        {"kind": kind} if kind else None
    )
    rejects(prepared_input, "unknown_scope")


def test_not_applicable_remains_distinct(prepared_input):
    result = build(prepared_input)
    source = prepared_input[1]["documents"][0]
    assert row(prepared_input[1], "definition")["scope"]["combinations"][0]["plan"] == {
        "kind": "not_applicable"
    }
    assert source["document_version"] == "fixture-v1"
    assert "product_version" not in result["sources"][0]


@pytest.mark.parametrize(
    "field", ["original_sha256", "extracted_text_sha256", "parser_binding"]
)
def test_stale_source_and_parser_pins(prepared_input, field):
    prepared_input[1]["documents"][0][field] = "changed"
    rejects(prepared_input, "stale_source")


def test_original_file_change(prepared_input):
    root, manifest = prepared_input
    (root / manifest["documents"][0]["path"]).write_text("changed")
    rejects(prepared_input, "stale_source")


@pytest.mark.parametrize(
    "field,value", [("start", 1), ("quote", "paraphrase"), ("end", True)]
)
def test_stale_exact_offsets_and_quote(prepared_input, field, value):
    row(prepared_input[1], "medical")["span"][field] = value
    rejects(prepared_input, "stale_span")


def test_partition_gap_is_not_called_complete(prepared_input):
    span = row(prepared_input[1], "medical")["span"]
    span["start"] += 1
    span["quote"] = span["quote"][1:]
    rejects(prepared_input, "incomplete_partition")


def test_trailing_source_omission_is_rejected(prepared_input):
    prepared_input[1]["documents"][0]["units"].pop()
    rejects(prepared_input, "incomplete_partition")


def test_missing_dependency_rejected(prepared_input):
    row(prepared_input[1], "hospital")["dependencies"][0]["unit_id"] = "missing"
    rejects(prepared_input, "missing_dependency")


def test_unspecified_dependencies_rejected(prepared_input):
    del row(prepared_input[1], "hospital")["dependencies"]
    rejects(prepared_input, "unknown_dependencies")


def test_excluded_dependency_rejected(prepared_input):
    row(prepared_input[1], "hospital")["dependencies"][0]["unit_id"] = "editor"
    rejects(prepared_input, "excluded_dependency")


@pytest.mark.parametrize("key", ["complete_conditions", "dependencies_complete"])
def test_hashes_cannot_replace_review(prepared_input, key):
    row(prepared_input[1], "hospital")["review"][key] = False
    rejects(prepared_input, "unreviewed_unit")


def test_duplicate_identity_rejected(prepared_input):
    row(prepared_input[1], "definition")["unit_id"] = "hospital"
    rejects(prepared_input, "ambiguous_unit")


def test_cycle_is_not_silently_cut(prepared_input):
    root, manifest = prepared_input
    unit = row(manifest, "definition")
    unit["scope"] = copy.deepcopy(row(manifest, "hospital")["scope"])
    unit["dependencies"] = [
        {"unit_id": "hospital", "reason": "definition", "reference_span": unit["span"]}
    ]
    rejects((root, manifest), "dependency_cycle")


def test_necessary_context_over_budget_fails_with_report(prepared_input):
    result = build(prepared_input)
    total = output(result, "hospital")["tokens"]
    prepared_input[1]["max_chunk_tokens"] = total - 1
    report = rejects(prepared_input, "oversized_unit")
    assert any(r["status"] == "blocked_oversized" for r in report["units"])
    assert "do not establish" in report["semantic_completeness"]


def test_dependency_count_and_depth_are_bounded(prepared_input):
    root, manifest = prepared_input
    notice = row(manifest, "notice")
    notice["dependencies"] = [
        {"unit_id": "hospital", "reason": "condition", "reference_span": notice["span"]}
    ]
    manifest["max_dependencies"] = 1
    rejects((root, manifest), "dependency_count")
    manifest["max_dependencies"] = 16
    manifest["max_dependency_depth"] = 1
    rejects((root, manifest), "dependency_depth")


def test_cross_scope_condition_is_not_universal(prepared_input):
    unit = row(prepared_input[1], "daily")
    unit["dependencies"] = [
        {"unit_id": "medical", "reason": "condition", "reference_span": unit["span"]}
    ]
    rejects(prepared_input, "cross_scope_dependency")


def test_same_text_different_source_has_distinct_provenance_and_native_content(
    prepared_input,
):
    root, manifest = prepared_input
    doc = copy.deepcopy(manifest["documents"][0])
    doc["source_id"], doc["path"] = "another-source", "another.md"
    (root / doc["path"]).write_bytes(
        (root / manifest["documents"][0]["path"]).read_bytes()
    )
    for unit in doc["units"]:
        unit["unit_id"] += "-other"
        for dep in unit["dependencies"]:
            dep["unit_id"] += "-other"
    manifest["documents"].append(doc)
    result = build((root, manifest))
    first, other = output(result, "notice"), output(result, "notice-other")
    assert first["primary"][0]["quote"] == other["primary"][0]["quote"]
    assert first["evidence_id"] != other["evidence_id"]
    assert first["content_sha256"] != other["content_sha256"]


def second_source(inputs, *, product=False, version=False):
    root, manifest = inputs
    doc = copy.deepcopy(manifest["documents"][0])
    doc["path"] = "second.md"
    if version:
        doc["document_version"] = "fixture-v2"
    else:
        doc["source_id"] = "second-source"
    if product:
        doc["product"]["id"] = "another-product"
        doc["product"]["label"] = "测试产品"
    (root / doc["path"]).write_bytes(
        (root / manifest["documents"][0]["path"]).read_bytes()
    )
    for unit in doc["units"]:
        unit["unit_id"] += "-other"
        for dep in unit["dependencies"]:
            dep["unit_id"] += "-other"
    manifest["documents"].append(doc)
    row(manifest, "hospital")["dependencies"][0]["unit_id"] = "definition-other"


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({}, "cross_source_dependency"),
        ({"product": True}, "cross_product_dependency"),
        ({"version": True}, "cross_version_dependency"),
    ],
)
def test_cross_document_dependencies_are_explicitly_refused(
    prepared_input, kwargs, code
):
    second_source(prepared_input, **kwargs)
    rejects(prepared_input, code)


def test_unsupported_composite_never_skips_other_source_hash_validation(prepared_input):
    second_source(prepared_input)
    prepared_input[1]["documents"][1]["original_sha256"] = "changed"
    rejects(prepared_input, "stale_source")


def test_copied_primary_retains_one_evidence_identity(prepared_input):
    root, manifest = prepared_input
    definition = row(manifest, "definition")
    definition["disposition"] = "primary"
    result = build((root, manifest))
    primary = output(result, "definition")["primary"][0]
    context = next(
        s
        for s in output(result, "hospital")["context"]
        if s.get("unit_id") == "definition"
    )
    assert primary["evidence_id"] == context["evidence_id"]
    assert primary["role"] == "primary" and context["role"] == "context"


def test_report_preserves_exclusion_and_no_semantic_claim(prepared_input):
    result = build(prepared_input)
    assert result["report"]["status"] == "prepared_not_semantically_approved"
    editor = next(r for r in result["report"]["units"] if r["unit_id"] == "editor")
    assert editor["status"] == "excluded_by_review"
    assert editor["reason"]
    assert all(
        "本节仅记录软件测试说明" not in u["content"]
        for u in result["documents"][0]["units"]
    )
    assert result["prepared_sha256"] == module.canonical_sha256(
        {k: v for k, v in result.items() if k != "prepared_sha256"}
    )


def test_changed_during_preparation_fails(prepared_input, monkeypatch):
    root, manifest = prepared_input
    original = module.chunking_by_token_size

    def mutate(*args, **kwargs):
        chunks = original(*args, **kwargs)
        (root / manifest["documents"][0]["path"]).write_text("changed")
        return chunks

    monkeypatch.setattr(module, "chunking_by_token_size", mutate)
    rejects(prepared_input, "source_changed")


def test_unsupported_format_is_explicit(prepared_input):
    root, manifest = prepared_input
    (root / "file.docx").write_bytes(b"unsupported")
    manifest["documents"][0]["path"] = "file.docx"
    rejects(prepared_input, "source_unreadable")


def test_native_tiktoken_counts_every_copied_exception(prepared_input):
    root, manifest = prepared_input
    tokenizer = module.TiktokenTokenizer(model_name=manifest["tokenizer_model"])
    result = module.prepare_import(manifest, root)
    assert result["tokenizer"]["injected"] is False
    for unit in result["documents"][0]["units"]:
        assert unit["tokens"] == len(tokenizer.encode(unit["content"]))
    hospital = output(result, "hospital")
    manifest["max_chunk_tokens"] = hospital["tokens"] - 1
    with pytest.raises(module.PreparationError) as caught:
        module.prepare_import(manifest, root)
    assert caught.value.code == "oversized_unit"


def test_scope_copy_cannot_bypass_excluded_disposition(prepared_input):
    manifest = prepared_input[1]
    editor = row(manifest, "editor")
    row(manifest, "medical")["scope"]["combinations"][0]["evidence"] = [editor["span"]]
    rejects(prepared_input, "excluded_scope_evidence")


def test_scope_context_uses_are_in_report(prepared_input):
    result = build(prepared_input)
    identity = next(r for r in result["report"]["units"] if r["unit_id"] == "identity")
    assert identity["status"] == "copied_context_only"
    assert set(identity["copied_into_unit_ids"]) == {
        u["unit_id"] for u in result["documents"][0]["units"]
    }


def test_not_applicable_cannot_smuggle_business_condition(prepared_input):
    row(prepared_input[1], "hospital")["dependencies"][0]["reason"] = "condition"
    rejects(prepared_input, "cross_scope_dependency")


def test_transitive_later_dependencies_remain_whole(prepared_input):
    root, manifest = prepared_input
    hospital = row(manifest, "hospital")
    notice = row(manifest, "notice")
    notice["dependencies"] = [
        {"unit_id": "hospital", "reason": "condition", "reference_span": notice["span"]}
    ]
    result = build((root, manifest))
    notice_output = output(result, "notice")
    assert hospital["span"]["quote"] in notice_output["content"]
    assert row(manifest, "definition")["span"]["quote"] in notice_output["content"]
    assert len(notice_output["dependency_edges"]) == 2


def test_native_chunker_content_drift_is_blocked(prepared_input, monkeypatch):
    original = module.chunking_by_token_size

    def drift(*args, **kwargs):
        chunks = original(*args, **kwargs)
        chunks[0]["content"] += " changed"
        return chunks

    monkeypatch.setattr(module, "chunking_by_token_size", drift)
    rejects(prepared_input, "native_chunk_mismatch")


def test_symlink_escape_is_rejected(prepared_input):
    root, manifest = prepared_input
    (root / "escape.md").symlink_to(root.parent / "outside.md")
    manifest["documents"][0]["path"] = "escape.md"
    rejects(prepared_input, "ambiguous_source")


def test_cli_inspect_is_unapproved_and_cannot_overwrite(prepared_input, monkeypatch):
    import json
    import sys

    root, manifest = prepared_input
    target = root / "inspection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_units",
            "inspect",
            str(root / manifest["documents"][0]["path"]),
            str(target),
            "--source-id",
            "review-source",
            "--document-version",
            "v1",
        ],
    )
    module.main()
    result = json.loads(target.read_text())
    assert result["review_status"] == "unapproved_source_inspection"
    assert "units" not in result
    with pytest.raises(FileExistsError):
        module.main()


def test_cli_preparation_failure_writes_no_artifact(prepared_input, monkeypatch):
    import json
    import sys

    root, manifest = prepared_input
    manifest["documents"][0]["original_sha256"] = "changed"
    manifest_path, target = root / "review.json", root / "prepared.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_units",
            "prepare",
            str(manifest_path),
            str(target),
            "--root",
            str(root),
        ],
    )
    with pytest.raises(module.PreparationError):
        module.main()
    assert not target.exists()


def test_real_pdf_parser_keeps_physical_page_provenance(tmp_path):
    """A generated two-page PDF exercises pypdf, not real policy correctness."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    pdf = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in (
        "Synthetic PDF product. All plans and responsibilities.",
        "A AND B required; exception C OR D; later E required.",
    ):
        page = pdf.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    path = tmp_path / "synthetic.pdf"
    with path.open("wb") as handle:
        pdf.write(handle)
    info = module.inspect_source(path, "pdf-source", "pdf-v1")
    text = info.pop("text")
    pages = info.pop("pages")
    entire = {"start": 0, "end": len(text), "quote": text}
    manifest = {
        "schema_version": 1,
        "tokenizer_model": "gpt-4o-mini",
        "max_chunk_tokens": 8192,
        "documents": [
            {
                **info,
                "path": path.name,
                "product": {
                    "id": "synthetic-pdf",
                    "label": "Synthetic PDF product",
                    "identity_span": exact(text, "Synthetic PDF product"),
                },
                "units": [
                    {
                        "unit_id": "pdf-unit",
                        "span": entire,
                        "disposition": "primary",
                        "scope": {
                            "combinations": [
                                {
                                    "plan": {"kind": "universal"},
                                    "responsibility": {"kind": "universal"},
                                    "evidence": [entire],
                                }
                            ]
                        },
                        "review": {
                            "complete_conditions": True,
                            "dependencies_complete": True,
                        },
                        "dependencies": [],
                    }
                ],
            }
        ],
    }
    result = module.prepare_import(manifest, tmp_path, CharacterTokenizer())
    unit = result["documents"][0]["units"][0]
    assert len(pages) == 2
    assert unit["primary"][0]["physical_pages"] == [1, 2]
    assert unit["primary"][0]["quote"] == text
    assert "exception C OR D" in unit["content"]


def test_pdf_unreadable_page_fails_closed(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="no readable text"):
        module.inspect_source(path, "blank", "v1")


def test_cli_inspection_is_private_and_never_replaces_existing(
    prepared_input, monkeypatch
):
    import stat
    import sys

    root, manifest = prepared_input
    source = root / manifest["documents"][0]["path"]
    target = root / "private-inspection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_units",
            "inspect",
            str(source),
            str(target),
            "--source-id",
            "fixture",
            "--document-version",
            "v1",
        ],
    )
    module.main()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    original = target.read_bytes()
    target.chmod(0o640)
    with pytest.raises(FileExistsError):
        module.main()
    assert target.read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_cli_rejects_duplicate_manifest_keys_before_output(tmp_path, monkeypatch):
    import sys

    source, target = tmp_path / "ambiguous.json", tmp_path / "out.json"
    source.write_text('{"schema_version":1,"schema_version":2}')
    monkeypatch.setattr(
        sys, "argv", ["import_units", "prepare", str(source), str(target)]
    )
    with pytest.raises(ValueError):
        module.main()
    assert not target.exists()
