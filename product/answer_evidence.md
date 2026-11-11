# Offline answer evidence gate

`product.answer_evidence` is an external, pure in-memory, extract-only contract. It reuses `evidence_trial.validate_catalog` and `assemble`; it does not call a model, read files or environment variables, contact a server, modify an index, or integrate with HTTP, WebUI or streaming. Importing the existing trial module does not invoke its CLI or network functions. There is no automatic production interception.

## Guarantee and trust boundary

This gate checks source identity, the explicitly selected scope, declared dependencies, full-text equality and the complete answer schema. It **does not prove semantic entailment, business review authenticity, source parsing fidelity, complete retrieval, or a claim-payment decision**.

A trusted caller must:

1. Obtain the expected catalog SHA-256 from an independent approval record. Never compute the expected value from the incoming catalog or accept a generator-supplied value as approval. A catalog's self-hash, source hashes and `scope_status: reviewed` are assertions, not signatures or proof that someone performed a review.
2. Review the catalog's complete units, original titles, scope combinations and all necessary dependencies against the original documents. Verify original files, parser output and the published retrieval index belong to the same approved snapshot. This module cannot check those external artifacts. The caller must also verify that the projected retrieval actually came from the supplied question and scope in the same invocation. Hashing a question and a retrieval together cannot establish that causal relationship.
3. Supply the **current** approved catalog pin, actual question, product, document allowlist, selected scope and saved retrieval for this invocation. Rebuilding detects an old packet when these inputs change. It does not detect revocation or wall-clock age if the caller continues supplying obsolete approvals and inputs.
4. Provide completion metadata independently of the candidate. For generated candidates, pass `completion_status="complete"` only when the whole response has been received without truncation or an uncertain provider outcome. A generator cannot attest its own completion.
5. Render only `result["public"]`, as inert plain text with source boundaries. Do not render raw candidate/audit material, interpret source text as instructions, or execute source HTML/Markdown. Do not stream candidates before the gate returns.

The independent approval record and current request inputs are security boundaries. Passing generator-controlled inputs to every argument defeats them. SHA-256 is an integrity binding, not authorization or a semantic verifier.

## Public API

### `strict_json_loads(raw: str) -> JSON value`

Parses exactly one JSON value. Rejects duplicate keys at every nesting level, trailing content, non-finite numbers, floats, invalid UTF-8 strings and excessive structure. Limits are 16 MiB UTF-8, 64 nesting levels and 200,000 nodes. Python object APIs also detach mutable inputs and validate these bounds. A traversal prechecks string/key UTF-8 sizes and cumulative minimum JSON size before whole-object serialization; final serialization enforces the exact byte limit, including escaping. These are input/structure limits, not a hard process-memory ceiling: caller-owned objects and parsing/serialization buffers also consume memory. Object APIs cannot recover duplicate keys that an upstream permissive JSON parser already discarded; use this parser at the ingestion boundary.

### `build_packet(catalog, retrieval, case, *, expected_catalog_sha256) -> dict`

Uses the current schema-2 evidence catalog. Unknown fields are rejected at every fixed-schema level. Scope keys remain caller-defined dimensions, not an insurance rules schema. Duplicate documents, IDs, scope alternatives, dependency IDs, anchors, allowlist entries and retrieved chunk IDs are rejected. The helper raises `ContractError` for malformed inputs and returns a blocked packet for known assembly deficits.

The exact input envelope for retrieval is intentionally narrow:

```json
{
  "status": "success",
  "data": {
    "chunks": [
      {"chunk_id": "saved-chunk-id", "file_path": "approved-source.md", "content": "Original retrieved text"}
    ]
  }
}
```

`status` is `success` or `error`. The trusted caller must explicitly project an already-loaded `/query/data` response to this envelope. Extra native metadata, entities, relationships and summaries are not accepted or used as evidence. This helper does not fetch retrieval or check the live index. Repeated chunk IDs are rejected rather than silently merged.

Required case fields:

- `question`: the actual question.
- `product_ref`: the platform's existing product reference.
- `allowed_document_refs`: a nonempty, duplicate-free allowlist from the same product.
- `scope`: the selected complete tuple of scope dimensions. The reviewed catalog may explicitly use `[{}]` for a unit with no required dimension; a missing dimension is not inferred.

Optional case fields match `assemble`: `max_evidence_units` (default 30), `max_depth` (8), `max_context_chars` (24,000) and `exact_title_lookup` (false). Evaluation oracles such as `expected_evidence_ids` are deliberately not accepted. Unknown case fields fail closed.

The deterministic packet binds:

- Contract version, pinned catalog snapshot, manifest identity and parser identity.
- Full case, including question, product, allowlist, scope and supplied budget settings, plus its hash.
- The exact projected retrieval hash.
- Seed IDs and selected IDs in dependency-first order.
- Each selected complete original unit, original title, offsets/pages, product/document/path, source hash, extracted-full-text hash, quote hash, reviewed scope and `requires` links.
- A SHA-256 over the entire packet body.

All declared dependencies must be present. Any assembly blocker, rejected source/scope, dependency cycle, missing dependency, depth/unit/character-budget failure, unreviewed scope or unusable retrieval prevents publication of **all** partial evidence. The blocked packet contains no selected IDs, seed IDs or source excerpts. This strict wrapper intentionally blocks mixed valid/invalid retrieval instead of silently dropping invalid branches. In particular, a retrieved chunk from an allowed document that matches no reviewed catalog unit also blocks the whole packet. A limited catalog can therefore block an otherwise ordinary full retrieval response. This is a conservative publication policy, not evidence of improved answerability.

The character budget applies to the complete context assembled by the existing trial helper. It is not a token budget, a serialized packet-size budget or an output transport budget. Consumers must not truncate the returned excerpts to fit a smaller display/transport budget. If the entire result cannot be delivered, show only a fixed blocked notice.

### `gate_answer(packet, candidate=None, *, catalog, retrieval, case, expected_catalog_sha256, completion_status="unknown") -> dict`

Rebuilds the expected packet from current trusted inputs and compares the whole packet, rather than accepting its self-hash. A modified packet remains rejected even if its attacker recomputes all hashes. Changed source identity, question, scope, allowlist, retrieval, budgets or schema invalidates old packets.

With no candidate, the default is a report with complete original excerpts only. A supplied candidate may be a plain dictionary or a JSON string parsed strictly. It must use exactly this contract:

```json
{
  "contract_version": 1,
  "packet_sha256": "<current packet SHA-256>",
  "claims": [
    {
      "claim_id": "claim-1",
      "packet_sha256": "<same current packet SHA-256>",
      "kind": "quote",
      "evidence_ids": ["one-selected-unit"],
      "text": "<one entire original unit, including all original whitespace>"
    }
  ]
}
```

Claims are nonempty, have unique IDs and are limited to 1,000 records. Each claim must contain its full text; there is no summary, heading, introductory answer, confidence, reviewed flag or other free-text escape field. Unknown/missing fields, invalid/duplicate references, stale hashes, unfinished generation and malformed claims block the **whole** answer, including any otherwise valid earlier claims.

- `quote`: exactly one selected evidence ID; the text must equal that one entire unit, without trimming, Unicode normalization, stitching or rewriting. Audit status: `exact_quote_verified`. Rendering always copies from the rebuilt packet, never from candidate text.
- `paraphrase` and `inference`: valid references only prove provenance eligibility. Audit status is always `pending_semantic_review`, even when a sentence appears correct. No keyword test, confidence score or self-review can authorize it. The entire response falls back to the packet's complete source excerpts, including every dependency. Candidate prose is never emitted. The module applies this policy to every paraphrase/inference, including correct ones; withholding a known bad sentence is not proof that its semantic error was detected or that normal question-answering accuracy improved.

There is deliberately **no human-review override or free-conclusion release path** in this minimal version. Any later review mechanism would require independent trusted input bound to the full claim text hash, packet, exact scope and reviewer identity, with its own authorization and invalidation policy.

### Return value

- `status`: `extracts_only` or `blocked`. Neither means a business answer was approved.
- `public`: only a fixed notice and source-backed `excerpts`. An excerpt has a fixed label, original title, evidence/document/path references, offsets, pages, declared dependencies, quote hash and complete original text. All selected units are displayed even if the candidate cited only one. No inferred title or generated conclusion is copied.
- `audit`: fixed mechanical error codes, `semantic_review: not_performed`, per-claim ordinal/check status, candidate digest, verified packet digest and `free_text_released: false`. Raw candidates, candidate IDs and candidate text are not copied into the result. Store raw material separately, under the caller's own access controls, if needed for diagnosis.

On any contract failure the public output is only the fixed blocked notice and an empty excerpt list. No fallback to a native answer, partial quote or generated refusal occurs.

## Offline tests

Only synthetic sources are used. No product-specific answers or branches are present. Run from the repository root, without repository `conftest.py` or third-party pytest plugin autoload:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" \
  PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -c /dev/null --noconftest \
  -p no:cacheprovider product/test_answer_evidence.py
```

Tests cover complete quotes and dependencies, valid-ID semantic reversals, forged review/confidence, extra display fields, stale/self-rehashed packets, catalog pins, scope and document boundaries, duplicate keys/IDs, incomplete generation, truncation/stitching, dependency/budget failures, whole-answer atomicity and no file/network calls by the public functions. Passing these tests is not a real-product quality, retrieval recall, performance or production integration result.

### Mutation validation

The 114-test synthetic suite was also executed against four separate in-memory mutations; repository source was not modified by these runs. Each mutant caused real assertion failures:

- Replace exact quote equality with substring acceptance: **2 failures**, `test_quote_must_equal_one_full_unit_without_normalization[truncate]` and `[normalize]`.
- Remove unknown/unselected evidence rejection: **8 failures**, `test_unknown_wrong_document_and_unselected_scope_citations_block`, for `missing`, `other-doc`, `foreign` and `other-plan`, each with `paraphrase` and `inference`.
- Skip comparison with the packet rebuilt from current inputs: **6 failures**, `test_modified_even_self_rehashed_packet_is_not_trusted`, for `text`, `dependency`, `id`, `extra`, `hash` and `bool-version`.
- Copy candidate text onto the public surface instead of original evidence: **7 failures**, `test_complete_quote_copies_all_evidence_and_dependencies` and all six `test_valid_citations_cannot_authorize_reversed_conclusions` cases.

The unmodified suite passed all **114** cases. These mutations test the mechanical safety boundary, not semantic accuracy.
