# Offline evidence trial inputs

These files are external diagnostic inputs. They do not modify LightRAG, create a
product master, import documents, call a model, or establish a separate product
workspace. Run commands from the repository root, using the product environment
with its pinned `pypdf` dependency:

```bash
python -m product.trial_examples.verify_examples
```

Schema 2 requires reviewed `extracted_text_sha256` on each document and reviewed
`text_sha256` on every unit. Prepare compares both before copying any evidence;
missing digests, parser drift and stale spans fail closed. These values come
from the existing verification records, not freshly generated expected values.
Schema 1 manifests/catalogs must not be silently reused.

The verifier checks original file SHA-256, unmodified extracted-text SHA-256,
Unicode `[start, end)` spans, per-span SHA-256, exact unique anchors, boundary
strings and dependency IDs. PDF extraction uses
`product.prepare_context.extract_source` directly. Verification snapshots were
created with pypdf 6.21.4. A different extraction result must be reviewed, not
silently assigned replacement offsets.

## Synthetic tuple and dependency fixture

**This fixture is fabricated software test data, not insurance advice or an
actual insurance product.**

Use `synthetic/manifest.json` with `synthetic/retrieval.json` and one case file:

- `case_valid_one.json`: exact tuple `(计划一, 基础)`.
- `case_valid_two.json`: exact tuple `(计划二, 附加)`.
- `case_reject_cartesian_one.json`: reject `(计划一, 附加)`.
- `case_reject_cartesian_two.json`: reject `(计划二, 基础)`.

The saved retrieval contains only `synthetic:paired`. Valid cases must expand its
`requires` dependency to `synthetic:common`. The two scope objects are an OR of
complete tuples; their dimensions must never be expanded into a Cartesian
product. `expected_evidence_ids` are offline set-comparison oracles only, never
retrieval instructions. Rejected cases have an empty expected set. They should
produce no admissible evidence, not an affirmative answer.

## 京康优护 limited source pilot

Use `jingkangyouhu/manifest.json` and any `jingkangyouhu/case_*.json`. The source is
the repository's original `北京人寿京康优护医疗保险条款.pdf`; no PDF copy or rewritten
insurance fact sheet is included. `file_path` is its original basename and must
match the existing native retrieval reference. The manifest's `source_root`
resolves to the repository root independently of the current working directory.

The pilot locates complete source spans for 2.1, 2.3, the introductory common
premise of 2.5, the complete 2.5.2 optional-benefit subclause, the complete 2.6
exclusions clause, and all of section 7 definitions. The latter is intentionally
kept intact so definition references in the selected clauses do not become
isolated paraphrases. Page headers and original extraction whitespace remain.
Physical PDF page numbers in `verification.json` count the cover page as page 1.

All real units have `scope_status: "unreviewed"`. Their `[{}]` scope is a schema
placeholder, **not a reviewed assertion of universal applicability**. The
platform's selected option, insured person's eligibility, policy schedule,
endorsements, other contractual dependencies, and document version have not
been validated. Do not promote these units to reviewed automatically. This
pilot does not establish complete dependency coverage or a whole-product
review. Its questions contain no invented gold answers or expected evidence IDs.
A fail-closed trial should report these units as unreviewed and withhold them
from admissible answer evidence until the appropriate review is completed.

`retrieval_constructed.json` is a deliberately constructed `/query/data`-shaped
response quoting the exact 2.5.2 source span. It allows an offline check of
locator resolution and the unreviewed gate. It is **not** measured LightRAG
retrieval, a stored live result, or evidence of retrieval quality. For a real
retrieval diagnostic, use the user's native `/query/data` response from the
already-imported original document in the existing shared workspace.

## Caller references and limitations

Every `placeholder:platform-*` value must be replaced consistently with the
caller's existing platform product/document reference before a real trial. No
new product or document registry is implied. Keep any real retrieval response
private. No credentials, account data, or environment secrets are needed for
these offline examples. A successful source verification proves provenance and
boundary integrity, not coverage correctness, entitlement, or insurance advice.
