# Sprint 152 Working Notes

## Goal

Sprint 152 promotes selected generated report freshness checks for
claim-bearing families without converting local generated rows into broad
release proof.

## Starting Evidence

- Sprint 141 created the normalized report-family architecture in
  `tests/corpus/manifests/report_families.tsv`,
  `tests/corpus/schemas/report_index_fields.md`, and
  `scripts/normalize_report_index.py`.
- Sprint 150 added QR generated-local oracle/report rows and stale-output
  cleanup in `scripts/run_corpus_oracle.py`.
- Sprint 151 added partial-SVD generated-local oracle/report rows, strengthened
  report-index tests, and left Sprint 152 the explicit residual of deciding
  generated report freshness publication policy.
- Current generated-local oracle rows emit `generated_present_unchecked`
  freshness status and remain advisory/local-only unless a caller explicitly
  requires generated families.
- Hosted CI lanes currently publish package, dead-code, coverage, and platform
  evidence through GitHub Actions logs/artifacts, but those logs are external
  evidence and must not be conflated with local generated report freshness.

## Item-To-Day Owner Map

| Sprint 152 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Generated Family Selection | Days 1-3 | Day 1 records the baseline, Day 2 audits candidate families, Day 3 selects families and policy class. |
| Item 2: Freshness Policy | Days 4-5 | Day 4 defines policy semantics, Day 5 maps policy into generator metadata and failure design. |
| Item 3: Generator Stabilization | Days 5-6 | Day 5 designs command/path/metadata stabilization, Day 6 implements the batch. |
| Item 4: Freshness Gate Implementation | Days 7-8 | Day 7 designs report-index gate behavior, Day 8 implements tests and checks. |
| Item 5: CI/Artifact Policy | Days 9-10 | Day 9 decides local/CI artifact posture, Day 10 implements selected follow-through. |
| Item 6: Documentation Alignment | Day 11 | Day 11 aligns report, corpus, maintainer, schema, and high-level guidance. |
| Item 7: Validation And Closeout | Days 12-14 | Day 12 runs integrated regeneration proof, Day 13 runs final quality gates and residual review, Day 14 closes with Sprint 153 handoff. |

## Stop Conditions

- A missing generated report row is treated as pass evidence.
- A generated-local row is promoted to hosted CI, release, package, ABI,
  performance, or state-of-the-art proof without an explicit policy row and
  validation lane.
- An optional-data skip or deferred row is counted as freshness proof.
- A report family is made required without a stable command, artifact path,
  support tier, owner, failure message, and regeneration instructions.
- Strict generated freshness fails but the failure is downgraded without a
  documented advisory/deferred classification.
- A benchmark, sentinel, coverage, dead-code, package, CI, or documentation row
  is flattened into generic pass evidence without preserving row meaning and
  non-claims.
- Hosted workflow logs are cited as source-controlled freshness artifacts.
- Generated `build/` or `coverage/` outputs are added to source control as
  release proof.
- Required Python/report checks or C quality gates fail after implementation
  changes.

## Daily Log

### Day 1: Freshness Intake

- Re-read the Sprint 152 section of
  `docs/planning/EPIC_13/PROJECT_PLAN.md`. The user-provided Epic 12 line
  reference points to an older Sprint 142 section, so Sprint 152 planning uses
  the Epic 13 project plan and output directory requested by the task.
- Created the Sprint 152 artifact directory and Day 1 generated-report
  baseline artifact.
- Reviewed Sprint 141 report-index architecture, including report-family
  contracts, normalized row fields, freshness states, advisory/generated
  behavior, and source-controlled schema docs.
- Reviewed Sprint 150 QR handoff: `23` solver-backed QR rows, `26` generated
  QR oracle rows, advisory generated freshness, and stale generated-output
  cleanup before current oracle/report output is written.
- Reviewed Sprint 151 partial-SVD handoff: four maintained partial-SVD
  fixtures, `26` generated-local partial-SVD rows, `29` total generated oracle
  rows in the combined command, and `generated_present_unchecked` advisory
  warnings as the main Sprint 152 residual.
- Inventoried generated report producer scripts:
  `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py`,
  `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`,
  `scripts/large_matrix_guardrails.sh`, `scripts/deadcode_workflow.sh`, and
  `scripts/deadcode_report.py`.
- Inventoried generated report families in
  `tests/corpus/manifests/report_families.tsv`: corpus, oracle, benchmark,
  sentinel, guardrail, deadcode, coverage, package, ci, documentation,
  report_index, and runtime_backend.
- Inventoried current workflow surfaces: Linux package install proof,
  dead-code artifact upload, coverage artifact upload, macOS install/pkg-config
  proof, and Windows CMake/CTest/install validation lanes.
- Captured stop conditions for missing rows, local-only generated rows,
  optional skips, required-family promotion, strict freshness failures, row
  meaning flattening, hosted-log overclaiming, generated artifact commits, and
  failed required checks.
- Day 2 handoff: audit candidate generated families for promotion readiness,
  especially oracle/corpus rows from Sprints 150-151, benchmark/sentinel/
  guardrail generated reports, dead-code and coverage artifacts, and hosted CI
  package/platform evidence boundaries.

### Day 2: Family Audit

- Created the generated family candidate audit artifact in
  `artifacts/day2-generated-family-candidate-audit.md`.
- Audited the current report-family manifest in
  `tests/corpus/manifests/report_families.tsv`.
- Audited `scripts/normalize_report_index.py` generated-row ingestion and
  freshness behavior for oracle, benchmark, sentinel, guardrail, coverage,
  dead-code, package, CI, documentation, report-index, and runtime-backend
  families.
- Audited `tests/test_normalize_report_index.py` for current missing-generated,
  required-generated, generated-present, stale, and strict-generated coverage.
- Scored candidate families by claim value, command/path stability,
  local/CI suitability, failure clarity, and claim-boundary safety.
- Identified the strongest Sprint 152 candidates:
  `oracle/generated_reference`, `oracle/solver_backed`, and supporting
  `report_index/missing_generated` rows.
- Kept corpus fixture/generator/expected rows as source-controlled
  prerequisites rather than generated freshness targets.
- Classified benchmark, sentinel advisory, guardrail, dead-code, coverage,
  package, CI, documentation, and runtime-backend rows as advisory,
  source-controlled, or deferred unless Day 3 selects a narrower safely
  closable subset.
- Day 3 handoff: select the generated families for Sprint 152 closure, likely
  the oracle rows plus missing-generated policy support, while preserving
  local-only support tier and explicit non-claims for hosted CI, package, ABI,
  performance, platform, and state-of-the-art evidence.

### Day 3: Family Selection

- Created the generated family selection artifact in
  `artifacts/day3-generated-family-selection.md`.
- Selected `oracle/solver_backed` and `oracle/generated_reference` as the
  Sprint 152 generated freshness publication target because they directly
  support the Sprint 150 QR and Sprint 151 partial-SVD maintained corpus
  evidence.
- Selected `report_index/missing_generated` as supporting policy
  infrastructure so missing selected generated families are visible and
  actionable instead of silently omitted.
- Kept corpus fixture/generator/expected rows as source-controlled
  prerequisites governed by Git and schema validation rather than generated
  freshness targets.
- Deferred required freshness promotion for benchmark, sentinel, guardrail,
  dead-code, and coverage families because they carry higher runtime,
  platform, advisory-triage, or overclaim risk.
- Kept package and CI rows source-controlled or hosted-external only; hosted
  workflow logs are not local generated report freshness artifacts.
- Preserved non-claims for broad QR correctness, broad partial-SVD correctness,
  raw vector/basis/sign/orientation parity, external-library parity, hosted CI
  proof, package-manager availability, shared-library ABI support, broad
  platform support, portable performance, benchmark superiority, coverage
  completeness, zero dead code, and state-of-the-art status.
- Defined rollback rules for nondeterministic oracle output, stale-file
  contamination, weak required-family diagnostics, strict freshness false
  failures, generated artifact commits, hosted-CI dependency, overclaiming,
  skip/defer pass confusion, and unresolved validation failures.
- Day 4 handoff: define exact selected-oracle freshness semantics for missing,
  required, generated-present, fresh, stale, strict, advisory, row-count, and
  command/path metadata behavior.

### Day 4: Freshness Policy Design

- Created the freshness policy design artifact in
  `artifacts/day4-freshness-policy-design.md`.
- Reviewed current `scripts/normalize_report_index.py` semantics for
  `--family`, `--require-generated`, `--check`, `--check-freshness`,
  `--strict-generated`, and `--advisory-ok`.
- Defined the selected freshness state model for `source_controlled`,
  `not_generated`, `generated_present_unchecked`, `fresh`, `stale`,
  `optional_data_skip`, `deferred`, and `unsupported` rows.
- Defined the selected local required oracle command posture:
  generate with `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`,
  normalize with `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`,
  and require oracle freshness with
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Defined strict local oracle policy as a candidate closeout check after
  command/path/metadata stabilization:
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness`.
- Defined required metadata fields for selected oracle rows: command,
  source commit, source branch, generated timestamp, platform, compiler,
  configuration, support tier, artifact path, status, claim scope, non-claims,
  row counts, solver families, and fixture keys.
- Defined local/CI failure policy for missing artifacts, stale commits,
  unchecked rows, comparison failures, row-count mismatches, fixture-key
  mismatches, optional skips, and non-selected generated families.
- Preserved the generated-local versus release-proof boundary: selected oracle
  freshness remains local-only until Days 9-10 explicitly decide otherwise.
- Day 5 handoff: design canonical oracle commands, selected artifact paths,
  row-count and fixture-key checks, actionable required-family diagnostics,
  strict freshness diagnostics, and documentation wording.

### Day 5: Generator Stabilization Design

- Created the generator stabilization design artifact in
  `artifacts/day5-generator-stabilization-design.md`.
- Inspected `scripts/run_corpus_oracle.py` command handling, output paths,
  stale-output cleanup, oracle TSV writing, report TSV writing, skip TSV
  writing, and manifest writing.
- Inspected selected oracle manifest fields: command, source commit, source
  branch, platform, compiler, configuration, oracle row count, solver family
  counts, fixture keys, support tier, claim boundary, and non-claims.
- Chose the canonical selected local oracle generation command:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`.
- Recorded accepted development variants for QR-only, partial-SVD-only, and
  temporary output-directory test paths while keeping the combined command as
  the Sprint 152 strict row-count owner.
- Defined canonical selected output paths under `build/corpus/oracle/`,
  `build/corpus-reports/`, and `build/report-index/`, with generated outputs
  remaining ignored and uncommitted.
- Defined row-level metadata stabilization requirements for oracle row ID,
  fixture key, solver family, operation, comparison kind, command, commit,
  branch, timestamp, platform, compiler, configuration, support tier, status,
  claim scope, and non-claims.
- Defined manifest-level policy expectations for the combined command:
  `52` total oracle rows, `3` generated-reference rows, `23` QR rows,
  `26` partial-SVD rows, solver families `partial_svd,qr,unknown`, and the
  selected QR plus partial-SVD fixture-key set.
- Confirmed existing stale-output cleanup is the right selected behavior:
  remove old oracle TSVs plus stale report `index.tsv`, `skips.tsv`, and
  `manifest.txt` before writing current output.
- Designed actionable diagnostics for missing artifacts, stale commits,
  unexpected row counts, missing solver families, missing fixture keys,
  oracle comparison failures, and unchecked rows under strict policy.
- Day 6 handoff: preserve cleanup behavior, add policy helpers for canonical
  selected oracle command/row-count/fixture-key checks as needed, improve
  required-family diagnostics, and keep generated build artifacts out of
  source control.

### Day 6: Generator Stabilization Implementation

- Created the generator stabilization implementation artifact in
  `artifacts/day6-generator-stabilization-implementation.md`.
- Updated `scripts/normalize_report_index.py` with the canonical selected
  oracle generation command:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`.
- Added selected oracle policy constants for `52` total generated oracle rows:
  `3` generated-reference `unknown` rows, `23` QR solver-backed rows, and
  `26` partial-SVD solver-backed rows.
- Added the selected QR and partial-SVD fixture-key set used to reject partial
  or mismatched generated oracle output when oracle freshness is required or
  strict generated freshness is requested.
- Improved missing required oracle diagnostics to name the expected artifact
  pattern and canonical regeneration command.
- Improved stale oracle diagnostics to include recorded commit, current commit,
  artifact path, and canonical regeneration command.
- Added oracle comparison-failure diagnostics that name fixture key, artifact
  path, and the canonical regeneration command under required or strict
  generated policy.
- Preserved current stale-output cleanup behavior in
  `scripts/run_corpus_oracle.py`; generated oracle/report files are still
  removed before writing current output.
- Updated `tests/test_normalize_report_index.py` with synthetic selected
  oracle rows so the complete selected family passes required freshness and a
  partial selected family fails with `oracle_selected_row_count`.
- Ran focused validation:
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`,
  and
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Day 7 handoff: design the explicit freshness gate behavior around the new
  selected oracle policy, including required/strict/advisory command surfaces,
  failure wording, and check integration boundaries.

### Day 7: Freshness Gate Design

- Created the freshness gate design artifact in
  `artifacts/day7-freshness-gate-design.md`.
- Inspected current report-index freshness implementation in
  `scripts/normalize_report_index.py`, including selected oracle row-count,
  solver-family, fixture-key, stale, missing, and failure diagnostics added on
  Day 6.
- Inspected current focused coverage in `tests/test_normalize_report_index.py`,
  including missing required generated families, advisory/stale behavior,
  partial-SVD oracle stale strictness, complete selected oracle proof, and
  partial selected oracle rejection.
- Defined selected gate scope for `oracle/generated_reference`,
  `oracle/solver_backed`, and supporting `report_index/missing_generated`
  rows.
- Defined required oracle assertions for artifact presence, current commit,
  no comparison failures, `52` selected rows, solver-family counts
  `unknown=3`, `qr=23`, `partial_svd=26`, solver-family presence, and selected
  fixture-key presence.
- Defined strict oracle behavior: fail stale, failing, or incomplete selected
  oracle rows; keep advisory non-selected families scoped and non-claiming.
- Defined advisory/deferred compatibility boundaries for benchmark, sentinel,
  guardrail, coverage, dead-code, package, CI, documentation, and
  runtime-backend families.
- Defined CLI behavior for the required oracle gate, strict oracle gate, and
  advisory family freshness checks.
- Wrote the Day 8 implementation checklist: strengthen tests for missing
  required oracle output, stale required rows, oracle comparison failure,
  missing solver family, missing fixture key, advisory/deferred compatibility,
  and focused Python/report validation.
- Day 8 handoff: implement the executable gate coverage from the Day 7 matrix
  without widening selected generated rows beyond local fixture-level oracle
  evidence.

### Day 8: Freshness Gate Implementation

- Created the freshness gate implementation artifact in
  `artifacts/day8-freshness-gate-implementation.md`.
- Expanded the synthetic selected-oracle fixture helper in
  `tests/test_normalize_report_index.py` so focused tests can generate
  complete, partial, stale, failing, missing-solver-family, and
  missing-fixture-key oracle outputs without requiring a compiled solver
  library for every mismatch case.
- Added executable coverage for missing required oracle artifacts. The failure
  now asserts the selected artifact pattern and canonical regeneration command.
- Added executable coverage for stale required oracle rows. The failure asserts
  recorded commit, current commit, and artifact path diagnostics.
- Added executable coverage for oracle comparison failures under required
  freshness. The failure asserts fixture key and artifact path diagnostics.
- Added executable coverage for missing selected solver-family output with
  `oracle_selected_solver_families`.
- Added executable coverage for missing selected fixture keys with
  `oracle_selected_fixture_keys`, including a case where total row count and
  solver-family counts remain correct.
- Added compatibility coverage proving `coverage` remains advisory by default
  and only fails when explicitly required, while `package` remains
  source-controlled advisory freshness.
- Re-ran focused validation:
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`,
  and
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Day 9 handoff: decide whether selected generated freshness gates stay
  local-only or get a hosted CI/artifact policy, while preserving local-only
  support tier and non-claims.

### Day 9: CI And Artifact Policy Design

- Created the CI/artifact policy artifact in
  `artifacts/day9-ci-artifact-policy.md`.
- Audited `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml` for existing hosted CI roles, generated
  report commands, and artifact uploads.
- Confirmed there is no existing hosted selected oracle freshness lane and no
  upload of `build/corpus/oracle/`, `build/corpus-reports/`, or
  `build/report-index/`.
- Confirmed existing hosted uploads are limited to dead-code report artifacts
  and coverage HTML; those remain advisory/supporting context, not selected
  oracle freshness proof.
- Audited `.gitignore` and confirmed `build/` and `coverage/` already exclude
  selected oracle, corpus-report, report-index, dead-code, benchmark, and
  coverage generated artifacts from source control.
- Decided Sprint 152 keeps selected oracle freshness as a local-required gate,
  not a hosted CI artifact publication lane.
- Defined the maintained local command sequence:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`
  followed by
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Defined the Day 10 follow-through: add a stable local command surface,
  preferably a Makefile target such as `report-index-oracle-freshness`, while
  leaving hosted workflow files and artifact uploads unchanged unless a command
  alignment issue is discovered.
- Preserved platform/compiler boundaries: local oracle rows record platform and
  compiler context but do not imply hosted CI proof, Linux/macOS/Windows
  parity, package/ABI support, performance claims, or state-of-the-art status.
- Day 10 handoff: implement the local command surface and validate it without
  uploading or committing generated oracle artifacts.

### Day 10: CI And Artifact Policy Implementation

- Created the CI/artifact implementation artifact in
  `artifacts/day10-ci-artifact-implementation.md`.
- Added the local Makefile target `report-index-oracle-freshness`.
- The target depends on `$(LIB)`, regenerates selected combined oracle output
  with
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`,
  then runs
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.
- Left hosted workflow files unchanged. No CI lane was added for selected
  oracle freshness in Sprint 152.
- Left artifact uploads unchanged. No upload was added for
  `build/corpus/oracle/`, `build/corpus-reports/`, or `build/report-index/`.
- Preserved existing hosted dead-code and coverage uploads as advisory or
  supporting context, not selected oracle freshness proof.
- Validated the new local command path with `make report-index-oracle-freshness`.
- Re-ran focused Python/report validation:
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`
  and `python3 tests/test_normalize_report_index.py`.
- Day 11 handoff: align maintainer/report-index documentation with
  `make report-index-oracle-freshness`, the selected local-only artifact
  policy, and the current failure-remediation messages.

### Day 11: Documentation Alignment

- Created the documentation alignment artifact in
  `artifacts/day11-documentation-alignment.md`.
- Updated `README.md` so the QR capability section names
  `make report-index-oracle-freshness` as the selected local oracle/report
  freshness gate and preserves hosted CI, platform, package/ABI, performance,
  and state-of-the-art non-claims.
- Updated `docs/solver_selection.md` so QR and partial-SVD evidence point at
  the selected local oracle freshness gate, while partial-SVD-only oracle
  generation remains a focused local debugging variant.
- Updated `docs/algorithm.md` so QR corpus evidence uses the selected combined
  local gate and treats QR-only oracle runs as focused debug variants.
- Updated `docs/maintainer_guide.md` with a selected oracle freshness gate
  section documenting command, generated paths, `52` selected rows, row-count
  split, selected failure classes, local-only artifact policy, and non-claims.
- Updated QR and partial-SVD maintenance guidance to include
  `make report-index-oracle-freshness` and to state that QR-only or
  partial-SVD-only runs do not satisfy the selected combined row-count policy
  by themselves.
- Updated normalized report-index workflow guidance with the preferred
  selected oracle target and selected oracle error classes.
- Updated `tests/corpus/schemas/report_index_fields.md` with the selected
  oracle freshness gate contract and diagnostic identifiers.
- Updated `tests/corpus/manifests/report_families.tsv` so both selected oracle
  contract rows name `make report-index-oracle-freshness` as the maintained
  generator command.
- Searched active docs for stale oracle command wording. Remaining QR-only and
  partial-SVD-only command references are intentional and labeled as focused
  debugging variants.
- Re-ran `make report-index-oracle-freshness`,
  `python3 scripts/validate_corpus_schema.py`,
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`,
  and `python3 tests/test_normalize_report_index.py`.
- Day 12 handoff: run the integrated regeneration proof using the selected
  target, validate normalized index behavior, and record current generated
  artifact policy evidence.

### Day 12: Integrated Regeneration And Policy Validation

- Created the integrated regeneration validation artifact in
  `artifacts/day12-integrated-regeneration-validation.md`.
- Ran the selected oracle freshness target:
  `make report-index-oracle-freshness`.
- Regenerated the selected local oracle outputs under ignored build paths:
  `build/corpus/oracle/corpus.oracle.tsv`,
  `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, and
  `build/corpus-reports/manifest.txt`.
- Confirmed the selected oracle output contains `52` rows: `23` QR
  solver-backed rows, `26` partial-SVD solver-backed rows, and `3`
  generated-reference rows.
- Regenerated and checked the combined corpus/oracle normalized index:
  `128` rows total, including `74` corpus rows and `54` oracle rows.
- Ran required selected oracle freshness and strict selected oracle freshness;
  both exited `0` with `0` freshness errors and the expected row-level
  `generated_present_unchecked` warnings.
- Ran advisory/deferred freshness checks for coverage, dead-code, package, and
  runtime-backend families; all exited `0` while preserving advisory and
  source-controlled classifications.
- Ran focused validation:
  `python3 scripts/validate_corpus_schema.py`,
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`,
  and `python3 tests/test_normalize_report_index.py`.
- Searched active documentation and report-family metadata for stale selected
  oracle command wording. Remaining QR-only and partial-SVD-only command
  references are intentional focused debugging variants.
- Ran `git diff --check`.
- Day 13 handoff: run final quality and residual review for the Sprint 152
  changed surfaces, clean generated Python caches, and confirm no ignored
  generated report output is accidentally staged as release proof.

### Day 13: Full Quality Gate And Residual Review

- Created the quality-gate and residual-review artifact in
  `artifacts/day13-quality-gate-residual-review.md`.
- Checked for modified or untracked `.c` and `.h` files with
  `git diff --name-only -- '*.c' '*.h'` and
  `git ls-files --others --exclude-standard -- '*.c' '*.h'`; both were empty.
- Used the focused Python/report/documentation gate because Sprint 152 changed
  Makefile, Python report-index code, Python tests, documentation,
  report-family metadata, and planning artifacts only.
- Ran `make report-index-oracle-freshness`; it passed and reported selected
  oracle freshness OK.
- Ran `python3 scripts/validate_corpus_schema.py`; it passed.
- Ran `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`
  and `python3 tests/test_normalize_report_index.py`; both passed.
- Regenerated and checked the combined corpus/oracle normalized index with
  `128` rows.
- Ran strict selected oracle freshness; it exited `0` with no freshness errors
  and the expected `52` row-level `generated_present_unchecked` warnings.
- Ran advisory/source-controlled freshness checks for coverage, dead-code,
  package, and runtime-backend families; they exited `0` with `11`
  advisory/source-controlled rows.
- Searched active docs and report-family metadata for stale selected oracle
  command wording. Remaining QR-only and partial-SVD-only command references
  are intentional focused debugging variants, and no stale `105 rows`
  reference remains.
- Removed Python `__pycache__` output and ran `git diff --check`.
- Recorded residual generated-family owners for benchmark, sentinel, guardrail,
  dead-code, coverage, report-index missing-output visibility, hosted CI,
  package, and runtime-backend governance.
- Day 14 handoff: finalize Sprint 152 notes and artifacts, prepare the Sprint
  153 ABI/package handoff, rerun the final lightweight checks, and confirm no
  ignored generated output is staged as release proof.

### Day 14: Closeout And Sprint 153 Handoff

- Created the closeout artifact in `artifacts/day14-closeout-summary.md`.
- Created the Sprint 153 ABI/package handoff artifact in
  `artifacts/sprint153-abi-package-handoff.md`.
- Finalized Sprint 152 retrospective inputs: selected generated freshness gate,
  policy decisions, validation status, claim changes, residuals, follow-up
  risks, and non-claims.
- Re-ran final lightweight validation:
  `make report-index-oracle-freshness`,
  `python3 scripts/validate_corpus_schema.py`,
  `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --output build/report-index/normalized-index.tsv`,
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`,
  `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness`,
  and
  `python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --family runtime_backend --check-freshness`.
- Confirmed final selected oracle generated output has `52` rows: `23` QR,
  `26` partial-SVD, and `3` generated-reference rows.
- Confirmed the final normalized corpus/oracle index has `128` rows:
  `74` corpus rows and `54` oracle rows.
- Rechecked active documentation, report-family metadata, and Sprint 152
  artifacts for stale selected oracle wording and stale row-count wording.
  Remaining QR-only and partial-SVD-only command references are intentional
  focused debugging variants.
- Confirmed `build/` generated report output is ignored, not staged, and not
  described as release proof.
- Removed Python `__pycache__` output after validation and ran
  `git diff --check`.
- Sprint 152 is ready for retrospective: selected generated freshness
  publication is closed locally, residual generated families are assigned, and
  Sprint 153 has a shared-library ABI product-decision handoff.
