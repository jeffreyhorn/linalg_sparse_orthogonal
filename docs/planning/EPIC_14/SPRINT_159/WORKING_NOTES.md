# Sprint 159 Working Notes

## Goal

Sprint 159 promotes selected local-only generated oracle and comparison
freshness checks into reviewed hosted evidence without broadening solver,
platform, package, performance, external-library parity, ABI, or
state-of-the-art claims.

## Starting Evidence

- Sprint 157 evidence contract template `T157-02` defines hosted generated
  report promotion as selected QR, partial-SVD, oracle, and comparison
  generated freshness evidence with explicit support-tier boundaries.
- Sprint 158 closed generated API HTML publication with a local-only
  no-commit policy and `make docs-check`; hosted report promotion is separate
  from Doxygen/API HTML publication.
- Current Makefile freshness gates are:
  - `make report-index-oracle-freshness`
  - `make report-index-comparison-freshness`
- Current generator and normalizer scripts are:
  - `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`
  - `python3 scripts/run_external_comparison.py --target qr-minnorm`
  - `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`
  - `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`
- Current report-family metadata keeps `oracle` and `comparison` generated
  rows local-only until a later sprint explicitly changes hosted publication
  and support-tier policy.

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-159` |
| Starting commit | `b53810ba514b030a0cbe6153cd92e9760a51b5b3` |
| Starting commit summary | `b53810ba Merge pull request #176 from jeffreyhorn/sprint-158` |
| Upstream state | created from current `master` after PR #176 merge |
| Initial Day 1 scope note | The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but the Sprint 159 section exists in `docs/planning/EPIC_14/PROJECT_PLAN.md`; the requested Sprint 159 plan/artifacts path remains under `docs/planning/EPIC_14/SPRINT_159/`. |

## Candidate Freshness Surface

| Surface | Current command or owner | Current support tier | Day 1 interpretation |
| --- | --- | --- | --- |
| Combined QR and partial-SVD oracle freshness | `make report-index-oracle-freshness` | local-only generated | Candidate for hosted promotion only after family selection, runtime budget, artifact policy, and strict row semantics. |
| QR minimum-norm comparison freshness | `make report-index-comparison-freshness` | local-only generated | Candidate for hosted promotion if runtime and artifact semantics remain narrow and deterministic. |
| Oracle generator | `scripts/run_corpus_oracle.py` | local generator | Source owner for selected oracle rows; not pass evidence without a freshness gate. |
| Comparison generator | `scripts/run_external_comparison.py` | local generator | Source owner for selected comparison rows; not broad external-library parity. |
| Report normalizer | `scripts/normalize_report_index.py` | local/generated report index | Candidate semantics owner for missing/stale/failing selected rows. |
| Report-family metadata | `tests/corpus/manifests/report_families.tsv` | source-controlled advisory metadata | Defines row meaning, support tier, claim scope, artifact pattern, and non-claims; not a fresh pass by itself. |
| CI workflows | `.github/workflows/*.yml` | reviewed/supplemental hosted lanes by file and job | Future owner for hosted execution, artifact upload, and support-tier wording. |
| Maintainer and public docs | `docs/maintainer_guide.md`, `tests/corpus/README.md`, `README.md`, `docs/solver_selection.md` | source-controlled claim guidance | Must move only after hosted evidence and row semantics are selected. |

## Explicit Non-Goals

- Do not promote every generated report family to reviewed hosted evidence.
- Do not treat ignored local files under `build/`, `coverage/`, or
  `docs/api/` as source-controlled pass evidence.
- Do not broaden QR or partial-SVD wording into broad solver correctness,
  LAPACK/NumPy/SciPy/SuiteSparse/Eigen parity, broad external-library
  comparison, or state-of-the-art claims.
- Do not use oracle/comparison freshness as package, ABI, shared-library,
  dynamic-loader, package-manager, platform, or performance proof.
- Do not change generated API HTML policy from Sprint 158.
- Do not add hosted CI work before family selection, runtime budget,
  artifact-retention policy, and selected-row failure semantics are explicit.

## Stop Conditions

- A proposed hosted row cannot be tied to a concrete report family, command,
  artifact path, claim scope, support tier, and non-claim boundary.
- Hosted runtime is unknown or likely unstable.
- Artifact retention, summary output, or failure diagnostics are ambiguous.
- Advisory/local-only families are accidentally promoted by workflow naming,
  report metadata, or public documentation.
- Stale, missing, skipped, or failing selected rows can pass silently.
- Public docs imply broader claims than the selected hosted row evidence
  supports.
- C/header files change without `make format && make lint && make test`.
- Documentation-only changes fail `git diff --check` or whitespace hygiene.

## Daily Log

### Day 1: Promotion Intake

- Re-read the Sprint 159 section of
  `docs/planning/EPIC_14/PROJECT_PLAN.md`.
- Confirmed the prompt path mismatch: `EPIC_12` lines 96-130 are Sprint 139,
  while the requested Sprint 159 title appears in Epic 14.
- Created Sprint 159 working notes and artifact directory structure under
  `docs/planning/EPIC_14/SPRINT_159/`.
- Recorded branch baseline: `sprint-159` at
  `b53810ba514b030a0cbe6153cd92e9760a51b5b3`, created from current `master`
  after PR #176.
- Reviewed Sprint 157 hosted-report evidence contract and Sprint 158
  generated API HTML closeout/handoff.
- Inventoried candidate promotion surfaces: Makefile oracle/comparison
  freshness targets, generator scripts, normalizer script, report-family
  metadata, CI workflow artifact owners, and public/maintainer docs.
- Established non-goals and stop conditions for broad solver, parity,
  platform, package, ABI, performance, generated-API, and state-of-the-art
  claims.
- Day 2 handoff: classify each candidate family as reviewed-hosted candidate,
  supplemental-hosted, advisory-local, or deferred, and tie selected families
  to exact claim/support-tier rows.

### Day 2: Family Selection

- Reviewed current oracle, comparison, QR, partial-SVD, corpus, and
  report-index rows in `tests/corpus/manifests/report_families.tsv`,
  `docs/maintainer_guide.md`, `tests/corpus/README.md`,
  `docs/solver_selection.md`, and `README.md`.
- Selected these reviewed-hosted candidates for runtime-budget review:
  - `oracle/solver_backed` QR selected rows from
    `make report-index-oracle-freshness`;
  - `oracle/solver_backed` partial-SVD selected rows from
    `make report-index-oracle-freshness`;
  - `comparison/qr_minnorm` selected rows from
    `make report-index-comparison-freshness`.
- Classified `oracle/generated_reference` as supplemental-hosted candidate:
  it may be uploaded or summarized as context for solver-backed rows, but
  should not be the primary public claim surface by itself.
- Classified broad `report_index/missing_generated` output and all
  non-selected benchmark, sentinel, guardrail, coverage, dead-code, package,
  runtime-backend, documentation, CI, and corpus metadata families as
  advisory-local or deferred.
- Mapped selected candidates to current claim wording: QR fixture-local
  rank/nullity/nullspace/minimum-norm evidence, partial-SVD fixture-local
  clustered/repeated/rank-deficient/projector/sparse-low-rank/fail-closed
  evidence, and one QR minimum-norm generated comparison.
- Recorded minimum hosted output requirements for selected oracle and
  comparison rows, including generated TSVs, manifests, skip/dependency
  summaries, normalized freshness diagnostics, and deterministic CI summaries.
- Day 3 handoff: measure local runtime for the selected candidate commands,
  confirm output size, and set hosted timeout and retention expectations
  before editing CI.

### Day 3: Runtime Plan

- Defined the runtime measurement matrix for selected hosted candidates
  S159-H01, S159-H02, S159-H03, and supplemental S159-S01.
- Split measurement into cold, warm, generator-only, normalizer-only, and
  selected Make-gate runs so Day 4 can separate build cost from report
  generation and freshness-check cost.
- Recorded the exact commands to measure:
  - `make report-index-oracle-freshness`
  - `make report-index-comparison-freshness`
  - `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`
  - `python3 scripts/run_external_comparison.py --target qr-minnorm`
  - `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`
  - `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`
- Defined output inventories for oracle and comparison artifact paths, summary
  fields, and size/count measurements.
- Drafted hosted timeout expectations, rerun policy, and stop conditions:
  no CI edits until selected gates complete locally with stable runtime,
  bounded artifact size, deterministic summaries, and clear stale/missing/
  skip/fail diagnostics.
- Day 4 handoff: execute the measurement matrix, capture runtime and output
  size, and approve or demote each selected candidate before CI work begins.

### Day 4: Runtime Budget Evidence

- Ran selected oracle and comparison freshness measurements on `sprint-159`
  at commit `b53810ba514b030a0cbe6153cd92e9760a51b5b3` on `darwin-x86_64`.
- Measured oracle freshness:
  - cold gate: `26.22s` real
  - warm gates: `6.65s` and `5.67s` real
  - generator-only: `4.56s` real
  - normalizer-only: `0.26s` real
- Measured comparison freshness:
  - cold gate: `21.94s` real
  - warm gates: `1.94s` and `1.77s` real
  - generator-only: `1.03s` real
  - normalizer-only: `0.26s` real
- Captured generated output inventory:
  - oracle outputs: `53`, `54`, `2`, and `16` lines for oracle TSV, report
    index, skips, and manifest; total measured size `125595` bytes.
  - comparison outputs: `5`, `5`, `5`, `7`, `36`, and `24` lines for
    project observations, baseline observations, dependency status, study,
    summary, and manifest; total measured size `16836` bytes.
- Confirmed selected row counts:
  - oracle `comparison_status=pass`: `52`
  - oracle solver-family counts: `qr=23`, `partial_svd=26`, `unknown=3`
  - comparison `status=pass`: `6`
- Recorded dependency behavior: source-controlled helper and Python baseline
  passed; optional NumPy/SciPy package baselines remained deferred and are not
  pass evidence.
- Set Day 4 hosted timeout recommendations: `10 minutes` each for oracle and
  comparison jobs, or `15 minutes` for a combined selected freshness job.
- Kept S159-H01, S159-H02, S159-H03, and supplemental S159-S01 eligible for
  hosted CI design; no selected candidate was demoted on runtime or artifact
  size grounds.
- Day 5 handoff: design the hosted freshness lane using selected rows only,
  serial execution, scoped artifacts, and explicit support-tier wording.

### Day 5: CI Surface Design

- Reviewed `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml` plus the current support-tier wording in
  README, maintainer guide, solver selection, and corpus docs.
- Selected `.github/workflows/ci.yml` as the Day 6 implementation target
  because Linux is the enforced source-of-truth reviewed baseline and the
  selected gates are maintained Makefile/report-index commands.
- Rejected macOS/Windows workflow changes for Sprint 159 Day 6:
  - macOS install/package lanes should remain static-first package proof, not
    generated report promotion.
  - Windows remains CMake-first and should not inherit Unix Makefile
    report-index proof or imply Windows Makefile parity.
- Designed one new serialized Linux job named
  `Linux reviewed hosted oracle/comparison freshness` with `timeout-minutes:
  15`.
- Designed selected steps:
  - checkout;
  - run `make report-index-oracle-freshness`;
  - run `make report-index-comparison-freshness`;
  - upload selected oracle, comparison, and diagnostics artifacts with 7-day
    retention under scoped artifact names.
- Defined PR failure semantics:
  stale/missing/failing/partial/row-count mismatch or required dependency
  failure is a product failure; optional NumPy/SciPy defers are not pass
  evidence; runner service outages may be retried only when selected rows were
  not produced.
- Day 6 handoff: implement the Linux job only, preserve staged/support-tier
  comments, do not promote broad report-index freshness, and keep artifacts
  scoped to selected oracle and QR minimum-norm comparison rows.

### Day 6: Hosted Freshness Implementation

- Implemented the selected hosted freshness lane in
  `.github/workflows/ci.yml`.
- Added the top-level Linux reviewed baseline comment entry for the selected
  hosted oracle/comparison freshness path.
- Added one serialized Linux job:
  `generated-report-freshness`, named
  `Linux reviewed hosted oracle/comparison freshness`, with
  `timeout-minutes: 15`.
- Wired selected commands exactly through maintained Make targets:
  - `make report-index-oracle-freshness`
  - `make report-index-comparison-freshness`
- Added scoped artifact upload with 7-day retention:
  `sprint159-hosted-oracle-comparison-freshness`.
- Kept macOS and Windows workflows unchanged so Sprint 159 does not imply
  macOS/Windows report-index parity, Windows Makefile parity, Windows
  `pkg-config` execution parity, package/ABI proof, or broad platform proof.
- Preserved non-promoted generated families by adding workflow comments that
  limit reviewed hosted evidence to selected QR/partial-SVD oracle rows and
  selected QR minimum-norm comparison rows.
- Day 7 handoff: define the detailed artifact publication policy and decide
  whether Day 6's combined artifact should remain combined or split into
  separate oracle/comparison artifact uploads for reviewer clarity.

### Day 7: Artifact Publication Design

- Reviewed the Day 6 hosted job and artifact upload shape.
- Decided Day 8 should split the combined artifact into two scoped uploads:
  - `sprint159-oracle-freshness`
  - `sprint159-comparison-qr-minnorm`
- Designed deterministic hosted summary steps for oracle and comparison
  outputs using shell/Python one-liners against generated TSVs and manifests.
- Set artifact retention at 7 days for both selected artifact groups.
- Chose strict missing-file behavior for split uploads:
  `if-no-files-found: error`, because each upload runs after its corresponding
  selected freshness command.
- Defined row-state summary expectations:
  - passing oracle summary prints total rows, QR rows, partial-SVD rows,
    generated-reference rows, and pass count.
  - passing comparison summary prints selected row count, pass count, and
    dependency pass/defer counts.
  - missing, stale, failing, partial, row-set mismatch, or required dependency
    failures remain command failures before upload.
  - optional NumPy/SciPy defers are printed as context and are not pass
    evidence.
- Day 8 handoff: update `.github/workflows/ci.yml` to add deterministic
  summaries, split uploads, and strict missing-file handling without changing
  selected commands or support-tier scope.

### Day 8: Artifact Publication Implementation

- Updated `.github/workflows/ci.yml` to replace the Day 6 combined artifact
  upload with split selected artifact uploads:
  - `sprint159-oracle-freshness`
  - `sprint159-comparison-qr-minnorm`
- Added deterministic hosted oracle summary output after
  `make report-index-oracle-freshness`.
- Added deterministic hosted QR minimum-norm comparison summary output after
  `make report-index-comparison-freshness`.
- Set `retention-days: 7` and `if-no-files-found: error` on both selected
  artifact uploads.
- Preserved `if: always()` on uploads so generated diagnostics are published
  when files exist after a failure.
- Kept selected commands unchanged and did not upload broad
  `build/report-index/normalized-index.tsv`.
- Confirmed workflow YAML parses and documentation/workflow whitespace checks
  pass.
- Day 9 handoff: audit normalizer semantics for selected hosted rows,
  especially oracle generated-present warning wording during successful strict
  selected freshness checks.

### Day 9: Normalizer Semantics Audit

- Audited `scripts/normalize_report_index.py` selected oracle and comparison
  freshness behavior.
- Confirmed required selected oracle missing, stale, failed, row-count,
  solver-family, and fixture-key mismatches already fail the normalizer.
- Confirmed required selected comparison missing and stale generated families
  fail through generic required-family freshness handling.
- Confirmed selected comparison row-set mismatch and selected non-pass status
  are implemented in `selected_comparison_policy_diagnostics()`, but focused
  test coverage should be added before Day 10 relies on those semantics as a
  hosted claim gate.
- Identified the main ambiguity: successful selected oracle freshness can
  still print generic strict-freshness warnings for current-commit generated
  rows, which makes reviewed hosted pass evidence look partially unchecked.
- Identified secondary wording ambiguity: selected comparison skip/defer
  diagnostics use `comparison_optional_rows`, which can blur selected proof
  rows with optional NumPy/SciPy dependency rows.
- Recorded promoted-row semantics and Day 10 test/fixture needs in
  `artifacts/day9-normalizer-semantics-audit.md`.
- Day 10 handoff: add focused selected comparison normalizer tests, tighten
  selected current-commit generated-row diagnostics, and preserve advisory
  local-only/source-controlled behavior for unpromoted families.

### Day 10: Normalizer Semantics Implementation

- Updated `scripts/normalize_report_index.py` so selected required
  current-commit oracle and comparison generated rows report as `fresh`
  instead of generic strict `generated_present_unchecked` warnings.
- Preserved stale selected row behavior: stale source commits still produce
  required-family `freshness: error` diagnostics with remediation commands.
- Changed selected comparison skip/defer wording from
  `comparison_optional_rows` to `comparison_selected_rows` so selected proof
  rows remain distinct from optional NumPy/SciPy dependency context.
- Added synthetic selected comparison fixtures in
  `tests/test_normalize_report_index.py`.
- Added focused selected comparison tests for complete row sets, missing rows,
  unexpected rows, duplicate row IDs, stale rows, failed rows, and deferred
  selected rows.
- Strengthened selected oracle pass coverage to assert successful required
  freshness no longer emits unchecked-warning wording.
- Validated with:
  - `python3 tests/test_normalize_report_index.py`
  - `make report-index-oracle-freshness`
  - `make report-index-comparison-freshness`
  - `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`
  - `make lint`
  - `git diff --check -- scripts/normalize_report_index.py tests/test_normalize_report_index.py .github/workflows/ci.yml docs/planning/EPIC_14/SPRINT_159`
- Day 11 handoff: align maintainer and public documentation with the tightened
  selected hosted freshness semantics without broadening support-tier,
  package, ABI, platform, performance, external-parity, or state-of-the-art
  claims.

### Day 11: Documentation Alignment

- Updated `README.md` command guidance to list both selected freshness gates
  and identify them as mirrored by reviewed Linux hosted CI.
- Updated README QR evidence wording so the selected hosted lane is described
  as reviewed Linux execution for selected oracle/comparison artifacts, not as
  broad QR parity or platform proof.
- Updated `docs/maintainer_guide.md` selected oracle and comparison freshness
  sections to describe Sprint 159 split hosted artifacts and current selected
  row semantics.
- Updated normalized report-index interpretation in the maintainer guide so
  current-commit selected rows are read as `fresh`, while stale, missing,
  failing, skipped, deferred, duplicate, unexpected, or incomplete selected
  rows remain non-pass states.
- Updated `tests/corpus/README.md` to clarify that Sprint 159 hosted CI covers
  only selected required oracle and QR minimum-norm comparison rows, not broad
  report-index freshness or every local-only family.
- Updated `docs/solver_selection.md` QR and partial-SVD evidence boundaries to
  mention the reviewed Linux hosted report-freshness lane without converting
  it into broad platform, package, performance, external-parity, or
  state-of-the-art proof.
- Wrote the Sprint 160 QR comparison handoff in
  `artifacts/day11-documentation-alignment.md`.
- Day 12 handoff: run the selected freshness commands, focused normalizer
  tests, documentation whitespace checks, and any lightweight docs validation
  needed after the public/maintainer wording changes.

### Day 12: Local Validation

- Ran `make report-index-oracle-freshness`; selected oracle freshness passed,
  normalized `54` rows, and current selected generated rows reported `fresh`.
- Ran `make report-index-comparison-freshness`; selected QR minimum-norm
  comparison freshness passed, normalized `7` rows, and six selected generated
  rows reported `fresh`.
- Ran `python3 tests/test_normalize_report_index.py`; focused normalizer tests
  passed, including selected comparison complete/missing/stale/duplicate/
  unexpected/fail/defer coverage.
- Ran `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`; Python syntax compilation passed.
- Ran `make docs-check`; Doxygen generation and API docs coverage passed.
- Ran `git diff --check` and trailing-whitespace scans for changed Sprint 159
  workflow, docs, scripts, and tests; hygiene checks passed.
- Confirmed no `.c` or `.h` files are modified, so the required
  `make format && make lint && make test` C/header gate is not required for
  Day 12.
- Recorded validation details in
  `artifacts/day12-local-validation.md`.
- Day 13 handoff: review hosted readiness in `.github/workflows/ci.yml`,
  including job naming, timeout, summaries, artifact paths, and the boundary
  between local-only row metadata and reviewed Linux hosted execution.

### Day 13: Hosted Readiness Review

- Re-read the `generated-report-freshness` workflow job as a single reviewer
  path.
- Confirmed the hosted readiness shape:
  - Linux-only reviewed job named
    `Linux reviewed hosted oracle/comparison freshness`;
  - `timeout-minutes: 15`;
  - selected oracle command: `make report-index-oracle-freshness`;
  - selected comparison command: `make report-index-comparison-freshness`;
  - split artifact uploads:
    `sprint159-oracle-freshness` and
    `sprint159-comparison-qr-minnorm`;
  - `retention-days: 7`;
  - `if-no-files-found: error`.
- Confirmed the workflow does not upload broad
  `build/report-index/normalized-index.tsv`.
- Confirmed local Day 12 generated files exist for every hosted upload path.
- Parsed `.github/workflows/ci.yml` with Ruby YAML successfully.
- Recorded rerun expectations:
  runner service outages may be rerun, but selected generator, normalizer,
  summary, or artifact failures should be treated as product failures unless
  logs show an infrastructure-only root cause.
- Confirmed local-only/advisory boundaries remain explicit in README,
  maintainer guide, corpus README, solver-selection docs, and Sprint 159
  artifacts.
- Finalized the Sprint 160 QR comparison handoff in
  `artifacts/day13-hosted-readiness.md`.
- Day 14 handoff: prepare closeout and retrospective inputs, summarize the
  promoted hosted evidence surface, list final validations, and ensure all
  Sprint 159 artifacts are complete before retrospective creation.

### Day 14: Closeout

- Confirmed all Day 1-13 artifacts and the Sprint 159 plan/working notes are
  present under `docs/planning/EPIC_14/SPRINT_159/`.
- Re-ran final targeted validation:
  - `make report-index-oracle-freshness`
  - `make report-index-comparison-freshness`
  - `python3 tests/test_normalize_report_index.py`
  - `make docs-check`
  - `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py`
  - Ruby YAML parse for `.github/workflows/ci.yml`
  - `git diff --check`
  - trailing-whitespace scan
- Confirmed selected oracle freshness still passes with `54` normalized rows
  and current selected generated rows reported as `fresh`.
- Confirmed selected comparison freshness still passes with `7` normalized
  rows and six selected generated comparison rows reported as `fresh`.
- Confirmed `docs-check` passes with `18` checked-in public headers, `18`
  generated reference pages, and `18` generated source pages.
- Confirmed no `.c` or `.h` files are modified, so the required
  `make format && make lint && make test` C/header gate is not required for
  Day 14; Day 10 already ran `make lint` successfully for extra confidence.
- Recorded promoted row, unpromoted row, final validation, claim wording, and
  retrospective inputs in `artifacts/day14-closeout.md`.
- Sprint 159 closeout state: ready for retrospective drafting after the user
  asks for it.
