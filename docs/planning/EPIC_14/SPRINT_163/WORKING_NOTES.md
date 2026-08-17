# Sprint 163 Working Notes

## Sprint Goal

Publish a methodology-bound performance/report artifact for selected canonical
benchmark and sentinel rows while preserving the project's non-superiority and
local-evidence boundaries.

## Source Artifact Note

The Sprint 163 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but
the current merged planning source for Sprint 163 is
`docs/planning/EPIC_14/PROJECT_PLAN.md` in the section "Sprint 163:
Methodology-Bound Performance Publication".

## Branch Baseline

- Branch: `sprint-163`
- Starting commit: `5f0b0027`
- Sprint 162 handoff: Windows package validation remains static-first and
  CMake-first; Windows Makefile install parity, Windows `pkg-config` execution
  parity, package-manager support, shared-library support, dynamic ABI
  compatibility, runtime-loader behavior, broad Windows parity, and performance
  superiority remain explicit non-claims.
- Sprint 163 must keep package proof separate from performance proof.

## Current Performance Evidence Surfaces

| Surface | Owner | Current Role | Sprint 163 Implication |
| --- | --- | --- | --- |
| Benchmark build surface | `Makefile`, `benchmarks/*.c` | `make bench-build` and `make tooling-build` compile benchmark/example binaries without running long workloads. | Compile health can support publication readiness, but does not provide timing evidence. |
| Full benchmark surface | `make bench` | Runs the broad benchmark set. | Too broad for a bounded Sprint 163 publication unless a narrow subset is explicitly selected. |
| Fast runtime lane | `make bench-fast` | Runs a short subset of scaling, fill-in, COLAMD, AMD-QG, and reorder measurements. | Useful as runtime confidence context, not a portable performance claim. |
| Canonical report | `make bench-canonical-report`, `scripts/bench_canonical_report.sh` | Emits threshold-free local CSV rows and `index.tsv` / `manifest.txt` metadata for `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse`. | Primary candidate for methodology-bound publication because it already records commit, branch, platform, compiler, build mode, thread count, command, and artifact identity. |
| Performance sentinels | `make performance-sentinels`, `scripts/performance_sentinels.sh` | Emits `sentinels.tsv`, `manifest.txt`, wall-check output, and threshold-free Cholesky CSC / LDLT KKT rows. | Strong candidate for publication if S5 thresholded rows remain separate from S2/S3 threshold-free context. |
| Wall-check gate | `make wall-check`, `scripts/wall_check.sh`, `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` | Narrow local pass/fail timing regression gate. | Any publication must identify this as local threshold evidence, not broad speedup proof. |
| Large-matrix guardrails | `make large-matrix-guardrails`, `scripts/large_matrix_guardrails.sh` | Structural large-matrix report lanes with explicit pass/skip/fail rows. | Adjacent evidence; not selected for Day 1 unless later methodology work chooses it. |
| Normalized report index | `scripts/normalize_report_index.py` and report freshness targets | Maintainer navigation and freshness aid across generated report families. | Can link selected report rows, but must not turn generated rows into release proof. |
| Public benchmark docs | `README.md`, `benchmarks/README.md` | State benchmark rows are branch-local measurements and not portable performance guarantees. | Docs already contain the non-overclaiming language Sprint 163 should preserve. |
| CI lanes | `.github/workflows/ci.yml` and platform workflows | Provide supplemental runtime, report-freshness, and platform confidence lanes. | CI timing should be treated as environment-bound evidence, not portable superiority evidence. |

## Explicit Non-Goals

Sprint 163 does not claim or attempt to prove:

- portable performance superiority;
- state-of-the-art status;
- broad platform performance parity;
- package evidence reuse as performance evidence;
- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- runtime-backend superiority;
- external-library parity;
- OpenMP speedup portability;
- broad report-index freshness as release proof;
- benchmark rows as release proof without methodology fields and claim
  boundaries;
- Windows Makefile install parity;
- Windows `pkg-config` execution parity.

## Working Assumptions

- Generated benchmark artifacts stay under ignored `build/` paths unless a
  later day explicitly selects source-controlled publication metadata.
- A single benchmark run can be published only as local methodology-bound
  evidence, not as a portable result.
- Hard gates and threshold-free rows need different support-tier and
  claim-boundary language.
- The selected publication surface should be small enough to regenerate locally
  and in CI without turning Sprint 163 into broad benchmark governance work.
- Existing package/static-first guards are separate evidence and should not be
  cited as performance proof.

## Stop Conditions

Stop and revise before proceeding if a change:

- describes benchmark output as portable or state-of-the-art performance;
- uses package, install, ABI, or Windows package proof as performance proof;
- collapses S5 `wall-check` gate rows with S2/S3 threshold-free sentinel rows;
- makes broad generated report freshness a release claim;
- weakens Sprint 162 retained package non-claim wording;
- modifies C or header files without running the required code quality checks;
- adds a long-running benchmark gate to mandatory CI without a narrow
  methodology decision.

## Daily Log

### Day 1: Sprint Intake And Performance Surface Inventory

- Re-read the Sprint 163 Epic 14 project-plan section and recorded the prompt
  path mismatch as a source artifact note.
- Reviewed Sprint 162 retained package non-claim handoff and separated package
  proof from Sprint 163 performance proof.
- Inventoried benchmark, sentinel, wall-check, report-index, generated report,
  documentation, CI, and package-boundary surfaces.
- Recorded explicit non-goals, assumptions, and stop conditions.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2: Canonical Row Candidate Inventory

- Inspected `Makefile`, `benchmarks/README.md`,
  `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`,
  `scripts/wall_check.sh`, report-index targets, and CI references for
  candidate publication rows.
- Built the source-backed candidate row register in
  `artifacts/day2-row-inventory.md`.
- Identified primary publication candidates:
  `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`,
  `bench_eigs_reuse`, sentinel S5 wall-check rows, sentinel S2 Cholesky CSC
  rows, and sentinel S3 LDLT KKT rows.
- Rejected or deferred rows that are correctness, package, corpus, comparison,
  broad benchmark, generated-freshness, or structural guardrail evidence rather
  than selected performance-publication evidence.
- Recorded blockers for Day 3 selection: canonical raw CSV rows still need a
  methodology contract, S5 needs explicit baseline/threshold framing, and S2/S3
  must stay threshold-free context rather than pass/fail evidence.

### Day 3: Surface Selection

- Selected the narrow Sprint 163 publication surface:
  `make bench-canonical-report` and `make performance-sentinels`.
- Classified the four canonical benchmark artifact rows as published
  threshold-free local measurements, provided Day 4 defines the required
  methodology contract before report or documentation edits.
- Classified sentinel S5 rows as the only selected thresholded local timing
  gate, with baseline and threshold provenance required everywhere they are
  surfaced.
- Classified sentinel S2 and S3 rows as selected threshold-free local context
  rows, not pass/fail evidence.
- Deferred full benchmark runs, fast runtime lanes, large-matrix guardrails,
  report freshness rows, package proof, correctness/corpus rows, and API
  documentation freshness rows out of the Sprint 163 performance-publication
  surface.
- Created `artifacts/day3-surface-selection.md`.

### Day 4: Methodology Contract

- Defined the required methodology fields for published canonical rows,
  published sentinel rows, supplemental rows, advisory rows, and local-only raw
  artifacts.
- Defined row-state semantics for present, missing, stale, malformed, skipped,
  deferred, failed, and threshold-free report rows.
- Distinguished S5 hard threshold timing gates from canonical/S2/S3
  threshold-free methodology reports.
- Recorded variance and repeat semantics: current selected rows are
  single-repeat local rows until a later implementation explicitly adds repeat
  counts or variance fields.
- Wrote public caveat wording that blocks portable performance superiority,
  state-of-the-art, broad platform parity, package-proof reuse, backend
  superiority, and OpenMP speedup claims.
- Created `artifacts/day4-methodology-contract.md`.

### Day 5: Report Schema And Script Gap Analysis

- Compared `scripts/bench_canonical_report.sh`,
  `scripts/performance_sentinels.sh`, `scripts/normalize_report_index.py`,
  `tests/corpus/manifests/report_families.tsv`, `Makefile`,
  `README.md`, and `benchmarks/README.md` against the Day 4 methodology
  contract.
- Confirmed selected report commands already emit the core provenance needed
  for local publication: command, artifact, UTC time, commit, branch, platform,
  compiler, build mode, and thread context.
- Identified canonical report gaps: no explicit `support_tier`,
  `claim_boundary`, row state, repeat count, warmup state, variance state,
  baseline state, threshold state, or local-only caveat fields in `index.tsv`.
- Identified sentinel report gaps: S5 lacks an explicit baseline provenance
  field in row output, and S2/S3 need explicit repeat, warmup, variance, and
  row-state semantics if published beyond current docs.
- Confirmed normalized report-index behavior preserves family-level non-claims
  and separates S5 hard-gate rows from S2/S3 advisory rows.
- Created `artifacts/day5-schema-gap-analysis.md`.

### Day 6: Report Enhancement Implementation I

- Updated `scripts/bench_canonical_report.sh` to append canonical methodology
  fields without renaming or removing existing `index.tsv` columns.
- Added canonical row classification fields: `report_family`, `status`,
  `support_tier`, `claim_boundary`, `baseline`, `threshold`,
  `repeat_semantics`, `warmup`, `variance`, `fixture_or_workload`,
  `matrix_size`, `backend_context`, and `methodology_notes`.
- Strengthened the canonical report manifest with the same methodology and
  non-superiority caveats required by Day 4.
- Preserved unselected benchmark behavior by changing only the selected
  canonical report script and not touching `make bench`, `bench-fast`, or
  exploratory benchmark targets.
- Ran focused checks:
  - `bash -n scripts/bench_canonical_report.sh`
  - `make bench-canonical-report`
  - `python3 scripts/normalize_report_index.py --family benchmark --output build/report-index/normalized-index.tsv`
- Created `artifacts/day6-report-implementation-1.md`.

### Day 7: Report Enhancement Implementation II

- Updated `scripts/performance_sentinels.sh` to append sentinel methodology
  fields without renaming or removing existing `sentinels.tsv` columns.
- Added sentinel methodology fields: `baseline_provenance`,
  `repeat_semantics`, `warmup`, `variance`, and `methodology_notes`.
- Preserved S5 as the thresholded local wall-check gate and S2/S3 as
  threshold-free local backend-context rows.
- Strengthened the sentinel manifest with explicit S5/S2/S3 caveats and
  non-superiority/non-portability wording.
- Updated `scripts/normalize_report_index.py` so normalized benchmark and
  sentinel `configuration` text preserves the new methodology fields.
- Ran focused checks:
  - `bash -n scripts/performance_sentinels.sh`
  - `make performance-sentinels`
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`
  - `python3 tests/test_normalize_report_index.py`
- Created `artifacts/day7-report-implementation-2.md`.

### Day 8: Gate Classification And Publication Policy

- Classified Sprint 163 selected rows into hard timing gates,
  threshold-free reports, generated local-only raw artifacts, advisory
  normalized-index rows, supplemental rows, hosted-only evidence, and deferred
  rows.
- Defined publication eligibility:
  methodology summaries and schema descriptions may be source-controlled, while
  generated timing CSV/TSV/text artifacts stay under ignored `build/` paths
  unless a future sprint explicitly promotes a stable example.
- Defined CI/hosted expectations: selected local commands may be used for
  local validation and artifact capture, but local generated rows cannot be
  cited as hosted proof unless the hosted job itself runs and publishes them.
- Recorded drift, variance, flaky timing, stale output, skip, defer, malformed,
  and failed-row policy.
- Preserved Day 9/Day 10 public documentation updates as planned follow-up
  rather than broadening Day 8 into docs rewrite work.
- Created `artifacts/day8-gate-classification.md`.

### Day 9: Documentation Alignment I

- Updated `benchmarks/README.md` for the selected Sprint 163 report behavior.
- Documented appended canonical methodology fields in `index.tsv`.
- Documented appended sentinel methodology fields in `sentinels.tsv` and
  related manifest context.
- Added explicit interpretation rules for canonical `measurement` rows, S5
  `pass`/`fail`/`skip` rows, S2/S3 `report` rows, `warmup=not_recorded`,
  `variance=not_recorded`, `baseline=n/a`, and `threshold=n/a`.
- Preserved the existing benchmark docs boundary that generated report outputs
  stay under ignored `build/` paths and must be regenerated rather than
  hand-edited.
- Created `artifacts/day9-benchmark-docs.md`.

### Day 10: Documentation Alignment II

- Updated `README.md` performance-summary wording for the selected
  methodology-bound canonical and sentinel report behavior.
- Updated `docs/maintainer_guide.md` to document appended methodology fields,
  row-state interpretation, local-only generated-artifact policy, normalized
  report-index preservation, and package/performance separation.
- Updated `tests/corpus/schemas/report_index_fields.md` so the report-index
  contract notes Sprint 163 benchmark and sentinel methodology fields without
  turning them into pass/fail benchmark proof.
- Scanned public, benchmark, maintainer, and report-index docs for
  non-superiority and unsupported-claim wording.
- Created `artifacts/day10-public-docs.md`.

### Day 11: Selected Benchmark And Sentinel Validation

- Ran selected report and sentinel commands:
  - `make bench-canonical-report`
  - `make performance-sentinels`
- Ran focused script and normalizer checks:
  - `bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh`
  - `python3 tests/test_normalize_report_index.py`
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`
- Verified generated canonical rows:
  - 4 rows
  - no missing methodology fields
  - `status=measurement`
  - `claim_boundary=local_threshold_free`
  - repeat semantics split between `configured_repeat_1` and
    `benchmark_default`
- Verified generated sentinel rows:
  - 19 rows total
  - no missing appended methodology fields
  - 3 S5 rows with `status=pass` and `claim_boundary=local_wall_gate`
  - 8 S2 rows with `status=report` and
    `claim_boundary=local_threshold_free`
  - 8 S3 rows with `status=report` and
    `claim_boundary=local_threshold_free`
- Created `artifacts/day11-selected-validation.md`.

### Day 12: Cross-Surface Validation And Quality Gate

- Re-ran selected benchmark/report/sentinel checks after documentation and
  policy updates.
- Ran report-index and schema checks affected by the sprint:
  - `python3 tests/test_normalize_report_index.py`
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`
  - `python3 scripts/validate_corpus_schema.py`
- Ran package-boundary validation because Day 10 docs mention package/platform
  support boundaries:
  - `bash scripts/static_package_deferral_check.sh`
- Ran documentation unsupported-claim scans across README, benchmark docs,
  maintainer guide, and report-index schema notes; hits are non-claims or
  boundaries.
- Verified selected generated row semantics with a focused Python check:
  canonical rows remain 4 `measurement/local_threshold_free` rows with
  `baseline=n/a` and `threshold=n/a`; sentinel rows remain 3 S5
  `pass/local_wall_gate` rows, 8 S2 `report/local_threshold_free` rows, and 8
  S3 `report/local_threshold_free` rows.
- Confirmed no `.c` or `.h` files changed, so `make format`, `make lint`, and
  `make test` are not required by the Sprint 163 Day 12 changed-file gate.
- Created `artifacts/day12-cross-surface-validation.md`.

### Day 13: Evidence Review

- Traced each supported Sprint 163 performance-publication statement to the
  selected command, generated row surface, documentation surface, and validation
  artifact.
- Confirmed canonical benchmark rows are local-only threshold-free
  `measurement` rows, not portable performance proof.
- Confirmed S5 remains the only hard local wall-check gate, while S2/S3 remain
  threshold-free backend-context `report` rows.
- Confirmed normalized report-index rows preserve methodology fields for
  navigation without becoming hosted, release, package, ABI, platform,
  performance, backend-superiority, or state-of-the-art proof.
- Scanned sensitive wording across README, benchmark docs, maintainer guide,
  report-index schema notes, and selected report scripts; hits are retained
  non-claims or boundary statements.
- Wrote the Sprint 164 API-header handoff, keeping API/header documentation
  work separate from Sprint 163 performance-publication evidence.
- Created `artifacts/day13-evidence-review.md`.

### Day 14: Closeout And Retrospective Prep

- Re-ran the final targeted validation bundle for the changed-file surface:
  - `bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh`
  - `make bench-canonical-report`
  - `make performance-sentinels`
  - `python3 tests/test_normalize_report_index.py`
  - `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`
  - `python3 scripts/validate_corpus_schema.py`
  - `bash scripts/static_package_deferral_check.sh`
- Confirmed the final validation bundle passed, including `26` normalized
  benchmark/sentinel rows and the static package deferral guard.
- Recorded selected performance publication closeout: canonical rows are
  local-only threshold-free measurements, S5 is the hard local wall gate, S2/S3
  are threshold-free backend-context report rows, and normalized report-index
  rows preserve methodology metadata for navigation.
- Recorded retained non-claims for portable performance, state-of-the-art
  evidence, hosted CI proof, package proof, package-manager proof, ABI proof,
  runtime-loader proof, OpenMP speedup proof, backend superiority, external
  library parity, and release proof.
- Prepared retrospective inputs from the complete Day 1 through Day 14 artifact
  set plus working notes.
- Preserved the Sprint 164 API-header handoff from Day 13.
- Created `artifacts/day14-closeout.md`.
