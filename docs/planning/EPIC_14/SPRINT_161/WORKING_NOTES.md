# Sprint 161 Working Notes

## Goal

Publish the first bounded partial-SVD comparison family with subspace-safe
metrics and generated freshness checks.

Sprint 161 implements the Epic 14 Sprint 161 section in
`docs/planning/EPIC_14/PROJECT_PLAN.md`. The user prompt referenced the older
Epic 12 project-plan path, but the current branch and plan place Sprint 161
under Epic 14.

## Branch Baseline

- Branch: `sprint-161`
- Starting commit: `8589a7e6 Merge pull request #178 from jeffreyhorn/sprint-160`
- Starting state: Sprint 160 has landed the descriptor-backed comparison
  runner pattern, selected QR comparison freshness checks, hosted generated
  evidence publication path, and a partial-SVD handoff.

## Starting Evidence

| Surface | Current State | Sprint 161 Implication |
| --- | --- | --- |
| Comparison runner | `scripts/run_external_comparison.py` is descriptor-backed for `qr-minnorm` and `qr-compatible-ls`. | Reuse the same model for one selected partial-SVD target. |
| Dense SVD reference helper | `tests/svd_external_dense_reference.py` provides source-controlled fixtures including `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, and `partial_svd_nonsym_rect10x8_k3`. | Prefer a fixture whose singular values can be compared without raw singular-vector identity. |
| Partial-SVD corpus | `tests/corpus/README.md` documents Sprint 140/Sprint 151 generated partial-SVD rows and selected oracle freshness. | Treat existing corpus/oracle proof as input, not as comparison publication. |
| Report metadata | `tests/corpus/manifests/report_families.tsv` has QR comparison metadata only. | Add partial-SVD comparison metadata before generated rows become selected evidence. |
| Normalizer | `scripts/normalize_report_index.py` expects selected QR comparison rows and `26` selected partial-SVD oracle rows. | Extend selected comparison freshness only after row IDs and semantics are fixed. |
| Runner tests | `tests/test_run_external_comparison.py` covers QR dispatch, files, rows, metadata, and optional dependency context. | Add focused partial-SVD runner tests without weakening QR behavior. |
| Normalizer tests | `tests/test_normalize_report_index.py` covers selected QR comparison row-state failures. | Add complete, missing, unexpected, duplicate, stale, fail, and defer cases for the selected partial-SVD comparison. |
| C proof owners | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, `tests/test_svd_partial_helpers.h`, and `tests/test_svd_partial_shared_helpers.h` own implementation behavior. | Keep C tests unchanged unless implementation behavior or fixture helpers change. |
| Public docs | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, and `tests/corpus/README.md` already constrain partial-SVD corpus claims. | Later docs must add comparison wording without broad parity or state-of-the-art claims. |

## Candidate Target Families

| Candidate | Initial Disposition | Reason |
| --- | --- | --- |
| `partial_svd_diag6_k2` | Selected first comparison target | Source-controlled dense helper exists; diagonal values are deterministic; top-k singular values are stable; raw vector identity can be avoided. |
| `partial_svd_tall_diag_8x5_k3` | Defer for first comparison | Adds tall rectangular shape and zero-row handling after the first path is proven. |
| `partial_svd_nonsym_rect10x8_k3` | Defer for first comparison | Nonsymmetric rectangular behavior needs more careful metric and tolerance review. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Defer for first comparison | Repeated or clustered spectra require subspace-safe semantics and avoid vector-order identity. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Defer for first comparison | Rank-deficient projector behavior is important but broader than the first publication target. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | Defer for first comparison | Sparse output and drop-tolerance behavior need separate claim boundaries. |
| `partial_svd_fail_closed_diag6_k2_v1` | Defer for first comparison | Fail-closed and recovery rows should not define the first passing comparison family. |

## Explicit Non-Goals

Sprint 161 does not claim or attempt to prove:

- broad partial-SVD correctness;
- raw singular-vector identity;
- vector sign, ordering, or orientation identity;
- repeated-spectrum vector ordering;
- convergence-rate superiority;
- partial-result guarantees after fail-closed outcomes;
- broad sparse-output or drop-tolerance optimality;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- platform portability, package-manager, shared-library ABI, release,
  performance, or state-of-the-art evidence.

## Assumptions

- The first comparison target remains `local_only` unless a later hosted
  promotion explicitly earns stronger wording.
- Source-controlled helpers are acceptable as fixture-local references; no
  external package baseline is required for pass evidence.
- Optional dependency rows, if present, are context only and cannot satisfy
  selected freshness.
- Generated comparison artifacts should follow Sprint 160's pattern:
  project observation, baseline observation, dependency status, study rows,
  summary, and manifest.
- Existing C/H proof owners remain untouched unless Sprint 161 changes solver
  behavior or fixture-helper behavior.

## Stop Conditions

Stop and reassess if a proposed row lacks any of:

- exact fixture key;
- stable row ID;
- metric and tolerance;
- support tier;
- artifact path;
- claim scope;
- explicit non-claims.

Also stop if the work requires raw singular-vector identity, treats skip/defer
as pass evidence, broadens public docs into parity claims, or modifies `.c` or
`.h` files without running the full required quality gate.

## Daily Log

### Day 1

- Re-read the Sprint 161 Epic 14 project-plan section.
- Reviewed Sprint 160's partial-SVD handoff and the existing QR comparison
  publication pattern.
- Inventoried current partial-SVD corpus, oracle, comparison runner, metadata,
  normalizer, test, and documentation surfaces.
- Recorded initial candidate family dispositions and non-goals.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2

- Selected `partial_svd_diag6_k2` as the first bounded partial-SVD comparison
  family.
- Probed the source-controlled dense helper:
  `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2`
  returned `OK 2`, `9`, and `6`.
- Deferred tall, nonsymmetric, repeated-spectrum, rank-deficient,
  sparse-output, and fail-closed candidates until the first comparison
  publication path is closed.
- Defined initial row IDs, output path, support tier, claim scope,
  documentation owners, and raw-vector-identity non-claims.
- Created `artifacts/day2-target-selection.md`.

### Day 3

- Finalized the selected comparison metric contract for
  `partial_svd_diag6_k2`.
- Locked the required selected row IDs for project status, baseline status,
  top-k singular values, max singular-value delta, residual norm, U/V
  orthogonality diagnostics, and diagonal projector diagnostics.
- Defined tolerance classes for status, singular values, residuals,
  orthogonality, projector diagnostics, row freshness, and row-state handling.
- Kept convergence, fail-closed, recovery, raw vector identity, vector
  sign/order identity, repeated-spectrum ordering, and external-library parity
  outside the claim surface.
- Created `artifacts/day3-metric-contract.md`.

### Day 4

- Designed the comparison-runner extension for `partial-svd-diag6-k2`.
- Mapped the target descriptor, project probe, baseline probe, study-row
  builder, expected-row validation, self-check, metadata, Makefile freshness,
  runner tests, normalizer tests, and documentation touch points.
- Defined stable output artifacts under
  `build/comparison/partial_svd_diag6_k2/`.
- Recorded the source-controlled `report_families.tsv` row shape and failure
  diagnostic matrix for solver, baseline, parse, tolerance, freshness,
  convergence, and fail-closed cases.
- Created `artifacts/day4-harness-design.md`.

### Day 5

- Implemented `partial-svd-diag6-k2` in
  `scripts/run_external_comparison.py`.
- Added a partial-SVD C probe path that builds the 6x6 diagonal fixture, runs
  `sparse_svd_partial()` with `k=2` and economy vectors, and emits the Day 3
  selected metrics.
- Added SVD dense-helper baseline parsing against
  `tests/svd_external_dense_reference.py`.
- Added partial-SVD study-row generation for the ten selected row IDs while
  preserving existing QR comparison rows.
- Extended `tests/test_run_external_comparison.py` to verify the new target's
  dispatch, output files, row IDs, metrics, metadata, support tier, and
  optional dependency context.
- Created `artifacts/day5-harness-implementation.md`.

### Day 6

- Added the source-controlled `comparison/partial_svd_diag6_k2` row to
  `tests/corpus/manifests/report_families.tsv`.
- Recorded the generator command, artifact pattern, support tier, claim scope,
  owner, and non-claims for the selected partial-SVD comparison family.
- Extended `tests/test_run_external_comparison.py` to verify report-family
  metadata and required source-controlled helper dependency rows.
- Verified optional NumPy/SciPy dependency rows remain `defer` context and do
  not become pass evidence.
- Created `artifacts/day6-expected-rows.md`.

### Day 7

- Designed the focused proof-owner test scope for the selected partial-SVD
  comparison family.
- Kept `tests/test_run_external_comparison.py` as the runner dispatch,
  artifact, metadata, dependency, and row-shape owner.
- Identified `scripts/normalize_report_index.py` and
  `tests/test_normalize_report_index.py` as the Day 8 owners for selected
  comparison freshness promotion.
- Deferred C proof-owner changes because Day 5/Day 6 did not modify `.c` or
  `.h` solver behavior or fixture helpers.
- Defined valid, stale, missing, unexpected, duplicate, skipped, deferred,
  tolerance-failing, malformed, and valid-row test cases.
- Created `artifacts/day7-test-design.md`.

### Day 8

- Expanded selected comparison freshness in
  `scripts/normalize_report_index.py` to include the ten
  `partial_svd_diag6_k2` comparison rows and the
  `build/comparison/partial_svd_diag6_k2/study.tsv` artifact.
- Updated `tests/test_normalize_report_index.py` to synthesize selected QR
  plus partial-SVD comparison rows.
- Added focused row-state coverage for missing, unexpected, duplicate, stale,
  fail, defer, and skip behavior over the expanded selected row set.
- Updated `make report-index-comparison-freshness` to regenerate
  `partial-svd-diag6-k2` before checking selected comparison freshness.
- Verified `make report-index-comparison-freshness` passes locally with the
  expanded selected comparison row set.
- Created `artifacts/day8-focused-tests.md`.

### Day 9

- Designed the normalized row and freshness interpretation for the selected
  `partial_svd_diag6_k2` comparison family.
- Classified the family as `generated_local` and `local_only`, with no hosted,
  package, ABI, platform, performance, release, external parity, or
  state-of-the-art claim.
- Defined reviewer inspection expectations for `study.tsv`, `summary.md`,
  `manifest.tsv`, project observations, baseline observations, dependency
  status, and normalized index rows.
- Identified public documentation that still describes comparison freshness as
  QR-only and must be updated on Day 11.
- Created `artifacts/day9-report-design.md`.

### Day 10

- Verified selected partial-SVD comparison rows appear in normalized report
  output.
- Strengthened `tests/test_normalize_report_index.py` so the complete
  selected comparison case asserts the ten normalized
  `partial_svd_diag6_k2` rows, `pass` status, `local_only` support tier,
  artifact identity, and raw-vector-identity non-claim boundary.
- Confirmed selected freshness still fails stale, missing, duplicate,
  unexpected, fail, defer, and skip cases through the focused normalizer test.
- Re-ran `make report-index-comparison-freshness` and confirmed the expanded
  selected comparison gate passes locally.
- Created `artifacts/day10-report-integration.md`.

### Day 11

- Updated public validation wording so the selected comparison freshness gate
  is described as QR plus partial-SVD, not QR-only.
- Added `partial_svd_diag6_k2` guidance to solver-selection docs with the
  fixture-local row meanings and the local-only evidence boundary.
- Expanded corpus documentation with the partial-SVD comparison target,
  artifact path, ten generated row meanings, and explicit non-claims.
- Added the selected comparison freshness contract to
  `tests/corpus/schemas/report_index_fields.md`, including the three
  source-controlled contract rows and 22 generated selected rows.
- Preserved non-claims for broad QR/SVD/partial-SVD correctness, raw
  singular-vector identity, external parity, platform, package, ABI,
  performance, release, and state-of-the-art evidence.
- Created `artifacts/day11-docs-alignment.md`.

### Day 12

- Ran the selected partial-SVD comparison generator for
  `partial-svd-diag6-k2` and confirmed it regenerated all split artifacts and
  passed the project-vs-baseline comparison.
- Re-ran both selected QR comparison generators to protect Sprint 160 behavior.
- Ran `make report-index-comparison-freshness`; the expanded selected
  comparison gate passed with `25` normalized rows.
- Ran `make report-index-oracle-freshness`; the selected oracle gate passed
  with `54` normalized rows.
- Ran the combined corpus/oracle/comparison normalized index check and
  confirmed `153` rows.
- Ran schema validation, focused normalizer and external-comparison tests,
  Python compile checks, and `git diff --check`.
- Confirmed no `.c` or `.h` files are modified, so the full C quality gate is
  not required for Day 12.
- Created `artifacts/day12-validation.md`.

### Day 13

- Traced the selected `partial_svd_diag6_k2` claim from the source-controlled
  report-family row through generated study rows, normalizer selected row IDs,
  focused tests, and public/maintainer documentation.
- Confirmed the generated comparison family has ten selected rows and remains
  `local_only` evidence.
- Confirmed optional NumPy/SciPy dependency rows remain `defer` context and
  cannot be interpreted as passing evidence.
- Reviewed public and maintainer docs for unsupported broad SVD,
  partial-SVD, raw singular-vector, external parity, hosted, release,
  platform, package, ABI, performance, and state-of-the-art claims.
- Prepared the Sprint 162 Windows package handoff as a separate package parity
  product decision, not an extension of Sprint 161 comparison evidence.
- Created `artifacts/day13-evidence-review.md`.

### Day 14

- Re-ran final targeted validation for the changed-file surface:
  `make report-index-comparison-freshness`,
  `make report-index-oracle-freshness`,
  `python3 scripts/validate_corpus_schema.py`,
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_run_external_comparison.py`, Python compile checks, and
  `git diff --check`.
- Confirmed comparison freshness passed with `25` rows, oracle freshness passed
  with `54` rows, and the combined corpus/oracle/comparison normalized index
  reported `153` rows.
- Confirmed no `.c` or `.h` files are modified, so the full C quality gate is
  not required for Sprint 161 closeout.
- Reviewed changed files for stale paths and unsupported positive claims; the
  selected claim remains fixture-local to `partial_svd_diag6_k2`.
- Closed optional NumPy/SciPy dependency rows as `defer` context only, not pass
  evidence.
- Prepared the Sprint 161 retrospective input set and reaffirmed the Sprint 162
  Windows package parity handoff.
- Created `artifacts/day14-closeout.md`.
