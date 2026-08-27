# Sprint 183 Day 13: Claim Review And Hardening

## Scope

Day 13 reconciled the Sprint 183 implementation, manifests, workflows, docs,
tests, validation records, and residual risks before closeout.

## Project-Plan Reconciliation

| Item | Status | Evidence |
| --- | --- | --- |
| 183.1 Family Selection | Complete | Days 1-4 selected exactly one additional family: Cholesky SPD tridiagonal solve. |
| 183.2 Fixture and Metric Contract | Complete | Days 5-6 defined and implemented `cholesky_spd_tridiag_5`, RHS, expected solution, tolerances, six rows, helper support, and helper tests. |
| 183.3 Harness Extension | Complete | Days 7-8 designed and implemented `cholesky-spd-tridiag-5`, `cholesky_spd_solve`, helper dispatch, dependency rows, and focused runner tests. |
| 183.4 Report Integration | Complete | Days 9-10 added report-family metadata, selected target metadata, freshness generation, Linux/macOS workflow uploads, and workflow guards. |
| 183.5 Documentation Alignment | Complete | Day 11 aligned README, solver-selection, maintainer, corpus, and report-index schema docs. |
| 183.6 Validation | In progress | Day 12 completed integrated validation; Day 14 should close the sprint and record final handoff. |

## Claim Consistency

| Surface | Claim state |
| --- | --- |
| Runner target | `cholesky-spd-tridiag-5` generates one fixture-local Cholesky SPD solve comparison. |
| Fixture | `cholesky_spd_tridiag_5`, a 5x5 SPD tridiagonal matrix with expected solution `[1, 2, 3, 4, 5]`. |
| Baseline | `tests/chol_external_dense_reference.py cholesky_spd_tridiag_5`. |
| Project probe | `sparse_cholesky_factor(A)` plus `sparse_cholesky_solve(A, rhs, x)`. |
| Selected row count | 6 rows. |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| Freshness command | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5`. |
| Hosted platforms | Linux and macOS selected comparison lanes only. |
| Windows boundary | Windows report freshness remains formally deferred. |

The selected target manifest, report-family manifest, runner metadata,
normalizer tests, selected workflow guard, README, solver-selection docs,
maintainer guide, corpus README, and report-index schema docs all describe the
same bounded claim.

## Non-Claims

The retained non-claims are explicit across manifests, runner output, docs, and
tests:

- no broad Cholesky correctness
- no broad SPD coverage
- no broad reordering coverage or reordering parity
- no CSC-vs-linked-list parity
- no factor-layout identity
- no fill superiority
- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem parity
- no Windows report freshness
- no package-manager proof
- no shared-library ABI proof
- no performance superiority
- no release proof
- no state-of-the-art claim

## Diagnostic Hardening Review

No Day 13 code hardening was needed.

Existing checks already fail with actionable Cholesky-specific diagnostics:

- unsupported runner targets list `cholesky-spd-tridiag-5`;
- selected manifest tests validate required fields, row counts, platform
  metadata, and non-claims;
- normalizer tests assert Cholesky row IDs, artifact path, subfamily, and
  non-claims;
- workflow guard tests assert the selected Cholesky target appears in Linux and
  macOS lanes;
- workflow drift tests fail clearly if the Cholesky selected `study.tsv` upload
  path is missing;
- Windows workflow guards reject selected freshness commands and selected
  comparison artifact names while the Sprint 182 deferral is active.

## Retrospective Inputs

- The sprint successfully added one bounded selected comparison family without
  widening optional dependency, package, platform, performance, release, or
  state-of-the-art claims.
- The strongest implementation pattern was reusing the existing solve-shaped
  comparison machinery for Cholesky rather than inventing a separate report
  shape.
- Day 8 intentionally used a temporary report-family metadata bypass; Day 9
  removed it once the manifest row existed. This kept runner work incremental
  but required careful follow-through.
- Day 12 found one invalid guessed target, `make test_cholesky`; the actual
  repository validation path is `build/test_cholesky` or full `make test`.
- Full `make format && make lint && make test` passed locally even though no
  tracked C/header diffs remained after formatting.

## Sprint 184 Risk Notes

- Hosted Linux/macOS selected comparison freshness still needs CI execution on
  the eventual PR.
- Windows report freshness remains deliberately deferred; future promotion must
  add a Windows-safe generator path, exact selected artifact scope, manifest
  metadata, workflow guard updates, and documentation in one change.
- Future additional comparison families should start from the selected target
  manifest and report-family row contract before adding workflow uploads.

## Validation

| Command | Status |
| --- | --- |
| `python3 - <<'PY' ... selected Cholesky manifest consistency probe ... PY` | Pass |
| `python3 - <<'PY' ... Cholesky runner consistency probe ... PY` | Pass |
| `python3 - <<'PY' ... Linux/macOS workflow Cholesky upload probe ... PY` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `git status --short -- build/comparison build/report-index` | Pass |
| `git diff --check` | Pass |
