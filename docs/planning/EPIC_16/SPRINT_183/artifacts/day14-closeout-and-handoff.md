# Sprint 183 Day 14: Closeout And Handoff

## Closed Claim

Sprint 183 adds one selected bounded external comparison family:
`cholesky-spd-tridiag-5`.

The maintained claim is fixture-local:

- fixture key: `cholesky_spd_tridiag_5`
- matrix: 5x5 SPD tridiagonal, diagonal 4 and off-diagonal -1
- RHS: `[2, 4, 6, 8, 16]`
- expected solution: `[1, 2, 3, 4, 5]`
- selected helper: `tests/chol_external_dense_reference.py`
- project probe: `sparse_cholesky_factor(A)` plus
  `sparse_cholesky_solve(A, rhs, x)`
- selected rows: 6 solve-shaped comparison rows
- tolerance: `1e-10`
- generated artifact: `build/comparison/cholesky_spd_tridiag_5/study.tsv`

The selected comparison freshness gate now covers five fixture-local families:

| Target | Rows | Hosted selected platforms |
| --- | ---: | --- |
| `qr-minnorm` | 6 | Linux, macOS |
| `qr-compatible-ls` | 6 | Linux, macOS |
| `partial-svd-diag6-k2` | 10 | Linux, macOS |
| `lu-nonsym-square-5` | 6 | Linux, macOS |
| `cholesky-spd-tridiag-5` | 6 | Linux, macOS |

## Changed Surfaces

| Surface | Summary |
| --- | --- |
| Helper | `tests/chol_external_dense_reference.py` supports key-based `cholesky_spd_tridiag_5` fixture input. |
| Runner | `scripts/run_external_comparison.py` includes `cholesky-spd-tridiag-5`, Cholesky project probe generation, dense helper dispatch, dependency rows, and non-claims. |
| Makefile | `make report-index-comparison-freshness` regenerates the Cholesky selected comparison output before normalization. |
| Manifests | `report_families.tsv` and `selected_report_targets.tsv` register `comparison/cholesky_spd_tridiag_5` and `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`. |
| Workflows | Linux and macOS selected comparison freshness lanes summarize and upload the six Cholesky generated files. |
| Tests | Helper, runner, normalizer, selected manifest, and selected workflow guard tests cover Cholesky rows, metadata, artifacts, and drift behavior. |
| Docs | README, solver-selection, maintainer, corpus, and report-index schema docs describe the bounded Cholesky comparison and non-claims. |

## Validation Summary

Day 12 ran the integrated validation pass:

| Command | Status |
| --- | --- |
| `python3 tests/test_chol_external_dense_reference.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `build/test_cholesky` | Pass |
| `make report-index-comparison-freshness` | Pass |
| `bash scripts/static_package_deferral_check.sh` | Pass |
| `bash scripts/package_manager_deferral_check.sh` | Pass |
| `make format` | Pass |
| `make lint` | Pass |
| `make test` | Pass |

Day 13 then rechecked the claim-guarding surfaces:

| Command | Status |
| --- | --- |
| selected Cholesky manifest consistency probe | Pass |
| Cholesky runner consistency probe | Pass |
| Linux/macOS workflow Cholesky upload probe | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |

`make test_cholesky` is not a repository Makefile target. The actual focused
Cholesky validation path is `build/test_cholesky` or full `make test`.

## Final Artifact Hygiene

Generated comparison and report-index outputs remain local build artifacts and
are not staged:

- `build/comparison/`
- `build/report-index`

Day 12 `make format` did not leave tracked C/header diffs. Full
`make format`, `make lint`, and `make test` passed locally after the code and
documentation changes.

## Non-Claims

Sprint 183 does not claim:

- broad Cholesky correctness
- broad SPD coverage
- broad reordering coverage or reordering parity
- CSC-vs-linked-list parity
- factor-layout identity
- fill superiority
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem parity
- Windows report freshness
- package-manager proof
- shared-library ABI proof
- performance superiority
- release proof
- state-of-the-art status

Windows report freshness remains formally deferred under the Sprint 182
decision record. No Sprint 183 selected target lists `windows`, and no Windows
workflow selected report freshness command or selected comparison upload name
was added.

## Handoff

PR review should focus on:

- whether the selected target manifest row exactly matches runner output,
  workflow uploads, docs, and normalizer tests;
- whether hosted Linux/macOS selected comparison lanes upload only the six
  Cholesky files and avoid broad `build/comparison/**` upload paths;
- whether the retained Windows deferral boundary stays intact;
- whether non-claims remain strong enough to prevent broad Cholesky, platform,
  package, performance, release, or state-of-the-art interpretation.

Future Sprint 184 work should treat LDLT KKT, iterative SPD/nonsymmetric
solves, eigensolver comparisons, backend telemetry, and performance comparisons
as separate candidate families with fresh fixture contracts and non-claim
reviews.

## Retrospective Inputs

- Reusing the existing solve-shaped comparison machinery kept the new family
  bounded and reduced schema churn.
- Manifest-first selected target metadata made Day 10 workflow guard updates
  straightforward and fail-closed.
- The temporary Day 8 report-family metadata bypass was acceptable only because
  Day 9 removed it immediately when the report-family row landed.
- Full local quality validation was useful even without tracked C/header diffs
  after formatting because the runner emits a generated C project probe.

## Final Checks

| Command | Status |
| --- | --- |
| `ls docs/planning/EPIC_16/SPRINT_183/artifacts` | Pass |
| `git diff --name-only | rg '\.(c|h)$' || true` | Pass; no tracked C/header diffs |
| `git status --short -- build/comparison build/report-index` | Pass |
| `find scripts tests -name __pycache__ -type d -print` | Pass after final cleanup; no cache dirs |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `git diff --check` | Pass |
