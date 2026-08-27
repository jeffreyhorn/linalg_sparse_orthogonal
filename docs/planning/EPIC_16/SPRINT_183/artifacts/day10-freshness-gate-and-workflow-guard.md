# Sprint 183 Day 10: Freshness Gate And Workflow Guard Update

## Scope

Day 10 promoted the selected Cholesky SPD tridiagonal comparison into the
hosted Linux and macOS selected comparison freshness lanes while preserving the
Sprint 182 Windows report freshness deferral.

## Workflow Updates

| Workflow | Update |
| --- | --- |
| `.github/workflows/ci.yml` | Added `cholesky-spd-tridiag-5` to the selected comparison summary target list with 6 expected rows. |
| `.github/workflows/ci.yml` | Added the six `build/comparison/cholesky_spd_tridiag_5/` files to the fail-closed selected comparison upload artifact. |
| `.github/workflows/macos-ci.yml` | Added `cholesky-spd-tridiag-5` to the selected comparison summary target list with 6 expected rows. |
| `.github/workflows/macos-ci.yml` | Added the six `build/comparison/cholesky_spd_tridiag_5/` files to the fail-closed selected comparison upload artifact. |
| `.github/workflows/macos-ci.yml` | Updated the hosted selected comparison lane comment to include Cholesky SPD tridiagonal solve. |

The uploaded Cholesky file set matches the selected target manifest contract:

| File |
| --- |
| `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv` |
| `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv` |
| `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv` |
| `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `build/comparison/cholesky_spd_tridiag_5/summary.md` |
| `build/comparison/cholesky_spd_tridiag_5/manifest.tsv` |

## Guard Updates

`tests/test_selected_comparison_workflow.py` now checks that the selected
Cholesky target is present in both Linux and macOS selected comparison lanes.
The existing manifest-driven guard continues to require every selected
manifest-owned file path, reject broad `build/comparison/**` uploads, require
fail-closed uploads, and validate summary row-count checks.

Day 10 also added a Cholesky-specific drift test that removes
`build/comparison/cholesky_spd_tridiag_5/study.tsv` from the Linux upload block
and verifies the guard fails clearly.

## Windows Boundary

Windows report freshness remains formally deferred. Day 10 did not add
`windows` to selected target manifest `workflow_platforms`, did not add a
Windows selected freshness command, and did not add Windows selected comparison
artifacts. The workflow guard still rejects selected freshness commands and
selected comparison artifact names in the Windows workflow.

## Freshness Result

`make report-index-comparison-freshness` regenerated the selected comparison
outputs and reported 39 fresh comparison rows. The selected Cholesky rows were
fresh against the current branch head.

## Validation

| Command | Status |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `make report-index-comparison-freshness` | Pass |
| `git status --short -- build/comparison build/report-index` | Pass |
| `git diff --check` | Pass |
