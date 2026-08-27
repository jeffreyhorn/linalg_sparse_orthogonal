# Sprint 183 Day 11: Documentation Alignment

## Scope

Day 11 aligned public and maintainer documentation with the selected Cholesky
SPD tridiagonal comparison family added during Sprint 183.

## Documentation Updates

| File | Update |
| --- | --- |
| `README.md` | Extended selected comparison non-claims to include broad Cholesky correctness, broad SPD coverage, and CSC-vs-linked-list parity. |
| `docs/solver_selection.md` | Added fixture-local Cholesky SPD tridiagonal selected comparison evidence to direct-solver guidance. |
| `docs/solver_selection.md` | Added a bounded Cholesky selected comparison paragraph with tolerances, hosted evidence boundary, and non-claims. |
| `docs/maintainer_guide.md` | Added `make report-index-comparison-freshness` and `cholesky_spd_tridiag_5` to the Cholesky trust-boundary row. |
| `docs/maintainer_guide.md` | Updated selected comparison freshness guidance for QR, partial-SVD, LU, and Cholesky families. |
| `tests/corpus/README.md` | Updated selected comparison docs from four to five fixture-local families and added the Cholesky target row. |
| `tests/corpus/schemas/report_index_fields.md` | Updated report-index schema guidance for manifest-selected QR, partial-SVD, LU, and Cholesky comparison reports. |

## Claim Boundary

The documentation describes `cholesky_spd_tridiag_5` as fixture-local generated
comparison evidence only. It checks Cholesky SPD factor/solve status, residual
norm, solution norm, solution values, and project-vs-baseline max absolute
delta against the selected source-controlled dense Cholesky reference helper
with `1e-10` tolerances.

The updated docs keep these non-claims explicit:

- no broad Cholesky correctness
- no broad SPD coverage
- no reordering parity
- no CSC-vs-linked-list parity
- no fill superiority
- no external-library parity
- no Windows report freshness
- no broad platform, package, ABI, performance, release, or state-of-the-art proof

## Source Of Truth

The selected target manifest remains authoritative for target keys, expected
rows, row IDs, commands, artifact patterns, required files, workflow artifacts,
support tiers, freshness policies, claim scopes, non-claims, and owners.

## Scan Notes

Current README, solver-selection, maintainer, corpus README, and report-index
schema wording now refer to the selected QR, partial-SVD, LU, and Cholesky
comparison set where current docs describe the active selected gate. Historical
planning records still describe their original sprint-era selected comparison
sets and were left unchanged.

## Validation

| Command | Status |
| --- | --- |
| `rg -n "selected QR plus partial-SVD plus LU|QR, Partial-SVD, And LU|selected QR, partial-SVD, and LU|four fixture-local comparison|manifest-selected QR, partial-SVD, and LU|selected QR, partial-SVD, and LU" README.md docs/solver_selection.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md` | Pass; no matches |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `git status --short -- build/comparison build/report-index` | Pass |
| `git diff --check` | Pass |
