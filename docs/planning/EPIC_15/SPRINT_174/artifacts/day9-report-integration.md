# Sprint 174 Day 9: Report Index Integration

## Purpose

Promote the Day 8 linked-list LU comparison target into the selected
report-index freshness surface with source-controlled proof-owner metadata and
bounded claim language.

## Source-Controlled Updates

Updated `scripts/normalize_report_index.py`:

- added six selected LU comparison row IDs to `SELECTED_COMPARISON_ROW_IDS`;
- added `build/comparison/lu_nonsym_square_5/study.tsv` to
  `SELECTED_COMPARISON_ARTIFACTS`.

Updated `Makefile`:

- added `python3 scripts/run_external_comparison.py --target
  lu-nonsym-square-5` to `report-index-comparison-freshness`.

Updated `tests/corpus/manifests/report_families.tsv`:

- added the `comparison	lu_nonsym_square_5` proof-owner row;
- recorded generator command
  `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5`;
- recorded artifact pattern `build/comparison/lu_nonsym_square_5/study.tsv`;
- set support tier to `local_only`;
- set freshness policy to `generated_compare_inputs`;
- bounded the claim to one fixture-level linked-list LU square-solve
  comparison against the selected source-controlled dense reference helper.

Updated `tests/test_run_external_comparison.py`:

- enabled report-family metadata checks for `lu-nonsym-square-5` now that the
  source-controlled proof-owner row exists.

## Selected Row IDs

The normalized report-index now requires these LU generated rows:

```text
comparison_lu_nonsym_square_5_project_status_v1
comparison_lu_nonsym_square_5_baseline_status_v1
comparison_lu_nonsym_square_5_residual_norm_v1
comparison_lu_nonsym_square_5_solution_norm_v1
comparison_lu_nonsym_square_5_solution_values_v1
comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1
```

The comparison selected set now contains 32 normalized rows:

- 24 generated comparison evidence rows from QR minimum-norm, QR compatible
  least-squares, partial-SVD diagonal top-k, and linked-list LU square solve;
- 4 source-controlled report-contract rows;
- 4 freshness advisory rows.

## Claim Boundary

The LU report-family row uses this bounded claim:

```text
Generated comparison rows record one local fixture-level linked-list LU
square-solve comparison for lu_nonsym_square_5 against the selected
source-controlled dense reference helper.
```

The row explicitly excludes broad LU correctness, nonsymmetric solve parity,
LU CSR parity, sparse-direct solver parity, pivoting superiority,
factor-layout identity, NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity,
external-library ecosystem parity, hosted CI proof, release proof, platform
portability proof, package-manager proof, shared-library ABI proof, performance
superiority, and state-of-the-art claims.

## Validation Notes

Running the normalizer directly before regenerating all selected artifacts
failed because only the LU artifact existed locally and the selected set now
requires all selected comparison families. That is expected ownership behavior:
`make report-index-comparison-freshness` is the proof-owner command because it
regenerates every selected comparison artifact before normalizing freshness.

Successful validation:

```text
python3 tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
make report-index-comparison-freshness
git diff --check
```

`make report-index-comparison-freshness` regenerated:

```text
build/comparison/qr_minnorm/study.tsv
build/comparison/qr_compatible_ls/study.tsv
build/comparison/partial_svd_diag6_k2/study.tsv
build/comparison/lu_nonsym_square_5/study.tsv
```

and completed with:

```text
normalize-report-index: freshness ok (32 rows)
report-index-comparison-freshness: passed (local-only generated comparison freshness)
```

No `.c` or `.h` source files were modified. The full C quality gate is not
required for Day 9.
