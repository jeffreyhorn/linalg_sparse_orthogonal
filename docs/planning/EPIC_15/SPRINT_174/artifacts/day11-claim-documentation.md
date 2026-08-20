# Sprint 174 Day 11: Claim Documentation Update

## Purpose

Update maintained public and maintainer documentation so the selected
comparison freshness gate names the new `lu-nonsym-square-5` family with
fixture-local scope, exact artifact ownership, row counts, and non-claims.

## Updated Documentation

### `README.md`

- Updated quick validation wording from selected QR plus partial-SVD comparison
  freshness to selected QR plus partial-SVD plus LU comparison freshness.
- Updated normalized report-index guidance so the selected comparison gate is
  limited to:
  - QR minimum-norm;
  - QR compatible least-squares;
  - partial-SVD diag6 k2;
  - linked-list LU nonsymmetric square solve.
- Updated the public evidence boundary to name `lu_nonsym_square_5` and reject
  broad LU correctness, nonsymmetric solve correctness, LU CSR parity,
  package/ABI, performance, platform, release, and state-of-the-art claims.

### `docs/maintainer_guide.md`

- Added `make report-index-comparison-freshness` as a maintained evidence
  owner for the linked-list LU row in the solver-family trust table.
- Added the `lu-nonsym-square-5` runner command to the selected comparison
  freshness workflow.
- Added the six generated LU comparison artifacts under
  `build/comparison/lu_nonsym_square_5/`.
- Updated selected comparison expectations from three contract rows plus 22
  generated rows to four contract rows plus 28 generated rows.
- Added the four-family row-count breakdown and clarified that selected QR and
  LU families use the same six row names.
- Expanded the non-claims to include broad LU correctness, nonsymmetric solve
  correctness, and LU CSR parity.

### `docs/solver_selection.md`

- Added fixture-local linked-list LU comparison evidence for
  `lu_nonsym_square_5` to the direct-solver selection table.
- Updated QR wording so the selected comparison gate points to the selected
  partial-SVD and LU comparisons.
- Added a selected LU comparison boundary paragraph with comparator,
  diagnostics, tolerance, local/hosted interpretation, and non-claims.

### `tests/corpus/README.md`

- Renamed the selected comparison freshness section to include LU.
- Updated the gate from three to four selected fixture-local comparison
  families.
- Added the `lu-nonsym-square-5` target row with fixture, meaning, and artifact.
- Documented that the selected LU family contributes the same six generated
  rows as QR comparison families.
- Added broad LU, nonsymmetric solve, and LU CSR non-claims.

### `tests/corpus/schemas/report_index_fields.md`

- Updated the selected comparison freshness section from Sprint 161 to Sprint
  174.
- Updated selected reports from QR plus partial-SVD to QR plus partial-SVD plus
  LU.
- Updated expected counts from three contract rows plus 22 generated rows to
  four contract rows plus 28 generated rows.
- Added the `lu_nonsym_square_5` six-row expectation and LU non-claims.

### `benchmarks/README.md`

- Updated the report-index handoff table so
  `make report-index-comparison-freshness` points at all selected comparison
  subdirectories, including `lu_nonsym_square_5`.
- Updated normalized-index interpretation from QR minimum-norm only to selected
  QR, partial-SVD, and LU comparison studies.
- Updated generated-output guidance from `build/comparison/qr_minnorm/` to
  `build/comparison/*/`.

## Claim Boundary

The selected LU comparison claim is:

```text
fixture-local linked-list LU square-solve comparison for lu_nonsym_square_5
against the selected source-controlled dense LU reference helper
```

The documentation does not claim broad LU correctness, broad nonsymmetric
solve parity, LU CSR parity, sparse-direct solver parity, pivoting superiority,
factor-layout identity, external-library parity, platform support,
package-manager support, shared-library ABI support, runtime-loader support,
performance superiority, release readiness, or state-of-the-art status.

## Claim Scans

Ran a targeted stale-wording scan:

```text
rg -n "selected QR (and|\\+) partial-SVD comparison|selected QR and partial-SVD comparison|selected QR \\+ partial-SVD comparison|three source-controlled comparison|22 generated comparison|three fixture-local comparison|QR and partial-SVD comparison reports|QR plus partial-SVD comparison" README.md docs tests/corpus benchmarks/README.md
```

Remaining matches are historical planning artifacts only, including prior
Sprint 161 and early Sprint 174 intake/design notes. Maintained public,
maintainer, corpus, schema, solver-selection, and benchmark docs now use the
updated selected QR plus partial-SVD plus LU wording.

Ran a package/ABI non-claim scan:

```text
rg -n "package-manager proof|shared-library ABI proof|package/ABI support|package proof; ABI proof" README.md docs/maintainer_guide.md docs/solver_selection.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md benchmarks/README.md
```

The matches are non-claim wording only.

## Validation

Commands run:

```text
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

All passed.

No `.c` or `.h` files were modified. The full C quality gate is not required
for Day 11.
