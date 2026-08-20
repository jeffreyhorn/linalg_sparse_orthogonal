# Day 6: Fixture Implementation

## Purpose

Implement and validate the selected linked-list LU fixture surface before
comparison-runner integration.

## Implementation Summary

Day 6 did not need a new mathematical fixture because Sprint 174 selected the
existing `lu_nonsym_square_5` fixture already shared by:

- `tests/lu_external_dense_reference.py`;
- `tests/test_sparse_lu.c`.

Day 6 added a focused source-controlled helper test:

```text
tests/test_lu_external_dense_reference.py
```

The test fixes the fixture contract before Day 8 runner implementation by
checking:

- the exact 5x5 nonsymmetric matrix;
- fixture-key lookup for `lu_nonsym_square_5`;
- right-hand side generation from `x_true = [1, 2, 3, 4, 5]`;
- dense helper solve output against the expected solution;
- CLI output contract `OK 5` plus five numeric solution values;
- unknown fixture failure without pass evidence.

## Fixture Contract Now Guarded

| Contract element | Guard |
| --- | --- |
| Matrix entries | `test_lu_nonsym_square_5_fixture_matrix_and_rhs` |
| Fixture key | `test_lu_nonsym_square_5_fixture_matrix_and_rhs` |
| Right-hand side | `test_lu_nonsym_square_5_fixture_matrix_and_rhs` |
| Dense solution | `test_lu_nonsym_square_5_dense_solution_matches_fixture_contract` |
| CLI success output | `test_lu_nonsym_square_5_cli_contract` |
| Unknown fixture failure | `test_unknown_fixture_fails_without_pass_evidence` |

## Expected Values

The guarded fixture contract remains:

```text
x_true = [1.0, 2.0, 3.0, 4.0, 5.0]
b = [12.5, 10.5, 18.0, 24.0, 48.0]
```

The CLI emits values equivalent to:

```text
OK 5
1
2
3.0000000000000004
4
4.9999999999999991
```

The test uses `1e-12` helper-level tolerance for comparing dense helper output
with the exact expected solution. This is tighter than the planned generated
comparison report tolerance `1e-10`; the generated report tolerance remains
unchanged.

## Implementation Boundary

Day 6 intentionally did not:

- add the `lu-nonsym-square-5` target to `scripts/run_external_comparison.py`;
- add selected LU comparison row IDs to `scripts/normalize_report_index.py`;
- add report-family manifest rows;
- update `make report-index-comparison-freshness`;
- generate or stage `build/comparison/lu_nonsym_square_5/`;
- change `.c` or `.h` files.

Those are Day 7 through Day 10 harness, report, and freshness tasks.

## Validation

Day 6 validation passed:

```sh
python3 tests/test_lu_external_dense_reference.py
python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
git diff --check
```

No `.c` or `.h` files changed on Day 6, so the full C quality gate is not
required for this day.

## Completion Check

Day 6 completion criteria are met:

- selected fixture definitions and helper behavior are source-controlled and
  guarded;
- fixture names and diagnostics are stable enough for report integration;
- implementation does not widen solver-family claims.
