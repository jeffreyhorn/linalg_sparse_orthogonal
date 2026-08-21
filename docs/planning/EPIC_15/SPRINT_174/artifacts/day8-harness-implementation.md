# Sprint 174 Day 8: Harness Extension Implementation

## Purpose

Implement the narrow generated-comparison harness extension designed on Day 7
for the selected linked-list LU fixture family.

## Implemented Runner Target

Added a new `scripts/run_external_comparison.py` target:

```text
--target lu-nonsym-square-5
```

Target metadata:

```text
comparison_kind: lu
fixture_key: lu_nonsym_square_5
subfamily: lu_nonsym_square_5
operation: square_solve
output_dir: build/comparison/lu_nonsym_square_5
support_tier: local_only
claim_scope: fixture-local linked-list LU square-solve comparison only
```

The target embeds the selected Day 4 matrix entries, right-hand side,
expected solution, expected solution norm, and `1e-10` residual/solution
tolerances. The selected comparison remains source-controlled by runner
metadata and helper code; generated artifacts stay under `build/`.

## Project Probe

Extended the runner's temporary C probe generator with
`solve_mode == "lu_square_solve"`.

The generated probe:

- constructs the 5x5 `lu_nonsym_square_5` matrix;
- solves with `sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12)`;
- calls `sparse_lu_solve(A, rhs, x)` after successful factorization;
- computes residual norm from the original embedded matrix entries;
- computes solution norm and solution values; and
- emits the existing observation keys:
  `status`, `residual_norm`, `solution_norm`, and `solution_values`.

The probe remains temporary and is removed by default through the existing
`--keep-temp` behavior.

## Baseline Adapter

Added a narrow LU baseline adapter in `scripts/run_external_comparison.py`.
The adapter preserves the Day 6 helper contract:

```text
python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
```

The helper still emits `OK 5` plus five solution values. The runner computes
the baseline residual norm and solution norm from target metadata and parsed
solution values, then emits the same baseline observation keys consumed by the
existing six-row study writer.

## Generated Outputs

Successfully generated the selected LU comparison artifacts:

```text
build/comparison/lu_nonsym_square_5/project_observations.tsv
build/comparison/lu_nonsym_square_5/baseline_observations.tsv
build/comparison/lu_nonsym_square_5/dependency_status.tsv
build/comparison/lu_nonsym_square_5/study.tsv
build/comparison/lu_nonsym_square_5/summary.md
build/comparison/lu_nonsym_square_5/manifest.tsv
```

The generated `study.tsv` emitted exactly six passing rows:

```text
comparison_lu_nonsym_square_5_project_status_v1
comparison_lu_nonsym_square_5_baseline_status_v1
comparison_lu_nonsym_square_5_residual_norm_v1
comparison_lu_nonsym_square_5_solution_norm_v1
comparison_lu_nonsym_square_5_solution_values_v1
comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1
```

Observed local comparison diagnostics:

```text
project residual norm: 5.3290705182007514e-15
baseline residual norm: 8.1402896778041619e-15
residual delta: 2.8112191596034105e-15
solution norm delta: 0
solution max absolute delta: 8.8817841970012523e-16
```

All observed deltas are within the selected `1e-10` tolerance.

## Focused Test Coverage

Extended `tests/test_run_external_comparison.py` so the focused runner suite
generates and validates the LU target's required files, manifest target,
fixture key, row IDs, metrics, pass statuses, dependency rows, support tier,
and artifact path. Report-family metadata assertions are deliberately skipped
for this target until Day 9 adds the selected report-index row and freshness
ownership.

## Deferred To Day 9

Day 8 intentionally does not add the LU rows to the normalized report-index
selected set. Day 9 owns:

- `SELECTED_COMPARISON_ROW_IDS`;
- `SELECTED_COMPARISON_ARTIFACTS`;
- the `comparison	lu_nonsym_square_5` report-family manifest row; and
- `make report-index-comparison-freshness` wiring.

## Validation

Commands run:

```text
python3 tests/test_lu_external_dense_reference.py
python3 scripts/run_external_comparison.py --self-check
python3 tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --target lu-nonsym-square-5
git diff --check
```

All passed.

No `.c` or `.h` source files were modified. The only C compiled for this day
is temporary generated probe source, so the full C quality gate is not required
for Day 8.
