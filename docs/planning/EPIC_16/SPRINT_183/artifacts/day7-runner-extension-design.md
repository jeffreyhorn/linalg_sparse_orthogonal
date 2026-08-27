# Sprint 183 Day 7: Runner Extension Design

## Purpose

Design the comparison runner changes needed for the selected Cholesky SPD
tridiagonal external comparison family before Day 8 implementation.

## Target Registration

Add one target to `scripts/run_external_comparison.py`:

| Field | Value |
| --- | --- |
| Key | `cholesky-spd-tridiag-5` |
| `comparison_kind` | `cholesky` |
| `fixture_key` | `cholesky_spd_tridiag_5` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `operation` | `cholesky_spd_solve` |
| `output_dir` | `build/comparison/cholesky_spd_tridiag_5/` |
| `rows` | `5` |
| `cols` | `5` |
| `rhs` | `[2.0, 4.0, 6.0, 8.0, 16.0]` |
| `expected_solution` | `[1.0, 2.0, 3.0, 4.0, 5.0]` |
| `expected_solution_norm` | `7.416198487095663` |
| `residual_tolerance` | `1e-10` |
| `solution_tolerance` | `1e-10` |
| `baseline_value_count` | `5` |
| `solve_mode` | `cholesky_spd_solve` |

Target entries:

| Row | Col | Value |
| ---: | ---: | ---: |
| 0 | 0 | 4 |
| 0 | 1 | -1 |
| 1 | 0 | -1 |
| 1 | 1 | 4 |
| 1 | 2 | -1 |
| 2 | 1 | -1 |
| 2 | 2 | 4 |
| 2 | 3 | -1 |
| 3 | 2 | -1 |
| 3 | 3 | 4 |
| 3 | 4 | -1 |
| 4 | 3 | -1 |
| 4 | 4 | 4 |

Summary title:

```text
Cholesky SPD Tridiagonal External Comparison Study
```

Success message:

```text
external-comparison: cholesky-spd-tridiag-5 project-vs-baseline comparison passed
```

## Project Probe Design

Extend `project_probe_source` with `solve_mode == "cholesky_spd_solve"`.

The generated C probe should:

1. include `sparse_cholesky.h`;
2. build the matrix from target entries using the existing generated-entry
   path;
3. call `sparse_cholesky_factor(A)`;
4. if factorization succeeds, call `sparse_cholesky_solve(A, rhs, x)`;
5. emit the existing solve fields:
   - `status`;
   - `residual_norm`;
   - `solution_norm`;
   - `solution_values`.

Do not add backend, reorder, fill, timing, or factor-layout fields. Those would
widen the claim beyond the selected fixture-local solve comparison.

## Baseline Dispatch Design

Add Cholesky-specific branches:

| Function | Required behavior |
| --- | --- |
| `baseline_name` | Return `source-controlled-dense-cholesky-reference`. |
| `baseline_version` | Return `chol_external_dense_reference.py`. |
| `comparison_configuration` | Use a new Sprint 183 stage marker such as `sprint183_day8_comparison_logic`. |
| `dependency_status_rows` | Mark `tests/chol_external_dense_reference.py` as the required source-controlled helper. |
| `run_baseline_reference` | Dispatch Cholesky targets to a Cholesky solve parser. |

The Cholesky parser should match the LU solve parser shape:

1. invoke `python3 tests/chol_external_dense_reference.py cholesky_spd_tridiag_5`;
2. require the first line to be `OK 5`;
3. parse exactly five float solution values;
4. compute baseline residual from `descriptor_entries`, target RHS, and parsed
   solution;
5. compute solution norm from parsed solution;
6. return `status`, `solution_values`, `residual_norm`, `solution_norm`,
   `baseline_command`, `baseline_helper_path`, Python executable, and Python
   version.

Day 8 may factor this parser into a shared solve-baseline helper if it keeps
the edit small. Avoid broad refactoring.

## Output File Contract

The new target must emit the standard selected comparison files:

- `project_observations.tsv`;
- `baseline_observations.tsv`;
- `dependency_status.tsv`;
- `study.tsv`;
- `summary.md`;
- `manifest.tsv`.

The output directory is:

```text
build/comparison/cholesky_spd_tridiag_5/
```

Generated files remain ignored and unstaged. Day 8 should inspect local output
with `git status --short -- build/comparison` before finishing.

## Study Row Contract

Reuse existing solve-shaped row generation in `comparison_study_rows`.

Expected row IDs:

```text
comparison_cholesky_spd_tridiag_5_project_status_v1
comparison_cholesky_spd_tridiag_5_baseline_status_v1
comparison_cholesky_spd_tridiag_5_residual_norm_v1
comparison_cholesky_spd_tridiag_5_solution_norm_v1
comparison_cholesky_spd_tridiag_5_solution_values_v1
comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1
```

Expected row count: 6.

The existing `expected_study_row_ids` solve branch already returns the correct
six-row pattern for non-partial-SVD targets, so adding the target should be
enough for self-check coverage.

## Focused Runner Tests

Extend `tests/test_run_external_comparison.py`:

| Area | Expected change |
| --- | --- |
| `TARGET_EXPECTATIONS` | Add `cholesky-spd-tridiag-5` with fixture, subfamily, operation, helper path, command, artifact pattern, expected metrics, and success message. |
| Unsupported-target diagnostic | Require `cholesky-spd-tridiag-5` to appear in supported target output. |
| Output generation loop | Existing loop should generate and validate the Cholesky target once added to `TARGET_EXPECTATIONS`. |
| Dependency rows | Existing helper assertion should validate `tests/chol_external_dense_reference.py`. |
| Report metadata | Existing metadata assertions should validate report-family row once Day 9 adds manifest metadata. |

Day 8 can set `require_report_family_metadata=False` temporarily for the new
target only if manifest integration is intentionally deferred to Day 9. If that
temporary test bypass is used, Day 9 must remove it.

## Self-Check Plan

Run:

```text
python3 scripts/run_external_comparison.py --self-check
```

Expected behavior:

- each target, including Cholesky, validates its expected row IDs;
- missing-row validation fails with `missing_selected_row`;
- duplicate-row validation fails with `duplicate_selected_row`;
- non-pass selected row validation fails with `metric_tolerance_miss`;
- optional `numpy` and `scipy` rows remain `defer`.

If self-check fails after adding Cholesky, Day 8 should treat it as a runner
contract error and stop before report integration.

## Failure Behavior

| Failure | Expected class |
| --- | --- |
| Unknown target | `unsupported_target` |
| Missing project library | `missing_project_library` |
| Probe compile failure | `project_build_failed` |
| Unsupported project solve mode | `unsupported_target` |
| Missing Cholesky helper | `missing_baseline_helper` |
| Cholesky helper exits nonzero | `baseline_command_failed` |
| Helper first line is not `OK 5` | `baseline_malformed_output` |
| Helper emits wrong value count | `baseline_malformed_output` |
| Helper emits non-numeric values | `baseline_malformed_output` |
| Project solution has wrong length | `project_probe_failed` |
| Baseline solution has wrong length | `baseline_malformed_output` |
| Selected row missing | `missing_selected_row` |
| Selected row duplicated | `duplicate_selected_row` |
| Selected row non-pass | `metric_tolerance_miss` |

## Day 8 Handoff

Implementation order:

1. Add Cholesky target metadata and non-claims.
2. Add `cholesky_spd_solve` probe mode and header include.
3. Add Cholesky baseline/helper/dependency dispatch.
4. Extend runner tests.
5. Run self-check and focused runner tests.
6. Generate local Cholesky comparison output for inspection.
7. Confirm `build/comparison/` remains unstaged.
8. Write Day 8 implementation notes.

## Validation

Day 7 changes planning artifacts only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| Runner implementation can proceed without revisiting the fixture contract. | Complete | Target fields, probe behavior, baseline parsing, and row IDs are fixed. |
| Generated output shape matches existing selected comparison patterns. | Complete | The standard six files and six solve rows are reused. |
| Failure behavior is defined before code changes. | Complete | Expected `ComparisonError` classes are specified. |
