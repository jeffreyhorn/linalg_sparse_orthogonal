# Sprint 191 Day 6: Reference Execution

## Purpose

Harden and test the bounded reference execution path for the selected
`qr-incompatible-ls` comparison family.

## Implementation Summary

Day 6 confirmed that `qr-incompatible-ls` can reuse the existing QR dense
reference execution path in `scripts/run_external_comparison.py`:

- `run_baseline_reference()` selects `tests/qr_external_dense_reference.py`
  for QR targets;
- the helper is invoked through `sys.executable`;
- helper output is normalized into `status`, `solution_values`,
  `residual_norm`, `solution_norm`, `baseline_command`,
  `baseline_helper_path`, `baseline_python_executable`, and
  `baseline_python_version`;
- `dependency_status_rows()` reports required `python3` and
  `tests/qr_external_dense_reference.py` rows plus deferred optional NumPy and
  SciPy rows.

No new external package dependency was introduced.

## Structured Reference Observations

For `qr-incompatible-ls`, the source-controlled helper currently normalizes to:

| Observation | Value |
| --- | --- |
| `status` | `success` |
| `solution_values` | `1.9999999999999998,-1` |
| `residual_norm` | `1.7320508075688772` |
| `solution_norm` | within `1e-15` of `2.2360679774997898` |
| `baseline_helper_path` | `tests/qr_external_dense_reference.py` |
| `baseline_command` | command includes `qr_overdetermined_incompatible_4x2` |

The tiny representation difference in solution norm is expected because the
helper emits `1.9999999999999998` for the first solution component. The
comparison remains well inside the selected `1e-10` tolerance.

## Added Tests

Day 6 extended `tests/test_run_external_comparison.py` with focused reference
execution coverage:

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_ls_reference_observations_and_dependencies()` | Asserts normalized baseline observations, required helper dependency status, and deferred NumPy/SciPy non-proof rows. |
| `test_qr_incompatible_ls_reference_parser_rejects_malformed_output()` | Monkeypatches malformed `OK 2` helper output and verifies `baseline_malformed_output`. |
| `test_qr_incompatible_ls_reference_reports_command_failure()` | Monkeypatches helper command failure and verifies `baseline_command_failed`. |
| `test_qr_incompatible_ls_dependency_reports_missing_helper()` | Uses an empty temporary root to verify `baseline_helper_missing` dependency status and `missing_baseline_helper`. |

These tests complement the Day 5 fixture contract test and the existing
selected target generation test.

## Failure-Path Semantics

| Failure path | Verified behavior |
| --- | --- |
| Missing helper file | Required helper dependency row reports `error` with `baseline_helper_missing`; direct baseline execution raises `missing_baseline_helper`. |
| Malformed helper count | Baseline execution raises `baseline_malformed_output` before pass evidence is accepted. |
| Helper command failure | Baseline execution raises `baseline_command_failed` and preserves the failure message. |
| Optional NumPy/SciPy absence | Rows remain `defer` with `optional_package_baseline_not_selected`; they are not pass evidence. |

## Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 6.

## Day 7 Handoff

Day 7 should focus on project-side observations:

- verify the generated C probe solves the 4-by-2 incompatible least-squares
  fixture and reports `SPARSE_SUCCESS`;
- assert project residual, solution norm, and solution values match the Day 3
  contract;
- add focused failure coverage for project observation mismatch if useful;
- keep changes isolated to the runner/test path unless a real project-side
  solver defect is found.
