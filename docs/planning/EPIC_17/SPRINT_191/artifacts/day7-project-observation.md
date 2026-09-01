# Sprint 191 Day 7: Project Observation

## Purpose

Validate and harden project-side observations for the selected
`qr-incompatible-ls` comparison family.

## Implementation Summary

Day 7 confirmed that the existing generated C probe path can run the selected
4-by-2 QR incompatible least-squares fixture without solver changes. The probe
uses the runner descriptor added on Day 5 and the generic QR
`least_squares_solve` project path:

- build a temporary C probe from `project_probe_source()`;
- create the sparse fixture from `QR_INCOMPATIBLE_LS_ENTRIES`;
- call `sparse_qr_factor()`;
- call `sparse_qr_solve()`;
- emit `status`, `residual_norm`, `solution_norm`, and `solution_values`;
- normalize those fields through `project_observation_rows()`.

No `.c` or `.h` source changes were required.

## Generated Project Observations

The target-specific generator produced these project observations:

| Metric | Value | Status | Status reason |
| --- | --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | `pass` | `project_status_match` |
| `residual_norm` | `1.7320508075688772` | `pass` | `project_residual_matches_expected` |
| `solution_norm` | `2.2360679774997894` | `pass` | `project_solution_norm_within_tolerance` |
| `solution_values` | `1.9999999999999996,-1.0000000000000002` | `pass` | `project_solution_values_within_tolerance` |

The small representation differences in solution values are inside the
selected `1e-10` tolerance and agree with the source-controlled dense helper.

## Added Project-Side Tests

Day 7 extended `tests/test_run_external_comparison.py` with:

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_ls_project_probe_observations()` | Runs the actual generated project probe and asserts status, residual, solution norm, solution values, project command naming, and project observation pass rows. |
| `test_qr_incompatible_ls_project_rows_reject_residual_mismatch()` | Verifies a synthetic zero residual fails the expected nonzero residual contract. |
| `test_qr_incompatible_ls_project_rows_reject_solution_mismatch()` | Verifies a synthetic solution-value mismatch fails the solution-values observation row. |

These tests make the project-side evidence deterministic and tied to the Day 3
fixture contract.

## Failure Diagnostics

| Failure path | Verified behavior |
| --- | --- |
| Project probe does not emit required fields | `parse_key_values()` raises `project_probe_failed` with missing field names. |
| Project residual does not match `expected_residual_norm` | `project_observation_rows()` marks `residual_norm` as `fail` with `project_residual_expected_mismatch`. |
| Project solution values do not match expected solution | `project_observation_rows()` marks `solution_values` as `fail` with `project_solution_values_tolerance_miss`. |
| Project QR status is not `SPARSE_SUCCESS` | `project_status` row fails with `project_status_mismatch`. |

## Validation

Commands run:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
python3 tests/test_run_external_comparison.py
column -t -s $'\t' build/comparison/qr_incompatible_ls/project_observations.tsv
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- `qr-incompatible-ls` generated successfully;
- generated project observations passed for status, residual, solution norm,
  and solution values;
- `tests/test_run_external_comparison.py` passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 7.

## Day 8 Handoff

Day 8 should integrate `qr-incompatible-ls` into the broader study and
manifest surfaces:

- add report-family metadata for `qr_incompatible_ls`;
- add selected target manifest metadata;
- add the target to `make report-index-comparison-freshness`;
- update Linux/macOS exact artifact upload scopes if hosted selected
  comparison freshness should include the new target;
- keep Windows selected comparison metadata unchanged unless a separate hosted
  proof is added.
