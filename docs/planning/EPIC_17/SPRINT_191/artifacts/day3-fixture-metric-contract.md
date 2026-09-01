# Sprint 191 Day 3: Fixture and Metric Contract

## Purpose

Define the exact fixture, numerical behavior, metrics, tolerances, row IDs,
and claim boundary for the selected `qr-incompatible-ls` comparison before
runner and manifest implementation begins.

## Fixture Contract

| Field | Contract |
| --- | --- |
| Target key | `qr-incompatible-ls` |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Subfamily | `qr_incompatible_ls` |
| Solver family | QR |
| Operation | `least_squares_solve` |
| Matrix shape | 4 rows by 2 columns |
| Matrix entries | `(0,0)=1`, `(1,1)=1`, `(2,0)=1`, `(2,1)=1`, `(3,0)=2`, `(3,1)=-1` |
| Right-hand side | `[1.0, -2.0, 2.0, 5.0]` |
| Expected least-squares solution | `[2.0, -1.0]` |
| Expected solution norm | `2.2360679774997898` (`sqrt(5)`) |
| Expected residual norm | `1.7320508075688772` (`sqrt(3)`) |
| Fixture source | Existing source-controlled helper `tests/qr_external_dense_reference.py` and matching C fixture in `tests/test_qr_solve.c`. |
| Fixture ownership | Handwritten deterministic fixture; no generated fixture file and no external data download. |

The fixture is intentionally inconsistent. The residual is expected to be
nonzero, so Sprint 191 must not model this target as `residual_norm <= 1e-10`.
The meaningful comparison is agreement between the project residual and the
source-controlled dense-reference residual.

## Reference Contract

The selected reference path is:

```sh
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
```

Current helper output:

```text
OK 3
1.9999999999999998
-1
1.7320508075688772
```

The helper returns two solution components followed by the residual norm. The
runner may compute the solution norm from the returned solution values, as it
does for existing QR least-squares targets.

## Metric Contract

| Metric row | Row kind | Expected value | Comparison basis | Tolerance kind | Tolerance value | Pass meaning |
| --- | --- | --- | --- | --- | ---: | --- |
| `project_status` | `metric_comparison` | `SPARSE_SUCCESS` | Project probe status | `status_only` | n/a | Project QR factor/solve completed for the fixture. |
| `baseline_status` | `dependency_status` | `success` | Dense helper status | `status_only` | n/a | Source-controlled dense reference helper completed. |
| `residual_norm` | `metric_comparison` | `1.7320508075688772` | `abs(project_residual - baseline_residual)` | `absolute` | `1e-10` | Project and baseline agree on the nonzero least-squares residual. |
| `solution_norm` | `metric_comparison` | `2.2360679774997898` | `abs(project_solution_norm - baseline_solution_norm)` | `absolute` | `1e-10` | Project and baseline agree on solution norm. |
| `solution_values` | `metric_comparison` | `2,-1` | Max absolute per-component solution delta | `absolute_per_component` | `1e-10` | Project and baseline agree on solution values. |
| `project_vs_baseline_max_abs_delta` | `metric_comparison` | `<=1e-10` | Max absolute per-component solution delta | `absolute` | `1e-10` | Project and baseline solution vectors agree within tolerance. |

## Expected Study Rows

The selected target should emit exactly these six study row IDs:

```text
comparison_qr_overdetermined_incompatible_4x2_project_status_v1
comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1
comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1
comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1
comparison_qr_overdetermined_incompatible_4x2_solution_values_v1
comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1
```

## Generated Artifact Contract

The target should write artifacts under:

```text
build/comparison/qr_incompatible_ls/
```

Required generated files:

```text
project_observations.tsv
baseline_observations.tsv
dependency_status.tsv
study.tsv
summary.md
manifest.tsv
```

The selected manifest row should use:

- `artifact_pattern=build/comparison/qr_incompatible_ls/study.tsv`;
- `expected_rows=6`;
- `freshness_policy=generated_compare_inputs`;
- `support_tier=local_only` unless later hosted evidence changes the tier;
- Linux/macOS workflow metadata only if those workflows are updated with exact
  artifact paths;
- no Windows metadata unless a later day adds hosted proof for this exact
  target.

## Runner Implication

Existing solve-style study rows already compare project residual against
baseline residual in `comparison_study_rows()`. However,
`project_observation_rows()` and `baseline_observation_rows()` currently mark
solve-style observation residual rows as pass only when
`residual_norm <= residual_tolerance`. That is correct for compatible solves
but wrong for this intentionally incompatible fixture.

Implementation days should add a small target-level contract, such as
`expected_residual_norm`, so project and baseline observation rows pass when
their residuals match the expected nonzero residual within tolerance. This
keeps the generated observation files coherent and prevents the selected
target from carrying failing observation rows while the study comparison rows
pass.

## Claim Boundary

Allowed claim:

> Selected QR incompatible least-squares comparison rows are fresh for
> `qr_overdetermined_incompatible_4x2` against the selected source-controlled
> dense QR reference helper.

Required paired non-claims:

> This does not claim broad QR parity, broad least-squares parity, global
> rank-threshold policy, broad rank-deficient solve behavior, NumPy parity,
> SciPy parity, LAPACK parity, SuiteSparse parity, Eigen parity, broad
> external-library ecosystem parity, Windows report freshness expansion,
> package-manager proof, shared-library ABI proof, performance superiority,
> release proof, or state-of-the-art status.

## Failure Modes

| Failure mode | Expected handling |
| --- | --- |
| Project QR factor or solve fails | Project status row fails and selected freshness reports generated comparison failure. |
| Dense helper missing | Dependency/status diagnostics fail with a clear missing-helper message; absence is not pass evidence. |
| Dense helper emits malformed output | Baseline parser fails with `baseline_malformed_output`. |
| Project residual differs from baseline residual | `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` fails. |
| Project solution differs from baseline solution | `solution_values` and `project_vs_baseline_max_abs_delta` rows fail. |
| Generated rows are stale | Normalizer freshness reports source-commit mismatch and the exact regeneration command. |
| Generated row set is missing, duplicated, or unexpected | Selected comparison freshness reports row-set mismatch. |

## Day 3 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '95,135p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
sed -n '1120,1475p' scripts/run_external_comparison.py
sed -n '1620,1775p' scripts/run_external_comparison.py
sed -n '1935,1975p' scripts/run_external_comparison.py
git diff --check
```

No `.c` or `.h` files were changed on Day 3, so `make format && make lint &&
make test` is not required.
