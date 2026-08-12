# First Narrow QR Minimum-Norm Comparison Study

## Scope

This source-controlled study snapshot publishes the first Sprint 154 narrow
external comparison result.

It covers one fixture-local QR minimum-norm solve:

- target: `qr-minnorm`;
- fixture: `qr_underdetermined_minnorm_2x4`;
- operation: `minnorm_solve`;
- project lane: `sparse_qr_solve_minnorm`;
- baseline lane: source-controlled external-process dense reference helper
  `tests/qr_external_dense_reference.py`.

Reproduce the generated local artifacts with:

```sh
make report-index-comparison-freshness
```

Generated outputs remain ignored local artifacts under
`build/comparison/qr_minnorm/`. This source-controlled study is the reviewable
publication snapshot.

## Provenance Snapshot

| Field | Value |
| --- | --- |
| Source branch | `sprint-154` |
| Source commit at generation | `0cd072882197d7b16fc3da7d27fac04aff883a50` |
| Worktree state at generation | `dirty` |
| Project version | `2.2.0` |
| Platform | `darwin-x86_64` |
| Compiler | `Apple clang version 11.0.0 (clang-1100.0.33.17)` |
| Baseline Python | `3.14.5 (main, May 10 2026, 10:21:34) [Clang 17.0.0 (clang-1700.6.4.2)]` |
| Baseline helper | `tests/qr_external_dense_reference.py` |
| Generated artifact root | `build/comparison/qr_minnorm/` |

The dirty worktree state is explicit provenance for this development-branch
publication snapshot. It is not release proof.

## Selected Rows

| Metric | Project value | Baseline value | Delta | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | | | status-only | `pass` |
| `baseline_status` | | `success` | | status-only | `pass` |
| `residual_norm` | `1.5700924586837752e-16` | `0` | `1.5700924586837752e-16` | `1e-10` absolute | `pass` |
| `solution_norm` | `0.99999999999999989` | `1` | `1.1102230246251565e-16` | `1e-10` absolute | `pass` |
| `solution_values` | `0.49999999999999989,0.49999999999999989,0.5,0.5` | `0.5,0.5,0.5,0.5` | `1.1102230246251565e-16` | `1e-10` absolute per component | `pass` |
| `project_vs_baseline_max_abs_delta` | `0.49999999999999989,0.49999999999999989,0.5,0.5` | `0.5,0.5,0.5,0.5` | `1.1102230246251565e-16` | `1e-10` absolute | `pass` |

All six selected generated comparison rows passed.

## Dependency Status

| Dependency | Status | Required | Interpretation |
| --- | --- | --- | --- |
| `python3` | `pass` | yes | The selected interpreter was available for the source-controlled helper. |
| `tests/qr_external_dense_reference.py` | `pass` | yes | The selected dense reference helper was available. |
| `numpy` | `defer` | no | Optional package baseline was not selected; this is not pass evidence. |
| `scipy` | `defer` | no | Optional package baseline was not selected; this is not pass evidence. |

## Interpretation

This study supports only this fixture-local statement:

`sparse_qr_solve_minnorm` agrees with the selected source-controlled dense
reference helper for `qr_underdetermined_minnorm_2x4` on project status,
baseline status, residual norm, solution norm, solution values, and maximum
absolute project-vs-baseline solution delta under the recorded command,
commit, platform, compiler, and local-only support tier.

## Non-Claims

This study does not claim:

- broad QR parity;
- NumPy parity;
- SciPy parity;
- LAPACK parity;
- SuiteSparse parity;
- Eigen parity;
- external-library ecosystem parity;
- hosted CI proof;
- release proof;
- platform portability proof;
- package-manager proof;
- shared-library or ABI proof;
- performance superiority;
- state-of-the-art status.

## Residual Comparative Gaps

The following comparison lanes remain deferred:

- QR comparison beyond `qr_underdetermined_minnorm_2x4`;
- optional NumPy baseline;
- optional SciPy baseline;
- LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and other ecosystem baselines;
- QR raw Q/R basis, sign/orientation, pivot-order, and rank-threshold policy
  comparisons;
- broad rank-deficient, nullspace, economy-mode, reorder, and sparse-mode QR
  comparisons;
- partial-SVD comparison publication in the same normalized `comparison`
  family;
- portable runtime or performance comparison;
- hosted CI comparison publication;
- package-manager, shared-library, loader, and ABI comparison lanes.
