# Day 8: Baseline Runner Implementation

## Scope

Day 8 implemented the selected external-process dense baseline path for the
first narrow QR minimum-norm comparison target:
`qr_underdetermined_minnorm_2x4`.

The implementation keeps the Sprint 154 claim boundary intact: the baseline is
the source-controlled dense reference helper executed as a separate process. It
is not NumPy, SciPy, LAPACK, SuiteSparse, Eigen, package-manager, hosted CI,
performance, ABI, or state-of-the-art evidence.

## Implementation

Updated `scripts/run_external_comparison.py` to add:

- baseline helper discovery for `tests/qr_external_dense_reference.py`;
- baseline command execution through the current Python interpreter;
- strict `OK 6` protocol parsing for four solution values, residual norm, and
  solution norm;
- baseline observation rows in
  `build/comparison/qr_minnorm/baseline_observations.tsv`;
- dependency status diagnostics in
  `build/comparison/qr_minnorm/dependency_status.tsv`;
- manifest provenance for baseline command, helper path, Python executable,
  Python version, baseline type, and generated artifact paths;
- fail-closed error classes for missing helper, failed baseline command, and
  malformed baseline output;
- explicit defer rows for NumPy and SciPy optional package baselines.

The runner still writes project observations separately in
`project_observations.tsv`. Project-vs-baseline comparison rows remain a Day 9
handoff item.

## Generated Artifact Shape

The Day 8 run generates:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/manifest.tsv`

Baseline observation rows from local validation:

| Metric | Value | Status | Status reason |
| --- | --- | --- | --- |
| `baseline_status` | `success` | `pass` | `baseline_status_success` |
| `baseline_residual_norm` | `0` | `pass` | `baseline_residual_within_tolerance` |
| `baseline_solution_norm` | `1` | `pass` | `baseline_solution_norm_within_tolerance` |
| `baseline_solution_values` | `0.5,0.5,0.5,0.5` | `pass` | `baseline_solution_values_within_tolerance` |

Dependency diagnostics from local validation:

| Dependency | Status | Required | Meaning |
| --- | --- | --- | --- |
| `python3` | `pass` | `yes` | selected interpreter is available |
| `tests/qr_external_dense_reference.py` | `pass` | `yes` | selected helper exists |
| `numpy` | `defer` | `no` | optional package baseline not selected |
| `scipy` | `defer` | `no` | optional package baseline not selected |

## Validation

Ran:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
```

Result:

- project-side QR minimum-norm scaffold passed;
- baseline QR minimum-norm scaffold passed;
- baseline emitted `success`, residual norm `0`, solution norm `1`, and
  solution values `0.5,0.5,0.5,0.5`;
- NumPy and SciPy remained `defer`, not pass evidence.

## Non-Claims

Day 8 does not claim:

- project-vs-baseline metric agreement;
- NumPy or SciPy parity;
- broad QR parity;
- external-library ecosystem parity;
- package-manager support;
- performance superiority;
- hosted CI coverage;
- shared-library or ABI support.

## Day 9 Handoff

Day 9 should add the selected comparison rows:

- project residual norm versus baseline residual norm;
- project solution norm versus baseline solution norm;
- project solution values versus baseline solution values;
- project-vs-baseline maximum absolute solution delta.

Those rows should fail closed on missing, malformed, duplicate, or
out-of-tolerance selected metrics and should preserve the same claim boundary.
