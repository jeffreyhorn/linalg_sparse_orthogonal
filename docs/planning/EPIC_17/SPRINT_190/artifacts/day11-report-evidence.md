# Sprint 190 Day 11: Report Evidence

## Purpose

Regenerate and inspect selected Cholesky comparison evidence for the Sprint 190
Windows workflow decision, while keeping local evidence separate from hosted
Windows pass evidence.

## Regeneration Commands

The selected Cholesky comparison report was regenerated locally through the new
CMake probe path:

```sh
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake
```

The target-specific freshness guard then passed:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

The freshness output reported six generated Cholesky rows as fresh against
`HEAD` and did not require QR, LU, partial-SVD, oracle, benchmark, or broad
report-index artifacts.

## Generated Local Evidence

Generated directory:

```text
build/comparison/cholesky_spd_tridiag_5/
```

Generated files:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

Generated study rows:

| Row ID | Status |
| --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `pass` |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `pass` |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `pass` |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `pass` |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `pass` |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `pass` |

## Provenance Snapshot

| Field | Value |
| --- | --- |
| `source_commit` | `4155eee320cea528513603130da41bf887de6d7b` |
| `source_branch` | `sprint-190` |
| `worktree_state` | `dirty` |
| `platform` | `darwin-x86_64` |
| `compiler` | `cmake-probe:default:Release` |
| `target` | `cholesky-spd-tridiag-5` |
| `fixture_key` | `cholesky_spd_tridiag_5` |
| `study_path` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |

The dirty worktree state is expected because Sprint 190 changes are still
uncommitted. It is not hosted Windows evidence.

## Dependency Snapshot

| Dependency | Status | Required | Reason |
| --- | --- | --- | --- |
| `python3` | `pass` | `yes` | `selected_interpreter_available` |
| `tests/chol_external_dense_reference.py` | `pass` | `yes` | `baseline_helper_available` |
| `numpy` | `defer` | `no` | `optional_package_baseline_not_selected` |
| `scipy` | `defer` | `no` | `optional_package_baseline_not_selected` |

Optional dependency defers remain context only and are not pass evidence.

## Consistency Checks

The generated report matches the selected decision record:

- exactly six selected Cholesky rows;
- all six rows pass;
- `source_commit` matches current `HEAD`;
- artifact path is the selected Cholesky `study.tsv`;
- CMake-probe provenance is recorded;
- optional NumPy/SciPy rows remain deferred;
- no unselected selected comparison target is required by the target-specific
  freshness command.

The test suite also verifies that stale Cholesky output, failed Cholesky rows,
row-set mismatches, broad upload paths, missing upload files, reused artifact
names, and unsupported Windows target promotion fail clearly.

## Residual Risks

- Local CMake-probe success on macOS is not hosted Windows pass evidence.
- `selected_report_targets.tsv` still omits `windows`; manifest promotion is a
  separate review surface.
- The hosted `windows-2022` job must pass before the lane can be cited as
  reviewed Windows selected Cholesky freshness evidence.
- The generated `build/` evidence remains ignored local output and is not
  committed.

## Validation

Commands run:

- `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused Day 11 validation commands passed.

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` C gate is not required for Day 11.
