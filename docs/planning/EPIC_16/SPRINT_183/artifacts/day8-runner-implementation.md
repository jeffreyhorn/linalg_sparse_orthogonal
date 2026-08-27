# Sprint 183 Day 8: Runner Implementation

## Purpose

Implement the selected Cholesky external comparison target in the local
comparison runner, add focused runner coverage, and inspect local generated
output without staging generated artifacts.

## Runner Changes

Implemented `cholesky-spd-tridiag-5` in
`scripts/run_external_comparison.py`.

Target metadata:

| Field | Value |
| --- | --- |
| `comparison_kind` | `cholesky` |
| `fixture_key` | `cholesky_spd_tridiag_5` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `operation` | `cholesky_spd_solve` |
| `output_dir` | `build/comparison/cholesky_spd_tridiag_5/` |
| `rhs` | `[2.0, 4.0, 6.0, 8.0, 16.0]` |
| `expected_solution` | `[1.0, 2.0, 3.0, 4.0, 5.0]` |
| `expected_solution_norm` | `7.416198487095663` |
| `baseline_value_count` | `5` |
| `solve_mode` | `cholesky_spd_solve` |

Added `cholesky_spd_solve` support to the generated C project probe. The probe
includes `sparse_cholesky.h`, factors with `sparse_cholesky_factor`, solves
with `sparse_cholesky_solve`, and emits the existing solve observation fields.

## Baseline Dispatch

Added Cholesky-specific branches for:

- `baseline_name`;
- `baseline_version`;
- `comparison_configuration`;
- `dependency_status_rows`;
- `run_baseline_reference`.

The LU solve-baseline parser was generalized into
`run_solve_baseline_reference`, now shared by LU and Cholesky. Cholesky invokes:

```text
python3 tests/chol_external_dense_reference.py cholesky_spd_tridiag_5
```

The parser expects `OK 5`, parses five float values, and computes residual and
solution norm from target entries and RHS.

## Focused Tests

Extended `tests/test_run_external_comparison.py` with the Cholesky target
expectations:

- target registration and unsupported-target diagnostics;
- required output files;
- exact fixture key, subfamily, and operation;
- six expected metrics and row IDs;
- dependency row for `tests/chol_external_dense_reference.py`;
- success message;
- manifest and summary behavior.

Report-family metadata validation for Cholesky is temporarily disabled with
`require_report_family_metadata=False` because Day 9 owns
`report_families.tsv` integration. Day 9 must remove that bypass after adding
the manifest row.

## Generated Output Inspection

Generated local output:

```text
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5
```

Generated files:

- `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/study.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/summary.md`;
- `build/comparison/cholesky_spd_tridiag_5/manifest.tsv`.

Inspection summary:

| Field | Observed |
| --- | --- |
| Study rows | 6 |
| Statuses | all `pass` |
| Baseline name | `source-controlled-dense-cholesky-reference` |
| Baseline version | `chol_external_dense_reference.py` |
| Required helper | `tests/chol_external_dense_reference.py` |
| Optional package rows | `numpy=defer`, `scipy=defer` |
| Project residual | `5.7560540319981793e-15` |
| Baseline residual | `5.7560540319981793e-15` |
| Max project-vs-baseline delta | `0` |
| Configuration | `stage=sprint183_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only` |

`git status --short -- build/comparison` reported no staged or unstaged tracked
changes, so generated report artifacts remain ignored and unstaged.

## Validation

| Command | Status |
| --- | --- |
| `python3 scripts/run_external_comparison.py --self-check` | Pass |
| `python3 tests/test_chol_external_dense_reference.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` | Pass |
| `git status --short -- build/comparison` | Pass |
| `git diff --check` | Pass |

No `.c` or `.h` source files were modified. The runner generated and compiled a
temporary C probe, but no source-controlled C quality gate is required for Day
8 under the sprint instructions.

## Day 9 Handoff

Day 9 should:

1. add `comparison/cholesky_spd_tridiag_5` to `report_families.tsv`;
2. add `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to
   `selected_report_targets.tsv`;
3. remove the temporary `require_report_family_metadata=False` Cholesky test
   bypass;
4. update Makefile freshness generation;
5. update Linux/macOS workflow selected-comparison file lists and guards;
6. update manifest, normalizer, and selected workflow tests;
7. run selected comparison freshness and inspect generated output.

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| Selected family can generate deterministic comparison output locally. | Complete | Cholesky output generated with six passing selected rows. |
| Runner tests cover success and important failure paths. | Complete | Focused runner tests and self-check cover registration, row IDs, output files, dependency rows, unsupported target, and selected-row validation failures. |
| Generated local artifacts remain ignored and unstaged. | Complete | `git status --short -- build/comparison` reported no tracked changes. |
