# Day 4 Harness Extension Design

## Purpose

Day 4 designs how the selected `qr-compatible-ls` family should be wired into
the existing external comparison harness without weakening the current
`qr-minnorm` family.

No code changes are made on Day 4. This document is the implementation
checklist for Day 5.

## Current Harness Shape

`scripts/run_external_comparison.py` currently has a single-target design:

| Current Element | Current Value |
| --- | --- |
| target | `qr-minnorm` |
| fixture key | `qr_underdetermined_minnorm_2x4` |
| generator key | `qr_underdetermined_minnorm_2x4_generator_v1` |
| default output root | `build/comparison/qr_minnorm/` |
| project probe | generated C program calling `sparse_qr_solve_minnorm` |
| baseline helper | `tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4` |
| selected rows | six hardcoded row IDs for `qr_underdetermined_minnorm_2x4` |
| summary title | QR minimum-norm external comparison study |

This shape should be generalized just enough to add `qr-compatible-ls`.

## Target Descriptor Design

Day 5 should introduce a target descriptor concept so both comparison families
share the same output, provenance, validation, and summary pipeline.

Recommended descriptor fields:

| Field | `qr-minnorm` Value | `qr-compatible-ls` Value |
| --- | --- | --- |
| `target` | `qr-minnorm` | `qr-compatible-ls` |
| `fixture_key` | `qr_underdetermined_minnorm_2x4` | `qr_overdetermined_compatible_5x3` |
| `operation` | `minnorm_solve` | `least_squares_solve` |
| `subfamily` | `qr_minnorm` | `qr_compatible_ls` |
| `output_dir` | `build/comparison/qr_minnorm/` | `build/comparison/qr_compatible_ls/` |
| `baseline_value_count` | `6` | `4` |
| `rhs` | `[1.0, 1.0]` | `[2.0, -2.5, 4.0, -0.5, 2.0]` |
| `expected_solution` | `[0.5, 0.5, 0.5, 0.5]` | `[1.0, -2.0, 0.5]` |
| `expected_solution_norm` | `1.0` | `2.2912878474779199` |
| `residual_tolerance` | `1e-10` | `1e-10` |
| `solution_tolerance` | `1e-10` | `1e-10` |
| `project_solve_mode` | `sparse_qr_solve_minnorm` | `sparse_qr_factor` plus `sparse_qr_solve` |
| `claim_scope` | fixture-local QR minimum-norm comparison only | fixture-local QR compatible least-squares comparison only |

The descriptor should own selected row IDs instead of deriving them from a
single global `FIXTURE_KEY`.

## Selected Row Map

The new family should emit these rows under
`build/comparison/qr_compatible_ls/study.tsv`:

| Row ID | Metric | Expected |
| --- | --- | --- |
| `comparison_qr_overdetermined_compatible_5x3_project_status_v1` | `project_status` | `SPARSE_SUCCESS` |
| `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1` | `baseline_status` | `success` |
| `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1` | `residual_norm` | project and baseline residual delta `<=1e-10` |
| `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1` | `solution_norm` | project and baseline norm delta `<=1e-10` |
| `comparison_qr_overdetermined_compatible_5x3_solution_values_v1` | `solution_values` | componentwise project and baseline delta `<=1e-10` |
| `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `<=1e-10` |

Day 5 should preserve the current `qr-minnorm` six-row set exactly.

## Output Path Policy

The new family should write the same artifact names as the current family:

| Artifact | New Path |
| --- | --- |
| project observations | `build/comparison/qr_compatible_ls/project_observations.tsv` |
| baseline observations | `build/comparison/qr_compatible_ls/baseline_observations.tsv` |
| dependency status | `build/comparison/qr_compatible_ls/dependency_status.tsv` |
| study rows | `build/comparison/qr_compatible_ls/study.tsv` |
| summary | `build/comparison/qr_compatible_ls/summary.md` |
| manifest | `build/comparison/qr_compatible_ls/manifest.tsv` |

The implementation may keep `--output-dir` for one target at a time, but the
Makefile should eventually run both targets explicitly with stable output
roots.

## Command Flow

Day 5 should keep the existing command flow:

1. Parse CLI target.
2. Resolve repository root, library, and output directory.
3. Resolve the target descriptor.
4. Reset only the selected target's output directory.
5. Collect source, branch, worktree, project version, platform, Python, and
   compiler provenance.
6. Build or reuse `build/libsparse_lu_ortho.a`.
7. Compile a temporary project probe from descriptor matrix/RHS/solve mode.
8. Run the source-controlled dense baseline helper for the descriptor fixture.
9. Parse project and baseline observations.
10. Emit project observations, baseline observations, dependency status,
    study rows, summary, and manifest.
11. Validate selected rows for the active descriptor.
12. Exit non-zero for any selected row or required dependency failure.

## Project Probe Design

The compatible least-squares probe should use direct deterministic entries for
the selected matrix:

```text
row 0: [1, 0, 2]
row 1: [0, 1, -1]
row 2: [2, -1, 0]
row 3: [1, 1, 1]
row 4: [3, 0, -2]
rhs:   [2, -2.5, 4, -0.5, 2]
```

The probe should:

- construct a `5x3` sparse matrix;
- call `sparse_qr_factor`;
- call `sparse_qr_solve`;
- compute residual norm from `A*x - b`;
- compute solution norm from `x`;
- emit stable key/value lines:
  - `status=SPARSE_SUCCESS`;
  - `residual_norm=<float>`;
  - `solution_norm=<float>`;
  - `solution_values=<comma-separated floats>`.

Do not call `sparse_qr_solve_minnorm` for this family. That function remains
the owner for `qr-minnorm`.

## Baseline Runner Design

The baseline command is:

```sh
python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3
```

The parser must require:

- first line `OK 4`;
- next three values as solution values;
- fourth value as residual norm.

The baseline observation layer should compute or record solution norm
`2.2912878474779199` for parity with project observations.

## Failure Diagnostic Matrix

| Failure Class | Trigger | Required Diagnostic |
| --- | --- | --- |
| `unsupported_target` | Target is not one of the descriptor keys. | Target value and supported target list. |
| `missing_fixture_metadata` | Descriptor fixture entries/RHS are missing or inconsistent. | Target and fixture key. |
| `missing_baseline_helper` | Dense helper path is absent. | Helper path and target. |
| `baseline_command_failed` | Baseline helper exits non-zero. | Command, exit code, and output. |
| `baseline_malformed_output` | Baseline protocol is not `OK <count>` plus numeric values. | Expected count, observed output, and fixture key. |
| `project_build_failed` | Static library or probe compilation fails. | Compiler command and captured output. |
| `project_probe_failed` | Probe exits non-zero or omits required keys. | Probe path or command and captured output. |
| `metric_tolerance_miss` | Selected metric exceeds tolerance or status mismatches. | Row ID, metric, expected, project, baseline, delta, tolerance. |
| `missing_selected_row` | Descriptor row set is incomplete. | Missing row IDs and artifact path. |
| `duplicate_selected_row` | Descriptor row ID appears more than once. | Duplicate row IDs and artifact path. |
| `unexpected_selected_row` | Descriptor emits selected rows outside expected IDs. | Unexpected row IDs and artifact path. |
| `unsupported_claim_boundary` | Claim scope or non-claims are broadened. | Field name and offending wording. |

## Touched Surfaces

| Surface | Expected Day 5+ Role |
| --- | --- |
| `scripts/run_external_comparison.py` | Add descriptor-backed target support and `qr-compatible-ls` generation. |
| `Makefile` | Later run both selected targets from `report-index-comparison-freshness`. |
| `scripts/normalize_report_index.py` | Later extend selected comparison row policy to include both target row sets and artifact paths. |
| `tests/test_normalize_report_index.py` | Later add focused row-set, stale, duplicate, unexpected, fail, and defer tests for the new selected family. |
| `tests/corpus/manifests/report_families.tsv` | Later add or update comparison family metadata for `qr_compatible_ls`. |
| `docs/maintainer_guide.md`, `docs/solver_selection.md`, `README.md`, `tests/corpus/README.md` | Later align public and maintainer wording after generated evidence exists. |
| `.github/workflows/ci.yml` | Later update hosted artifact paths only after runtime and selected freshness behavior are validated. |

No C/header implementation changes are expected for the harness design. If Day
5 discovers a required C/header change, the full gate becomes
`make format && make lint && make test`.

## Validation Plan

Initial Day 5 script implementation should run:

- `python3 scripts/run_external_comparison.py --self-check`;
- `python3 scripts/run_external_comparison.py --target qr-minnorm`;
- `python3 scripts/run_external_comparison.py --target qr-compatible-ls`;
- focused Python syntax check for touched scripts;
- `git diff --check`.

After the normalizer is extended, later days should also run:

- `make report-index-comparison-freshness`;
- `python3 tests/test_normalize_report_index.py`;
- `make docs-check` if documentation changes;
- full `make format && make lint && make test` only if `.c` or `.h` files
  change.

## Completion Check

- Harness changes are scoped to `qr-compatible-ls` and preserve `qr-minnorm`.
- Generated outputs have stable names and reviewable diagnostics.
- Failure classes are defined before implementation.
- Touched surfaces and validation requirements are known before Day 5.
