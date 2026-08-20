# Day 5: Comparator Output Design

## Purpose

Design the comparator invocation, generated output schema, fail-closed
semantics, stale-output handling, and report-index integration for the selected
linked-list LU comparison target before implementation.

## Selected Target Contract

| Field | Value |
| --- | --- |
| Runner target | `lu-nonsym-square-5` |
| Report family | `comparison` |
| Subfamily | `lu_nonsym_square_5` |
| Fixture key | `lu_nonsym_square_5` |
| Operation | `square_solve` |
| Output directory | `build/comparison/lu_nonsym_square_5/` |
| Main study artifact | `build/comparison/lu_nonsym_square_5/study.tsv` |
| Baseline helper | `tests/lu_external_dense_reference.py` |
| Baseline type | `external-process-source-controlled-helper` |
| Support tier | `local_only` |

## Comparator Invocation

The baseline comparator command should be:

```sh
python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
```

The helper output contract is:

```text
OK 5
<x0>
<x1>
<x2>
<x3>
<x4>
```

For the selected fixture, the expected values are fixture-local and
tolerance-checked against:

```text
1,2,3,4,5
```

The comparator remains source-controlled and local. It is not NumPy, SciPy,
LAPACK, SuiteSparse, Eigen, or any external package parity proof.

## Project Probe Design

The project probe should follow the existing `scripts/run_external_comparison.py`
temporary C-probe pattern:

1. Build a temporary C probe linked against `build/libsparse_lu_ortho.a`.
2. Construct `lu_nonsym_square_5` from the Day 4 matrix.
3. Compute `b = A * [1, 2, 3, 4, 5]`.
4. Copy `A` into an LU work matrix.
5. Run `sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12)`.
6. Run `sparse_lu_solve(LU, b, x_project)`.
7. Emit key/value observations:
   - `status`
   - `residual_norm`
   - `solution_norm`
   - `solution_values`
   - `project_probe_command`

The existing non-partial-SVD observation parser already expects
`status`, `residual_norm`, `solution_norm`, and `solution_values`. Day 8 can
reuse that path if the LU target provides `expected_solution`,
`expected_solution_norm`, `residual_tolerance`, and `solution_tolerance`.

## Generated Files

The LU target should write the same six local files as existing selected
comparison targets:

| File | Role |
| --- | --- |
| `project_observations.tsv` | Project LU probe observations normalized into rows. |
| `baseline_observations.tsv` | Dense helper observations normalized into rows. |
| `dependency_status.tsv` | Baseline/helper dependency status rows. |
| `study.tsv` | Selected generated comparison rows consumed by report-index normalization. |
| `summary.md` | Human-readable local summary of generated rows and non-claims. |
| `manifest.tsv` | Provenance for target, fixture, source commit, branch, platform, compiler, commands, and generated paths. |

All files remain ignored local generated output under `build/`. None should be
committed as Sprint 174 source evidence.

## `study.tsv` Row Schema

Use the existing `STUDY_FIELDS` schema from `scripts/run_external_comparison.py`:

```text
comparison_row_id
report_family
subfamily
row_kind
fixture_key
operation
metric
baseline_name
baseline_type
baseline_version
baseline_command
baseline_python_executable
baseline_python_version
project_name
project_version
project_command
source_commit
source_branch
worktree_state
platform
compiler
configuration
expected_value
project_value
baseline_value
delta_value
tolerance_kind
tolerance_value
status
status_reason
caveat
artifact_path
generated_at_utc
support_tier
claim_scope
non_claims
```

No new schema fields are needed for the LU target.

## Selected Generated Rows

The LU target should emit exactly six selected rows:

| Row ID | Row kind | Metric | Tolerance kind |
| --- | --- | --- | --- |
| `comparison_lu_nonsym_square_5_project_status_v1` | `metric_comparison` | `project_status` | `status_only` |
| `comparison_lu_nonsym_square_5_baseline_status_v1` | `dependency_status` | `baseline_status` | `status_only` |
| `comparison_lu_nonsym_square_5_residual_norm_v1` | `metric_comparison` | `residual_norm` | `absolute` |
| `comparison_lu_nonsym_square_5_solution_norm_v1` | `metric_comparison` | `solution_norm` | `absolute` |
| `comparison_lu_nonsym_square_5_solution_values_v1` | `metric_comparison` | `solution_values` | `absolute_per_component` |
| `comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1` | `metric_comparison` | `project_vs_baseline_max_abs_delta` | `absolute` |

The selected row list should be added to `SELECTED_COMPARISON_ROW_IDS` only
after the generator can emit them. The selected artifact list should add:

```text
build/comparison/lu_nonsym_square_5/study.tsv
```

## Manifest Design

The LU target manifest should keep the existing manifest fields and set:

| Manifest key | Value |
| --- | --- |
| `target` | `lu-nonsym-square-5` |
| `fixture_key` | `lu_nonsym_square_5` |
| `baseline_name` | `source-controlled-dense-lu-reference` |
| `baseline_type` | `external-process-source-controlled-helper` |
| `baseline_version` | `lu_external_dense_reference.py` |
| `configuration` | `stage=sprint174_lu_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only` |

Source commit, branch, worktree state, platform, compiler, generated time, and
command paths should keep the existing runner behavior.

## Failure Semantics

The implementation must fail closed for:

| Failure | Required behavior |
| --- | --- |
| Unknown target | Raise `unsupported_target`; no generated pass evidence. |
| Missing baseline helper | Raise `missing_baseline_helper`; no generated pass evidence. |
| Baseline command exits nonzero | Raise `baseline_command_failed`; no generated pass evidence. |
| Baseline output malformed | Raise `baseline_malformed_output`; no generated pass evidence. |
| Project probe compile/run failure | Raise `project_probe_failed`; no generated pass evidence. |
| Missing selected row | Raise `missing_selected_row`; freshness fails. |
| Duplicate selected row | Raise `duplicate_selected_row`; freshness fails. |
| Any selected row non-pass | Raise `metric_tolerance_miss`; freshness fails. |
| Stale source commit in generated row | `normalize_report_index.py --check-freshness` fails. |
| Missing selected artifact | `normalize_report_index.py --check-freshness` fails. |
| Unexpected selected row count | `normalize_report_index.py --check-freshness` fails. |

Missing optional packages such as NumPy or SciPy must remain irrelevant to this
target. The LU helper does not depend on optional external packages, so optional
dependency absence cannot manufacture pass evidence.

## Stale-Output Handling

The target should reuse the existing `reset_output_dir(output_dir)` behavior:

- create `build/comparison/lu_nonsym_square_5/` if missing;
- remove prior files directly under that directory before generation;
- write a fresh manifest and study at the current source commit;
- rely on `normalize_report_index.py --family comparison --require-generated
  comparison --check-freshness` to reject stale source commits, missing
  artifacts, row-count mismatches, and non-pass rows.

Day 10 should add a negative proof where practical, such as deleting
`build/comparison/lu_nonsym_square_5/study.tsv` or editing a copied generated
row source commit, then proving freshness fails.

## Report-Index Integration Plan

Implementation should update these surfaces in order:

1. Add `lu-nonsym-square-5` to `scripts/run_external_comparison.py`.
2. Add LU row IDs to `SELECTED_COMPARISON_ROW_IDS`.
3. Add `build/comparison/lu_nonsym_square_5/study.tsv` to
   `SELECTED_COMPARISON_ARTIFACTS`.
4. Add a source-controlled `comparison/lu_nonsym_square_5` row to
   `tests/corpus/manifests/report_families.tsv`.
5. Update `make report-index-comparison-freshness` to run
   `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5`.
6. Update comparison docs after generation and freshness checks pass.

The source-controlled report-family row should use:

- `row_meaning=external_process_dense_reference_comparison`;
- `row_origin=generated_local`;
- `support_tier=local_only`;
- `freshness_policy=generated_compare_inputs`;
- `artifact_pattern=build/comparison/lu_nonsym_square_5/study.tsv`;
- claim scope bounded to linked-list LU on `lu_nonsym_square_5`;
- non-claims excluding LU CSR, direct CSR/CSC public solve API, broad LU,
  broad nonsymmetric parity, package, ABI, platform, performance, release, and
  state-of-the-art claims.

## Validation Plan

Implementation days should use this minimum validation sequence:

```sh
make build/libsparse_lu_ortho.a
python3 scripts/run_external_comparison.py --target lu-nonsym-square-5
python3 scripts/run_external_comparison.py --self-check
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
make report-index-comparison-freshness
```

If `.c` or `.h` files change, the full C quality gate is required:

```sh
make format && make lint && make test
```

## Day 5 Validation

Day 5 is planning-only comparator-output design. No `.c` or `.h` files
changed, so the full C quality gate is not required. `git diff --check` is the
required day-level hygiene check.

## Completion Check

Day 5 completion criteria are met:

- comparator output is schema-stable before implementation;
- stale or missing comparator output has clear failure behavior;
- report integration requirements are known before runner changes.
