# Sprint 174 Day 7: Harness Extension Design

## Purpose

Design the narrow generated-comparison harness extension for the selected
linked-list LU fixture family before implementation. Day 7 intentionally does
not add the runner target yet; it fixes the CLI, output paths, row contract,
cleanup behavior, and claim boundary that Day 8 through Day 10 will implement
and wire into freshness checks.

## Existing Entry Points

The selected family should extend the existing generated-comparison runner
instead of creating a parallel harness:

- `scripts/run_external_comparison.py`
  - `TARGETS` owns the supported `--target` names and fixture metadata.
  - `run_project_probe()` builds a temporary C probe against
    `build/libsparse_lu_ortho.a` and parses key-value observations.
  - `run_baseline_reference()` invokes a source-controlled dense-reference
    helper and converts helper output into baseline observations.
  - `comparison_study_rows()` emits the six non-partial-SVD comparison rows:
    project status, baseline status, residual norm, solution norm, solution
    values, and project-vs-baseline max absolute delta.
  - `expected_study_row_ids()` and `validate_selected_study_rows()` enforce
    exact selected-row membership and pass status.
  - `reset_output_dir()` clears generated files in the selected output
    directory before each run.
- `scripts/normalize_report_index.py`
  - `SELECTED_COMPARISON_ROW_IDS` defines generated comparison rows that must
    be present when comparison freshness is required.
  - `SELECTED_COMPARISON_ARTIFACTS` defines the generated `study.tsv` files
    that must exist and match current source provenance.
- `Makefile`
  - `report-index-comparison-freshness` builds the library, regenerates the
    selected comparison targets, and runs report-index freshness checks.
- `tests/corpus/manifests/report_families.tsv`
  - the `comparison` rows document the generator command, artifact pattern,
    support tier, claim scope, non-claims, owner, and introduction sprint.

## Selected Target Contract

Add exactly one generated comparison target:

```text
target: lu-nonsym-square-5
comparison_kind: lu
fixture_key: lu_nonsym_square_5
subfamily: lu_nonsym_square_5
operation: square_solve
output_dir: build/comparison/lu_nonsym_square_5
baseline_helper: tests/lu_external_dense_reference.py
baseline_command: python3 tests/lu_external_dense_reference.py lu_nonsym_square_5
support_tier: local_only
claim_scope: fixture-local linked-list LU square-solve comparison only
```

The target metadata should include the Day 4 fixture values so the runner can
generate the project probe and validate the source-controlled helper without
consulting generated artifacts:

```text
rows: 5
cols: 5
nnz: 19
rhs: 12.5,10.5,18.0,24.0,48.0
expected_solution: 1.0,2.0,3.0,4.0,5.0
expected_solution_norm: 7.416198487095663
residual_tolerance: 1e-10
solution_tolerance: 1e-10
solve_mode: lu_square_solve
pivot_mode: SPARSE_PIVOT_COMPLETE
factor_tolerance: 1e-12
baseline_output_kind: solution_only
baseline_value_count: 5
```

## Baseline Adapter Decision

The existing generic non-partial-SVD baseline path is QR-shaped: it invokes
`tests/qr_external_dense_reference.py` and expects the helper to emit solution
values plus residual metadata. The selected LU helper was intentionally guarded
on Day 6 as a solution-only CLI:

```text
OK 5
1
2
3.0000000000000004
4
4.9999999999999991
```

Day 8 should preserve that helper contract and add a narrow LU baseline adapter
inside `scripts/run_external_comparison.py`. The adapter should:

1. Select `tests/lu_external_dense_reference.py` when
   `comparison_kind == "lu"`.
2. Parse exactly five solution values from the `OK 5` output.
3. Compute `solution_norm` in the runner from the parsed baseline vector.
4. Compute the baseline residual norm in the runner from the target matrix,
   right-hand side, and parsed baseline vector.
5. Emit the same baseline observation keys already consumed by
   `comparison_study_rows()`: `status`, `solution_values`, `residual_norm`,
   `solution_norm`, `baseline_command`, `baseline_helper_path`,
   `baseline_python_executable`, and `baseline_python_version`.

This keeps Day 6's source-controlled helper guard stable and avoids expanding
the dense helper into a second report generator.

## Project Probe Design

Day 8 should extend the project-probe branch with `comparison_kind == "lu"` or
with a dedicated `solve_mode == "lu_square_solve"` source generator. The probe
must:

- construct the Day 4 5x5 sparse matrix with the exact 19 nonzero entries;
- build the right-hand side from the selected target metadata;
- copy the matrix into an LU working matrix if the public API mutates factor
  input;
- call `sparse_lu_factor(..., SPARSE_PIVOT_COMPLETE, 1e-12)`;
- call `sparse_lu_solve(...)`;
- compute residual norm against the original matrix and right-hand side;
- compute solution norm and comma-separated solution values; and
- emit the existing required project observation keys:
  `status`, `residual_norm`, `solution_norm`, and `solution_values`.

The temporary C source remains generated under the existing
`sparse-comparison-*` temp directory and should be removed unless `--keep-temp`
is provided.

## Generated Output Plan

The generated output directory is:

```text
build/comparison/lu_nonsym_square_5/
```

Each run should replace generated files in that directory through the existing
`reset_output_dir()` path and produce:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

These files stay generated and are not source-controlled. The source-controlled
freshness surface is the selected row list, selected artifact list, report
family manifest row, Make target wiring, and documentation.

## Study Row Contract

The target should reuse the current non-partial-SVD `STUDY_FIELDS` schema and
the existing six-row contract:

```text
comparison_lu_nonsym_square_5_project_status_v1
comparison_lu_nonsym_square_5_baseline_status_v1
comparison_lu_nonsym_square_5_residual_norm_v1
comparison_lu_nonsym_square_5_solution_norm_v1
comparison_lu_nonsym_square_5_solution_values_v1
comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1
```

All six rows are hard pass/fail evidence for only the selected fixture, command,
commit, platform, compiler, configuration, artifact, and support tier.

## Claim Boundary

The report-family row and generated study rows must use bounded language:

```text
Generated comparison rows record one local fixture-level linked-list LU
square-solve comparison for lu_nonsym_square_5 against the selected
source-controlled dense reference helper.
```

Required non-claims:

```text
no broad LU correctness; no broad nonsymmetric solve parity; no LU CSR parity;
no sparse-direct solver parity; no pivoting superiority; no factor-layout
identity; no external-library ecosystem parity; no NumPy parity; no SciPy
parity; no LAPACK parity; no SuiteSparse parity; no Eigen parity; no hosted CI
proof; no release proof; no platform portability proof; no package-manager
proof; no shared-library ABI proof; no performance superiority; no
state-of-the-art claim
```

## Make And Report-Index Wiring Plan

Implementation should land in this order:

1. Add the `lu-nonsym-square-5` target and LU adapter to
   `scripts/run_external_comparison.py`.
2. Extend focused runner tests for target registration, self-check row IDs, and
   generated LU output.
3. Add the six LU row IDs to `SELECTED_COMPARISON_ROW_IDS`.
4. Add `build/comparison/lu_nonsym_square_5/study.tsv` to
   `SELECTED_COMPARISON_ARTIFACTS`.
5. Add a `comparison	lu_nonsym_square_5` row to
   `tests/corpus/manifests/report_families.tsv`.
6. Add `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5`
   to `make report-index-comparison-freshness`.

## Validation Plan

No `.c` or `.h` files are expected to change for this harness extension because
the project probe source is generated by Python at runtime. If implementation
does touch C or public headers, the full gate becomes mandatory:

```text
make format
make lint
make test
```

Expected focused checks for the harness implementation:

```text
python3 tests/test_lu_external_dense_reference.py
python3 tests/test_run_external_comparison.py
make build/libsparse_lu_ortho.a
python3 scripts/run_external_comparison.py --target lu-nonsym-square-5
python3 scripts/run_external_comparison.py --self-check
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
make report-index-comparison-freshness
git diff --check
```

Day 7 completion is satisfied when the implementation target, output ownership,
row contract, Make/report-index ownership, and non-claim boundary are explicit
before code changes begin.
