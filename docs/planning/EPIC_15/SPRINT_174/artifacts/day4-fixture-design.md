# Day 4: Fixture Design

## Purpose

Define the exact fixture matrix, right-hand side, expected outputs, tolerance
policy, diagnostics, ownership, and deferred edge cases for the selected
linked-list LU generated comparison family before implementation.

## Selected Fixture

Sprint 174 uses the existing deterministic fixture:

```text
lu_nonsym_square_5
```

The fixture is already defined in:

- `tests/lu_external_dense_reference.py`
- `tests/test_sparse_lu.c`

No new mathematical fixture is needed for Sprint 174. Reusing the existing
fixture avoids accidental drift between C tests and generated comparison
reports.

## Matrix Specification

The selected matrix is:

```text
A =
[
  [ 4.0, -1.0,  0.0,  2.0,  0.5],
  [ 1.5,  5.0, -2.0,  0.0,  1.0],
  [ 0.0,  2.0,  6.0, -1.0,  0.0],
  [ 3.0,  0.0,  1.0,  7.0, -2.0],
  [-1.0,  0.5,  0.0,  2.0,  8.0],
]
```

Fixture properties:

| Field | Value |
| --- | --- |
| Rows | 5 |
| Columns | 5 |
| Nonzeros | 19 |
| Structural class | square nonsymmetric sparse matrix |
| Value class | deterministic double values |
| Solver family | linked-list LU |
| Pivot mode | `SPARSE_PIVOT_COMPLETE` |
| Factor tolerance | `1e-12` |

This fixture is deliberately small. It is a generated comparison correctness
fixture, not a performance, scalability, fill, or broad nonsymmetric corpus
fixture.

## Right-Hand Side And Expected Solution

The expected solution is:

```text
x_true = [1.0, 2.0, 3.0, 4.0, 5.0]
```

The right-hand side is generated as `b = A * x_true`:

```text
b = [12.5, 10.5, 18.0, 24.0, 48.0]
```

The dense reference helper currently returns:

```text
OK 5
1
2
3.0000000000000004
4
4.9999999999999991
```

The fixture-local dense reference diagnostics from Day 4 are:

| Diagnostic | Value |
| --- | ---: |
| infinity norm of dense-reference solution | `4.999999999999999` |
| 2-norm of dense-reference solution | `7.416198487095663` |
| infinity norm of dense-reference residual | `7.105427357601002e-15` |

## Project Solve Path

The generated comparison family should mirror the existing C external LU test:

1. Build `SparseMatrix *A` from the matrix above.
2. Compute `b = A * [1, 2, 3, 4, 5]`.
3. Copy `A` into `LU`.
4. Run `sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12)`.
5. Run `sparse_lu_solve(LU, b, x_project)`.
6. Compare `x_project` with the dense reference solution emitted by
   `tests/lu_external_dense_reference.py lu_nonsym_square_5`.

The generated comparison runner may implement the project solve in a temporary
C harness, following existing `scripts/run_external_comparison.py` patterns,
or by a narrow reusable direct-solve path if Day 5 proves the schema remains
bounded.

## Expected Generated Rows

The selected fixture should emit six generated comparison rows:

| Row ID | Metric | Expected status | Tolerance |
| --- | --- | --- | ---: |
| `comparison_lu_nonsym_square_5_project_status_v1` | `project_status` | `pass` | exact status |
| `comparison_lu_nonsym_square_5_baseline_status_v1` | `baseline_status` | `pass` | exact status |
| `comparison_lu_nonsym_square_5_residual_norm_v1` | `residual_norm` | `pass` | `1e-10` |
| `comparison_lu_nonsym_square_5_solution_norm_v1` | `solution_norm` | `pass` | `1e-10` |
| `comparison_lu_nonsym_square_5_solution_values_v1` | `solution_values` | `pass` | `1e-10` |
| `comparison_lu_nonsym_square_5_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `pass` | `1e-10` |

The row shape intentionally matches existing direct-solve comparison rows for
QR solve fixtures. It should not add pivot layout, factor values, fill, timing,
or singular-status rows in Sprint 174.

## Tolerance Policy

| Check | Tolerance kind | Tolerance value | Failure message should name |
| --- | --- | ---: | --- |
| Project status | exact | `pass` | fixture key, project status, solver call that failed |
| Baseline status | exact | `pass` | fixture key, baseline status, helper command output |
| Residual norm | absolute upper bound | `1e-10` | fixture key, residual norm, tolerance |
| Solution norm | absolute delta | `1e-10` | fixture key, project norm, baseline norm, tolerance |
| Solution values | max absolute delta | `1e-10` | fixture key, vector index, project value, baseline value, tolerance |
| Project vs baseline max delta | max absolute delta | `1e-10` | fixture key, max delta, tolerance |

The threshold matches the existing `tests/test_sparse_lu.c` external
dense-reference assertions. Implementation should not loosen it without a new
artifact explaining the numerical reason.

## Ownership And Storage

| Surface | Owner |
| --- | --- |
| Fixture matrix | Source-controlled helper/test definitions in `tests/lu_external_dense_reference.py` and `tests/test_sparse_lu.c`. |
| Generated comparison target | `scripts/run_external_comparison.py --target lu-nonsym-square-5`. |
| Generated output | ignored local files under `build/comparison/lu_nonsym_square_5/`. |
| Report-family contract | source-controlled row in `tests/corpus/manifests/report_families.tsv`. |
| Required generated row IDs/artifacts | `scripts/normalize_report_index.py`. |
| Freshness target | `make report-index-comparison-freshness`. |
| Public/maintainer wording | README, `tests/corpus/README.md`, `docs/maintainer_guide.md`, and `benchmarks/README.md` if needed. |

## Deferred Edge Cases

| Edge case | Deferred rationale |
| --- | --- |
| `lu_singular_square_4` report family | Existing singular helper/test coverage remains test evidence; Sprint 174 selected one nonsingular solve report family. |
| LU CSR dense solve comparison | Requires public API and CSR/CSC claim boundary design before generated report support. |
| Pivoting strategy comparison | Would invite pivot-internal and factorization-layout claims outside the selected fixture-local solve result. |
| Multiple pivot modes | Selected solve path uses `SPARSE_PIVOT_COMPLETE`; other modes remain outside this report family. |
| Matrix Market nonsymmetric corpus | Larger corpus work would need data, optional-data, and platform/runtime policy. |
| Performance/fill timing | Generated comparison is correctness evidence only. |
| Hosted comparison promotion | Existing reviewed hosted selected comparison lane may later include this target only after workflow and claim updates; Day 4 does not promote hosting. |

## Claim Boundary

The fixture supports only this future claim after implementation and
validation:

> One local generated report family compares linked-list LU on
> `lu_nonsym_square_5` against the source-controlled dense LU helper with
> fixture-local solution and residual tolerances.

It does not support LU CSR correctness, singular solve behavior as a generated
report family, broad linked-list LU correctness, broad nonsymmetric solver
parity, external-library ecosystem parity, package/ABI support, broad platform
support, performance, release, or state-of-the-art claims.

## Day 4 Validation

Day 4 is planning-only fixture design. No `.c` or `.h` files changed, so the
full C quality gate is not required.

Day 4 ran a helper-derived fixture check to record:

- matrix values;
- right-hand side values;
- dense reference solution;
- dense-reference solution norms;
- dense-reference residual norm.

`git diff --check` is the required day-level hygiene check.

## Completion Check

Day 4 completion criteria are met:

- fixture design is precise enough for implementation;
- expected outputs are bounded to `lu_nonsym_square_5`;
- deferred edge cases are documented before code changes.
