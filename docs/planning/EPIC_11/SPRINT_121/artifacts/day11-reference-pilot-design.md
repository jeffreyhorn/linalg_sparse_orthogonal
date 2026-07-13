# Sprint 121 Day 11 - Reference Pilot Design

## Purpose

Day 11 designs one bounded external dense-reference comparison lane for Day 12
implementation. The pilot compares singular values from this library's SVD
against a small pure-Python dense reference on one deterministic rectangular
rank-deficient fixture.

This artifact is design-only. No C source, header, Makefile, CMake, CTest,
workflow, package, benchmark, public API, or production source surfaces are
changed by Day 11.

## Scope

| Field | Value |
|---|---|
| Sprint/day | Sprint 121 Day 11 |
| Artifact owner | Sprint 121 SVD external dense-reference pilot |
| Solver or behavior family | SVD singular-value external dense-reference comparison |
| Touched surfaces on Day 11 | Planning artifact and working notes only |
| Planned Day 12 test owner | `tests/test_svd.c` |
| Explicitly out of scope | Broad LAPACK/SciPy/NumPy parity, external package dependency, QR solve parity, partial-SVD parity, low-rank optimality, SuiteSparse fixtures, benchmark timing, public API expansion, CTest count changes, package/install lanes, platform support claims, and state-of-the-art claims. |

## Selected Pilot

| Field | Decision |
|---|---|
| Pilot name | Rectangular dense SVD external singular-value reference |
| Fixture | Deterministic 6x4 dense rectangular matrix with full column rank and mixed signs. |
| Primary comparison | Full SVD singular values from `sparse_svd` against an external pure-Python dense reference. |
| Reference model | Python script computes eigenvalues of `A^T A` with a bounded symmetric Jacobi iteration, then returns sorted square-rooted singular values. |
| External dependency policy | Use only Python standard library; do not require NumPy, SciPy, LAPACK, or system BLAS. |
| Failure model | Python command failure after launch is a test failure; missing `python3` is a skip through the existing external-reference helper; Windows skips explicitly to match existing external-reference tests. |
| Implementation strategy | Add a small Python helper plus one `test_svd.c` test using the existing `tf_read_external_reference_vector` helper. |

## Baseline

| Baseline item | Current value |
|---|---|
| Existing SVD proof owner | `tests/test_svd.c` owns full SVD singular values, reconstruction, orthogonality, rank, pseudoinverse, condition number, dense low-rank, sparse low-rank, and internal-reference partial-SVD checks. |
| Existing external-reference pattern | `tests/test_sparse_lu.c`, `tests/test_chol_csc.c`, and `tests/test_ldlt_csc.c` use Python helpers and `tf_read_external_reference_vector` for bounded external dense references. |
| Existing helper gate | `tests/test_solver_helpers.h` exposes `tf_read_external_reference_vector` when `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` is defined before inclusion. |
| Current product truth references | Sprint 118 product truth and Sprint 121 Days 2-10 artifacts require external references to stay bounded and explicitly non-parity. |
| Current non-claims | Sprint 121 does not claim LAPACK, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, external dense-library, or state-of-the-art numerical parity. |

## Fixture Taxonomy

| Fixture | Symmetry | Definiteness | Rank | Conditioning/scaling | Sparsity pattern | Expected behavior |
|---|---|---|---|---|---|---|
| `svd_rect_fullrank_6x4` | Nonsymmetric rectangular | Not applicable | 4 | Moderate scale, no extreme conditioning | Dense 6x4 | Library full SVD and pure-Python dense reference agree on the four singular values within the local tolerance. |

## Matrix Construction

Planned fixture key: `svd_rect_fullrank_6x4`.

Matrix:

```text
[ 3.0  -1.0   0.0   2.0 ]
[ 0.0   4.0   1.0  -1.0 ]
[ 2.0   0.0   3.0   0.5 ]
[ 5.0   3.0   4.0   1.5 ]
[-1.0   5.0   4.0  -0.5 ]
[ 3.0   4.0   7.0   2.5 ]
```

The last three rows are deterministic linear combinations of the first three
rows, so the fixture is rectangular and rank deficient without using a
diagonal-only spectrum.

Construction rules:

- C test builds the sparse matrix locally in `tests/test_svd.c`.
- Python helper builds the same dense matrix from the fixture key.
- The Python helper computes `A^T A`, diagonalizes it with a bounded Jacobi
  routine, clamps tiny negative roundoff to zero, takes square roots, and
  emits singular values in descending order.
- The C test calls `sparse_svd(A, &svd, 0, 0)` and compares `svd.S`.

## Oracle Or Reference Source

| Oracle/reference | Invocation | Trust boundary | Skip/error handling |
|---|---|---|---|
| External pure-Python dense SVD reference | `python3 tests/svd_external_dense_reference.py svd_rect_fullrank_6x4` | Independent dense arithmetic path for one fixed small fixture; not a LAPACK/SciPy/NumPy oracle and not a broad SVD correctness proof. | Missing `python3` skips through `tf_read_external_reference_vector`; Python script `ERROR` output is a test failure; Windows skips explicitly. |
| Library full SVD | `sparse_svd(A, &svd, 0, 0)` | Product behavior under test. | Allocation or SVD failure is a test failure. |
| Singular-value comparison | Max absolute difference over four singular values. | Bounded value comparison only; does not compare vectors, signs, subspace bases, performance, or low-rank optimality. | Difference above tolerance is a test failure. |

## Tolerance And Acceptance Model

| Metric | Tolerance | Rationale |
|---|---:|---|
| Singular-value max absolute difference | `1e-8` | The fixture is small and moderately scaled, but the pure-Python Jacobi reference and library SVD use different dense arithmetic paths. |
| Singular-value ordering | Descending order | Both reference and library should publish sorted singular values. |
| Rank implication | Do not assert rank from external reference in this pilot | Rank threshold semantics remain owned by Day 8 rank fixtures and `sparse_svd_rank`. |
| Vector/subspace agreement | Not checked | Singular vectors can differ by signs or basis choices in rank-deficient subspaces. |
| Runtime | No timing assertion | External helper runtime is not performance evidence. |

## Planned File And Build Impact

| Surface | Day 12 planned change |
|---|---|
| `tests/svd_external_dense_reference.py` | New pure-Python dense reference helper for the selected fixture. |
| `tests/test_svd.c` | Add one external dense-reference test and local fixture/read helper. |
| `tests/test_solver_helpers.h` | No planned change; reuse existing helper with `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`. |
| Makefile | No planned change because the pilot stays inside existing `test_svd`. |
| CMake | No planned change because the pilot stays inside existing `test_svd`. |
| CTest count | No planned change; existing `test_svd` registration remains the owner. |
| Public docs | No planned change; non-claim wording stays in Sprint 121 artifacts. |

## Planned Test Shape

| Test | Purpose |
|---|---|
| `test_svd_external_dense_reference_rect_fullrank_6x4` | Build the 6x4 full-column-rank fixture, read four external singular values, run `sparse_svd`, and assert the max singular-value difference is below `1e-8`. |

Do not add a broad SVD external-reference helper header from this single pilot.
Keep the fixture key, Python command, Windows skip behavior, and tolerance
visible at the `test_svd.c` call site.

## Focused Validation Checklist

Day 12 will modify `.c` and add a Python test helper, so it must run:

1. `make format`
2. `make build/test_svd && ./build/test_svd`
3. `make lint`
4. `make test`
5. `git diff --check`
6. Focused trailing-whitespace scan over
   `docs/planning/EPIC_11/SPRINT_121`, `tests/test_svd.c`, and
   `tests/svd_external_dense_reference.py`.

Because Makefile, CMake, and CTest registration are not planned to change,
`make source-list-check`, CMake configure/build, and `ctest -N` are not
required unless Day 12 changes the planned file/build impact.

## Unsupported Or Expected-Failure Cases

| Case | Disposition | Reason |
|---|---|---|
| NumPy/SciPy/LAPACK invocation | Unsupported | The pilot must not depend on external numerical packages or imply parity with them. |
| Singular-vector or subspace comparison | Unsupported | Rank-deficient singular-vector bases are not unique and would require separate subspace-owner design. |
| Partial-SVD comparison | Deferred | Current partial-SVD evidence uses internal full SVD and deterministic fixtures; external partial-SVD parity is broader. |
| QR external comparison | Deferred | Day 9 already added deterministic QR least-squares evidence; Sprint 121 selects one SVD external lane. |
| SuiteSparse fixtures | Unsupported | The pilot should remain small, deterministic, and fast. |
| Windows external execution | Skipped | Matches existing external-reference test policy. |
| Performance comparison | Unsupported | External helper timing is not product evidence. |

## Rollback Checklist

If Day 12 validation fails:

1. Remove `tests/svd_external_dense_reference.py`.
2. Remove the external-reference helper/test additions from `tests/test_svd.c`.
3. Re-run `make format && make lint && make test` because `.c` changed.
4. Re-run `git diff --check` and the focused whitespace scan.
5. Record the failed pilot and reason in the Day 12 artifact and Sprint 121
   residual queue.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | None | Do not update. |
| Solver-selection docs | None | Do not update. |
| Examples/tutorial | None | Do not update. |
| Benchmark/performance wording | None | Do not update. |
| Package/platform docs | None | Do not update. |

## Non-Claims Preserved

- The pilot does not prove LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos,
  Eigen, or broad external dense-library parity.
- The pilot does not prove singular-vector, subspace, partial-SVD, low-rank,
  pseudoinverse, least-squares, or QR parity.
- The pilot does not prove performance, scalability, platform support, package
  support, ABI stability, or state-of-the-art behavior.
- The pilot does not add or change public API.

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| Implement the bounded SVD external dense-reference pilot | Sprint 121 Day 12 | This artifact. |
| Decide whether more SVD external fixtures are worth adding | Future SVD oracle owner | Sprint 121 retrospective residual queue. |
| Decide whether QR should receive its own external dense-reference lane | Future QR oracle owner | Day 9 QR least-squares artifact plus this artifact. |
| Keep partial-SVD external parity out of Sprint 121 unless separately designed | Future partial-SVD oracle owner | Day 10 partial-SVD artifact. |

## Completion Check

| Criterion | Status |
|---|---|
| Fixture taxonomy is recorded. | Complete. |
| Oracle or reference trust boundary is recorded. | Complete. |
| Tolerances are explicit. | Complete. |
| Unsupported cases are explicit. | Complete. |
| Validation commands are recorded. | Complete. |
| Drift and non-claims are recorded. | Complete. |
| Residual handoff is recorded. | Complete. |
