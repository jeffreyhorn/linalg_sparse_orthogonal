# Sprint 129 Day 10 Minimum-Norm Helper Ownership Gate

## Purpose

Day 10 decides whether minimum-norm helper movement is safe and
behavior-specific enough for Day 11. The current tests intentionally keep
minimum-norm behavior split by owner: QR solve fixtures, COLAMD/minimum-norm
behavior lanes, SVD pseudoinverse lanes, fallback/refinement/zero-row lanes,
and SuiteSparse submatrix smoke coverage.

The gate does not move code. It defines what Day 11 may move, what must stay
visible at call sites, and what should remain deferred.

## Current Ownership Map

| Owner | File | Current lanes | Must remain visible |
| --- | --- | --- | --- |
| QR solve owner | `tests/test_qr_solve.c` | `test_qr_solve_minnorm_underdetermined_known_solution` and `qr_underdetermined_minnorm_2x4` external dense-reference fixture | Fixture matrix, RHS, expected exact solution, residual, norm tolerance, and external-reference skip/error handling. |
| COLAMD/minimum-norm owner | `tests/test_colamd.c` | Null args, 2 x 4 exact solution, minimality comparison, 3 x 6 exact values, 5 x 10 exact values, COLAMD ordering, rank-deficient, square fallback, 1 x n, refinement, zero-row, QR-vs-SVD-pseudoinverse, and `west0067` submatrix smoke | Behavior name, matrix shape, expected values or comparison candidate, tolerance, norm bound, diagnostics, and owner-specific options. |
| SVD pseudoinverse owner | `tests/test_svd.c` and `tests/test_svd_helpers.h` | Diagonal pseudoinverse, Moore-Penrose condition, rectangular pseudoinverse, underdetermined pseudoinverse minimum-norm solution | Pseudoinverse storage layout, Moore-Penrose condition, SVD tolerance, matrix dimensions, and pseudoinverse-specific diagnostics. |
| Shared QR helpers | `tests/test_qr_helpers.h` | Matrix builders, residual helpers, external-reference reader helpers | QR-specific fixture names and external-reference protocols; no generic QR/SVD minimum-norm API by default. |
| Shared SVD helpers | `tests/test_svd_helpers.h` | `tf_svd_pinv_first_moore_penrose_error` and SVD reconstruction/error helpers | SVD ownership, pseudoinverse layout, and Moore-Penrose interpretation. |

## Duplicate And Near-Duplicate Inventory

| Pattern | Current locations | Movement risk | Day 10 decision |
| --- | --- | --- | --- |
| 2 x 4 exact minimum-norm fixture construction | `tests/test_qr_solve.c`, `tests/test_colamd.c`, `tests/test_svd.c` | The same shape supports QR solve, COLAMD/minimum-norm, and SVD pseudoinverse claims with different tolerances and diagnostics. | Do not create a generic fixture helper on Day 10. Day 11 may extract only a behavior-specific builder if call-site expectations stay visible. |
| `A*x` residual loops | `tests/test_qr_solve.c`, `tests/test_colamd.c` | A generic assertion helper could hide owner-specific tolerance and residual interpretation. | Defer generic residual helper movement. Keep tolerances at call sites. |
| Vector norm checks | `tests/test_colamd.c`, `tests/test_qr_solve.c`, `tests/test_svd.c` | Norm value is part of the behavior claim, not incidental plumbing. | Do not abstract assertion policy; simple local `vec_norm2` use can remain. |
| SVD pseudoinverse application to `b` | `tests/test_colamd.c`, `tests/test_svd.c` | Storage-layout differences are easy to obscure and SVD-pseudoinverse must not become a global QR oracle. | Defer movement unless helper name is explicitly SVD-owned and layout-specific. |
| Moore-Penrose `A*A^+*A` error | `tests/test_svd_helpers.h`, `tests/test_svd.c` | Already correctly SVD-owned. Moving it to QR helpers would blur ownership. | Keep in `tests/test_svd_helpers.h`. |
| SuiteSparse `west0067` submatrix construction | `tests/test_colamd.c` | Corpus submatrix smoke has support-tier/runtime implications and should not be generalized silently. | Keep owner-local unless a future optional/corpus helper gate exists. |

## Candidate Helper Movement Table

| Candidate | Proposed name | Owner | Day 10 decision | Day 11 promotion requirements |
| --- | --- | --- | --- | --- |
| 2 x 4 minimum-norm fixture builder | `tf_qr_minnorm_make_split_constraints_2x4` | QR helper or owner-local static helper | Tentatively promotable | Must preserve expected solution/norm/tolerance at call sites; update QR solve and COLAMD only if duplication is reduced without creating a QR/SVD oracle. |
| 2 x 4 pseudoinverse fixture builder | `tf_svd_pinv_make_split_constraints_2x4` | SVD helper | Deferred | Only useful if multiple SVD pseudoinverse tests consume it; must not be shared back into QR solve as a generic minimum-norm oracle. |
| Minimum-norm residual assertion | `tf_qr_minnorm_assert_exact_solution` | QR/COLAMD helper | Rejected for Day 11 | Would hide tolerances, expected values, and behavior-specific diagnostics. |
| SVD pseudoinverse apply helper | `tf_svd_pinv_apply_column_major_to_rhs` | SVD helper | Deferred | Needs layout-specific name, explicit dimensions, and focused SVD plus COLAMD QR-vs-SVD validation if shared. |
| `west0067` first-30-row underdetermined builder | `tf_colamd_minnorm_make_west0067_first30` | COLAMD owner-local helper | Deferred | Needs corpus support-tier and runtime policy; not a general minimum-norm helper. |
| Generic minimum-norm fixture/assertion helper | `tf_minnorm_*` | Shared | Rejected | Name is too broad and would blur QR solve, COLAMD, SVD pseudoinverse, fallback, refinement, zero-row, and SuiteSparse semantics. |

## Day 11 Acceptance Checklist

Day 11 may move at most one helper, and only if every item below is satisfied
before editing code:

1. Helper name encodes behavior and owner; no generic `tf_minnorm_*` or
   QR/SVD-neutral minimum-norm helper.
2. Call sites keep matrix shape, RHS, expected solution or norm bound,
   tolerance, and diagnostic text visible.
3. The helper does not turn SVD pseudoinverse into a global QR oracle.
4. Fallback, refinement, zero-row, COLAMD ordering, rank-deficient, and
   SuiteSparse lanes remain owner-specific.
5. Public headers, package headers, ABI, CMake, and Makefile source lists do
   not change unless a separate build/API gate is explicitly satisfied.
6. Focused validation is pinned before edits:
   `make build/test_qr_solve && ./build/test_qr_solve`,
   `make build/test_colamd && ./build/test_colamd`, and
   `make build/test_svd && ./build/test_svd` when SVD-owned behavior is
   touched.
7. If any `.c` or `.h` file changes, run the full gate:
   `make format && make lint && make test`.

If any item is missing, Day 11 must explicitly defer helper movement.

## Call-Site Tolerance Policy

- Exact small QR/COLAMD minimum-norm fixtures keep expected component values
  and norm tolerances at the call site.
- Residual checks keep their tolerance and norm/residual interpretation at the
  owner call site.
- SVD pseudoinverse checks keep storage-layout comments and Moore-Penrose
  metric interpretation at the SVD call site.
- QR-vs-SVD pseudoinverse comparisons may use SVD output as a bounded
  cross-check for the named fixture only; they do not create a global SVD
  oracle for QR minimum-norm behavior.
- SuiteSparse submatrix checks keep support-tier, runtime, shape, and norm
  bound diagnostics in the COLAMD/minimum-norm owner.

## Files Changed

| File | Change |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 10 helper ownership notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day10-minnorm-helper-ownership-gate.md` | Recorded minimum-norm ownership map, movement candidates, acceptance checklist, tolerance policy, and non-claims. |

No C source, header, Python helper, Matrix Market data, build file, maintainer
guide, public API, or public wording file changed for Day 10.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 10. The day defines an internal
helper movement gate and does not add a new accepted evidence lane, helper
protocol, public API behavior, external fixture key, or user-visible claim.

## Validation

Day 10 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No code quality gate is required for Day 10 because no `.c` or `.h` file
changed for this day.

## Non-Claims Preserved

- No generic QR/SVD/minimum-norm helper API.
- No public API, package, ABI, CMake, Makefile, CI, CTest, or install-header
  claim.
- No broad QR minimum-norm, SVD pseudoinverse, COLAMD, fallback, refinement,
  zero-row, SuiteSparse corpus, optional-data, platform, backend, performance,
  scalability, or memory claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No helper candidate hides solver-specific behavior or tolerances. | Complete | Candidate table rejects generic assertion helpers and requires visible shape, expected values, tolerances, and diagnostics at call sites. |
| QR solve, COLAMD, SVD, fallback, refinement, zero-row, and SuiteSparse lanes remain behavior-specific. | Complete | Ownership map and acceptance checklist keep each lane in its current behavior owner unless a narrow Day 11 helper passes all gates. |
| Full validation requirements are explicit before any code movement. | Complete | Day 11 checklist defines focused QR solve/COLAMD/SVD validation and full quality gate for any `.c` or `.h` edits. |
