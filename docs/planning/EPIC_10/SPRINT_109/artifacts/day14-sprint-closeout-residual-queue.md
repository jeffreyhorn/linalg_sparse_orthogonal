# Day 14 Sprint 109 Closeout & Residual Queue

## Purpose

Day 14 closes Sprint 109 by reconciling outcomes against the project-plan
items, publishing the no-duplicate completed-work list, ordering downstream
residuals, and preparing retrospective-ready notes.

## Item-by-Item Closeout

| Item # | Item | Status | Evidence |
|---:|---|---|---|
| 1 | Residual Debt Intake and Dependency Ordering | Completed | Day 1 re-read Sprint 108 residual debt, excluded completed helpers, and ordered eigensolver, matrix-shell, and giant-test work. |
| 2 | Eigensolver Dense Jacobi Source Boundary | Completed | Days 2-3 defined the private owner, source-list order, Make/CMake parity, and focused validation plan. |
| 3 | Eigensolver Dense Jacobi Extraction | Completed | Day 4 moved only `s21_dense_sym_jacobi` into `src/sparse_eigs_dense_internal.c`; Day 5 validated focused Make and CMake lanes. |
| 4 | Eigensolver Behavior-Sensitive Boundary Audit | Completed as no-move audit | Days 6-7 documented grow-m, refinement, shared kernels, dispatch/defaults, handle/workspace, and shift-invert no-go conditions. |
| 5 | Matrix Shell Candidate Boundary Contract | Completed as future-owner contract | Days 8-9 selected `src/sparse_matrix_io.c` as a future Matrix Market owner and deferred movement behind private builder ownership. |
| 6 | Giant-Test Proof-Owner Cleanup Pass | Completed | Days 10-11 selected and implemented the QR exact-RHS setup cleanup in `tests/test_qr.c`. |
| 7 | Validation, Metrics, and Residual Closeout | Completed | Days 12-14 captured focused integration, full quality gate, metrics, drift evidence, and this residual queue. |

## Completed Work That Must Not Be Duplicated

Downstream sprints should not repeat these completed Sprint 109 outcomes:

- dense Jacobi boundary review for `s21_dense_sym_jacobi`;
- private dense Jacobi extraction into `src/sparse_eigs_dense_internal.c`;
- source-list registration of `src/sparse_eigs_dense_internal.c` in
  `Makefile`, `CMakeLists.txt`, and `build-metadata/library_sources.txt`;
- focused eigensolver Make and CMake validation for the dense Jacobi split;
- grow-m/refinement/shared-kernel no-move audit for Sprint 109;
- dispatch/defaults/handle/workspace/shift-invert no-move audit for Sprint
  109;
- Matrix Market `src/sparse_matrix_io.c` future-owner selection;
- Matrix Market no-move decision due to `SparseBuildEntry` and
  `sparse_matrix_build_from_entries` ownership;
- QR exact-RHS helper cleanup in `tests/test_qr.c`;
- Day 12 focused integration and drift evidence;
- Day 13 full quality gate and metrics evidence.

## Dense Jacobi Final Status

Status: completed.

`s21_dense_sym_jacobi` now has a single implementation owner:

```text
src/sparse_eigs_dense_internal.c
```

The split did not add a public header declaration or install surface. The
source is registered in all reviewed library source-list owners and validated
by focused eigensolver lanes plus the full Day 13 quality gate.

## Eigensolver No-Go Conditions

Status: deferred for future behavior-specific work.

Do not move the following eigensolver areas without a dedicated owner contract
and focused validation:

- grow-m backend orchestration;
- refinement helpers and in-place public result mutation;
- shared Lanczos/MGS/spectrum-selection kernels;
- public default option construction and validation;
- backend dispatch and cleanup control flow;
- public handle/workspace lifecycle;
- shift-invert matrix copy, factorization, inverse-Ritz post-processing, and
  LDLT path reporting.

These areas remain behavior owners, not simple line-count cleanup.

## Matrix-Shell Final Status

Status: future-owner contract completed; code movement deferred.

Future owner candidate:

```text
src/sparse_matrix_io.c
```

Allowed future scope:

- `sparse_save_mm`;
- `sparse_load_mm`;
- Matrix Market parsing/formatting helpers;
- Matrix Market errno and checked stream-write behavior.

Required prerequisite before movement:

- resolve private ownership for `SparseBuildEntry` and
  `sparse_matrix_build_from_entries`, because the builder is shared by
  `sparse_copy`, `sparse_transpose`, and `sparse_load_mm`.

No matrix-shell code changed in Sprint 109.

## Giant-Test Cleanup Final Status

Status: one bounded cleanup completed; remaining proof-owner debt deferred.

Completed:

- added `make_qr_exact_rhs` as a local static setup helper in
  `tests/test_qr.c`;
- replaced seven repeated exact-RHS setup blocks;
- preserved solver calls, rank checks, residual labels, tolerances,
  reconstruction checks, refinement assertions, and QR-vs-LU comparison logic
  at call sites.

Deferred:

- QR sequential RHS fill helper for non-exact least-squares/refinement smoke;
- LDLT CSC external dense-reference oracle cleanup;
- per-solver iterative exact-RHS helper cleanup;
- SVD storage-layout proof-loop cleanup.

## Dependency-Ordered Residual Queue

1. Matrix builder ownership decision:
   define whether `SparseBuildEntry` and `sparse_matrix_build_from_entries`
   become a private builder source before any Matrix Market split.
2. Matrix Market source split:
   move load/save behavior only after builder ownership is resolved, then run
   focused matrix tests and solver-smoke fixture lanes.
3. Eigensolver behavior-owner validation:
   if future movement is desired, select one behavior-sensitive eigensolver
   owner at a time, starting with a narrow private owner and direct validation
   for defaults, dispatch, workspace, or shift-invert semantics.
4. QR sequential RHS cleanup:
   consider a small `tests/test_qr.c` setup-only helper only if it does not
   hide least-squares or refinement proof values.
5. LDLT CSC external oracle cleanup:
   treat as a dedicated oracle-lane review because it couples external Python
   references, Windows skips, LDLT factorization, and dense solve comparison.
6. Iterative exact-RHS cleanup:
   split by solver family rather than introducing one broad helper across CG,
   GMRES, BiCGSTAB, and MINRES proof lanes.
7. SVD proof-loop cleanup:
   preserve storage-layout and orthogonality loops inline unless a future
   helper can keep stride, rank, and residual proof explicit at call sites.

## Validation Closeout

Sprint 109 validation evidence:

- Day 5 focused Make and CMake eigensolver validation passed;
- Day 9 focused matrix and solver-smoke validation passed for the no-move
  Matrix Market decision;
- Day 11 focused QR validation passed;
- Day 12 focused Make and CMake integration validation passed;
- Day 12 CTest registration remained at 54 tests;
- Day 13 `make format && make lint && make test` passed;
- Day 13 final test output reported `All tests passed.`

## Metrics Closeout

Final recorded metrics:

| Metric | Value |
|---|---:|
| `src/sparse_eigs.c` lines | 1412 |
| `src/sparse_eigs_dense_internal.c` lines | 129 |
| `tests/test_qr.c` lines | 3194 |
| Library sources | 46 |
| CTest registrations | 54 |
| `s21_dense_sym_jacobi` implementation owners | 1 |
| `make_qr_exact_rhs` call sites | 7 |
| Public/header diffs | 0 |
| Helper-target changes | 0 |

## Retrospective-Ready Notes

Sprint 109 delivered one low-risk source-boundary extraction and one bounded
giant-test cleanup while intentionally rejecting broader movement without
proof. The most important pattern is that maintainability work remained
reviewable because each movement candidate was either validated as small and
private or deferred with explicit no-go conditions.

Main outcome:

- dense Jacobi now has a private source owner;
- QR repeated exact-RHS setup is consolidated;
- Matrix Market movement has a clear future owner and prerequisite;
- behavior-heavy eigensolver code remains intentionally unmoved;
- all required validation passed.

Primary downstream risk:

- future maintainability work must avoid treating public behavior owners and
  proof-owner tests as mechanical line-count cleanup.

## Completion Criteria Status

- Every Sprint 109 deliverable has a completed, deferred, or rejected status.
- Downstream residuals are dependency-ordered.
- No sprint-exit claim exceeds validation evidence.
- Sprint 109 is ready for retrospective creation.
