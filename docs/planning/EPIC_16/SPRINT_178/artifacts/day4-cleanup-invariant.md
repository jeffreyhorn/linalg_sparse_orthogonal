# Sprint 178 Day 4: Cleanup Invariant Record

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Purpose

Document the cleanup behavior, no-publication rules, retry semantics, and
unsupported breadth for the selected `sparse_matmul()` workspace allocation
proof before implementation and regression tests begin.

## Selected Scope

| Field | Value |
| --- | --- |
| Selected subsystem | `sparse_matmul()` workspace allocation |
| Public entry point | `sparse_err_t sparse_matmul(const SparseMatrix *A, const SparseMatrix *B, SparseMatrix **C)` |
| Owner file | `src/sparse_matrix.c` |
| Declaration owner | `include/sparse_matrix.h` |
| Expected test owner | `tests/test_matmul.c` or a focused matrix allocation-failure test if a separate executable is justified |
| Selected failure sites | `acc`, `nz_flag`, and `touched` workspace allocations after `sparse_create(m, nc)` succeeds |

## Ownership Trace

The selected path has this ownership sequence:

1. Validate `C`.
2. Set `*C = NULL`.
3. Validate `A`, `B`, and shape compatibility.
4. Allocate temporary output matrix `out` with `sparse_create(m, nc)`.
5. Allocate local workspace arrays:
   - `acc` with `sparse_calloc_idx_array`;
   - `nz_flag` with `sparse_calloc_idx_array`;
   - `touched` with `sparse_malloc_idx_array`.
6. On selected workspace allocation failure, free any allocated workspace,
   free `out`, and return `SPARSE_ERR_ALLOC`.
7. On success, populate `out`, free workspace, publish `out` through `*C`,
   and return `SPARSE_OK`.

## Public-State Invariants

| Invariant | Required behavior |
| --- | --- |
| Null output pointer | If `C == NULL`, return `SPARSE_ERR_NULL` and do not perform selected allocations. |
| Output initialization | If `C != NULL`, set `*C = NULL` before validating input matrices or allocating output/workspace. |
| Null inputs | If `A == NULL` or `B == NULL`, return `SPARSE_ERR_NULL` and leave `*C == NULL`. |
| Shape mismatch | If `A->cols != B->rows`, return `SPARSE_ERR_SHAPE` and leave `*C == NULL`. |
| Selected workspace allocation failure | Return `SPARSE_ERR_ALLOC` and leave `*C == NULL`. |
| Successful retry | After `sparse_alloc_test_reset()`, the same valid `A` and `B` inputs can be multiplied successfully. |
| Input matrix ownership | `A` and `B` remain caller-owned and reusable after selected injected failures. |
| Success publication | `*C` is assigned only after all selected workspace allocations and product construction succeed. |

## Internal Cleanup Invariants

| Failure site | Required cleanup |
| --- | --- |
| `acc` allocation failure | `acc`, `nz_flag`, and `touched` are either `NULL` or freed safely; `out` is freed; `*C` remains `NULL`. |
| `nz_flag` allocation failure | `acc` is freed; `nz_flag` and `touched` are either `NULL` or freed safely; `out` is freed; `*C` remains `NULL`. |
| `touched` allocation failure | `acc` and `nz_flag` are freed; `touched` is either `NULL` or freed safely; `out` is freed; `*C` remains `NULL`. |
| Later product construction error | Local workspace is freed; `out` is freed; `*C` remains `NULL`. This path is adjacent but not the selected Sprint 178 allocation-failure proof. |
| Success | `acc`, `nz_flag`, and `touched` are freed; `out` is transferred to caller through `*C`. |

## Retry Semantics

The regression suite should use this retry pattern for each selected failure
site:

1. Build small valid input matrices `A` and `B`.
2. Configure the private fail-at-count hook for the selected workspace
   allocation site.
3. Call `sparse_matmul(A, B, &C)`.
4. Assert `SPARSE_ERR_ALLOC`.
5. Assert `C == NULL`.
6. Reset the hook with `sparse_alloc_test_reset()`.
7. Call `sparse_matmul(A, B, &C)` again.
8. Assert `SPARSE_OK`.
9. Assert the expected numeric product.
10. Free `C`, `A`, and `B`.

The tests should not add public allocation-failure APIs. They may include the
private internal allocation header from test code if that follows the Sprint
176 pattern.

## Failure-Count Planning Notes

Day 5 should confirm exact fail-at counts, but the intended selected
workspace sites are ordered as:

| Selected site | Relative target after output matrix creation |
| --- | --- |
| `acc` | first selected workspace helper allocation |
| `nz_flag` | second selected workspace helper allocation |
| `touched` | third selected workspace helper allocation |

Because `sparse_create(m, nc)` also uses wrapped helper allocations, the
harness design must either account for those earlier helper calls or introduce
a private, subsystem-local helper that targets the selected workspace block
without changing public API.

## Unsupported Breadth

This proof does not cover:

- `sparse_create()` shell allocation;
- `sparse_insert()` node/slab allocation during product flush;
- all sparse matrix operations;
- matrix copy, transpose, CSR/CSC conversion, or build-helper allocation;
- direct solvers or factorization paths;
- QR, LDLT, Cholesky, SVD, eigensolver, graph, reorder, or iterative solver
  workspaces;
- package/install flows;
- generated-report or generated API tooling;
- state-of-the-art reliability or external-library parity.

## Documentation Boundary

If the Sprint 178 proof passes, public and maintainer wording may say only:

`sparse_matmul()` workspace allocation has deterministic allocation-failure
cleanup evidence with no partial output publication and successful retry after
reset.

Wording must not say or imply broad matrix, solver, package, generated-tool,
or whole-library allocation-failure coverage.

## Testability Checklist

| Assertion | Testable without public API changes? | Notes |
| --- | --- | --- |
| selected failure returns `SPARSE_ERR_ALLOC` | Yes | Use private allocation hook from tests. |
| `*C` remains `NULL` | Yes | Initialize local output pointer and inspect after failure. |
| inputs remain reusable | Yes | Retry multiplication after reset using same `A` and `B`. |
| retry succeeds | Yes | Assert `SPARSE_OK` and expected entries. |
| workspace is freed | Indirect locally; direct leak proof depends on sanitizer/Valgrind outside this sprint's guaranteed local checks. | Day 7-8 tests should still assert no stale output and successful retry. |
| temporary `out` is freed | Indirect locally; direct leak proof depends on sanitizer/Valgrind outside this sprint's guaranteed local checks. | Cleanup path already calls `sparse_free(out)`; tests can prove no publication and retry. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Cleanup assertions can be converted directly into tests | Complete | Public-state and internal cleanup invariant tables name the expected assertions for each selected failure site. |
| Retry behavior is observable without changing public API | Complete | Retry semantics use existing public `sparse_matmul()` plus private test hook reset. |
| Broad allocation-failure non-claims are preserved | Complete | Unsupported breadth and documentation boundary reject adjacent broad claims. |
