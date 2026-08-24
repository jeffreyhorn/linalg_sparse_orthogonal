# Sprint 178 Day 3: Subsystem Selection Detail

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Purpose

Select exactly one subsystem for Sprint 178 allocation-failure proof and
freeze the entry points, allocation sites, cleanup paths, out-of-scope
surfaces, and retained non-claims before implementation begins.

## Selection Decision

Sprint 178 selects:

**Selected subsystem:** `sparse_matmul()` workspace allocation in
`src/sparse_matrix.c`.

This is the selected subsystem for deterministic allocation-failure proof
batch 2. No direct solver, decomposition, eigensolver, package/install, or
generated-tooling allocation path is selected by this decision.

## Selection Rationale

| Criterion | Assessment |
| --- | --- |
| User value | High. Sparse matrix multiplication is a public matrix operation and a building block for examples, solver validation, reconstruction tests, and downstream users. |
| Public state exposure | High. The public contract returns an error code and writes the output through `SparseMatrix **C`; failure should not publish a partial result. |
| Deterministic failure sites | Strong. The selected workspace arrays use wrapped allocation helpers that already consult the private fail-at-count hook. |
| Cleanup observability | Strong. On selected workspace allocation failure, the function frees any partial workspace arrays, frees the temporary output matrix, and returns `SPARSE_ERR_ALLOC` before assigning `*C`. |
| Retry feasibility | Strong. Tests can reuse the same input matrices after reset and verify the product succeeds. |
| Implementation risk | Medium-low. The proof can target existing wrapped allocation sites without converting broad solver raw allocations. |
| Claim risk | Low if wording names only `sparse_matmul()` workspace allocation and preserves all broad allocation-failure non-claims. |

## In-Scope Entry Point

| Entry point | File | Scope |
| --- | --- | --- |
| `sparse_matmul(const SparseMatrix *A, const SparseMatrix *B, SparseMatrix **C)` | `src/sparse_matrix.c`; declaration in `include/sparse_matrix.h` | Public sparse matrix-matrix multiplication output/workspace allocation behavior. |

## Selected Failure Sites

The selected deterministic failure sites are the three workspace allocations
after the temporary output matrix is created:

| Failure site | Allocation helper | Expected failure behavior |
| --- | --- | --- |
| accumulator workspace | `sparse_calloc_idx_array(nc, sizeof(sparse_scalar_t), &acc)` | Return `SPARSE_ERR_ALLOC`, free any local workspace allocated before failure, free the temporary output matrix, and leave `*C == NULL`. |
| nonzero flag workspace | `sparse_calloc_idx_array(nc, sizeof(int), &nz_flag)` | Return `SPARSE_ERR_ALLOC`, free `acc`, free the temporary output matrix, and leave `*C == NULL`. |
| touched-column workspace | `sparse_malloc_idx_array(nc, sizeof(idx_t), &touched)` | Return `SPARSE_ERR_ALLOC`, free `acc`, free `nz_flag`, free the temporary output matrix, and leave `*C == NULL`. |

The selected failure sites intentionally start after `sparse_create(m, nc)`
succeeds. `sparse_create()` shell allocation is a separate candidate subsystem
and is not selected for this sprint.

## Ownership Paths In Scope

| Owner | Expected rule |
| --- | --- |
| input matrices `A` and `B` | Inputs remain owned by the caller and must remain reusable after injected workspace failure. |
| output pointer `C` | `sparse_matmul()` sets `*C = NULL` before validation/allocation and must leave it `NULL` on selected workspace allocation failure. |
| temporary output matrix `out` | Owned by `sparse_matmul()` until success; must be freed on selected workspace allocation failure. |
| `acc` | Local workspace; freed on later selected workspace allocation failures and normal success. |
| `nz_flag` | Local workspace; freed on later selected workspace allocation failures and normal success. |
| `touched` | Local workspace; freed on selected failure paths after allocation and normal success. |

## Required Test Shape

Sprint 178 regression coverage should prove:

1. a deterministic injected failure at each selected workspace allocation site
   returns `SPARSE_ERR_ALLOC`;
2. `*C` remains `NULL` after each selected injected failure;
3. input matrices can still be used after each failure;
4. a retry after `sparse_alloc_test_reset()` succeeds and returns the expected
   product;
5. no test relies on public allocation-failure API.

## Out-Of-Scope Allocation Paths

| Out-of-scope path | Reason |
| --- | --- |
| `sparse_create()` shell allocation | This is a separate matrix lifecycle subsystem and would broaden the proof beyond `sparse_matmul()` workspace allocation. |
| `sparse_insert()` node/slab allocation during product flush | Node allocation uses matrix internals and may require separate observability; selected Day 3 proof stays on deterministic workspace allocations. |
| Matrix copy, transpose, CSR/CSC conversion, and build helpers | These remain future matrix allocation candidates. |
| LU CSR conversion/factorization | High-value but many raw allocations remain outside the existing hook. |
| LDLT, QR, Cholesky, SVD, and eigensolver workspaces | Valuable future candidates, but broader cleanup graphs and raw allocations make them poor fit for this bounded proof. |
| Package/install flows and generated-report tooling | Outside the Sprint 178 selected subsystem and outside current allocation hook scope. |

## Retained Non-Claims

Sprint 178 may claim only that `sparse_matmul()` workspace allocation has
deterministic allocation-failure cleanup evidence after the proof passes.
It must not claim:

- broad allocation-failure safety for matrix construction;
- broad allocation-failure safety for all sparse matrix operations;
- direct solver allocation-failure coverage;
- eigensolver allocation-failure coverage;
- QR, LDLT, Cholesky, SVD, or LU workspace coverage;
- package/install allocation-failure coverage;
- generated-tooling allocation-failure coverage;
- state-of-the-art reliability or external-library parity.

## Day 4 Cleanup-Invariant Inputs

Day 4 should turn this selection into explicit invariants:

- `C == NULL` returns `SPARSE_ERR_NULL` before any selected allocation path;
- invalid or shape-mismatched inputs still return their documented errors
  before selected workspace allocation;
- selected workspace allocation failure returns `SPARSE_ERR_ALLOC`;
- `*C` remains `NULL` on selected workspace allocation failure;
- `A` and `B` remain reusable after selected injected failure;
- temporary `out` is freed on selected workspace allocation failure;
- retry after reset succeeds and produces the expected sparse product.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Exactly one subsystem is selected | Complete | Selected subsystem is `sparse_matmul()` workspace allocation only. |
| Selected failure sites are deterministic and testable | Complete | Failure sites are the wrapped `acc`, `nz_flag`, and `touched` workspace allocations. |
| Scope does not imply broad allocation-failure coverage | Complete | Out-of-scope paths and retained non-claims reject broader matrix, solver, package, generated-tooling, and reliability claims. |
