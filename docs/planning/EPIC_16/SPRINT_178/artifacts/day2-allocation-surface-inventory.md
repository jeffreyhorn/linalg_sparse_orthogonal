# Sprint 178 Day 2: Allocation Surface Inventory

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Purpose

Inventory candidate allocation-heavy subsystems and compare their fitness for
the Sprint 178 deterministic allocation-failure proof. Day 2 does not select
the subsystem; it creates a comparable evidence base for Day 3 selection.

## Inventory Inputs

- `src/sparse_alloc_internal.c`
- `src/sparse_alloc_internal.h`
- `tests/test_iterative.c`
- `tests/test_iterative_handle_helpers.h`
- `Makefile`
- `CMakeLists.txt`
- matrix surfaces: `src/sparse_matrix.c`,
  `src/sparse_matrix_build_internal.c`, `src/sparse_csr.c`
- direct solver surfaces: `src/sparse_lu.c`, `src/sparse_lu_csr.c`,
  `src/sparse_ldlt.c`, `src/sparse_ldlt_csc.c`, `src/sparse_qr.c`
- decomposition/workspace surfaces: `src/sparse_ldlt_csc.c`,
  `src/sparse_qr.c`, `src/sparse_svd.c`, `src/sparse_svd_partial.c`

## Current Hook Reachability

| Surface | Current state | Sprint 178 implication |
| --- | --- | --- |
| Wrapped allocation helpers | `sparse_malloc_array`, `sparse_calloc_array`, `sparse_malloc_idx_array`, and `sparse_calloc_idx_array` consult the private fail-at-count hook. | Paths already using these helpers can usually be tested with the existing deterministic hook. |
| Raw `malloc`/`calloc`/`realloc` | Many solver and decomposition files allocate directly. | These paths are not currently reached by the private hook and would require a scoped conversion or local harness. |
| Hook semantics | `remaining == 0` fails the next wrapped allocation once, then resets; positive counts decrement before the single failure. | Tests can target exact wrapped allocation order, but should avoid relying on broad incidental allocation counts. |
| Public API exposure | Hook controls remain private/internal in `src/sparse_alloc_internal.*`. | Sprint 178 should not add public test-injection API. |

## Candidate Surface Matrix

| Candidate | Owner files | Public state exposure | Cleanup observability | Retry feasibility | Hook reachability | Implementation risk | Day 2 fitness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix shell allocation | `src/sparse_matrix.c`, public matrix headers, `tests/test_sparse_matrix.c` | High: `sparse_create()` returns either a usable matrix or `NULL`. | High: shell buffers are zeroed/freed through `sparse_matrix_free_shell_buffers()`. | High: retrying `sparse_create()` after reset should produce a valid matrix. | Strong for row/column headers and permutation arrays because shell buffers use wrapped helpers; initial struct allocation uses raw `malloc`. | Low to medium. | Strong candidate for deterministic proof. |
| Matrix copy/conversion workspace | `src/sparse_matrix.c`, `src/sparse_matrix_build_internal.c`, `src/sparse_csr.c`, matrix tests | Medium to high: failures should not mutate existing input matrices or publish partial output. | Medium: output objects can be checked for `NULL`/error and inputs can be rechecked. | High for retrying copy/multiply/conversion after reset. | Mixed: several workspace arrays use wrapped helpers, while node slabs and some construction paths use raw allocation. | Medium. | Good candidate if limited to wrapped workspace paths. |
| Matrix multiplication workspace | `src/sparse_matrix.c`, matrix arithmetic tests | High: `*C` should not receive stale or partial output on workspace failure. | High: accumulator arrays and output matrix cleanup are explicit. | High: same inputs can be multiplied again after reset. | Strong for accumulator arrays through wrapped helpers; output matrix shell also uses wrapped helpers after raw struct allocation. | Medium. | Strong candidate if public output no-publication is easy to assert. |
| LU CSR conversion/factorization | `src/sparse_lu_csr.c`, `src/sparse_lu.c`, LU tests | High: factor objects and solver state are user-visible through solve/factor APIs. | Medium: many cleanup paths exist, but ownership is broader. | Medium: retry is likely feasible but must account for factorization state. | Weak: many important allocations are raw `malloc`/`calloc`/`realloc`. | High. | High-value candidate, but risky for this sprint unless tightly scoped. |
| LDLT CSC workspace/factorization | `src/sparse_ldlt.c`, `src/sparse_ldlt_csc.c`, LDLT tests | High: factor/workspace state must not be partially published. | Medium to high for workspace object allocation; broader factorization is more complex. | Medium: retry likely feasible for selected small matrices. | Weak to mixed: workspace allocation uses raw `calloc`; some sparse matrix helpers are wrapped. | High. | Good future candidate, but likely too broad unless limited to one workspace allocation helper. |
| QR factorization workspace | `src/sparse_qr.c`, QR tests | High: factor object and solver state are externally visible. | Medium: `sparse_qr_free()` exists, but factorization has many staged buffers. | Medium: retry with same matrix is possible. | Mixed: file has many wrapped helper calls and raw allocations. | High. | Valuable but broad; better after lower-risk proof pattern is extended. |
| SVD/partial-SVD workspace | `src/sparse_svd.c`, `src/sparse_svd_partial.c`, SVD tests | High: factor/result structures and dense buffers are externally visible. | Medium: cleanup functions exist, but dense workspace staging is complex. | Medium: retry is possible but numerical and workspace setup add noise. | Mixed: partial SVD has many wrapped helper calls; full SVD uses many raw allocations. | High. | Defer unless Sprint 178 intentionally chooses a complex proof. |

## Candidate Failure Sites

| Candidate | Concrete failure sites to inspect |
| --- | --- |
| Matrix shell allocation | `sparse_matrix_alloc_shell_buffers()` allocations for `row_headers`, `col_headers`, `row_perm`, `inv_row_perm`, `col_perm`, and `inv_col_perm`; `sparse_create()` cleanup after shell allocation failure. |
| Matrix multiplication workspace | `sparse_matmul()` allocation of `acc`, `nz_flag`, and `touched`; cleanup of `out` on workspace failure; no-publication behavior for `*C`. |
| Matrix copy/conversion | temporary `entries` arrays in copy/sort paths; build helper `row_tails`/`col_tails`; CSR conversion buffers if selected. |
| LU CSR conversion/factorization | `LuCsr` allocation, `row_ptr`, `col_idx`, `values`, `write_pos`, factorization work arrays, compaction arrays, and dense block temporaries. |
| LDLT CSC workspace | `LdltCscWorkspace` allocation, dense column/pattern/marker arrays, row-adjacency arrays, supernode dense/pivot buffers. |
| QR/SVD workspace | QR permutation/beta/vector buffers; Householder staging buffers; SVD bidiagonal, dense U/V workspace, partial-SVD work arrays. |

## Cleanup Observability Notes

- Matrix shell allocation has the clearest cleanup boundary: on failure,
  `sparse_create()` returns `NULL` and the partially allocated shell buffers
  are released before the struct is freed.
- Matrix multiplication has a useful public no-publication rule: if workspace
  allocation fails, the output pointer should not be left with a partial
  product and the same inputs should work after reset.
- LU CSR and LDLT/QR/SVD factorization have higher user value but broader
  cleanup graphs; they may need additional instrumentation or conversion from
  raw allocation to wrapped helpers before deterministic injection can reach
  the intended sites.
- Workspace-only proofs can be narrowly bounded, but the public claim must
  name the workspace owner rather than the entire solver family.

## Retry Feasibility Notes

| Candidate | Retry feasibility |
| --- | --- |
| Matrix shell allocation | Strong: call `sparse_create()` with injection, reset, call again, then validate dimensions/permutations. |
| Matrix multiplication workspace | Strong: call multiplication with injection, verify error/no output publication, reset, call again, validate numeric result. |
| Matrix copy/conversion | Strong if scoped to deterministic wrapped workspace allocation; verify input matrix unchanged and retry output matches. |
| LU CSR factorization | Medium: retry should be possible but may need careful factor-state cleanup checks. |
| LDLT/QR/SVD workspace | Medium: retry is possible for small fixtures, but cleanup ownership and numerical setup are noisier. |

## Day 2 Recommendation For Day 3

The strongest low-risk Sprint 178 candidates are:

1. Matrix shell allocation.
2. Matrix multiplication workspace.
3. Matrix copy/conversion workspace.

These candidates have user-visible behavior, deterministic wrapped allocation
sites, observable cleanup or no-publication expectations, and simple retry
fixtures. Direct solver and decomposition candidates remain valuable, but
many key paths use raw allocation and have a larger cleanup graph, so they
should be selected only if Day 3 intentionally accepts the higher scope.

## Non-Selection Statement

Day 2 does not select the Sprint 178 subsystem. Selection is deferred to Day
3 after comparing the candidate matrix against Sprint 177 Gate 1.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| At least three candidate subsystems are comparable | Complete | Candidate matrix compares matrix shell, matrix multiply, matrix copy/conversion, LU CSR, LDLT CSC, QR, and SVD/partial-SVD surfaces. |
| Failure and cleanup paths are mapped to concrete files | Complete | Candidate failure-site table maps each family to owner files and allocation sites. |
| No subsystem is selected before feasibility is assessed | Complete | Day 2 records recommendations only; final selection is deferred to Day 3. |
