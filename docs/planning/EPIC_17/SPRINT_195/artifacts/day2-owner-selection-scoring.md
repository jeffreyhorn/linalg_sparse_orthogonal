# Sprint 195 Day 2: Owner Selection Scoring

## Purpose

Select exactly one Sprint 195 reliability owner using allocation density,
cleanup complexity, stale-output risk, retry clarity, user impact,
deterministic hook feasibility, and review cost.

## Decision

Sprint 195 selects:

`sparse_symbolic_cholesky()` in `src/sparse_etree.c`

The selected proof lane is the symbolic Cholesky out-struct construction path.
It is not the whole etree subsystem, not `sparse_analyze()`, not
`sparse_symbolic_lu()`, and not broad direct-solver allocation-failure proof.

## Selection Rationale

| Criterion | Evidence |
| --- | --- |
| Allocation density | The owner allocates `sym->col_ptr`, `sym->row_idx`, child-list work arrays, marker/temp arrays, column-row pointer arrays, column-row counts, and per-column propagated row sets. |
| Cleanup complexity | Failure paths call `sparse_symbolic_free(sym)` and free temporary arrays after partial publication. This gives meaningful cleanup behavior to prove. |
| Stale-output risk | The caller passes `sparse_symbolic_t *sym`; a failure must not leave success-looking `col_ptr`, `row_idx`, `n`, or `nnz` state behind. |
| Retry clarity | Existing `test_etree` fixtures already prove successful symbolic output for 1x1, diagonal, tridiagonal, arrow, dense, known 5x5, and SuiteSparse cases. |
| User impact | Symbolic Cholesky supports analysis and direct-solver setup for maintained workflows. It is visible through symbolic and analysis APIs without requiring public API changes. |
| Hook feasibility | Most allocations already use `sparse_malloc_idx_array`, `sparse_calloc_idx_array`, or `sparse_malloc_array`; one `sym->col_ptr` direct `malloc` is a bounded Day 3/Day 4 harness-design question. |
| Review cost | The proof can live in the existing `test_etree` proof-owner binary with a focused gate, avoiding a new solver family or broad source-list rewrite. |

## Ranked Candidate Table

| Rank | Candidate | Total score | Disposition |
| ---: | --- | ---: | --- |
| 1 | `sparse_symbolic_cholesky()` symbolic out-struct owner | 31 | Selected primary owner. |
| 2 | `chol_csc_alloc()` / `chol_csc_workspace_alloc()` | 28 | Fallback owner. |
| 3 | `sparse_symbolic_lu()` | 27 | Deferred because it composes too many owners. |
| 4 | Narrow `sparse_matrix` constructor path | 26 | Deferred because claim boundaries are hard to keep narrow. |
| 5 | Broad LDLT CSC owner | 25 | Rejected for Sprint 195 due to algorithm and backend complexity. |
| 6 | LU CSR solve/factor lane | 24 | Deferred; valuable but direct allocator and output mutation semantics increase proof cost. |
| 7 | Linked-list LU solve workspace lane | 23 | Deferred; caller-owned `x` mutation makes stale-output proof less crisp. |
| 8 | Broad QR factor/solve owner | 23 | Rejected due to recent QR churn and numerical behavior risk. |

## Fallback Owner

The fallback owner is Cholesky CSC object/workspace construction:
`chol_csc_alloc()` and `chol_csc_workspace_alloc()` in `src/sparse_chol_csc.c`.

It is a good fallback because out-pointer semantics are clear and existing
tests already cover null, badarg, zero-size, grow, and workspace basics. It is
not the primary choice because its allocations currently use direct `calloc`,
so deterministic fail-at-count proof would need more harness conversion before
tests can force individual failure sites.

## Existing Coverage For Selected Owner

| Existing test surface | Current proof | Sprint 195 gap |
| --- | --- | --- |
| `test_symbolic_null_args` | Invalid inputs return `SPARSE_ERR_NULL`. | Does not exercise allocation failure. |
| `test_symbolic_1x1` | Minimal successful symbolic output is correct. | Does not prove retry after a failed attempt. |
| `test_symbolic_diagonal`, `test_symbolic_tridiag`, `test_symbolic_arrow`, `test_symbolic_dense`, `test_symbolic_known_5x5` | Normal symbolic structures match expected shape or numeric Cholesky containment. | Do not assert cleanup after partial allocation failure. |
| `test_symbolic_free_zeroed` | `sparse_symbolic_free(NULL)` and freeing zeroed structs are safe. | Does not prove partially initialized structs are cleaned after failure. |
| Make/CMake registration | `test_etree` is registered in both Make and CMake. | No focused symbolic allocation-failure gate or label exists yet. |

## Rejected Breadth

Sprint 195 explicitly excludes:

- all `src/sparse_etree.c` allocation-failure behavior;
- `sparse_symbolic_lu()` L/U publication and cleanup;
- `sparse_analyze()` output-state cleanup;
- `sparse_etree_compute()`, `sparse_etree_postorder()`, and `sparse_colcount()`
  standalone proofs except where needed to prepare a successful selected-owner
  fixture;
- all direct solvers, eigensolvers, SVD, QR, matrix construction, conversion,
  IO, package/install, report-generation, and benchmark allocation paths;
- exhaustive allocator behavior, concurrency behavior, and state-of-the-art
  reliability claims.

## Day 3 Handoff

Day 3 should trace `sparse_symbolic_cholesky()` line by line and record:

1. every allocation point and whether the current hook can force it;
2. exactly when `sym` is zeroed, partially populated, freed, and returned;
3. what state callers may observe after each failure class;
4. retry setup using an existing small fixture;
5. whether converting the `sym->col_ptr` direct `malloc` to
   `sparse_malloc_array` is required for complete deterministic proof;
6. the focused gate shape for `test_etree` symbolic allocation-failure tests.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

`git diff --check` passes.
