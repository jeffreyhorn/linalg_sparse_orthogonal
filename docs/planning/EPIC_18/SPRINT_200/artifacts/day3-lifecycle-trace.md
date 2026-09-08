# Sprint 200 Day 3: Lifecycle Trace

## Purpose

Trace the selected `sparse_symbolic_lu()` owner before implementation so Day 4
can turn concrete allocation, publication, cleanup, retry, and input
preservation behavior into invariants.

## Selected Boundary

The selected owner is exactly `sparse_symbolic_lu()` in
`src/sparse_etree.c:412-665`. Sprint 200 does not expand this proof to
`sparse_analyze()`, all symbolic helpers, direct solvers, matrix construction,
platform OOM behavior, or concurrent allocation-hook behavior.

## Lifecycle Trace

| Step | Code reference | Owner action | Cleanup or publication rule |
| --- | --- | --- | --- |
| 1 | `src/sparse_etree.c:414-419` | Validate `A`, output pointers, and square shape. | No allocation has occurred. |
| 2 | `src/sparse_etree.c:427-429` | Allocate temporary interaction matrix `B`. | Later failures must call `sparse_free(B)`. |
| 3 | `src/sparse_etree.c:433-465` | If `perm` is present, allocate and fill `seen` and `inv_perm`. | Invalid permutation or allocation failure frees permutation workspaces and `B`. |
| 4 | `src/sparse_etree.c:467-495` | Allocate `row_cols`, build row interaction candidates, then free `row_cols` and `inv_perm`. | `row_cols` failure returns allocation error without output publication. |
| 5 | `src/sparse_etree.c:497-510` | Propagate insertion or diagonal-fill failures from temporary `B`. | Frees `B`; matrix-construction breadth remains outside the selected proof unless isolated later. |
| 6 | `src/sparse_etree.c:514-525` | Allocate `parent`, `postorder`, and `cc`. | Frees partial workspaces and `B` on allocation failure. |
| 7 | `src/sparse_etree.c:527-543` | Compute etree, postorder, colcount, and local `sym_full`. | Common cleanup frees workspaces and `B`; `sym_full` is owned locally until final publication. |
| 8 | `src/sparse_etree.c:551-561` | If U is requested, zero caller `sym_U`, set `n`, and allocate `u_cnt`. | `u_cnt` failure frees `sym_full`; `sym_U` needs an explicit failure invariant. |
| 9 | `src/sparse_etree.c:572-626` | Allocate and fill `sym_U->col_ptr`, compute `nnz`, allocate `sym_U->row_idx`. | Failures must free `u_cnt`, `sym_U`, and `sym_full`. |
| 10 | `src/sparse_etree.c:628-649` | Fill and sort U row indices, then free `u_cnt`. | No current failure after row fill. |
| 11 | `src/sparse_etree.c:652-657` | Publish `sym_L` by assigning `sym_full`, or free `sym_full` for U-only success. | `sym_L` is delayed until all requested work succeeds. |
| 12 | `src/sparse_etree.c:659-664` | Free common workspaces and temporary `B`. | Return final status. |

## Allocation Reachability

| Allocation site | Current deterministic hook reachability | Day 4/5 implication |
| --- | --- | --- |
| `sparse_create(n, n)` for `B` | Not directly proven from the selected owner today. | Avoid claiming unless existing matrix internals make this reachable in focused tests. |
| Direct `calloc` for `seen` | Not hook-controlled. | Wrapper conversion may be needed for valid-permutation allocation proof. |
| Direct `malloc` for `inv_perm` | Not hook-controlled. | Wrapper conversion may be needed for valid-permutation allocation proof. |
| `sparse_malloc_idx_array` for `row_cols` | Hook-controlled. | Good early failure point. |
| `sparse_insert(B, ...)` internals | Mixed, partly matrix owner. | Treat propagated failures carefully to avoid broad matrix claims. |
| `sparse_malloc_idx_array` for `parent`, `postorder`, `cc` | Hook-controlled. | Good middle failure points. |
| `sparse_symbolic_cholesky()` local `sym_full` allocations | Hook-controlled inside the existing symbolic Cholesky proof path. | Can be claimed as propagated symbolic-LU intermediate failure, not as a new symbolic Cholesky proof. |
| `sparse_calloc_idx_array` for `u_cnt` | Hook-controlled. | Good U-building failure point. |
| Direct `malloc` for `sym_U->col_ptr` | Not hook-controlled. | Important stale-output point; likely needs narrow wrapper conversion. |
| `sparse_malloc_idx_array` for `sym_U->row_idx` | Hook-controlled. | Good late U-building failure point after partial U publication. |

## Caller-Owned Inputs

| Input | Preservation expectation |
| --- | --- |
| `A` | `sparse_symbolic_lu()` reads matrix structure and values but should not mutate `A`; tests should sample dimensions and representative entries after failure. |
| `perm` | Valid permutation input is read-only; tests should compare all elements after injected failure. |
| `sym_L` | Caller owns struct storage; Day 4 must decide whether failure leaves sentinel fields unchanged or requires zero/free-safe state. |
| `sym_U` | Caller owns struct storage; because the function zeroes and partially fills it, Day 4 must require free-safe failed state after U-building failures. |

## Retry Points

| Retry path | Evidence to add later |
| --- | --- |
| L+U natural order | Failure followed by successful `sparse_symbolic_lu(A, NULL, &L, &U)` on the same matrix. |
| L-only | Failure followed by successful L-only call, proving no stale `sym_L` dependency. |
| U-only | Failure followed by successful U-only call, proving partial U cleanup is sufficient. |
| Valid permutation | Failure followed by successful call with the same unchanged `perm`. |

## Unsupported Breadth

This trace deliberately does not claim:

- broad `sparse_analyze()` allocation-failure ownership;
- helper-level etree/postorder/colcount proof;
- broad sparse matrix allocation or insertion reliability;
- direct-solver factor/solve reliability;
- OS OOM behavior;
- concurrent use of process-global allocation hooks;
- hosted CI ownership or platform parity.

## Day 4 Handoff

Day 4 should formalize invariants for:

1. `sym_L` delayed publication and failed-call state;
2. `sym_U` free-safe partial publication after zeroing;
3. caller-owned `A` and `perm` preservation;
4. retry success after `sparse_alloc_test_reset()`;
5. wrapper conversion decisions for direct `calloc`/`malloc` paths.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 3 validation command.
