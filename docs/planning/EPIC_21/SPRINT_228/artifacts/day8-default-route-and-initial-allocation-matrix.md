# Sprint 228 Day 8: Default Route and Initial Allocation Matrix

## Outcome

Day 8 completes repeated selector verification and makes every allocation in
the maintained reconstruction boundary injectable. The focused sweep reaches
each output-shell/map/slab allocation, observes `SPARSE_ERR_ALLOC` with NULL
output at every failure index, crosses the first successful completion index,
and succeeds again after reset.

This is deliberately an **initial reconstruction-boundary matrix**, not an
exhaustive whole-call allocation claim. The bidiagonalization and SVD-factor
allocations catalogued below still use raw allocation in several locations;
Day 9 owns their conversion and the full public-call sweep.

## Default-route proof

The checked selector has only three outcomes:

- unset or exact `on` sets `SVD_LOWRANK_ROUTE_OUTER`;
- exact `off` sets `SVD_LOWRANK_ROUTE_DENSE`;
- every other set value returns `SPARSE_ERR_BADARG` before decomposition.

After SVD and tolerance resolution, `src/sparse_svd.c` dispatches the outer
route and returns immediately on success or failure. The checked dense
`calloc(m*n, sizeof(double))` appears only after that return boundary in the
exact-`off` branch. Consequently an ordinary unset call cannot reach the old
reconstruction accumulator. This is a control-flow proof; the dense
bidiagonalization workspace is separate and remains intentionally present.

The repeated-selector regression runs unset, `on`, `off`, `on`, `off`, `on`,
then two malformed values in one process. Every valid result matches the
analytical default baseline exactly, both malformed calls clear stale output,
and the original caller environment is restored.

## Initial injectable allocation matrix

The internal-only `sparse_create_injectable()` and
`sparse_insert_injectable()` surfaces reuse the production matrix constructor
and insertion implementation while routing the matrix object and first node
slab through the existing checked allocation hook. Public matrix API behavior
and ABI are unchanged.

The pinned 3x3, rank-two direct-factor fixture produces two nodes, which fit in
one slab. Its exact allocation sequence is:

| Failure index | Allocation | Expected result |
| ---: | --- | --- |
| 0 | `SparseMatrix` shell | `SPARSE_ERR_ALLOC`, NULL output |
| 1 | row-header array | `SPARSE_ERR_ALLOC`, NULL output |
| 2 | column-header array | `SPARSE_ERR_ALLOC`, NULL output |
| 3 | row permutation | `SPARSE_ERR_ALLOC`, NULL output |
| 4 | inverse-row permutation | `SPARSE_ERR_ALLOC`, NULL output |
| 5 | column permutation | `SPARSE_ERR_ALLOC`, NULL output |
| 6 | inverse-column permutation | `SPARSE_ERR_ALLOC`, NULL output |
| 7 | first output-node slab | `SPARSE_ERR_ALLOC`, NULL output; partial matrix destroyed |
| 8 | no failure reached | success, two sorted diagonal nodes |

Every failure preserves sigma, U, and Vt byte-for-byte. A clean retry after
the entire sweep produces the same successful output. The helper owns and
destroys every partial matrix until it publishes complete success.

## Whole-call allocation inventory

### Economy SVD decomposition and factors

For a non-wide `m x n` input with `k=min(m,n)`, the public call reaches:

1. bidiagonal dense workspace `W[m*n]`;
2. bidiagonal `diag[k]`, `superdiag[k-1]`, `u_vecs[k]`, `u_betas[k]`,
   `v_vecs[k-1]`, and `v_betas[k-1]` arrays;
3. Householder scratch `hv[max(m,n)]`;
4. one left reflector of lengths `m, m-1, ..., m-k+1` per step;
5. one right reflector of lengths `n-1, n-2, ..., n-k+1` per applicable step;
6. QR copies `bd_diag[k]` and `bd_super[k-1]`;
7. economy work factors `U_work[m*k]` and `V_work[n*k]`;
8. published `Vt[k*n]`; `U_work` becomes the published U factor.

Several of these still use raw `malloc`/`calloc`; they are the first Day 9
instrumentation target. Full-mode padding allocations are not reached because
low-rank reconstruction always requests economy mode.

### Wide-input transpose prelude

When `m<n`, bidiagonalization first creates `A^T`: an `nnz` entry array,
transpose matrix shell/maps, row/column tail arrays, and enough node slabs for
the transposed entries. It then runs the non-wide list above on `A^T`. The
entry and tail arrays are checked; transpose shell and node allocations are
not yet fully injectable.

### Reconstruction and output

- Maintained outer route: injectable matrix shell, six map/header arrays, and
  one node slab per `SPARSE_NODES_PER_SLAB` retained output entries. It has no
  route-specific `m*n` accumulator.
- Exact-`off` compatibility route: public matrix shell/maps, raw dense
  `accum[m*n]`, and output node slabs. This remains intentionally reachable
  only through exact `off`.
- All-zero shortcut: public empty matrix shell/maps and no nodes. It bypasses
  both reconstruction loops and needs inclusion in Day 9's public-call sweep.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | Passed: 131 tests, 17,568 assertions, no failures or skips |

The isolated 64-bit focused run and final full quality gates are recorded in
the working notes after completion.

## Day 9 handoff

Convert the catalogued raw allocations on the economy-SVD and wide-transpose
paths to the checked hook, include the all-zero and exact-`off` branches, then
run exhaustive public-call allocation and numeric failure sweeps through
success. Preserve the Day 8 direct reconstruction sweep as the smaller owner
boundary diagnostic.
