# Sprint 228 Day 1: Reconstruction Path and Allocation Inventory

**Date:** 2026-11-25

**Source:** `363600ea27e854a4334cff5fa06ace6529045ccd`

**Branch:** `sprint-228`

**Decision state:** Current-state baseline only; Day 2 owns the accepted contract and measurement protocol.

## Scope and conclusion

`sparse_svd_lowrank_sparse()` always computes a full economy SVD before reconstructing a sparse approximation. The current default then allocates an additional zeroed `m×n` double buffer, accumulates all retained rank-one terms into it, and inserts threshold-qualified cells into a sparse matrix. Exact environment value `SPARSE_SVD_LOWRANK_OUTER=on` instead computes each cell in a scalar accumulator and inserts it directly. The latter removes only the additional reconstruction buffer; it does not remove dense bidiagonalization work or the economy `U` and `Vt` factors.

No production behavior was changed on Day 1.

## Call graph

```text
sparse_svd_lowrank_sparse(A, rank_k, drop_tol, &result)
  |
  +-- validate output pointer; set *result = NULL
  +-- validate A, rank, and original row/column state
  +-- sparse_svd_compute(compute_uv=1, economy=1)
  |    |
  |    +-- sparse_bidiag_factor(A)
  |    |    +-- wide A: sparse_transpose(A), then recurse on A^T
  |    |    +-- dense W[m*n] and Householder reflector storage
  |    +-- copy bidiagonal arrays
  |    +-- form U_work[m*k] and V_work[n*k]
  |    +-- implicit bidiagonal QR iteration and triplet sorting
  |    +-- retain sigma[k], U[m*k], Vt[k*n]
  |
  +-- sigma[0] == 0: free SVD, return empty sparse_create(m,n)
  +-- drop_tol <= 0: use DBL_EPSILON * sigma[0]
  +-- parse_svd_lowrank_outer()
       |
       +-- exact "on": sparse_svd_lowrank_outer_product()
       |    +-- sparse_create(m,n)
       |    +-- for i, j: scalar sum over s; threshold; sparse_insert
       |
       +-- otherwise: dense-intermediate route
            +-- sparse_create(m,n)
            +-- checked calloc(m*n, sizeof(double))
            +-- for s, i, j: accumulate rank-one terms
            +-- for i, j: threshold; sparse_insert
```

Both nonzero routes call `sparse_svd_free()` after reconstruction and transfer only the sparse output to the caller on success.

## Ownership and allocation inventory

Let `k = min(m,n)`, `q` be output `nnz`, and `S = SPARSE_NODES_PER_SLAB` (currently 4096).

| Layer | Storage and size | Owner while live | Cleanup/transfer | Overflow protection |
| --- | --- | --- | --- | --- |
| Wide-input transpose | Entry array, transposed sparse shell/maps, row/column-tail workspaces, and `ceil(nnz(A)/S)` slabs | `sparse_bidiag_factor` wide branch | `sparse_free(At)` after recursive factorization | Checked `idx_t` conversions in transpose/build helpers; shell buffers use checked array allocators |
| Bidiagonal dense work | `W[m*n]` doubles (or transposed dimensions with the same product) | `sparse_bidiag_factor` | Freed before return | Manual product and byte-limit checks |
| Bidiagonal metadata | `diag[k]`, `superdiag[k-1]`, pointer tables, beta arrays, one `max(m,n)` work vector | `sparse_bidiag_t` during factorization/SVD extraction | `sparse_bidiag_free()` | Dimension-to-byte checks precede allocation |
| Householder vectors | Left lengths `m-s`; right lengths `n-s-1` for applicable steps | `sparse_bidiag_t` | `sparse_bidiag_free()` | Prior maximum-dimension byte checks; individual raw allocations |
| Economy SVD factors | `sigma[k]`, `U[m*k]`, `Vt[k*n]` doubles | Local `sparse_svd_t` | `sparse_svd_free()` after output construction | Checked products and `SIZE_MAX / sizeof(double)` guards |
| Temporary SVD vectors | `V_work[n*k]` while forming `Vt`; copied bidiagonal arrays | `sparse_svd_compute` | Freed before successful return | Checked products/bytes |
| Sparse output shell | One matrix object; row/column header arrays; two forward/inverse permutation arrays per dimension | Output `SparseMatrix` | Caller receives on success; `sparse_free()` on error | Six shell arrays use checked allocation wrappers; object allocation is raw |
| Sparse output entries | `ceil(q/S)` full node slabs | Output `SparseMatrix` | Caller receives on success; `sparse_free()` on error | No count product at insert time; slab allocation occurs on demand |
| Dense-route scratch | `m*n` zeroed doubles | `sparse_svd_lowrank_sparse` | Freed after threshold pass or before error return | `sparse_size_mul_overflow(m,n)` plus byte-limit guard |
| Outer-route scratch | One scalar `val` per active loop iteration | Stack | Ends per cell | No heap allocation; factor indexing uses previously checked dimensions |
| Selector | Borrowed pointer returned by `getenv` | Process environment | No owned storage | Exact string comparison only |

### Important memory boundary

The removable allocation is exactly the dense-route `accum[m*n]`. It is additional to:

- bidiagonalization's dense `W[m*n]` temporary;
- retained economy factors `U[m*k]` and `Vt[k*n]` plus `sigma[k]` during reconstruction; and
- the sparse output shell, maps, and node slabs.

Therefore a successful Sprint 228 change can claim removal of the default reconstruction accumulator, not sparse-memory full SVD.

## Representative allocation-call count

These are source-derived successful heap-allocation calls, not peak-live allocations or allocator-injection observations. They exclude construction of input `A` and assume a square/tall input (`m >= n`), `k > 1`, economy vectors enabled, nonzero output, and one output slab.

For economy SVD:

- `sparse_bidiag_factor`: eight fixed allocations plus `k` left and `k-1` right reflector vectors = `2k + 7` calls;
- post-factor SVD: copied diagonal/superdiagonal, `U_work`, `V_work`, and `Vt` = five calls;
- total decomposition/factors = `2k + 12` calls;
- sparse output shell/maps = seven calls;
- first output slab = one call;
- dense route only: accumulator = one call.

The maintained 32×32 rank-4 synthetic cross-route fixture has `k=32` and four retained diagonal entries, hence one node slab:

| Component | Dense default | Forced outer |
| --- | ---: | ---: |
| Bidiagonalization and economy SVD | 76 | 76 |
| Sparse output shell/maps | 7 | 7 |
| Output node slabs | 1 | 1 |
| Route-specific reconstruction buffer | 1 (`32*32*8 = 8,192` bytes) | 0 |
| **Successful allocation calls** | **85** | **84** |

This one-call difference understates the memory difference because the dense call scales as `8*m*n` bytes, while an output slab is allocated identically by both routes. For wide inputs, transpose construction adds its own entry array, sparse shell/maps, two tail workspaces, and node slabs; those are decomposition costs, not reconstruction-route costs.

## Allocation-injection coverage gap

`sparse_alloc_test_fail_after()` controls only allocations made through `sparse_malloc_*` and `sparse_calloc_*`. Current low-rank reconstruction also uses raw allocation for:

- bidiagonal dense work, metadata, reflector vectors, and SVD work/factor arrays;
- the dense `accum` buffer;
- the public `sparse_create()` matrix object; and
- node slabs reached through public `sparse_insert()`.

Consequently, existing fail-after tests cannot exhaustively reach the current route's failure surface. This is an explicit input to Days 8–9, not evidence that those failures are safe.

## Observed behavior baseline

| Condition | Current result | Output state |
| --- | --- | --- |
| `result == NULL` | `SPARSE_ERR_NULL` | No writable output exists |
| Valid `result`, `A == NULL` | `SPARSE_ERR_NULL` | Cleared to NULL before validation |
| `rank_k <= 0` or `rank_k > min(m,n)` | `SPARSE_ERR_BADARG` | NULL |
| Nonidentity row/column state | `SPARSE_ERR_BADARG` | NULL |
| SVD allocation or convergence failure | Propagated `SPARSE_ERR_ALLOC` or `SPARSE_ERR_NOT_CONVERGED` | NULL |
| Output shell, accumulator, or insert allocation failure | `SPARSE_ERR_ALLOC` | NULL; partial output freed |
| `sigma[0] == 0` | `SPARSE_OK` with empty `m×n` sparse matrix | Caller-owned matrix; no route scratch |
| `drop_tol <= 0` | Replaced by `DBL_EPSILON * sigma[0]` | Normal success/failure semantics |
| Positive finite `drop_tol` | Retain a cell exactly when `fabs(value) >= drop_tol` | Equality is retained; no duplicate cell insertion |
| NaN or positive-infinite `drop_tol` | Currently reaches reconstruction; all finite comparisons are false, producing empty success | Caller-owned empty matrix; Day 2 must decide whether to reject |
| Selector exactly `on` | Outer route | Normal success/failure semantics |
| Selector unset, `off`, empty, or unrecognized | Dense route | Normal success/failure semantics |

Both routes iterate output coordinates in ascending physical row then column order, call `sparse_insert` at most once per cell, and return identity permutation maps. Thus successful output has unique coordinates, sorted row chains, sorted column chains, and caller ownership through `sparse_free()`. Input `A` is read but not modified.

## Existing test and benchmark surface

- `tests/test_svd.c` covers diagonal, sparse thresholding, dense-versus-sparse comparison, rank-one, rectangular, invalid argument, synthetic route equality, and `nos4`/`bcsstk04` route equality.
- `tests/test_svd_partial_corpus.c` includes one low-rank sparse call but does not independently compare both reconstruction routes.
- `tests/test_sprint8_integration.c` exercises low-rank reconstruction and broader SVD integration.
- `benchmarks/bench_svd.c` measures full and partial SVD; it does not isolate sparse low-rank reconstruction memory.
- Historical Sprint 29 comments report selected memory behavior, but Sprint 228 does not accept those historical measurements as exact-source evidence.

## Baseline execution

Environment:

- Apple Clang 17.0.0, C11, optimized Make build, 32-bit `idx_t`;
- Darwin 24.6.0 x86_64;
- source SHA shown above.

| Command | Result |
| --- | --- |
| `make build/test_svd` | Pass; focused binary built |
| `env -u SPARSE_SVD_LOWRANK_OUTER ./build/test_svd` | 114 tests, 2,067 assertions, zero failures, zero skips; harness 0.136 s |
| `SPARSE_SVD_LOWRANK_OUTER=on ./build/test_svd` | 114 tests, 2,067 assertions, zero failures, zero skips; harness 0.142 s |

In both executions, the explicit cross-route tests reported relative Frobenius difference `0.000e+00` for the 32×32 synthetic case and for `nos4` and `bcsstk04` at ranks 10 and 50. These runs establish a reproducible baseline only. They are not isolated timing evidence, do not cover independent expected values for every route, and do not settle the Day 2 tolerance or selector contract.

## Day 2 inputs

Day 2 must resolve the following before implementation:

1. Reject or explicitly define NaN and infinite `drop_tol` values.
2. Decide whether the environment selector is retained as compatibility/diagnostic behavior and how invalid values behave.
3. Pin square, tall, wide, zero, rank-deficient, clustered, cancellation, and threshold-boundary fixtures with independent references.
4. Freeze elementwise/normwise comparison tolerances and structural invariants without using one production route as the oracle for the other.
5. Separate reconstruction allocation accounting from full-SVD factors and whole-process RSS in the measurement protocol.
