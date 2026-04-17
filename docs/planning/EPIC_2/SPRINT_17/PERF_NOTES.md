# Sprint 17 — CSC Numeric Backend Performance Notes

These numbers are produced by the Day 12 benchmarks
`benchmarks/bench_chol_csc.c` and `benchmarks/bench_ldlt_csc.c`.
They drive the Day 13 tuning of `SPARSE_CSC_THRESHOLD` and shape the
Day 14 README performance section.

## Reproducing

```bash
make build/bench_chol_csc build/bench_ldlt_csc
./build/bench_chol_csc --repeat 10
./build/bench_ldlt_csc --repeat 10
```

Optional: `./build/bench_chol_csc path/to/matrix.mtx --repeat 10` for a
single matrix.

## Machine

Darwin 24.6.0 on Apple silicon; Apple Clang (from Xcode Command Line
Tools); release (-O2) library.  Absolute numbers will vary between
machines; speedup ratios are the interesting signal.

## Cholesky — CSC vs linked-list

All runs use AMD fill-reducing reordering for both paths.  Residual
`||A x - b||_∞ / ||b||_∞` with b = A · 1.  10-repeat average.

| matrix         |   n   |  nnz  | factor_ll | factor_csc | solve_ll | solve_csc | speedup_csc | res |
|----------------|------:|------:|----------:|-----------:|---------:|----------:|------------:|----:|
| `nos4.mtx`     |  100  |   594 |   0.36 ms |    0.16 ms |  0.011 ms|  0.006 ms |      **2.3×** | 6e-16 |
| `bcsstk04.mtx` |  132  |  3648 |   3.05 ms |    0.81 ms |  0.063 ms|  0.021 ms |      **3.8×** | 1e-15 |

Times are in milliseconds (ms).  `factor_ll` = linked-list factor,
`factor_csc` = Sprint 17 CSC factor.  Residuals match to double-
precision round-off on both paths.

**Headline:** the CSC factor is **2.3–3.8× faster** than the linked-
list kernel on the two SPD SuiteSparse matrices in the default corpus.
Day 5's scatter-gather column kernel replaces linked-list pointer-
chasing with contiguous column traversals; the bigger the supernodal
structure, the larger the win (bcsstk04, which has denser fill, shows
the 3.8× end of the range).

The supernodal entry point (Day 11) currently delegates to the same
scalar kernel, so its timing tracks CSC scalar to within variance.
Adding a native batched dense-kernel path is a follow-up.

## LDL^T — CSC vs linked-list

Day 8's CSC LDL^T is a wrapper: it expands the lower triangle to a
full symmetric `SparseMatrix` and calls the linked-list
`sparse_ldlt_factor`.  The benchmark mostly measures the expansion
overhead versus the linked-list path's internal AMD reorder.

| matrix         |   n   |  nnz  | factor_ll | factor_csc | speedup_csc | res |
|----------------|------:|------:|----------:|-----------:|------------:|----:|
| `nos4.mtx`     |  100  |   594 |   0.38 ms |    0.43 ms |       0.88× | 6e-16 |
| `bcsstk04.mtx` |  132  |  3648 |   4.62 ms |    3.27 ms |       1.41× | 1e-15 |

**Headline:** no systematic speedup on the current LDL^T wrapper —
this is expected.  A native CSC LDL^T kernel (with 1×1 / 2×2 Bunch-
Kaufman pivoting in packed storage and proper symmetric swaps) is the
next LDL^T deliverable; these numbers become the baseline.

## Threshold guidance (`SPARSE_CSC_THRESHOLD`)

`SPARSE_CSC_THRESHOLD` defaults to 100 in `include/sparse_matrix.h`.
The 10-repeat Cholesky numbers above show a speedup at n = 100
(`nos4`, 2.3×) — comfortably above 1× — so the 100 default is a
reasonable first-cut crossover.  Matrices smaller than ~30-50 columns
may spend more time on CSC conversion than on elimination and are
best left on the linked-list path.  Override with
`-DSPARSE_CSC_THRESHOLD=N` at compile time.

## Current opt-in path

Today the CSC Cholesky kernel is reached through the internal
`chol_csc_factor` / `chol_csc_factor_solve` APIs (see
`src/sparse_chol_csc_internal.h`).  The public
`sparse_cholesky_factor_opts` continues to run the linked-list
kernel unchanged — a transparent size-based dispatch through that
entry point is tracked as Day 13+/post-sprint work (requires a
CSC → linked-list writeback so factored state can live in the
caller-owned `SparseMatrix`).

## Follow-up ideas driven by these numbers

1. Add `make bench` entry points dedicated to the CSC comparison so the
   numbers land in CI.
2. Source a larger SuiteSparse corpus (n ≥ 1000) where fill is heavy
   and the supernodal dense-kernel path can show a multiplier beyond
   the scalar CSC's 3.8× (expect 2–4× more on top).
3. A native CSC LDL^T kernel so the LDL^T ratio moves from ~1× to
   something competitive with Cholesky's 3–4×.
