# Sprint 17 — CSC Numeric Backend Performance Notes

These numbers are produced by the Day 12 benchmarks
`benchmarks/bench_chol_csc.c` and `benchmarks/bench_ldlt_csc.c`.
They drive the Day 13 tuning of `SPARSE_CSC_THRESHOLD` and shape the
Day 14 README performance section.

## Reproducing

```bash
make build/bench_chol_csc build/bench_ldlt_csc
./build/bench_chol_csc --repeat 5
./build/bench_ldlt_csc --repeat 5
```

Optional: `./build/bench_chol_csc path/to/matrix.mtx --repeat 5` for a
single matrix.

## Machine

Darwin 24.6.0 on Apple silicon; Apple Clang (from Xcode Command Line
Tools); release (-O2) library.  Absolute numbers will vary between
machines; speedup ratios are the interesting signal.

## Fair-comparison methodology

Both benchmarks now run AMD fill-reducing reordering (or its
`sparse_analyze` equivalent) *inside* the per-repetition timed region
on both paths.  The linked-list entry point
`sparse_cholesky_factor_opts(AMD)` / `sparse_ldlt_factor_opts(AMD)`
re-runs AMD on every call, so including AMD on the CSC side gives an
apples-to-apples one-shot factor comparison.  Earlier drafts of these
numbers precomputed AMD once outside the loop on the CSC side, which
inflated the speedup by the AMD cost (dominant on small matrices).
See the "analyze-once, factor-many" section below for the workflow
where AMD amortization is legitimate.

## Cholesky — CSC vs linked-list

All runs use AMD fill-reducing reordering for all paths.  Residual
`||A x - b||_∞ / ||b||_∞` with b = A · 1.  5-repeat average.

| matrix         |   n   |  nnz  | factor_ll | factor_csc | factor_csc_sn | solve_ll | solve_csc | speedup_csc | speedup_csc_sn | res |
|----------------|------:|------:|----------:|-----------:|--------------:|---------:|----------:|------------:|---------------:|----:|
| `nos4.mtx`     |  100  |   594 |   2.02 ms |    1.22 ms |       1.00 ms | 0.015 ms |  0.009 ms |      **1.65×** |       **2.01×** | 6e-16 |
| `bcsstk04.mtx` |  132  |  3648 |   8.03 ms |    7.12 ms |       6.61 ms | 0.108 ms |  0.034 ms |      **1.13×** |       **1.22×** | 1e-15 |

Times are in milliseconds (ms).  `factor_ll` = linked-list factor,
`factor_csc` = Sprint 17 CSC scalar factor, `factor_csc_sn` = Sprint 17
CSC supernodal entry point (Day 11).  Residuals match to double-
precision round-off on both paths.

**Headline:** the CSC factor is **1.1–2.0× faster** than the linked-
list kernel on the two SPD SuiteSparse matrices in the default corpus,
once AMD is measured on both sides.  Day 5's scatter-gather column
kernel replaces linked-list pointer-chasing with contiguous column
traversals; the gain is most visible on small matrices where the
linked-list per-edge overhead is a larger fraction of total work
(nos4, 2.0× on the supernodal path).

The supernodal entry point (Day 11) currently delegates to the same
scalar kernel, so its timing tracks CSC scalar to within variance
plus a modest win from dense-block reuse.  Adding a native batched
dense-kernel path is a follow-up.

Solve times show a larger relative gap (CSC's triangular solves are
~2–3× faster), but solve cost is tiny compared to factor so this is
not the headline number.

## LDL^T — CSC vs linked-list

Day 8's CSC LDL^T is a wrapper: it expands the lower triangle to a
full symmetric `SparseMatrix` and calls the linked-list
`sparse_ldlt_factor`.  The benchmark measures that overhead.

| matrix         |   n   |  nnz  | factor_ll | factor_csc | speedup_csc | res |
|----------------|------:|------:|----------:|-----------:|------------:|----:|
| `nos4.mtx`     |  100  |   594 |   0.64 ms |    2.41 ms |       0.27× | 6e-16 |
| `bcsstk04.mtx` |  132  |  3648 |  12.76 ms |    9.63 ms |       1.33× | 1e-15 |

**Headline:** no systematic speedup on the current LDL^T wrapper —
on small matrices the CSC↔linked-list expansion dominates and the
wrapper is slower (nos4, 0.27×); on larger matrices it breaks even
or modestly wins (bcsstk04, 1.33×).  A native CSC LDL^T kernel (with
1×1 / 2×2 Bunch-Kaufman pivoting in packed storage and proper
symmetric swaps) is the next LDL^T deliverable; these numbers become
the baseline.

## Analyze-once, factor-many workflow

The numbers above measure a **one-shot** factor call.  The
`sparse_analyze` / `sparse_factor_numeric` split (Sprint 14) lets you
run AMD + symbolic analysis once and reuse the result across many
numeric refactorizations with the same pattern (e.g. a time-stepping
simulation where the matrix values change but the nonzero pattern is
fixed).  In that workflow AMD is amortized to zero, and the CSC
numeric kernel advantage shows up undiluted.  The `speedup_csc`
column in this document measures the *one-shot* case deliberately —
it is the comparison most relevant when only a single solve is
needed.  Callers running many refactorizations of the same pattern
should expect a larger CSC advantage because the linked-list path's
per-call AMD is the more significant cost gap.

## Threshold guidance (`SPARSE_CSC_THRESHOLD`)

`SPARSE_CSC_THRESHOLD` defaults to 100 in `include/sparse_matrix.h`.
The 5-repeat Cholesky numbers above show a one-shot speedup at
n = 100 (`nos4`, 1.65×–2.01×) — comfortably above 1× — so the 100
default is a reasonable first-cut crossover.  Matrices smaller than
~30-50 columns may spend more time on CSC conversion than on
elimination and are best left on the linked-list path.  Override
with `-DSPARSE_CSC_THRESHOLD=N` at compile time.

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
   the scalar CSC's 1.1–2.0× (expect 2–4× more on top once a native
   batched dense kernel replaces the scalar delegate).
3. A native CSC LDL^T kernel so the LDL^T ratio moves from ~0.3–1.3×
   to something competitive with Cholesky's scalar numbers.
4. Benchmark the `sparse_analyze` / `sparse_factor_numeric` split on a
   time-stepping workload to quantify the analyze-once, factor-many
   speedup (expected to be larger than the one-shot numbers here).
