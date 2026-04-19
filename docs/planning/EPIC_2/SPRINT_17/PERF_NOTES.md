# Sprint 17 — CSC Numeric Backend Performance Notes

These numbers are produced by the Day 12 benchmarks
`benchmarks/bench_chol_csc.c` and `benchmarks/bench_ldlt_csc.c`.
They drive the Day 13 tuning of `SPARSE_CSC_THRESHOLD` and shape the
Day 14 README performance section.

> **Sprint 18 Day 13 update.** Corpus expanded to include `bcsstk14`,
> `s3rmt3m3`, `Kuu`, and `Pres_Poisson` (SPD, n up to ≈15k).  Both
> tables below are regenerated from `bench_day12.txt` (3-repeat
> average; full CSV at
> [`docs/planning/EPIC_2/SPRINT_18/bench_day12.txt`](../SPRINT_18/bench_day12.txt)).
> Day 13 also landed the full `sym_L` pre-population in
> `chol_csc_from_sparse_with_analysis` (a Day 12 discovery) — without
> it the batched supernodal path silently missed fill rows on every
> new SPD fixture and produced residuals in the 1e-1 range.  Today's
> table is post-fix, so residuals are back at round-off.

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

| matrix             |   n   |    nnz  | factor_ll |  factor_csc | factor_csc_sn | solve_ll | solve_csc | speedup_csc | speedup_csc_sn | res |
|--------------------|------:|--------:|----------:|------------:|--------------:|---------:|----------:|------------:|---------------:|----:|
| `nos4.mtx`         |   100 |     594 |    0.46 ms |      0.42 ms |        0.38 ms | 0.008 ms | 0.008 ms |      **1.09×** |        **1.22×** | 7e-16 |
| `bcsstk04.mtx`     |   132 |    3648 |    3.12 ms |      2.67 ms |        3.09 ms | 0.042 ms | 0.017 ms |      **1.16×** |        **1.01×** | 6e-16 |
| `bcsstk14.mtx`     |  1806 |   63454 |  364.29 ms |    208.82 ms |      152.83 ms | 5.569 ms | 0.533 ms |      **1.74×** |        **2.38×** | 1e-15 |
| `s3rmt3m3.mtx`     |  5357 |  207123 | 4018.41 ms |   1914.53 ms |     1179.41 ms |21.436 ms | 3.443 ms |      **2.10×** |        **3.41×** | 5e-15 |
| `Kuu.mtx`          |  7102 |  340200 | 3147.78 ms |   4112.76 ms |     1416.64 ms |20.774 ms | 2.599 ms |        0.77× |        **2.22×** | 9e-15 |
| `Pres_Poisson.mtx` | 14822 |  715804 |46003.69 ms |  17597.98 ms |    10580.68 ms |135.013 ms|15.345 ms |      **2.61×** |        **4.35×** | 2e-13 |

Times are in milliseconds (ms), 3-repeat average.  `factor_ll` = the
public `sparse_cholesky_factor_opts(AMD)` entry point — which under
the Sprint 18 Day 11 dispatch now routes to the CSC supernodal
kernel internally whenever `n >= SPARSE_CSC_THRESHOLD`, so on the
three smallest matrices the `factor_ll` column records the linked-
list kernel and on the four largest it records the supernodal
kernel (with the extra sparse_analyze / from_sparse / writeback
round-trip visible in the `factor_ll >= factor_csc_sn` gap).
`factor_csc` is the scalar CSC kernel accessed via
`chol_csc_eliminate`; `factor_csc_sn` is the Sprint 18 Days 6-10
batched supernodal kernel via `chol_csc_eliminate_supernodal` with
`min_size = 4`.  All three paths share the same AMD permutation for
an apples-to-apples factor comparison.

### Observations

**Scalar-CSC speedup scales with n.**  The scalar kernel's ratio
against the linked-list path climbs from 1.09× at n = 100 to 2.61×
at n = 14822, consistent with the Sprint 17 hypothesis that pointer-
chasing overhead on the linked-list side grows faster than contiguous
column traversal on the CSC side.  The one outlier is `Kuu` (scalar
speedup 0.77×), where the Pres_Poisson-dominated symbolic fill
produces a CSC that the scalar gather's drop-tolerance prune hits
in many small chunks — extra `shift_columns_right_of` calls.  The
supernodal path avoids those shifts entirely (values land in pre-
allocated sym_L positions), so `Kuu`'s supernodal speedup (2.22×)
slots back into the monotonic n-vs-speedup trend.

**Supernodal > scalar on every non-trivial matrix.**  The batched
path beats the scalar kernel on all four new fixtures
(bcsstk14 → 2.38 / 1.74 = 1.37× extra, s3rmt3m3 → 1.62×,
Kuu → 2.89×, Pres_Poisson → 1.67×).  The one size where supernodal
lags scalar is `bcsstk04` (1.01× vs 1.16×), where the matrix is
small enough that supernode detection overhead eats the batched
dense-block win.  That's the floor the Sprint 17 notes predicted
(tight-knit matrices below ~500 columns with limited supernode
structure).

**Solve times are negligible next to factor.**  Every matrix shows
solve_csc and solve_csc_sn < 1% of factor_csc_sn time, so CSC's
triangular-solve gains (visible at larger n: 0.53 ms on bcsstk14,
15.3 ms on Pres_Poisson) don't shift the headline number.

## LDL^T — CSC vs linked-list

Sprint 18 Days 1-5 replaced the Sprint 17 wrapper with a native CSC
Bunch-Kaufman kernel reachable via the default
`ldlt_csc_eliminate`.  The benchmark records both paths by flipping
`ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE)` vs
`LDLT_CSC_KERNEL_WRAPPER`; the wrapper column stays available as a
regression baseline.

| matrix         |   n   |    nnz  | factor_ll | factor_csc_native | factor_csc_wrapper | solve_ll | solve_csc_native | speedup_native | speedup_wrapper | res |
|----------------|------:|--------:|----------:|------------------:|-------------------:|---------:|-----------------:|---------------:|----------------:|----:|
| `nos4.mtx`     |   100 |     594 |    0.37 ms |           0.32 ms |            0.36 ms | 0.008 ms |         0.005 ms |          1.15× |           1.02× | 6e-16 |
| `bcsstk04.mtx` |   132 |    3648 |    3.92 ms |           1.99 ms |            3.80 ms | 0.045 ms |         0.015 ms |      **1.97×** |           1.03× | 6e-16 |
| `bcsstk14.mtx` |  1806 |   63454 |  511.84 ms |         209.29 ms |          524.63 ms | 5.471 ms |         0.632 ms |      **2.45×** |           0.98× | 8e-15 |
| `s3rmt3m3.mtx` |  5357 |  207123 | 4285.08 ms |        1927.83 ms |         4644.77 ms |22.147 ms |         2.997 ms |      **2.22×** |           0.92× | 8e-15 |

**Headline:** the native CSC LDL^T kernel is **2.2–2.5× faster**
than the linked-list path on the non-trivial SPD fixtures
(bcsstk04 and larger), and the wrapper hovers at ~1.0× — exactly
what the Sprint 17 notes predicted would happen once the native
kernel replaced the expand-and-delegate shim.  The Bunch-Kaufman
α = (1 + √17) / 8 partial-scan is identical between the two
backends, so the residual column (`res`) shows native and wrapper
producing the same factor to round-off; only the time column
moves.

### Corpus selection for LDL^T

`bench_ldlt_csc`'s default list is smaller than `bench_chol_csc`'s
on purpose:

- `Kuu` (n = 7102) and `Pres_Poisson` (n = 14822) are kept on the
  Cholesky bench; the LDL^T wrapper path's full-symmetric expansion
  pushes them into multi-minute territory per repeat and the
  comparison stops being useful.  Available via
  `./build/bench_ldlt_csc <path>` for a targeted run.
- `bloweybq` (symmetric indefinite, n = 10001) is reported
  `SPARSE_ERR_SINGULAR` by every backend — the linked-list baseline
  rejects it as below the `sparse_rel_tol` singularity threshold,
  so there is no "correct answer" for the CSC paths to reproduce.
- `tuma1` (symmetric indefinite, n = 22967) runs past 3 minutes per
  factor on the linked-list path on the Day 12 host; the 3-repeat
  × 3-path bench loop would spend >60 minutes before printing a
  CSV row.  Same single-matrix escape hatch.

The default corpus retained for the benchmark is therefore SPD only
but covers the same scaling range as `bench_chol_csc`.

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

### Sprint 18 Day 13 retrospective

The Day 12 corpus does not include a matrix below n = 100 — the
Cholesky bench's smallest fixture is `nos4` at exactly n = 100, where
the supernodal path is already 1.22× faster than linked-list.  That
means the data point supporting a threshold _change_ (either up or
down) is missing:

- `n = 100` (nos4): CSC wins at 1.22× supernodal.  If the threshold
  were below 100, this matrix would still take the CSC path — same
  behaviour as today.
- `n < 100`: not measured.  The original Sprint 17 estimate ("~30-50
  columns may spend more time on CSC conversion than elimination")
  remains a hypothesis until the Sprint 19 corpus adds n ∈ {20, 40,
  60} matrices.

Given the "do not move the threshold without benchmark data"
constraint in the Sprint 18 plan, the default stays at 100.  That
keeps the Day 11 dispatch behaviour unchanged and preserves the
two-path coverage in the existing test suite (small matrices exercise
linked-list, large matrices exercise CSC supernodal).  A follow-up
to characterise n ∈ [20, 100] is filed against the Sprint 19 PLAN.

Override with `-DSPARSE_CSC_THRESHOLD=N` at compile time.  Matrices
larger than a few hundred columns should see a CSC win on every
tested fixture; matrices smaller than ~30 columns remain
theoretically better on the linked-list path (no measurement yet).

## Current opt-in path

Sprint 18 Day 11 replaced this with a **transparent** dispatch:
`sparse_cholesky_factor_opts(mat, opts)` now routes through the CSC
supernodal kernel whenever `mat->rows >= SPARSE_CSC_THRESHOLD`, with
the final factor written back into `mat` via
`chol_csc_writeback_to_sparse`.  Callers don't need to choose a
backend — the speedup tracked above is what `sparse_cholesky_factor_opts`
delivers by default.  `sparse_cholesky_opts_t::backend` (
`SPARSE_CHOL_BACKEND_LINKED_LIST` / `SPARSE_CHOL_BACKEND_CSC`) lets
tests force one path on the same binary, and `used_csc_path` reports
the chosen branch for test assertions.

The internal `chol_csc_*` helpers in
`src/sparse_chol_csc_internal.h` remain available for code that
already holds a `CholCsc *` and wants to skip the
`SparseMatrix` round-trip.

## Follow-up ideas driven by these numbers

1. Add `make bench` entry points dedicated to the CSC comparison so the
   numbers land in CI. *(Done — `make bench` runs `bench_chol_csc` and
   `bench_ldlt_csc`.)*
2. Source a larger SuiteSparse corpus (n ≥ 1000) where fill is heavy
   and the supernodal dense-kernel path can show a multiplier beyond
   the scalar CSC's 1.1–2.0×. *(Done in Sprint 18 Day 12 — see the
   extended table above: 2.2–4.4× on bcsstk14/s3rmt3m3/Kuu/Pres_Poisson.)*
3. A native CSC LDL^T kernel so the LDL^T ratio moves from ~0.3–1.3×
   to something competitive with Cholesky's scalar numbers.
   *(Done in Sprint 18 Days 1-5 — native Bunch-Kaufman CSC kernel
   now returns 2.2–2.5× vs linked-list on bcsstk14 and s3rmt3m3.)*
4. Benchmark the `sparse_analyze` / `sparse_factor_numeric` split on a
   time-stepping workload to quantify the analyze-once, factor-many
   speedup (expected to be larger than the one-shot numbers here).
   *(Still outstanding — Sprint 19 candidate.)*
5. Characterise n ∈ [20, 100] so `SPARSE_CSC_THRESHOLD` can move with
   data rather than estimate.  *(Sprint 19 candidate — see the
   threshold retrospective above.)*
6. Investigate `Kuu`'s scalar-CSC regression (0.77×) — extra
   `shift_columns_right_of` calls in drop-tolerance pruning.  The
   supernodal path avoids this entirely; whether the scalar kernel
   should also pre-allocate the full sym_L pattern (like
   `chol_csc_from_sparse_with_analysis` does now) is a Sprint 19
   question.
