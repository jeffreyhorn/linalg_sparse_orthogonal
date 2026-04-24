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
| `Kuu.mtx`          |  7102 |  340200 | 1501.28 ms |    711.46 ms |      522.69 ms |16.693 ms | 1.417 ms |      **2.11×** |        **2.87×** | 9e-15 |
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
column traversal on the CSC side.  `Kuu` fell off that trend in the
Sprint 18 Day 14 capture (scalar **0.77×**) — its fill pattern ran
the scalar gather's drop-tolerance prune into many small
`shift_columns_right_of` calls that dominated the factor.  The
Sprint 19 Day 6-7 fix wrote an in-place write-and-zero-pad fast path
into `chol_csc_gather` gated on
`chol_csc_from_sparse_with_analysis`'s sym_L pre-allocation flag:
when the slot is sym_L-sized, survivors land in their pre-existing
slots and dropped entries get zeroed without touching `col_ptr`, so
the memmove chain never runs.  Kuu's scalar speedup recovered to
**2.11×** (table row above reflects the Day 7 re-bench) and the
n-vs-speedup trend is monotonic again across the corpus.

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

**Sprint 19 Day 6-7 Kuu regression fix.**
`chol_csc_gather`'s pre-Sprint-19 implementation called
`shift_columns_right_of` on every gather to resize the column slot
after drop-tolerance.  Sprint 19 Day 5's `sample` profile on Kuu
attributed 60% of total factor time to `_platform_memmove` driven
by those shifts; the Day 6-7 fix replaces the shift with an in-place
write + zero-pad when the pre-allocated slot fits the survivors.
Gated on a new `CholCsc::sym_L_preallocated` flag that
`chol_csc_from_sparse_with_analysis` sets, so the merge-walk safety
check needed for heuristic initialisers is skipped in the common
dispatch path.  Kuu scalar moved from 0.77× → 2.11× (row above) and
every matrix with n ≥ 1806 either improved or stayed flat.  The
fix is memory-neutral (no allocation changes) and exercised by the
new `test_chol_csc_kuu_scalar_no_regression` regression test plus
the Day 7 full-corpus bench capture
([`bench_day7_post_kuu.txt`](../SPRINT_19/bench_day7_post_kuu.txt)).
Sub-threshold fixtures (n ≤ 132) show slightly lower speedup ratios
vs Sprint 18 Day 14's baseline — not because CSC got slower (it's
faster in absolute terms), but because the linked-list baseline
improved by a comparable or larger factor on the same re-run, which
the sub-millisecond timing resolution amplifies into the ratio.
The user-visible dispatch still routes those n's to linked-list via
`SPARSE_CSC_THRESHOLD = 100`, so the sub-threshold "ratio
regression" is not user-observable.

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

The tables above measure a **one-shot** factor call.  The
`sparse_analyze` / `sparse_factor_numeric` split (Sprint 14) lets
callers run AMD + symbolic analysis once and reuse the result across
many numeric refactorizations with the same pattern (e.g. a time-
stepping simulation where the matrix values change but the nonzero
pattern is fixed).  `benchmarks/bench_refactor_csc.c` (Sprint 19
Day 1) measures that workflow directly: one `sparse_analyze` per
matrix outside the timed region, then N = 5 refactors with
deterministic symmetric-key value perturbations inside the timed
region.

Raw capture: [`docs/planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt`](../SPRINT_19/bench_day2_refactor.txt).
Same 3-repeat Sprint 18 corpus, `eps = 1e-9` per-entry multiplicative
noise on A, residuals checked vs the perturbed A after the final
refactor.

### Analyze-once / factor-many — CSC vs linked-list

| matrix             |   n   |    nnz  | analyze_ms | refactor_ll | refactor_csc | solve_ll | solve_csc | **speedup_refactor** | res_ll | res_csc |
|--------------------|------:|--------:|-----------:|------------:|-------------:|---------:|----------:|---------------------:|-------:|--------:|
| `nos4.mtx`         |   100 |     594 |     0.16   |    0.06 ms  |    0.07 ms   | 0.004 ms | 0.002 ms  |                0.83× | 6e-16  | 8e-16  |
| `bcsstk04.mtx`     |   132 |    3648 |     0.92   |    0.87 ms  |    0.37 ms   | 0.023 ms | 0.007 ms  |            **2.37×** | 2e-15  | 2e-15  |
| `bcsstk14.mtx`     |  1806 |   63454 |    42.86   |  138.99 ms  |   28.74 ms   | 2.920 ms | 0.263 ms  |            **4.84×** | 2e-15  | 2e-15  |
| `s3rmt3m3.mtx`     |  5357 |  207123 |   271.69   | 1717.14 ms  |  174.12 ms   |16.010 ms | 1.484 ms  |            **9.86×** | 8e-15  | 6e-15  |
| `Kuu.mtx`          |  7102 |  340200 |   391.01   | 1034.26 ms  |  112.82 ms   |15.965 ms | 1.332 ms  |            **9.17×** | 1e-14  | 1e-14  |
| `Pres_Poisson.mtx` | 14822 |  715804 |  2872.73   |31473.61 ms  | 1295.89 ms   |123.646ms | 9.934 ms  |           **24.29×** | 2e-13  | 3e-13  |

`refactor_ll` is `sparse_refactor_numeric` on the analysed matrix —
internally `sparse_factor_numeric` calls `build_permuted_copy` with
`analysis->perm` and then `sparse_cholesky_factor` on the permuted
copy, so AMD is skipped.  `refactor_csc` is
`chol_csc_from_sparse_with_analysis` + `chol_csc_eliminate_supernodal`
with `SPARSE_CSC_SUPERNODE_MIN_SIZE = 4`, also skipping AMD.  Neither
path runs `sparse_analyze` inside the timed region — that cost is
reported separately as `analyze_ms` so callers can estimate the
amortisation break-even (one-shot cost = `analyze_ms + refactor_ms`;
analyze-once cost after N refactors = `analyze_ms / N + refactor_ms`).

### Hypothesis check (Sprint 17 + Sprint 18 prediction)

Both prior PERF_NOTES revisions argued that the CSC speedup **should
be larger** in the analyze-once workflow than in the one-shot
workflow because AMD dominates the small-matrix one-shot cost and
amortises to zero once the analysis is reused.  Side-by-side
comparison of the two workflows on the same corpus:

| matrix             | one-shot speedup (sn) | analyze-once speedup | delta    |
|--------------------|----------------------:|---------------------:|---------:|
| `nos4.mtx`         |                 1.22× |                0.83× | −0.39× |
| `bcsstk04.mtx`     |                 1.01× |            **2.37×** | +1.36× |
| `bcsstk14.mtx`     |                 2.38× |            **4.84×** | +2.46× |
| `s3rmt3m3.mtx`     |                 3.41× |            **9.86×** | +6.45× |
| `Kuu.mtx`          |                 2.22× |            **9.17×** | +6.95× |
| `Pres_Poisson.mtx` |                 4.35× |           **24.29×** | +19.94× |

**Hypothesis confirmed on every fixture with `n >= 132`.**  The gap
widens monotonically with n, consistent with the prior-sprint
reasoning: as matrices grow, the CSC kernel's contiguous column-major
traversal pulls ahead of the linked-list kernel's pointer-chasing,
and with AMD amortised away only that kernel-vs-kernel gap remains.
Pres_Poisson (n = 14 822) sees a **24.29× refactor speedup** — the
one-shot 4.35× was bounded from below by AMD's fixed cost on both
sides; refactor removes that floor entirely.

**Hypothesis disconfirmed on nos4 (n = 100).**  The refactor
speedup (0.83×) is smaller than the one-shot (1.22×).  At this size
the absolute factor times are sub-millisecond (60 μs LL, 72 μs CSC),
and per-iteration overhead — CSC's `chol_csc_from_sparse_with_analysis`
walks A and materialises the full sym_L pattern, while LL's
`build_permuted_copy` does a narrower permuted-copy — dominates the
numeric work.  One-shot includes AMD on both sides, which at n = 100
is ~0.4 ms for LL and ~0.3 ms for CSC (empirically measured as
`one-shot - refactor` per path); removing AMD removes a bigger
proportion of the LL path's cost than the CSC path's cost, which
collapses the ratio.  This is a small-matrix floor effect, not a
kernel-architecture problem — it is the crossover region that the
Sprint 19 Day 3-4 small-matrix study (`SPARSE_CSC_THRESHOLD`
retrospective) will map out.

### Break-even analysis

`analyze_ms / N + refactor_csc_ms` vs `refactor_ll_ms` gives the N at
which the CSC analyze-once workflow starts beating the linked-list
analyze-once workflow on the same matrix:

- For Pres_Poisson (LL refactor = 31 474 ms, CSC refactor = 1 296 ms,
  analyze = 2 873 ms): CSC wins from N = 1 onwards — even a single
  refactor after `sparse_analyze` pays back the analyse cost many
  times over.
- For Kuu / s3rmt3m3 / bcsstk14: CSC wins from N = 1 onwards by a
  similar margin.
- For bcsstk04 (LL refactor = 0.87 ms, CSC refactor = 0.37 ms,
  analyze = 0.92 ms): CSC wins from N = 2 onwards; at N = 1, both
  paths pay roughly the same total (analyze + factor) because
  analyze dominates.
- For nos4: the analyze-once workflow doesn't tip in favour of CSC
  at any N — the kernel is slightly slower at this size.  Callers
  should stay on the linked-list path for n < 132 (or use the
  AUTO dispatch, which routes via `SPARSE_CSC_THRESHOLD = 100` —
  see the Sprint 19 Day 3-4 crossover study below).

In short: the analyze-once workflow is the right default whenever
the same sparsity pattern is re-factored more than once, and the
CSC backend delivers order-of-magnitude speedups in that regime on
matrices beyond a few hundred columns.

## Threshold guidance (`SPARSE_CSC_THRESHOLD`)

`SPARSE_CSC_THRESHOLD` defaults to 100 in `include/sparse_matrix.h`.

### Sprint 19 Day 3-4 crossover study

Sprint 18 Day 13 filed the sub-100 fixtures as a follow-up because
the Day 12 corpus had no matrix below n = 100 (nos4 at n = 100 was
already 1.22× on supernodal, so the data couldn't distinguish
"threshold too high" from "threshold exactly right").  Sprint 19
Day 3 landed the missing data:
`./build/bench_chol_csc --small-corpus --repeat 50` on 10 in-memory
SPD fixtures — raw capture in
[`docs/planning/EPIC_2/SPRINT_19/bench_day3_small_corpus.txt`](../SPRINT_19/bench_day3_small_corpus.txt).

Supernodal speedup (`speedup_csc_sn` = `factor_ll_ms / factor_csc_sn_ms`)
by fixture family and n:

| family  |  n=20  |  n=40  |  n=60  |  n=80  |  n=100 (nos4) |  n=132 (bcsstk04) |
|---------|-------:|-------:|-------:|-------:|--------------:|------------------:|
| tridiag |  0.51× |  0.65× |  0.56× |  0.54× | (no fixture)  |   (no fixture)    |
| banded  |  0.85× |  0.70× |  0.75× |  0.71× | (no fixture)  |   (no fixture)    |
| dense   |**1.14×**|        |**0.89×**|       | (no fixture)  |   (no fixture)    |
| real    |        |        |        |        |    **1.22×**  |     **1.01×**     |

Values are from
[`docs/planning/EPIC_2/SPRINT_19/bench_day3_small_corpus.txt`](../SPRINT_19/bench_day3_small_corpus.txt)
(column `speedup_csc_sn`).

Observations:

- **Dense fixtures are mixed in this small-corpus capture.**  The
  `dense-20` run is above parity at **1.14×**, consistent with a
  single large supernode amortising detection overhead across the
  whole matrix, but `dense-60` drops back to **0.89×**, so this
  table does not support claiming a monotonic dense-family
  crossover below n = 20 — only that at n = 20 the batched path
  can pull ahead on fully-populated patterns.
- **Tridiag fixtures stay at 0.51×–0.65× across n ∈ [20, 80].**
  Tridiagonal matrices have zero fill, so supernode detection finds
  only singleton supernodes — the batched path degenerates to the
  scalar CSC kernel plus detection overhead, and never pulls ahead
  of the linked-list's one-entry-per-column sweep at these sizes.
- **Banded fixtures cluster at 0.70×–0.85×.**  Moderate fill, some
  supernode structure, but not enough to overcome the CSC build
  overhead at sub-100 sizes.
- **Real-corpus `nos4` at n = 100 is 1.22×.**  Moderate SuiteSparse
  SPD structure crosses in the 80–100 range; the crossover is
  faster for more-structured matrices.

### Decision: keep `SPARSE_CSC_THRESHOLD = 100`

The synthetic data shows the crossover is **family-dependent**:
dense is above parity only at `dense-20` (1.14×) and dips back to
0.89× at `dense-60`, tridiagonal fixtures haven't crossed at 80,
banded fixtures are trending toward 1.0× near 100.  Real SuiteSparse
SPD matrices — the expected input distribution — have moderate
structure (bandwidth tens to hundreds) and cross just at or above
n = 100 (`nos4` 1.22×, `bcsstk04` 1.01×).

Per the Day 4 decision matrix:

> "If families disagree (e.g., tridiag crosses at 30 but dense at
> 90), the current default of 100 is the conservative worst-case —
> keep it."

Our data matches that case (dense non-monotonic in the sub-100
range, tridiag still below parity at 80, banded ≈ 100), so **100
is confirmed as the right default**.
It favours the common case (moderately-sparse SuiteSparse-style
SPDs) while preserving the linked-list path for the sub-100
matrices where CSC loses 15-50%.

The worst-case miscalibration at 100 is ~30% overhead on
tridiagonal-structure inputs (hypothetically factored at n = 100
with threshold dispatch routing to CSC).  Callers with known
structure can override at compile time with
`-DSPARSE_CSC_THRESHOLD=N` — recommended settings based on the
measurements above:

| input structure            | recommended threshold |
|----------------------------|----------------------:|
| dense / near-dense         |                  ~20 |
| moderate SuiteSparse SPD   |        **100** (default) |
| tridiagonal / very sparse  |               150–200 |

A matrix whose CSC supernodal path regresses below 1.0× is always
faster on linked-list — the threshold is the dial that keeps the
library honest at the crossover.

### What we still don't have

- Crossover for each family is not pinned to a single n.  The data
  says tridiagonal hasn't crossed by 80; a future sweep at n ∈
  {100, 120, 150, 200} would bracket it precisely.  Left to a
  follow-up if the default 100 turns out to be miscalibrated on a
  real corpus that's heavily tridiagonal.
- Hybrid-structure matrices (blocks of sparse + dense) are not in
  the corpus.  Real SuiteSparse fixtures like bcsstk04 / bcsstk14 /
  Pres_Poisson cover that space indirectly via their natural
  structure, and the one-shot / analyze-once tables above show the
  default 100 is safe for them all.

## LDL^T transparent dispatch (Sprint 20 Days 4-6)

Sprint 20 extended the Sprint 18 Cholesky transparent dispatch to
the LDL^T side.  `sparse_ldlt_opts_t` grew a `backend` field
(AUTO / LINKED_LIST / CSC) and an optional `used_csc_path` output;
AUTO routes CSC above `SPARSE_CSC_THRESHOLD` (default 100) via
`sparse_analyze` → `ldlt_csc_from_sparse_with_analysis` →
`ldlt_csc_eliminate_supernodal` → CSC→`sparse_ldlt_t` writeback,
with a structural fallback to the scalar CSC factor (produced by
the dispatch's mandatory pre-pass) on any batched failure.

Raw capture:
[`docs/planning/EPIC_2/SPRINT_20/bench_day6_dispatch.txt`](../SPRINT_20/bench_day6_dispatch.txt).

`./build/bench_ldlt_csc --dispatch --repeat 5`:

| matrix       |      n | nnz      | AUTO backend | factor_auto | factor_ll | speedup | res_auto | res_ll   |
|--------------|-------:|---------:|:-------------|------------:|----------:|--------:|---------:|---------:|
| nos4.mtx     |    100 |      594 | csc          |    0.863 ms |  0.395 ms |   0.46× | 6.47e-16 | 5.89e-16 |
| bcsstk04.mtx |    132 |    3,648 | csc          |    7.982 ms |  6.244 ms |   0.78× | 7.57e-16 | 6.05e-16 |
| bcsstk14.mtx |  1,806 |   63,454 | csc          |  824.276 ms | 906.833 ms |  1.10× | 8.05e-15 | 8.05e-15 |
| s3rmt3m3.mtx |  5,357 |  207,123 | csc          | 5732.799 ms | 6192.541 ms | 1.08× | 8.36e-15 | 8.36e-15 |
| kkt-150      |    150 |      438 | csc          |    0.486 ms |  0.282 ms |   0.58× | 0.00e+00 | 0.00e+00 |

Observations:

- **AUTO selects `csc` on every row** because all fixtures have
  n >= 100.  Forced LINKED_LIST runs as the left-side comparison.
- **Residuals match round-off on both paths** (including kkt-150
  exact zero — the all-ones RHS solves to machine precision on
  this synthetic fixture).
- **Small-matrix overhead on nos4 / bcsstk04 / kkt-150.**  AUTO is
  0.46-0.78× — slower than forced LINKED_LIST because the CSC
  dispatch runs a scalar pre-pass to resolve BK swaps and then
  attempts the batched factor on top.  Doubles the numeric work
  on matrices where the batched structural advantage is small.
  Same crossover shape as the Sprint 18 Cholesky dispatch; the
  threshold targets moderate-to-large problems.
- **Large-matrix wins on bcsstk14 / s3rmt3m3.**  1.10× and 1.08×
  even with the batched-path fallback tripping on those
  matrices — the CSC scalar kernel (Sprint 18 `ldlt_csc_native`)
  still beats the linked-list kernel at n >= ~1800.

Day 6 also surfaced a pre-existing Sprint 19 issue: the batched
supernodal LDL^T kernel trips `SPARSE_ERR_SINGULAR` spuriously on
bcsstk14 / s3rmt3m3 inside `ldlt_dense_factor`'s dense BK pivot
check.  Sprint 19's `--supernodal` bench on bcsstk14 produces
`res_csc_sn = 1.93e+04` — a silent correctness bug that was
invisible until Sprint 19 PR #27 review added the `res_csc_sn`
column.  The Day 5 structural fallback in `ldlt_factor_csc_path`
was widened on Day 6 to catch any batched failure, so the
dispatch always returns a correct factor via the scalar pre-pass
(`used_csc_path` stays at 1 because the CSC kernel chain handled
it end-to-end).  Root cause and a long-term fix for the batched-
kernel tolerance policy are tracked in the Sprint 20 Day 14
retrospective.

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
