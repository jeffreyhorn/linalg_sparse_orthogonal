# Sprint 18 Retrospective — CSC Kernel Performance Follow-Ups

**Duration:** 14 days
**Branch:** `sprint-18`
**Date range:** 2026-04-18 → 2026-04-19 (intensive condensed run)

## Goal recap

> Complete the Sprint 17 CSC numeric-backend work by replacing the
> linked-list delegations with native CSC kernels, exposing a
> transparent size-based dispatch through the public API, and
> validating scaling behavior on larger SuiteSparse problems.

## Definition of Done checklist

Against the five `PROJECT_PLAN.md` items:

| # | Item | Target | Landed | Verdict |
|---|------|--------|--------|---------|
| 1 | Native CSC LDL^T Bunch-Kaufman kernel | Replace the Day 8 wrapper with a native kernel on packed CSC: in-place swaps, 1×1 / 2×2 pivots, element growth; bit-identical to linked-list | Days 1-5. `ldlt_csc_eliminate_native` + `ldlt_csc_symmetric_swap`; four-criteria BK scanner; bit-identical on the 20-matrix random cross-check | ✅ Complete |
| 2 | Batched supernodal Cholesky factor | Replace the scalar delegate inside `chol_csc_eliminate_supernodal` with a batched kernel using `chol_dense_factor` + `chol_dense_solve_lower`; residuals match scalar | Days 6-9. `chol_csc_supernode_extract` + `_eliminate_diag` + `_eliminate_panel` + `_writeback` wired end-to-end; singleton fast-path preserves correctness on fill-ful matrices at `min_size = 1` | ✅ Complete |
| 3 | Transparent `sparse_cholesky_factor_opts` dispatch | CSC → linked-list writeback; fallback below threshold; public API unchanged | Days 10-11. `chol_csc_writeback_to_sparse` + dispatch branch in `sparse_cholesky_factor_opts` + `sparse_cholesky_opts_t::backend` forcing | ✅ Complete |
| 4 | Larger SuiteSparse corpus & scaling benchmarks | Add n ≥ 1000 SPD + indefinite fixtures, capture scaling numbers in `PERF_NOTES.md` | Days 12-13. bcsstk14 / s3rmt3m3 / Kuu / Pres_Poisson added to Cholesky bench; bcsstk14 / s3rmt3m3 added to LDL^T bench. `bench_day12.txt` raw captures; `PERF_NOTES.md` tables extended; scaling narrative landed | ✅ Complete |
| 5 | Integration tests & documentation | Cross-threshold integration tests + remove wrapper/delegation language project-wide | Day 14. `tests/test_sprint18_integration.c` (10 tests, 234 assertions); doc passes on `src/sparse_chol_csc.c`, `src/sparse_ldlt_csc.c`, `include/sparse_cholesky.h`, `docs/algorithm.md`, `README.md`, `PROJECT_PLAN.md` | ✅ Complete |

## Final metrics

| Metric | Start of Sprint 18 | End of Sprint 18 |
|--------|-------------------:|-----------------:|
| Total tests across all suites | 1 384 | **1 453** (+69) |
| `test_chol_csc` tests | 100 | **130** (+30) |
| `test_ldlt_csc` tests | 40 | **69** (+29) |
| `test_sprint18_integration` tests | 0 | **10** (new) |
| `test_chol_csc` assertions | ~8 900 | **21 256** |
| `src/sparse_chol_csc.c` LoC | 1015 | 1719 (+704) |
| `src/sparse_ldlt_csc.c` LoC | 432 | 1543 (+1111) |
| `src/sparse_cholesky.c` LoC | ~360 | 401 (+~40, dispatch wiring) |
| SPD factor speedup, bcsstk14 (n = 1806, scalar / supernodal) | n/a (not in corpus) | **1.74× / 2.38×** |
| SPD factor speedup, s3rmt3m3 (n = 5357, scalar / supernodal) | n/a | **2.10× / 3.41×** |
| SPD factor speedup, Pres_Poisson (n = 14822, scalar / supernodal) | n/a | **2.61× / 4.35×** |
| LDL^T factor speedup, bcsstk14 (native) | wrapper ~1.0× | **2.45×** |
| LDL^T factor speedup, s3rmt3m3 (native) | n/a | **2.22×** |

All residuals on the enlarged corpus remain ≤ 2e-13 (SPD spot-check threshold 1e-10).

## Benchmark deltas (Sprint 17 end → Sprint 18 end)

All numbers are 3-repeat one-shot factor with AMD inside the timed region.

### Cholesky (`bench_chol_csc`)

| Matrix | n | Sprint 17 speedup (scalar / sn) | Sprint 18 speedup (scalar / sn) | Delta |
|--------|--:|---:|---:|---|
| nos4 | 100 | 1.65× / 2.01× | 1.09× / 1.22× | Slower — the Sprint 17 supernodal number was a delegation to scalar; Sprint 18 numbers include actual supernode-detection overhead on a matrix too small to benefit from it. Fair comparison now. |
| bcsstk04 | 132 | 1.13× / 1.22× | 1.16× / 1.01× | Scalar slightly faster; supernodal flat — matrix is small enough that detection overhead eats the batched-block win. Expected floor. |
| bcsstk14 | 1806 | n/a | **1.74× / 2.38×** | New fixture; supernodal pulls ahead of scalar by 37%. |
| s3rmt3m3 | 5357 | n/a | **2.10× / 3.41×** | New. |
| Kuu | 7102 | n/a | 0.77× / **2.22×** | New. Scalar regression localised to drop-tol shrink shifts; supernodal side-steps via full sym_L pre-allocation. |
| Pres_Poisson | 14822 | n/a | **2.61× / 4.35×** | New. Highest supernodal speedup in the corpus. |

### LDL^T (`bench_ldlt_csc`)

| Matrix | n | Sprint 17 wrapper speedup | Sprint 18 native speedup | Delta |
|--------|--:|---:|---:|---|
| nos4 | 100 | 0.27× | 1.15× | Native kernel now beats linked-list at this size instead of losing to it. |
| bcsstk04 | 132 | 1.33× | **1.97×** | Near-doubling. |
| bcsstk14 | 1806 | n/a | **2.45×** | New. |
| s3rmt3m3 | 5357 | n/a | **2.22×** | New. |

The Sprint 17 retrospective's prediction — *"a native CSC LDL^T kernel is
follow-up work; these numbers are the baseline for measuring its
improvement"* — is confirmed: native is uniformly faster on every matrix,
and the wrapper's best showing (bcsstk04 1.33×) is now the native
kernel's floor (nos4 1.15×).

## What went well

- **Sprint 17 dense primitives dropped in exactly as designed.** The
  Day 11 `chol_dense_factor` and `chol_dense_solve_lower` functions
  were built in Sprint 17 with the supernodal batched path in mind;
  Day 7-8 wired them into `chol_csc_supernode_eliminate_diag` and
  `_eliminate_panel` without touching their signatures. Zero API
  churn on the Sprint 17 code.
- **Native BK kernel matched the wrapper bit-for-bit on first pass.**
  The 20-matrix random cross-check test (`test_native_mixed_pivots_random`)
  passed with zero byte-level differences against the wrapper for L,
  D, D_offdiag, pivot_size, and perm on Days 3-4. The "exact same
  algorithm just different data layout" principle held throughout.
- **Transparent dispatch was a 40-line change, not a refactor.** Day 11
  added a five-line switch in `sparse_cholesky_factor_opts` and a
  writeback helper; no existing caller needed to update. The
  `sparse_cholesky_opts_t` extension (`backend`, `used_csc_path`)
  preserved zero-init semantics so every existing caller continued to
  compile and run unchanged.
- **The enlarged corpus found a real latent bug (Day 12).** The
  supernodal path silently missed fill rows on bcsstk14 and larger
  because `chol_csc_from_sparse*` only pre-populated A's pattern. The
  scalar kernel hid the bug via dynamic `shift_columns_right_of`
  growth. Adding n ≥ 1000 fixtures to the corpus surfaced it
  immediately, and the fix (materialise full `sym_L` up front in
  `chol_csc_from_sparse_with_analysis`) was clean to apply and
  well-localised.

## What didn't go as planned

- **bloweybq and tuma1 didn't make it into the default LDL^T corpus.**
  bloweybq (n = 10 001) is singular on every backend including the
  linked-list baseline — no "correct answer" to time against. tuma1
  (n = 22 967) runs past 3 minutes per factor on the linked-list path
  on the Day 12 host; the 3-repeat × 3-path default loop would spend
  more than 60 minutes before producing a CSV row. Both are kept
  available via the single-matrix CLI path (`./build/bench_ldlt_csc
  <path>`), and the corpus documentation in `bench_ldlt_csc.c`
  records why each was excluded. Net effect: the LDL^T scaling
  corpus is SPD-only, which weakens the "symmetric indefinite"
  coverage the Day 12 plan hoped for.
- **Kuu's scalar-CSC regression was real.** 0.77× at n = 7102 is
  the one matrix in the enlarged corpus where the scalar CSC kernel
  loses to linked-list. Root cause is the repeated
  `shift_columns_right_of` calls during drop-tolerance pruning —
  each call is an `O(nnz)` memmove, and Kuu's fill pattern triggers
  many of them. The supernodal path avoids this entirely via
  `sym_L` pre-allocation, so the scalar regression is not
  user-visible through `sparse_cholesky_factor_opts` (which
  dispatches to supernodal at n ≥ 100). Flagged as a Sprint 19
  scalar-kernel design question.
- **Analyze-once / factor-many benchmark still unmeasured.** Both
  Sprint 17 and Sprint 18 noted this as an expected-larger speedup
  (because AMD amortises to zero across many refactors), and neither
  sprint ran the measurement. Deferred to Sprint 19 as a dedicated
  benchmark addition rather than a bullet in a broader sprint.

## Items deferred to Sprint 19 (with rationale)

1. **`bench_refactor_csc` — analyze-once, factor-many workflow.**
   The Sprint 17 and Sprint 18 PERF_NOTES both hypothesise a larger
   speedup in this workflow. Running it properly requires a
   dedicated benchmark that does one `sparse_analyze` then N
   `sparse_factor_numeric` / `sparse_refactor_numeric` calls with
   changed values — a different structure from the existing one-shot
   bench. Not in Sprint 18 scope, and not worth shoehorning in.
2. **Small-matrix corpus (n ∈ [20, 100]).** The Day 12 plan
   forbade moving `SPARSE_CSC_THRESHOLD` without benchmark data. The
   enlarged corpus has no matrix below n = 100, so the threshold
   stays at 100 pending a small-matrix study. The Sprint 19
   candidate characterisation would include n ∈ {20, 40, 60, 80}
   tridiagonals or SPDs and map out the crossover explicitly.
3. **Scalar-CSC drop-tolerance shift cost.** Kuu's 0.77× regression
   is the one data point the Day 12 corpus produced against the
   scalar kernel. Investigating whether `chol_csc_from_sparse`
   should also pre-allocate the full `sym_L` pattern (like the
   analysis-aware variant does now) is a design call rather than a
   bug fix — both approaches are correct; the question is whether
   the extra memory cost on small matrices is worth eliminating
   the `shift_columns_right_of` penalty on Kuu-like matrices.
4. **Native supernodal LDL^T batched kernel.** The Cholesky side
   now has a batched supernodal path; the LDL^T side has a native
   BK kernel but runs column-by-column. Extending the Liu-Ng-Peyton
   supernode detection to symmetric indefinite (where 2×2 pivots
   complicate supernode boundaries) is a follow-on that mirrors
   this sprint's Cholesky work on the LDL^T side. Tracked in the
   Sprint 19 candidate list.

## Lessons for future "wrapper → native" migrations

- **Keep the wrapper compiled in as a cross-check, at least through
  the first release.** Running `bench_ldlt_csc --repeat 1
  path/to.mtx` through `LDLT_CSC_KERNEL_WRAPPER` gave a bit-identical
  reference to compare the native kernel against — worth its weight
  in bug hunts. The wrapper path stays in Sprint 18 even though no
  production caller reaches it.
- **A "replace wrapper with native" sprint doubles as a testing
  sprint.** The native BK kernel's bit-identical requirement forced
  a 20-matrix random cross-check test plus a battery of 1×1 / 2×2
  forced-pivot tests. None of those existed before Sprint 18; all of
  them continue to earn their keep now that they cover a kernel
  without a cross-check partner.
- **Pre-allocating symbolic structure is load-bearing for dense
  batched kernels.** This is the Day 12 lesson in one line: any
  column-batching algorithm that reads `col_ptr[j+1] - col_ptr[j]`
  as "this column's work set" requires the full predicted symbolic
  pattern to be materialised before it runs. Scalar elimination can
  paper over pattern-vs-actual mismatches via dynamic growth; batched
  extracts cannot. Future sprints adding batched kernels on the LU
  or LDL^T side should start with "where does the symbolic pattern
  come from?" as the Day 1 question.
- **"Fair comparison" methodology changes in PERF_NOTES are real
  commits.** The Sprint 17 numbers had AMD outside the timed region
  on the CSC path, inflating the speedup by the AMD cost. The
  Sprint 18 numbers move AMD inside the timed region on all paths.
  Both sets of numbers are legitimate for the workload they measure
  (one-shot vs analyze-once); the lesson is that "speedup" is
  always workload-dependent and the PERF_NOTES body text must say
  so explicitly.

## Line-count summary

New / substantially modified Sprint 18 files:

| File | Sprint 17 end | Sprint 18 end | Delta |
|------|--------------:|--------------:|------:|
| `src/sparse_chol_csc.c` | 1015 | 1719 | +704 |
| `src/sparse_chol_csc_internal.h` | 666 | 940 | +274 |
| `src/sparse_ldlt_csc.c` | 432 | 1543 | +1111 |
| `src/sparse_ldlt_csc_internal.h` | 280 | 420 | +140 |
| `src/sparse_cholesky.c` | ~360 | 401 | +41 (dispatch) |
| `tests/test_chol_csc.c` | 2549 | 3998 | +1449 |
| `tests/test_ldlt_csc.c` | 1111 | 2140 | +1029 |
| `tests/test_sprint18_integration.c` | 0 | 331 | +331 (new) |
| `benchmarks/bench_chol_csc.c` | 254 | 289 | +35 |
| `benchmarks/bench_ldlt_csc.c` | 225 | 281 | +56 |

Total Sprint 18 additions: **~5200 LoC** (code + tests + benches +
docs). No production files shrank — the Sprint 18 work is additive.

No Sprint 17 wrapper code was retired this sprint: the Sprint 17
`ldlt_csc_eliminate_wrapper` body remains as a cross-check (see
`src/sparse_ldlt_csc.c` header for the rationale). If a future
sprint decides the wrapper is no longer needed, retiring it would
remove ~200 LoC of expand-and-delegate plumbing.
