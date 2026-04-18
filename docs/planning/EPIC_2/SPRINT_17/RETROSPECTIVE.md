# Sprint 17 Retrospective — CSR/CSC Numeric Backend for Cholesky and LDL^T

**Duration:** 14 days
**Branch:** `sprint-17`
**Date range:** 2026-04-16 → 2026-04-17 (intensive condensed run)

## Goal recap

> Extend the CSR working-format strategy (proven in Sprint 10 for LU) to
> Cholesky and LDL^T, making compressed formats the primary numeric
> backend for all direct solvers while keeping the linked list as the
> mutable front end.

## Definition of Done checklist

Against the six PROJECT_PLAN.md items:

| # | Item | Target | Landed | Verdict |
|---|------|--------|--------|---------|
| 1 | CSC format for Cholesky | Convert linked-list ↔ CSC, symbolic-aware allocation | `CholCsc` struct, `chol_csc_alloc` / `_free` / `_grow`, `chol_csc_from_sparse`, `chol_csc_from_sparse_with_analysis`, `chol_csc_to_sparse`, `chol_csc_validate` | ✅ Complete |
| 2 | CSC Cholesky elimination | Scatter-gather kernel, fill-in, drop tolerance, triangular solve | Day 4-6 scalar kernel with `scatter` / `cmod` / `cdiv` / `gather` / `end_column` workspace helpers, `chol_csc_solve` / `_perm`, `chol_csc_factor` / `_solve` shims | ✅ Complete |
| 3 | CSC LDL^T elimination | BK 1×1/2×2 in CSC | `LdltCsc` storage + native CSC solve ✅; BK elimination currently delegates to the linked-list kernel via full-symmetric expansion ⚠️ | ⚠️ Partial (native BK kernel deferred) |
| 4 | Supernodal detection + dense kernels | Detection + batched dense factor per supernode | Detection (`chol_csc_detect_supernodes`) ✅, dense primitives (`chol_dense_factor`, `chol_dense_solve_lower`) ✅, `chol_csc_eliminate_supernodal` dispatch ✅; batched dense-kernel integration deferred | ⚠️ Partial (detection ships, batched factor deferred) |
| 5 | Benchmarks and validation | ≥ 2× speedup vs linked-list | One-shot, AMD included on both paths: **nos4 scalar 1.65× / supernodal 2.01×; bcsstk04 scalar 1.13× / supernodal 1.22×** Cholesky factor speedup on CSC. Residuals match linked-list to 1e-15 | ⚠️ Partial — target met on `nos4` supernodal, below target on `bcsstk04` for one-shot. Analyze-once / factor-many workflow expected to exceed 2× but not yet benchmarked |
| 6 | Documentation | README + algorithm.md + file-level design comments | README perf section + feature-list update; `docs/algorithm.md` 3 new sections; `PERF_NOTES.md`; file-level design comments in both .c files; cross-link in `sparse_lu_csr.h` | ✅ Complete |

## Final metrics

| Metric | Start of Sprint 17 | End of Sprint 17 |
|--------|-------------------:|-----------------:|
| Test suites | 35 | 41 |
| Total unit tests | ~1145 | **1384** (+239) |
| Public + internal API surface | ~150 | +~35 (`chol_csc_*`, `ldlt_csc_*`, `chol_dense_*`) |
| SuiteSparse SPD factor speedup (bcsstk04, one-shot scalar / sn) | 1× baseline | **1.13× / 1.22×** |
| SuiteSparse SPD factor speedup (nos4, one-shot scalar / sn) | 1× baseline | **1.65× / 2.01×** |

New Sprint 17 files (total **6532 LoC** — code + tests + benches + docs):

| File | LoC |
|------|----:|
| `src/sparse_chol_csc_internal.h` | 666 |
| `src/sparse_chol_csc.c` | 1015 |
| `src/sparse_ldlt_csc_internal.h` | 280 |
| `src/sparse_ldlt_csc.c` | 432 |
| `tests/test_chol_csc.c` | 2549 |
| `tests/test_ldlt_csc.c` | 1111 |
| `benchmarks/bench_chol_csc.c` | 254 |
| `benchmarks/bench_ldlt_csc.c` | 225 |

New Sprint 17 test counts:

| Suite | Tests | Assertions |
|-------|------:|-----------:|
| `test_chol_csc` | 100 | 20 682 |
| `test_ldlt_csc` | 40 | 314 |

Benchmark numbers (5-repeat `make bench`, on-machine, one-shot factor
with AMD reorder inside the timed region on both paths):

| Matrix | n | nnz | factor_ll | factor_csc | factor_csc_sn | **speedup (scalar / sn)** | residual |
|--------|---:|----:|----------:|-----------:|--------------:|--------------------------:|---------:|
| Cholesky `nos4` | 100 | 594 | 2.02 ms | 1.22 ms | 1.00 ms | **1.65× / 2.01×** | 6e-16 |
| Cholesky `bcsstk04` | 132 | 3648 | 8.03 ms | 7.12 ms | 6.61 ms | **1.13× / 1.22×** | 1e-15 |
| LDL^T `nos4` | 100 | 594 | 0.64 ms | 2.41 ms | — | 0.27× | 6e-16 |
| LDL^T `bcsstk04` | 132 | 3648 | 12.76 ms | 9.63 ms | — | 1.33× | 1e-15 |

## What went well

- **Sprint 10 pattern translated directly.** The CSR LU scatter-gather
  template (`scatter → cmod → cdiv → gather`) mapped cleanly onto
  Cholesky with column pointers instead of row pointers.  Day 4's
  scaffolding shipped in one session, Day 5's fill-in handling in
  another — no algorithm-level surprises.
- **Sprint 14 symbolic analysis paid off on Day 3.** The exact
  `sym_L.nnz` prediction let `chol_csc_from_sparse_with_analysis`
  size capacity precisely; elimination never called `chol_csc_grow`
  on any test matrix whose prediction matched.  This is exactly what
  Sprint 14 promised.
- **CSC was the right choice over CSR for Cholesky.** Both triangular
  sweeps (forward `L·y = b` and backward `L^T·x = y`) walk the same
  column slice — no transpose materialisation.  The header design
  comment makes this explicit for future maintainers.
- **Fair-comparison methodology surfaced.**  Day 12's initial numbers
  (2.6× / 3.5×) were measured with AMD reordering outside the CSC
  timed region, which inflated the ratio by the AMD cost.  Round-10
  review feedback flagged this; re-measuring with AMD inside both
  paths' timed regions brought the one-shot numbers to honest
  1.13–2.01× territory.  The fair numbers are what ship; the original
  numbers were a measurement bug rather than a capability bug.  The
  analyze-once / factor-many workflow (Sprint 14 split) is where the
  CSC advantage is expected to re-emerge, since AMD amortizes away
  across many numeric refactorizations of the same pattern.
- **Honest documentation of the wrapper.** The Day 8 LDL^T wrapper is
  explicitly documented as a baseline, with its header spelling out
  why a native kernel is future work.  Tests still cross-check
  bit-for-bit against the linked-list path, so any native kernel
  replacement has a strong correctness oracle.

## What didn't

- **Native CSC LDL^T kernel did not land.** Day 8's 10-hour budget
  proved tight for implementing full Bunch-Kaufman (four criteria,
  2×2 block pivots, symmetric swaps in packed CSC storage,
  element-growth tracking).  Decision: ship a correctness-first
  wrapper that delegates to `sparse_ldlt_factor`, with a clear header
  design block explaining the choice.  The solve path *is* native CSC
  (Day 9), so half the LDL^T pipeline lives in compressed form.
- **Batched supernodal Cholesky factor did not land.** Day 11's
  batched-dense-kernel integration requires a workspace layout change
  (per-supernode m×s dense panel with cross-supernode cmod
  accumulation), which is a substantial refactor on its own.
  Decision: ship the dense primitives tested standalone plus the
  detection-and-dispatch entry point, and let a future sprint wire
  the batched kernel without API changes.
- **Transparent dispatch via `sparse_cholesky_factor_opts` did not
  land** in Day 12.  The writeback of a CSC factor into a caller-
  owned `SparseMatrix` in place is a non-trivial internal refactor
  that risked destabilising existing tests.  Decision: document the
  opt-in path (`chol_csc_factor_solve`), add `SPARSE_CSC_THRESHOLD`
  as a hint, flag transparent dispatch as follow-up.
- **Initial tridiagonal supernode test was wrong (Day 10).** I
  expected "no supernodes" for a tridiagonal L at `min_size = 2`,
  but the last two columns of any tridiagonal L (pattern `{n-2,
  n-1}` then `{n-1}`) satisfy all three Liu-Ng-Peyton conditions.
  The kernel was right, my expectation was wrong.  Test updated;
  added a comment explaining the trailing-chain behaviour.
- **A clang-tidy false positive slowed Day 9.** The path-sensitive
  array-bound analyzer couldn't prove `k+1 < n` inside the 2×2-pivot
  branch of `ldlt_csc_solve` even though `ldlt_csc_validate`
  enforces the invariant.  Required three `NOLINT` markers with
  justifying comments to get lint green.
- **First forced-2×2 test (Day 8) used a matrix that took a 1×1 pivot
  with row swap instead.** BK Criterion 3 matches when the trailing
  diagonal is ≥ α·|off-diag|; the original `[[0, 1], [1, 2]]` fell
  through to a swap-and-1×1 pivot.  Fixed by using `[[0.1, 1], [1,
  0.3]]` where both diagonals are small enough to force Criterion 4.

## Items deferred (with rationale)

All deferrals are explicit in the code with `/* follow-up */` comments
and in the PROJECT_PLAN.md status column:

1. **Native CSC LDL^T Bunch-Kaufman kernel.** The Day 8 wrapper is
   numerically correct; replacing the linked-list delegation with a
   native kernel is purely a performance follow-up.  Expected to live
   in Sprint 18+ alongside similar work.
2. **Batched dense-kernel supernodal Cholesky factor.**
   `chol_csc_eliminate_supernodal` currently detects supernodes and
   delegates to the scalar kernel.  Replacing the delegation with a
   batched dense factor on each supernode's diagonal block + panel
   would drop elimination time further for structural-mechanics
   matrices.  Builds directly on the Day 11 dense primitives.
3. **Transparent `sparse_cholesky_factor_opts` dispatch with
   threshold.** Requires writing a CSC → linked-list result into the
   caller-owned `SparseMatrix` without disturbing `factor_norm`,
   `reorder_perm`, and the row/col permutation arrays.
4. **Larger SuiteSparse corpus.** The default fixtures cap at n =
   132.  With n ≥ 1000 matrices, the scalar CSC kernel's speedup
   should grow (pointer-chasing overhead is proportionally larger)
   and the supernodal batched path would show visible scaling.

## Lessons for future "linked-list → compressed" refactors

1. **Name the pattern match early.** Sprint 17's plan explicitly
   referenced Sprint 10's CSR LU design block.  Having a concrete
   prior example with the same shape meant Days 1-5 were translation
   work, not original design.  Future compressed-format work
   (symbolic QR? compressed SVD block ops?) should do the same.
2. **Ship detection before replacing kernels.** Day 10 landed
   `chol_csc_detect_supernodes` as a standalone capability, verified
   independently on canonical structures.  Day 11 could then dispatch
   on top of it without re-proving detection correctness.  Don't
   bundle "new structure + new kernel" — that couples two kinds of
   risk.
3. **A correctness-first wrapper is an honest deliverable.** Day 8's
   LDL^T wrapper is slower than a native kernel would be, but it's
   numerically identical to the reference and lets every downstream
   test pass.  Future sprints can replace the wrapper's internals
   transparently.  Shipping a wrapper is strictly better than
   shipping a half-implemented native kernel with known bugs.
4. **Per-column CSC slot resizing via `memmove` is the right
   default.** Day 5's `shift_columns_right_of` handles both growth
   (fill-in) and shrink (drop tolerance) by moving the tail columns
   and bumping `col_ptr`.  `chol_csc_grow` handles capacity.
   Together this was enough for every test matrix; no more complex
   per-column slot reservation was needed.
5. **Benchmark-driven documentation.** The Day 12 benchmark numbers
   drove the README perf section, the algorithm.md performance table,
   and the `SPARSE_CSC_THRESHOLD` default (100 was validated by
   nos4 crossing 1× at that size).  Numbers in prose unbacked by a
   reproducible benchmark are aspirations, not claims — and numbers
   measured with mismatched timing boundaries are worse than no
   numbers, because they bake a methodology bug into shipping docs
   (see "Fair-comparison methodology surfaced" above).

## Final regression status

- `make clean && make` — builds clean.
- `make format` — clean across all source and tests.
- `make lint` — clang-tidy + cppcheck clean (3 intentional NOLINT markers in the LDL^T solve's 2×2-block branch, each with a justifying comment citing the `ldlt_csc_validate` invariant).
- `make test` — **1384 / 1384 pass across 41 suites**, 1 094 595 total assertions.
- `make sanitize` (UBSan) — full suite clean.
- `make bench` — one-shot fair-comparison (AMD included on both paths): Cholesky CSC scalar 1.13–1.65×, supernodal 1.22–2.01× over linked-list; LDL^T CSC wrapper 0.27× (nos4) / 1.33× (bcsstk04) baseline.

## Files changed / added

```
 modified:  Makefile
 modified:  CMakeLists.txt
 modified:  README.md
 modified:  include/sparse_cholesky.h
 modified:  include/sparse_lu_csr.h            (cross-link comment)
 modified:  include/sparse_matrix.h            (SPARSE_CSC_THRESHOLD)
 modified:  docs/algorithm.md                  (3 new sections)
 modified:  docs/planning/EPIC_2/PROJECT_PLAN.md
 added:     docs/planning/EPIC_2/SPRINT_17/PLAN.md
 added:     docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md
 added:     docs/planning/EPIC_2/SPRINT_17/RETROSPECTIVE.md
 added:     src/sparse_chol_csc_internal.h
 added:     src/sparse_chol_csc.c
 added:     src/sparse_ldlt_csc_internal.h
 added:     src/sparse_ldlt_csc.c
 added:     tests/test_chol_csc.c
 added:     tests/test_ldlt_csc.c
 added:     benchmarks/bench_chol_csc.c
 added:     benchmarks/bench_ldlt_csc.c
```
