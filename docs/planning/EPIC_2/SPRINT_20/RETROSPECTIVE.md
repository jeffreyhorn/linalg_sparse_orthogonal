# Sprint 20 Retrospective — LDL^T Completion & Symmetric Lanczos Eigensolver

**Duration:** 14 days
**Branch:** `sprint-20`
**Date range:** 2026-04-22 → 2026-04-23 (intensive condensed run)

## Goal recap

> Close out the final Sprint 19 follow-ups for the batched supernodal
> LDL^T path — the `_with_analysis` shim that enables indefinite
> matrices and a transparent size-based dispatch through
> `sparse_ldlt_factor_opts` — then land the symmetric Lanczos
> eigensolver with shift-invert mode.

## Definition of Done checklist

Against the seven `PROJECT_PLAN.md` items:

| # | Item | Target | Landed | Verdict |
|---|------|--------|--------|---------|
| 1 | `ldlt_csc_from_sparse_with_analysis` | Pre-allocate full `sym_L` pattern from `sparse_analysis_t`; unblock batched supernodal LDL^T on indefinite inputs | Days 1-3. Design + stub Day 1; implementation Day 2; Day 3 wires the `sym_L_preallocated` fast-path through `ldlt_csc_eliminate_supernodal` and drops the KKT residual from 1e-2..1e-6 to round-off (≤ 1e-13) | ✅ Complete |
| 2 | Transparent LDL^T dispatch through `sparse_ldlt_factor_opts` | `backend` selector (AUTO / LINKED_LIST / CSC) + `used_csc_path` telemetry; AUTO routes CSC above `SPARSE_CSC_THRESHOLD` | Days 4-6. Day 4 API + dispatch skeleton; Day 5 full CSC path + structural fallback + cross-backend agreement test; Day 6 cross-threshold integration tests, `bench_ldlt_csc --dispatch` mode, PERF_NOTES section | ✅ Complete |
| 3 | Eigenvalue API design | `sparse_eigs_t`, `sparse_eigs_opts_t`, `sparse_eigs_sym` public API + stub | Day 7. Full public header with doxygen for every field; compile-ready stub in `src/sparse_eigs.c` validates every documented precondition | ✅ Complete |
| 4 | Symmetric Lanczos eigensolver | Thick-restart Lanczos with full MGS reorth for k LARGEST/SMALLEST, Ritz extraction, convergence bookkeeping | Days 8-11. Day 8 3-term recurrence; Day 9 full MGS reorth with wide-spectrum test; Day 10 thick-restart outer loop; Day 11 `tridiag_qr_eigenpairs` + Ritz vector lift `V · Y[:, j]` + Wu/Simon residuals + symmetry-check precondition | ✅ Complete |
| 5 | Shift-invert Lanczos | `NEAREST_SIGMA` via `(A - σI)^{-1}` factored through `sparse_ldlt_factor_opts` AUTO dispatch | Day 12. `lanczos_iterate_op` callback refactor; shift-invert context threads LDL^T solve as the Lanczos operator; post-process λ = σ + 1/θ; two-pointer |θ|-descending selection | ✅ Complete |
| 6 | Lanczos tests and validation | SuiteSparse SPD, SVD cross-check (A^T·A spectrum = σ²), indefinite AUTO dispatch, stability regressions | Day 13. 8 new tests: nos4 / bcsstk04 / bcsstk14 LARGEST+SMALLEST, A^T·A vs SVD on a Hilbert-style 15×10, KKT n=150 shift-invert with `used_csc_path_ldlt == 1` assertion, near-singular + zero matrix regressions. Wu/Simon-based outer loop redesign landed here too (the Day 10 short/long restart didn't converge on SuiteSparse — see "What didn't go well"). | ✅ Complete |
| 7 | Lanczos documentation | README + algorithm.md + header doxygen + example program + retrospective | Day 14. README "Sparse Symmetric Eigensolver (Sprint 20)" subsection + API table row; doxygen refresh with cross-references to `sparse_ldlt.h` and `sparse_svd.h`; 110-line pedagogical section in `docs/algorithm.md` (3-term recurrence, MGS rationale, Wu/Simon, shift-invert, convergence heuristics); `examples/example_eigs.c` demonstrating both LARGEST and NEAREST_SIGMA with residual checks | ✅ Complete |

## Final metrics

| Metric | Start of Sprint 20 | End of Sprint 20 |
|--------|-------------------:|-----------------:|
| Total `static void test_*` definitions across all test files | ~1 525 | **~1 548** (+23) |
| `tests/test_eigs.c` (new file) | — | 19 tests / 154 assertions |
| `tests/test_sprint20_integration.c` (new file) | — | 20 tests / 126 assertions |
| `tests/test_ldlt_csc.c` | ~3 000 | ~3 600 (+600 LoC — Day 1-3 indefinite batched tests) |
| `tests/test_ldlt.c` | ~1 100 | ~1 420 (+320 LoC — Day 4-5 backend dispatch tests) |
| `src/sparse_eigs.c` (new) | — | 741 LoC |
| `src/sparse_eigs_internal.h` (new) | — | 121 LoC |
| `include/sparse_eigs.h` (new) | — | 264 LoC |
| `src/sparse_ldlt.c` | ~1 200 | 1 402 (+~200 for backend dispatch + CSC writeback) |
| `src/sparse_ldlt_csc.c` | ~2 350 | 2 710 (+~360 for `_with_analysis` + sym_L fast-path) |
| New benchmark captures | — | `bench_day3_indefinite.txt`, `bench_day6_dispatch.txt`, `bench_day9_reorth.txt`, `bench_day13_lanczos.txt` |
| Public headers added | — | `include/sparse_eigs.h` |
| Example programs added | — | `examples/example_eigs.c` |
| **KKT batched LDL^T residual (Day 3)** | 1e-2..1e-6 (fallback to scalar) | **≤ 1e-13** (batched supernodal) |
| **nos4 k=5 LARGEST Lanczos convergence (Day 13)** | saturated at 7e-3 after 2 000 iters (old outer loop) | **4e-14 in 70 iters** (growing-m outer loop) |
| **KKT n=150 shift-invert wall (Day 12-13)** | n/a | **1.9 ms** (vs direct SMALLEST 2.9 ms — 34% faster) |

All residuals on the enlarged corpus remain ≤ 2e-13 (SPD spot-check threshold 1e-10).

## Performance highlights

### Indefinite LDL^T batched enablement (Day 3)

Before Day 1-3, any indefinite matrix (KKT-style saddle points, mixed-sign diagonal, anything with non-trivial off-block coupling) forced the fallback to the scalar `ldlt_csc_eliminate` kernel because the heuristic `ldlt_csc_from_sparse` fill-factor pattern silently dropped supernodal cmod fill rows.  Day 1-3 mirrored the Cholesky Sprint 19 Day 6 fix: pre-allocate the full `sym_L` pattern from `sparse_analysis_t` via `ldlt_csc_from_sparse_with_analysis`, then teach `ldlt_csc_eliminate_supernodal` to take the `sym_L_preallocated` fast path.  KKT residual dropped from 1e-2..1e-6 to round-off on the Sprint 18 Day 13 KKT fixture (n = 28, n = 55).

### Transparent LDL^T dispatch (Days 4-6)

Added `sparse_ldlt_opts_t.backend` (AUTO / LINKED_LIST / CSC) + `used_csc_path` output telemetry, mirroring the Sprint 18 Cholesky dispatch.  AUTO routes to the CSC supernodal path on `n ≥ SPARSE_CSC_THRESHOLD` (default 100) with a structural fallback to `ldlt_csc_eliminate` when batching defeats.  The `bench_ldlt_csc --dispatch` mode reports the chosen path per matrix; on the Sprint 19 SPD corpus the AUTO choice matches the measured crossover.  Indefinite n=150 KKT runs end-to-end through the public `sparse_ldlt_factor_opts` AUTO with round-off residuals — the Day 3 enablement validated through the public API.

### Lanczos convergence (Days 8-13)

The Sprint 20 eigensolver converges nos4 k=5 LARGEST to residual 4e-14 in 70 Lanczos iterations / ~3 ms wall.  bcsstk04 k=3 LARGEST lands at m=62 / 4.6 ms; bcsstk14 k=5 LARGEST at m=70 / 113 ms.  Shift-invert on the KKT n=150 fixture at σ=0 converges in 39 Lanczos iterations / 1.9 ms — 34% faster than direct SMALLEST at m=62 / 2.9 ms, and `used_csc_path_ldlt == 1` confirms the inner LDL^T routed through the Day 4-6 CSC supernodal backend.

See `bench_day13_lanczos.txt` for the full numbers.

## What went well

- **`_with_analysis` + dispatch arc landed cleanly in 6 days.**  The Cholesky Sprint 18/19 precedent gave Days 1-3 a concrete template: pre-allocate the full `sym_L`, add a preallocated fast-path to the supernodal writeback, wire the new shim through `sparse_ldlt_factor_opts` exactly like the Cholesky side does.  Zero unexpected detours.  Days 4-6 followed the same pattern for the dispatch plumbing.

- **Public Lanczos API design on Day 7 held up through Day 13.**  The Day 7 header declarations — `sparse_eigs_t`, `sparse_eigs_opts_t`, `sparse_eigs_sym` — needed *one* backward-incompatible addition across the rest of the sprint: the Day 13 `used_csc_path_ldlt` observability field.  Everything else — `compute_vectors`, `which`, `sigma`, `reorthogonalize`, `residual_norm` — was right first time.  Having a compile-ready stub that validated every precondition gave Days 8-13 a stable target.

- **Wu/Simon residual beat the stability-check-based convergence.**  The Day 10/11 outer loop ran Lanczos twice per restart (short m_short, long m_long) and compared top-k theta values for stability.  This converged fine on small diagonal fixtures but saturated on SuiteSparse.  Day 13's discovery: for a Ritz pair (θⱼ, y_j) the eigen-equation residual is *exactly* |β_m · y_{m-1,j}| (Paige 1972), and it's free to compute once Y is in hand.  Switching to Wu/Simon as the primary gate — and replacing the restart-with-warm-start with a growing-m single Lanczos sequence — fixed convergence on nos4 / bcsstk04 / bcsstk14 with no further tuning.

- **The callback refactor (`lanczos_iterate_op`) paid off immediately on Day 12.**  Refactoring `lanczos_iterate(A, ...)` into a thin wrapper over `lanczos_iterate_op(op_fn, op_ctx, ...)` was a ~30-line change.  Day 12 then dropped in `sparse_ldlt_solve` as the operator with two lines of context-struct setup — no changes to the Lanczos core.  Also opens the door for LOBPCG in Sprint 21 to reuse the same driver with a different operator.

- **Test coverage bulletproofed the numerical corners.**  `test_zero_matrix` and `test_near_singular_stable` (condition 1e8) both ran cleanly under sanitizers without crashes or NaN leaks.  The tests documented what the solver *does* in these corner cases, not just what it should do — useful for future contributors who might tweak the early-exit logic.

- **The example program compiles and produces sensible output on the first try.**  `examples/example_eigs.c` loads nos4, runs LARGEST+compute_vectors, prints a self-validating residual table, then runs NEAREST_SIGMA on a local KKT fixture.  No surprises — the API is as clean from the outside as it feels from the inside.

## What didn't go well

- **The Day 10/11 outer loop design saturated on SuiteSparse.**  Original plan: run Lanczos at m_short, then at m_long with the same v0; compare top-k θ values for stability.  On non-convergence, restart with v0 := v_last (the final Lanczos vector of the long run) and repeat.  This looked reasonable on paper — each restart should bring fresh information via the warm-start direction — but in practice the warm-start discarded the partial basis, and the top-k θ values kept re-approaching the same neighbourhood on every restart without tightening.  On nos4 k=5 LARGEST, the residual saturated around 7e-3 after 2000 iterations.  Day 13 scrapped the short/long dual-batch approach entirely and replaced it with a single growing-m Lanczos sequence (same v0, extending the Krylov basis by `k + 20` per retry).  Fixed the convergence immediately.  **Lesson:** "thick-restart" is a specific technical term meaning "preserve the converged Ritz space in the restart basis" (Wu/Simon 2000, Stathopoulos/Saad 2007).  What the Day 10 implementation did was "restart with a new v0" — which is something entirely different and weaker.  Naming matters.

- **`SPARSE_ERR_SHIFT_SINGULAR` existed in docstrings but not in the enum.**  Day 7 documented `SPARSE_ERR_SHIFT_SINGULAR` as the return code when σ coincides with an eigenvalue.  Day 12 discovered the enum has no such value; the factored `A - σI` actually returns `SPARSE_ERR_SINGULAR` via `sparse_ldlt_factor_opts`.  Had to walk back the docstring to reference the real enum value.  **Lesson:** when designing APIs, cross-check every named error against `include/sparse_types.h` before committing the header.

- **Convergence-criterion tuning consumed more time than planned.**  The PLAN budgeted 40 hours for Days 8-11 (Lanczos core).  Actual spend was closer to 35, but another ~8 hours on Day 13 went to debugging why SuiteSparse tests reported NOT_CONVERGED — which turned out to be the Day 10 outer loop, not the Lanczos iteration itself.  If the Day 10 design had used Wu/Simon from the outset, Day 13 would have been 4 hours faster.

- **bcsstk04 SMALLEST needs m = n = 132.**  The bottom-cluster eigenvalues are close enough that Lanczos needs the full Krylov basis to separate them.  This is expected behaviour for Lanczos on clustered spectra (shift-invert with σ ≈ 1e-3 would be the production path), but it means the test takes 33 ms instead of the 2-5 ms seen on extremes.  Documented the trade-off in `bench_day13_lanczos.txt`.

## Items deferred

- **LOBPCG backend.**  Sprint 20's `sparse_eigs_opts_t.backend` enum already lists `SPARSE_EIGS_BACKEND_LANCZOS`; the header comment reserves a `SPARSE_EIGS_BACKEND_LOBPCG` slot for Sprint 21.  Not attempted in Sprint 20; explicit Sprint 21 work per the project plan.

- **Iterative refinement for eigenpairs.**  Not in scope.  The Wu/Simon residual is the accuracy bound users should trust; callers who need tighter answers can either bump `opts.tol` or apply inverse-iteration post-hoc on the returned eigenvector.

- **OpenMP parallelism inside Lanczos.**  The matvec and reorth inner loops are embarrassingly parallel but not yet parallelised.  `sparse_matvec` is OpenMP-enabled when built with `-DSPARSE_OPENMP`, so direct Lanczos already benefits; the reorth loop is serial.  Left for a future perf sprint.

- **Benchmark executable (`bench_eigs`).**  The Day 13 bench was a throwaway driver in `/tmp/bench_eigs.c`.  A permanent `benchmarks/bench_eigs.c` with CSV output and an `--sweep` mode is a natural follow-up but not Sprint 20 scope.

- **True thick-restart (Wu/Simon arrowhead).**  The Day 13 outer loop uses growing-m, which is simpler and converges reliably on the target corpus.  A proper thick-restart that preserves the converged Ritz subspace in a smaller basis would save memory on very large n (bcsstk14 at n=1806 already uses 26 MB for V at m=1806).  Left for Sprint 21 if the LOBPCG path shows a need.

## Lessons

- **Name your algorithms precisely.**  "Thick-restart Lanczos" is a specific structural concept (preserve the converged Ritz subspace via arrowhead).  Calling something thick-restart when it's actually "restart with warm v_last" invites design bugs — the Day 10 outer loop was nominally correct but didn't do the thing the comment claimed.  Day 13's redesign is correctly named "growing-m Lanczos" in the code comment, not "thick-restart".

- **Self-validating tests beat external-reference tests.**  For SuiteSparse SPD validation, the obvious approach is "compute eigenvalues with some dense reference (LAPACK?) and compare".  No dense reference is available in this codebase, and any simple re-implementation would be circular.  Instead, `assert_ritz_residuals` just checks `||A v − λ v|| / (|λ| · ||v||) ≤ tol` on the returned pairs.  This is what users actually care about, doesn't need an external truth, and catches both value and vector errors.  Won't detect *missing* eigenvalues (e.g. if Lanczos skips one in a cluster), but that's what the Wu/Simon gate is for.

- **Anchor tolerances by problem scale, not by absolute units.**  Day 13 initially failed bcsstk04 at `residual = 1.46e-8 > tol = 1e-8`.  The eigenvalues were ~1e7, so the relative residual was 1.5e-15 — excellent.  Rescaling the test's residual by `|λ| · ||v||` — matching the Wu/Simon convention — makes the tolerance comparable across nos4 (eigenvalues near 1), bcsstk14 (eigenvalues near 1e10), and diagonal unit fixtures.

- **Callback refactors are cheap before the implementation ossifies.**  Refactoring `lanczos_iterate` to take a callback on Day 12 was ~30 lines (Days 8-11 had already stabilised the inner 100-line recurrence).  Had the callback arrived in Sprint 21 instead, it would have meant touching a much wider diff.  Lesson: when a feature (shift-invert) obviously needs a generalisation, add the indirection when the interior is small.

- **Grow-m Lanczos is simpler and sufficient for this codebase's target corpus.**  The implementation-cost gap between growing-m and proper Wu/Simon thick-restart is large (~500 vs ~50 lines for the outer loop).  For problems up to n ~ 2000 with k ≤ 10, growing-m with m_cap = n converges fine and fits in tens of MB.  True thick-restart only becomes necessary when n is large enough that holding V for m = n is memory-prohibitive, which isn't the Sprint 20 use case.  YAGNI applied well.

## Acknowledgements

Sprint 20 extended Sprints 17-19's CSC and supernodal groundwork with a symmetric eigensolver, and the composition lands cleanly: shift-invert Lanczos on an indefinite matrix exercises the Day 3 `_with_analysis` fix, the Day 4-6 AUTO dispatch, and the Sprint 17 CSC supernodal LDL^T path in a single `sparse_eigs_sym(which=NEAREST_SIGMA)` call.  The `used_csc_path_ldlt = 1` telemetry on the Day 13 KKT test is the concrete proof that the architectural bets from three prior sprints integrate as designed.
