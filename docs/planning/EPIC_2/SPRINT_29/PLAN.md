# Sprint 29 Plan: SVD Improvements, Eigenpair Refinement, Progress Callbacks, CI Hardening & Epic 2 Wrap-Up

**Sprint Duration:** 14 days
**Goal:** Address remaining review findings and the final Sprint 20 deferred follow-up: fix the dense-in-disguise SVD paths, add an opt-in inverse-iteration refinement post-pass for `sparse_eigs_sym` eigenpairs (deferred from Sprint 20), add progress/cancel callbacks for long-running routines, add Windows/macOS CI, improve the sparse low-rank approximation, calibrate the long-running coverage gate + fix the build-and-test bench-step CI hang (both inherited as pre-existing failures from PRs #28-32), absorb two Sprint-28 deferrals (`bench_reorder --reorder-via-analyze` flag + macOS-15+ tsan handling), and close out Epic 2 with final documentation and validation.  Items routed from `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 section (lines 757-802) + `SPRINT_28/RETROSPECTIVE.md` Items-deferred #3 + "What didn't go well" tsan-on-macOS entry.

**Starting Point:** Sprint 28 (PR #36, merged at `34734bf`) shipped: three new advisory env vars (`SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`, `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES` selector list, `SPARSE_SUPERNODAL_POSTORDER={off,on}` — canonical name post-PR-#36-review with legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias for back-compat); zero production default flips.  Sprint 28 Day-12 24-setting × 6-fixture cross-corpus matrix CONFIRMED Sprint 27 default + Sprint 28 Item-4 `SPARSE_SUPERNODAL_POSTORDER=on` tied for Pres_Poisson best at 0.9226× (bit-equal by symmetric-permutation invariance).  **The literal 0.85× Pres_Poisson target was FORMALLY RETIRED with Sprint 28's empirical evidence** (6 consecutive sprints + non-pipeline pivot demonstrating the floor is structural; `non_pipeline_decision.md` "Literal 0.85× Target — Formal Retirement").  Sprint 28 inheritance: cross-platform `tf_setenv` / `tf_unsetenv` macros in `tests/test_framework.h` (per PR-#36 review); `_POSIX_C_SOURCE 200809L` at the top of env-mutating test files; canonical `SPARSE_SUPERNODAL_POSTORDER` env var name with `SPARSE_ND_SUPERNODAL_POSTORDER` legacy alias; CI tsan job hangs on macOS 15.7 in `__tsan::CheckAndProtect → get_dyld_hdr` (deferred to Sprint 29 Item 10b).  Headline gates: `make wall-check` PASS (Pres_Poisson ND ~5 s vs 47 s baseline 1.5× ceiling = 70.5 s); `make sanitize` CLEAN; `make tsan` blocked on macOS 15.7 platform issue (Sprint 27 inheritance); `make lint` EXIT 0 (109 NOLINT); `make test` 2068 assertions PASS.

**End State:** `sparse_svd_lowrank_sparse()` rewritten to use outer-product accumulation directly into sparse output (no m×n dense intermediate; documented memory + wall-time wins).  `sparse_svd_full_opts.full_u_v` option lights up non-economy U and V^T output.  `sparse_eigs_sym_opts.refine` (default off) + `sparse_eigs_sym_opts.refine_max_iters` add an opt-in inverse-iteration refinement post-pass that reuses the Sprint 20 shift-invert path (`sparse_ldlt_factor_opts` AUTO dispatch) at the converged Ritz value.  `sparse_progress_cb_t` callback type + `opts.progress_cb` / `opts.progress_user` fields land across LU / Cholesky / LDL^T / QR / CG / GMRES / MINRES / BiCGSTAB / Lanczos / LOBPCG / ND, with callback-return-value-driven cancellation.  GitHub Actions adds Windows (MSVC via CMake) + macOS (Apple Clang + Homebrew GCC) build matrix.  `sparse_get_err()` accessor variant lands (or the silent-zero-on-error contract is documented across the accessor headers — Day-10 decides).  `make coverage` calibrated against operating reality (either tightened to ≥ 95 % aggregate or `COV_THRESHOLD` lowered to a defensible target with documentation).  `make bench` CI step replaced with `bench-fast` (`--skip-factor`) or moved to a separate nightly workflow that doesn't gate PR merges.  `bench_reorder --reorder-via-analyze` flag lets the standard bench harness exercise Sprint-28-era env vars (Sprint-28 Item-10a absorption).  `make tsan` gets macOS-15+ handling (Linux-CI tsan job OR macOS-version-gated target emitting a non-zero-exit "blocked on macOS 15+; routing to inherited validation" message; Sprint-28 Item-10b absorption).  `README.md` updated with all Epic-2 APIs (LDL^T, IC, MINRES, BiCGSTAB, eigensolvers, eigenpair refinement, COLAMD, ND, progress callbacks).  `INSTALL.md` updated for new CI platforms.  `docs/planning/EPIC_2/SPRINT_29/RETROSPECTIVE.md` + `EPIC_2_RETROSPECTIVE.md` filled in single-pass.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to 168 hours — exactly the 14×12 ceiling.  Items 1-9 from PROJECT_PLAN.md sum to 168 hrs nominally; Item 10's 6-hr Sprint-28-deferral absorption rides within Items 5/6/8 (per the PROJECT_PLAN.md Item-10 routing: Item-10a `bench_reorder` flag folds into Item-8 CI bench-step work; Item-10b macOS-15+ tsan handling rides on Item-5/6 CI matrix work).  Item 9's 24-hr PROJECT_PLAN.md estimate lands as 20 hrs of allocated time across Days 13-14 — the README + INSTALL + retrospective work fits in 20 hrs with the remaining 4 hrs absorbed by retrospective scaffolding that starts incrementally as each item closes during Days 2-12 (each item-close day spends ~30 min updating the in-flight retro skeleton).  Risk concentration is Item 8 (28 hrs across Days 11-13): final integration + coverage-gate + bench-step are inherited red-since-PR-#28 problems with no fixed scope; if Day-13's CI bench-step work over-runs, Day 14's retrospective absorbs the spillover.

---

## Day 1: Item 1 — Sparse Low-Rank Without Dense Accumulator (Design + Skeleton)

**Theme:** Open the sprint with Item 1's sparse low-rank refactor.  `sparse_svd_lowrank_sparse()` currently builds the rank-k approximation through an m×n dense intermediate, which is O(m·n) memory + O(m·n·k) wall — for the Pres_Poisson-class corpus (m = n = 14 822, typical k = 50) that's ~1.7 GB intermediate, dominating both memory and compute over the actual rank-k product.  Day 1 maps the existing call path + outer-product algebra + designs the sparse accumulator data structure.

**Time estimate:** 12 hours

### Tasks
1. Re-read `src/sparse_svd.c::sparse_svd_lowrank_sparse()` + the upstream `sparse_svd_thin()` Lanczos-bidiag path; map where the m×n dense intermediate is allocated + how it's consumed.  Walk `bench_svd.c`'s low-rank micro-benchmark numbers (Sprint 17 / 18 captures in `SPRINT_17/PERF_NOTES.md` / `SPRINT_18/PERF_NOTES.md`) to establish the Sprint-29 baseline.
2. Design the outer-product accumulator: each Lanczos pair (σ_i, u_i, v_i) contributes a rank-1 outer product `σ_i · u_i · v_i^T` summed into the running result.  Sparse output target: a `SparseMatrix *` with linked-list insertion (the same convention `sparse_create` + `sparse_insert` already follows).  Per-iteration cost: O(nnz(u_i) · nnz(v_i)) per outer product; total cost: O(k · m_nz · n_nz) which beats O(m·n·k) when m_nz · n_nz < m · n (i.e. when u_i / v_i are sparse OR when we threshold by σ_i magnitude).
3. Decide on the thresholding strategy: a per-pair `tol` cutoff (drop entries below `σ_i · |u_ij| · |v_ik| < tol`) versus a top-K-per-row pruning.  Pick one; document the rejection rationale for the other in the design doc.
4. Draft `tests/test_svd.c::test_sparse_svd_lowrank_outer_product_matches_dense` as failing-as-expected: build a small synthetic SPD with known low-rank structure (e.g. n=32, rank=4), call both the old dense-intermediate path (if still callable) and the new outer-product path, assert ||A_dense - A_sparse||_F / ||A_dense||_F ≤ 1e-10.  RUN_TEST commented out until Day 2.
5. Stub `src/sparse_svd.c::sparse_svd_lowrank_outer_product()` (new internal helper) with the signature + a no-op body; default callers still route through the dense-intermediate path.  Add env-var gate `SPARSE_SVD_LOWRANK_OUTER={off (default), on}` to expose the new path during Day-2 validation.
6. Write `docs/planning/EPIC_2/SPRINT_29/lowrank_design_day1.md` (parallel to Sprint 28's `pivot_decision_day1.md`): existing path's O(m·n) intermediate cost + corpus measurements; new path's outer-product algebra + thresholding choice; expected wall + memory wins; LOC estimate.
7. Run `make format && make lint && make test` (no functional change Day 1; sanity check the workspace + the new test stub compiles).

### Deliverables
- `docs/planning/EPIC_2/SPRINT_29/lowrank_design_day1.md` design doc with algebra + thresholding choice + LOC estimate
- `src/sparse_svd.c` skeleton for `sparse_svd_lowrank_outer_product()` (default-off env-var gate)
- `tests/test_svd.c::test_sparse_svd_lowrank_outer_product_matches_dense` failing-as-expected stub (RUN_TEST commented out)
- All quality checks clean

### Completion Criteria
- Design doc names the thresholding strategy + documents the rejection rationale for the alternative
- Skeleton compiles; env-var gate `SPARSE_SVD_LOWRANK_OUTER=off` produces bit-identical existing behaviour
- Stubbed test compiles + trips the expected assertion when RUN_TEST is uncommented (manual dry-run)
- `make format && make lint && make test` clean

---

## Day 2: Item 1 Close — Outer-Product Implementation + Bench Validation

**Theme:** Implement the outer-product accumulator per Day-1's design; validate against the dense baseline on a small SPD then sweep the SuiteSparse corpus.  Goal: produce ≥ 50 % memory + ≥ 30 % wall reduction on the n ≥ 5000 fixtures vs the dense-intermediate path; document fixture-by-fixture deltas in `lowrank_sweep_day2.txt`.

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_svd_lowrank_outer_product()` in `src/sparse_svd.c`: take the Lanczos-bidiag pairs (σ_i, u_i, v_i) from the existing path, fold each rank-1 outer product into the sparse output via the chosen Day-1 thresholding strategy.
2. Light up `test_sparse_svd_lowrank_outer_product_matches_dense` (Day 1 stub) under both `SPARSE_SVD_LOWRANK_OUTER=off` (existing path) + `=on` (new path); assert relative-Frobenius residual ≤ 1e-10 on the n=32 synthetic.
3. Add corpus-safety test `test_sparse_svd_lowrank_outer_product_corpus_safety`: load nos4 / bcsstk04 / bcsstk14 (small + mid), run both paths at k ∈ {10, 50}, assert ||A_off - A_on||_F / ||A_off||_F ≤ 1e-10 on every fixture × k combination.
4. Extend `bench_svd.c` (or write a one-off `/tmp/bench_lowrank_day2.c`) to measure wall + peak-memory under both paths for the corpus.  Capture to `docs/planning/EPIC_2/SPRINT_29/lowrank_sweep_day2.txt`.
5. Decide the default flip: if the sweep clears ≥ 50 % memory + ≥ 30 % wall reduction without corpus regression past 5 %, flip `SPARSE_SVD_LOWRANK_OUTER` default to `on` in `parse_svd_lowrank_outer()` + update `docs/algorithm.md` SVD subsection.  Otherwise ship as advisory.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_svd.c::sparse_svd_lowrank_outer_product()` lit up + bit-validated against the dense path
- `tests/test_svd.c` 2 new tests passing under both env settings
- `docs/planning/EPIC_2/SPRINT_29/lowrank_sweep_day2.txt` corpus bench (wall + peak-memory)
- Production default flip OR advisory documentation per the sweep verdict
- `docs/algorithm.md` SVD subsection updated with the new env var + per-fixture deltas (if flipped)
- All quality checks clean

### Completion Criteria
- Outer-product path matches dense path to ≤ 1e-10 Frobenius residual on synthetic + 3 corpus fixtures
- Corpus sweep captured with real numeric values (no `?` placeholders per Sprint 27 Day-13 lesson)
- Default-flip decision recorded in `lowrank_sweep_day2.txt` with bench evidence
- `make format && make lint && make test && make wall-check` clean

---

## Day 3: Item 2 — Full SVD U/V Output Beyond Economy Mode (Design + Implementation)

**Theme:** Extend `sparse_svd_full_opts` (or the equivalent options struct) to optionally output the complete (non-economy) U (m×m) and V^T (n×n).  Today's path only ships economy U (m×k) + V^T (k×n) where k = min(m, n).  Full output is needed for callers doing rank-deficient null-space analysis or full-spectrum reconstruction.

**Time estimate:** 12 hours

### Tasks
1. Read `src/sparse_svd.c` + `include/sparse_svd.h`: map the current economy-only output's data flow; identify where U / V^T are sized + populated; the new full-mode path needs to extend U with orthonormal columns (Gram-Schmidt against the existing economy columns) for the `m - k` missing rank entries (and similarly V^T's `n - k` rows for `m < n`).
2. Add an opts flag (`opts.full_u_v` boolean) + a return-size contract: when off, returns the existing economy shapes; when on, U is `m × m` and V^T is `n × n` (padded with orthonormal columns / rows that complete the basis).
3. Implement the full-mode path: after the economy SVD lands, run modified Gram-Schmidt against the unit vectors `e_{k+1}, ..., e_m` (deflating against the economy U columns) to fill the remaining U columns; symmetric for V^T.  Sprint 21's reorth helpers (`reorth_v_against_basis`) provide the building blocks.
4. Add `tests/test_svd.c::test_svd_full_u_v_orthonormality`: for a 16×8 random matrix, request full mode, assert U is 16×16 orthonormal (||U^T U - I||_F ≤ 1e-10) and V^T is 8×8 orthonormal.
5. Add `tests/test_svd.c::test_svd_full_u_v_reconstruction`: same matrix, assert `||A - U·Σ·V^T||_F` is within roundoff (≤ 1e-10) under both economy and full modes (full mode reconstruction must match economy mode's accuracy since the extra basis columns are orthogonal to A's range).
6. Run `make format && make lint && make test`.

### Deliverables
- `include/sparse_svd.h` API extension: `full_u_v` opts field with the size contract documented
- `src/sparse_svd.c` full-mode path landed (economy + Gram-Schmidt padding)
- 2 new tests in `tests/test_svd.c` covering orthonormality + reconstruction under full mode
- All quality checks clean (economy-mode path bit-identical when `full_u_v == false`)

### Completion Criteria
- Full-mode U is `m × m` orthonormal + V^T is `n × n` orthonormal (residuals ≤ 1e-10)
- Reconstruction residual matches economy-mode within roundoff
- `make format && make lint && make test` clean

---

## Day 4: Item 2 Close + Item 3 — Eigenpair Refinement Design Kickoff

**Theme:** Close Item 2 with corpus-safety tests + opts-doc updates (4 hrs); kick off Item 3's optional inverse-iteration refinement post-pass for `sparse_eigs_sym` (4 hrs design + 4 hrs design-doc draft).  Item 3 is deferred from Sprint 20 retrospective.

**Time estimate:** 12 hours (4 hrs Item 2 close + 8 hrs Item 3 design)

### Tasks
1. **Item 2 close (4 hrs):** Add `test_svd_full_u_v_economy_mode_unchanged` — assert economy-mode behaviour bit-identical to pre-Sprint-29 with `full_u_v=false`.  Update `include/sparse_svd.h` doc-comments with the full-mode size contract + a deprecation-free transition path (existing callers see no change because the opts field defaults to false).
2. **Item 3 design (8 hrs):** Re-read `docs/planning/EPIC_2/SPRINT_20/RETROSPECTIVE.md` "Items deferred" entry on eigenpair iterative refinement; map `src/sparse_eigs.c` shift-invert dispatch (`sparse_ldlt_factor_opts` AUTO path); identify where the converged Ritz values + vectors land in the public `sparse_eigs_t` result struct.
3. Design the refinement loop: for each converged `(λ_i, v_i)`, run inverse iteration with the converged Ritz value as the shift (reusing the same factored shift produced by the Sprint 20 path).  Convergence criterion: ||A·v_i - λ_i·v_i|| / ||v_i|| ≤ machine_eps OR `opts.refine_max_iters` hit.  Loop body: solve `(A - λ_i·I) y = v_i`; normalize `v_i' = y / ||y||`; update `λ_i' = v_i'^T A v_i'`.
4. Specify the API surface: `opts.refine` (boolean, default off for back-compat) + `opts.refine_max_iters` (int, default 5).  The Wu/Simon residual gate remains the production accuracy contract; this is opt-in for downstream callers (deflation pipelines, response-function evaluation) per the Sprint 20 retro's framing.
5. Stub `tests/test_eigs.c::test_eigs_refine_default_off_unchanged` + `test_eigs_refine_tightens_residual` (failing-as-expected — Day 5 lights them up).
6. Write `docs/planning/EPIC_2/SPRINT_29/refinement_design_day4.md`: design + API surface + integration points + Sprint-20 retrospective citation + back-compat contract.
7. Run `make format && make lint && make test`.

### Deliverables
- `tests/test_svd.c::test_svd_full_u_v_economy_mode_unchanged` passing (Item 2 corpus-safety close)
- `include/sparse_svd.h` doc-comment updates for the `full_u_v` opts field
- `docs/planning/EPIC_2/SPRINT_29/refinement_design_day4.md` Item 3 design doc
- `tests/test_eigs.c::test_eigs_refine_default_off_unchanged` + `test_eigs_refine_tightens_residual` failing-as-expected stubs (RUN_TEST commented out)
- All quality checks clean

### Completion Criteria
- Item 2's economy-mode-unchanged test passes (existing behaviour bit-identical with `full_u_v=false`)
- `refinement_design_day4.md` documents the API surface + back-compat contract + Sprint-20-retrospective citation
- Stubbed Day-5 tests compile + trip the expected failure message under RUN_TEST
- `make format && make lint && make test` clean

---

## Day 5: Item 3 Close — Eigenpair Refinement Implementation + Tests

**Theme:** Implement the inverse-iteration refinement loop per Day-4's design; light up the stubbed tests; validate on small synthetic + corpus.

**Time estimate:** 12 hours

### Tasks
1. Add `opts.refine` + `opts.refine_max_iters` to `sparse_eigs_sym_opts` in `include/sparse_eigs.h`.  Default off + default `max_iters=5`.
2. Implement the refinement loop in `src/sparse_eigs.c`: after Lanczos / LOBPCG converges, iterate the inverse-iteration step on each `(λ_i, v_i)` until residual ≤ machine_eps OR `max_iters` exhausted.  Reuse the Sprint-20 shift-invert factored matrix (the existing `sparse_ldlt_factor_opts` AUTO dispatch already builds a factored A - σI for the convergence path; pass that same factor into the refinement loop to avoid re-factoring).
3. Light up `test_eigs_refine_default_off_unchanged` (Day 4 stub): assert bit-identical eigenpairs with `opts.refine=false` vs Sprint 28 pre-merge.
4. Light up `test_eigs_refine_tightens_residual`: load a synthetic SPD with known clustered eigenvalues (close pairs that stress Wu/Simon's tolerance), run without refinement → assert residual ≤ `opts.tol`, run with refinement → assert residual ≤ 1e-13 (tighter than `opts.tol`).
5. Add `test_eigs_refine_lobpcg_backend` — same contract but routed through `SPARSE_EIGS_BACKEND_LOBPCG`; verify both backends share the refinement post-pass.
6. Add `test_eigs_refine_max_iters_budget` — assert the loop respects `opts.refine_max_iters=1` (single inverse-iteration step) and produces a partial-refinement result rather than over-iterating.
7. Run `make format && make lint && make test`.

### Deliverables
- `include/sparse_eigs.h` `opts.refine` + `opts.refine_max_iters` API extension
- `src/sparse_eigs.c` refinement loop landed; reuses Sprint-20 shift-invert factored matrix
- 4 new tests passing (default-off-unchanged, tightens-residual, lobpcg-backend, max-iters-budget)
- All quality checks clean

### Completion Criteria
- Eigenpair residuals under `opts.refine=true` ≤ 1e-13 on the clustered-eigenvalue synthetic (vs `opts.tol` ~1e-10 without refinement)
- Default-off path bit-identical to Sprint 28 (no production behaviour change without `opts.refine=true`)
- Both Lanczos + LOBPCG backends share the post-pass
- `make format && make lint && make test` clean

---

## Day 6: Item 4 — Progress/Cancel Callbacks (Design + First Routines)

**Theme:** Design the callback API + integrate into the first batch of long-running routines (LU, Cholesky, LDL^T).  Per PROJECT_PLAN.md: callback return value signals cancellation.

**Time estimate:** 12 hours

### Tasks
1. Design the callback type in `include/sparse_types.h`:
   ```c
   typedef struct {
       const char *phase;     /* e.g. "factor", "solve", "fm_pass" */
       idx_t step;            /* monotonic counter within phase */
       idx_t total;            /* expected total steps if known, else 0 */
       double elapsed_s;       /* wall time since phase start */
   } sparse_progress_t;

   typedef int (*sparse_progress_cb_t)(const sparse_progress_t *p, void *user);
   ```
   Callback returns 0 to continue, non-zero to cancel.  Caller-side: cancellation surfaces as `SPARSE_ERR_CANCELLED` (new error code).
2. Add `opts.progress_cb` + `opts.progress_user` fields to the existing options structs that drive long-running routines: `sparse_lu_factor_opts`, `sparse_cholesky_factor_opts`, `sparse_ldlt_factor_opts`, `sparse_iterative_opts` (CG / GMRES / MINRES / BiCGSTAB), `sparse_eigs_sym_opts`, `sparse_qr_factor_opts`.  Default `NULL` (no callback) preserves Sprint 28 behaviour.
3. Add `SPARSE_ERR_CANCELLED` to `include/sparse_types.h`'s error enum + update `sparse_strerror()`.
4. Integrate the first batch (LU + Cholesky + LDL^T factorization paths): emit progress events at meaningful boundaries (e.g. per column for scalar elimination, per supernode for supernodal paths); check the callback return after every emission; propagate `SPARSE_ERR_CANCELLED` cleanly up the call chain (free intermediate state, leave caller-visible output untouched per the existing failure-rollback contract).
5. Add `tests/test_integration.c::test_progress_cb_lu_emits + _cancel` — synthetic 100×100 LU; callback that counts emissions + asserts `>= n` calls; cancellation test asserts factor returns SPARSE_ERR_CANCELLED + leaves the input matrix unmodified.  Similar tests for Cholesky + LDL^T.
6. Run `make format && make lint && make test`.

### Deliverables
- `include/sparse_types.h` callback type + `SPARSE_ERR_CANCELLED` error code
- `opts.progress_cb` / `opts.progress_user` added to LU / Cholesky / LDL^T / iterative / eigs / QR opts structs
- LU / Cholesky / LDL^T factor paths emit progress + honour cancellation
- 6 new tests in `tests/test_integration.c` (LU / Cholesky / LDL^T × emits + cancel)
- All quality checks clean (default-NULL-callback path bit-identical to Sprint 28)

### Completion Criteria
- Callback emissions are monotonic + cover the entire factor phase on a 100×100 synthetic
- Cancellation cleanly returns SPARSE_ERR_CANCELLED + leaves the input matrix unmodified (no partial-factor corruption)
- Default-NULL-callback runs at zero overhead (`make wall-check` Pres_Poisson ND ≤ Sprint-28 baseline + noise margin)
- `make format && make lint && make test` clean

---

## Day 7: Item 4 Close + Item 5 Start (Windows CI)

**Theme:** Finish Item 4 across the remaining routines (QR, iterative solvers, Lanczos, LOBPCG, ND) (4 hrs); kick off Item 5's Windows CI investigation + initial GitHub Actions job (8 hrs).

**Time estimate:** 12 hours (4 hrs Item 4 close + 8 hrs Item 5 start)

### Tasks
1. **Item 4 close (4 hrs):** Integrate callbacks into QR factorization (`src/sparse_qr.c`), the four iterative solvers (CG / GMRES / MINRES / BiCGSTAB in `src/sparse_iterative.c`), Lanczos (`src/sparse_eigs.c`), LOBPCG (`src/sparse_eigs.c`), and nested-dissection (`src/sparse_reorder_nd.c`).  Per-routine progress + cancel semantics match Day-6's LU/Cholesky/LDL^T patterns (emit at meaningful boundaries; cancel returns SPARSE_ERR_CANCELLED + leaves input unmodified).
2. Add `test_progress_cb_iterative_solvers_emits_cancel` covering CG / GMRES / MINRES / BiCGSTAB on a 50×50 indefinite synthetic.  Add `test_progress_cb_lanczos_lobpcg_emits_cancel`.  Add `test_progress_cb_nd_emits_cancel` on Pres_Poisson (skip-load if not present per the existing test pattern).
3. **Item 5 start (8 hrs):**
   - Read `CMakeLists.txt` + `Makefile` parity audit from Sprint 11; identify any MSVC-incompatible patterns introduced in Sprints 12-28.  Known suspects: `_putenv_s` already wrapped in `tests/test_framework.h` per Sprint 28 PR-#36; `strtok_r` already replaced with portable tokenizer in `src/sparse_graph.c` per Sprint 28 PR-#36.  Sweep `src/` + `tests/` for any other POSIX-only API calls (`fork`, `dup2`, `mmap` — unlikely but check).
   - Draft `.github/workflows/windows-ci.yml` (parallels the existing Linux job in `.github/workflows/build.yml`): MSVC + CMake configure + build + ctest run.  Initial config: `windows-latest` runner; `-G "Visual Studio 17 2022"` generator; `cmake --build . --config Release`; `ctest -C Release --output-on-failure`.
   - Open a pre-CI scratch build locally if Windows access is available (or push a draft workflow + iterate on failures).
4. Run `make format && make lint && make test`.

### Deliverables
- Item 4 fully integrated across QR / 4 iterative solvers / Lanczos / LOBPCG / ND
- 3 new test sets covering the remaining routines (iterative-solvers / Lanczos+LOBPCG / ND)
- `.github/workflows/windows-ci.yml` draft committed (may be red on first push — Day 8 closes failures)
- All quality checks clean

### Completion Criteria
- All 11 listed routines emit progress + honour cancellation per the Day-6 pattern
- `make wall-check` Pres_Poisson ND ≤ Sprint-28 baseline + 5 % (default-NULL-callback overhead bounded)
- `.github/workflows/windows-ci.yml` lands; the first CI run may fail (red is acceptable Day 7; Day-8 closes)
- `make format && make lint && make test` clean

---

## Day 8: Item 5 Close (Windows CI) + Item 10b (macOS-15+ tsan)

**Theme:** Close Item 5 with a green Windows CI run (8 hrs); land the Sprint-28 Item-10b macOS-15+ tsan handling (4 hrs) per `SPRINT_28/RETROSPECTIVE.md` "What didn't go well" entry.

**Time estimate:** 12 hours (8 hrs Item 5 close + 4 hrs Item 10b)

### Tasks
1. **Item 5 close (8 hrs):** Iterate on `.github/workflows/windows-ci.yml` until green.  Expected blockers (in priority order):
   - `_POSIX_C_SOURCE` references in test files: confirm Sprint-28's `tf_setenv` / `tf_unsetenv` macros work on MSVC's `_putenv_s` path (PR-#36 verified the macro layer; verify the actual CI build).
   - clang-format flag differences: macOS clang-format-22 vs MSVC's older bundled version; address via `.clang-format` style adjustments or a relaxed `make format-check` for the Windows job.
   - Path-separator handling in `tests/data/suitesparse/*.mtx` loads: should already work (the existing `sparse_load_mm` path uses portable `fopen`), but verify.
   - Any `<unistd.h>` includes: MSVC doesn't ship it; replace with `<io.h>` or `<process.h>` guarded by `#ifdef _MSC_VER`.
2. Add `_POSIX_C_SOURCE` guards to any test files using `tf_setenv` / `tf_unsetenv` that Sprint 28 missed (Sprint 28 caught the 3 supernodal test files; verify across the full test suite).
3. **Item 10b macOS-15+ tsan (4 hrs):** Per PROJECT_PLAN.md Item 10(b): two viable options, pick one:
   - **Option (i) Linux-CI tsan job**: `.github/workflows/tsan-ci.yml` runs `make tsan` on `ubuntu-latest` (where TSan's dyld handling doesn't hit the macOS-15 bug).  This rides on the Item-5/6 CI matrix expansion.
   - **Option (ii) macOS-version-gated `make tsan`**: detect macOS 15+ at make-time, emit "tsan blocked on macOS 15+; routing to inherited validation per SPRINT_28 PR #36" + return non-zero exit so CI doesn't silently pass.
   - Pick option (i) — the Linux runner is free in GitHub Actions and runs tsan without the dyld-init hang.  Option (ii) routes the same intent through a less-discoverable error message.
4. Smoke-test the Linux tsan CI job by running `make tsan` on a Docker `ubuntu:22.04` container locally (or push + iterate); verify the full test suite passes under tsan.
5. Run `make format && make lint && make test`.

### Deliverables
- `.github/workflows/windows-ci.yml` green (Item 5 close)
- Any cross-platform portability fixes scattered across `src/` + `tests/` (likely small)
- `.github/workflows/tsan-ci.yml` green on Linux (Item 10b close)
- `docs/planning/EPIC_2/SPRINT_29/windows_ci_decision.md` brief notes documenting any non-obvious portability adjustments
- All quality checks clean

### Completion Criteria
- GitHub Actions Windows job is green on the sprint-29 branch's HEAD
- GitHub Actions tsan-on-Linux job is green on the sprint-29 branch's HEAD
- `make format && make lint && make test` clean on macOS (existing platform) + Linux (CI) + Windows (CI)

---

## Day 9: Item 6 — macOS CI Job (Apple Clang + Homebrew GCC)

**Theme:** Add the macOS GitHub Actions job; cover both Apple Clang (default `cc` on macOS-latest) and Homebrew GCC (`brew install gcc` + use `gcc-13` or similar).  Verify coverage + packaging scripts work on macOS per PROJECT_PLAN.md Item 6.

**Time estimate:** 12 hours

### Tasks
1. Draft `.github/workflows/macos-ci.yml`: matrix on `compiler ∈ {apple-clang, homebrew-gcc}` × `runner = macos-latest`; both run `make test && make wall-check`; the Apple Clang job also runs `make sanitize`.
2. The Homebrew GCC job needs `brew install gcc libomp` + setting `CC=gcc-13` (or whichever current version Homebrew ships).  Document the version pin choice in the workflow comments so future updates can be tracked.
3. Verify the make targets that touch coverage / packaging work on macOS:
   - `make coverage` (lcov tooling availability — Apple ships `xcrun llvm-cov`, may need `brew install lcov` for `genhtml` etc.)
   - `make install` + `pkg-config` flow per `sparse.pc.in`
4. Sweep `src/` for any `__APPLE__` / `__GLIBC__` portability gates introduced by Sprint 28 PR-#36 (`_POSIX_C_SOURCE` macros, etc.); confirm they work on Homebrew GCC too (which uses glibc-style features.h gates).
5. Iterate on `.github/workflows/macos-ci.yml` until green.  Expected blockers: OpenMP linkage on Apple Clang (`-Xpreprocessor -fopenmp -lomp`); `clock_gettime` already covered by `_POSIX_C_SOURCE`; any `__attribute__((cleanup))` or compiler-extension usage.
6. Run `make format && make lint && make test` locally.

### Deliverables
- `.github/workflows/macos-ci.yml` green on both Apple Clang + Homebrew GCC matrix entries
- Any cross-platform portability fixes for the Homebrew GCC path
- `docs/planning/EPIC_2/SPRINT_29/macos_ci_decision.md` brief notes on the OpenMP-linkage handling + version pins
- All quality checks clean

### Completion Criteria
- GitHub Actions macOS job is green on both compiler matrix entries on the sprint-29 branch's HEAD
- `make coverage` + `make install` confirmed working on macOS (manual local verification acceptable if CI doesn't run them)
- `make format && make lint && make test` clean across Linux + macOS + Windows

---

## Day 10: Item 7 — API Accessor Error Reporting

**Theme:** Per PROJECT_PLAN.md Item 7: add a `sparse_get_err()` variant that returns error codes alongside values, OR document the silent-zero-on-error contract explicitly in all accessor headers.  Pick one based on a quick design study.

**Time estimate:** 12 hours

### Tasks
1. Inventory the existing accessor API surface: `sparse_get()`, `sparse_get_value()` (if separate), `sparse_get_diag()`, anything else that returns a scalar value without an error code.  Map their current silent-zero-on-error / silent-undefined contract.
2. Design study: weigh **Option (a) new `sparse_get_err(matrix, i, j, &val)` returning `sparse_err_t`** vs **Option (b) document the silent-zero-on-error contract explicitly + add `sparse_get_last_error()` to the global accessor**.  Pick one based on:
   - **API surface impact**: option (a) duplicates every accessor function; option (b) adds one global function.
   - **Caller ergonomics**: option (a) is more verbose but unambiguous; option (b) is concise but requires a separate query call to disambiguate.
   - **Existing patterns**: review the wider library — does `sparse_norm()` etc. already use a return-error-code pattern?  If yes, option (a) aligns; if no, option (b) preserves the existing aesthetic.
3. Implement the chosen option in `include/sparse_matrix.h` (and adjacent accessor headers).
4. Add tests in `tests/test_sparse_matrix.c`:
   - For option (a): `test_sparse_get_err_returns_correct_error_on_out_of_range` + `test_sparse_get_err_returns_value_on_success`
   - For option (b): `test_silent_zero_contract_documented_in_header` (read the header docstring, assert the expected wording — brittle but cheap; OR test `sparse_get_last_error()` after a known-fail accessor call)
5. Update `include/sparse_matrix.h` + `include/sparse_lu.h` + `include/sparse_qr.h` etc. accessor docstrings to explicitly state the silent-zero contract (Option b) or the new `_err` variant (Option a).
6. Write `docs/planning/EPIC_2/SPRINT_29/accessor_error_decision.md` documenting the design choice + rejection rationale for the alternative.
7. Run `make format && make lint && make test`.

### Deliverables
- `include/sparse_matrix.h` (et al.) API extension or docstring updates per the chosen option
- 2 new tests covering the new behaviour
- `docs/planning/EPIC_2/SPRINT_29/accessor_error_decision.md` design doc
- All quality checks clean

### Completion Criteria
- Either `sparse_get_err()` works correctly OR all accessor headers explicitly document the silent-zero contract + `sparse_get_last_error()` is exposed
- New tests pass
- `make format && make lint && make test` clean

---

## Day 11: Item 8 — Final Integration Testing (Part 1)

**Theme:** Begin Item 8's 28-hour final-integration block: full regression under all sanitizers + cross-feature integration tests for new Sprint 11-28 features (focus: callback behaviour from Item 4 + eigenpair refinement from Item 3 + Sprint-28-era ND env vars + Sprint-29's new SVD paths).

**Time estimate:** 12 hours

### Tasks
1. Run `make clean && make test && make sanitize && make wall-check && make tsan` locally + the new Windows / macOS / Linux-tsan CI jobs (Days 7-9 lit them up).  Expected: all green except possibly minor flakes that need investigation.
2. Write cross-feature integration tests in `tests/test_sprint29_integration.c`:
   - **Progress callback × eigsolver × refinement**: Lanczos with `opts.progress_cb` set + `opts.refine=true` — verify callback fires across both phases (Lanczos iterations + refinement post-pass) + cancellation mid-refinement leaves the partial eigenpairs intact.
   - **Full-mode SVD × low-rank outer-product**: request full U/V via Day-3's API, then call `sparse_svd_lowrank_outer_product` on the result — verify the rank-k reconstruction matches.
   - **Sprint-28-era ND env var × Sprint-29 callbacks**: `SPARSE_SUPERNODAL_POSTORDER=on` + `sparse_reorder_nd` with `opts.progress_cb` — verify the supernodal postorder post-pass fires AFTER the multilevel partition's progress events.
3. Begin coverage analysis: run `make coverage` + inspect per-file coverage; tabulate which files are below the 95 % threshold + the cause (test gaps vs cold paths that don't merit testing).  Capture to `docs/planning/EPIC_2/SPRINT_29/coverage_audit_day11.md`.
4. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `tests/test_sprint29_integration.c` with 3+ cross-feature tests landed + passing
- `docs/planning/EPIC_2/SPRINT_29/coverage_audit_day11.md` per-file coverage analysis (informational; Day 12's decision builds on it)
- All sanitizer + wall-check + CI jobs green on the sprint-29 branch's HEAD
- All quality checks clean

### Completion Criteria
- 3+ cross-feature integration tests pass under all sanitizers
- Coverage audit identifies the per-file gaps + the proposed Day-12 resolution path (tighten tests vs lower COV_THRESHOLD)
- `make test && make sanitize && make wall-check` clean

---

## Day 12: Item 8 — Coverage-Gate Calibration (Part 2)

**Theme:** Calibrate `make coverage` per the Day-11 audit.  Two paths: (a) tighten the test suite to push aggregate ≥ 95 %, OR (b) lower `COV_THRESHOLD` to a defensible target (~85 %) with documentation of which file groups carry which coverage levels.  Per PROJECT_PLAN.md Item 8, the choice depends on the Day-11 audit's per-file picture.

**Time estimate:** 12 hours

### Tasks
1. Per Day-11's audit, decide path (a) tighten or path (b) lower-threshold:
   - **Path (a)**: pick the 3-5 lowest-coverage files where added tests would yield the most aggregate movement (per-file × file-line-count).  Write targeted tests to push aggregate ≥ 95 %.  Estimate: ~2 hrs per file × 5 files = 10 hrs.
   - **Path (b)**: lower `COV_THRESHOLD` from 95 % to 85 % (or a Day-11-evidence-justified value).  Document the file-group breakdown in `docs/planning/EPIC_2/SPRINT_29/coverage_threshold_decision.md`: numeric kernels at 90-95 %; iterative solvers at 80-85 %; benchmarking + test-only helpers carved out as not-covered intentionally.  Update `Makefile`'s `COV_THRESHOLD` variable + the `.github/workflows/*.yml` coverage step.  Estimate: ~3 hrs.
2. Default to path (b) unless the Day-11 audit identifies low-hanging-fruit gaps that would clear ≥ 8 percentage points aggregate movement in < 10 hrs — the operating-reality calibration is the documented PROJECT_PLAN.md "Sprint 24 Days 12-13 inheritance" path.
3. Whichever path lands, capture the decision + reasoning in `coverage_threshold_decision.md` (Day-11's audit + Day-12's decision + bench evidence).
4. Re-run `make coverage` locally + push to CI to verify the gate now passes.
5. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_29/coverage_threshold_decision.md` decision doc
- Either tightened tests OR lowered `COV_THRESHOLD` per the decision
- CI coverage gate passing on the sprint-29 branch's HEAD
- All quality checks clean

### Completion Criteria
- `make coverage` passes the calibrated threshold locally + in CI
- Decision doc records the path (tighten vs lower) + the per-file-group breakdown
- `make test && make sanitize && make wall-check` clean

---

## Day 13: Item 8 Close — CI Bench-Step Fix + Item 10a (bench_reorder flag) + Item 9 Prep

**Theme:** Close Item 8 with the CI bench-step fix (4 hrs) + Sprint-28-Item-10a `bench_reorder --reorder-via-analyze` flag (~2-3 hrs); start Item 9's Epic 2 retrospective prep (~5-6 hrs).

**Time estimate:** 12 hours (4 hrs Item 8 close + 3 hrs Item 10a + 5 hrs Item 9 part 1)

### Tasks
1. **Item 8 close — CI bench-step fix (4 hrs):** Per PROJECT_PLAN.md Item 8: `make bench`'s 6-hour-runner-timeout failure on PRs #31 + #32 must close.  Two options:
   - **Option (i) `bench-fast` / `bench-ci` target**: a `--skip-factor` variant of `make bench` that captures the symbolic-pass timings without the multi-minute Pres_Poisson numeric factor.  Updates `Makefile` + CI workflow.
   - **Option (ii) move `make bench` to a nightly schedule**: a separate GitHub Actions cron workflow (e.g. `.github/workflows/bench-nightly.yml`) that runs the full bench daily off the PR critical path.
   - Pick option (i) — keeps the bench evidence on PRs without the runtime explosion.  Default the CI's `build-and-test` to `make bench-fast`; document that `make bench` (full) ships as a developer-side opt-in for deep wall-time investigations.
2. **Item 10a `bench_reorder --reorder-via-analyze` flag (3 hrs):** Per `SPRINT_28/RETROSPECTIVE.md` Items-deferred #3: the existing `bench_reorder`'s perm-pre-applied + `sparse_analyze(REORDER_NONE)` path doesn't fire Sprint-28-era env vars (`SPARSE_SUPERNODAL_POSTORDER`, et al.) because `analysis->perm` is NULL.  Add `--reorder-via-analyze` flag that routes through `sparse_analyze(REORDER_AMD)` or `sparse_analyze(REORDER_ND)` instead of the manual pre-apply path; the new flag exposes the env-var dispatch from-tree (no more ad-hoc `/tmp/bench_day*.c` helpers).  Folds naturally into Item-8 because the CI bench-step fix is already touching `bench_reorder.c`.
3. **Item 9 prep (5 hrs):** Begin drafting `docs/planning/EPIC_2/SPRINT_29/RETROSPECTIVE.md` + `docs/planning/EPIC_2/EPIC_2_RETROSPECTIVE.md`:
   - Sprint-29 RETROSPECTIVE.md follows the Sprint-28 template (Status, Goal recap, DoD checklist, Final metrics, Performance highlights, What went well / surprised us / didn't go well, Items deferred, Lessons, Sprint 30+ inputs, Day-by-day capsule, Day-budget vs estimate, DoD verification, Acknowledgements).
   - EPIC_2_RETROSPECTIVE.md is the broader 18-sprint Epic 2 wrap-up: summary table of all sprints, cumulative metrics (ND/AMD nnz_L trajectory across Sprints 22-28; eigensolver capabilities; CSC kernel speedups; etc.), open question journal (literal 0.85× target retirement; supernodal numeric-factor kernel as Sprint-30+ followup; etc.).
4. Update `README.md` "Features" section with Sprint-29 additions: progress callbacks, eigenpair refinement, full SVD U/V output, sparse low-rank improvements.  Update `INSTALL.md` with the new Windows + macOS CI matrix entries.
5. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `Makefile` `bench-fast` / `bench-ci` target + CI workflow updated to use it
- `benchmarks/bench_reorder.c` `--reorder-via-analyze` flag landed
- `docs/planning/EPIC_2/SPRINT_29/RETROSPECTIVE.md` + `EPIC_2_RETROSPECTIVE.md` skeletons stubbed with section structure (Day 14 fills in single-pass)
- `README.md` Sprint-29-additions draft
- `INSTALL.md` macOS + Windows updates
- All quality checks clean

### Completion Criteria
- CI `build-and-test` `make bench` step finishes in < 5 min (down from 6-hour timeout)
- `bench_reorder --reorder-via-analyze` fires the `SPARSE_SUPERNODAL_POSTORDER=on` dispatch (manual verification: set env + run + observe nnz_L change on Pres_Poisson)
- `RETROSPECTIVE.md` skeletons compile + have all section headers ready for Day-14 fill-in
- `make format && make lint && make test && make sanitize && make wall-check` clean

---

## Day 14: Item 9 Close — Epic 2 Retrospective + README + PR

**Theme:** Single-pass closing day per the Sprints 25/26/27/28 retrospective lesson ("single Day-14 retro that absorbs the Day-13 work matches the actual time spent").  Fill in both Sprint-29 + Epic-2 retrospectives; close README + INSTALL sweep; open Sprint 29 PR; address reviewer feedback (estimated 2-3 hrs buffer in the 12-hour budget).

**Time estimate:** 12 hours

### Tasks
1. Validate the Day-12 / Day-13 tests pass under the Sprint-29 default-flipped configuration (`make clean && make test && make sanitize && make wall-check`).  All CI jobs (Linux, Windows, macOS, tsan-on-Linux) green on the sprint-29 branch's HEAD.  If any test trips — particularly the Item-2 full-mode SVD + Item-3 refinement + Item-4 callback + Item-10a `bench_reorder` flag tests — root-cause + either fix in-place (small fix) or document as Sprint 30+ routing in `RETROSPECTIVE.md`.
2. Fill in `docs/planning/EPIC_2/SPRINT_29/RETROSPECTIVE.md` single-pass:
   - **Status**: Sprint 29 final outcome — items closed + items deferred.
   - **Goal recap**: the Sprint 29 charter.
   - **Definition of Done checklist**: per-item ✓/✗ + reference commits.
   - **Final metrics**: SVD low-rank wall + memory deltas (Day 2); eigenpair refinement residual deltas (Day 5); callback overhead (Day 7 wall-check delta); coverage threshold final value (Day 12).
   - **Performance highlights**: production default flips (Day-2 SVD low-rank if flipped; otherwise advisory); largest single-fixture improvement.
   - **What went well / What surprised us / What didn't go well**.
   - **Items deferred (route to Sprint 30+)**: any items that didn't close + reason.
   - **Lessons**: Sprint 29-specific (per-day or per-item).
   - **Sprint 30+ inputs**: concrete handoff items.
   - **Day-by-day capsule** + **Day-budget vs estimate** + **DoD verification** tables.
   - **Acknowledgements**.
3. Fill in `docs/planning/EPIC_2/EPIC_2_RETROSPECTIVE.md` single-pass (the broader 18-sprint wrap-up):
   - **Summary table** of all 18 sprints (titles + key deliverables + actual hours).
   - **Cumulative metrics**: Pres_Poisson ND/AMD nnz_L trajectory Sprint 22 → 28; CSC kernel speedups Sprints 17-19; eigensolver convergence Sprints 20-21; ordering quality Sprints 22-28.
   - **Production default flips landed across Epic 2** (HCC default Sprint 27 Day 2; `nd_base_threshold = 128` Sprint 27 Day 3; SVD low-rank outer-product Sprint 29 Day 2 if flipped; etc.).
   - **Advisory env vars shipped across Epic 2** (cross-reference per axis).
   - **Open question journal**: 0.85× Pres_Poisson target retirement (Sprint 28); supernodal numeric-factor kernels (Sprint-30+); etc.
   - **Lessons** at the Epic level (not just per-sprint).
4. Update `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 section: status flip from "in flight" to "Complete"; record actual hours vs estimated 168; cite the Sprint-29 closures (production default flips landed, advisory env vars, etc.).
5. Final `make format && make lint && make test && make sanitize && make wall-check && make tsan` (the last via the Linux CI tsan job since macOS 15+ tsan is platform-blocked) — must all be clean before PR.
6. Open Sprint 29 PR; request review; address any reviewer feedback (estimated 2-3 hrs of buffer in the 12-hour budget).

### Deliverables
- `docs/planning/EPIC_2/SPRINT_29/RETROSPECTIVE.md` filled in single-pass with all 12 sections
- `docs/planning/EPIC_2/EPIC_2_RETROSPECTIVE.md` Epic-2 wrap-up retrospective
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 status flip to "Complete" + actuals vs estimate
- `README.md` finalised with Sprint-29 additions (built on Day-13 draft)
- `INSTALL.md` finalised with platform updates
- Sprint 29 PR opened + (ideally) merged
- All quality checks clean (format, lint, test, sanitize, wall-check, tsan-on-Linux)

### Completion Criteria
- All Sprint 29 tests pass under the default-flipped configuration on Linux + macOS + Windows + tsan-on-Linux
- `RETROSPECTIVE.md` + `EPIC_2_RETROSPECTIVE.md` record per-item + per-Epic outcomes with concrete Sprint 30+ routing if needed
- `make format && make lint && make test && make sanitize && make wall-check` clean; Linux-CI tsan job green
- Sprint 29 PR is mergeable (CI green, reviewer feedback addressed)
- Epic 2 is formally Complete

---

## Total Time Budget

| Day | Theme | Hours |
|-----|-------|-------|
| 1 | Item 1 — Sparse low-rank design + skeleton | 12 |
| 2 | Item 1 close — outer-product implementation + bench validation | 12 |
| 3 | Item 2 — Full SVD U/V output design + impl part 1 | 12 |
| 4 | Item 2 close (4h) + Item 3 design kickoff (8h) | 12 |
| 5 | Item 3 close — eigenpair refinement implementation + tests | 12 |
| 6 | Item 4 — Progress/cancel callbacks design + first batch (LU/Chol/LDLT) | 12 |
| 7 | Item 4 close (4h) + Item 5 Windows CI start (8h) | 12 |
| 8 | Item 5 Windows CI close (8h) + Item 10b macOS-15+ tsan handling (4h) | 12 |
| 9 | Item 6 — macOS CI (Apple Clang + Homebrew GCC) | 12 |
| 10 | Item 7 — API accessor error reporting | 12 |
| 11 | Item 8 — Final integration testing + cross-feature tests + coverage audit | 12 |
| 12 | Item 8 — Coverage-gate calibration | 12 |
| 13 | Item 8 close (4h) + Item 10a bench_reorder flag (3h) + Item 9 prep (5h) | 12 |
| 14 | Item 9 close — Sprint-29 + Epic-2 retrospective + README + PR | 12 |

**Total: 168 hours** — exactly the 14×12 ceiling.  Items 1-9 from PROJECT_PLAN.md sum to 168 hrs nominally; Item 10's 6-hr Sprint-28-deferral absorption rides within Items 5/6/8 (Item-10a `bench_reorder` flag folds into Item-8 CI bench-step work on Day 13; Item-10b macOS-15+ tsan handling folds into Item-5 Windows CI close + macOS CI on Day 8).  Item 9's 24-hr PROJECT_PLAN.md estimate lands as 20 hrs of allocated time across Days 13-14 — the remaining 4 hrs absorb into retrospective scaffolding that starts incrementally as each item closes during Days 2-12 (each item-close day spends ~30 min updating the in-flight retro skeleton, so the Day-14 single-pass fill operates on a partially-populated draft rather than from scratch).
