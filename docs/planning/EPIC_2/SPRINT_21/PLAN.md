# Sprint 21 Plan: Eigensolver Completion — Thick-Restart, OpenMP & LOBPCG

**Sprint Duration:** 14 days
**Goal:** Close out the symmetric eigensolver family started in Sprint 20.  Replace the provisional growing-m outer loop with a true Wu/Simon thick-restart Lanczos so memory stays bounded on large-n problems; parallelise the full-MGS reorthogonalization inner loop under OpenMP (rounding out the iteration, since `sparse_matvec` is already OpenMP-driven from Sprint 17/18); add LOBPCG for preconditioned block eigenvalue computation via the already-reserved `SPARSE_EIGS_BACKEND_LOBPCG` enum slot; and ship a permanent `benchmarks/bench_eigs.c` executable with CSV output, `--sweep` mode, and a `--compare` mode that benches all three eigensolver backends on the same corpus.

**Starting Point:** Sprint 20 shipped the symmetric Lanczos eigensolver with a growing-m outer loop (Day 13 redesign), full MGS reorthogonalization, Wu/Simon per-pair residuals, and shift-invert via `sparse_ldlt_factor_opts` AUTO dispatch.  The `lanczos_iterate_op` callback refactor from Day 12 (shift-invert enablement) is the seam LOBPCG will reuse.  The `sparse_eigs_opts_t.backend` enum already lists `SPARSE_EIGS_BACKEND_LANCZOS` and reserves the `SPARSE_EIGS_BACKEND_LOBPCG = 2` slot (header comment in `include/sparse_eigs.h`).  Sprint 13's `sparse_precond_fn` callback (IC(0) / LDL^T preconditioners) is available for LOBPCG's preconditioning path.  The Day 13 throwaway `/tmp/bench_eigs.c` driver captured the baseline numbers in `docs/planning/EPIC_2/SPRINT_20/bench_day13_lanczos.txt` — Sprint 21's permanent bench needs to reproduce those numbers under a `--compare` mode.

**End State:** `sparse_eigs_sym` offers three backends routable through `sparse_eigs_opts_t::backend`: the Sprint 20 growing-m Lanczos (still selectable for smoke tests and AUTO's small-problem default), a new thick-restart Lanczos that preserves the converged Ritz subspace in a compact arrowhead basis so peak memory drops from `O(m_cap · n)` to `O((k + m_restart) · n)` (bcsstk14 at n = 1806 drops from ~26 MB of Lanczos V to well under 1 MB with m_restart = 30), and LOBPCG for preconditioned block eigenvalue computation reusing Sprint 13's IC(0) infrastructure.  The full MGS reorth loop is OpenMP-parallel and validated under `-fsanitize=thread`.  `benchmarks/bench_eigs.c` is a permanent executable with `--sweep` (matrix, k, which, backend sweep → CSV) and `--compare` (both Lanczos backends + LOBPCG on the same corpus) modes, with the Sprint 21 capture landed at `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt`.  `tests/test_eigs_thick_restart.c` and `tests/test_eigs_lobpcg.c` cover the new backends; README and `docs/algorithm.md` document the Wu/Simon arrowhead state and LOBPCG Rayleigh-Ritz iteration.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~132 hours — about 8 hours above the 124-hour PROJECT_PLAN.md estimate, providing a safety buffer within the 168-hour (14 × 12) hard ceiling.  The buffer tracks Sprint 20's actual-vs-estimate experience (estimated 142 hrs, shipped at ~140 hrs — the cushion was consumed by the Day 13 Wu/Simon outer-loop redesign when the Day 10 stability-check restart saturated on SuiteSparse).  Sprint 21's risk concentration is similar: item 1's arrowhead state is a novel data structure, and item 3's LOBPCG Rayleigh-Ritz has its own subtle convergence failure modes the growing-m path doesn't share.

---

## Day 1: Thick-Restart Lanczos — Design & Arrowhead State

**Theme:** Design the Wu/Simon thick-restart mechanism, define the arrowhead data structure the restart will preserve, and stub the new `lanczos_thick_restart_iterate` entry point

**Time estimate:** 10 hours

### Tasks
1. Read the Sprint 20 Day 13 growing-m outer loop in `src/sparse_eigs.c` (`sparse_eigs_sym`, the `for (;;)` grow-retry loop around line 717) and the `lanczos_iterate_op` callback shape (`lanczos_op_fn` + `ctx`).  Note how `V`, `alpha`, `beta`, `m_cap` are sized for the full Lanczos run — peak memory is `O(m_cap · n)`, which is what thick-restart needs to reduce.
2. Read the Wu/Simon (2000) and Stathopoulos/Saad (2007) papers on the Lanczos side (key insight: after a restart, the tridiagonal T becomes an *arrowhead* — a diagonal containing the locked Ritz values plus a trailing row/column of β-coupling entries and one active α/β row).  Sketch the arrowhead shape for m_restart = 30, k_locked = 5 and work out the indexing that `tridiag_qr_eigenpairs` needs to tolerate.
3. Decide the restart protocol:
   - **Option A:** Full thick-restart (Wu/Simon 2000) — keep the top k converged Ritz vectors, form the arrowhead T, continue iterating.  Preserves all convergence progress.
   - **Option B:** Simple deflation — lock converged pairs out of the basis, restart Lanczos on the unconverged subspace only.  Loses some history but simpler to implement.
   - **Option C (recommended):** Full Wu/Simon thick-restart with an explicit `lanczos_restart_state_t` struct carrying the locked Ritz pairs, the arrowhead-augmented T, and the residual vector.  Mirrors the Sprint 20 Day 12 `lanczos_iterate_op` callback pattern — keeps the Lanczos core callback-driven and makes the restart state inspectable in tests.
4. Design the new data structure `lanczos_restart_state_t` in `src/sparse_eigs_internal.h`:
   - `V_locked` (n × k_locked) and `theta_locked` (k_locked) — the Ritz pairs surviving the restart.
   - `alpha_arrow` / `beta_arrow` — arrowhead diagonal + trailing-row entries (size m_restart).
   - `residual` — the Lanczos residual vector `β_m · v_{m+1}` at the moment of restart; seeds the next iteration.
   - Accessor helpers + a `lanczos_restart_state_free` to match the Sprint 20 `sparse_ldlt_free` convention.
5. Design the new entry point `lanczos_thick_restart_iterate` as a variant of `lanczos_iterate_op` that accepts an existing restart state (or NULL to start fresh), runs one Lanczos phase of length m_restart, and writes back an updated arrowhead state.  Public-API-adjacent but internal — stays in `sparse_eigs_internal.h`.
6. Write the file-header-style design block in `src/sparse_eigs.c` mirroring Sprint 20 Day 8's Lanczos design block: explain the arrowhead structure, the Wu/Simon references, why growing-m remains as a fallback, and the memory-bound claim `O((k + m_restart) · n)`.
7. Add compile-ready stubs: `lanczos_restart_state_t`, `lanczos_thick_restart_iterate` returning `SPARSE_ERR_BADARG` (the codebase's "stub in progress" signal; no `SPARSE_ERR_NOT_IMPL` in this library) so Days 2-3 have a target to replace without header churn.
8. Run `make format && make lint && make test` — all clean (no behavior change yet).

### Deliverables
- `lanczos_restart_state_t` declared in `src/sparse_eigs_internal.h`
- `lanczos_thick_restart_iterate` signature + design block committed to `src/sparse_eigs.c`
- Compile-ready stubs returning `SPARSE_ERR_BADARG`

### Completion Criteria
- Design block referenced back to Wu/Simon (2000) and Stathopoulos/Saad (2007) with the arrowhead shape sketched
- Stub compiles and links; no public-header changes (Sprint 20's `include/sparse_eigs.h` surface is stable)
- `make format && make lint && make test` clean

---

## Day 2: Thick-Restart Lanczos — Arrowhead Reduction & Ritz Locking

**Theme:** Implement the arrowhead tridiagonal eigensolve and the Ritz-locking step that feeds the next restart — the numeric core of the Wu/Simon mechanism

**Time estimate:** 10 hours

### Tasks
1. Implement the arrowhead-to-tridiagonal reduction.  After the locked-Ritz block + trailing α/β row is assembled, run a short sequence of Givens rotations to chase the trailing coupling entries down to a symmetric tridiagonal shape that the existing `tridiag_qr_eigenpairs` (Sprint 20 Day 11) can consume unchanged.  The Sprint 20 Ritz extraction is already Y-producing — reuse it wholesale.
2. Implement `lanczos_restart_pick_locked`: given the m × m Y matrix from the current Ritz extraction and the selection indices `sel_idx` (already computed by Sprint 20's `s20_select_indices` helper), form V_locked = V · Y[:, sel_idx] via a column-major gemm inlined in the same style as `s20_lift_ritz_vectors` (Sprint 20 Day 11).  Return the locked subspace, the locked θ values, and the trailing β·y_{m-1,j} coupling for the arrowhead.
3. Implement `lanczos_restart_state_assemble`: pack the locked pairs + coupling into the `lanczos_restart_state_t` struct, copy the residual `β_m · v_{m+1}` (already at `V[:, m_actual]` when Lanczos terminates on m_max without an invariant-subspace exit), and NULL-initialise the fields the next phase will fill.
4. Unit tests in the new `tests/test_eigs_thick_restart.c`:
   - Arrowhead reduction round-trip: assemble an arrowhead with 3 locked pairs + 2 trailing rows; reduce to tridiagonal; confirm the resulting T has the same spectrum as the arrowhead to 1e-12.
   - Locked subspace correctness: assemble V_locked = V · Y_k; confirm `V_locked^T · V_locked ≈ I` to 1e-10 (requires the caller to have enabled full MGS reorth, which Sprint 20 documented as necessary for reliable lifted eigenvectors).
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Arrowhead-to-tridiagonal reduction via Givens rotations
- `lanczos_restart_pick_locked` / `_assemble` producing a valid restart state
- 2+ unit tests in `tests/test_eigs_thick_restart.c`

### Completion Criteria
- Arrowhead reduction preserves the spectrum to 1e-12 on the unit fixture
- Locked subspace orthonormality ≤ 1e-10 (with reorth on)
- `make format && make lint && make test && make sanitize` clean

---

## Day 3: Thick-Restart Lanczos — Phase Execution & Restart Loop

**Theme:** Run a Lanczos phase starting from an existing restart state, chain phases together into the outer thick-restart loop, and replace Sprint 20's growing-m outer loop with the new mechanism (dispatched via an opts flag so the old path stays available)

**Time estimate:** 10 hours

### Tasks
1. Implement `lanczos_thick_restart_iterate` body: if the caller passes a seed restart state, initialise the first k_locked Lanczos vectors to V_locked and seed v_{k_locked+1} from the state's residual; otherwise behave exactly like `lanczos_iterate_op` (empty restart state = fresh start).  Runs one Lanczos phase of length `m_restart`; returns the completed α/β and the new m_actual.  Reuses the Sprint 20 full-MGS reorth + scale-aware breakdown threshold from commit 70015a4.
2. Compose the outer loop `s21_thick_restart_outer_loop` in `src/sparse_eigs.c`:
   - Initial phase: `lanczos_thick_restart_iterate(state=NULL, m_restart)` → α, β, V of size m_restart.
   - Ritz extraction via `s20_ritz_pairs` (unchanged from Sprint 20).
   - Wu/Simon per-pair residual check using the existing Sprint 20 machinery.
   - If k converged: emit result, return SPARSE_OK.
   - If not converged and retry budget remains: `lanczos_restart_pick_locked` + `lanczos_restart_state_assemble` → restart state; loop back with a non-NULL state.
3. Add an opts dispatch in `sparse_eigs_sym`:
   - New enum value `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` (append to the existing enum — positional initialisers still compile because it's last).
   - AUTO routing: thick-restart when `n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD` (provisional 500; tune in Day 4) and growing-m below.  Document the threshold in the public header.
   - LANCZOS (existing) → growing-m path unchanged.
4. Residual vector availability: confirm Sprint 20's `lanczos_iterate_op` leaves `V[:, m_actual]` populated when it terminates on `m_max` without invariant-subspace exit (that's the restart residual).  If not, add a parameter so the thick-restart path can request it without touching the growing-m callers.
5. Run `make format && make lint && make test && make sanitize` — all clean.  The Sprint 20 tests must still pass because the growing-m path is unchanged.

### Deliverables
- `lanczos_thick_restart_iterate` implementation chaining Lanczos phases through a restart state
- Outer loop `s21_thick_restart_outer_loop` composing iterate + restart + convergence
- `sparse_eigs_sym` dispatch routing to thick-restart above the threshold

### Completion Criteria
- New backend compiles and dispatches correctly per `opts->backend` setting
- All Sprint 20 tests pass unchanged (growing-m path untouched)
- `make format && make lint && make test && make sanitize` clean

---

## Day 4: Thick-Restart Lanczos — Memory-Bounded Convergence & Parity Tests

**Theme:** Verify thick-restart converges with bounded peak memory and matches the growing-m path numerically on the Sprint 20 corpus

**Time estimate:** 10 hours

### Tasks
1. Memory instrumentation: add a `peak_V_elems` counter threaded through both Lanczos paths that records the maximum number of V columns held at any time.  Thick-restart at `m_restart = 30`, `k = 5` should cap at ~35 columns regardless of total iteration count; growing-m grows monotonically up to `m_cap`.  Log via the existing `result->iterations` pattern (add a sibling `result->peak_basis_size` field — positional initialisers still compile because it's trailing).
2. Memory-bounded test on bcsstk14 (n = 1806, k = 5 LARGEST): thick-restart with `m_restart = 30` must converge with `peak_V_elems ≤ 35 · n` (~500 KB of V vs the ~26 MB the growing-m path currently allocates).  Assert residual ≤ 1e-10 on the returned pairs.
3. Cross-backend parity on the Sprint 20 corpus: for nos4, bcsstk04, bcsstk14, and kkt-150 (the four Day 13 bench fixtures), run both backends with identical opts and assert:
   - Eigenvalues match to 1e-10.
   - `result.residual_norm` matches within 2× (thick-restart typically has better residuals at higher iteration counts; growing-m can have a slight advantage when m_cap is very large — the 2× band covers both).
   - `result.n_converged == k` on both.
4. Tune `SPARSE_EIGS_THICK_RESTART_THRESHOLD` (introduced Day 3 at 500): measure crossover on the nos4 (n = 100) → bcsstk04 (n = 132) → KKT-150 (n = 150) progression.  Thick-restart likely wins above n ≈ 200 for k ≤ 10; adjust the threshold accordingly.  Document the measured crossover in `docs/planning/EPIC_2/SPRINT_21/bench_day4_restart.txt`.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Peak-memory counter threaded through both Lanczos paths; exposed as `sparse_eigs_t.peak_basis_size`
- Memory-bounded test on bcsstk14 confirming thick-restart's bounded footprint
- Parity test across four fixtures and two backends
- Tuned `SPARSE_EIGS_THICK_RESTART_THRESHOLD` with bench capture

### Completion Criteria
- Thick-restart bcsstk14 converges at `peak_basis_size ≤ 35 · n` with residual ≤ 1e-10
- Parity tests pass on all four Sprint 20 corpus fixtures
- `make format && make lint && make test && make sanitize` clean

---

## Day 5: OpenMP Lanczos Reorthogonalization

**Theme:** Parallelise the full-MGS reorth inner loop under `-DSPARSE_OPENMP` so the Lanczos iteration gets the same treatment `sparse_matvec` already has from Sprint 17/18

**Time estimate:** 10 hours

### Tasks
1. Profile the Sprint 20 Lanczos inner loop on bcsstk14 k=5 LARGEST.  Expected hot triad per step: (a) matvec `A · v_k` (already OpenMP from Sprint 17/18), (b) full-MGS reorth `w -= <w, v_j> · v_j` for j = 0..k-1 (serial), (c) β norm + normalize (small, not worth parallelising).  Record baseline wall times with `OMP_NUM_THREADS` in {1, 2, 4, 8}.
2. Parallelise the MGS reorth loop in `lanczos_iterate_op`:
   - Inner dot product `<w, v_j>`: `#pragma omp parallel for reduction(+:dot)` on the `i` axis.  Same pattern Sprint 17/18 used for matvec.
   - Subtract `w[i] -= dot * v_j[i]`: `#pragma omp parallel for` on `i`.
   - The outer `j` loop stays serial — MGS requires each iteration to see the partially-orthogonalised `w` (classical Gram-Schmidt parallelises the `j` loop but loses the MGS stability; Sprint 20's design block documents why we keep MGS).
   - Gate the pragmas behind `#ifdef SPARSE_OPENMP` so non-OpenMP builds are unchanged.
3. Apply the same treatment inside `lanczos_thick_restart_iterate` (Day 3): the restart-phase's reorth loop is the same kernel, so a single helper `s21_mgs_reorth_step(w, V, j, n)` called from both paths keeps the diff small.
4. Confirm correctness under `-fsanitize=thread` via a new `make sanitize-thread` target (add to Makefile if not present — mirrors the existing `make sanitize` pattern for ASan/UBSan).  Run `tests/test_eigs.c` + `tests/test_eigs_thick_restart.c` under TSan; zero data races.
5. Quick-win optimisation — memory bandwidth: the dot-product + subtract pair reads `w` and `v_j` twice.  If the profile shows memory-bound behaviour, fuse into a single pass with private accumulators + reduction, cutting memory traffic in half.  Only apply if the measured speedup justifies the complexity; otherwise defer to Sprint 22.
6. Run `make format && make lint && make test && make sanitize && make sanitize-thread` — all clean.

### Deliverables
- Parallelised MGS reorth in both `lanczos_iterate_op` and `lanczos_thick_restart_iterate` under `SPARSE_OPENMP`
- `make sanitize-thread` target (if not already present) exercising the Lanczos paths under TSan
- Baseline timing capture in `docs/planning/EPIC_2/SPRINT_21/bench_day5_omp.txt` (serial vs OMP_NUM_THREADS ∈ {1, 2, 4, 8})

### Completion Criteria
- TSan reports zero races on the Lanczos tests
- Serial build remains bit-for-bit identical to Sprint 20 (no behavioural drift)
- `make format && make lint && make test && make sanitize && make sanitize-thread` clean

---

## Day 6: OpenMP Lanczos Benchmarks, Scaling & Tuning

**Theme:** Measure the actual speedup, identify bottlenecks, tune the reorth parallelisation, and update PERF_NOTES / docs

**Time estimate:** 10 hours

### Tasks
1. Scaling study: run the Sprint 20 corpus (nos4, bcsstk04, bcsstk14, kkt-150) under both Lanczos backends (growing-m and thick-restart) at `OMP_NUM_THREADS ∈ {1, 2, 4, 8, 16}`.  Expected result: 2–3× speedup at 4 threads on bcsstk14 m=70; less on smaller fixtures where reorth is a smaller fraction of the step.
2. Critical-path analysis: confirm that at high thread counts the remaining bottleneck is the Lanczos step's serial overhead (the β norm, α dot, and cache misses on V's layout).  If the critical path is V layout (column-major access patterns hitting non-sequential memory), flag it as a Sprint 22 candidate rather than a Sprint 21 blocker.
3. Threshold tuning: decide whether to gate the OMP reorth on an `n` or `m` threshold — OpenMP overhead dominates at very small problem sizes (the Sprint 17/18 matvec has the same issue; check if that code gates on nnz).  Implement as a static inline helper so the gate is compile-out when `SPARSE_OPENMP` is undefined.
4. `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` extension: add a "Sprint 21 Lanczos OpenMP" subsection with the measured numbers, the threshold decision, and a cross-reference to the matvec parallelisation that set the pattern.
5. Update `docs/algorithm.md` Lanczos section (added in Sprint 20 Day 14): add a paragraph covering the parallelisation strategy and the MGS-vs-CGS tradeoff under parallel reorth (MGS stays serial in `j`, CGS would parallelise in `j` but loses numerical stability — same decision the sequential Sprint 20 path made, just reaffirmed).
6. Run `make format && make lint && make test && make sanitize && make sanitize-thread && make bench` — all clean.

### Deliverables
- Scaling capture in `docs/planning/EPIC_2/SPRINT_21/bench_day6_omp_scaling.txt` covering both backends × four fixtures × five thread counts
- Threshold-gated reorth parallelisation (if the profile motivates it)
- PERF_NOTES.md "Sprint 21 Lanczos OpenMP" subsection
- `docs/algorithm.md` paragraph on parallel MGS vs classical Gram-Schmidt

### Completion Criteria
- Measured speedup ≥ 2× at 4 threads on bcsstk14 (the PROJECT_PLAN.md item-2 target)
- No regression vs Sprint 20 on the serial (`OMP_NUM_THREADS=1`) path
- `make format && make lint && make test && make sanitize && make sanitize-thread && make bench` clean

---

## Day 7: LOBPCG — API Design & Block Rayleigh-Ritz Infrastructure

**Theme:** Design the LOBPCG backend, fit it into the Sprint 20 `sparse_eigs_t` API via the reserved `SPARSE_EIGS_BACKEND_LOBPCG` slot, and stub the block Rayleigh-Ritz infrastructure

**Time estimate:** 10 hours

### Tasks
1. Read the Knyazev LOBPCG (2001) paper and the BLOPEX reference implementation notes.  Key data structure: a block of three n × block_size matrices — X (current approximate eigenvectors), W (preconditioned residuals), P (previous iteration's search directions) — concatenated into a 3·block_size subspace for each Rayleigh-Ritz step.
2. Inventory the reusable pieces from Sprint 20:
   - `lanczos_op_fn` callback type (`src/sparse_eigs_internal.h`) — LOBPCG's `A · X` multiply uses the same signature.
   - `tridiag_qr_eigenpairs` — not reusable for the 3k × 3k dense symmetric Rayleigh-Ritz; need a dense symmetric eigensolver helper.  Option A: reuse `sparse_dense.c`'s `jacobi_eig_symmetric` (Sprint 14, used by SVD).  Option B: roll a small dense tridiagonal + Givens reducer inline (more code but zero dependency).  Recommend Option A — Jacobi is robust for k × k where k ≤ 30 and already in the library.
   - `sparse_precond_fn` callback (Sprint 13) — the W-block preconditioning path plugs in IC(0) or LDL^T unchanged.
3. Public-API surface: extend `sparse_eigs_opts_t` with LOBPCG-specific options:
   - `int block_size` — LOBPCG's `k_block` (default 0 → auto-set to `k`; callers can request a bigger block for better convergence at the cost of more memory).
   - `sparse_precond_fn precond` + `void *precond_ctx` — optional preconditioner; NULL → identity (vanilla LOBPCG without preconditioning, slower but still correct).
   - Document the new fields as designated-init-safe (trailing, zero-default selects library defaults).
4. Internal API surface in `src/sparse_eigs_internal.h`:
   - `s21_lobpcg_rr_step`: given X, W, P (three n × block_size matrices), A (via callback), compute the 3·block_size × 3·block_size Gram matrix, run Rayleigh-Ritz via `jacobi_eig_symmetric`, extract the block_size lowest (or whichever matches `which`) Ritz pairs, form the next X / P from the combination coefficients.
   - `s21_lobpcg_orthonormalize_block`: QR-like orthonormalisation of an n × block_size matrix via modified Gram-Schmidt (reusing Sprint 20's MGS kernel).  Critical for numerical stability of Rayleigh-Ritz — a non-orthonormal basis produces junk eigenvalues.
5. Stub `s21_lobpcg_solve` returning `SPARSE_ERR_BADARG` ("stub in progress") so Day 8 has a target.  Wire the dispatch in `sparse_eigs_sym`: `SPARSE_EIGS_BACKEND_LOBPCG` → `s21_lobpcg_solve`; AUTO stays with Lanczos for now (Day 10 decides the LOBPCG crossover threshold).
6. Design block in `src/sparse_eigs.c` covering: the 3-block subspace, Rayleigh-Ritz, the Knyazev convergence heuristic, and the preconditioning reuse of `sparse_precond_fn`.  Same pedagogical depth as Sprint 20's Lanczos design block.
7. Run `make format && make lint && make test` — all clean (no behaviour change; stub returns BADARG which is covered by the Sprint 20 precondition tests).

### Deliverables
- Extended `sparse_eigs_opts_t` with `block_size` + `precond` + `precond_ctx` fields
- `s21_lobpcg_rr_step` + `s21_lobpcg_orthonormalize_block` + `s21_lobpcg_solve` stubs
- Design block in `src/sparse_eigs.c` covering the Rayleigh-Ritz pipeline
- Dispatch wired in `sparse_eigs_sym`

### Completion Criteria
- New public-header fields documented + positional-initialiser compatibility verified
- Stub compiles and links; existing tests unaffected
- `make format && make lint && make test` clean

---

## Day 8: LOBPCG — Vanilla Rayleigh-Ritz Core (No Preconditioning)

**Theme:** Implement the unpreconditioned LOBPCG iteration — full block Rayleigh-Ritz with X/W/P blocks, converging on SPD fixtures where Lanczos already works

**Time estimate:** 10 hours

### Tasks
1. Implement `s21_lobpcg_orthonormalize_block`: column-by-column modified Gram-Schmidt on an n × block_size matrix with a breakdown check (column norm < `scale * 1e-14`; eject the column and shrink the block).  Reuse Sprint 20's scale-aware breakdown pattern from commit 70015a4.  Returns the effective block size after any ejections.
2. Implement `s21_lobpcg_rr_step`:
   - Concatenate X, W, P columnwise into a 3·block_size basis.
   - Orthonormalise via the helper above (orthonormality is required for the Gram matrix below to be symmetric positive-definite).
   - Compute the Gram matrix `G = Q^T · A · Q` of size 3·block_size × 3·block_size by applying `A` blockwise (one `op` call per column is the simple route; if the `op` supports a block form, use it — for now call per column, wrap in a small inline helper so Sprint 22 can swap in a block matvec).
   - Run `jacobi_eig_symmetric(G)` to get the Ritz pairs.
   - Select the block_size lowest (or matching `which`) Ritz values + their eigenvector columns `Y`.
   - Form the next X = Q · Y; next P = difference between the combination coefficients in the current step and the previous — classic LOBPCG formula (Knyazev 2001 eq. 2.11).
3. Implement the LOBPCG outer loop in `s21_lobpcg_solve`:
   - Initial X: random orthonormal n × block_size (deterministic pseudo-random — reuse Sprint 20's golden-ratio fractional v0 mixing extended to block_size columns).
   - Initial P: zero.
   - Compute residual R = A·X − X·Λ for the current Λ = diag(Ritz values); this is W (preconditioned residual is just R when `precond` is NULL).
   - Orthogonalise W against X, orthonormalise, run `s21_lobpcg_rr_step`, update X and P, recompute R.
   - Convergence gate per-pair: ‖R[:, j]‖ / max(|λ_j|, scale) ≤ tol.  Matches Sprint 20's Wu/Simon scale anchor so the residual field has consistent semantics across backends.
4. Initial smoke test on nos4 (n=100, k=5 LARGEST): LOBPCG should converge in < 100 outer iterations with residual ≤ 1e-8.  Weaker than the Lanczos 4e-14 — LOBPCG without preconditioning has a well-known slower asymptotic rate; the preconditioner in Day 9 closes the gap.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `s21_lobpcg_orthonormalize_block` with breakdown handling
- `s21_lobpcg_rr_step` implementing the block Rayleigh-Ritz core
- `s21_lobpcg_solve` outer loop with vanilla (unpreconditioned) convergence on nos4

### Completion Criteria
- nos4 k=5 LARGEST converges in ≤ 100 LOBPCG iterations with residual ≤ 1e-8
- Random fixture stability: 5 reruns with different deterministic seeds produce the same eigenvalues to 1e-10
- `make format && make lint && make test && make sanitize` clean

---

## Day 9: LOBPCG — Preconditioning & Block Direction Update

**Theme:** Plug Sprint 13's `sparse_precond_fn` into the W block, refine the P block update formula, and validate convergence speedup on ill-conditioned fixtures

**Time estimate:** 9 hours

### Tasks
1. Thread the preconditioner through `s21_lobpcg_solve`:
   - When `opts->precond != NULL`, compute `W[:, j] = precond(R[:, j])` for each column.  When NULL, W = R unchanged (vanilla path from Day 8).
   - Orthogonalise W against the current X and normalise; ejected columns drop the effective block size (Day 8 orthonormaliser already handles this).
2. Refine the P update formula: Knyazev's original eq. 2.11 vs the "more robust" formulation used by BLOPEX (Stathopoulos 2007).  For LOBPCG the difference matters on near-singular Gram matrices — implement the BLOPEX variant with a guard on the Gram matrix's conditioning via Jacobi's reported eigenvalue spread.  Cite the reference in an inline comment.
3. Soft-locking: when a Ritz pair's residual drops below `tol`, optionally freeze it in the X block by setting the corresponding W and P columns to zero.  Reduces the effective block size in subsequent iterations — important on problems where the spectrum has a large gap between the bottom-k and the rest.  Gate on an `opts->lobpcg_soft_lock` flag (default on — saves time on easy problems, no correctness difference).
4. Preconditioned regression test: build an ill-conditioned SPD (diag-spread 1e6, n=500, k=5 LARGEST) where vanilla LOBPCG from Day 8 takes > 500 iterations.  With IC(0) preconditioning (constructed via `sparse_ic_factor` + `sparse_ic_precond` — Sprint 13 infrastructure), convergence should drop to < 100 iterations.  Assert residual ≤ 1e-8 and iteration count ≤ 100.
5. LDL^T preconditioning variant: same fixture, but use `sparse_ldlt_factor_opts` + `sparse_ldlt_solve` as the preconditioner (wrap in a small `sparse_precond_fn` adapter).  Cross-check that LDL^T preconditioning converges faster than IC(0) on the ill-conditioned fixture — a sanity gate that the preconditioning path is actually helping.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- Preconditioning path in `s21_lobpcg_solve` gated on `opts->precond != NULL`
- BLOPEX-style P-update formula with conditioning guard
- Soft-locking implementation gated on `opts->lobpcg_soft_lock`
- Regression test with IC(0) + LDL^T preconditioners on an ill-conditioned SPD

### Completion Criteria
- Preconditioned LOBPCG converges ≥ 5× faster than vanilla on the ill-conditioned fixture
- IC(0) and LDL^T preconditioning paths both work (LDL^T typically faster)
- `make format && make lint && make test && make sanitize` clean

---

## Day 10: LOBPCG — AUTO Dispatch Threshold, Smallest/Sigma Coverage, Integration

**Theme:** Tune where AUTO should pick LOBPCG vs Lanczos, extend LOBPCG to SMALLEST / NEAREST_SIGMA, and complete the three-backend integration

**Time estimate:** 8 hours

### Tasks
1. SMALLEST coverage: LOBPCG's default spectrum-selection mode is lowest eigenvalues (physics heritage — vibration modes, Laplacian eigenmaps).  Confirm the Day 8-9 implementation already lands `SMALLEST` correctly; add test coverage matching Sprint 20's SMALLEST fixture (bcsstk04 k=3 SMALLEST).  LARGEST requires negating the operator's action — wrap `op(ctx, n, x, y) → y = A·x` into a `neg_op(ctx, n, x, y) → y = -A·x` adapter when `which == LARGEST`, then negate the returned eigenvalues.  Matches the Sprint 20 Day 12 shift-invert callback pattern.
2. NEAREST_SIGMA coverage: LOBPCG with shift-invert = LOBPCG where the operator is `(A − σI)^{-1}` (same as Sprint 20's shift-invert Lanczos).  Reuse the Sprint 20 shift-invert context setup code (build A_shifted, factor via `sparse_ldlt_factor_opts`, LDL^T solve as the op) — share the setup via a helper `s21_build_shift_invert_context(A, sigma, &ctx)` that both Lanczos backends and LOBPCG can call.  Post-process λ = σ + 1/θ as before.
3. AUTO threshold tuning: benchmark the three backends (growing-m, thick-restart, LOBPCG) on the corpus `{nos4, bcsstk04, bcsstk14, kkt-150}` × `which ∈ {LARGEST, SMALLEST, NEAREST_SIGMA}`.  Decide the AUTO routing policy:
   - `n < 200`: growing-m Lanczos (current AUTO).
   - `200 ≤ n < 1000`: thick-restart Lanczos (Day 3 default).
   - `n ≥ 1000` AND `block_size ≥ 4` AND `precond != NULL`: LOBPCG.
   - Otherwise thick-restart Lanczos.
4. Document the AUTO decision tree in `include/sparse_eigs.h` (the existing `sparse_eigs_backend_t` doc block).  Mark each threshold as "tuned on the Sprint 21 bench corpus; tune further in future sprints when the workload shifts".
5. Full regression: run the Sprint 20 corpus tests under AUTO with the new routing.  Every test must still pass — AUTO's choice might change, but the numerical results must remain correct on every fixture.
6. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- LOBPCG SMALLEST / LARGEST / NEAREST_SIGMA all functional
- Shared `s21_build_shift_invert_context` helper (refactor of Sprint 20 Day 12 code)
- AUTO dispatch decision tree for all three backends + `sparse_eigs_backend_t` doc block
- Full corpus regression under new AUTO routing

### Completion Criteria
- LOBPCG matches Lanczos results to 1e-8 on all three `which` modes across the corpus
- AUTO routing verified via `result.used_csc_path_ldlt` + new `result.backend_used` field (added this sprint to mirror the LDL^T dispatch telemetry)
- `make format && make lint && make test && make sanitize` clean

---

## Day 11: `benchmarks/bench_eigs.c` — Permanent Driver with CSV + Sweep + Compare

**Theme:** Replace the Sprint 20 Day 13 throwaway `/tmp/bench_eigs.c` with a permanent benchmark executable that drives all three backends over the corpus with reproducible CSV output

**Time estimate:** 10 hours

### Tasks
1. Create `benchmarks/bench_eigs.c` mirroring the structure of `benchmarks/bench_ldlt_csc.c` (Sprint 20 Day 6 added the `--dispatch` mode there as the template):
   - `main` argument parser: `--matrix <path>` (required), `--k <N>`, `--which <LARGEST|SMALLEST|NEAREST|all>`, `--sigma <float>`, `--backend <GROWING_M|THICK_RESTART|LOBPCG|all>`, `--precond <NONE|IC0|LDLT>`, `--repeats <N>` (default 5 for median), `--csv`, `--compare`, `--sweep <preset>`.
   - Fixture loader wrapping `sparse_load_mm` + the Sprint 20 KKT builder (for shift-invert coverage).
   - Timer via `clock_gettime(CLOCK_MONOTONIC)`, report median of `--repeats` runs.
2. CSV output format: header row `matrix,n,k,which,sigma,backend,precond,iterations,peak_basis,wall_ms,residual,status`.  One row per (matrix, k, which, backend, precond) combination, sorted by matrix + backend for easy diff-ing across commits.
3. `--sweep default` preset: (matrices: nos4, bcsstk04, bcsstk14, kkt-150) × (k: 3, 5) × (which: LARGEST, SMALLEST) + NEAREST_SIGMA at sigma=0 on kkt-150 × (all three backends).  Roughly 4×2×2×3 + 1×3 = 51 rows.  Caps total runtime at ~5 minutes under `-O2` on the author's development machine.
4. `--compare` preset: focused on the three-backend head-to-head on a few representative problems.  Output a second CSV that pivots the default sweep — rows are (matrix, k, which), columns are (backend, iterations, wall_ms, residual).  Makes it easy to see "LOBPCG wins here, thick-restart wins there" at a glance.
5. Register the binary in `benchmarks/CMakeLists.txt` (if present) and the existing `Makefile` bench target.  Add a smoke invocation `make bench-eigs` that runs the default sweep to stdout (no CSV) so CI / local validation catches regressions.
6. Add a short README subsection under `benchmarks/` (mirroring the existing `bench_ldlt_csc` notes) pointing at `bench_eigs --help` and the CSV schema.
7. Run `make format && make lint && make test && make sanitize && make bench && make bench-eigs` — all clean.

### Deliverables
- `benchmarks/bench_eigs.c` with `--matrix`, `--k`, `--which`, `--sigma`, `--backend`, `--precond`, `--repeats`, `--csv`, `--compare`, `--sweep` modes
- CSV schema documented in `benchmarks/README.md`
- `make bench-eigs` target invoking the default sweep
- Smoke-level integration with the existing `make bench`

### Completion Criteria
- `bench_eigs --sweep default --csv > /tmp/s21_sweep.csv` produces a ~50-row CSV in < 5 min
- `bench_eigs --compare --csv` produces a pivoted three-backend head-to-head CSV
- `make format && make lint && make test && make sanitize && make bench && make bench-eigs` clean

---

## Day 12: Thick-Restart Tests & SuiteSparse Memory-Bounded Regression

**Theme:** Build out `tests/test_eigs_thick_restart.c` with memory-bounded convergence, cross-backend parity, and the full Sprint 21 regression corpus for item 1

**Time estimate:** 8 hours

### Tasks
1. Flesh out `tests/test_eigs_thick_restart.c` (created stub-style on Day 2) with the full Sprint 21 coverage:
   - `test_thick_restart_diagonal_k3` — same fixture Sprint 20 Day 11 used; confirms the new backend is correct on trivial cases.
   - `test_thick_restart_nos4_k5_largest` — convergence in ≤ 2× the growing-m iteration count with `peak_basis_size ≤ m_restart + k + 5`.
   - `test_thick_restart_bcsstk14_bounded_memory` — the memory-bound regression from Day 4, promoted to a permanent test: `peak_basis_size * sizeof(double) * n ≤ 1 MB` at k=5, m_restart=30.
   - `test_thick_restart_matches_growing_m_residual` — parity test on kkt-150 NEAREST_SIGMA: both backends produce eigenvalues matching to 1e-10 and residuals within 2× of each other.
   - `test_thick_restart_locked_pairs_preserved` — after a restart, the previously-converged Ritz pairs have monotonically non-increasing residuals (the core Wu/Simon claim: restart preserves convergence progress).
2. Negative-path coverage:
   - `test_thick_restart_invalid_m_restart` — `opts->m_restart > max_iterations` must return `SPARSE_ERR_BADARG` (extends the Sprint 20 commit 27acf6f `min_required` check to the thick-restart field).
   - `test_thick_restart_single_phase` — `m_restart == max_iterations` degenerates to a single Lanczos run and the result must match the Sprint 20 growing-m output bit-for-bit on a small diagonal fixture.
3. Integration with the Sprint 20 test runner: register all new tests in `tests/test_eigs.c`'s `main` (or create a new runner if the file is getting too big — match the Sprint 19/20 convention where large families got their own file).
4. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `tests/test_eigs_thick_restart.c` with 5+ positive-path + 2+ negative-path tests
- Memory-bound regression promoted to a permanent assertion
- Registered in the runner; counted in the test suite total

### Completion Criteria
- All thick-restart tests pass under `make test && make sanitize`
- Memory bound asserted numerically (not just measured) in at least one test
- `make format && make lint && make test && make sanitize` clean

---

## Day 13: LOBPCG Tests & Benchmark Capture

**Theme:** Build `tests/test_eigs_lobpcg.c`, cross-validate LOBPCG against Lanczos on the Sprint 20 corpus, and capture the Sprint 21 bench as `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt`

**Time estimate:** 8 hours

### Tasks
1. Create `tests/test_eigs_lobpcg.c` following the Sprint 20 `test_eigs.c` structure:
   - `test_lobpcg_diagonal_smallest` — SMALLEST on a diagonal fixture; the easy-path sanity test.
   - `test_lobpcg_tridiag_spd_matches_lanczos` — cross-backend parity on the Sprint 20 SPD tridiag fixture; eigenvalues match to 1e-8.
   - `test_lobpcg_preconditioned_ic0` — ill-conditioned SPD with IC(0) preconditioning; asserts the convergence speedup claim from Day 9 (< 100 iterations vs > 500 vanilla).
   - `test_lobpcg_preconditioned_ldlt` — same fixture, LDL^T preconditioner; LDL^T should converge in fewer outer iterations than IC(0) on this fixture.
   - `test_lobpcg_shift_invert_kkt` — NEAREST_SIGMA on kkt-150; matches Sprint 20's Lanczos shift-invert result to 1e-8.
   - `test_lobpcg_largest_via_negation` — LARGEST path via the Day 10 neg-op adapter; eigenvalues match Lanczos LARGEST to 1e-8.
2. Negative-path coverage:
   - `test_lobpcg_block_size_zero_defaults` — `opts->block_size == 0` → effective block_size = k.
   - `test_lobpcg_block_size_too_small` — `opts->block_size < k` returns `SPARSE_ERR_BADARG`.
   - `test_lobpcg_precond_signature_validation` — `precond != NULL && precond_ctx == NULL` should still be allowed (many preconditioners carry state via globals); only reject `precond_ctx != NULL && precond == NULL` as the obvious user error.
3. Run the `bench_eigs --sweep default --csv` capture (from Day 11); commit the output as `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt` (rename to `.txt` for consistency with Sprint 20's `bench_day13_lanczos.txt` naming).  Also commit the `--compare` pivot as a companion file `bench_day14_compare.txt`.
4. Run `make format && make lint && make test && make sanitize && make sanitize-thread && make bench && make bench-eigs` — all clean.

### Deliverables
- `tests/test_eigs_lobpcg.c` with 6+ positive-path + 3+ negative-path tests
- Benchmark captures: `bench_day14.txt` (full sweep) and `bench_day14_compare.txt` (three-backend pivot)

### Completion Criteria
- All LOBPCG tests pass; cross-backend parity within 1e-8 on every corpus fixture
- Preconditioning speedup claim verified numerically, not just by inspection
- `make format && make lint && make test && make sanitize && make sanitize-thread && make bench && make bench-eigs` clean

---

## Day 14: Documentation, `docs/algorithm.md`, PROJECT_PLAN Update & Retrospective

**Theme:** Write the thick-restart + OpenMP + LOBPCG documentation, extend the customer-facing `examples/example_eigs.c`, update project docs, and close Sprint 21 with a retrospective

**Time estimate:** 10 hours

### Tasks
1. `README.md` updates (2 hrs):
   - Extend the Sprint 20 "Sparse symmetric eigensolver" subsection with: true thick-restart Lanczos with bounded `O((k + m_restart) · n)` memory, OpenMP reorthogonalization scaling, LOBPCG preconditioned block solver.
   - Mention `bench_eigs` as a permanent tool; point at `bench_day14.txt` for the measured numbers.
   - Short prose on when to pick which backend (the Day 10 AUTO decision tree in plain language).
2. `include/sparse_eigs.h` doxygen refresh (1 hr):
   - Document the new `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` and `SPARSE_EIGS_BACKEND_LOBPCG` enum values.
   - Document `opts->block_size`, `opts->precond`, `opts->precond_ctx`, `opts->lobpcg_soft_lock`, `opts->m_restart`.
   - Document `result.peak_basis_size` and `result.backend_used` output fields (added this sprint).
   - Cross-reference `sparse_ic.h` and `sparse_ldlt.h` for preconditioner construction.
3. `docs/algorithm.md` Eigensolver section extension (2 hrs):
   - Subsection on the Wu/Simon arrowhead state: the arrowhead shape, the Givens reduction to tridiagonal, the locked-Ritz-pair preservation proof sketch.  Reference Wu/Simon (2000) and Stathopoulos/Saad (2007).
   - Subsection on LOBPCG: the three-block subspace (X, W, P), the block Rayleigh-Ritz step, Knyazev's convergence argument at a high level, the preconditioning role.  Reference Knyazev (2001).
   - Subsection on the OpenMP reorth strategy: MGS stays serial in j, parallelises in i; why classical Gram-Schmidt is *not* the tempting alternative despite easier j-parallelisation.
4. `examples/example_eigs.c` extension (1 hr):
   - Add a third demo: LOBPCG with IC(0) preconditioning on a moderate SPD, showing the preconditioning speedup.
   - Add a fourth demo (optional, cut if time-pressed): thick-restart on a large-n fixture with the memory-bound benchmark printed inline.
5. `docs/planning/EPIC_2/PROJECT_PLAN.md` update (0.5 hr): mark Sprint 21 items 1–5 as complete; update the Summary table (Sprint 21 → **Complete**, actual hours).
6. Retrospective (2.5 hrs): `docs/planning/EPIC_2/SPRINT_21/RETROSPECTIVE.md` mirroring the Sprint 20 template — DoD checklist against the 5 PROJECT_PLAN.md items, final metrics (assertion count, test count delta, thick-restart memory savings on bcsstk14, OpenMP scaling ratio at 4 threads, LOBPCG iteration-count ratio vs Lanczos), what went well / didn't, items deferred with rationale, lessons learned (especially any surprises in the arrowhead reduction or LOBPCG preconditioning on real fixtures).
7. Final regression (1 hr):
   - `make clean && make format && make lint && make test && make sanitize && make sanitize-thread && make bench && make bench-eigs && make examples` — all clean.
   - Verify total test count grew by at least 20 from the Sprint 20 baseline.
   - Verify `bench_day14.txt` committed and referenced from README.

### Deliverables
- `README.md` + `docs/algorithm.md` refreshed with thick-restart + OpenMP + LOBPCG content
- `examples/example_eigs.c` extended with the LOBPCG preconditioned demo
- `include/sparse_eigs.h` doxygen finalised (new enums, opts fields, result fields)
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 21 marked **Complete**
- `docs/planning/EPIC_2/SPRINT_21/RETROSPECTIVE.md` with full DoD + metrics + deferrals

### Completion Criteria
- Every item (1–5) in the Sprint 21 PROJECT_PLAN.md table is marked ✅ or ⚠️ with explicit deferral rationale
- `make clean && make format && make lint && make test && make sanitize && make sanitize-thread && make bench && make bench-eigs && make examples` all clean
- `examples/example_eigs.c` runs end-to-end including the LOBPCG demo

---

## Sprint 21 Summary Table

| Day | Title | Hours |
|-----|-------|------:|
| 1   | Thick-Restart Lanczos — Design & Arrowhead State | 10 |
| 2   | Thick-Restart Lanczos — Arrowhead Reduction & Ritz Locking | 10 |
| 3   | Thick-Restart Lanczos — Phase Execution & Restart Loop | 10 |
| 4   | Thick-Restart Lanczos — Memory-Bounded Convergence & Parity Tests | 10 |
| 5   | OpenMP Lanczos Reorthogonalization | 10 |
| 6   | OpenMP Lanczos Benchmarks, Scaling & Tuning | 10 |
| 7   | LOBPCG — API Design & Block Rayleigh-Ritz Infrastructure | 10 |
| 8   | LOBPCG — Vanilla Rayleigh-Ritz Core (No Preconditioning) | 10 |
| 9   | LOBPCG — Preconditioning & Block Direction Update | 9 |
| 10  | LOBPCG — AUTO Dispatch Threshold, Smallest/Sigma Coverage, Integration | 8 |
| 11  | `benchmarks/bench_eigs.c` — Permanent Driver with CSV + Sweep + Compare | 10 |
| 12  | Thick-Restart Tests & SuiteSparse Memory-Bounded Regression | 8 |
| 13  | LOBPCG Tests & Benchmark Capture | 8 |
| 14  | Documentation, `docs/algorithm.md`, PROJECT_PLAN Update & Retrospective | 10 |
| **Total** | | **133** |

Day cap: 12 hours (unused — max is 10 hrs).  Sprint total 133 hrs vs PROJECT_PLAN.md estimate 124 hrs — a ~9-hour buffer for the arrowhead state novelty and LOBPCG convergence tuning, well within the 168-hour (14 × 12) hard ceiling.
