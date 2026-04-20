# Sprint 19 Plan: CSC Kernel Tuning & Native Supernodal LDL^T

**Sprint Duration:** 14 days
**Goal:** Close out the Sprint 18 CSC follow-ups surfaced in `docs/planning/EPIC_2/SPRINT_18/RETROSPECTIVE.md`: quantify the analyze-once / factor-many speedup the Sprint 17 + Sprint 18 `PERF_NOTES.md` hypothesise, characterise `SPARSE_CSC_THRESHOLD`'s crossover with sub-100 fixtures, fix the scalar-CSC regression on Kuu-like fill patterns, restore the LDL^T scalar kernel's sparse-row scaling by adding a row-adjacency index, and extend the Sprint 18 batched supernodal path from Cholesky to symmetric indefinite LDL^T.

**Starting Point:** Sprint 18 shipped the native CSC Bunch-Kaufman LDL^T kernel, the batched supernodal Cholesky path, transparent dispatch in `sparse_cholesky_factor_opts`, and a larger SuiteSparse corpus (bcsstk14 / s3rmt3m3 / Kuu / Pres_Poisson on the SPD side; bcsstk14 / s3rmt3m3 on the LDL^T side).  The retrospective flagged five follow-ups that this sprint closes out:

- **Analyze-once / factor-many workflow is still unmeasured.**  Sprint 17 and Sprint 18 `PERF_NOTES.md` both hypothesise a larger CSC speedup in the multi-refactor workflow (AMD amortises across N refactors) but neither sprint ran the benchmark.  No `benchmarks/bench_refactor_csc.c` exists.
- **`SPARSE_CSC_THRESHOLD` sits at 100 with no sub-100 data.**  The Day 12 plan forbade moving the threshold without benchmark data; the enlarged corpus has no matrix below n = 100.  The small-matrix study (n ∈ {20, 40, 60, 80}) is the missing measurement.
- **Scalar-CSC regresses to 0.77× on Kuu (n = 7102).**  The scalar kernel's repeated `shift_columns_right_of` calls during drop-tolerance pruning are the dominant cost on this matrix family; every other fixture in the Sprint 18 corpus is monotonic in n-vs-speedup.  The supernodal path sidesteps the problem via full `sym_L` pre-allocation, so the regression isn't user-visible through `sparse_cholesky_factor_opts`, but the scalar kernel should still be restored to the monotonic trend.
- **`ldlt_csc_cmod_unified` Phase A does O(step_k · log nnz) per step.**  The linked-list reference (`acc_schur_col` in `src/sparse_ldlt.c`) iterates only over prior columns where `L(col, j) != 0` via the cross-linked row list, so the CSC kernel loses the reference's sparse-row scaling.  Flagged by Copilot during PR #26 review.
- **LDL^T has a native column-by-column kernel but no batched supernodal path.**  The Cholesky side got one in Sprint 18 Days 6-10 and delivers 2.2–4.4× on larger SPD fixtures; the LDL^T side runs column-by-column.  Extending the batched path to symmetric indefinite requires handling 2×2 pivots at supernode boundaries (a 2×2 pivot can legally split a supernode).

**End State:** `benchmarks/bench_refactor_csc.c` ships and quantifies the analyze-once speedup on the enlarged corpus; `SPARSE_CSC_THRESHOLD` is confirmed at 100 (or re-tuned) with sub-100 measurements on record; the scalar-CSC kernel's `shift_columns_right_of` regression on Kuu is resolved and the n-vs-speedup trend is monotonic; `LdltCsc` carries a row-adjacency index and `ldlt_csc_cmod_unified` iterates only the contributing prior columns; and a batched supernodal LDL^T path mirrors the Sprint 18 Cholesky helpers (`_extract` / `_writeback` / `_eliminate_diag` / `_eliminate_panel`) with a new dense LDL^T block factor primitive, producing bit-identical output to the scalar kernel and delivering a measurable speedup on bcsstk14 / s3rmt3m3.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~152 hours — about 16 hours under the 168-hour ceiling documented in `PROJECT_PLAN.md`, leaving room for per-day overruns on the more complex LDL^T supernodal work (Days 10-14) without breaching the cap.

---

## Day 1: `bench_refactor_csc` — Harness & Scaffolding

**Theme:** Build the dedicated analyze-once / factor-many benchmark so Day 2 can run it across the Sprint 18 corpus

**Time estimate:** 10 hours

### Tasks
1. Design block at the top of a new `benchmarks/bench_refactor_csc.c` explaining the workflow difference vs the Sprint 17/18 one-shot `bench_chol_csc`:
   - One-shot workflow: `sparse_cholesky_factor_opts(AMD)` per call — AMD reordering runs every time, linear scaling with n refactors.
   - Analyze-once workflow: `sparse_analyze` once → N × `sparse_refactor_numeric` with perturbed values but identical pattern. AMD amortises.  CSC speedup is hypothesised larger because only the numeric factor cost remains in the comparison.
2. Implement the harness:
   - CLI mirrors `bench_chol_csc`: `--repeat R` for repeats, positional arg for a single-matrix run, default corpus pulled from `tests/data/suitesparse/`.
   - For each matrix: run one `sparse_analyze` + AMD (outside the timed region); then time N = 10 refactors with small random perturbations to A's values (keep pattern identical so `sparse_refactor_numeric` applies).
   - Two backend paths timed side-by-side: linked-list (via `sparse_ldlt_factor_opts` / `sparse_cholesky_factor_opts` with `SPARSE_CHOL_BACKEND_LINKED_LIST`) and CSC (supernodal, `SPARSE_CHOL_BACKEND_CSC`).
   - CSV output: `matrix, n, nnz, analyze_ms, refactor_ll_ms, refactor_csc_ms, solve_ll_ms, solve_csc_ms, speedup_refactor, res_ll, res_csc`.
3. Wire into the Makefile `BENCH_SRCS` list and into `CMakeLists.txt`' s `add_executable` alongside the existing bench binaries.
4. Small-matrix smoke run (nos4 + bcsstk04) to confirm the harness produces sensible numbers (non-zero refactor times, residuals ≤ 2e-13).
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- `benchmarks/bench_refactor_csc.c` building under both Makefile and CMake
- CLI parity with `bench_chol_csc` (`--repeat`, positional path, `--help`)
- Smoke-run CSV on nos4 + bcsstk04 produces sensible numbers

### Completion Criteria
- `./build/bench_refactor_csc --repeat 3` runs to completion on the default corpus without crashing
- CSV header + two data rows (nos4, bcsstk04) readable by `column -t`
- `make format && make lint && make test` clean

---

## Day 2: `bench_refactor_csc` — Corpus Run & `PERF_NOTES.md` Update

**Theme:** Run the analyze-once benchmark on the full Sprint 18 corpus and land the measurement into `PERF_NOTES.md` alongside the one-shot numbers

**Time estimate:** 10 hours

### Tasks
1. Full corpus run: `./build/bench_refactor_csc --repeat 5` on nos4, bcsstk04, bcsstk14, s3rmt3m3, Kuu, Pres_Poisson.  Capture to `docs/planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt` for reference.
2. Cross-check residuals: for each matrix, verify the final refactor produces `||A·x - b||_∞ / ||b||_∞ ≤ 1e-10`.  Any matrix above the threshold is a regression — stop and investigate before writing up.
3. Extend `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`:
   - New section **Analyze-once / factor-many — CSC vs linked-list** with the same column structure as the existing one-shot Cholesky table (matrix, n, nnz, refactor_ll, refactor_csc, speedup_refactor, residuals).
   - Short narrative: is the analyze-once speedup larger than the one-shot speedup on every matrix?  On which matrices is the gap biggest?  The Sprint 17 hypothesis is that AMD dominates the one-shot cost for small matrices, so the gap should widen there; confirm or disconfirm with the data.
   - If the hypothesis disconfirms (analyze-once speedup ≤ one-shot speedup), include a 1-2 paragraph analysis of why (e.g., `sparse_refactor_numeric` doing more work than expected).
4. Link the new section from the `docs/algorithm.md` Performance section (a single sentence pointer) and from the sprint retrospective section (Day 14).
5. Run `make format && make lint && make test && make bench` — all clean.  The new `bench_refactor_csc` participates in `make bench` going forward.

### Deliverables
- `bench_day2_refactor.txt` raw capture committed under `SPRINT_19/`
- `PERF_NOTES.md` extended with the analyze-once table + narrative
- `docs/algorithm.md` cross-link

### Completion Criteria
- Every matrix in the corpus produces a CSV row (no failures)
- Residuals across all runs ≤ 1e-10
- Hypothesis either confirmed (table shows larger speedup in analyze-once) or disconfirmed-with-analysis (narrative explains why)
- `make format && make lint && make test && make bench` clean

---

## Day 3: Small-Matrix Corpus — Synthetic Fixtures & Bench Integration

**Theme:** Add n ∈ {20, 40, 60, 80} fixtures below the current `SPARSE_CSC_THRESHOLD` so Day 4 has the data to retune (or confirm) the threshold

**Time estimate:** 10 hours

### Tasks
1. Decide fixture types by consulting `docs/algorithm.md` for the n-vs-speedup trend signatures we expect to see:
   - Four tridiagonal SPDs at n ∈ {20, 40, 60, 80} (minimal fill, stress pointer-chasing overhead).
   - Four random banded SPDs at the same n with bandwidth 3-5 (moderate fill, stresses the scalar kernel's `shift_columns_right_of`).
   - Two random dense SPDs at n ∈ {20, 60} (max fill, stresses the supernodal path's detection overhead on small supernodes).
2. Add a small-matrix corpus mode to `bench_chol_csc`:
   - New CLI flag `--small-corpus` that substitutes the default matrix list with the 10 synthetic fixtures above.
   - Fixtures generated in-memory via a seeded RNG (document the seed in the header comment) so reproducibility doesn't depend on an on-disk `.mtx`.
   - CSV row format unchanged — the "matrix" column becomes `tridiag-20`, `banded-40`, `dense-20` etc.
3. Smoke-run `./build/bench_chol_csc --small-corpus --repeat 3` and verify residuals ≤ 2e-13 across all 10 fixtures.
4. Capture baseline numbers (with the current SPARSE_CSC_THRESHOLD = 100) to `docs/planning/EPIC_2/SPRINT_19/bench_day3_small_corpus.txt`.
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- 10 synthetic SPD fixtures generated in-memory from a seeded RNG
- `bench_chol_csc --small-corpus` CLI flag
- Baseline CSV capture on the small-matrix corpus

### Completion Criteria
- Every fixture factors successfully through all three backend paths (linked-list, CSC scalar, CSC supernodal)
- Residuals ≤ 2e-13 across all fixtures
- `make format && make lint && make test` clean

---

## Day 4: `SPARSE_CSC_THRESHOLD` Retrospective — Analysis & Decision

**Theme:** Map the crossover n from Day 3's numbers and either confirm the default at 100 or re-tune

**Time estimate:** 10 hours

### Tasks
1. Plot (in ASCII via `column -t` / `gnuplot` if available) the n-vs-speedup curves from `bench_day3_small_corpus.txt` for each fixture family (tridiag, banded, dense).  Look for the n where `speedup_csc_sn > 1.0` crosses over from below.
2. Decision matrix:
   - If the crossover is at n ≤ 50 on every family, the current default of 100 is conservative — consider lowering to 60 or 80.
   - If the crossover is at n ≥ 100 on every family, the current default is aggressive — consider raising.
   - If families disagree (e.g., tridiag crosses at 30 but dense at 90), the current default of 100 is the conservative worst-case — keep it.
3. Update `include/sparse_matrix.h`'s `SPARSE_CSC_THRESHOLD` macro:
   - If the data supports a change, update the default value and refresh the doc comment with the measurement that justifies it (n, fixture family, speedup).
   - If the data confirms 100, remove the "pending small-matrix study" hedge from the current doc comment and replace with the measured data.
4. Update `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` "Threshold guidance" section:
   - Remove the Sprint 18 Day 13 "deferred to Sprint 19" language.
   - Replace with a table showing the crossover data across the three families.
   - Document the final decision (keep at 100 / raise / lower) with rationale.
5. If the threshold changed: re-run `make test` to confirm no tests hard-code the old value (any that do should use `SPARSE_CSC_THRESHOLD` via `include/sparse_matrix.h`).
6. Run `make format && make lint && make test && make bench` — all clean.

### Deliverables
- Decision: `SPARSE_CSC_THRESHOLD` default either confirmed at 100 or re-tuned to a new value with supporting measurement
- `PERF_NOTES.md` "Threshold guidance" section updated with the small-matrix study results
- `include/sparse_matrix.h` doc comment refreshed

### Completion Criteria
- `PERF_NOTES.md` now contains crossover data rather than a "pending" note
- `SPARSE_CSC_THRESHOLD` documented value matches measured crossover (± a small safety margin)
- `make format && make lint && make test && make bench` clean

---

## Day 5: Kuu Regression — Profile & Design

**Theme:** Confirm `shift_columns_right_of` is the dominant cost on Kuu's scalar path and decide which of the two architecturally-legitimate fixes to apply

**Time estimate:** 10 hours

### Tasks
1. Profile `./build/bench_chol_csc tests/data/suitesparse/Kuu.mtx --repeat 5` under `time` / `perf stat` / macOS `sample` to confirm scalar CSC at 0.77× vs linked-list is dominated by `shift_columns_right_of` memmoves, not something else (e.g., drop-tolerance threshold miscalibration).  Capture profile output to `docs/planning/EPIC_2/SPRINT_19/profile_day5_kuu.txt`.
2. Sanity-check the hypothesis: count the number of `shift_columns_right_of` calls during scalar elimination of Kuu (instrument locally, measure, then revert the instrumentation). Compare against the much smaller fixtures (bcsstk04, bcsstk14) to confirm the per-call cost × call count explains the regression.
3. Evaluate the two fix options:
   - **Option A — pre-allocate full `sym_L` in `chol_csc_from_sparse`:** Match `chol_csc_from_sparse_with_analysis` by computing the symbolic L pattern upfront (costs one extra `sparse_analyze`-equivalent walk of A) and skipping the dynamic column growth entirely.  Simplifies the scalar kernel but costs memory on small matrices.
   - **Option B — batch the shifts inside `chol_csc_gather`:** Accumulate the delta sizes for multiple columns and issue a single bulk memmove at the end of a batch.  Keeps the dynamic-growth path but amortises the memmove cost.
4. Decision rule: pick Option A if the small-matrix corpus from Day 3 shows minimal memory overhead on the sub-100 fixtures; Option B if the memory cost is significant.  Write the decision into `docs/planning/EPIC_2/SPRINT_19/kuu_fix_decision.md` with the supporting numbers.
5. Sketch the implementation plan for the chosen option as a comment block in `src/sparse_chol_csc.c` near the function that will change (`chol_csc_from_sparse` for Option A, `chol_csc_gather` for Option B).
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- Profile data confirming `shift_columns_right_of` as the dominant cost on Kuu
- `kuu_fix_decision.md` with the selected option and rationale
- Sketch comment in `src/sparse_chol_csc.c` outlining the implementation

### Completion Criteria
- Profile data shows `shift_columns_right_of` (or the function it calls) dominating scalar Kuu factor time
- Decision doc cites specific memory / speedup numbers rather than qualitative reasoning
- `make format && make lint && make test` clean

---

## Day 6: Kuu Regression — Implementation

**Theme:** Land the chosen fix with tests covering both the Kuu-style regression case and the small-matrix memory path

**Time estimate:** 12 hours

### Tasks
1. Implement the fix from Day 5's decision doc.  Rough scope (Option A — full sym_L pre-alloc):
   - Add an internal helper `chol_csc_symbolic_pattern(const SparseMatrix *mat, idx_t **col_ptr_out, idx_t **row_idx_out, idx_t *nnz_out)` that runs the equivalent of `sparse_analyze`'s symbolic phase but without the full `sparse_analysis_t` wrapper (so `chol_csc_from_sparse` callers don't need to manage analysis lifetime).
   - Modify `chol_csc_from_sparse` to call the helper, pre-allocate with the full sym_L pattern, and distribute A's lower-triangle values into their matching positions (same pattern as `_with_analysis`).
   - Drop the dynamic `shift_columns_right_of` invocations from `chol_csc_gather` — columns now have their final pattern pre-allocated with zero-valued fill slots, and gather just writes into existing slots.
2. Alternative scope (Option B — batched shifts): implement the batch-shift logic inside `chol_csc_gather` and update the drop-tolerance bookkeeping to amortise memmoves.
3. Regression tests:
   - New test `test_chol_csc_kuu_scalar_no_regression`: factor Kuu through `chol_csc_eliminate` and assert residual ≤ 1e-10 AND nnz(L) matches (to within drop-tolerance grain) the current supernodal path's output on the same matrix.
   - Existing `test_chol_csc_eliminate_*` tests must all pass unchanged — the behavioural contract is unchanged, only the internal storage path differs.
4. Re-run `./build/bench_chol_csc tests/data/suitesparse/Kuu.mtx --repeat 5` and confirm scalar CSC now beats linked-list (target: > 1.5× on Kuu, matching the monotonic trend from the other corpus fixtures).
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `chol_csc_from_sparse` (or `chol_csc_gather`) updated with the Day 5 fix
- `test_chol_csc_kuu_scalar_no_regression` test passing
- Before/after Kuu bench numbers showing scalar CSC back above 1.5×

### Completion Criteria
- Scalar CSC factor time on Kuu drops to < 0.7 × linked-list factor time (speedup > 1.5×)
- Every existing `test_chol_csc` test passes (no regression)
- `make format && make lint && make test && make sanitize` clean

---

## Day 7: Kuu Regression — Full Corpus Bench & Narrative Update

**Theme:** Verify the fix across the entire Sprint 18 corpus (not just Kuu) and update `PERF_NOTES.md`

**Time estimate:** 10 hours

### Tasks
1. Full-corpus `./build/bench_chol_csc --repeat 3` run comparing scalar CSC speedups before and after the Day 6 fix.  Capture to `docs/planning/EPIC_2/SPRINT_19/bench_day7_post_kuu.txt`.
2. Regression sweep:
   - For each matrix in {nos4, bcsstk04, bcsstk14, s3rmt3m3, Kuu, Pres_Poisson}, compute the delta `speedup_csc_post - speedup_csc_pre` from Sprint 18 Day 14's `bench_day14.txt` vs today's run.
   - Any matrix where the delta is negative (regression) blocks the fix — investigate and revert / tune.  All matrices should either stay flat or improve.
3. Update `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`:
   - Scalar-CSC row for Kuu: replace the 0.77× with the new speedup.
   - New paragraph in the "Observations" block documenting the fix (`chol_csc_from_sparse` pre-populates the full `sym_L` pattern to match `_with_analysis`, eliminating the `shift_columns_right_of` memmoves).
   - Mark the Sprint 19 deferred item as resolved; the "scalar-CSC drop-tolerance shift cost" bullet in the Sprint 18 retrospective's Sprint 19 deferral list moves to "completed" in the Sprint 19 retrospective (Day 14).
4. If the supernodal path's output on Kuu changes (unlikely — the fix is scalar-only), re-verify the `test_dispatch_day12_bcsstk14_residual` test and its siblings.
5. Memory check: for the small-matrix corpus from Day 3, measure the peak RSS via `/usr/bin/time -l` on `bench_chol_csc --small-corpus` before and after the fix.  If Option A was chosen (pre-allocate sym_L), the memory cost should be visible but bounded — document the overhead.
6. Run `make format && make lint && make test && make sanitize && make bench` — all clean.

### Deliverables
- Full-corpus CSV capture before/after the fix
- `PERF_NOTES.md` Kuu row and observations paragraph updated
- Small-matrix memory overhead documented

### Completion Criteria
- No regression on any corpus matrix (every speedup flat or improved)
- Kuu scalar CSC speedup on the monotonic trend with the other fixtures
- `make format && make lint && make test && make sanitize && make bench` clean

---

## Day 8: LDL^T Row-Adjacency Index — Data Structure & Lifecycle

**Theme:** Add the row-adjacency index to `LdltCsc` so Day 9 can rewrite `ldlt_csc_cmod_unified` to iterate only contributing prior columns

**Time estimate:** 10 hours

### Tasks
1. Design block at the top of `src/sparse_ldlt_csc.c` explaining why the index is needed:
   - `ldlt_csc_cmod_unified` Phase A currently iterates `kp = 0..step_k-1` with a binary search per `kp` to find `L(col, kp)`.  That's O(step_k · log nnz) per elimination step.
   - The linked-list reference (`acc_schur_col` in `src/sparse_ldlt.c`) iterates only over the columns where `L(col, j) != 0` via the cross-linked row list on the `SparseMatrix` — native sparse-row scaling.
   - Add an auxiliary `row_adj` structure to `LdltCsc` that maps each row `r` to the list of prior columns `kp < r` where `L(r, kp)` is stored.  Populate incrementally as columns factor; rewrite Phase A (and Phase B for 2×2 pivots) to iterate the list instead of `[0, step_k)`.
2. Data structure choice:
   - Option A — per-row dynamic array (`idx_t **row_adj; idx_t *row_adj_count; idx_t *row_adj_cap`).  Simple, cache-friendly for small rows, but requires per-row growth logic.
   - Option B — global CSC transpose maintained alongside the primary CSC (`row_adj_col_ptr[n+1], row_adj_row_idx[]`).  Cache-friendly for global sweeps, but every column factor requires updating n-ish row pointers.
   - Pick Option A (the linked-list reference's behaviour is closest to per-row storage, and per-row growth is bounded by n · avg_fill which is < 2× the L capacity we already pay for).  Document the rejection of Option B in the design block.
3. Extend `LdltCsc`:
   - New fields `idx_t **row_adj;` (length n), `idx_t *row_adj_count;` (length n), `idx_t *row_adj_cap;` (length n).
   - Alloc hook in `ldlt_csc_alloc`: zero the three arrays.
   - Free hook in `ldlt_csc_free`: walk and free each non-null `row_adj[r]`, then free the three top-level arrays.
4. Incremental population helper `ldlt_csc_row_adj_append(LdltCsc *F, idx_t row, idx_t col)`:
   - Append `col` to `row_adj[row]`, growing the per-row array geometrically (2×) when capacity is hit.
   - Guard allocation overflow: `row_adj_cap[row] * sizeof(idx_t)` must not wrap `size_t` (same pattern as `chol_csc_grow`).
5. Unit tests (no kernel changes yet — just test the data structure):
   - Alloc / free round-trip with n = 10, zero rows filled.
   - Append 3 columns to row 5, verify `row_adj[5] = {0, 2, 4}` in insertion order.
   - Geometric growth stress: append 100 columns to a single row, verify contents and final `row_adj_cap[r] >= 100`.
6. Run `make format && make lint && make test` — all clean.

### Deliverables
- `LdltCsc` carries `row_adj` / `row_adj_count` / `row_adj_cap` fields
- Alloc / free / append helpers with overflow-safe growth
- 3+ unit tests covering the data structure in isolation

### Completion Criteria
- Append → read round-trip preserves insertion order
- Free releases all per-row allocations (valgrind / ASan clean)
- `make format && make lint && make test` clean

---

## Day 9: LDL^T Row-Adjacency Index — Wire Into `ldlt_csc_cmod_unified`

**Theme:** Rewrite Phase A and Phase B to iterate the row-adjacency list instead of `[0, step_k)`; verify bit-identical output

**Time estimate:** 10 hours

### Tasks
1. Modify `ldlt_csc_eliminate_native` to populate the row-adjacency index as columns factor:
   - After each column `k` is fully factored and written back to CSC, walk its stored row indices and call `ldlt_csc_row_adj_append(F, row, k)` for each.
   - Populate happens once per column; linked-list reference does the same via `sparse_insert`'s row/col chain maintenance.
2. Rewrite `ldlt_csc_cmod_unified` Phase A:
   - Replace the `for (idx_t kp = 0; kp < step_k; kp++)` + `ldlt_csc_lookup_Lrc` scan with `for (idx_t idx = 0; idx < F->row_adj_count[col]; idx++) { idx_t kp = F->row_adj[col][idx]; ... }`.
   - The inner logic (fetching `L(col, kp)`, applying the per-column contribution) stays the same; only the outer iteration changes.
   - Preserve the `kp < step_k` guard inside the loop — `row_adj[col]` may contain columns ≥ step_k if the index tracked a post-factor row (shouldn't happen, but defence-in-depth).
3. Rewrite Phase B (2×2 cross-term correction) analogously:
   - For each `kp` in `row_adj[col]`, if `pivot_size[kp] == 2` and `kp + 1 < step_k`, apply the 2×2 cross-term update.  The existing `while (kp + 1 < step_k) { ... kp += 1 or 2; }` loop structure rewrites to an index-into-row_adj walk with the same conditional logic.
4. Audit `ldlt_csc_scatter_symmetric` (the symmetric-swap helper): does it also walk the factored prefix?  If yes, apply the same row-adj traversal; if no, leave it alone and document the invariant in a comment.
5. Cross-check tests:
   - Every existing `test_ldlt_csc_eliminate_*` test must pass — the behavioural contract is bit-identical output.
   - New `test_ldlt_csc_row_adj_matches_reference` test: factor a 20-matrix random-indefinite corpus via the native kernel (with row-adj) and via the Sprint 17 wrapper (via `ldlt_csc_set_kernel_override`); assert L / D / D_offdiag / pivot_size / perm all match bit-for-bit.
6. Benchmark: `./build/bench_ldlt_csc --repeat 3` — capture to `docs/planning/EPIC_2/SPRINT_19/bench_day9_row_adj.txt` and compare against Sprint 18 Day 14's `bench_day14.txt`.  Speedup should be visible on matrices with non-dense factored rows (bcsstk14, s3rmt3m3).
7. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `ldlt_csc_cmod_unified` Phase A + Phase B iterate `row_adj[col]` instead of `[0, step_k)`
- Population hook in `ldlt_csc_eliminate_native` maintains the index
- New 20-matrix cross-check test confirms bit-identical output
- Benchmark showing speedup on bcsstk14 / s3rmt3m3

### Completion Criteria
- Every existing `test_ldlt_csc_*` test passes unchanged
- Cross-check finds zero divergences on the 20-matrix corpus
- `bench_ldlt_csc` shows measurable speedup on bcsstk14 and s3rmt3m3
- `make format && make lint && make test && make sanitize` clean

---

## Day 10: Native Supernodal LDL^T — Design & Supernode Detection

**Theme:** Design the batched LDL^T supernodal path and extend `chol_csc_detect_supernodes` to respect 2×2 pivot boundaries

**Time estimate:** 12 hours

### Tasks
1. Design block at the top of `src/sparse_ldlt_csc.c` documenting the batched LDL^T supernodal approach:
   - Mirror of Sprint 18 Days 6-10 Cholesky batched supernodal: extract → eliminate_diag → eliminate_panel → writeback.
   - Key difference: LDL^T uses Bunch-Kaufman pivoting with 1×1 and 2×2 pivot blocks.  A 2×2 pivot spans two columns (k, k+1), and those two columns **must not be split** across supernode boundaries — otherwise the batched eliminate_diag would factor the first column without seeing its pivot partner.
   - The Liu-Ng-Peyton detection therefore needs an extra condition: if column k is part of a 2×2 pivot with column k+1, and the current supernode ends at k, extend the supernode by one column to include k+1.
2. Implement `ldlt_csc_detect_supernodes(const LdltCsc *F, idx_t min_size, idx_t *super_starts, idx_t *super_sizes, idx_t *count)`:
   - Copy the three-condition Liu-Ng-Peyton check from `chol_csc_detect_supernodes` as a starting point (fundamental supernode conditions on sorted CSC columns).
   - Add a fourth condition: a supernode boundary at column k is only valid if `pivot_size[k] == 1` OR `k + 1` is outside the current supernode's extension range.  Equivalently: 2×2 pivots are atomic; a supernode cannot start or end mid-pivot.
   - Since `pivot_size` is known from a prior factor run, `ldlt_csc_detect_supernodes` is called AFTER `ldlt_csc_eliminate_native` runs once to produce `pivot_size`, then the batched path factors a subsequent refactor.  Document this two-pass model in the design block and note that the first factor uses the scalar kernel; only refactors use the batched path.
3. Alternative design (if pre-known pivot_size is too restrictive): detect supernodes during factoring, using a "rollback" mechanism when a 2×2 pivot crosses a tentative supernode boundary.  This is more complex; document both designs and pick one.  Recommended: pre-known pivot_size (simpler, matches Sprint 18 Cholesky's two-pass analysis + factor model).
4. Unit tests for `ldlt_csc_detect_supernodes`:
   - Dense 6×6 matrix with all 1×1 pivots: one supernode covering [0, 6) of size 6.
   - Dense 6×6 with a 2×2 pivot at (2, 3): either one supernode [0, 6) or two supernodes [0, 4) + [4, 6), depending on detection rules.  Verify the detection respects the 2×2 boundary.
   - Block-diagonal 8×8 with two 4×4 blocks, 2×2 pivot in each block: two supernodes with correct 2×2-aware boundaries.
5. Run `make format && make lint && make test` — all clean.

### Deliverables
- Design block in `src/sparse_ldlt_csc.c` covering batched LDL^T + 2×2 boundary handling
- `ldlt_csc_detect_supernodes` implementation
- 3+ unit tests on canonical LDL^T pivot patterns

### Completion Criteria
- Every detected supernode has `pivot_size[k] == 1 || (k+1 in supernode)` at both its start and end
- All unit tests pass
- `make format && make lint && make test` clean

---

## Day 11: Native Supernodal LDL^T — Dense LDL^T Block Factor Primitive

**Theme:** Build the column-major dense LDL^T-with-Bunch-Kaufman primitive that the batched eliminate_diag will call per supernode

**Time estimate:** 12 hours

### Tasks
1. Add `ldlt_dense_factor(double *A, double *D, double *D_offdiag, int8_t *pivot_size, idx_t n, idx_t lda, double tol, double *elem_growth_out)` alongside the existing `chol_dense_factor` in `src/sparse_chol_csc.c` (or a new `src/sparse_ldlt_dense.c` if the Cholesky file grows too large).
2. Implementation mirrors the reference column-by-column BK in `src/sparse_ldlt.c`:
   - For k = 0 .. n-1: pick a 1×1 or 2×2 pivot from column k (and optionally k+1 for the 2×2 case) using the four-criteria BK rule.
   - Apply symmetric row/column swaps on the dense buffer to move the chosen pivot into the (k, k) or ((k, k+1), (k, k+1)) slot.
   - Eliminate column k (or columns k, k+1 for 2×2): compute L(i, k) = A(i, k) / D(k) for 1×1, or the 2×2 block solve for 2×2, and update the trailing submatrix via rank-1 (1×1) or rank-2 (2×2) Schur complement.
   - Track element growth: `max |L(i, k)| / max |A(:, k)|` per step; return the maximum growth factor so the caller can detect instability.
3. Key difference from `chol_dense_factor`:
   - LDL^T doesn't take sqrt — `D(k)` holds the pivot scalar (or the 2×2 block's diagonal for 2×2 pivots), not sqrt(pivot).
   - `pivot_size[k]` output array lets the caller reconstruct the block structure; for 2×2 pivots `pivot_size[k] == pivot_size[k+1] == 2`.
   - `D_offdiag[k]` holds the off-diagonal of the 2×2 block for 2×2 pivots; zero for 1×1.
4. Unit tests in `tests/test_chol_csc.c` (or a new `tests/test_ldlt_dense.c`):
   - `ldlt_dense_factor` on a dense 4×4 indefinite: verify L·D·L^T reconstructs A to round-off.
   - 2×2 pivot forced via careful eigenvalue placement: verify `pivot_size[0] == pivot_size[1] == 2`, `D_offdiag[0] != 0`.
   - Mixed 1×1 / 2×2 on a 6×6: verify the reconstruction and the growth factor is bounded.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `ldlt_dense_factor` primitive with 1×1 / 2×2 pivot support and element-growth tracking
- 3+ unit tests verifying L·D·L^T reconstruction and pivot correctness

### Completion Criteria
- `ldlt_dense_factor` output matches the linked-list reference on every test matrix to within round-off
- Element growth factor reported matches the reference (within 1 ULP)
- `make format && make lint && make test && make sanitize` clean

---

## Day 12: Native Supernodal LDL^T — Extract / Writeback Helpers

**Theme:** Build the plumbing that moves an LDL^T supernode's diagonal block + panel between packed CSC and a dense column-major buffer

**Time estimate:** 12 hours

### Tasks
1. Implement `ldlt_csc_supernode_extract(const LdltCsc *F, idx_t s_start, idx_t s_size, double *dense, idx_t lda, idx_t *row_map, idx_t *panel_height_out)`:
   - Mirror of `chol_csc_supernode_extract` (Sprint 18 Day 6) but operates on `LdltCsc`'s embedded `CholCsc *L`.
   - Walk the first column of the supernode to collect its stored rows into `row_map`; subsequent columns must share the same below-diagonal pattern (supernodal invariant, verified on debug builds).
   - Scatter each column's `L->values` into the dense column-major buffer at positions given by `row_map`.
   - Key difference from Cholesky: no `D` / `D_offdiag` extraction here — those are handled by the diagonal block factor on Day 13.
2. Implement `ldlt_csc_supernode_writeback(LdltCsc *F, idx_t s_start, idx_t s_size, const double *dense, idx_t lda, const idx_t *row_map, idx_t panel_height, double drop_tol)`:
   - Mirror of `chol_csc_supernode_writeback` (Sprint 18 Day 10) with the same per-column drop rule (`|v| < drop_tol * |D(k)|` for below-diagonal off-diagonals).
   - The diagonal block's L values go back into `F->L`; the D values computed by `ldlt_dense_factor` go into `F->D`; `D_offdiag` and `pivot_size` likewise.
3. Unit tests (no eliminate_diag yet — just round-trip):
   - Dense 6×6 LDL^T fixture with precomputed factor: extract → memcpy → writeback reproduces the input exactly.
   - Block-diagonal 8×8 (two 4×4 blocks): extract each supernode, round-trip, verify independence.
   - Mixed 1×1 / 2×2 pivot supernode: verify `D_offdiag` survives the round-trip.
4. Run `make format && make lint && make test` — all clean.

### Deliverables
- `ldlt_csc_supernode_extract` + `ldlt_csc_supernode_writeback` helpers
- 3+ round-trip tests on canonical LDL^T supernode structures

### Completion Criteria
- Extract → writeback is exactly the identity on every supernode of the test fixtures
- `ldlt_csc_validate` passes after every writeback
- `make format && make lint && make test` clean

---

## Day 13: Native Supernodal LDL^T — Eliminate_Diag & Eliminate_Panel

**Theme:** Wire the Day 11 dense primitive into the extract/writeback flow so a supernode's diagonal block is factored in one dense call and its panel via a dense triangular solve

**Time estimate:** 12 hours

### Tasks
1. Implement `ldlt_csc_supernode_eliminate_diag(LdltCsc *F, idx_t s_start, idx_t s_size, double *dense, idx_t lda, const idx_t *row_map, idx_t panel_height, double tol)`:
   - Apply external cmod from all prior columns `k < s_start` that contribute to rows in the supernode's `row_map` — same left-looking pattern as `chol_csc_supernode_eliminate_diag` but with LDL^T's 1×1/2×2 pivot block structure in the cmod expansion.
   - The row-adjacency index from Day 8 accelerates this step: for each row `r` in `row_map`, iterate `F->row_adj[r]` instead of `0..s_start-1`.
   - Call `ldlt_dense_factor` on the `s_size × s_size` diagonal slab.  Capture `pivot_size` / `D` / `D_offdiag` outputs into local scratch arrays.
   - Return `SPARSE_ERR_SINGULAR` on element-growth overflow or near-zero pivot, matching the native kernel's error semantics.
2. Implement `ldlt_csc_supernode_eliminate_panel(const double *L_diag, const double *D, const double *D_offdiag, const int8_t *pivot_size, idx_t s_size, idx_t lda_diag, double *panel, idx_t lda_panel, idx_t panel_rows)`:
   - Forward-substitute each panel row against the factored diagonal: for each row `i`, solve `L_diag · x = panel[i, :]` — same as `chol_dense_solve_lower` but accounting for the D block.
   - For a column `j` in the diagonal block with `pivot_size[j] == 1`: simple 1D division.
   - For a 2×2 block at (j, j+1) with `pivot_size[j] == 2`: 2×2 block solve using `D(j), D_offdiag(j), D(j+1)`.
3. Wire into `ldlt_csc_eliminate_supernodal(LdltCsc *F, idx_t min_size)`:
   - Interleave the batched path for detected supernodes with the scalar scatter/cmod/cdiv/gather fallback for non-supernodal columns, same as `chol_csc_eliminate_supernodal`.
   - After each batched supernode factor, populate the row-adjacency index for all its columns (so subsequent columns' cmod can use it).
4. Cross-check tests:
   - Dense 8×8 indefinite: entire matrix is one supernode — batched path is one `ldlt_dense_factor` call; residual matches reference to round-off.
   - Block-diagonal 12×12 with two 6×6 indefinite blocks: two supernodes, batched factor per block.
   - Random indefinite 30×30 with AMD reorder: compare scalar native kernel vs batched — L, D, D_offdiag, pivot_size all bit-identical.
5. Run `make format && make lint && make test && make sanitize` — all clean.

### Deliverables
- `ldlt_csc_supernode_eliminate_diag` + `ldlt_csc_supernode_eliminate_panel`
- `ldlt_csc_eliminate_supernodal` end-to-end with interleaved batched / scalar dispatch
- 3+ cross-check tests comparing batched vs scalar on canonical indefinite structures

### Completion Criteria
- Batched and scalar paths produce bit-identical factor state on every test matrix
- `SPARSE_ERR_SINGULAR` surfaces correctly on non-factorable inputs
- `make format && make lint && make test && make sanitize` clean

---

## Day 14: Integration, Benchmarks, Documentation & Retrospective

**Theme:** Wire the batched LDL^T path into the transparent dispatch (if applicable), run the full corpus benchmark, and close out Sprint 19

**Time estimate:** 12 hours

### Tasks
1. Integration tests (3 hrs):
   - New `tests/test_sprint19_integration.c` mirroring `test_sprint18_integration.c`: cross-threshold dispatch + batched-vs-scalar agreement on 10+ SPD and indefinite fixtures.  Register in `Makefile` + `CMakeLists.txt`.
   - Force-both-paths test for LDL^T: `ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE)` + `ldlt_csc_eliminate_supernodal` vs `ldlt_csc_eliminate` on the same input — solutions must agree to round-off.
2. Benchmark runs (2 hrs):
   - `./build/bench_ldlt_csc --repeat 3` on the Sprint 18 corpus (nos4, bcsstk04, bcsstk14, s3rmt3m3).  Extend `bench_ldlt_csc.c` with a `factor_csc_sn_ms` column recording the batched supernodal path's time if the user passes `--supernodal` (default: scalar, preserves existing bench behaviour).
   - `./build/bench_refactor_csc --repeat 5` re-run to confirm the Day 1-2 analyze-once numbers haven't regressed.
   - `./build/bench_chol_csc --repeat 3` re-run to confirm the Cholesky side is unchanged.
   - Capture all three to `docs/planning/EPIC_2/SPRINT_19/bench_day14.txt`.
3. Documentation cleanup (3 hrs):
   - File header of `src/sparse_ldlt_csc.c`: describe the batched supernodal path alongside the native scalar kernel and the Sprint 17 wrapper (still present for cross-checks).
   - `include/sparse_ldlt.h`: if a public LDL^T factor_opts entry point was touched, document any new options.
   - `docs/algorithm.md`: add a "Supernodal LDL^T" subsection mirroring the Cholesky one from Sprint 18 Day 13, pointing at `bench_day14.txt` for current speedups.
   - `README.md`: refresh the CSC LDL^T performance table if the numbers on bcsstk14 / s3rmt3m3 moved.
   - `docs/planning/EPIC_2/PROJECT_PLAN.md`: mark Sprint 19 items 1-5 as complete and update the Summary table (Sprint 19 → **Complete**, actual hours if they diverged from the 168-hour estimate).
4. Retrospective (3 hrs):
   - Write `docs/planning/EPIC_2/SPRINT_19/RETROSPECTIVE.md` mirroring the Sprint 18 retrospective structure: DoD checklist against the 5 PROJECT_PLAN.md items, final metrics, benchmark deltas (Sprint 18 end → Sprint 19 end), what went well / didn't, items deferred with rationale, lessons for future "scalar → batched" migrations (drawing on both Sprint 18 Cholesky and Sprint 19 LDL^T experiences).
   - If any of items 1-5 did not land: explicit deferral rationale (what tried, what blocked, candidate sprint for follow-up).
5. Final regression (1 hr):
   - `make clean && make format && make lint && make test && make sanitize && make bench && make examples` — all clean.
   - Verify total test count (`make test 2>&1 | grep "Tests run" | awk '{sum += $NF} END {print sum}'`) has grown by at least 30 from the Sprint 18 baseline (1453).

### Deliverables
- `tests/test_sprint19_integration.c` with 10+ tests covering batched LDL^T + cross-threshold dispatch
- `bench_day14.txt` with the final Sprint 19 numbers for `bench_chol_csc`, `bench_ldlt_csc`, and `bench_refactor_csc`
- `docs/algorithm.md` + `README.md` + `PERF_NOTES.md` + `src/sparse_ldlt_csc.c` header refreshed
- `docs/planning/EPIC_2/SPRINT_19/RETROSPECTIVE.md` with full DoD + metrics + deferrals
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 19 marked **Complete**

### Completion Criteria
- Every item (1-5) in the Sprint 19 PROJECT_PLAN.md table is marked ✅ or ⚠️ with deferral rationale
- `make clean && make format && make lint && make test && make sanitize && make bench && make examples` all clean
- Total test count ≥ 1483 (Sprint 18 baseline 1453 + ≥30 new tests)
- Retrospective honestly assesses both wins and misses
