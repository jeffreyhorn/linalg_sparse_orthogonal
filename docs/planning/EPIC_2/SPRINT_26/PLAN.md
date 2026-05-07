# Sprint 26 Plan: ND Fill-Quality Closure (Sprint 25 deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-25 collectively missed (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 best opt-in 0.942× → Sprint 25 best opt-in 0.922× via setting 13).  Sprint 25's 96-measurement sweep is the strongest evidence yet that the residual 7.2pp gap requires intervention at the FINEST FM level (or pre-empting the multilevel pipeline entirely with geometric cut detection on regular grids) — three independent algorithmic axes acting at the coarsening / intermediate-FM / coarsest-bisection levels (Sprint 25 items 1-3) all wash out individually on Pres_Poisson.  This sprint pursues three concrete avenues from `SPRINT_25/RETROSPECTIVE.md` "Sprint 26 inputs" #1: multi-pass FM at the FINEST level beyond Sprint 23's 3 passes (annealing acceptance / different bucket-tie-break / thick-restart-style FM with global rollback), direct geometric cut detection on regular grids, and per-vertex separator scoring.  Also closes the Sprint 25 deferred items: HCC default-flip blocker (bcsstk14 produces degenerate `sep = 0` empty separator under HCC matching), the pre-existing `sparse_eigs.c:948` UBSan division-by-zero log surfaced by Sprint 25 Day 14 sanitize, per-recursion-level partition profiling extension (extends Sprint 25's `SPARSE_ND_PROFILE`), and `nd_base_threshold` re-sweep (Day 11 measured `nd_emit_natural` degenerate-fallback overhead at ~5.3 s on Pres_Poisson).  All items routed from `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Items deferred" + "Sprint 26 inputs" #1-4.

**Starting Point:** Sprint 25 (PR #33, merged at `91ed8c7`) shipped: three new ND env-var-gated algorithmic axes — `SPARSE_ND_COARSENING={heavy_edge,hcc}` (Day 1-3) selecting Heavy Connectivity Coarsening per Karypis-Kumar 1998 §5; `SPARSE_FM_INTERMEDIATE_PASSES` (Day 4-5; default 1, range [1, 10]) extending Sprint 23 Day 11's multi-pass FM from finest to intermediate uncoarsening levels; `SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}` (Day 6-8) substituting spectral bisection via Sprint 20-21 Lanczos eigensolver at the coarsest level — all default off / Sprint 24 behavior preserved bit-identically.  `SPARSE_ND_PROFILE` per-phase ND wall-clock instrumentation in `src/sparse_reorder_nd.c` (Day 11; `_Thread_local`-backed accumulators per the PR #33 review-fix commit `f465a86`).  `wall_check_baseline.txt` extended with `pres_poisson_nd_ms = 47 055` baseline + 1.5× per-key threshold (Day 12; `scripts/wall_check.sh` updated).  Day 9's 96-measurement sweep documented setting 13 (HCC + ratio=200) as the Pres_Poisson best at 0.9218× (-3pp vs Sprint 24 baseline; the headline win) and setting 15 (full set) as the corpus-wide best at Kuu 1.309× (-97pp; the largest single corpus win Sprint 25 produced).  Three of three default-flip attempts blocked: HCC by `bcsstk14 sep=0` finding (degenerate empty separator under HCC matching); intermediate-FM + spectral by neutral Pres_Poisson individual results.  See `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` for the full retrospective; `headline_summary.md` for the Day 9 sweep verdict; `nd_wall_time_decision.md` for the Day 11 variance-vs-cost classification.

**End State:** `sparse_graph_partition` handles HCC's `sep = 0` degenerate-cut case on bcsstk14 cleanly (either via HCC matching tightening at coarsening time or via a `sep = 0` fall-back that re-bisects with HEM); `test_partition_bcsstk14_smoke` passes under `SPARSE_ND_COARSENING=hcc`; `SPARSE_ND_COARSENING=hcc` either flips to default (if the HCC-vs-corpus profile is now flip-rule-clean) or stays advisory with the bcsstk14 path documented.  `src/sparse_eigs.c:948` UBSan division-by-zero log is cleared via the one-line `|| anchor == 0.0` guard extension; a regression test in `tests/test_eigs.c` pins the zero-spectrum path against future regression.  `SPARSE_ND_PROFILE` extends to per-recursion-level partition profiling via a `partition_ns_per_depth[MAX_ND_DEPTH]` accumulator array; `profile_day{N}_per_depth.txt` documents which recursion depth dominates Pres_Poisson ND wall.  `nd_base_threshold` re-sweep produces a per-fixture decision doc and either flips the default from 32 → ≥ 64 (saving the ~5.3 s `nd_emit_natural` overhead) or documents the trade-off as advisory.  `SPARSE_FM_FINEST_STRATEGY={baseline,annealing,thick_restart}` env var lands with whichever sub-axis Sprint 26 item 3's per-depth profile suggests; default `baseline` preserves Sprint 23 behavior; flipped if Pres_Poisson closes ≤ 0.85×.  `SPARSE_ND_GRID_CUT={off,on,auto}` env var lands with the regular-grid-detection heuristic + geometric median-row-or-column cut; default `off` initially, `auto` enables only when the heuristic fires.  `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` extends Sprint 24's existing env var with per-vertex scoring; gated as opt-in.  Pres_Poisson ND/AMD reaches ≤ 0.85× of AMD (or partial close documented + Sprint 27+ routed if a fifth algorithmic axis is needed).  `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` tightens from Sprint 24's 0.96× to whatever default ratio is achieved + 2pp noise margin.  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_26/bench_*.{csv,txt}` plus `finest_fm_decision.md` + `geometric_cut_decision.md` + `per_vertex_sep_decision.md` + `headline_summary.md`.  `docs/algorithm.md` ND subsection updates to describe the three new env vars + their per-fixture deltas; `SPRINT_22/PERF_NOTES.md` (or `SPRINT_26/PERF_NOTES.md` if the Sprint 22 file gets unwieldy after Sprint 25's closures) gets a "Sprint 26 closures" subsection.  `SPRINT_26/RETROSPECTIVE.md` ships filled in (single Day-14 retro per Sprint 25 retrospective lesson "single Day-14 retro that absorbs the Day-13 work matches the actual time spent").

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to 148 hours — exactly the PROJECT_PLAN.md estimate; ~20-hour slack against the 14×12 = 168-hour ceiling for variance / unexpected sub-axis trade-offs.  Risk concentration is item 5 (28 hrs across Days 6-8): FINEST-level FM annealing / thick-restart is novel; the three sub-axes (annealing acceptance / different bucket-tie-break / thick-restart-style global rollback) interact in ways that haven't been measured.  Item 6 (24 hrs across Days 9-10) is the secondary 0.85× candidate but is workload-specific (Pres_Poisson-class regular grids only); irregular fixtures must reject the geometric-cut heuristic cleanly.  Item 7 (24 hrs across Days 11-12) is the tertiary candidate with the smallest expected per-fixture movement but the cleanest safety profile (just a new lift-strategy enum value).  Items 1, 3, 4 are foundation work (~36 hrs combined) and ship in the first week; Item 8 + 9 are the closing-week deliverables (32 hrs combined Days 13-14).

---

## Day 1: Sprint Kickoff — `sparse_eigs.c:948` UBSan Quick-Win + HCC sep=0 Reading

**Theme:** Open the sprint with the smallest deferred item (item 2, ~4 hrs: the `sparse_eigs.c:948` UBSan division-by-zero one-line fix + regression test).  Use the remaining time to read Sprint 25's `coarsening_decision.md` "Two test failures surfaced under the new defaults" and instrument `src/sparse_graph.c::graph_coarsen_hcc` for the Day-2 bcsstk14 profile.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Items deferred" + "Sprint 26 inputs" + `coarsening_decision.md` "What didn't ship" + `nd_wall_time_decision.md` "What we learned from the profile" #3-4.  Pin the Sprint 25 baselines (Pres_Poisson default 0.952×, best opt-in 0.922× via setting 13; bcsstk14 `sep = 0` reproducer = `SPARSE_ND_COARSENING=hcc build/bench_reorder --only bcsstk14 --skip-factor`).
2. **Item 2 — `sparse_eigs.c:948` UBSan quick-win.**  Apply the one-line fix at `src/sparse_eigs.c:946`: extend `if (anchor < scale * 1e-12)` to `if (anchor < scale * 1e-12 || anchor == 0.0)`.  This guards the case where `scale == 0.0` (all eigenvalues exactly zero) AND the current Ritz value `tv_l == 0.0` (so `anchor = fabs(tv_l) == 0.0`), which the existing strict-less-than guard fails to catch.
3. **Item 2 regression test.**  Add `tests/test_eigs.c::test_eigs_zero_spectrum_no_div_by_zero` that constructs a zero-spectrum fixture (e.g. `sparse_matrix_t` with all-zero values, or an explicitly shifted operator `A - σI` at `σ = λ_0 + 0` for a constructed all-zero matrix), calls `sparse_eigs_sym`, and verifies (a) the call returns without UBSan complaint and (b) the residual norm is finite (not NaN from a 0/0).  Run `make sanitize` to confirm the UBSan log no longer fires.
4. Re-instrument `src/sparse_graph.c::graph_coarsen_hcc` with a SPARSE_HCC_DEBUG-gated `fprintf(stderr, ...)` that emits the cmap (matched-vertex assignment) at each call, so Day 2 can capture HCC's matching choices on bcsstk14 to identify the structural property triggering `sep = 0`.  Default off (one branch overhead when off).
5. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `src/sparse_eigs.c:946` extended with `|| anchor == 0.0` guard
- `tests/test_eigs.c::test_eigs_zero_spectrum_no_div_by_zero` regression test added + enabled in RUN_TEST block
- `src/sparse_graph.c::graph_coarsen_hcc` SPARSE_HCC_DEBUG-gated cmap-emit instrumentation
- `make sanitize` clean (no UBSan logs from sparse_eigs.c:948)
- All quality checks clean

### Completion Criteria
- `make sanitize` no longer emits the `src/sparse_eigs.c:948:38: runtime error: division by zero` log
- New `test_eigs_zero_spectrum_no_div_by_zero` passes under both default and `make sanitize` builds
- SPARSE_HCC_DEBUG=1 emits cmap on bcsstk14 reproducer; default-off code path is bit-identical to current master
- `make format && make lint && make test && make sanitize && make wall-check` clean

---

## Day 2: HCC bcsstk14 sep=0 — Root-Cause Profile + Fix Design

**Theme:** Capture HCC's matching choices on bcsstk14 under the Day-1 SPARSE_HCC_DEBUG instrumentation; identify the structural property of bcsstk14 that produces a degenerate coarse-level partition under HCC's `min(deg(u), deg(v))` weighting; design the fix (option (a) HCC matching tightening at coarsening time vs option (b) `sparse_graph_partition` `sep = 0` fall-back).  Day 2's gate is "decision doc + design — implementation lands Day 3".

**Time estimate:** 10 hours

### Tasks
1. Capture HCC's full coarsening trace on bcsstk14: `SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc build/bench_reorder --only bcsstk14 --skip-factor 2> /tmp/hcc_bcsstk14_trace.txt`.  Inspect the cmap at each coarsening level; identify the level where the partition collapses to a one-sided cut.
2. Compare bcsstk14's HCC trace against the (already-passing) HEM trace: `SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=heavy_edge build/bench_reorder --only bcsstk14 --skip-factor 2> /tmp/hem_bcsstk14_trace.txt`.  Diff the cmaps to identify which matching choices differ between HCC and HEM at the level where the partition diverges.
3. Profile the structural property of bcsstk14 that triggers the HCC degeneracy.  Hypothesis (per Sprint 25 `coarsening_decision.md`): bcsstk14's structural-mechanics provenance produces a degree distribution where HCC's `min(deg(u), deg(v))` weighting biases matching toward a pathological direction (e.g. all matches collapse onto the same side of a natural balance line).  Validate via histogram of bcsstk14's degree distribution + per-edge `min(deg(u), deg(v))` distribution; document in `docs/planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md`.
4. Design the fix.  Two options, pick one:
   - **Option (a)** — HCC matching tightening: add a per-vertex check that detects when the matching is producing a one-sided coarse graph and falls back to HEM-style matching for that vertex.  Pros: keeps the fix local to HCC; HEM-bcsstk14 already works (sep=97 ✓).  Cons: heuristic-tuning territory; risk of regressing other fixtures.
   - **Option (b)** — `sparse_graph_partition` `sep = 0` fall-back: detect the degenerate cut at partition emit time, log a fall-back warning, re-bisect with HEM matching.  Pros: cross-cutting; works regardless of which coarsening algorithm produced the degenerate cut.  Cons: extra cost on the degenerate path; logs may be noisy if the fall-back fires unexpectedly.
   - Pick based on Day 2's diagnosis: if the structural property points at HCC alone (plausible from Sprint 25's evidence), prefer (a) for locality; if a future coarsening algorithm could produce the same degeneracy, prefer (b) for robustness.  Document the choice + rationale in `hcc_sep_zero_diagnosis.md`.
5. Stub the chosen fix as a failing-as-expected test in `tests/test_graph.c`: `test_hcc_bcsstk14_no_degenerate_partition` that runs the bcsstk14 coarsening + bisection under `SPARSE_ND_COARSENING=hcc` and asserts `sep > 0`.  Currently fails; Day 3's fix lights it up.
6. Run `make format && make lint && make test`.

### Deliverables
- `/tmp/hcc_bcsstk14_trace.txt` + `/tmp/hem_bcsstk14_trace.txt` matching-choice traces (intermediate; not committed)
- `docs/planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md` root-cause analysis + fix-option decision (a vs b)
- Stubbed `tests/test_graph.c::test_hcc_bcsstk14_no_degenerate_partition` (failing — pin Day-3 fix)
- All quality checks clean (HCC-default-off path remains bit-identical)

### Completion Criteria
- `hcc_sep_zero_diagnosis.md` names the exact structural property of bcsstk14 that triggers HCC's degenerate cut + cites the coarsening level + matching choices
- Fix option (a or b) selected with documented rationale
- `test_hcc_bcsstk14_no_degenerate_partition` compiles and trips with "expected sep > 0, got 0" or analogous failure message
- `make format && make lint && make test` clean

---

## Day 3: HCC bcsstk14 sep=0 — Fix Implementation + Validation + HCC Default-Flip Re-Attempt

**Theme:** Implement the Day-2 chosen fix; validate that bcsstk14 sep > 0 holds under `SPARSE_ND_COARSENING=hcc`; corpus-bit-identical-or-better verification; re-attempt the HCC default flip (Sprint 25 Day 10's reverted experiment) — flip if the bcsstk14-fix-corrected profile is now flip-rule-clean per PLAN.md item 1's HCC flip rule.

**Time estimate:** 6 hours

### Tasks
1. Implement the Day-2 chosen fix (option (a) HCC matching tightening or option (b) `sparse_graph_partition` sep=0 fall-back) per `hcc_sep_zero_diagnosis.md`.
2. Run `tests/test_graph.c::test_hcc_bcsstk14_no_degenerate_partition` — must now pass.
3. Run `tests/test_graph.c::test_partition_bcsstk14_smoke` under `SPARSE_ND_COARSENING=hcc` — must now pass (this is the Sprint 25 Day 10 default-flip blocker).
4. Run `tests/test_reorder_nd.c::test_nd_bcsstk14_fill_vs_amd` under `SPARSE_ND_COARSENING=hcc` — bcsstk14 nnz_L should be ≤ Sprint 25 setting 13's 130 358 (Day 10 capture in `bench_day10_setting13_advisory.csv`).
5. Re-attempt the HCC default flip: change `parse_coarsening_strategy()` default in `src/sparse_graph.c` from `COARSENING_HEAVY_EDGE` to `COARSENING_HCC`.  Run the full corpus bench: `build/bench_reorder --skip-factor`.  Capture per-fixture nnz_L vs Sprint 25 setting 13 baseline; if all 6 fixtures bit-identical-or-improving + Pres_Poisson ≤ 0.937× (Sprint 25 HCC-alone target), flip is good; commit the flip.  If any fixture regresses past 5pp (Kuu was the original concern), revert + document.
6. Update `docs/planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md` with the post-fix verification results + flip outcome.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_graph.c` HCC sep=0 fix landed (option (a) or (b) per Day 2 design)
- `tests/test_graph.c::test_hcc_bcsstk14_no_degenerate_partition` enabled and passing
- `tests/test_graph.c::test_partition_bcsstk14_smoke` passes under `SPARSE_ND_COARSENING=hcc`
- HCC default-flip outcome (flipped or reverted) documented in `hcc_sep_zero_diagnosis.md` post-fix verification section
- All quality checks clean

### Completion Criteria
- `bcsstk14 sep > 0` under `SPARSE_ND_COARSENING=hcc` (verified by the new test + the existing partition smoke test)
- Corpus nnz_L bit-identical-or-improving vs Sprint 25 setting 13 baseline across all 6 fixtures
- HCC default flip decision recorded with bench evidence
- `make format && make lint && make test && make wall-check` clean

---

## Day 4: Per-Recursion-Level Partition Profiling Extension

**Theme:** Extend Sprint 25 Day 11's `SPARSE_ND_PROFILE` instrumentation in `src/sparse_reorder_nd.c` to break out `sparse_graph_partition` cost by recursion depth.  The Day 11 profile measured cumulative partition time across all 301 recursive calls on Pres_Poisson but didn't attribute cost by depth — Sprint 26 item 5 (FINEST FM) needs to know whether cost concentrates at the root, intermediate levels, or near the base threshold.

**Time estimate:** 12 hours

### Tasks
1. Add a `MAX_ND_DEPTH` constant in `src/sparse_reorder_nd.c` (sized to ~64 — covers any plausible ND recursion depth on n ≤ 10^9 graphs).  Add a `_Thread_local long long partition_ns_per_depth[MAX_ND_DEPTH]` accumulator + matching `_Thread_local idx_t partition_calls_per_depth[MAX_ND_DEPTH]` counter, parallel to the existing per-phase accumulators.
2. Wrap the `sparse_graph_partition` call in `nd_recurse` to accumulate to both the existing cumulative `nd_prof_partition_ns` AND the new per-depth `partition_ns_per_depth[depth]`.  `depth` is already a parameter to `nd_recurse`; just thread it through.  Bounds-check `depth < MAX_ND_DEPTH` to prevent OOB on degenerate-deep graphs (fall back to accumulating into `partition_ns_per_depth[MAX_ND_DEPTH - 1]` with a SPARSE_ND_PROFILE warning if that fires).
3. Extend the stderr emit at the end of `sparse_reorder_nd` to include the per-depth breakdown.  Format: a per-depth table with columns `depth, calls, total_ms, avg_ms` for depths where `calls > 0`.
4. Re-run the Day 11 5-run capture pattern: `SPARSE_ND_PROFILE=1 build/bench_reorder --only Pres_Poisson --skip-factor`.  Save the per-run output to `docs/planning/EPIC_2/SPRINT_26/profile_day4_per_depth.txt` (5 runs, with the same lightly-polished formatting note as Sprint 25 Day 11's profile).
5. Analyze the per-depth profile.  Hypothesis to validate: cost is concentrated at depths 0-2 (the largest subgraphs); leaf-AMD splice region at depths near `log2(n / nd_base_threshold)` is essentially free.  Document analysis in `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md`: per-depth fractions, identify which depth the FINEST-FM intervention (item 5) should target.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_reorder_nd.c` extended with per-depth partition accumulators
- `docs/planning/EPIC_2/SPRINT_26/profile_day4_per_depth.txt` 5-run capture
- `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md` analysis + Sprint-26 item 5 design input
- All quality checks clean

### Completion Criteria
- Per-depth instrumentation compiles + runs; emits readable per-depth table when `SPARSE_ND_PROFILE=1`
- 5-run Pres_Poisson capture + analysis identify the depth(s) where partition cost concentrates
- `per_recursion_profile_day4.md` provides design input for item 5's sub-axis selection (annealing / bucket-tie-break / thick-restart-style FM)
- `make format && make lint && make test && make wall-check` clean

---

## Day 5: `nd_base_threshold` Re-Sweep + `nd_emit_natural` Reduction Decision

**Theme:** Sprint 25 Day 11 measured `nd_emit_natural` (degenerate single-side partition fallback at small subgraphs) firing 32 times on Pres_Poisson at ~165 ms each = ~5.3 s of cumulative cost.  Raising `nd_base_threshold` from 32 → 64 (or higher) would skip the degenerate-partition cases at the cost of forcing leaf-AMD on larger subgraphs.  Sweep the threshold and decide whether to flip the default.

**Time estimate:** 8 hours

### Tasks
1. Sweep `sparse_reorder_nd_base_threshold` ∈ {32 (default), 48, 64, 96, 128} on the full Sprint 25 corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson) via the existing `bench_reorder --nd-threshold N` flag.  Capture nnz_L delta + wall-time delta + `nd_emit_natural` call count delta (use `SPARSE_ND_PROFILE=1`).  Save raw output to `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_sweep.txt`.
2. Build a per-fixture × per-threshold summary table: `(threshold, fixture, nnz_L, wall_ms, emit_natural_calls)`.  For each fixture, identify whether nnz_L regresses (leaf-AMD on n ~64 subgraphs may produce different fill than ND-recursion through them) and whether wall-time improves (the `~5.3 s emit_natural` saving on Pres_Poisson vs leaf-AMD's added cost).
3. Apply the flip rule: flip the default if (a) Pres_Poisson wall improves by ≥ 5 % AND (b) no fixture regresses nnz_L past 1pp.  Document the decision in `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_decision.md` with the per-fixture sweep table + flip outcome (flipped to N or stays at 32, plus per-fixture advisory if a per-workload tuning emerges).
4. If flipped: change `sparse_reorder_nd_base_threshold = N` in `src/sparse_reorder_nd.c`; update `sparse_reorder_nd_internal.h` doc-comment; re-run `make wall-check` to verify Pres_Poisson ND wall stays under the 70 583 ms 1.5× ceiling (it should be much faster if the flip is valid).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_sweep.txt` 5-threshold × 6-fixture × 2-metric (nnz_L + wall) capture
- `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_decision.md` per-fixture summary + flip decision
- `src/sparse_reorder_nd.c` default updated (if flipped) + comment block updated to reflect Sprint 26 measurement
- All quality checks clean

### Completion Criteria
- Sweep covers 5 thresholds × 6 fixtures + emit_natural call counts captured
- Decision doc justifies the flip-or-stay choice with per-fixture flip-rule application
- If flipped: `make wall-check` Pres_Poisson ND ≤ baseline (and ideally much lower if the ~5.3 s saving materializes)
- `make format && make lint && make test && make wall-check` clean

---

## Day 6: FINEST FM — Sub-Axis Selection + Design

**Theme:** Pick which of three Sprint 25 `RETROSPECTIVE.md` "Sprint 26 inputs" #1 sub-axes to implement: (a) annealing acceptance — accept worsening moves with decreasing probability; (b) different bucket-tie-break — currently FIFO, try LIFO or seeded random; (c) thick-restart-style FM with global rollback — track best cut across all passes, allow each pass to re-explore from that anchor.  Selection driven by Day 4's per-depth profile.  Day 6's gate is "design doc + sub-axis selection + skeleton — implementation lands Day 7".

**Time estimate:** 12 hours

### Tasks
1. Read `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md`'s analysis of where partition cost concentrates on Pres_Poisson.  If the profile shows cost concentrated at depths 0-2 (large subgraphs), the FINEST FM intervention has the most leverage.  If the profile shows cost evenly distributed, the per-depth picture doesn't strongly differentiate the three sub-axes.
2. Compare the three sub-axes against the Sprint 25 Day 5 saturation finding ("passes ≥ 5 saturate at 0.952×"):
   - **(a) Annealing acceptance**: accepts worsening moves to escape local minima.  Best-fit when FM is converging to a local-minimum cut that's not the global minimum.  Risk: too-aggressive annealing destabilises convergence.
   - **(b) Bucket-tie-break (LIFO / seeded random)**: changes the order in which equal-gain moves are processed.  Best-fit when the saturation is caused by FIFO consistently picking the same wrong tie-break.  Risk: small movement, may not move the needle.
   - **(c) Thick-restart-style FM**: tracks global best, restarts each pass from the anchor with random perturbation.  Best-fit when FM is getting stuck in different local minima per pass and Sprint 23's rollback-on-regress isn't enough.  Risk: 2-3× wall cost (each pass re-runs FM from scratch); needs careful budget bounding.
   - Pick based on Day 4's per-depth profile: if cost concentrates at one depth, sub-axis (c) thick-restart targets that depth most directly; if cost is distributed, sub-axis (a) annealing has the broadest applicability.  Document in `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md`: sub-axis selection + algorithm contract + sweep dimensions.
3. Sketch the `SPARSE_FM_FINEST_STRATEGY={baseline,annealing,thick_restart,bucket_tiebreak}` env-var gate in `src/sparse_graph.c::graph_refine_fm` (or wherever the finest-level FM passes are dispatched).  Default `baseline` (Sprint 23 behavior); on-value the chosen sub-axis; out-of-range / non-numeric input falls back to `baseline` matching Sprint 24/25 env-var validation patterns.
4. Stub `tests/test_graph.c::test_finest_fm_strategy_<name>_smoke` for the chosen sub-axis as failing-as-expected: pin the expected behavior (e.g. annealing accepts at least one worsening move; thick-restart returns to a checkpoint state; bucket-tie-break picks last-in vs first-in).  Day 7's implementation lights it up.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md` sub-axis selection + algorithm contract + env-var design
- `SPARSE_FM_FINEST_STRATEGY` env-var skeleton in `src/sparse_graph.c` (default `baseline`; chosen sub-axis stub falls through to baseline)
- Stubbed `tests/test_graph.c::test_finest_fm_strategy_<name>_smoke` (failing — pin Day-7 implementation)
- All quality checks clean

### Completion Criteria
- `finest_fm_design.md` justifies the sub-axis selection against Day 4's per-depth profile
- Env-var dispatch compiles + Sprint 23 baseline path stays bit-identical (default off)
- Sub-axis-specific test stub trips with "not yet implemented" or analogous skip message
- `make format && make lint && make test && make wall-check` clean

---

## Day 7: FINEST FM — Implementation

**Theme:** Implement the Day-6 chosen sub-axis (annealing / thick-restart / bucket-tie-break) end-to-end behind the `SPARSE_FM_FINEST_STRATEGY=<name>` env-var gate.  Day 7's gate is "library compiles + Sprint 23 default-path corpus tests bit-identical (env var off) + chosen sub-axis produces measurable behavior change when on".

**Time estimate:** 12 hours

### Tasks
1. Implement the chosen sub-axis per Day-6's `finest_fm_design.md`.  Re-use Sprint 23 Day 11's multi-pass FM scaffolding + Sprint 23 Day 10's gain-bucket structure (`src/sparse_graph_fm_buckets.h`); only the per-pass / per-move acceptance / restart logic changes.
2. Add the `SPARSE_FM_FINEST_STRATEGY` env-var read at the top of the FM dispatch in `graph_refine_fm` (matching Sprint 25 Day 4 `SPARSE_FM_INTERMEDIATE_PASSES` pattern).  Out-of-range / non-numeric / missing → default `baseline`.
3. Run the existing 39 partition tests under `SPARSE_FM_FINEST_STRATEGY=baseline` (default) — should all pass bit-identically to current master.
4. Run the same tests under `SPARSE_FM_FINEST_STRATEGY=<chosen>` — most will pass (the FM still produces a valid cut, just a different one); some determinism contracts may need re-validation.  Triage: if `test_partition_determinism_*` fails under the new sub-axis, root-cause and fix on Day 8.
5. Light up the Day-6 stub `test_finest_fm_strategy_<name>_smoke`: real assertions for the sub-axis-specific behavior (annealing accepts worsening; thick-restart returns to anchor; etc.).
6. Capture a quick-look bench: `SPARSE_FM_FINEST_STRATEGY=<chosen> build/bench_reorder --only Pres_Poisson --skip-factor` — does Pres_Poisson nnz_L move?  Save to `docs/planning/EPIC_2/SPRINT_26/bench_day7_finest_fm_quicklook.txt`.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_graph.c` finest-FM sub-axis implementation behind `SPARSE_FM_FINEST_STRATEGY=<chosen>` env-var gate
- Sprint 23 baseline path preserved bit-identically (default off, all 39 partition tests pass)
- `tests/test_graph.c::test_finest_fm_strategy_<name>_smoke` lit up with real assertions
- `docs/planning/EPIC_2/SPRINT_26/bench_day7_finest_fm_quicklook.txt` Pres_Poisson quick-look capture
- All quality checks clean

### Completion Criteria
- Library compiles under both `SPARSE_FM_FINEST_STRATEGY=baseline` and `=<chosen>`
- Sprint 23 default path tests bit-identical to current master
- Chosen sub-axis produces measurable behavior change (sub-axis-specific test passes; quick-look bench shows movement vs baseline, even if the sign isn't yet confirmed positive)
- `make format && make lint && make test && make wall-check` clean

---

## Day 8: FINEST FM — Cross-Corpus Sweep + Decision

**Theme:** Sweep the chosen sub-axis across the full Sprint 25 corpus + tunable parameters (annealing temperature schedule, thick-restart pass budget, bucket-tie-break seed); apply the Sprint 25-style flip rule (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression past 5pp); decide on default.

**Time estimate:** 8 hours

### Tasks
1. Sweep the chosen sub-axis's tunable parameter (e.g. annealing initial-temperature ∈ {0.01, 0.1, 1.0}; thick-restart pass-budget ∈ {3, 5, 10}; bucket-tie-break seed ∈ {0, 42, 12345}) on the full Sprint 25 corpus.  Capture per-fixture nnz_L + wall.  Save to `docs/planning/EPIC_2/SPRINT_26/finest_fm_sweep.txt`.
2. Build a per-fixture × per-parameter summary table.  Identify the parameter values that close Pres_Poisson the most + the values that trade-off favourably across the corpus.
3. Apply the PLAN.md flip rule (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression past 5pp).  Document the decision in `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md` with the per-fixture sweep table + flip outcome + per-fixture advisory if a workload-specific tuning emerges.
4. If a parameter value closes Pres_Poisson ≤ 0.85× cleanly: flip the default (change `parse_finest_fm_strategy()` default in `src/sparse_graph.c` from `FINEST_FM_BASELINE` to the chosen sub-axis); re-run the full corpus + wall-check.
5. If the sub-axis falls short of 0.85×: document the partial close + escalation to Day 9 (item 6 geometric cut) per `finest_fm_decision.md` "Escalation to item 6" section.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_sweep.txt` per-parameter × per-fixture capture
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md` flip decision + per-fixture advisory + escalation routing
- `src/sparse_graph.c` default updated (if flipped) + comment block updated
- All quality checks clean

### Completion Criteria
- Sweep covers ≥ 3 parameter values × 6 fixtures
- Decision doc applies the flip rule consistently with PLAN.md item 5
- If 0.85× achieved: default flipped + corpus bit-identical-or-improving + wall-check clean
- If 0.85× missed: escalation to Day 9 (item 6) documented + Sprint 27 routing prepared if items 6-7 also fall short
- `make format && make lint && make test && make wall-check` clean

---

## Day 9: Geometric Grid-Cut — Detection Heuristic + Design

**Theme:** Design the regular-grid-detection heuristic + the geometric median-row-or-column cut that pre-empts the multilevel pipeline on Pres_Poisson-class workloads.  Day 9's gate is "design doc + grid-detection heuristic + env-var skeleton — implementation lands Day 10".

**Time estimate:** 12 hours

### Tasks
1. Design the grid-detection heuristic in `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md`.  Key features to inspect:
   - Vertex degree histogram (regular 2D grids: ~4 (interior), ~3 (edges), ~2 (corners), with negligible variance).
   - Adjacency pattern regularity (each interior vertex shares exactly 4 neighbours with adjacent vertices in a "+" shape).
   - Total nnz vs n (regular 2D grid: nnz ≈ 5n; deviation > 10 % → reject as non-grid).
   - Optional: connectivity entropy (regular grids have low entropy in the adjacency-pattern signature).
   - Decision: a vertex-set passes "regular-grid" detection if (a) ≥ 90 % of vertices have degree ∈ {3, 4, 5}, (b) nnz / n ∈ [4.5, 5.5], (c) adjacency-pattern signature matches a 2D-grid template.  Tunable thresholds; sweep on Day 10.
2. Design the geometric median-row-or-column cut.  Two sub-cases:
   - **With coordinates** (rare in sparse linalg — coordinates aren't normally part of the matrix): bisect at the median of the longest dimension.
   - **Without coordinates** (the common case): infer an "axis" via bipartite-matching-derived spectral coordinates (the Fiedler vector's median already approximates this; could re-use Sprint 25 Day 6-8's `graph_bisect_coarsest_spectral` Lanczos pass).  Or simpler: use a BFS from a peripheral vertex to derive a vertex-distance field; bisect at the median distance.
   - Document the chosen algorithm + rationale.  Lift the bisection edge as the separator (matching Sprint 22's edge-to-vertex separator pattern).
3. Sketch the `SPARSE_ND_GRID_CUT={off,on,auto}` env-var gate in `src/sparse_reorder_nd.c::nd_recurse` (the entry point where the pipeline-vs-grid-cut decision happens).  Default `off`; `on` forces grid cut regardless of detection; `auto` enables only when the grid-detection heuristic fires.  Out-of-range / non-numeric / missing → default `off`.
4. Stub `tests/test_graph.c::test_grid_detection_pres_poisson` (Pres_Poisson must detect as grid; assertion on the detection function returning true) + `test_grid_detection_irregular_rejects` (Kuu / bcsstk14 must NOT detect as grid; assertion on false).  Day 10's implementation lights both up.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md` grid-detection heuristic + geometric-cut algorithm + env-var design
- `SPARSE_ND_GRID_CUT` env-var skeleton in `src/sparse_reorder_nd.c` (default off; falls through to multilevel pipeline)
- Stubbed `tests/test_graph.c::test_grid_detection_pres_poisson` + `test_grid_detection_irregular_rejects` (failing — pin Day-10 implementation)
- All quality checks clean

### Completion Criteria
- `geometric_cut_design.md` names exact detection thresholds + cut algorithm + env-var contract
- Env-var skeleton compiles; default `off` path is bit-identical to current master
- Both stub tests trip with "not yet implemented" or analogous skip message
- `make format && make lint && make test && make wall-check` clean

---

## Day 10: Geometric Grid-Cut — Implementation + Pres_Poisson Validation

**Theme:** Implement the Day-9 grid-detection heuristic + geometric median-row-or-column cut behind the `SPARSE_ND_GRID_CUT={off,on,auto}` env-var gate.  Validate on Pres_Poisson (must close to ≤ 0.85× to be worth the special path) + irregular fixtures (must NOT regress — the heuristic must reject them cleanly).

**Time estimate:** 12 hours

### Tasks
1. Implement the grid-detection heuristic per Day 9's `geometric_cut_design.md` in `src/sparse_reorder_nd.c` (or a new helper file `src/sparse_nd_grid.c` if the code grows past ~100 lines).  Light up `test_grid_detection_pres_poisson` + `test_grid_detection_irregular_rejects` with real assertions.
2. Implement the geometric median-row-or-column cut per Day 9's design.  Use the chosen axis-inference method (BFS-distance-field-median or coordinate-based depending on Day 9's choice).  Lift the bisection vertices as the separator.
3. Wire the `SPARSE_ND_GRID_CUT=auto` path: in `nd_recurse`, before calling `sparse_graph_partition`, run grid-detection; if it fires, call the geometric cut path; if not, fall through to the multilevel pipeline.  `=on` forces grid cut regardless; `=off` (default) ignores grid detection.
4. Validate on Pres_Poisson under `SPARSE_ND_GRID_CUT=auto`: nnz_L should be ≤ 0.85× AMD (the headline goal).  If yes, this is the secondary 0.85× candidate succeeding.  Capture to `docs/planning/EPIC_2/SPRINT_26/bench_day10_grid_cut_pres_poisson.txt`.
5. Validate on irregular fixtures (Kuu, bcsstk14, s3rmt3m3) under `SPARSE_ND_GRID_CUT=auto`: detection must reject them cleanly + nnz_L must be bit-identical to `=off` (since the grid-cut path doesn't fire).  Confirm via `bench_reorder` that the auto-detect path doesn't trip on irregular fixtures.
6. Capture full corpus bench under `SPARSE_ND_GRID_CUT=auto`: `build/bench_reorder --skip-factor`.  Save to `docs/planning/EPIC_2/SPRINT_26/bench_day10_grid_cut_corpus.txt`.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_reorder_nd.c` (or `src/sparse_nd_grid.c`) grid-detection + geometric-cut implementation
- `tests/test_graph.c` two grid-detection tests lit up + passing
- `docs/planning/EPIC_2/SPRINT_26/bench_day10_grid_cut_pres_poisson.txt` Pres_Poisson nnz_L under `=auto`
- `docs/planning/EPIC_2/SPRINT_26/bench_day10_grid_cut_corpus.txt` full-corpus capture under `=auto`
- All quality checks clean

### Completion Criteria
- Pres_Poisson detects as grid + grid-cut produces ≤ 0.85× nnz_L (the headline goal) OR documents the partial close + Sprint 27 routing
- All 4 irregular fixtures reject grid detection + bit-identical nnz_L to `SPARSE_ND_GRID_CUT=off`
- Corpus bench under `=auto` doesn't regress any fixture
- `make format && make lint && make test && make wall-check` clean

---

## Day 11: Per-Vertex Separator Scoring — Design + Implementation

**Theme:** Implement the third 0.85× candidate per Sprint 25 RETROSPECTIVE.md "Sprint 26 inputs" #1 (c): score boundary vertices individually by a "separator-suitability" function (degree × balance-impact + other features) and pick the top-K vertices regardless of side.  This is closer to AMD's pivot-selection style than Sprint 22 / 24's side-then-lift heuristics.  Day 11's gate is "design doc + implementation + Pres_Poisson quick-look".

**Time estimate:** 12 hours

### Tasks
1. Design the per-vertex separator-suitability function in `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_design.md`.  Candidate features to combine:
   - Vertex degree (high-degree vertices contribute more to separator size if added — score down).
   - Balance-impact (vertices on the boundary of the larger side improve balance if added to separator — score up).
   - Existing-separator adjacency (vertices already neighbour-of-separator are "natural" extensions — score up).
   - Optional: coarsening-cmap stability (vertices that landed in a "stable" coarse cluster — score up).
   - Pin the linear-combination weights (or sweep them on Day 12).
2. Add `parse_sep_lift_strategy()` (or extend Sprint 24 Day 6's existing parser) to recognize a new `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` value alongside `smaller_weight` (default) and `balanced_boundary` (Sprint 24 advisory).  Out-of-range / non-numeric / missing → default `smaller_weight`.
3. Implement the per-vertex scoring + top-K selection in `src/sparse_graph.c::graph_edge_separator_to_vertex_separator` (or a new helper called by it).  K is chosen to maintain Sprint 24 Day 6's 70/30 post-lift balance check; if the per-vertex pick would skew past 70/30, fall back to `smaller_weight` matching Sprint 24's pattern.
4. Capture quick-look on Pres_Poisson under `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex`: `build/bench_reorder --only Pres_Poisson --skip-factor`.  Does Pres_Poisson move?  Save to `docs/planning/EPIC_2/SPRINT_26/bench_day11_per_vertex_pres_poisson.txt`.
5. Capture quick-look on Kuu (must not regress past `balanced_boundary`'s -38pp Sprint 24 win): `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex build/bench_reorder --only Kuu --skip-factor`.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_design.md` scoring function + env-var design
- `src/sparse_graph.c` per-vertex scoring + top-K selection implementation behind `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` env-var gate
- `docs/planning/EPIC_2/SPRINT_26/bench_day11_per_vertex_{pres_poisson,kuu}.txt` quick-look captures
- All quality checks clean

### Completion Criteria
- `per_vertex_sep_design.md` names the scoring formula + tunable weights + 70/30 balance fallback
- Library compiles under all three values: `smaller_weight` (default; bit-identical), `balanced_boundary` (Sprint 24 advisory; bit-identical), `per_vertex` (Sprint 26 new; produces a different separator)
- Pres_Poisson + Kuu quick-looks captured; movement direction (win / neutral / regress) documented
- `make format && make lint && make test && make wall-check` clean

---

## Day 12: Per-Vertex Separator Scoring — Validation + Cross-Corpus Sweep + Decision

**Theme:** Validate per-vertex separator scoring on the full corpus; sweep tunable scoring weights if the Day-11 quick-look showed promise; apply the flip rule; decide whether to flip the default OR ship as advisory OR escalate to "primary if items 5-6 fall short" per PLAN.md item 7.

**Time estimate:** 12 hours

### Tasks
1. Sweep the per-vertex scoring weights on the full corpus.  Three weight combinations to try (Day 11's design pins the default; sweep alternatives):
   - **Balance-priority**: heavily weight balance-impact over degree; expects Kuu / bcsstk14 wins (the irregular fixtures where Sprint 24's `balanced_boundary` already shines).
   - **Degree-priority**: heavily weight (low) degree over balance; expects Pres_Poisson wins (regular grids where high-degree separator vertices destroy fill quality).
   - **Hybrid**: linear combination ~50/50; baseline.
   Capture full corpus bench for each weight combination.  Save to `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_sweep.txt`.
2. Build per-fixture × per-weight summary table.  Apply the flip rule (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression past 5pp); document in `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_decision.md`.
3. Cross-evaluate per-vertex against items 5-6 outcomes.  Three scenarios:
   - Items 5-6 already closed Pres_Poisson ≤ 0.85×: per-vertex is a corpus-wide-improvement candidate; flip default if it composes constructively.
   - Items 5-6 fell short: per-vertex is the primary 0.85× candidate; sweep more aggressively + try compound combinations with items 5-6.
   - Per-vertex itself closes ≤ 0.85×: this is the headline win.
4. If a weight combination produces a clear flip-rule-clean default win, change `parse_sep_lift_strategy()` default in `src/sparse_graph.c` from `SEP_LIFT_SMALLER_WEIGHT` to `SEP_LIFT_PER_VERTEX`; re-run full corpus + wall-check.
5. If per-vertex fails to flip: ship as advisory + document per-fixture recommendations in `per_vertex_sep_decision.md`.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_sweep.txt` weight × fixture sweep
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_decision.md` flip decision + per-fixture advisory + cross-evaluation against items 5-6
- `src/sparse_graph.c` default updated (if flipped) + comment block updated
- All quality checks clean

### Completion Criteria
- Sweep covers ≥ 3 weight combinations × 6 fixtures
- Decision doc applies the flip rule + cross-references items 5-6 outcomes
- If flipped: corpus bit-identical-or-improving + Pres_Poisson ≤ items-5-6's achievement (or better)
- `make format && make lint && make test && make wall-check` clean

---

## Day 13: Cross-Corpus Re-Bench + Production-Default Decisions + Test-Bound Tightening

**Theme:** Run the full Sprint 25 + Sprint 26 env-var combination matrix; pick the corpus-wide-best + Pres_Poisson-headline-best (Sprint 25 Day 9 pattern: setting 13 vs setting 15); tighten `test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 24's 0.96× to whatever the achieved default ratio + 2pp noise margin allows.

**Time estimate:** 12 hours

### Tasks
1. Build the combination matrix.  Sprint 25 had 16 settings (3 new env vars × 2-3 values each); Sprint 26 adds three new env vars (`SPARSE_FM_FINEST_STRATEGY`, `SPARSE_ND_GRID_CUT`, extended `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex`) + uses the items 1, 4 outcomes (HCC default-flip outcome + nd_base_threshold default).  Cap at ≤ 24 representative combinations to keep total bench wall under 8 hours (Sprint 25 Day 9's 96-measurement sweep took ~6 hours; Sprint 26 needs to fit a similar budget).
2. Run the combination matrix on the full Sprint 25 corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson).  Capture per-setting × per-fixture nnz_L + wall.  Save to `docs/planning/EPIC_2/SPRINT_26/bench_day13_combinations.{csv,txt}`.
3. Identify the corpus-wide-best setting (the Sprint 25 setting-15 analogue) and the Pres_Poisson-headline-best setting (the Sprint 25 setting-13 analogue).  Apply the flip rule per setting; flip defaults if a clear winner emerges (subject to Day-3's HCC default-flip outcome + the existing test contracts).
4. Tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 24's `≤ 0.96× nnz_amd` to whatever the achieved default ratio + 2pp noise margin allows.  If items 5-7 closed Pres_Poisson default to ≤ 0.85×, tighten to `≤ 0.87× nnz_amd`; if partial close to ≤ 0.90×, tighten to `≤ 0.92× nnz_amd`; if no default movement (Sprint 26 ships everything as advisory), bound stays at 0.96×.
5. Build `docs/planning/EPIC_2/SPRINT_26/headline_summary.md` summarizing the verdict on each headline gate (literal Pres_Poisson ≤ 0.85×; smaller-fixture corpus safety; default-flip outcomes; Sprint 27+ routing for any unmet items).  Format mirrors Sprint 25 Day 9's `headline_summary.md`.
6. Capture `bench_day13_amd_qg.{csv,txt}` via `build/bench_amd_qg` for the qg-AMD wall-time + nnz_L parity check (matches Sprint 25 Day 13 pattern).
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_26/bench_day13_combinations.{csv,txt}` combination matrix capture
- `docs/planning/EPIC_2/SPRINT_26/bench_day13_amd_qg.{csv,txt}` qg-AMD parity capture
- `docs/planning/EPIC_2/SPRINT_26/headline_summary.md` verdict + Sprint 27+ routing
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` tightened to whatever Sprint 26 achieved + 2pp
- All quality checks clean

### Completion Criteria
- Combination matrix runs ≥ 16 settings × 6 fixtures (≥ 96 measurements; matches Sprint 25 Day 9 scale)
- `headline_summary.md` provides a Verdict table with PASS/MISS for each gate
- Test bound tightened to reflect achieved default ratio
- `make format && make lint && make test && make wall-check` clean

---

## Day 14: Soak, Final Bench Capture, Tests + Docs + Retrospective + PR

**Theme:** Final cross-ordering + qg-AMD captures as the end-of-sprint headline, full corpus regression run, sanitize + tsan, single-pass retrospective body (per Sprint 25 retrospective lesson), tests + docs sweep, and PR open.  This is the day that gates the merge.

**Time estimate:** 12 hours

### Tasks
1. Final capture: re-run `bench_reorder.c` + `bench_amd_qg.c` once more.  Save to `bench_day14.{csv,txt}` + `bench_day14_amd_qg.{csv,txt}`.  Sanity-check nnz_L bit-identical to Day 13 (Day 14 should be tests + docs + retro only, no algorithmic change).
2. Run `make sanitize` (UBSan) against the full test suite.  Item 2's `sparse_eigs.c:948` UBSan fix should have cleared the Sprint 25 inherited log; verify Sprint 26's algorithmic additions (HCC sep=0 fix, finest-FM sub-axis, geometric grid-cut, per-vertex sep scoring) don't reintroduce sanitizer issues.
3. Run `make tsan` (with Homebrew LLVM clang per `Makefile` `tsan` target's note) against the full test suite.  Sprint 26's changes are single-threaded (no OpenMP additions); verify Sprint 25's tsan baseline still applies + the new finest-FM / grid-cut / per-vertex-sep paths don't introduce races.
4. Closing tests for items 1, 4, 5, 6, 7: audit the new tests added Days 1-12 are enabled in their test binaries' RUN_TEST blocks (no `#if 0` guards).  Add any missing tests called out by the items that haven't landed yet.
5. Update `docs/algorithm.md` ND subsection: describe the three new env vars (`SPARSE_FM_FINEST_STRATEGY`, `SPARSE_ND_GRID_CUT`, extended `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex`) + their per-fixture deltas; supersede Sprint 25's "0.85× literal target route to Sprint 26" caveat with the actual achievement.  Update the per-fixture advisory list with Sprint 26's new combinations.
6. Append a "Sprint 26 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open `docs/planning/EPIC_2/SPRINT_26/PERF_NOTES.md` if the Sprint 22 file is too long after Sprint 25's closures section).  Include: the Day-13 sweep summary table, what moved (Pres_Poisson ND/AMD ratio + which env vars contributed), what didn't move (fixtures where the new defaults were neutral), Sprint 27+ routing for any deferred items.
7. Fill in the Sprint 26 retrospective single-pass at `docs/planning/EPIC_2/SPRINT_26/RETROSPECTIVE.md` (no Day-13 stub per the Sprint 25 retro lesson): Goal recap; DoD checklist; Final metrics (Pres_Poisson ND/AMD ratio achieved + per-fixture corpus table); Performance highlights; What went well; What surprised us; What didn't go well; Items deferred (any 0.85× shortfall routes to Sprint 27 with concrete avenues); Lessons; Sprint 27 inputs; Acknowledgements (Karypis-Kumar 1998 + Sprint 23's gain-bucket FM as foundations); Day-by-day capsule with commit SHAs; Day-budget vs estimate; DoD verification.
8. Open the Sprint 26 PR (`gh pr create`) targeting `master`.  PR description summarises the nine items + the day-by-day commits + the headline numbers from `bench_day14.txt` vs Sprint 25's `bench_day14.txt`.
9. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `bench_day14.{csv,txt}` + `bench_day14_amd_qg.{csv,txt}` final captures
- `make sanitize` + `make tsan` clean (or pre-existing infrastructure gaps explicitly flagged in retro)
- New tests for items 1, 4, 5, 6, 7 enabled in RUN_TEST blocks
- `docs/algorithm.md` ND subsection updated with Sprint 26 env vars + advisory deltas
- `SPRINT_22/PERF_NOTES.md` (or `SPRINT_26/PERF_NOTES.md`) "Sprint 26 closures" subsection
- `RETROSPECTIVE.md` filled in (single Day-14 retro; all 13 sections from Sprint 25's pattern have content)
- Sprint 26 PR opened
- All quality checks clean

### Completion Criteria
- Final cross-ordering capture matches Day 13's output bit-identically on nnz_L (no regressions in Day 14's doc/test/retro work)
- `make sanitize` + `make tsan` clean against the full test suite (or documented as Sprint-27 infrastructure follow-ups)
- Retrospective written; all 13 sections have content (no `(Day 14 prose pending)` placeholders)
- PR opened; description references the headline Pres_Poisson ND/AMD ratio achieved + the three new env-var defaults (whichever flipped)
- `make format && make lint && make test && make sanitize && make wall-check` clean

---

## Sprint 26 Summary

**Total estimated hours:** 8 + 10 + 6 + 12 + 8 + 12 + 12 + 8 + 12 + 12 + 12 + 12 + 12 + 12 = 148 hours

**Item-to-day mapping:**

| Item | Days | Hours |
|------|------|-------|
| 1: HCC bcsstk14 sep=0 root-cause + fix | Days 2-3 (+ Day 1 prep) | 16 |
| 2: `sparse_eigs.c:948` UBSan quick-win | Day 1 | 4 |
| 3: Per-recursion-level partition profiling extension | Day 4 | 12 |
| 4: `nd_base_threshold` re-sweep + `nd_emit_natural` reduction | Day 5 | 8 |
| 5: Multi-pass FM at the FINEST level (annealing / thick-restart) | Days 6-8 | 32 (allotted; PROJECT_PLAN.md estimate 28) |
| 6: Direct geometric cut detection on regular grids | Days 9-10 | 24 |
| 7: Per-vertex separator scoring | Days 11-12 | 24 |
| 8: Cross-corpus re-bench + production-default decisions + test-bound tightening | Day 13 | 12 (allotted; PROJECT_PLAN.md estimate 16) |
| 9: Tests + docs + retrospective | Day 14 | 12 (allotted; PROJECT_PLAN.md estimate 16) |

Net: 4 hours over PROJECT_PLAN.md estimate for items 5 (added buffer for risky novel sub-axis exploration); 8 hours under estimate for items 8 + 9 combined (Sprint 25 Day 13/14 lesson: closing days are tighter than initially estimated when retrospective is single-pass).  Total: 148 hours, within the 14×12 = 168-hour ceiling with 20-hour slack for variance.

**Headline gates (must pass on Day 14):**

- Pres_Poisson ND/AMD ≤ 0.85× literal target met (Sprint 25 baseline 0.952× default / 0.922× best opt-in via setting 13; Sprint 26 closes the remaining 7.2pp via items 5-7 combined or any one of them individually)
- HCC default-flip unblocked: `bcsstk14 sep > 0` under `SPARSE_ND_COARSENING=hcc` (item 1 fix landed)
- `sparse_eigs.c:948` UBSan log cleared: `make sanitize` clean (item 2)
- All Sprint 25 nnz_L rows bit-identical or improve under Sprint 26's production defaults
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` asserts the tightened bound (target ≤ 0.85× pinned with 2pp noise margin if items 5-7 close the literal target)
- `make wall-check` exits 0 against Sprint 25 Day 12's expanded baseline
- `make format && make lint && make test && make sanitize && make wall-check` clean on Day 14

**Risk flags:**

- **Item 5 (FINEST FM annealing / thick-restart) is the primary 0.85× candidate but novel.**  Sprint 23 + Sprint 25 have established that finest-level FM saturates at 3 passes under the existing acceptance / tie-break / restart logic; the three sub-axes (a/b/c per Sprint 25 RETROSPECTIVE.md) haven't been measured in this codebase.  If all three sub-axes wash out (consistent with Sprint 25's pattern of all three independent algorithmic axes washing out), item 5 falls short and items 6-7 inherit the 0.85× burden.  Mitigation: Day 4's per-depth profile (item 3) informs Day 6's sub-axis selection, biasing toward whichever sub-axis attacks the depth where FM cost concentrates.
- **Item 6 (geometric grid-cut) is workload-specific.**  Pres_Poisson is the canonical fit; irregular fixtures must reject the heuristic cleanly.  If the heuristic over-fires (false positives on near-regular structural-mechanics matrices like s3rmt3m3), corpus correctness regresses.  Mitigation: tunable detection thresholds + `=auto` default-off-with-explicit-opt-in semantics.
- **Item 7 (per-vertex sep scoring) is the tertiary candidate with the smallest expected per-fixture movement.**  If items 5-6 succeed, item 7 is just an additional advisory.  If items 5-6 fall short, item 7 becomes the primary fallback — and per Sprint 24 Day 6 + Sprint 25 Day 9 evidence, separator-extraction-level interventions don't move Pres_Poisson much (Sprint 24's `balanced_boundary` is +0.1pp on Pres_Poisson).  Risk: the 0.85× literal target misses for the FOURTH sprint in a row.  Mitigation: Day 13's cross-corpus re-bench is the gate; if all three new axes wash out individually, document the partial close + escalate concrete avenues (e.g. multi-graph-coarsening with HCC + HEM + connectivity-aware in parallel, or LP-relaxation-based separator) to Sprint 27.
- **Item 1 HCC sep=0 fix may regress existing fixtures.**  Option (a) HCC matching tightening is heuristic territory — could over-correct and produce worse matchings on other fixtures.  Mitigation: Day 3's corpus-bit-identical-or-improving validation gates the fix landing; revert if any fixture regresses.
- **Combination matrix at Day 13 may explode past 24 settings if all 3 new axes need 2-3 values each + interact with Sprint 25's 3 axes.**  Sprint 25 Day 9 ran 16 settings × 6 fixtures in ~6 hours; Sprint 26 needs to fit similar bench wall-time within Day 13's 12-hour budget.  Mitigation: cap at ≤ 24 representative combinations (skip uninteresting cross-products) per the Day-13 task description; capture the rest as a Sprint-27 follow-up if any settings look promising on individual-axis sweeps.

---

**Branch:** `sprint-26`
**Target:** `master`
**Inheritances:**
- Sprint 22's modular ND + multilevel partition pipeline (foundation for items 1, 5, 6)
- Sprint 23's gain-bucket FM (foundation for item 5)
- Sprint 24's `make wall-check` + `SPARSE_ND_SEP_LIFT_STRATEGY` env var (foundation for items 7, 8, 9)
- Sprint 25's three new env vars (`SPARSE_ND_COARSENING`, `SPARSE_FM_INTERMEDIATE_PASSES`, `SPARSE_ND_COARSEST_BISECTION`) + `SPARSE_ND_PROFILE` instrumentation + `wall_check_baseline.txt` Pres_Poisson ND baseline (foundation for items 1, 3, 4, 8)
