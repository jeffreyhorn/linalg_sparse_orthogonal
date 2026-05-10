# Sprint 28 Plan: Non-Pipeline-Level Pres_Poisson Closure (Sprint 27 deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-27 collectively missed (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 best opt-in 0.942× → Sprint 25 best opt-in 0.922× → Sprint 26 best opt-in 0.9217× → Sprint 27 default 0.9226×; -7.3pp gap; **fifth consecutive sprint to miss**) via non-pipeline-level interventions — Sprint 27's 24-setting × 6-fixture matrix delivered the strongest empirical evidence yet that pipeline-level interventions cannot move this fixture, so this sprint pivots to a fundamentally different approach.  Three candidate non-pipeline pivots from `SPRINT_27/RETROSPECTIVE.md` "Items deferred" #1: (a) METIS-style multi-matching coarsening, (b) geometric domain decomposition exploiting Pres_Poisson's 2D mesh metadata, (c) supernodal reordering on the elimination tree.  Day 1 picks ONE based on a cost / upside / fit-with-existing-codebase study; if all three look infeasible within Sprint 28's budget, Day 1's fallback formally calibrates the target to the empirical ~0.92× floor with documentation rather than ship another miss.  Also closes the secondary Sprint 27 deferred items: formal gain-noise variant of thick-restart (Sprint 27 Day 11 simplified to partition-state random-flip under implementation pressure; the formal version routes here), multi-strategy FM ensemble (Sprint 27 PLAN.md parking-lot item), `test_nd_pres_poisson_fill_with_leaf_amd` bound calibration to whichever outcome (0.85× if target closes, empirical-floor + 2pp documented as the bound otherwise), and Pres_Poisson ND wall further reduction conditional on item 4's outcome and real-world workload motivation.  All items routed from `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred" + "Sprint 28 inputs" #1-5.

**Starting Point:** Sprint 27 (PR #35, merged at `79f7a85`) shipped: HCC default flip `heavy_edge → hcc` (Day 2; Kuu-safe degree-CV-detection-and-HEM-fall-through; closes the second of two HCC default-flip blockers Sprint 26 Day 3 only half-fixed); `nd_base_threshold` flip 96→128 (Day 3; relaxed 2pp flip rule absorbs the s3rmt3m3 +1.05pp boundary case Sprint 26 Day 5 rejected); `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` with orthogonal `SPARSE_ND_SEP_LIFT_WEIGHT={hybrid, balance, degree}` axis (Day 4; empirical validation of Sprint 26 Day 12's "70/30 gate dominates" hypothesis — fixed-K produces 6× spread on Kuu vs <1pp under dynamic-K); three new advisory env vars (`SPARSE_FM_FINEST_STRATEGY={annealing, thick_restart}` with sub-axes `SPARSE_FM_ANNEALING_SCHEDULE` / `SPARSE_FM_THICK_RESTART_PERTURB`; `SPARSE_ND_ROOT_BISECT=spectral` with `SPARSE_ND_ROOT_BISECT_MAX_N` threshold) — ALL three regressed Pres_Poisson and stayed advisory.  Cumulative Pres_Poisson default-path achievement: 0.950× → **0.9226×** (-2.7pp); cumulative ND wall reduction 12.2 s → 10.1 s (-17 % vs Sprint 26; **-73.5 % vs Sprint 25 baseline**).  Sprint 27's largest single-fixture improvement: **Kuu -35.3 % nnz_L** under setting 18 (`--nd-threshold 256` + `per_vertex_fixed_k × hybrid`).  Headline gates: literal Pres_Poisson ≤ 0.85× target — **MISS at 0.9226×** (-7.3pp gap; **fifth consecutive sprint to miss**); test bound `test_nd_pres_poisson_fill_with_leaf_amd` tightened from 0.96× to **0.94×** (Sprint 27 default 0.923× + 2pp noise margin); `make wall-check` PASS (Pres_Poisson ND ~10 s vs 47 s baseline 1.5× ceiling = 70.5 s); `make sanitize` + `make tsan` CLEAN.  Day-13's 24-setting × 6-fixture cross-corpus matrix CONFIRMED no advisory combination beats the production default on the headline fixture.  See `SPRINT_27/RETROSPECTIVE.md`, `headline_summary.md`, per-axis decision docs (`hcc_kuu_diagnosis.md`, `nd_base_threshold_decision.md`, `per_vertex_fixed_k_decision.md`, `annealing_fm_decision.md`, `root_spectral_decision.md`, `thick_restart_decision.md`).  Sprint 27 ships strong empirical evidence that **the multilevel pipeline + leaf-AMD reaches near-optimal cuts on Pres_Poisson that pipeline-level interventions cannot improve** — Sprint 28 pivots to non-pipeline-level machinery or accepts the empirical floor.

**End State:** `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` lands the gain-bucket-noise variant Sprint 27 Day 11 deviated from; the simplified `gauss_noise` value is retained for back-compat.  `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES=baseline,fifo,annealing` (default the three) lights up the multi-strategy ensemble that runs all three FM strategies in parallel per finest-level call and picks the lowest-cut result.  Day 1 picks ONE of (a) METIS-style multi-matching coarsening, (b) geometric domain decomposition, (c) supernodal reordering on the elimination tree, OR (d) empirical-floor calibration fallback; if (a)/(b)/(c), the chosen approach lands behind a new env-var gate (`SPARSE_ND_MULTI_MATCHING={off,on}` + `SPARSE_ND_MULTI_MATCHING_K=3` for (a); `SPARSE_ND_GEOMETRIC_DD={off,on,auto}` + `sparse_reorder_nd_with_coords()` API for (b); `SPARSE_ND_SUPERNODAL_POSTORDER={off,on}` for (c)) with Pres_Poisson nnz_L moved measurably; if (d), `test_nd_pres_poisson_fill_with_leaf_amd` stays at 0.94× and the 0.85× target is formally retired in `docs/algorithm.md` after 5 sprints of empirical evidence that the pipeline cannot reach it.  `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` calibrated to the Sprint 28 outcome (≤ 0.85× pinned with 2pp margin if the literal target closes; ≤ 0.94× retained otherwise).  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_28/bench_*.{csv,txt}` plus `gain_noise_decision.md` + `ensemble_fm_decision.md` + `non_pipeline_decision.md` + `headline_summary.md`.  `docs/algorithm.md` ND subsection updates with the new env vars + per-fixture deltas; `SPRINT_22/PERF_NOTES.md` gets a "Sprint 28 closures" subsection (or `SPRINT_28/PERF_NOTES.md` if the Sprint 22 file is too long).  `SPRINT_28/RETROSPECTIVE.md` ships filled in single-pass.  Pres_Poisson ND wall further reduction (Item 6) is conditional on Item 4's outcome — only fires if Item 4 lands a structural change that opens a new wall-reduction path OR real-world workloads motivate it; otherwise routes to Sprint 29 parking-lot.

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to 144 hours — exactly the PROJECT_PLAN.md estimate; ~24-hour slack against the 14×12 = 168-hour ceiling for variance / Day-1 fallback expansion / item-4 over-budget on the chosen non-pipeline approach.  Risk concentration is item 4 (60 hrs across Days 6-10): the chosen non-pipeline approach is novel infrastructure relative to the existing multilevel pipeline; (a) METIS-style multi-matching needs careful integration with `sparse_graph_hierarchy_build`'s match-loop; (b) geometric DD needs a coordinates-input API + Lanczos at full graph size; (c) supernodal-etree reordering needs to compose with the existing `sparse_analyze` postorder pass without breaking AMD/ND ordering bit-stability.  Items 1-2 (32 hrs combined Days 2-5) ship secondary deferrals as foundation work.  Item 3 (Day 1) is the pivot decision day — Day-1 fallback to (d) empirical-floor calibration shifts items 4-6's budget into Sprint 29 wrap-up and stops Sprint 28 at item 1 + item 2 + retrospective (the freed ~84 hrs become Sprint 29 absorbed work).  Item 5 (16 hrs across Days 12-13) is cross-corpus re-bench + production-default decisions + test-bound calibration; item 7 (16 hrs across Days 13-14) is the closing-week deliverables.  Item 6 (12 hrs Day 11) is conditional — if it doesn't fire, Day 11's budget pivots to item-4 over-budget absorption or item-5 / item-7 buffer.

---

## Day 1: Sprint Kickoff — Non-Pipeline-Level Pivot Decision Study

**Theme:** Open the sprint with item 3's pivot decision day.  Sprint 27's empirical conclusion was that pipeline-level interventions cannot move Pres_Poisson; Day 1 picks ONE of (a) METIS-style multi-matching coarsening, (b) geometric domain decomposition exploiting Pres_Poisson's 2D mesh metadata, (c) supernodal reordering on the elimination tree, OR (d) empirical-floor calibration fallback if all three look infeasible within the remaining 9-day budget.  Pattern matches Sprint 26 Day 9's `geometric_cut_decision.md` rejection: spend a focused day studying the candidates, document the cost / upside / fit, pick one.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred" #1 + `headline_summary.md` "Sprint 28 inputs" + Sprint 26 Day 9's `geometric_cut_design.md` (the rejection precedent — Pres_Poisson is high-order FE-mesh, mean degree 47.3, NOT a 2D grid; geometric DD must accept this and act on the FE-mesh structure rather than a regular-grid heuristic).
2. Profile candidate (a) METIS-style multi-matching coarsening:
   - Map the integration surface: `src/sparse_graph.c::sparse_graph_hierarchy_build` (single matching loop today; replace with K=3 parallel matchings — HEM / HCC / random-shuffle); `graph_coarsest_bisection` (one trial bisection per matching to score the post-coarsening graph).
   - Lines-of-code estimate: ~200 LOC for the K-matching loop + per-matching trial-bisect + best-cut selection; ~50 LOC for the env-var gates (`SPARSE_ND_MULTI_MATCHING={off,on}` + `SPARSE_ND_MULTI_MATCHING_K=3`).  Risk: the trial-bisect overhead at every coarsening level multiplies wall by K=3; Sprint 27 wall was 10 s, so worst-case ~30 s — still under the 70.5 s wall-check ceiling.
   - Literature precedent: Karypis-Kumar 1998 §5.4 "Multiple-Matching Coarsening"; Cited 5-15 % nnz reduction on FE meshes vs single-matching; LiveJournal / Pres_Poisson-class fixtures benefit most.
3. Profile candidate (b) geometric domain decomposition:
   - Map the integration surface: new `sparse_reorder_nd_with_coords(A, coords, opts, perm)` API; `src/sparse_reorder_nd.c::nd_recurse` extended to accept `coords` and project onto the principal axis via PCA (Sprint 20 Day 6 Lanczos eigensolver on the coordinate covariance matrix); `graph_edge_separator_to_vertex_separator` (Sprint 22 Day 4) consumes the geometric vertex sets.
   - Lines-of-code estimate: ~400 LOC (new API + PCA-projection + median-bisect + recurse with coords subsetting); ~80 LOC for env-var gates (`SPARSE_ND_GEOMETRIC_DD={off,on,auto}`).  Risk: requires a coordinates input that doesn't exist for arbitrary `SparseMatrix` — Pres_Poisson can be synthesised but the corpus doesn't ship coordinates; needs a coordinate-synthesis or coordinates-from-mesh-metadata path.
   - Literature precedent: Heath-Raghavan 1995 / Hendrickson-Leland 1995 "Multilevel algorithms for partitioning irregular graphs" §3 "Geometric methods"; Cited 10-30 % nnz reduction on regular FE meshes vs algebraic ND; high-order FE meshes (Pres_Poisson is a deg-47 mesh, NOT a deg-4 2D grid) benefit less but the principal-axis projection still tracks the dominant geometric direction.
4. Profile candidate (c) supernodal reordering on the elimination tree:
   - Map the integration surface: existing `chol_csc_detect_supernodes` (Sprint 17/18) + `sparse_analyze` postorder pass; the new path computes a supernode-grouping postorder permutation that runs after the existing AMD/ND ordering and groups supernodes contiguously in the final order.
   - Lines-of-code estimate: ~150 LOC (postorder + supernode-grouping permutation + composition with AMD/ND); ~30 LOC for env-var gates (`SPARSE_ND_SUPERNODAL_POSTORDER={off,on}`).  Risk: this is fundamentally a numeric-factorization optimization (improves blocking + supernodal kernels' efficiency), NOT a fill-reduction optimization — may not move Pres_Poisson's nnz_L at all even if it lands cleanly; literature precedent (Liu 1990, Davis 2006 §6.5) shows ≤ 5 % nnz reduction.
5. Compare candidates: build `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` with a 4-row × 3-column table (cost LOC, integration risk, expected Pres_Poisson nnz_L upside) + a recommendation paragraph.  Pick ONE.
6. Pick fallback (d) if all three exceed the remaining 9-day budget OR all three show < 5 % expected Pres_Poisson upside per the literature: document the empirical-floor calibration plan (no test bound change since Sprint 27 Day 13 already tightened to 0.94×; `docs/algorithm.md` adds a paragraph noting the 0.85× target was retired after 5 sprints of evidence; the freed item-4 / item-5 / item-6 budget shifts to Sprint 29 wrap-up).
7. Run `make format && make lint && make test` (no code changes — sanity check the workspace is clean).

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` 3-candidate cost / risk / upside table + chosen-pivot recommendation (a / b / c) OR fallback-(d) empirical-floor calibration plan
- All quality checks clean (no code changes Day 1)

### Completion Criteria
- `pivot_decision_day1.md` names the chosen pivot (a / b / c / d) with documented rationale grounded in the literature precedent + integration-cost estimate
- If (a) / (b) / (c) is picked, items 4-6 implementation plan lights up for Days 6-11; if (d) is picked, items 4-6 budget formally shifts to Sprint 29 and Sprint 28 ends at Day 5 (item 1-2) + Day 14 (retrospective) with ~84 hrs slack absorbed back to Sprint 29
- `make format && make lint && make test` clean

---

## Day 2: Item 1 — Formal Gain-Noise Variant of Thick-Restart FM (Design + Implementation)

**Theme:** Sprint 27 Day 11 simplified the `gauss_noise` thick-restart variant to "random-flip with k drawn proportional to a half-Gaussian" under implementation-time pressure (`thick_restart_design.md` Day-10 deviation note).  The formal Day-10 design called for adding a Gaussian noise term to the gain-bucket pick step in `graph_refine_fm` — `noisy_gain = gain + sigma * |max_gain| * randn()` where `sigma` decreases with pass number — rather than perturbing the post-pass partition state.  Day 2 implements the formal variant.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md` "Day-10 deviation note" + `thick_restart_decision.md` "Sprint 28 inputs" — pin the Day-10 design's intent (perturb the gain-bucket comparator, not the partition state) and the empirical evidence the simplified `gauss_noise` regressed Pres_Poisson +4.7-11.5pp.
2. Implement the formal variant.  In `src/sparse_graph.c::graph_refine_fm`, when `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`, the gain-bucket pick step adds Gaussian noise: `noisy_gain = gain + sigma * |max_gain| * randn()` where `sigma_k = sigma_0 * (1 - k/K)` for K total passes (linear decay) or `sigma_0 * α^k` for α=0.5 (exponential decay; matches Sprint 27 annealing's schedule).  Default schedule: linear decay (cheaper to compute; predictable cutoff).  Use the existing xorshift32 RNG seeded per `(n, k)` pair (Sprint 27 Day 6 + Day 11 pattern) for deterministic reproduction across runs.
3. Add `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` enum value to the existing parser (Sprint 27's `random_flip` / `boundary_shuffle` / `gauss_noise` enum); the simplified `gauss_noise` value stays for back-compat (Sprint 27 captures + replays must still produce identical output).
4. Add `tests/test_graph.c::test_finest_fm_gain_noise_formal_disrupts_baseline` that runs the same fixture under `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` vs `=baseline` and asserts the cuts differ (smoke evidence the new code path is exercised); pin determinism via two-run stability check.
5. Quick Pres_Poisson + corpus interim sweep (full sweep waits for Day 12-13's cross-corpus rebench): single-fixture nnz_L delta vs Sprint 27 default + vs Sprint 27 simplified `gauss_noise`.  Goal is rough sanity: does the formal variant beat Sprint 27's regressing simplified version?  Capture interim raw output to `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_interim_day2.txt`.
6. Run `make format && make lint && make test`.

### Deliverables
- `src/sparse_graph.c` formal gain-noise variant landed behind `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`
- `tests/test_graph.c::test_finest_fm_gain_noise_formal_disrupts_baseline` passing + pinning determinism
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_interim_day2.txt` Pres_Poisson + corpus interim sweep
- All quality checks clean (gain_noise_formal-default-off path remains bit-identical)

### Completion Criteria
- Formal gain-noise variant compiles + the new test passes (deterministic + differs-from-baseline)
- Sprint 27 simplified `gauss_noise` value still produces Sprint-27-identical output (back-compat verified)
- `make format && make lint && make test` clean

---

## Day 3: Item 1 Close + Item 2 Design Kickoff — Multi-Strategy FM Ensemble

**Theme:** Close item 1 (full corpus sweep + decision doc); kick off item 2 (multi-strategy FM ensemble — Sprint 27 PLAN.md parking-lot item).  Sprint 27's Day-13 24-setting matrix established the empirical baseline; item 2 runs all three FM strategies (baseline + FIFO + annealing) per finest-level call and picks the lowest-cut result, exploring 3× the FM landscape per partition at 2-3× wall budget.

**Time estimate:** 8 hours (4 hrs item 1 close + 4 hrs item 2 design)

### Tasks
1. **Item 1 close (4 hrs):** Run the full Sprint 27 corpus sweep under `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` × 2 schedule sub-axes (linear, exponential) on the 6 fixtures (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson).  Capture nnz_L + wall to `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_sweep.txt`.
2. Write `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md`: per-fixture nnz_L delta vs Sprint 27 default + vs Sprint 27 simplified `gauss_noise`; flip-or-stay decision (advisory unless formal variant beats Sprint 27 default on Pres_Poisson AND has cleaner flip-rule application than Sprint 27's simplified gauss_noise).  Most likely outcome based on the empirical pattern: advisory only — Sprint 27 evidence shows the multilevel pipeline + leaf-AMD reaches near-optimal cuts on Pres_Poisson regardless of FM-cascade tweaks; document and move on.
3. **Item 2 design (4 hrs):** Read `docs/planning/EPIC_2/SPRINT_27/PLAN.md` parking-lot section + `SPRINT_27/RETROSPECTIVE.md` "Items deferred" #3 — pin the ensemble's contract (run baseline + FIFO + annealing in parallel per finest-level call; pick lowest-cut result; 2-3× wall budget; deterministic given fixed seeds).
4. Write `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md`: ensemble runner architecture; per-strategy result struct (cut, wall_ns, deterministic-seed); strategy-pick formula (lowest cut wins; tie → first-by-listed-order); env-var gates (`SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES=baseline,fifo,annealing` selector list); back-compat (existing strategy values unchanged).
5. Stub `tests/test_graph.c::test_finest_fm_ensemble_picks_best_strategy` as failing-as-expected: pin the pick-correctness contract (run a synthetic where one strategy provably dominates — e.g. baseline FM finds the optimal cut, FIFO/annealing land worse — and assert the ensemble picks baseline's cut).  Day 4's implementation lights it up.
6. Run `make format && make lint && make test`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_sweep.txt` 2-schedule × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md` flip-or-stay decision + per-fixture delta
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md` architecture + env-var design
- Stubbed `tests/test_graph.c::test_finest_fm_ensemble_picks_best_strategy` (failing — pin Day-4 implementation)
- All quality checks clean

### Completion Criteria
- `gain_noise_decision.md` records the flip-or-stay outcome with per-fixture bench evidence
- `ensemble_fm_design.md` documents the ensemble runner + strategy-pick contract; env-var design is explicit
- `make format && make lint && make test` clean

---

## Day 4: Item 2 — Multi-Strategy FM Ensemble Implementation

**Theme:** Implement the ensemble runner per Day 3's design.  Default selector `baseline,fifo,annealing` runs all three strategies per finest-level FM call; pick lowest-cut result; deterministic across runs given fixed env state.

**Time estimate:** 12 hours

### Tasks
1. Implement `SPARSE_FM_FINEST_STRATEGY=ensemble` dispatch in `src/sparse_graph.c::graph_refine_fm`.  On-value, parse `SPARSE_FM_ENSEMBLE_STRATEGIES` (default `baseline,fifo,annealing`); for each listed strategy, run the existing strategy code path with the partition state cloned per-strategy; track each strategy's resulting cut + wall_ns; pick the strategy with the lowest cut (tie → first listed); apply that strategy's partition assignment as the final result.
2. Add per-strategy partition-state cloning (each strategy modifies the partition independently; the ensemble runner must deep-copy the input partition before each strategy call to avoid cross-strategy contamination).
3. Add ensemble-pick instrumentation behind `SPARSE_FM_ENSEMBLE_DEBUG=1`: print per-strategy cut + wall + which-strategy-won for the partition call.  Useful for Day 5's sweep + decision-doc evidence.
4. Light up `tests/test_graph.c::test_finest_fm_ensemble_picks_best_strategy` (Day 3 stub) — must now pass with the synthetic-pick correctness assertion.
5. Add `tests/test_graph.c::test_finest_fm_ensemble_deterministic` that runs the same Pres_Poisson partition under the ensemble twice and asserts bit-identical cuts (deterministic-seed contract).
6. Quick Pres_Poisson + corpus interim sweep: `SPARSE_FM_FINEST_STRATEGY=ensemble` on 6 fixtures.  Capture interim raw output to `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_interim_day4.txt`.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_graph.c` ensemble runner landed behind `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES` selector
- `tests/test_graph.c::test_finest_fm_ensemble_picks_best_strategy` passing
- `tests/test_graph.c::test_finest_fm_ensemble_deterministic` passing
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_interim_day4.txt` 6-fixture interim sweep
- All quality checks clean (ensemble-default-off path remains bit-identical; `make wall-check` PASS — ensemble adds K-strategy multiplier but capped K=3 keeps Pres_Poisson under the 70.5 s ceiling)

### Completion Criteria
- Ensemble compiles + both new tests pass
- Sprint 27 individual strategy values still produce Sprint-27-identical output (back-compat verified)
- `make wall-check` PASS — Pres_Poisson under ensemble ≤ 30 s (3× Sprint 27 default + slack)
- `make format && make lint && make test && make wall-check` clean

---

## Day 5: Item 2 Close — Ensemble Sweep + Decision

**Theme:** Run the full Sprint 27 corpus sweep under the ensemble + variants of the strategy selector list; pick the corpus-wide-best ensemble configuration; write the flip-or-stay decision.  The ensemble must close Pres_Poisson meaningfully (≤ 0.92× — better than Sprint 27 default 0.923×) without wall blowup past 2× of Sprint 27's ~10 s, OR document as advisory.

**Time estimate:** 4 hours

### Tasks
1. Run the full corpus sweep: `SPARSE_FM_FINEST_STRATEGY=ensemble` × {selector list variants: `baseline,fifo,annealing` (default); `baseline,annealing` (drop FIFO if FIFO never wins); `fifo,annealing` (drop baseline if baseline never wins); `baseline,fifo,annealing,thick_restart` (4-way ensemble — Sprint 27's thick_restart strategy joins)} × 6 fixtures.  Capture nnz_L + wall + strategy-win-counts (via `SPARSE_FM_ENSEMBLE_DEBUG=1`) to `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_sweep.txt`.
2. Write `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_decision.md`: per-fixture nnz_L delta vs Sprint 27 default; per-selector wall delta; which strategy wins most often per fixture (the strategy-win-counts inform whether the ensemble is doing real work or just picking baseline 95 % of the time).  Flip rule: ensemble flips to default if (a) Pres_Poisson improves ≥ 1pp vs Sprint 27 default AND (b) no fixture regresses past 5pp AND (c) wall stays under 2× of Sprint 27 default.  Most likely outcome: advisory only (Sprint 27 evidence pattern + ensemble wall cost).
3. If flipped: change `parse_finest_fm_strategy()` default in `src/sparse_graph.c` from baseline to ensemble; update `sparse_reorder_nd_internal.h` doc-comment; re-run `make wall-check` to verify Pres_Poisson ND wall stays under the per-key threshold.
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_sweep.txt` 4-selector × 6-fixture sweep with strategy-win-counts
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_decision.md` flip-or-stay decision + per-selector evidence
- `src/sparse_graph.c` default updated (if flipped) + `sparse_reorder_nd_internal.h` comment block updated
- All quality checks clean

### Completion Criteria
- Sweep covers 4 selectors × 6 fixtures + strategy-win-counts captured
- Decision doc records the flip-or-stay choice with per-fixture flip-rule application
- If flipped: `make wall-check` Pres_Poisson ND ≤ 2× Sprint 27 default
- `make format && make lint && make test && make wall-check` clean

---

## Day 6: Item 4 — Implementation Day 1 (Chosen Non-Pipeline Approach)

**Theme:** Begin Day-1's chosen non-pipeline approach.  Concrete tasks vary by Day-1 pick: (a) METIS-style multi-matching → start with the K-matching loop scaffolding in `sparse_graph_hierarchy_build`; (b) geometric DD → start with the `sparse_reorder_nd_with_coords()` API + PCA-projection skeleton; (c) supernodal-etree reordering → start with the postorder-permutation computation.  This day is foundation: data structures, env-var gates, default-off skeleton; full implementation lands across Days 7-10.

**Time estimate:** 12 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` chosen-pivot rationale + integration-cost estimate.
2. Per pick, set up the Day-1 scaffolding:
   - **(a) METIS-style multi-matching:** add `SPARSE_ND_MULTI_MATCHING={off,on}` + `SPARSE_ND_MULTI_MATCHING_K=3` env-var parsers; stub the K-matching loop in `sparse_graph_hierarchy_build` (default behaviour bit-identical when off); design the per-matching trial-bisection scoring.
   - **(b) Geometric DD:** add `sparse_reorder_nd_with_coords(A, coords, opts, perm)` public API + `SPARSE_ND_GEOMETRIC_DD={off,on,auto}` env-var parser; stub the PCA-projection helper that operates on a `(n, 2)` or `(n, 3)` coordinate matrix.
   - **(c) Supernodal-etree reordering:** add `SPARSE_ND_SUPERNODAL_POSTORDER={off,on}` env-var parser; stub the postorder-permutation computation that consumes `chol_csc_detect_supernodes`'s output and produces a permutation that groups supernodes contiguously.
3. Stub failing-as-expected tests for the Day-1 chosen approach:
   - **(a):** `tests/test_graph.c::test_multi_matching_picks_best_cut_synthetic` — synthetic where one matching produces a clearly better cut than the other two; assert the K-matching loop picks that one.
   - **(b):** `tests/test_reorder_nd.c::test_geometric_dd_pres_poisson_synth` — synthesise Pres_Poisson with known (x, y) coordinates; assert the principal-axis projection bisection produces a cut that respects the coordinate structure.
   - **(c):** `tests/test_reorder_nd.c::test_supernodal_postorder_etree_contract` — assert the postorder permutation groups supernodes contiguously per `chol_csc_detect_supernodes`'s output.
4. Verify default-off path: under env-var unset, the new code path doesn't fire; existing tests still pass bit-identically.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Scaffolding for the Day-1 chosen non-pipeline approach: env-var parsers, public API stubs (if applicable), default-off skeleton in the appropriate `src/` files
- Stubbed failing-as-expected test(s) for the Day-1 chosen approach (Day 7-10 implementation lights them up)
- All quality checks clean (default-off path remains bit-identical)

### Completion Criteria
- Env-var parsers compile + accept the expected values + default-off behaviour preserved
- Stubbed test(s) compile and trip with the expected failure message
- `make format && make lint && make test && make wall-check` clean

---

## Day 7: Item 4 — Implementation Day 2

**Theme:** Continue Day-1's chosen non-pipeline approach.  Day 7 lands the core algorithmic step (per pick: K-matching score-and-pick / coordinate-PCA-projection / supernodal-postorder permutation).

**Time estimate:** 12 hours

### Tasks
1. Per Day-1 pick, implement the core algorithmic step:
   - **(a) METIS-style multi-matching:** implement the K=3 parallel matchings (HEM / HCC / random-shuffle) inside the coarsening loop; for each, run a trial bisection on the resulting coarse graph + score by min-cut; pick the matching with the lowest min-cut score; proceed with coarsening using the picked matching.  Determinism: each matching uses a fixed seed (HEM = 0; HCC = 1; random-shuffle = `(level, n) % UINT32_MAX`).
   - **(b) Geometric DD:** implement the PCA-projection — given a `(n, 2)` or `(n, 3)` coordinate matrix, compute the covariance matrix, find the largest eigenvalue's eigenvector via Sprint 20 Day 6's Lanczos (full graph at this size — Pres_Poisson n=14 822 is within the spectral threshold); project all vertices onto the principal axis; bisect at the median value; lift the boundary as separator.
   - **(c) Supernodal-etree reordering:** implement the postorder permutation that groups supernodes contiguously; compose with the existing AMD/ND ordering by applying the supernode-grouping permutation as a post-pass (the AMD/ND ordering is the input; the supernode-grouping permutation reorders within supernodes only, preserving the AMD/ND elimination tree structure).
2. Light up the Day-6 stubbed test(s) — must now pass:
   - **(a):** `test_multi_matching_picks_best_cut_synthetic`.
   - **(b):** `test_geometric_dd_pres_poisson_synth`.
   - **(c):** `test_supernodal_postorder_etree_contract`.
3. Quick Pres_Poisson + corpus interim measurement: run `bench_reorder --only Pres_Poisson` under the new path with `--skip-factor`; capture nnz_L + wall to `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day7.txt`.  Goal is "is the path moving Pres_Poisson nnz_L meaningfully?" — directional, not headline.
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Core algorithmic step landed for Day-1's chosen non-pipeline approach
- Day-6 stubbed test(s) now passing
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day7.txt` Pres_Poisson + corpus interim measurement
- All quality checks clean

### Completion Criteria
- Core algorithmic step compiles + the new test(s) pass
- Pres_Poisson interim measurement is captured (delta vs Sprint 27 default reported with sign + magnitude)
- `make format && make lint && make test && make wall-check` clean

---

## Day 8: Item 4 — Implementation Day 3

**Theme:** Continue the non-pipeline approach.  Day 8 lands the remaining algorithmic steps + edge-case handling per the Day-1 pick.

**Time estimate:** 12 hours

### Tasks
1. Per Day-1 pick, land the remaining algorithmic steps + edge cases:
   - **(a) METIS-style multi-matching:** add the trial-bisection scoring's wall budget guard (skip the trial bisection if cumulative trial-bisect time exceeds 50 % of single-matching wall; fall back to single-matching with the first listed matching to avoid wall blowup); add per-matching seed determinism verification.
   - **(b) Geometric DD:** add the recursive call (each half of the geometric bisection recurses with the corresponding coordinate subset); add the small-subgraph base case (when `n <= sparse_reorder_nd_base_threshold`, fall through to leaf-AMD per Sprint 23 Day 7's pattern); add the auto-mode that detects whether coordinates are usable (coordinates non-NULL + dimensions = 2 or 3 + no NaN/Inf).
   - **(c) Supernodal-etree reordering:** integrate with `sparse_analyze`'s postorder pass — the supernode-grouping permutation runs after `sparse_analyze` returns, before the user calls `sparse_factor_numeric`; ensure the permutation composition with AMD/ND keeps the elimination tree's parent pointers consistent.
2. Add corpus-safety tests:
   - **(a):** `tests/test_graph.c::test_multi_matching_corpus_no_smaller_fixture_regress` — assert nnz_L on small fixtures (nos4, bcsstk04) doesn't regress past 5pp.
   - **(b):** `tests/test_reorder_nd.c::test_geometric_dd_no_coords_falls_through` — assert `sparse_reorder_nd` (without coords) still produces Sprint-27-identical output when `SPARSE_ND_GEOMETRIC_DD=auto` (the auto detection rejects no-coords inputs).
   - **(c):** `tests/test_chol_csc.c::test_supernodal_postorder_residual_unchanged` — assert numeric factorization residuals stay within Sprint 27's ≤ 1e-8 bound after the supernode-grouping permutation.
3. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Edge-case handling + corpus-safety integration for Day-1's chosen approach
- New corpus-safety tests passing
- All quality checks clean

### Completion Criteria
- Edge cases compile + corpus-safety tests pass
- Default-off path still bit-identical
- `make format && make lint && make test && make wall-check` clean

---

## Day 9: Item 4 — Implementation Day 4

**Theme:** Continue the non-pipeline approach.  Day 9 runs a focused Pres_Poisson sweep — does the path actually close the 0.85× target or land short?  This is the gating measurement for Day-10's flip-or-stay decision.

**Time estimate:** 12 hours

### Tasks
1. Run the full Sprint 27 corpus sweep under Day-1's chosen non-pipeline approach: 6 fixtures × the relevant env-var combinations (per pick: K ∈ {2, 3, 5} for (a); on/auto for (b); on/off composed with AMD/ND for (c)).  Capture nnz_L + wall to `docs/planning/EPIC_2/SPRINT_28/non_pipeline_sweep.txt`.
2. Compute the headline-target verdict: did Pres_Poisson land ≤ 0.85× of AMD nnz_L?  If yes — the literal target is closed for the first time in 5 sprints; document as a flip candidate (Days 12-13 cross-corpus matrix verifies the flip-rule). If no — record the achieved ratio + delta from target; document the per-fixture deltas + corpus safety.
3. Tune the env-var defaults per pick (e.g. for (a), pick the K value that minimizes Pres_Poisson nnz_L without smaller-fixture regress past 5pp; for (b), pick auto vs on per the corpus safety; for (c), pick the supernode-grouping granularity).
4. Run `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` under the tuned default — does it pass at the Sprint 27 0.94× bound?  If the literal 0.85× target closed, prepare the test bound tightening for Day 13 (don't tighten yet — wait for the cross-corpus matrix).
5. Capture interim decision direction in `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day9.md`: headline verdict (closed / partial / missed) + per-fixture per-tuning deltas + corpus safety + Day 10's close-out plan.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_sweep.txt` 6-fixture × env-var-tuning sweep
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day9.md` headline verdict + Day-10 close-out plan
- Tuned env-var defaults committed (default-off / default-auto per pick + the tuning value)
- All quality checks clean

### Completion Criteria
- Sweep covers the relevant env-var combinations × 6 fixtures with raw output
- `non_pipeline_interim_day9.md` records the literal 0.85× target outcome (closed / partial / missed) with bench evidence
- Tuned defaults compile + corpus safety still holds
- `make format && make lint && make test && make wall-check` clean

---

## Day 10: Item 4 Close — Decision Doc + Final Tuning

**Theme:** Close item 4 with a decision doc that records the flip-or-stay outcome for the chosen non-pipeline approach.  If the literal 0.85× target closed, prepare the production-default flip for Day 13; if partial, document as advisory; if missed, route to Sprint 29 with empirical evidence that even non-pipeline approaches don't move Pres_Poisson on this codebase + corpus.

**Time estimate:** 12 hours

### Tasks
1. Finalise the Day-9 sweep + tuning into `docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md`: pick the corpus-wide-best tuning + Pres_Poisson-headline-best tuning; document the flip-or-stay rationale.  Three outcome classes:
   - **Closed:** Pres_Poisson ≤ 0.85× of AMD; flip the env-var default to on (or auto for (b)); `non_pipeline_decision.md` documents the close + cross-corpus flip-rule plan for Day 13.
   - **Partial:** Pres_Poisson moved measurably (≥ 1pp better than Sprint 27 default 0.923×) but didn't reach 0.85×; advisory only — the new env var ships at default-off for the production default; `non_pipeline_decision.md` documents the partial close + Sprint 29+ routing if a fundamentally-different next-step is motivated.
   - **Missed:** Pres_Poisson didn't move (≤ 1pp delta from Sprint 27 default); advisory only or removed depending on corpus safety; `non_pipeline_decision.md` documents the empirical evidence + concedes the literal 0.85× target appears to be at-or-below the empirical floor for this codebase + corpus.
2. Add `tests/test_reorder_nd.c::test_non_pipeline_pres_poisson_close_to_target` (commented-out RUN_TEST line if Sprint 28's verdict misses; uncommented + tightened bound if Sprint 28 closes the target).  Mirror Sprint 27's pattern of "ship the test scaffolding even if the assertion is commented out — future sprints can light it up if a closing combination emerges".
3. If the verdict is "closed", prepare the test bound tightening: `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from 0.94× to 0.85× + 2pp = 0.87× (or whatever Sprint 28's achieved + 2pp).  Don't commit the tightening today — wait for Day 13's cross-corpus matrix to confirm the flip-rule is clean; today just stage the diff.
4. If the verdict is "missed" / "partial", begin drafting the Day-14 retrospective's "Items deferred" section noting that even non-pipeline approaches don't move Pres_Poisson on this codebase + corpus; route the literal 0.85× target to Sprint 29 with documented empirical evidence.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md` flip-or-stay verdict + per-tuning evidence
- `tests/test_reorder_nd.c::test_non_pipeline_pres_poisson_close_to_target` scaffolding (commented-out or uncommented per verdict)
- Staged test bound tightening diff (if closed; commit waits for Day 13)
- Draft Day-14 retrospective notes (if partial / missed)
- All quality checks clean

### Completion Criteria
- `non_pipeline_decision.md` records the verdict (closed / partial / missed) with bench evidence + flip-or-stay rationale
- Test scaffolding lands per the verdict
- `make format && make lint && make test && make wall-check` clean

---

## Day 11: Item 6 (Conditional) — Pres_Poisson ND Wall Further Reduction

**Theme:** Conditional item.  Sprint 27 cut Pres_Poisson ND wall -73.5 % vs Sprint 25 baseline (38 s → 10 s); further reduction would need pipeline restructuring.  Day 11 only fires if (a) item 4's chosen non-pipeline approach lands a structural change that opens a new wall-reduction path, OR (b) real-world Sprint-28-cycle workloads surface a need.  If neither trigger fires, Day 11's 12 hrs becomes slack for item-4 over-budget absorption or item-5 / item-7 buffer.

**Time estimate:** 12 hours (conditional — slips to slack if trigger conditions don't fire)

### Tasks
1. Check the trigger conditions:
   - **(a)** Did item 4's non-pipeline approach land a structural change?  E.g. (b) geometric DD pre-empts the multilevel pipeline entirely on coordinate-input fixtures — wall reduction comes from skipping coarsening + FM at the root level.  (c) supernodal-etree reordering doesn't change the partition wall (it's a post-permutation), so (c) doesn't trigger Item 6.  (a) METIS-style multi-matching adds wall (K-matching multiplier) — also doesn't trigger Item 6 (regression, not improvement).
   - **(b)** Did real-world Sprint-28-cycle workloads surface a need?  Check the GitHub issues + any user reports since Sprint 27 merged at `79f7a85` for "ND wall too slow" feedback.
   - If neither trigger fires, end this day at task 1 + commit no-op + reallocate the 11 remaining hours: 4 hrs to item-4 over-budget absorption (if any), 4 hrs to item-5 prep (Day 12's cross-corpus matrix scaffolding), 3 hrs to item-7 prep (Day 13's docs + retrospective drafting).
2. **If trigger fires** — pick a candidate optimisation:
   - **Deepen-coarsen-and-reuse-symbolic:** analyze the elimination tree once across nested partition calls; the recursive ND descent currently re-analyses each subgraph; share the etree analysis across the parent call's children.
   - **Parallelise FM gain-bucket picks via OpenMP:** Sprint 21 Day 5-6's reorth-OpenMP pattern is the template; gate via `SPARSE_OPENMP` + `SPARSE_FM_OMP_MIN_N` threshold (default 5000).
   - **Batch-process leaf-AMD calls:** the recursive ND descent emits leaf-AMD calls one at a time; batching them into a single AMD invocation amortizes setup cost.
3. **If trigger fires** — implement the picked optimisation; run a Pres_Poisson ND wall measurement under `make wall-check`; capture to `docs/planning/EPIC_2/SPRINT_28/wall_reduction_decision.md` + flip-or-stay decision (flip if wall improves ≥ 10 % without nnz_L regression).
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- If trigger fires: implementation + `docs/planning/EPIC_2/SPRINT_28/wall_reduction_decision.md` decision doc + flip outcome
- If trigger doesn't fire: Day 11 budget reallocated to item-4 over-budget / item-5 prep / item-7 prep
- All quality checks clean

### Completion Criteria
- Trigger condition checked + documented (fired / no-op)
- If fired: wall improvement ≥ 10 % vs Sprint 27 baseline without nnz_L regression
- `make format && make lint && make test && make wall-check` clean

---

## Day 12: Item 5 — Cross-Corpus Re-Bench Day 1

**Theme:** Run the Sprint 28 cross-corpus matrix combining the new Sprint 28 axes (formal gain-noise thick-restart, ensemble FM, item-4's chosen non-pipeline approach) with the Sprint 27 Day-13 24-setting baseline.  Cap at ≤24 representative settings matching Sprint 27's pattern; capture raw output to `bench_day12_combinations.csv` + `.txt`.

**Time estimate:** 8 hours

### Tasks
1. Design the ≤24-setting matrix.  Template from Sprint 27 Day 13's combinations:
   - 1: Sprint 26 default (HEM, t=96) — for historical reference.
   - 2: Sprint 27 default (HCC + t=128) — the inherited baseline.
   - 3-6: Sprint 28 item-4 axis variants (per Day-10 pick: e.g. K ∈ {2, 3, 5} for (a); on/auto for (b); on/off composed with AMD/ND for (c)).
   - 7-9: Sprint 28 item-1 axis variants (`gain_noise_formal` × {linear, exponential schedule}).
   - 10-13: Sprint 28 item-2 axis variants (ensemble × {`baseline,fifo,annealing` (default), `baseline,annealing`, `fifo,annealing`, `baseline,fifo,annealing,thick_restart` 4-way}).
   - 14-20: Sprint 27 advisory recipes (Kuu opt-in `--nd-threshold 256 + per_vertex_fixed_k × hybrid`, etc. — verify Sprint 28 changes don't regress them).
   - 21-24: Sprint 28 stack combinations (item-4 × item-1, item-4 × item-2, item-4 × item-1 × item-2 — does stacking the new axes help or hurt?).
2. Run the matrix via `bench_reorder.c` per Sprint 27 Day 13's pattern; save raw output to `docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.csv` + `.txt`.  Use real numeric measurements throughout — no `?` placeholders (Sprint 27 Day 13 review-comment lesson).
3. Begin building `docs/planning/EPIC_2/SPRINT_28/headline_summary.md`: top 5 by Pres_Poisson nnz_L; top 5 by corpus-wide nnz_L; identify the intersection.  Day 13 finishes the summary + production-default decisions + test-bound calibration.
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.csv` + `.txt` ≤24-combination × 6-fixture sweep with real numeric values throughout
- `docs/planning/EPIC_2/SPRINT_28/headline_summary.md` started (top-5 lists; full verdict in Day 13)
- All quality checks clean

### Completion Criteria
- Matrix covers ≤24 combinations × 6 fixtures with raw csv + txt captures (no `?` placeholder rows)
- `headline_summary.md` records the top-5-by-Pres_Poisson + top-5-by-corpus-wide intersection
- `make format && make lint && make test && make wall-check` clean

---

## Day 13: Item 5 Close + Item 7 Prep — Production-Default Decisions + Test-Bound Calibration

**Theme:** Close item 5 with the production-default decisions + test-bound calibration; begin item 7 prep (drafting `docs/algorithm.md` updates + retrospective scaffolding).

**Time estimate:** 12 hours (8 hrs item 5 close + 4 hrs item 7 prep)

### Tasks
1. **Item 5 close (8 hrs):** Finalise `docs/planning/EPIC_2/SPRINT_28/headline_summary.md` with the Sprint 28 verdict on the literal 0.85× target (closed / partial / missed) + per-axis flip outcomes.
2. Production-default decisions per axis:
   - `SPARSE_FM_THICK_RESTART_PERTURB`: flip default to `gain_noise_formal` if Day 3's decision rule cleared (most likely advisory only — see Day 3 most-likely outcome).
   - `SPARSE_FM_FINEST_STRATEGY`: flip default to `ensemble` if Day 5's decision rule cleared (most likely advisory only — wall cost typically blocks the flip).
   - Item-4 axis (per pick: `SPARSE_ND_MULTI_MATCHING` / `SPARSE_ND_GEOMETRIC_DD` / `SPARSE_ND_SUPERNODAL_POSTORDER`): flip default to on / auto if Day 9-10's verdict was "closed" or "partial with clear flip-rule application".
3. Calibrate `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd`:
   - **If item 4 closed (Pres_Poisson ≤ 0.85×):** tighten from 0.94× to Sprint 28's-achieved + 2pp (e.g. 0.87× if achieved 0.85×); commit the previously-staged Day 10 diff.
   - **If item 4 partial / missed:** keep the bound at 0.94× (Sprint 27's tightening); document in `headline_summary.md` that the literal 0.85× target is formally retired with Sprint 28's empirical evidence (5 sprints + non-pipeline approaches).
4. Light up the `tests/test_reorder_nd.c::test_non_pipeline_pres_poisson_close_to_target` test from Day 10 if the verdict is "closed" + the cross-corpus matrix confirms the flip-rule.
5. **Item 7 prep (4 hrs):** Draft `docs/algorithm.md` ND subsection updates listing the new Sprint 28 env vars + their per-fixture deltas; supersede Sprint 27's "0.85× literal target route to Sprint 28+" caveat with the actual Sprint 28 outcome.  Stub `docs/planning/EPIC_2/SPRINT_28/RETROSPECTIVE.md` with the section skeleton (What shipped / Items deferred / Sprint 29 inputs / Process lessons / Day-by-day capsule / Day-budget vs estimate / DoD verification).
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_28/headline_summary.md` Sprint 28 verdict + per-axis flip outcomes
- Production defaults flipped per Day-13 verdict (item-1 / item-2 / item-4 axes)
- `test_nd_pres_poisson_fill_with_leaf_amd` bound calibrated (tightened or kept per verdict)
- `test_non_pipeline_pres_poisson_close_to_target` lit up if verdict is "closed"
- `docs/algorithm.md` ND subsection draft updates
- `docs/planning/EPIC_2/SPRINT_28/RETROSPECTIVE.md` stubbed
- All quality checks clean

### Completion Criteria
- `headline_summary.md` records the literal 0.85× target outcome (closed via what / partial / retired) with bench evidence
- Test bound calibrated with bench evidence
- `RETROSPECTIVE.md` stub has the section skeleton ready for Day 14's single-pass fill
- `make format && make lint && make test && make wall-check` clean

---

## Day 14: Item 7 Close — Tests + Docs + Retrospective

**Theme:** Single-pass closing day per Sprints 25/26/27 retrospective lesson ("single Day-14 retro that absorbs the Day-13 work matches the actual time spent").  New tests validation + `docs/algorithm.md` ND subsection update + `SPRINT_22/PERF_NOTES.md` "Sprint 28 closures" subsection + `SPRINT_28/RETROSPECTIVE.md` filled in.

**Time estimate:** 12 hours

### Tasks
1. Validate the Day 12-13 tests are passing under the Sprint 28 default-flipped configuration (re-run `make test && make sanitize && make tsan && make wall-check`).  If any tests trip — particularly the item-4 close-to-target tests — root-cause and either fix in-place (small fix) or document as Sprint 29+ routing in `RETROSPECTIVE.md`.
2. Finalise `docs/algorithm.md` ND subsection updates from Day 13's draft: describe the Sprint 28 env vars (`SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`, `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES`, item-4's chosen-pivot env vars per Day-1 pick) + their per-fixture deltas; supersede Sprint 27's "0.85× literal target route to Sprint 28+" caveat with the actual Sprint 28 outcome (closed / partial-with-Sprint-29-routing / retired-with-empirical-floor).
3. Append a "Sprint 28 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open `docs/planning/EPIC_2/SPRINT_28/PERF_NOTES.md` if the Sprint 22 file is too long after Sprint 27's closures section).  Per-axis closure summary + cross-corpus best-combination citation + any production-default flips' before/after numbers.
4. Fill in `docs/planning/EPIC_2/SPRINT_28/RETROSPECTIVE.md` single-pass.  Sections:
   - **Status:** Sprint 28 final outcome — items closed + items deferred + literal 0.85× verdict.
   - **Goal recap:** the Sprint 28 charter.
   - **Definition of Done checklist:** per-item ✓/✗ + reference commits.
   - **Final metrics:** ND/AMD nnz(L) ratios Sprint 22 → Sprint 28 (extending the Sprint 27 retrospective table); Pres_Poisson ND wall (if Item 6 fired); largest single-fixture improvement.
   - **Performance highlights:** any production default flips + their cumulative impact.
   - **What went well / What surprised us / What didn't go well**.
   - **Items deferred (route to Sprint 29+):** any items that didn't close + reason.
   - **Lessons:** Sprint 28-specific (per-day or per-item).
   - **Sprint 29 inputs:** concrete handoff items.
   - **Day-by-day capsule** + **Day-budget vs estimate** + **DoD verification** tables.
   - **Acknowledgements**.
5. Update `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 28 section: status flip from "in flight" to "Complete"; record actual hours vs estimated 144; cite the literal 0.85× verdict; route any partial-close items to the Sprint 29 section.
6. Final `make format && make lint && make test && make sanitize && make tsan && make wall-check` — must all be clean before PR.
7. Open Sprint 28 PR; request review; address any reviewer feedback (estimated 2-3 hrs of buffer in the 12-hour budget).

### Deliverables
- `docs/algorithm.md` ND subsection updated with Sprint 28 env vars + per-fixture deltas
- `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or `SPRINT_28/PERF_NOTES.md`) "Sprint 28 closures" subsection appended
- `docs/planning/EPIC_2/SPRINT_28/RETROSPECTIVE.md` filled in single-pass with all 12 sections
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 28 status flip to "Complete" + actuals vs estimate
- Sprint 28 PR opened + (ideally) merged
- All quality checks clean (format, lint, test, sanitize, tsan, wall-check)

### Completion Criteria
- All Sprint 28 tests pass under the default-flipped configuration
- `docs/algorithm.md` describes the new env vars accurately + cites the per-fixture deltas with bench evidence
- `RETROSPECTIVE.md` records the literal 0.85× target outcome (closed / partial / retired) with concrete Sprint 29+ routing if needed
- `make format && make lint && make test && make sanitize && make tsan && make wall-check` all clean
- Sprint 28 PR is mergeable (CI green, reviewer feedback addressed)

---

## Total Time Budget

| Day | Theme | Hours |
|-----|-------|-------|
| 1 | Item 3 — Non-pipeline-level pivot decision study | 8 |
| 2 | Item 1 — Formal gain-noise thick-restart variant (design + impl) | 8 |
| 3 | Item 1 close (4h) + Item 2 design kickoff (4h) | 8 |
| 4 | Item 2 — Multi-strategy FM ensemble implementation | 12 |
| 5 | Item 2 close — ensemble sweep + decision | 4 |
| 6 | Item 4 — Implementation Day 1 (chosen non-pipeline approach scaffolding) | 12 |
| 7 | Item 4 — Implementation Day 2 (core algorithmic step) | 12 |
| 8 | Item 4 — Implementation Day 3 (edge cases + corpus safety) | 12 |
| 9 | Item 4 — Implementation Day 4 (Pres_Poisson sweep + interim verdict) | 12 |
| 10 | Item 4 close — decision doc + final tuning | 12 |
| 11 | Item 6 (conditional) — Pres_Poisson ND wall further reduction | 12 |
| 12 | Item 5 — Cross-corpus re-bench Day 1 | 8 |
| 13 | Item 5 close (8h) + Item 7 prep (4h) — production-default decisions + test-bound calibration | 12 |
| 14 | Item 7 close — tests + docs + retrospective | 12 |

**Total: 144 hours** — exactly the PROJECT_PLAN.md estimate; ~24-hour slack against the 14×12 = 168-hour ceiling for variance / Day-1 fallback expansion / item-4 over-budget on the chosen non-pipeline approach.  Day 11's Item 6 is conditional — if its trigger conditions don't fire, the 12 hrs becomes slack absorbed into item-4 over-budget / item-5 prep / item-7 prep.  Day-1 fallback to (d) empirical-floor calibration shifts items 4-6's budget (84 hrs) to Sprint 29 wrap-up and stops Sprint 28 at item 1 (12 hrs) + item 2 (20 hrs) + item 7 (16 hrs) = 48 hrs spread across Days 2-5 + Day 14.
