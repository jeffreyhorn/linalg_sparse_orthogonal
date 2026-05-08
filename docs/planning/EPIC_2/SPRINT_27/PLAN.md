# Sprint 27 Plan: ND Fill-Quality Closure II (Sprint 26 deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-26 collectively missed (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 best opt-in 0.942× → Sprint 25 best opt-in 0.922× → Sprint 26 best opt-in 0.9217× unchanged; -7.2pp gap; **fourth consecutive sprint to miss**).  Sprint 26's 96+ measurements across Days 7-12 closed the algorithmic-tie-break and per-phase-scoring exploration spaces without moving the headline; Sprint 27 pivots to **structural interventions at the multilevel pipeline level** — root-level Fiedler-axis bisection (extending Sprint 25's coarsest-level spectral) and annealing-acceptance FM at the finest level (now affordable under Sprint 26 Day 5's -68 % Pres_Poisson wall improvement).  Also closes the secondary deferred items: HCC Kuu-safe matching variant (Day 13's combination matrix surfaced Kuu HCC-alone +14.6pp regress as the SECOND HCC default-flip blocker, after Sprint 26 Day 3's bcsstk14 sep=0 fix unlocked the FIRST); tunable fixed-K per-vertex selection (Day 12's finding that the dynamic-K + 70/30 balance gate masks per-vertex weight-scheme differences); larger `nd_base_threshold` beyond 96 with relaxed flip rule (Day 5's strict ≥1pp regression cap stopped t=96 from going to t=128); thick-restart-style FM as a conditional fallback if annealing falls short.  All items routed from `docs/planning/EPIC_2/SPRINT_26/RETROSPECTIVE.md` "Items deferred" + per-axis decision docs.

**Starting Point:** Sprint 26 (PR #34, merged at `511941f`) shipped: `nd_base_threshold` default flipped 32→96 (Day 5; Pres_Poisson ND wall 38.1 s → 12.2 s, -67.9 %, nnz_L bit-stable -0.21pp); HCC bcsstk14 sep=0 blocker FIXED (Day 3) via `_Thread_local force_hem_override` fall-back path in `sparse_graph_partition` (bcsstk14 under HCC now produces sep=97); `sparse_eigs.c:948` UBSan division-by-zero log CLEARED (Day 1 one-line `|| anchor == 0.0` guard); three new advisory env vars: `SPARSE_FM_FINEST_STRATEGY={baseline,fifo}` (Day 7; bucket-tie-break FIFO via tails[]; parser already accepts `annealing` and `thick_restart` values that fall through to baseline), `SPARSE_ND_SEP_LIFT_STRATEGY={smaller_weight,balanced_boundary,per_vertex,per_vertex_balance,per_vertex_degree}` (Day 10/12; per-vertex separator scoring with 3 weight schemes — empirically converge to bit-identical outputs on 5 of 6 fixtures because dynamic-K + 70/30 balance gate dominates), per-recursion-depth `SPARSE_ND_PROFILE` extension (Day 4; identifies depths 6-9 as the partition-cost concentration zone with 60-200 ms constant overhead floor).  Headline gates: literal Pres_Poisson ≤ 0.85× target — **MISS at 0.9217×** (-7.2pp gap; Sprint 25 setting 13 best opt-in unchanged); Pres_Poisson < Sprint 25 default — PASS (-0.2pp via Day 5 flip); smaller-fixture corpus safety — PASS; HCC bcsstk14 fix — PASS; UBSan log — CLEAR; test bound — STAY at 0.96× (Items 5-7 didn't move default); `make wall-check` PASS; `make sanitize` + `make tsan` CLEAN.  Day 13's 12-setting × 6-fixture combination matrix surfaced the SECOND HCC default-flip blocker (Kuu HCC-alone +14.6pp regress; CV=0.425 highest in corpus); Day 11/12 measured all three per-vertex weight schemes producing bit-identical Pres_Poisson outputs because the dynamic-K + 70/30 balance gate dominates the score formula.  Sprint 26 ships strong empirical evidence that **tie-break-and-scoring-style interventions don't move Pres_Poisson** — the 0.85× target requires structural intervention at the multilevel pipeline level, not the FM-cascade or separator-extraction level.  See `SPRINT_26/RETROSPECTIVE.md`, `headline_summary.md`, `nd_base_threshold_decision.md`, `finest_fm_decision.md`, `geometric_cut_design.md` (Item 6 rejection rationale on Day 9), `per_vertex_sep_decision.md` for per-day decision rationale.

**End State:** `SPARSE_FM_FINEST_STRATEGY=annealing` lights up the parser-accepts-but-falls-through-to-baseline value Sprint 26 Day 6 left stubbed; annealing accepts worsening moves with `exp(-Δgain / T)` probability where T decreases with pass number (default schedule selectable from {linear, exponential, cosine}).  `SPARSE_ND_ROOT_BISECT={multilevel (default), spectral}` env var extends Sprint 25 Day 6-8's coarsest-level spectral to the ROOT level: full-graph Fiedler-vector projection + median bisection, with multilevel-pipeline fallback when Lanczos fails or `n` exceeds a tunable threshold.  HCC Kuu-safe matching variant (option (a) adaptive weighting on high-CV fixtures OR (b) per-edge weight-equality break — pick on Day 1) lands; if flip-rule clean, HCC default flips from `heavy_edge` to `hcc`.  `nd_base_threshold` re-sweeps under a 2pp relaxed flip rule; if t ≥ 128 passes the relaxed rule, default flips again.  Fixed-K per-vertex selection mode (`SPARSE_ND_SEP_LIFT_K=N` or `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`) lands so the 3 per-vertex weight schemes finally differentiate.  `SPARSE_FM_FINEST_STRATEGY=thick_restart` (conditional fallback) lights up the second parser stub — Days 10-12 budget conditional on items 4-5 outcomes.  Pres_Poisson ND/AMD reaches ≤ 0.85× of AMD (or partial close documented + Sprint 28+ routed if structural pipeline-level interventions also fall short).  `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` tightens from Sprint 24's 0.96× to whatever default ratio is achieved + 2pp noise margin.  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_27/bench_*.{csv,txt}` plus `annealing_fm_decision.md` + `root_spectral_decision.md` + `hcc_kuu_safe_decision.md` + `headline_summary.md`.  `docs/algorithm.md` ND subsection updates to describe the new env vars + their per-fixture deltas; `SPRINT_22/PERF_NOTES.md` gets a "Sprint 27 closures" subsection.  `SPRINT_27/RETROSPECTIVE.md` ships filled in (single Day-14 retro per Sprints 25/26 retrospective lesson).

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to 152 hours — exactly the PROJECT_PLAN.md estimate; ~16-hour slack against the 14×12 = 168-hour ceiling for variance / multi-strategy ensemble parking-lot work if budget permits.  Risk concentration is item 5 (32 hrs across Days 7-9): root-level spectral bisection on the FULL Pres_Poisson Laplacian (n=14 822) is novel infrastructure — Lanczos at this size is 5-10 s wall, and the median-cut fallback path needs careful tuning.  Item 4 (28 hrs across Days 5-7) is the PRIMARY 0.85× candidate; annealing has the broadest applicability of the FINEST-FM sub-axes Sprint 26 Day 6 stubbed.  Item 6 (24 hrs across Days 10-12) is the conditional fallback; if items 4 + 5 land Pres_Poisson ≤ 0.85×, item 6's budget pivots to multi-strategy-ensemble parking-lot work + extra item 7/8 buffer.  Items 1-3 (36 hrs combined Days 1-4) are foundation work that ships in the first week; items 7 + 8 (32 hrs combined Days 12-14) are the closing-week deliverables.

---

## Day 1: Sprint Kickoff — HCC Kuu Profiling + Matching Trace + Fix Design

**Theme:** Open the sprint with item 1's diagnostic phase.  Sprint 26 Day 13's combination matrix surfaced Kuu HCC-alone +14.6pp regress as the SECOND HCC default-flip blocker (after Sprint 26 Day 3 fixed the FIRST — bcsstk14 sep=0).  Capture HCC's matching choices on Kuu under Sprint 26 Day 1's `SPARSE_HCC_DEBUG` instrumentation; profile Kuu's degree distribution to quantify how the bimodal CV=0.425 (highest in corpus) interacts with HCC's `min(deg(u), deg(v))` weighting; pick fix option (a) adaptive weighting OR (b) per-edge weight-equality break.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_26/RETROSPECTIVE.md` "Items deferred" + `headline_summary.md` "Default-flip rule application" + `hcc_sep_zero_diagnosis.md` (Sprint 26 Day 2's diagnosis pattern is the template for Day 1's Kuu diagnosis).
2. Capture HCC's full coarsening trace on Kuu: `SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=hcc build/bench_reorder --only Kuu --skip-factor 2> /tmp/hcc_kuu_trace.txt`.  Inspect the cmap at each coarsening level; identify which level the matching biases noticeably toward boundary-with-boundary pairs.
3. Diff against the (already-passing) HEM trace: `SPARSE_HCC_DEBUG=1 SPARSE_ND_COARSENING=heavy_edge build/bench_reorder --only Kuu --skip-factor 2> /tmp/hem_kuu_trace.txt`.  Quantify the per-level matching divergence (count of vertices that match different partners under HCC vs HEM).
4. Profile Kuu's structural property triggering the regress: histogram of degree distribution (CV target reproduction = 0.425); per-edge `min(deg(u), deg(v))` distribution; per-vertex high-degree-vs-low-degree neighbour counts.  Document in `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md`.
5. Pick fix option:
   - **(a) Adaptive HCC weighting**: detect CV > 0.30 at coarsening time; soften the `min(deg)` factor (e.g. blend with sqrt(deg) or fall through to HEM entirely on high-CV fixtures).  Pros: localised; doesn't touch the matching loop's tie-break.  Cons: heuristic threshold; may not catch all cases.
   - **(b) Per-edge weight-equality break**: when two edges have identical HCC scores, prefer the edge whose endpoints have closer degrees (smaller `|deg(u) - deg(v)|`).  Pros: structural; works on any high-CV fixture without explicit detection.  Cons: per-edge comparator change; correctness validation needs more care.
   - Pick based on the diagnosis: if Kuu's regress is concentrated at one coarsening level under high-CV vertices, prefer (a) for surgical applicability; if the regress is distributed across levels, prefer (b) for structural robustness.  Document the choice + rationale in `hcc_kuu_diagnosis.md` "Day 1 fix-option selection".
6. Stub `tests/test_graph.c::test_hcc_kuu_no_default_flip_blocker` as a failing-as-expected test that runs the Kuu coarsening + reorder under `SPARSE_ND_COARSENING=hcc` and asserts nnz_L ≤ Sprint 26 default + 5pp.  Currently fails; Day 2's fix lights it up.
7. Run `make format && make lint && make test`.

### Deliverables
- `/tmp/hcc_kuu_trace.txt` + `/tmp/hem_kuu_trace.txt` matching-choice traces (intermediate; not committed)
- `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md` Kuu degree-distribution profile + per-level HCC-vs-HEM divergence + fix-option (a vs b) decision
- Stubbed `tests/test_graph.c::test_hcc_kuu_no_default_flip_blocker` (failing — pin Day-2 fix)
- All quality checks clean (HCC-default-off path remains bit-identical)

### Completion Criteria
- `hcc_kuu_diagnosis.md` names the exact structural property of Kuu that triggers HCC's +14.6pp regress + cites the coarsening level(s) where the divergence concentrates
- Fix option (a or b) selected with documented rationale
- `test_hcc_kuu_no_default_flip_blocker` compiles and trips with "expected nnz_L ≤ default + 5pp, got worse" or analogous failure message
- `make format && make lint && make test` clean

---

## Day 2: HCC Kuu-Safe Fix Implementation + HCC Default-Flip Re-Attempt

**Theme:** Implement the Day-1 chosen fix; validate that Kuu nnz_L stays within flip-rule budget under `SPARSE_ND_COARSENING=hcc`; corpus-bit-identical-or-better verification; re-attempt the HCC default flip (Sprint 26 Day 3's flip blocked by Kuu and bcsstk14; Sprint 26 only fixed bcsstk14).  Day 2's gate is "fix lands + Kuu test passes + flip-rule outcome documented".

**Time estimate:** 8 hours

### Tasks
1. Implement the Day-1 chosen fix (option (a) adaptive HCC weighting OR option (b) per-edge weight-equality break) per `hcc_kuu_diagnosis.md`.
2. Run `tests/test_graph.c::test_hcc_kuu_no_default_flip_blocker` — must now pass.
3. Run `tests/test_graph.c::test_partition_bcsstk14_smoke` under `SPARSE_ND_COARSENING=hcc` — must still pass (Sprint 26 Day 3's bcsstk14 fix must not regress under the Kuu-targeted change).
4. Run `tests/test_reorder_nd.c::test_nd_kuu_fill_vs_amd` under `SPARSE_ND_COARSENING=hcc` — Kuu nnz_L should be ≤ Sprint 26 default-strategy baseline + 5pp.
5. Re-run Sprint 26 Day 13's 12-setting × 6-fixture combination matrix capture (subset: settings involving HCC) under the new fix.  Check the flip rule: HCC default flip is good if (a) Pres_Poisson improves ≥ 1pp AND (b) no smaller-fixture regress past 5pp.  If clean, change `parse_coarsening_strategy()` default in `src/sparse_graph.c` from `COARSENING_HEAVY_EDGE` to `COARSENING_HCC`; commit the flip.  If Kuu still blocks (or another fixture surfaces a new blocker), revert + document.
6. Update `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md` with the post-fix verification results + flip outcome.  If flipped, append a "Default-flip rule satisfied" subsection citing the Day-2 combination matrix delta.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_graph.c` HCC Kuu-safe fix landed (option (a) or (b) per Day-1 design)
- `tests/test_graph.c::test_hcc_kuu_no_default_flip_blocker` enabled and passing
- `tests/test_graph.c::test_partition_bcsstk14_smoke` still passing under `SPARSE_ND_COARSENING=hcc` (no Sprint 26 Day 3 regress)
- HCC default-flip outcome (flipped or stayed) documented in `hcc_kuu_diagnosis.md` post-fix verification section
- All quality checks clean

### Completion Criteria
- Kuu nnz_L ≤ Sprint 26 default + 5pp under `SPARSE_ND_COARSENING=hcc`
- bcsstk14 still produces sep > 0 under HCC (no Sprint 26 Day 3 regress)
- Combination-matrix subset confirms flip-rule outcome with bench evidence
- `make format && make lint && make test && make wall-check` clean

---

## Day 3: `nd_base_threshold` Relaxed-Flip-Rule Re-Sweep

**Theme:** Sprint 26 Day 5 swept t ∈ {32, 48, 64, 96, 128} on the full corpus and found t=96 was the maximum threshold satisfying the strict ≥1pp regression cap (s3rmt3m3 +1.05pp at t=128 just past the gate).  Re-evaluate with a 2pp relaxed cap; if t=128 (or higher) passes the relaxed rule, flip the default again.  Also explore per-fixture-class advisory thresholds.  Day 3's gate is "decision doc + flip-or-stay outcome".

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_decision.md` "Day 5 sweep findings" + Sprint 26 Day 5's full sweep table.  Identify the per-fixture nnz_L deltas at t=128 vs t=96 (s3rmt3m3 +1.05pp was the blocker; check what other fixtures showed).
2. Re-sweep `sparse_reorder_nd_base_threshold` ∈ {96 (current default), 128, 192, 256} on the full Sprint 26 corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson) via the existing `bench_reorder --nd-threshold N` flag.  Capture nnz_L delta + wall-time delta + leaf-AMD call count delta (use `SPARSE_ND_PROFILE=1`).  Save raw output to `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_resweep.txt`.
3. Apply the relaxed flip rule: flip the default if (a) Pres_Poisson wall improves ≥ 5 % AND (b) no fixture regresses nnz_L past 2pp (was 1pp Sprint 26).  Document the decision in `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md` with the per-fixture sweep table + flip outcome (flipped to N or stayed at 96 + per-fixture-class advisory if a per-workload tuning emerges).
4. If flipped: change `sparse_reorder_nd_base_threshold = N` in `src/sparse_reorder_nd.c`; update `sparse_reorder_nd_internal.h` doc-comment to match (the Sprint 26 Day 5 wording stays mostly as-is, just the numeric default changes); re-run `make wall-check` to verify Pres_Poisson ND wall stays under the per-key threshold.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_resweep.txt` 4-threshold × 6-fixture × 2-metric capture
- `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md` per-fixture summary + relaxed-flip-rule outcome
- `src/sparse_reorder_nd.c` default updated (if flipped) + `sparse_reorder_nd_internal.h` comment block updated
- All quality checks clean

### Completion Criteria
- Sweep covers 4 thresholds × 6 fixtures + leaf-AMD call counts captured
- Decision doc justifies the flip-or-stay choice with per-fixture relaxed-flip-rule application
- If flipped: `make wall-check` Pres_Poisson ND ≤ baseline (and ideally lower if the larger-leaf-AMD saving materializes)
- `make format && make lint && make test && make wall-check` clean

---

## Day 4: Fixed-K Per-Vertex Selection Mode + 3-Scheme Sweep

**Theme:** Sprint 26 Day 12's empirical finding: all 3 per-vertex weight schemes (`per_vertex`, `per_vertex_balance`, `per_vertex_degree`) converge to bit-identical outputs on 5 of 6 fixtures because the dynamic-K + 70/30 balance gate dominates the score formula.  Add a fixed-K selection mode where K = min(boundary_count[0], boundary_count[1]) instead of dynamic balance-respecting termination; sweep the 3 weight schemes under fixed-K; verify they differentiate.

**Time estimate:** 12 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_decision.md` "Day 12 finding" + Sprint 26 Day 10/12 implementation in `src/sparse_graph.c::graph_separator_lift_per_vertex`.  Identify the dynamic-K termination point (the 70/30 balance gate) and the score formula's per-vertex weight component.
2. Add a fixed-K selection mode.  Two surface options, pick one:
   - **(a) New env var** `SPARSE_ND_SEP_LIFT_K=N` (parse with strtol + range-check + fallback to dynamic-K).  Adds an orthogonal axis to the existing `SPARSE_ND_SEP_LIFT_STRATEGY` enum.
   - **(b) New strategy enum value** `per_vertex_fixed_k` that selects K = min(boundary_count[0], boundary_count[1]).  Adds to the existing strategy axis.
   - Pick (b) for symmetry with the existing per-vertex strategy values; document briefly.
3. Implement the fixed-K mode.  When `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`, the lift loop terminates after exactly K vertices have been lifted (where K = min boundary count), regardless of the 70/30 balance state.  The per-vertex score formula stays the same; only the termination predicate changes.
4. Sweep the 3 weight schemes under fixed-K on the Sprint 26 corpus: `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` × {`SPARSE_ND_SEP_LIFT_WEIGHT={uniform, balance, degree}`}.  (If the existing implementation hardcodes the weight scheme into the strategy enum value, refactor: the weight scheme becomes orthogonal — new env var `SPARSE_ND_SEP_LIFT_WEIGHT` — so it can stack with fixed-K.)  Capture per-fixture nnz_L deltas; save to `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_sweep.txt`.
5. Add `tests/test_graph.c::test_per_vertex_fixed_k_differs_from_dynamic_k` that runs the same Pres_Poisson partition under dynamic-K and fixed-K and asserts the cuts differ (smoke-level evidence the new mode is exercised).
6. Document the 3-scheme empirical differentiation in `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md`: which scheme wins per-fixture under fixed-K, whether the differentiation is large enough to motivate a default-strategy flip, corpus-safety check.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `src/sparse_graph.c` fixed-K selection mode behind `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` (and possibly `SPARSE_ND_SEP_LIFT_WEIGHT={uniform, balance, degree}` for orthogonal weight axis)
- `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_sweep.txt` 3-weight × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md` differentiation analysis + flip-or-stay outcome
- `tests/test_graph.c::test_per_vertex_fixed_k_differs_from_dynamic_k` smoke test
- All quality checks clean

### Completion Criteria
- `per_vertex_fixed_k_decision.md` shows the 3 weight schemes producing different outputs on at least 2 fixtures (Sprint 26 Day 12 had bit-identical on 5 of 6)
- New env-var-gated path is bit-identical to current master when off
- `test_per_vertex_fixed_k_differs_from_dynamic_k` passes and pins the differs-from-dynamic-K behaviour
- `make format && make lint && make test && make wall-check` clean

---

## Day 5: Annealing FM — Design + Skeleton

**Theme:** Sprint 26 Day 6's design REJECTED annealing for cost reasons (+20-50 % wall expansion).  Day 5's `nd_base_threshold` flip dropped Pres_Poisson ND wall 38 s → 12 s, making annealing's wall budget affordable now.  Day 5's gate is "design doc + temperature schedule sub-axis selection + skeleton — implementation lands Day 6".

**Time estimate:** 12 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md` "Annealing rejected for cost" + `finest_fm_decision.md` "Sprint 27 inputs".  Pin the parser-already-accepts-but-falls-through-to-baseline state of the `SPARSE_FM_FINEST_STRATEGY=annealing` value (Sprint 26 Day 6 stubbed it).
2. Read Karypis-Kumar 1998 §6.3 (or whichever Kirkpatrick-1983 / Cherkassky-1995 reference best fits the FM-with-annealing setting); pin the standard temperature schedule + acceptance probability formulae.  Annealing acceptance: accept worsening moves with `P = exp(-Δgain / T)`; reject with `1 - P`.  T decreases per pass (typical schedule: T_0 = max gain in first pass; halve per pass; cutoff at T < 1 — moves below cutoff are rejected as in baseline).
3. Pick the temperature schedule sub-axis.  Three plausible variants:
   - **(a) Linear**: T_k = T_0 × (1 - k/K) for K total passes; simplest; predictable cutoff.
   - **(b) Exponential**: T_k = T_0 × α^k where α ≈ 0.5; classical; faster convergence; risk of premature freeze.
   - **(c) Cosine**: T_k = T_0/2 × (1 + cos(πk/K)); slow start, fast end; popular in modern training schedules.
   - Pick (b) exponential for Day 5 default (matches the classical formulation and is the cheapest to compute); leave (a) and (c) as Day 6 sweep dimensions if Day 6 timing permits.  Document in `docs/planning/EPIC_2/SPRINT_27/annealing_fm_design.md`: schedule selection + acceptance contract + sweep dimensions.
4. Light up the `SPARSE_FM_FINEST_STRATEGY=annealing` dispatch in `src/sparse_graph.c::graph_refine_fm` (the parser already accepts the value; Sprint 26 Day 6's stub falls through to baseline).  On-value, dispatch to a new `graph_refine_fm_annealing` entry point that calls the existing bucket-FM scaffolding with annealing-acceptance overlaid.  Out-of-range / non-numeric / missing → default `baseline` (Sprint 26 Day 6 behaviour preserved).
5. Stub `tests/test_graph.c::test_finest_fm_annealing_accepts_worsening` as failing-as-expected: pin the expected behaviour (under `SPARSE_FM_FINEST_STRATEGY=annealing` on a fixture where baseline FM would saturate, annealing accepts at least one worsening move per pass).  Day 6's implementation lights it up.
6. Add a `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential (default), cosine}` env var as the temperature-schedule sub-axis (Day 6 sweep dimension).  Stub the cosine + linear branches; default exponential.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_design.md` schedule selection + algorithm contract + env-var design
- `SPARSE_FM_FINEST_STRATEGY=annealing` dispatch in `src/sparse_graph.c` (skeleton; falls through to baseline-ish behaviour pending Day 6's implementation)
- `SPARSE_FM_ANNEALING_SCHEDULE` env var stub (3 values; default exponential)
- Stubbed `tests/test_graph.c::test_finest_fm_annealing_accepts_worsening` (failing — pin Day-6 implementation)
- All quality checks clean (annealing-default-off path remains bit-identical)

### Completion Criteria
- `annealing_fm_design.md` justifies the schedule selection against Sprint 26 Day 4 per-depth profile + KK1998 §6.3 reference
- Annealing dispatch compiles + Sprint 23 baseline path stays bit-identical (default off)
- Day-5 stub trips with "annealing not yet implemented; falls through" or analogous skip message
- `make format && make lint && make test && make wall-check` clean

---

## Day 6: Annealing FM — Implementation + Initial Pres_Poisson Measurement

**Theme:** Implement the Day-5 chosen exponential-schedule annealing acceptance end-to-end; instrument worsening-move counts; capture an interim Pres_Poisson ND nnz_L measurement under `SPARSE_FM_FINEST_STRATEGY=annealing` to gauge progress toward the 0.85× target.  Day 6's gate is "library compiles + Sprint 26 default-path corpus tests bit-identical (env var off) + annealing produces measurable behaviour change when on".

**Time estimate:** 12 hours

### Tasks
1. Implement annealing-acceptance per Day-5's `annealing_fm_design.md`.  Re-use Sprint 23 Day 11's multi-pass FM scaffolding + Sprint 23 Day 10's gain-bucket structure (`src/sparse_graph_fm_buckets.h`); only the per-move acceptance and per-pass temperature update logic changes.  Track worsening-move count per pass (stderr emit under a new `SPARSE_FM_ANNEALING_DEBUG=1` flag, default off).
2. Implement the linear + cosine temperature-schedule branches behind `SPARSE_FM_ANNEALING_SCHEDULE` (Day 5's stubs).  All three should produce different per-pass T sequences; document the expected sequence in the implementation file's header comment.
3. Run the existing 39 partition tests under `SPARSE_FM_FINEST_STRATEGY=baseline` (default) — should all pass bit-identically to current master.
4. Run the same tests under `SPARSE_FM_FINEST_STRATEGY=annealing` — most will pass (annealing produces a valid cut, just a different one); some determinism contracts may need re-validation under the random acceptance.  Triage: if `test_partition_determinism_*` fails, root-cause and fix on Day 7 (likely answer: add a `SPARSE_FM_ANNEALING_SEED` env var so the random acceptance is reproducible).
5. Capture an interim Pres_Poisson ND nnz_L measurement under `SPARSE_FM_FINEST_STRATEGY=annealing` (and the 3 schedule sub-axes) to gauge progress against the 0.85× target.  Save to `/tmp/annealing_pres_poisson_day6.txt` or `docs/planning/EPIC_2/SPRINT_27/annealing_interim_day6.txt`.
6. Light up the Day-5 stub `test_finest_fm_annealing_accepts_worsening`: real assertions for the worsening-move count > 0 under at least one pass.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Annealing acceptance landed in `src/sparse_graph.c::graph_refine_fm_annealing` behind `SPARSE_FM_FINEST_STRATEGY=annealing` env-var gate
- Linear + exponential + cosine temperature schedules wired behind `SPARSE_FM_ANNEALING_SCHEDULE` (default exponential)
- Existing 39 partition tests pass bit-identically under default-off; annealing produces a valid (different) cut when on
- Interim Pres_Poisson ND nnz_L measurement (saved to commit message or `annealing_interim_day6.txt`)
- All quality checks clean

### Completion Criteria
- Annealing accepts ≥ 1 worsening move per pass (verified by `test_finest_fm_annealing_accepts_worsening`)
- Sprint 23 baseline path bit-identical when env var off (verified by 39 partition tests)
- Pres_Poisson ND nnz_L under annealing measured + saved (used as input to Day 7's flip decision)
- `make format && make lint && make test && make wall-check` clean

---

## Day 7: Annealing FM Closing + Root-Level Spectral Bisection Design Kickoff

**Theme:** Close out item 4: full corpus sweep under annealing × 3 schedules (Day 7 first 4 hours); apply the flip rule; commit the flip if Pres_Poisson lands ≤ 0.85× without smaller-fixture regress past 5pp.  Then kick off item 5 (root-level spectral bisection) design (Day 7 last 8 hours): read the Sprint 25 Day 6-8 coarsest-spectral implementation; design the root-level extension; pick the Lanczos invocation strategy.

**Time estimate:** 12 hours (4h item 4 close + 8h item 5 design kickoff)

### Tasks
1. **Item 4 closing (4h).**  Run the full Sprint 26 corpus under `SPARSE_FM_FINEST_STRATEGY=annealing` × `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential, cosine}` (3 schedules × 6 fixtures = 18 measurements).  Capture per-fixture nnz_L deltas + wall-time deltas vs Sprint 26 default.  Save to `docs/planning/EPIC_2/SPRINT_27/annealing_fm_sweep.txt`.
2. Apply the annealing flip rule: flip `SPARSE_FM_FINEST_STRATEGY` default to `annealing` (with the best-performing schedule) if (a) Pres_Poisson lands ≤ 0.85× of AMD nnz_L AND (b) no smaller-fixture regress past 5pp.  Document in `docs/planning/EPIC_2/SPRINT_27/annealing_fm_decision.md`: per-fixture sweep table + flip outcome (default flipped or stays advisory).
3. If flipped: tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from 0.96× to whatever Pres_Poisson achieved + 2pp noise margin (Day 13's bound-tightening gate may revise this further).  If item 4 misses the 0.85× target, item 5 (root-level spectral) carries the headline-target weight; item 4 ships as advisory.
4. **Item 5 design kickoff (8h).**  Re-read `src/sparse_graph.c::graph_coarsest_bisection` + Sprint 25 Day 6-8's spectral path (`SPARSE_ND_COARSEST_BISECTION=spectral`).  Pin the Lanczos invocation pattern, the Fiedler-vector extraction, and the median-bisection-with-balance-fallback flow.
5. Design the root-level extension.  Two key decisions:
   - **(i) Lanczos cost characterisation**: at n=14 822 (Pres_Poisson), Lanczos on the full graph Laplacian is ~5-10 s wall (Sprint 26 Day 4's profile didn't include a root-level Lanczos timing — Day 7 should run a quick one-off measurement).  This is comparable to Sprint 26 Day 5's full ND wall (12 s) — plausibly affordable; document the tradeoff.
   - **(ii) Fallback threshold**: at what n should `SPARSE_ND_ROOT_BISECT=spectral` fall back to the multilevel pipeline?  Plausible bound: n > 100 000 makes Lanczos > 30 s (per Sprint 21 Day 5 scaling); n ≤ 1 000 makes the multilevel pipeline cheap enough not to bother with spectral.  Pick a tunable threshold via `SPARSE_ND_ROOT_BISECT_MAX_N=N` env var (default 50 000 — wide enough to include Pres_Poisson; small enough to not blow up on production-scale fixtures).
6. Sketch the `SPARSE_ND_ROOT_BISECT={multilevel (default), spectral}` env-var gate in `src/sparse_reorder_nd.c::sparse_reorder_nd` (the entry point — root-level intercept is at the top of the recursion).  Default `multilevel` (Sprint 26 behavior); on-value `spectral` invokes the new path.  Document the design in `docs/planning/EPIC_2/SPRINT_27/root_spectral_design.md`.
7. Stub `tests/test_reorder_nd.c::test_nd_root_spectral_pres_poisson_smoke` as failing-as-expected: pin the expected `SPARSE_ND_ROOT_BISECT=spectral` produces a different cut than `multilevel` on Pres_Poisson.  Day 8-9's implementation lights it up.
8. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_sweep.txt` 3-schedule × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_decision.md` flip outcome + best-schedule selection
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_design.md` Lanczos-cost characterisation + fallback-threshold design
- Stubbed `tests/test_reorder_nd.c::test_nd_root_spectral_pres_poisson_smoke` (failing — pin Day 8-9's implementation)
- If annealing flip lands: tightened `test_nd_pres_poisson_fill_with_leaf_amd` bound + production-default change
- All quality checks clean

### Completion Criteria
- `annealing_fm_decision.md` documents the per-fixture × per-schedule outcome + flip-rule application
- `root_spectral_design.md` justifies the Lanczos-cost-vs-multilevel-cost tradeoff with a Day-7 one-off Lanczos timing measurement
- Root-level spectral env var stub compiles + multilevel-default path stays bit-identical
- `make format && make lint && make test && make wall-check` clean

---

## Day 8: Root-Level Spectral Bisection — Design Close + Implementation

**Theme:** Close out the Day-7 design (formalise the `SPARSE_ND_ROOT_BISECT_MAX_N` threshold + Lanczos-failure fallback path); implement the root-level spectral bisection; verify the multilevel-default path stays bit-identical when off.  Day 8's gate is "library compiles + spectral path produces a Fiedler-vector-based cut on Pres_Poisson + multilevel default unchanged".

**Time estimate:** 12 hours

### Tasks
1. Finalise the design: lock the `SPARSE_ND_ROOT_BISECT_MAX_N` threshold via a Day-8 one-off corpus Lanczos timing pass (run `sparse_eigs_sym` shift-invert at σ ≈ 0+ε on each corpus fixture's full Laplacian; record wall-time + Lanczos convergence count).  Save to `docs/planning/EPIC_2/SPRINT_27/root_spectral_lanczos_timing.txt`.
2. Implement the root-level spectral bisection.  Entry point: at the top of `sparse_reorder_nd::nd_recurse` (or wherever the recursion's root call lives), if `SPARSE_ND_ROOT_BISECT=spectral` AND `n ≤ SPARSE_ND_ROOT_BISECT_MAX_N`, build the graph Laplacian, call `sparse_eigs_sym` for the second-smallest eigenvector, project all vertices onto the Fiedler vector, bisect at the median value (with 60/40 balance-tolerance fallback to multilevel — same fallback shape as Sprint 25 Day 8's coarsest-spectral path); emit the boundary edge as the separator.  If Lanczos fails to converge, fall back to multilevel pipeline (no error returned upstream).
3. Add the `SPARSE_ND_ROOT_BISECT` env var read at the top of `sparse_reorder_nd` (matching Sprint 25 Day 6 `SPARSE_ND_COARSEST_BISECTION` env-var read pattern).  Out-of-range / non-numeric / missing → default `multilevel`.
4. Run the existing 39 partition + reorder tests under `SPARSE_ND_ROOT_BISECT=multilevel` (default) — should all pass bit-identically to current master (the gate's off-by-default branch is bit-identical to Sprint 26).
5. Run the same tests under `SPARSE_ND_ROOT_BISECT=spectral` — Pres_Poisson + smaller fixtures should produce valid (possibly different) cuts.  Verify the determinism contracts hold (Lanczos is deterministic given a fixed seed; `test_partition_determinism_*` should pass).
6. Light up the Day-7 stub `test_nd_root_spectral_pres_poisson_smoke`: real assertions for "spectral path produces a different cut than multilevel on Pres_Poisson" + "Pres_Poisson nnz_L under spectral root-bisect is ≤ multilevel default".
7. Run `make format && make lint && make test && make wall-check` (the wall-check includes the Pres_Poisson ND threshold; spectral root-bisect may reduce or increase ND wall — Day 9's measurement decides).

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_lanczos_timing.txt` 6-fixture root-Laplacian Lanczos timing capture
- Root-level spectral bisection in `src/sparse_reorder_nd.c` behind `SPARSE_ND_ROOT_BISECT=spectral` env-var gate
- Existing 39 partition + reorder tests pass bit-identically under `SPARSE_ND_ROOT_BISECT=multilevel` (default off)
- Spectral path produces a valid cut on Pres_Poisson + smaller fixtures
- All quality checks clean

### Completion Criteria
- Lanczos at root level on Pres_Poisson converges within 100 iterations + < 30 s wall
- Multilevel-default path stays bit-identical when env var off (verified by 39 partition + reorder tests)
- `test_nd_root_spectral_pres_poisson_smoke` passes with the differs-from-multilevel + nnz_L-not-worse assertions
- `make format && make lint && make test && make wall-check` clean

---

## Day 9: Root-Level Spectral — Pres_Poisson Sweep + Decision

**Theme:** Run the full Sprint 26 corpus under `SPARSE_ND_ROOT_BISECT=spectral`; capture per-fixture nnz_L + wall deltas; apply the flip rule; decide whether to flip the default.  This is Sprint 27's SECONDARY 0.85× Pres_Poisson candidate (annealing FM is PRIMARY); Day 9 either confirms a closing of the headline target or routes the work to Sprint 28+.

**Time estimate:** 12 hours

### Tasks
1. Run the full Sprint 26 corpus (nos4, bcsstk04, Kuu, bcsstk14, s3rmt3m3, Pres_Poisson) under `SPARSE_ND_ROOT_BISECT=spectral` (with the Day-8 default `SPARSE_ND_ROOT_BISECT_MAX_N=50 000` — Pres_Poisson at n=14 822 fits; smaller fixtures fit; bcsstk14 at n=1 806 fits; s3rmt3m3 at n=5 357 fits).  Capture per-fixture nnz_L deltas + wall-time deltas vs Sprint 26 default + Sprint 27 Day 7 annealing-best.
2. If Pres_Poisson under root-spectral lands ≤ 0.85× of AMD nnz_L: this is the headline closure.  Apply the flip rule for `SPARSE_ND_ROOT_BISECT` default: flip if (a) Pres_Poisson lands ≤ 0.85× AND (b) no smaller-fixture regress past 5pp AND (c) wall stays under per-key threshold (root-level Lanczos may add 5-10 s on Pres_Poisson; verify `make wall-check` passes).  If flip-rule clean, change `SPARSE_ND_ROOT_BISECT` default to `spectral`; commit the flip.
3. If Pres_Poisson under root-spectral does NOT land ≤ 0.85×: combine with Day 7's annealing-best result — does the *combination* `SPARSE_FM_FINEST_STRATEGY=annealing + SPARSE_ND_ROOT_BISECT=spectral` get there?  Run that combination on Pres_Poisson; if combined lands ≤ 0.85×, the headline closes via the combined opt-in (default-flip outcomes individual; combined ships as advisory recipe in `docs/algorithm.md`).
4. Document in `docs/planning/EPIC_2/SPRINT_27/root_spectral_decision.md`: per-fixture sweep table + per-fixture wall delta + flip outcome + headline status (closed via item 4 alone / closed via item 5 alone / closed via item 4+5 combined / still missing — route item 6 thick-restart as Sprint 27's tertiary candidate).
5. Update `docs/algorithm.md` ND subsection with a "Sprint 27 root-level spectral" paragraph documenting the new env var + per-fixture deltas.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_sweep.txt` 6-fixture corpus sweep + (item 4 + item 5) combined sweep
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_decision.md` per-fixture summary + flip outcome + headline status
- If headline closes: `docs/algorithm.md` updated; `test_nd_pres_poisson_fill_with_leaf_amd` bound tightened to ≤ 0.85× + 2pp
- All quality checks clean

### Completion Criteria
- Sweep covers 6 fixtures × {root-spectral alone, root-spectral + annealing-best combined}
- Decision doc records the headline status (closed / partial / missing → item 6 routing)
- If flipped: `make wall-check` Pres_Poisson ND ≤ per-key threshold (Lanczos cost included in the new wall)
- `make format && make lint && make test && make wall-check` clean

---

## Day 10: Thick-Restart FM — Design (Conditional)

**Theme:** Item 6 (thick-restart FM) is the conditional fallback for Sprint 27's headline.  If items 4-5 closed Pres_Poisson ≤ 0.85× (Day 9 verdict), Day 10's budget pivots to the multi-strategy-ensemble parking-lot work + extra item 7/8 buffer.  If items 4-5 missed, Day 10 proceeds with item 6 design.

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_27/root_spectral_decision.md` (Day 9's verdict).  If Pres_Poisson closed ≤ 0.85× via items 4 / 5 / 4+5: pivot Day 10 to the multi-strategy-ensemble parking-lot work documented in PROJECT_PLAN.md "Multi-strategy FM ensemble (parking lot)".  Skip tasks 2-7 of this day; document the pivot in `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md` "Day 10 pivot — items 4-5 closed the headline".  If items 4-5 missed: proceed with tasks 2-7.
2. Re-read `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md` "Thick-restart rejected for cost (2-3× wall expansion)".  Sprint 26 Day 5's wall improvement (Pres_Poisson 38 → 12 s under default-flipped t=96) makes thick-restart 24 → 36 s plausibly affordable on Pres_Poisson; smaller fixtures get a smaller absolute cost.
3. Pin the parser-already-accepts-but-falls-through state of the `SPARSE_FM_FINEST_STRATEGY=thick_restart` value (Sprint 26 Day 6 stubbed it).  Design contract: thick-restart tracks the GLOBAL-best cut across all FM passes; restart each pass from the saved anchor with random perturbation rather than building only on the previous pass's result.
4. Pick the perturbation sub-axis.  Three plausible variants:
   - **(a) Random-vertex flip (k vertices)**: flip k randomly-selected vertices' partition assignments before each pass; k = 1 % of n.
   - **(b) Boundary-vertex shuffle**: among the boundary vertices (vertices on edges crossing the cut), shuffle their part assignments.
   - **(c) Gaussian noise on gain estimates**: perturb the initial-pass gain values with N(0, σ) noise; σ = max gain / 4.
   - Pick (a) random-vertex flip for Day 10 default (simplest; cheapest to compute); leave (b) and (c) as Day 11 sweep dimensions if Day 11 timing permits.  Document in `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md`.
5. Light up the `SPARSE_FM_FINEST_STRATEGY=thick_restart` dispatch in `src/sparse_graph.c::graph_refine_fm` (the parser already accepts the value; Sprint 26 Day 6's stub falls through to baseline).  On-value, dispatch to a new `graph_refine_fm_thick_restart` entry point.
6. Stub `tests/test_graph.c::test_finest_fm_thick_restart_returns_to_anchor` as failing-as-expected: pin the expected behaviour (under `SPARSE_FM_FINEST_STRATEGY=thick_restart`, the global-best cut is non-decreasing across passes; pass N+1's start state matches the global-best from passes 1..N modulo the perturbation).  Day 11's implementation lights it up.
7. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md` perturbation sub-axis selection + algorithm contract + env-var design (OR Day-10 pivot doc if items 4-5 closed the headline)
- `SPARSE_FM_FINEST_STRATEGY=thick_restart` dispatch in `src/sparse_graph.c` (skeleton; falls through to baseline-ish pending Day 11's implementation)
- Stubbed `tests/test_graph.c::test_finest_fm_thick_restart_returns_to_anchor` (failing — pin Day 11's implementation)
- All quality checks clean (thick-restart-default-off path remains bit-identical)

### Completion Criteria
- `thick_restart_design.md` justifies the perturbation sub-axis selection OR documents the Day-10 pivot to multi-strategy-ensemble work if Day 9 closed the headline
- Thick-restart dispatch compiles + Sprint 23 baseline path stays bit-identical (default off)
- Day-10 stub trips with "thick-restart not yet implemented; falls through" or analogous skip message (if not pivoted)
- `make format && make lint && make test && make wall-check` clean

---

## Day 11: Thick-Restart FM — Implementation (Conditional)

**Theme:** Implement the Day-10 chosen random-vertex-flip thick-restart end-to-end (or, if pivoted, implement the multi-strategy-ensemble parking-lot work from PROJECT_PLAN.md).  Day 11's gate is "library compiles + Sprint 26 default-path corpus tests bit-identical (env var off) + thick-restart produces measurable behaviour change when on".

**Time estimate:** 12 hours

### Tasks
1. If Day 10 pivoted to multi-strategy ensemble: implement the ensemble — run baseline + FIFO + annealing in parallel (or sequentially with a global-best filter); pick best cut per partition call.  Skip tasks 2-7; document the implementation in `multi_strategy_ensemble_design.md` + `multi_strategy_ensemble_decision.md`.  Day 12's tasks shift to ensemble sweep + decision.
2. Implement thick-restart per Day-10's `thick_restart_design.md`.  Re-use Sprint 23 Day 11's multi-pass FM scaffolding; track the global-best cut across all passes; before each pass except the first, restart from the saved anchor with k = 1 % × n vertices' partition assignments randomly flipped.
3. Implement the boundary-vertex-shuffle + gaussian-noise perturbation branches behind `SPARSE_FM_THICK_RESTART_PERTURB={random_flip (default), boundary_shuffle, gauss_noise}` (Day 10's stubs).
4. Run the existing 39 partition tests under `SPARSE_FM_FINEST_STRATEGY=baseline` (default) — should all pass bit-identically to current master.
5. Run the same tests under `SPARSE_FM_FINEST_STRATEGY=thick_restart` — most will pass (thick-restart produces a valid cut, just one with the global-best property); some determinism contracts may need re-validation under the random perturbation.  Triage: add a `SPARSE_FM_THICK_RESTART_SEED` env var so perturbations are reproducible.
6. Capture an interim Pres_Poisson ND nnz_L measurement under `SPARSE_FM_FINEST_STRATEGY=thick_restart` (and the 3 perturbation sub-axes) to gauge the partial-close potential.  Save to `docs/planning/EPIC_2/SPRINT_27/thick_restart_interim_day11.txt`.
7. Light up the Day-10 stub `test_finest_fm_thick_restart_returns_to_anchor`: real assertions for the global-best non-decreasing property.
8. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Thick-restart in `src/sparse_graph.c::graph_refine_fm_thick_restart` behind `SPARSE_FM_FINEST_STRATEGY=thick_restart` env-var gate
- Random-flip + boundary-shuffle + gauss-noise perturbations behind `SPARSE_FM_THICK_RESTART_PERTURB` (default random_flip)
- Existing 39 partition tests pass bit-identically under default-off; thick-restart produces a valid (different) cut when on
- Interim Pres_Poisson ND nnz_L measurement
- All quality checks clean

### Completion Criteria
- Thick-restart's global-best cut is non-decreasing across passes (verified by `test_finest_fm_thick_restart_returns_to_anchor`)
- Sprint 23 baseline path bit-identical when env var off
- `make format && make lint && make test && make wall-check` clean

---

## Day 12: Thick-Restart Closing + Cross-Corpus Re-Bench Prep + Item 8 Test Scaffolding

**Theme:** Close out item 6 (thick-restart sweep + decision; 4h); prep the Day-13 cross-corpus re-bench (compile the bench-driver wrapper; pin the ≤24-combination cap per Sprint 26 Day 13's pattern; 4h); scaffold the item-8 tests (HCC Kuu-safe corpus parity; fixed-K differs-from-dynamic-K assertion sharpening; annealing-FM accepts-worsening-moves test reliability check; root-level spectral Fiedler-cut validation; 4h).

**Time estimate:** 12 hours (4h item 6 close + 4h item 7 prep + 4h item 8 test scaffolding)

### Tasks
1. **Item 6 closing (4h).**  Run the full Sprint 26 corpus under `SPARSE_FM_FINEST_STRATEGY=thick_restart` × `SPARSE_FM_THICK_RESTART_PERTURB={random_flip, boundary_shuffle, gauss_noise}` (3 perturbations × 6 fixtures = 18 measurements).  Save raw output to `docs/planning/EPIC_2/SPRINT_27/thick_restart_sweep.txt`.  Document the verdict in `docs/planning/EPIC_2/SPRINT_27/thick_restart_decision.md`: per-fixture sweep table + best-perturbation selection + headline status (does thick-restart help close Pres_Poisson where annealing/spectral didn't?).  Thick-restart ships advisory unless it both lands the headline AND has cleaner flip-rule application than items 4-5.
2. **Item 7 prep (4h).**  Compile the cross-corpus re-bench driver: identify the ≤24 representative combinations matching Sprint 26 Day 13's pattern.  Plausible combinations include: (a) Sprint 26 default; (b) Sprint 27 default-after-flips (if any landed); (c) HCC + Kuu-safe-fix; (d) `nd_base_threshold` = best-Day-3 value; (e) annealing-best-schedule; (f) root-level-spectral; (g) the headline 4-axis combination (annealing + root-spectral + per-vertex-fixed-K + HCC); (h-x) per-axis individual variations.  Cap at 24; save the combination list + bench-driver invocation script to `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations_skeleton.txt`.
3. **Item 8 test scaffolding (4h).**  Add the four new tests as failing-as-expected: `test_hcc_kuu_safe_corpus_parity` (tests HCC under Kuu-safe-fix matches Sprint 26 default-strategy on the corpus); `test_per_vertex_fixed_k_three_schemes_differentiate` (tightens Day 4's smoke into a corpus assertion); `test_finest_fm_annealing_pres_poisson_close_to_target` (Pres_Poisson nnz_L under annealing's best schedule lands within 2pp of the target); `test_nd_root_spectral_pres_poisson_close_to_target` (same for root-spectral).  These tests pin the Sprint 27 expected outcomes; if Day 9's verdict closed the headline, they pass under the Sprint 27 default; if it missed, they trip and document the partial-close.
4. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_sweep.txt` 3-perturbation × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_decision.md` flip outcome + best-perturbation selection + headline-status update
- `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations_skeleton.txt` ≤24-combination cross-corpus re-bench plan
- 4 new item-8 tests scaffolded (failing-as-expected; Day 13's measurements light them up or document the partial-close)
- All quality checks clean

### Completion Criteria
- `thick_restart_decision.md` documents the per-perturbation outcome + advisory-or-default verdict
- Cross-corpus re-bench combination list pinned at ≤24 entries
- 4 new item-8 tests compile and trip with "Sprint 27 verdict pending Day 13" or analogous skip message
- `make format && make lint && make test && make wall-check` clean

---

## Day 13: Cross-Corpus Re-Bench + Production-Default Decisions + Test-Bound Tightening

**Theme:** Run the Day-12 ≤24-combination cross-corpus matrix end-to-end; pick the corpus-wide-best combination + Pres_Poisson-headline-best combination; flip defaults if a clear winner emerges (items 1-3 individual flip outcomes feed in here as inputs).  Tighten `test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 24's 0.96× to whatever Sprint 27 achieved + 2pp noise margin.

**Time estimate:** 12 hours

### Tasks
1. Run the cross-corpus matrix: ≤24 combinations × 6 fixtures × 2 metrics (nnz_L + wall) = ≤288 measurements.  Use `bench_reorder.c` + `bench_amd_qg.c` per Sprint 26 Day 13's pattern.  Save raw output to `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations.csv` + `.txt`.
2. Build the headline summary: pick the top 5 combinations by Pres_Poisson nnz_L; pick the top 5 by corpus-wide nnz_L; identify the intersection (Sprint 26 Day 13's setting 13 was the canonical "best both" — Sprint 27's analog).  Document in `docs/planning/EPIC_2/SPRINT_27/headline_summary.md`: per-combination per-fixture outcomes + Sprint 27 verdict on the literal 0.85× target (closed / partial / missing).
3. Production-default decisions per axis:
   - `SPARSE_ND_COARSENING`: flip default to `hcc` if Day 2's Kuu-safe fix + the cross-corpus matrix confirm HCC default is flip-rule clean.
   - `sparse_reorder_nd_base_threshold`: flip from 96 to the Day 3 winner if any.
   - `SPARSE_FM_FINEST_STRATEGY`: flip from `baseline` to `annealing` (or `thick_restart`) if Day 7 / 12's flip rules cleared.
   - `SPARSE_ND_ROOT_BISECT`: flip from `multilevel` to `spectral` if Day 9's flip rule cleared.
   - `SPARSE_ND_SEP_LIFT_STRATEGY`: stays advisory unless Day 4's fixed-K mode wins corpus-wide on the differentiated weight schemes.
4. Tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from 0.96× to whatever Sprint 27's achieved default ratio is + 2pp noise margin (e.g. if default lands 0.83×, tighten to 0.85×).  Update `tests/test_reorder_nd.c::test_nd_10x10_grid_matches_or_beats_amd_fill` if Sprint 27's flips moved that metric.
5. Light up the four item-8 tests scaffolded Day 12: under the Sprint 27 default-flipped configuration, the four tests should pass; if any trip, root-cause and fix or document as Sprint 28+ routing.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations.csv` + `.txt` ≤24-combination × 6-fixture sweep
- `docs/planning/EPIC_2/SPRINT_27/headline_summary.md` Sprint 27 verdict on the 0.85× target + per-axis flip outcomes
- Production defaults flipped per Day-13 verdict (HCC / nd_base_threshold / FINEST FM / root-bisect)
- `test_nd_pres_poisson_fill_with_leaf_amd` tightened from 0.96× to Sprint 27's-achieved + 2pp
- 4 item-8 tests pass under Sprint 27 default config
- All quality checks clean

### Completion Criteria
- Cross-corpus matrix covers ≤24 combinations × 6 fixtures with raw csv + txt captures
- `headline_summary.md` records the literal 0.85× target outcome (closed via what combination, or partial / missing → Sprint 28+ routing)
- Test bound tightened with bench evidence
- `make format && make lint && make test && make wall-check` clean

---

## Day 14: Tests + Docs + Retrospective

**Theme:** Single-pass closing day per Sprints 25/26 retrospective lesson ("single Day-14 retro that absorbs the Day-13 work matches the actual time spent").  New tests validation + `docs/algorithm.md` ND subsection update + `SPRINT_22/PERF_NOTES.md` "Sprint 27 closures" subsection + `SPRINT_27/RETROSPECTIVE.md` filled in.

**Time estimate:** 12 hours

### Tasks
1. Validate the Day 12-13 tests are passing under the Sprint 27 default-flipped configuration (re-run `make test && make sanitize && make wall-check`).  If any tests trip — particularly the four item-8 tests pinning Pres_Poisson outcomes — root-cause and either fix in-place (small fix) or document as Sprint 28+ routing in `RETROSPECTIVE.md`.
2. Update `docs/algorithm.md` ND subsection to describe the Sprint 27 env vars (`SPARSE_ND_ROOT_BISECT`, `SPARSE_FM_FINEST_STRATEGY={annealing,thick_restart}`, `SPARSE_FM_ANNEALING_SCHEDULE`, `SPARSE_FM_THICK_RESTART_PERTURB`, `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k`, `SPARSE_ND_SEP_LIFT_K`, possibly `SPARSE_ND_SEP_LIFT_WEIGHT`) + their per-fixture deltas; supersede Sprint 26's "0.85× literal target route to Sprint 27" caveat with the actual achievement (or partial close + Sprint 28+ routing).
3. Append a "Sprint 27 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md`.  Per-axis closure summary + cross-corpus best-combination citation.
4. Fill in `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` single-pass.  Sections: "What shipped" (per-day deliverables); "Items deferred" (anything from PROJECT_PLAN.md item list that didn't close + reason); "Sprint 28 inputs" (concrete handoff items: e.g. multi-strategy ensemble parking-lot if items 4-5 closed early; pipeline-level interventions if items 4-5 missed; HCC default-flip status; etc.); "Process lessons" (anything Sprint 27-specific worth recording).
5. Update `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 27 section: status flip from "in flight" to "Complete"; record actual hours vs estimated 152; cite the 0.85× verdict; route any partial-close items to the Sprint 28 section.
6. Final `make format && make lint && make test && make sanitize && make tsan && make wall-check` — must all be clean before PR.
7. Open PR; request review; address any reviewer feedback (estimated 2-3 hours of buffer in the 12-hour budget).

### Deliverables
- `docs/algorithm.md` ND subsection updated with Sprint 27 env vars + per-fixture deltas
- `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` "Sprint 27 closures" subsection appended
- `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` filled in single-pass with what-shipped + items-deferred + Sprint-28-inputs + process-lessons
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 27 status flip to "Complete" + actuals vs estimate
- Sprint 27 PR opened + (ideally) merged
- All quality checks clean (format, lint, test, sanitize, tsan, wall-check)

### Completion Criteria
- All Sprint 27 tests pass under the default-flipped configuration
- `docs/algorithm.md` describes the new env vars accurately + cites the per-fixture deltas with bench evidence
- `RETROSPECTIVE.md` records the literal 0.85× target outcome (closed / partial / missing) with concrete Sprint 28+ routing if needed
- `make format && make lint && make test && make sanitize && make tsan && make wall-check` all clean
- Sprint 27 PR is mergeable (CI green, reviewer feedback addressed)

---

## Total Time Budget

| Day | Theme | Hours |
|-----|-------|-------|
| 1 | Item 1A — HCC Kuu profiling + matching trace + fix design | 8 |
| 2 | Item 1B — HCC Kuu-safe fix impl + flip re-attempt | 8 |
| 3 | Item 2 — `nd_base_threshold` relaxed-flip-rule re-sweep | 8 |
| 4 | Item 3 — Fixed-K per-vertex selection + 3-scheme sweep | 12 |
| 5 | Item 4 design — Annealing FM design + skeleton | 12 |
| 6 | Item 4 impl — Annealing FM implementation + interim Pres_Poisson | 12 |
| 7 | Item 4 close (4h) + Item 5 design kickoff (8h) — Root-spectral | 12 |
| 8 | Item 5 — Root-level spectral design close + impl | 12 |
| 9 | Item 5 — Root-level spectral Pres_Poisson sweep + decision | 12 |
| 10 | Item 6 — Thick-restart FM design (conditional) | 8 |
| 11 | Item 6 — Thick-restart FM implementation (conditional) | 12 |
| 12 | Item 6 close (4h) + Item 7 prep (4h) + Item 8 test scaffold (4h) | 12 |
| 13 | Item 7 — Cross-corpus re-bench + production-default decisions | 12 |
| 14 | Item 8 — Tests + docs + retrospective | 12 |

**Total: 152 hours** — exactly the PROJECT_PLAN.md estimate; ~16-hour slack against the 14×12 = 168-hour ceiling for variance / multi-strategy-ensemble parking-lot work / extra item 7-8 buffer if items 4-5 close the headline early.
