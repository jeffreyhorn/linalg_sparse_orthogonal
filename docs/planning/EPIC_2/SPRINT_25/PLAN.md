# Sprint 25 Plan: ND Fill-Quality Follow-Up (Sprint 24 deferrals)

**Sprint Duration:** 14 days
**Goal:** Close the Pres_Poisson ND/AMD ≤ 0.85× literal target Sprints 22-24 collectively missed (Sprint 22 1.063× → Sprint 23 0.952× → Sprint 24 0.942× best opt-in) via algorithmic work outside Sprints 22-24's scope: Heavy Connectivity Coarsening (Karypis-Kumar 1998 §5), multi-pass FM at intermediate uncoarsening levels, and spectral bisection at the coarsest level.  Also profile + tighten Sprint 24's ND default-path wall-time drift (1.06-1.10× of Sprint 23 baseline) and add a Pres_Poisson ND wall line to `wall_check_baseline.txt`.  All items routed from `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` "Items deferred" + "Sprint 25 inputs" #1-3.

**Starting Point:** Sprint 24 (PR #32, merged at `7293e30`) shipped: a `make wall-check` regression-check gate + `wall_check_baseline.txt` (`scripts/wall_check.sh`); the (c) revert of Sprint 23 Days 2-5 restoring Sprint 22's variable-only quotient-graph AMD baseline (`src/sparse_reorder_amd_qg.c`); two new ND env-var-gated alternatives (`SPARSE_ND_COARSEN_FLOOR_RATIO` Day 5 + `SPARSE_ND_SEP_LIFT_STRATEGY` Day 6) — both default off / Sprint 22 behavior preserved bit-identically; tightened test bounds (`test_nd_pres_poisson_fill_with_leaf_amd` ≤ 0.96× Sprint 24 Day 7; `test_nd_10x10_grid_matches_or_beats_amd_fill` ≤ 1.17× Day 8).  Sprint 24's `bench_summary_day9.md` documents the headline outcomes: gate (a) qg-AMD wall on bcsstk14 = 125.8 ms (PASS, well under the 1.5× Sprint 22 ceiling of 210 ms); gate (b) qg-AMD nnz(L) bit-identical across 9/9 fixtures (PASS); gate (c) Pres_Poisson ND/AMD = 0.952× default / 0.942× best opt-in (MISS — 0.85× literal target routed to Sprint 25); gate (d) all Sprint 23 nnz_L rows bit-identical-or-better (PASS).  Sprint 25 inherits gate (c) per `SPRINT_24/RETROSPECTIVE.md` "Sprint 25 inputs" #1.

**End State:** `sparse_graph_hierarchy_build` runs Heavy Connectivity Coarsening behind `SPARSE_ND_COARSENING={hcc,heavy_edge}` (default flipped to `hcc` if it produces a clear corpus-wide win on Pres_Poisson + small fixtures, otherwise stays `heavy_edge`).  `graph_refine_fm` runs multi-pass FM at the second-finest and third-finest uncoarsening levels behind `SPARSE_FM_INTERMEDIATE_PASSES` (default 1 = Sprint 23 behavior; flipped to 2 or 3 if measurable cut tightening on Pres_Poisson without smaller-fixture regression).  `graph_coarsest_bisection` runs Fiedler-vector-based spectral bisection behind `SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}` (reusing the Sprint 20-21 Lanczos eigensolver via `sparse_eigs_sym` shift-invert at σ ≈ 0+ε, with median-partition + 60/40-balance fallback to GGGP; default `gggp` flipped to `spectral` if Pres_Poisson wins materialize).  Pres_Poisson ND/AMD reaches ≤ 0.85× of AMD (or partial close to ≤ 0.90× documented + Sprint 26 routed if a fourth algorithmic axis is needed).  ND default-path wall on Pres_Poisson is profiled; the 21 % drift Sprint 24 Day 8 measured is either tightened to ≤ Sprint 23 + 5 % or documented as run-to-run variance with `wall_check_baseline.txt` threshold matched.  A `pres_poisson_nd` baseline line ships in `wall_check_baseline.txt` with a 50 % threshold.  `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` tightens from 0.96× to whatever is achieved + 2pp.  Cross-corpus re-bench captures land in `docs/planning/EPIC_2/SPRINT_25/bench_*.{csv,txt}` plus `coarsening_decision.md` + `intermediate_fm_decision.md` + `spectral_bisection_decision.md` + `headline_summary.md`.  `docs/algorithm.md` ND subsection updates to describe the three new env vars + their per-fixture deltas; `SPRINT_22/PERF_NOTES.md` (or `SPRINT_25/PERF_NOTES.md` if the Sprint 22 file gets unwieldy) gets a "Sprint 25 closures" subsection.  `SPRINT_25/RETROSPECTIVE.md` ships filled in (single Day-14 retro per Sprint 24 retrospective lesson "skip the stub-vs-body distinction").

**Time budget:** Each day caps at 12 hours.  The day budgets below sum to ~132 hours — about 4 hours above the 128-hour PROJECT_PLAN.md estimate, providing a similar safety buffer to Sprint 24 (which estimated 130 hrs and shipped at ~80 actual; the (c) revert collapsed a 32-hr fix into a 1-day revert and made items 3 + 4 N/A).  Risk concentration is item 3 (32 hrs across Days 6-8): spectral bisection is novel infrastructure that depends on the Sprint 20-21 Lanczos eigensolver behaving well on small Laplacian matrices, and the 60/40 balance-tolerance fallback to GGGP is the safety net.  Item 1 HCC (28 hrs across Days 1-3) is the highest-confidence delivery — it's a known algorithmic substitution.  Items 5-6 (16 hrs across Days 11-12) are independent of items 1-4 and can be reshuffled if the algorithmic work over-runs.

---

## Day 1: Sprint Kickoff — HCC Reading & Design

**Theme:** Read Karypis-Kumar 1998 §5 (Heavy Connectivity Coarsening) end-to-end; document the algorithm contract (edge-weight × neighbour-count scoring; tie-break order; matched-vertex collapse rule); design the `SPARSE_ND_COARSENING={hcc,heavy_edge}` env-var gate so the Sprint 22 default path stays bit-identical when off.  Day 1's gate is "design doc + skeleton — implementation lands Day 2".

**Time estimate:** 8 hours

### Tasks
1. Re-read `docs/planning/EPIC_2/SPRINT_24/RETROSPECTIVE.md` "Sprint 25 inputs" + `nd_coarsen_floor_decision.md` + `nd_sep_strategy_decision.md`.  Pin the Sprint 24 baseline (Pres_Poisson ND/AMD = 0.952× default, 0.942× best opt-in) as the regression band item 1's HCC has to push past.
2. Read Karypis-Kumar 1998 "Multilevel k-way Partitioning Scheme for Irregular Graphs" §5 ("Heavy Connectivity Coarsening").  Capture the algorithm contract in `docs/planning/EPIC_2/SPRINT_25/hcc_design.md`: scoring function `score(u, v) = edge_weight(u, v) × min(degree(u), degree(v))` (or whichever variant the paper recommends — pin the exact formula + page reference); tie-break order; visit-order over vertices (sorted by degree ascending? random? order Sprint 22's heavy-edge matching uses); matched-vertex collapse rule (sums adjacency, sums weight, marks both as matched).
3. Sketch the `SPARSE_ND_COARSENING` env-var gate in `src/sparse_graph.c::sparse_graph_hierarchy_build` (or wherever the matching loop lives — confirm the entry point Day 1).  Default `heavy_edge` (Sprint 22 behavior); on-value `hcc`; out-of-range / non-numeric input falls back to `heavy_edge` matching Sprint 24 Day 5's `SPARSE_ND_COARSEN_FLOOR_RATIO` validation pattern (strtol + range-check + fallback).
4. Stub `tests/test_graph.c::test_hcc_match_selection_grid` and `test_hcc_match_selection_irregular` as failing tests that pin the expected match selection on a 5×5 grid + a small irregular fixture under `SPARSE_ND_COARSENING=hcc` (these tests fail until Day 3's implementation lands).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `docs/planning/EPIC_2/SPRINT_25/hcc_design.md` algorithm contract + `SPARSE_ND_COARSENING` env-var gate design
- Stubbed `tests/test_graph.c` HCC tests (failing — pin the expected output Day 3 will produce)
- Confirmed code-entry-point identification for Day 2's HCC matching loop landing
- All quality checks clean

### Completion Criteria
- `hcc_design.md` names the exact KK1998 scoring formula + tie-break with page references; cross-references the Sprint 22 heavy-edge matching for the modified-vs-replaced delta
- The two failing-as-expected HCC tests compile and run (their assertions trip with "matching not yet implemented" or analogous skip message)
- `make format && make lint && make test && make wall-check` clean (HCC tests pass conditional on `SPARSE_ND_COARSENING` not yet being recognised — Day 1 ships only design)

---

## Day 2: HCC Implementation — Core Matching Loop

**Theme:** Implement the HCC matching loop in `src/sparse_graph.c` behind the `SPARSE_ND_COARSENING=hcc` env-var gate.  Day 2's gate is "library compiles + Sprint 22 default-path corpus tests bit-identical (env var off) + HCC produces *some* matching when on (correctness validation deferred to Day 3)".

**Time estimate:** 10 hours

### Tasks
1. Implement the HCC matching loop following Day 1's `hcc_design.md`.  Re-use Sprint 22's coarsen-loop scaffolding (visit-order, matched-vertex tracking, coarse-graph emit) — only the per-vertex match-selection function changes from `argmax(edge_weight)` to `argmax(score(u, v))` per the Day-1 formula.
2. Add the `SPARSE_ND_COARSENING` env-var read at the top of `sparse_graph_hierarchy_build` (matching the Sprint 24 Day 5 `SPARSE_ND_COARSEN_FLOOR_RATIO` env-var read pattern).  Out-of-range / non-numeric / missing → default `heavy_edge`.  Validate the gate via a small `coarsening_strategy_t` enum + dispatch in the matching loop body.
3. Run the existing `tests/test_graph.c` 39 partition tests under `SPARSE_ND_COARSENING=heavy_edge` (default) — should all pass bit-identically to current master (the gate's off-by-default branch is bit-identical to Sprint 22 behaviour).
4. Run the same tests under `SPARSE_ND_COARSENING=hcc` — most will pass (HCC produces a valid matching, just a different one); some determinism contracts may fail because HCC's tie-break differs from heavy-edge's.  Triage: if `test_partition_determinism_*` fails under HCC due to non-deterministic tie-break, root-cause and fix on Day 3.
5. Capture an interim Pres_Poisson ND nnz_L delta under `SPARSE_ND_COARSENING=hcc` to gauge progress against the 0.942× → ≤ 0.85× target (Day 1's plan-target for HCC's contribution alone).
6. Run `make format && make lint && make test && make wall-check` (the wall-check stays at Sprint 24's baseline ceilings throughout Sprint 25).

### Deliverables
- HCC matching loop in `src/sparse_graph.c` behind `SPARSE_ND_COARSENING` env-var gate
- Existing 39 partition tests pass bit-identically under `SPARSE_ND_COARSENING=heavy_edge` (default off)
- HCC produces a valid (possibly different) matching under `SPARSE_ND_COARSENING=hcc`
- Interim Pres_Poisson ND nnz_L measurement under HCC (saved to commit message or `docs/planning/EPIC_2/SPRINT_25/hcc_interim_day2.txt`)
- All quality checks clean

### Completion Criteria
- Library compiles; `test_graph` 39/39 pass under `SPARSE_ND_COARSENING=heavy_edge` (default-off branch is bit-identical to Sprint 22)
- `SPARSE_ND_COARSENING=hcc` produces some matching (not necessarily corpus-wide-correct yet — Day 3 validates)
- Pres_Poisson ND nnz_L under HCC trends toward the 0.85× target (≤ 0.93× = on-track; > 0.94× = HCC alone insufficient, escalate on Day 3)
- `make format && make lint && make test && make wall-check` clean

---

## Day 3: HCC — Validation, Determinism, Tests

**Theme:** Complete HCC's correctness work — bit-identical fill across the full corpus + synthetic banded under `SPARSE_ND_COARSENING=hcc`, no determinism drift (`test_partition_determinism_*` passes), unit tests pin the match selection on synthetic 2D-grid + irregular fixtures.  Day 2's loop was structural; Day 3 cleans up edge cases and lands the assertions.

**Time estimate:** 10 hours

### Tasks
1. Run `tests/test_graph.c` 39 partition tests under `SPARSE_ND_COARSENING=hcc`.  If any fail (especially `test_partition_determinism_*`), root-cause: HCC's tie-break must produce the same matching given the same seed + input.  If determinism breaks, the visit-order over candidate edges is the likely culprit (sort by stable key, e.g. (score desc, source-id asc, target-id asc)).
2. Land the two stubbed HCC tests from Day 1 (`test_hcc_match_selection_grid`, `test_hcc_match_selection_irregular`) — they should now pass with the HCC implementation from Day 2.  Pin the expected matching: on a 5×5 grid the diagonal edges should match preferentially because their endpoints have higher mutual degree than perimeter edges; on an irregular fixture the high-connectivity edges should match first.
3. Run `bench_reorder.c` full SuiteSparse corpus under `SPARSE_ND_COARSENING=hcc` and verify nnz(L) trends per-fixture vs Sprint 24 Day 9 baseline.  Pres_Poisson should drop from 0.952× toward ≤ 0.85×; smaller fixtures should not regress past the noise band (≤ 5 percentage points).  Capture to `docs/planning/EPIC_2/SPRINT_25/bench_day3_hcc_only.{csv,txt}`.
4. If Pres_Poisson under HCC alone reaches ≤ 0.90×, mark item 1 on-track for the headline gate; if 0.90-0.94×, item 1 is the smaller of three contributions and items 2 + 3 must close the rest; if > 0.94×, escalate (HCC's algorithmic-correctness verified but its win is small on this fixture; the headline gate depends entirely on items 2 + 3).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- 39 partition tests pass under `SPARSE_ND_COARSENING=hcc` (all determinism contracts intact)
- 2 new HCC unit tests pass (`test_hcc_match_selection_grid`, `test_hcc_match_selection_irregular`)
- `bench_day3_hcc_only.{csv,txt}` cross-corpus capture under HCC
- Item-1 headline-progress assessment (≤ 0.90× = on-track; 0.90-0.94× = need items 2+3; > 0.94× = item 1 is small contribution)
- All quality checks clean

### Completion Criteria
- `make test` covers the new HCC unit tests
- `test_partition_determinism_*` passes bit-identically under `SPARSE_ND_COARSENING=hcc`
- Pres_Poisson ND nnz_L under HCC ≤ 0.94× (necessary for items 2+3 to close the remaining gap on Day 9-10's combined re-bench)
- Smaller fixtures (nos4, bcsstk04, bcsstk14, Kuu) don't regress past 5pp under HCC vs Sprint 24 Day 9 baseline
- `make format && make lint && make test && make wall-check` clean

---

## Day 4: Multi-Pass FM Intermediate — Design & Implementation

**Theme:** Extend Sprint 23 Day 11's `SPARSE_FM_FINEST_PASSES` (3 passes at finest only) to the second-finest and third-finest uncoarsening levels, configurable via `SPARSE_FM_INTERMEDIATE_PASSES` (default 1 = Sprint 23 behavior).  Day 4's gate is "library compiles + Sprint 23 default-path corpus tests bit-identical (env var = 1) + multi-pass at intermediate levels runs without crashes when env var > 1".

**Time estimate:** 10 hours

### Tasks
1. Read Sprint 23 Day 11's commit (`SPARSE_FM_FINEST_PASSES` introduction in `src/sparse_graph.c::graph_refine_fm`) and `SPRINT_23/RETROSPECTIVE.md` "Performance highlights" lesson "multi-pass FM's payoff scales with the cost of a single pass".  Confirm the gain-bucket FM (Sprint 23 Days 9-10) makes per-pass cost O(|E|) so 1-2 extra passes at the 2nd-finest level adds < 5 % wall-time.
2. Add `SPARSE_FM_INTERMEDIATE_PASSES` env-var read with strtol + range-check + fallback (matching Sprint 24 Day 5's pattern).  Range [1, 10]; default 1.
3. Plumb the env var into `graph_refine_fm`'s outer uncoarsening loop: at level depth `level_idx` in the multilevel hierarchy, run `intermediate_passes` passes if `level_idx == finest - 1` or `level_idx == finest - 2` (the second-finest and third-finest), else 1 pass (Sprint 22 default for all but the finest).  The finest level continues to use `SPARSE_FM_FINEST_PASSES` (Sprint 23 Day 11).
4. Verify the skipped-vertex re-insertion contract (Sprint 23 Day 10's bcsstk04 LDL^T residual hazard fix) holds across the new pass placements.  Re-run `test_ldlt_via_nd_dispatch` under `SPARSE_FM_INTERMEDIATE_PASSES=2` — residual must stay at the Sprint 23 Day 10 baseline of 6.02 e-12.
5. Run `make format && make lint && make test && make wall-check` (under default `SPARSE_FM_INTERMEDIATE_PASSES=1` the suite must remain bit-identical to current master).

### Deliverables
- `SPARSE_FM_INTERMEDIATE_PASSES` env var implemented in `src/sparse_graph.c`
- Default-1 path bit-identical to Sprint 23 Day 11 behavior (corpus tests pass bit-identically)
- Multi-pass at intermediate levels runs without crashes for `SPARSE_FM_INTERMEDIATE_PASSES=2` and =3
- Skipped-vertex re-insertion contract intact (`test_ldlt_via_nd_dispatch` residual at 6.02 e-12)
- All quality checks clean

### Completion Criteria
- Library compiles; full test suite passes under `SPARSE_FM_INTERMEDIATE_PASSES=1` (bit-identical to current master)
- `SPARSE_FM_INTERMEDIATE_PASSES=2` and `=3` execute without crashing on the full SuiteSparse corpus
- bcsstk04 LDL^T residual = 6.02 e-12 under `SPARSE_FM_INTERMEDIATE_PASSES=2`
- `make format && make lint && make test && make wall-check` clean

---

## Day 5: Multi-Pass FM Intermediate — Validation & Sweep

**Theme:** Sweep `SPARSE_FM_INTERMEDIATE_PASSES ∈ {1, 2, 3}` across the corpus to identify whether 2-3 passes at the 2nd-finest level produces measurable cut tightening on Pres_Poisson without smaller-fixture regression.  Document the per-fixture deltas for Day 9's combined-effect re-bench.

**Time estimate:** 10 hours

### Tasks
1. Run `bench_reorder.c` full SuiteSparse corpus under `SPARSE_FM_INTERMEDIATE_PASSES=1` (default), `=2`, `=3` separately.  Capture nnz(L) and reorder_ms per fixture × setting.  Save to `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_sweep.txt`.
2. Compute the per-fixture ND/AMD ratio under each setting; compare against Sprint 24 Day 9 baseline.  Flag the setting that maximizes Pres_Poisson tightening without regressing smaller fixtures past 5pp.  Default candidate: `SPARSE_FM_INTERMEDIATE_PASSES=2` if it tightens Pres_Poisson by ≥ 1pp; `=3` if 2 isn't enough; stay at `=1` if neither shows a measurable win.
3. Add a Day-5 unit test `test_fm_intermediate_passes_smoke` in `tests/test_graph.c` that validates `SPARSE_FM_INTERMEDIATE_PASSES=2` produces a partition (any partition) on a 100-vertex synthetic fixture — the smoke test pins the env var's plumbing without locking in a particular cut quality.
4. Document the sweep results in `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md`: the chosen flip target (or "stays at 1"), the per-fixture nnz_L deltas, and the wall-time cost (multi-pass at intermediate levels adds work; verify it stays < 5 % wall).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `intermediate_fm_sweep.txt` with 3 settings × 6 fixtures = 18 measurements
- `intermediate_fm_decision.md` with the chosen flip target + per-fixture rationale + wall-time impact
- New `test_fm_intermediate_passes_smoke` unit test
- All quality checks clean

### Completion Criteria
- Sweep capture covers all 3 settings × 6 fixtures (no missing rows)
- Chosen flip target meets the rule: Pres_Poisson tightens by ≥ 1pp AND no smaller fixture regresses past 5pp; if neither setting meets the rule, decision doc records "stays at 1" and item 2's contribution to the headline gate is 0pp
- `test_fm_intermediate_passes_smoke` passes
- `make format && make lint && make test && make wall-check` clean

---

## Day 6: Spectral Bisection — Lanczos Integration Design & Laplacian Setup

**Theme:** Design how `graph_coarsest_bisection` will reuse the Sprint 20-21 Lanczos eigensolver (`sparse_eigs_sym` with shift-invert at σ ≈ 0+ε) to compute the Fiedler vector (second-smallest eigenvalue's eigenvector) of the graph Laplacian.  Day 6's gate is "design doc + Laplacian construction code ready for Day 7's Lanczos call".

**Time estimate:** 10 hours

### Tasks
1. Read `include/sparse_eigs.h` + Sprint 20 Day 7's `sparse_eigs_sym` API + Sprint 20-21 shift-invert path.  Document the call contract in `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md`: input (CSR Laplacian L = D - A), shift σ ≈ 0+ε to suppress the trivial λ_0 = 0 eigenpair, request k=2 eigenpairs (λ_0, v_0) + (λ_1, v_1), use v_1 as the Fiedler vector.  Pin the relative residual tolerance (1e-8 default; 1e-6 if Lanczos struggles on small Laplacians).
2. Implement `graph_build_laplacian` helper in `src/sparse_graph.c`: takes a `sparse_graph_t *` and emits an `idx_t n` × `idx_t n` symmetric `SparseMatrix` representing L = D - A where D is the degree matrix (diagonal of vertex degrees) and A is the adjacency matrix.  For unit-weighted graphs (all edges weight 1): `L[i][j] = -1` for j adjacent to i, `L[i][i] = degree(i)`.  Free the matrix when done.
3. Stub `graph_coarsest_bisection_spectral` as a function that takes the coarsest graph + `idx_t *part` output and returns a {0, 1} bisection.  Day 6 lands the function signature + Laplacian-construction call + Lanczos-call placeholder; Day 7 implements the Fiedler-vector → median-partition logic + GGGP fallback.
4. Add the `SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}` env-var read + dispatch in `graph_coarsest_bisection`'s entry point.  Default `gggp` (Sprint 22 behavior); `spectral` calls the Day 7 spectral path; `brute` keeps the brute-force enumeration for n ≤ 20.
5. Stub `tests/test_graph.c::test_spectral_bisection_eigenvalue_ordering` and `test_spectral_bisection_gggp_fallback` — fail-as-expected until Day 7 + Day 8 implementations land.
6. Run `make format && make lint && make test && make wall-check` (default-`gggp` branch bit-identical to current master).

### Deliverables
- `spectral_bisection_design.md` with the Lanczos call contract + Laplacian construction + Fiedler-vector usage
- `graph_build_laplacian` helper in `src/sparse_graph.c`
- `graph_coarsest_bisection_spectral` stub + `SPARSE_ND_COARSEST_BISECTION` env-var gate + dispatch
- 2 stubbed spectral tests (failing as expected)
- All quality checks clean

### Completion Criteria
- Library compiles; `test_graph` 39/39 pass under default `SPARSE_ND_COARSEST_BISECTION=gggp` (bit-identical to current master)
- `SPARSE_ND_COARSEST_BISECTION=spectral` returns a not-yet-implemented sentinel cleanly (doesn't crash) — Day 7 lands the implementation
- `graph_build_laplacian` produces a valid CSR Laplacian for a 5-vertex test fixture (manual verification: row sums to 0, off-diagonals ≤ 0)
- `make format && make lint && make test && make wall-check` clean

---

## Day 7: Spectral Bisection — Implementation

**Theme:** Implement the Fiedler-vector → median-partition logic in `graph_coarsest_bisection_spectral`, including the 60/40 balance-tolerance check that falls back to GGGP if the Fiedler cut is too imbalanced.  This is the highest-load day of the sprint (12 hours) because spectral bisection has the most novel infrastructure.

**Time estimate:** 12 hours

### Tasks
1. Implement the Lanczos call in `graph_coarsest_bisection_spectral`: build the Laplacian via Day 6's `graph_build_laplacian`, set up `sparse_eigs_opts_t` with `n_eigenpairs = 2`, `which = SPARSE_EIGS_SMALLEST` (or whichever Sprint 20-21 enum names smallest-by-magnitude with shift-invert), `shift_sigma = 1e-6` (Day 6's σ ≈ 0+ε), call `sparse_eigs_sym(L, &opts, &result)`, extract `result.eigenvectors[1]` as the Fiedler vector v_1.
2. Implement the median-partition logic: find median of v_1 entries, set `part[i] = 0` if `v_1[i] < median` else `part[i] = 1`.  For ties at the median value (common on regular meshes), break by vertex-id ordering to preserve determinism.
3. Implement the 60/40 balance-tolerance check: count vertices in each side post-partition; if `min(|side_0|, |side_1|) / max < 0.4` (i.e. the imbalance is worse than 60/40), the Fiedler cut is too skewed for ND's recursion contract — fall back to GGGP via `graph_coarsest_bisection_gggp` (the Sprint 22 entry point, kept available for the fallback path).
4. Land the two stubbed spectral tests from Day 6 (`test_spectral_bisection_eigenvalue_ordering`, `test_spectral_bisection_gggp_fallback`).  The eigenvalue-ordering test verifies λ_0 ≈ 0 and λ_1 > 0 on a connected graph; the GGGP-fallback test builds a fixture where the Fiedler cut is intentionally imbalanced (e.g. star graph: one center + n-1 leaves; spectral cut puts center on one side, all leaves on the other = 1/(n-1) imbalance) and verifies the fallback fires.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `graph_coarsest_bisection_spectral` fully implemented: Laplacian → Lanczos → Fiedler vector → median partition → balance check → GGGP fallback
- 2 spectral unit tests pass (`test_spectral_bisection_eigenvalue_ordering`, `test_spectral_bisection_gggp_fallback`)
- All quality checks clean

### Completion Criteria
- `make test` covers the 2 new spectral tests
- `test_spectral_bisection_eigenvalue_ordering`: λ_0 ≈ 0 (≤ 1e-6), λ_1 > 0, eigenvector v_0 is approximately constant (Fiedler-of-a-connected-graph contract)
- `test_spectral_bisection_gggp_fallback`: star-graph fixture's spectral cut triggers the 60/40 fallback to GGGP
- `make format && make lint && make test && make wall-check` clean

---

## Day 8: Spectral Bisection — Edge Cases, Fallback, Validation

**Theme:** Handle edge cases (n=1, n=2, disconnected graph, Lanczos non-convergence) and validate that the spectral path produces tighter cuts on Pres_Poisson than GGGP without regressing other fixtures.  Day 8's gate is "spectral bisection is corpus-safe + Pres_Poisson sees a measurable cut-tightening win".

**Time estimate:** 10 hours

### Tasks
1. Edge-case handling in `graph_coarsest_bisection_spectral`:
   - `n = 1`: trivial; `part[0] = 0`; skip Lanczos.
   - `n = 2`: trivial; `part[0] = 0, part[1] = 1`; skip Lanczos.
   - Disconnected graph (multiple connected components): the Laplacian has multiple zero eigenvalues; the Fiedler vector is degenerate.  Detect via `λ_1 ≈ 0` (within 1e-6 of λ_0); fall back to GGGP if so.
   - Lanczos non-convergence: if `sparse_eigs_sym` returns a non-OK status, fall back to GGGP.
2. Add unit tests for each edge case in `tests/test_graph.c`: `test_spectral_bisection_n1`, `test_spectral_bisection_n2`, `test_spectral_bisection_disconnected`, `test_spectral_bisection_lanczos_failure` (the last uses an artificially-crafted Laplacian or simulates a failure).
3. Run `bench_reorder.c` full SuiteSparse corpus under `SPARSE_ND_COARSEST_BISECTION=spectral`.  Verify Pres_Poisson tightens vs GGGP baseline (the headline check); other fixtures should be neutral or improve, never regress past 5pp.  Capture to `docs/planning/EPIC_2/SPRINT_25/bench_day8_spectral_only.{csv,txt}`.
4. Decision: if Pres_Poisson under spectral alone reaches ≤ 0.90×, item 3 alone is on-track for the headline gate; if 0.90-0.94×, items 1-2 must contribute the rest; if > 0.94×, escalate (spectral's cut-quality win on Pres_Poisson didn't materialize as expected; the headline gate depends on items 1 + 2 + 5 env-var combinations).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- 4 new edge-case spectral tests (n=1, n=2, disconnected, Lanczos failure → GGGP fallback)
- `bench_day8_spectral_only.{csv,txt}` cross-corpus capture under spectral
- Item-3 headline-progress assessment
- All quality checks clean

### Completion Criteria
- All 4 edge-case spectral tests pass
- Pres_Poisson ND nnz_L under spectral ≤ 0.94× (necessary for items 1+2+3 combined to close the 0.85× gap on Day 9)
- No fixture regresses past 5pp under spectral
- `make format && make lint && make test && make wall-check` clean

---

## Day 9: Cross-Corpus Re-Bench Under All Combinations

**Theme:** Run `bench_reorder.c` + `bench_amd_qg.c` across the full corpus under all combinations of the three new env vars (`SPARSE_ND_COARSENING`, `SPARSE_FM_INTERMEDIATE_PASSES`, `SPARSE_ND_COARSEST_BISECTION`) × Sprint 24's two existing env vars (`SPARSE_ND_COARSEN_FLOOR_RATIO`, `SPARSE_ND_SEP_LIFT_STRATEGY`).  The matrix is large; bound it to ~16 representative combinations rather than full Cartesian product.

**Time estimate:** 10 hours

### Tasks
1. Define the bench matrix: 6 fixtures × ~16 env-var combinations.  Combinations to include (16 total): Sprint 24 default; HCC-only; intermediate-FM-only; spectral-only; HCC + intermediate-FM; HCC + spectral; intermediate-FM + spectral; all three new; HCC + Sprint 24 ratio=200; spectral + Sprint 24 balanced_boundary; full set (HCC + intermediate-FM + spectral + Sprint 24 ratio=200 + Sprint 24 balanced_boundary); plus a few intermediate exploratory points.
2. Script the sweep in a small bash driver: for each combination, set env vars, run `bench_reorder --skip-factor`, parse `Pres_Poisson,*,ND,...`'s nnz_L + reorder_ms.  Capture to `docs/planning/EPIC_2/SPRINT_25/bench_day9_combinations.{csv,txt}`.
3. Identify the corpus-wide-best combination (lowest Pres_Poisson ND/AMD without regressing smaller fixtures past 5pp).  Compare against Sprint 24 Day 9 baseline (0.952× default; 0.942× best opt-in).
4. Headline assessment: does any combination reach ≤ 0.85× on Pres_Poisson?  ≤ 0.90× partial close?  Document the headline finding in `docs/planning/EPIC_2/SPRINT_25/headline_summary.md` (Day-10's decision day will pick the production default).
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `bench_day9_combinations.{csv,txt}` with ~96 measurements (6 fixtures × 16 combinations)
- `headline_summary.md` initial draft naming the corpus-wide-best combination + its Pres_Poisson ND/AMD ratio
- All quality checks clean

### Completion Criteria
- Sweep captures all 16 × 6 = 96 measurements (no missing rows)
- Headline finding documented: best Pres_Poisson ND/AMD ≤ 0.85× (PASS) / ≤ 0.90× (partial close) / > 0.90× (route remaining gap to Sprint 26)
- `make format && make lint && make test && make wall-check` clean

---

## Day 10: Production-Default Decisions & Test-Bound Tightening

**Theme:** Pick the corpus-wide-best combination from Day 9's sweep; flip the three new env-var defaults if a clear winner emerges; tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 24's `≤ 0.96× nnz_amd` to whatever the achieved ratio + 2pp allows.  Land the per-item decision docs.

**Time estimate:** 6 hours

### Tasks
1. From Day 9's `headline_summary.md`, pick the production default for each of the three new env vars: flip `SPARSE_ND_COARSENING` to `hcc` if the corpus-wide HCC win is clear (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression); flip `SPARSE_FM_INTERMEDIATE_PASSES` to 2 or 3 per Day 5's decision; flip `SPARSE_ND_COARSEST_BISECTION` to `spectral` if it's the dominant Pres_Poisson contributor.  Update the env-var defaults in `src/sparse_graph.c` if any flip.
2. Write the three per-item decision docs:
   - `docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md` — HCC vs heavy_edge sweep findings + production-default flip rationale (or stays-at-`heavy_edge` rationale)
   - `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` (if not already filled in Day 5) — multi-pass intermediate sweep findings + production-default flip
   - `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_decision.md` — spectral vs gggp sweep findings + production-default flip
3. Tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from `≤ 0.96× nnz_amd` to whatever Day 9's best-default-combination achieves + 2pp.  Inline comment cross-references the Day-10 production-default flip.
4. Tighten `tests/test_reorder_nd.c::test_nd_10x10_grid_matches_or_beats_amd_fill` if Day 9 measurements show movement on the 10×10 grid (likely if HCC + intermediate-FM combine constructively at small n).
5. Re-run `bench_reorder.c` + `bench_amd_qg.c` under the new production defaults; verify the new defaults produce the bench numbers Day 9's sweep predicted (sanity-check the flip).  Capture to `docs/planning/EPIC_2/SPRINT_25/bench_day10_post_flip.{csv,txt}`.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Production-default flips landed in `src/sparse_graph.c` (three env vars; flip per the rule "Pres_Poisson tightens ≥ 1pp + smaller fixtures don't regress past 5pp")
- Three per-item decision docs (`coarsening_decision.md`, `intermediate_fm_decision.md`, `spectral_bisection_decision.md`) finalised
- Tightened `test_nd_pres_poisson_fill_with_leaf_amd` bound (target ≤ 0.85× pinned with 2pp noise margin)
- `bench_day10_post_flip.{csv,txt}` confirms the flip's bench numbers match Day 9's prediction
- All quality checks clean

### Completion Criteria
- All decision docs name the chosen flip target (or "stays at default") with the per-fixture rationale
- `test_nd_pres_poisson_fill_with_leaf_amd` asserts the tightened bound (e.g. `nnz_nd * 100 ≤ nnz_amd * 87` for ≤ 0.85× + 2pp = ≤ 0.87×)
- Day 10's bench numbers match Day 9's sweep prediction within ±5 % wall-time noise (nnz_L bit-identical)
- `make format && make lint && make test && make wall-check` clean

---

## Day 11: ND Wall-Time Profile + Tightening

**Theme:** Sprint 24 Day 8 measured 42.86 s default-path Pres_Poisson ND vs Sprint 23's 36.4 s baseline (21 % drift); profile with `clock_gettime` instrumentation to identify whether the drift is real algorithmic cost or run-to-run measurement variance.  If real cost, tighten so default Pres_Poisson ND wall ≤ Sprint 23 + 5 %; if variance, document the 21 % run-to-run band as a measured property.

**Time estimate:** 12 hours

### Tasks
1. Add `SPARSE_ND_PROFILE` env var to `src/sparse_reorder_nd.c` (matching Sprint 24 Day 1's `SPARSE_QG_PROFILE` pattern): when set, emit `clock_gettime` aggregates per phase (coarsening, FM uncoarsening per level, separator extraction, leaf AMD splice, perm assembly).  Off by default; production overhead negligible.
2. Profile Pres_Poisson ND under `SPARSE_ND_PROFILE=1` 5 times (consecutive runs, same machine, no concurrent load) and capture the per-phase totals + variance.  Save to `docs/planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt`.
3. Decision: if any single phase shows > 5 % drift vs Sprint 23 baseline (computed as the per-phase Sprint-23-equivalent measurement, where available; if not, the aggregate drift is the only signal), root-cause and tighten.  If all phases are within 5 % drift but the aggregate is still 21 % above Sprint 23 baseline, the drift is run-to-run measurement variance; document and proceed to Day 12 with the relaxed wall-check threshold.
4. If a real algorithmic cost is identified (likely candidates per the profile: leaf-AMD splice cost; FM bucket-array growth on Sprint 23's gain-bucket structure), implement the fix.  Re-profile to confirm the tightening.
5. Document findings in `docs/planning/EPIC_2/SPRINT_25/nd_wall_time_decision.md`: variance vs algorithmic-cost finding, fix (if any), Sprint-23-baseline vs Sprint-25-post-fix measurements.
6. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `SPARSE_ND_PROFILE` env-var-gated `clock_gettime` instrumentation in `src/sparse_reorder_nd.c`
- `profile_day11_pres_poisson_nd.txt` with 5 consecutive runs + per-phase breakdown
- `nd_wall_time_decision.md` documenting variance-vs-algorithmic-cost finding + any fix landed
- All quality checks clean

### Completion Criteria
- Profile capture identifies whether Sprint 24's 21 % drift is variance (5 runs span > 15 % range) or algorithmic cost (a single phase shows > 5 % drift consistently)
- If variance: documented; Day 12 sets the wall-check threshold to 50 % above the Sprint 25 measured baseline
- If algorithmic cost: fix landed; Sprint 25's post-fix Pres_Poisson ND wall ≤ Sprint 23 baseline + 5 % (~38 s)
- `make format && make lint && make test && make wall-check` clean

---

## Day 12: `make wall-check` Pres_Poisson ND Baseline Line

**Theme:** Add a `pres_poisson_nd` baseline entry to `wall_check_baseline.txt` with the Day 11 measurement; update `scripts/wall_check.sh` to parse the new key and run `bench_reorder --only Pres_Poisson --skip-factor`'s ND row in addition to the existing AMD row.  Threshold matches Day 11's variance-vs-algorithmic-cost finding (50 % above baseline if variance; 25 % above if Day 11 tightened the algorithmic cost).

**Time estimate:** 6 hours

### Tasks
1. Add a `pres_poisson_nd_ms=` line to `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` (or open `docs/planning/EPIC_2/SPRINT_25/wall_check_baseline.txt` if the Sprint 24 file gets unwieldy with the cumulative comment block).  Value = median of Day 11's 5 consecutive Pres_Poisson ND measurements.
2. Update `scripts/wall_check.sh` to:
   - Parse a third baseline key `pres_poisson_nd_ms` (matching the existing `bcsstk14_qg_amd_ms` and `pres_poisson_amd_ms` parser pattern from Sprint 24 Day 1).
   - Run `bench_reorder --only Pres_Poisson --skip-factor` (already done for the AMD row) and additionally extract the ND row's `reorder_ms` via `awk -F, '$1=="Pres_Poisson" && $3=="ND" {print $5; exit}'`.
   - Compare the ND measurement against the new baseline using the variable threshold (50 % if variance per Day 11; 25 % if algorithmic-cost-tightened per Day 11).  Threshold lives as a config constant near the top of the script.
3. Update `docs/algorithm.md` "Performance regression gates" subsection to describe the new ND baseline + threshold rationale (cross-reference Day 11's variance / algorithmic-cost decision).
4. Verify the gate fires on a synthetic regression: temporarily bump the baseline value by 10× and run `make wall-check` — must FAIL.  Restore the real baseline.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- `pres_poisson_nd_ms=` entry in `wall_check_baseline.txt` with the Day 11 measurement
- `scripts/wall_check.sh` updated with the new baseline key + ND row parser + variable threshold
- `docs/algorithm.md` "Performance regression gates" subsection extended for the new baseline
- All quality checks clean (including the new ND wall-check gate exiting 0)

### Completion Criteria
- `make wall-check` runs all three checks (bcsstk14 qg-AMD, Pres_Poisson AMD, Pres_Poisson ND) and reports each individually
- Synthetic-regression test (10× baseline bump) confirms the new ND gate fires correctly
- `make format && make lint && make test && make wall-check` clean

---

## Day 13: Closing Tests + Documentation Sweep

**Theme:** Add the closing tests called out by items 1-3, refresh `docs/algorithm.md`'s ND subsection to describe the three new env vars + their per-fixture deltas, append a "Sprint 25 closures" subsection to `SPRINT_22/PERF_NOTES.md`, and capture the final cross-corpus bench.

**Time estimate:** 10 hours

### Tasks
1. Audit new tests already added in Days 1-8 (HCC parity tests, multi-pass-intermediate FM smoke test, spectral bisection eigenvalue ordering + edge-case tests).  Verify they're enabled in each test binary's `RUN_TEST` block (no `#if 0` guards).  Add any missing tests called out by items 1-3 that haven't landed yet.
2. Update `docs/algorithm.md`'s ND subsection:
   - Add `SPARSE_ND_COARSENING` (Day 1-3) under step 1 (Coarsen) with HCC's per-fixture deltas
   - Add `SPARSE_FM_INTERMEDIATE_PASSES` (Day 4-5) under step 3 (Uncoarsen with FM)
   - Add `SPARSE_ND_COARSEST_BISECTION` (Day 6-8) under step 2 (Coarsest bisection)
   - Update the per-fixture advisory list in "Characteristics" with the Day 9-10 cross-corpus deltas under the new defaults
   - Supersede Sprint 24's "Pres_Poisson 0.7× literal target route to Sprint 25" caveat with the actual achievement (0.85× or partial close)
3. Append a "Sprint 25 closures" subsection to `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (or open `docs/planning/EPIC_2/SPRINT_25/PERF_NOTES.md` if the Sprint 22 file is too long after Sprint 24's closures section).  Include: the Day-9 sweep summary table, what moved (Pres_Poisson ND/AMD ratio + which env vars contributed), what didn't move (fixtures where the new defaults were neutral), Sprint 26 routing for any deferred items.
4. Capture the final cross-corpus bench under the production defaults: `bench_reorder.c` + `bench_amd_qg.c` full SuiteSparse + synthetic banded.  Save to `docs/planning/EPIC_2/SPRINT_25/bench_day13_final.{csv,txt}` + `bench_day13_amd_qg.{csv,txt}`.
5. Run `make format && make lint && make test && make wall-check`.

### Deliverables
- Closing tests for items 1-3 enabled in their test binaries
- `docs/algorithm.md` ND subsection updated with three new env vars + per-fixture advisory deltas
- `SPRINT_22/PERF_NOTES.md` (or `SPRINT_25/PERF_NOTES.md`) "Sprint 25 closures" subsection
- `bench_day13_final.{csv,txt}` + `bench_day13_amd_qg.{csv,txt}` final cross-corpus capture
- All quality checks clean

### Completion Criteria
- New tests pass under `make test` (all enabled, no skip blocks)
- `algorithm.md` ND subsection describes all five env vars (Sprint 24's two + Sprint 25's three) with consistent advisory format
- `PERF_NOTES.md` closures subsection cites Day 9-10's numbers and names the production-default flips
- `bench_day13_final.txt` confirms the production defaults match Day 10's `bench_day10_post_flip.txt` measurements (bit-identical nnz_L; wall ms within ±5 %)
- `make format && make lint && make test && make wall-check` clean

---

## Day 14: Soak, Final Bench Capture, Retrospective & PR Open

**Theme:** Final cross-ordering capture as the end-of-sprint headline, full corpus regression run, single-pass retrospective body (per Sprint 24 retrospective lesson "skip the stub-vs-body distinction"), and PR open.  This is the day that gates the merge.

**Time estimate:** 8 hours

### Tasks
1. Final capture: re-run `bench_reorder.c` + `bench_amd_qg.c` once more.  Save to `bench_day14.{csv,txt}` + `bench_day14_amd_qg.{csv,txt}`.  Sanity-check that nothing regressed since Day 13 — should be bit-identical on nnz_L since Day 13 was tests + docs only.
2. Run `make sanitize` (UBSan) against the full test suite.  Sprint 24 closed Sprint 23's flagged sanitize gap; verify Sprint 25's algorithmic additions (HCC, intermediate-FM, spectral) don't reintroduce sanitizer issues.
3. Run `make tsan` (with Homebrew LLVM clang per `Makefile` `tsan` target's note) against the full test suite.  Sprint 25's changes are single-threaded (no OpenMP additions); verify Sprint 22's tsan baseline still applies + the new Lanczos call in spectral bisection doesn't leak races into the partition path.
4. Fill in the Sprint 25 retrospective single-pass at `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` (no Day-13 stub per the Sprint 24 retro lesson): Goal recap; DoD checklist; Final metrics (Pres_Poisson ND/AMD ratio achieved + per-fixture corpus table); Performance highlights; What went well; What surprised us; What didn't go well; Items deferred (any 0.85× shortfall routes to Sprint 26 with concrete avenues); Lessons; Sprint 26 inputs; Acknowledgements (Karypis-Kumar 1998 §5 for HCC; Sprint 20-21 Lanczos eigensolver for spectral bisection); Day-by-day capsule; DoD verification.
5. Open the Sprint 25 PR (`gh pr create`) targeting `master`.  PR description summarises the seven items + the day-by-day commits + the headline numbers from `bench_day14.txt` vs Sprint 24's `bench_day9.txt`.
6. Run `make format && make lint && make test && make sanitize && make wall-check`.

### Deliverables
- `bench_day14.{csv,txt}` + `bench_day14_amd_qg.{csv,txt}` final captures
- `make sanitize` + `make tsan` clean (or pre-existing infrastructure gaps explicitly flagged in retro)
- `RETROSPECTIVE.md` filled in (single Day-14 retro; all 13 sections from Sprint 24's pattern have content)
- Sprint 25 PR opened
- All quality checks clean

### Completion Criteria
- Final cross-ordering capture matches Day 13's output bit-identically on nnz_L (no regressions in Day 13's doc/test sweep)
- `make sanitize` + `make tsan` clean against the full test suite (or documented as Sprint-26 infrastructure follow-ups)
- Retrospective written; all 13 sections have content (no `(Day 14 prose pending)` placeholders)
- PR opened; description references the headline Pres_Poisson ND/AMD ratio achieved + the three new env-var defaults (whichever flipped)
- `make format && make lint && make test && make sanitize && make wall-check` clean

---

## Sprint 25 Summary

**Total estimated hours:** 8 + 10 + 10 + 10 + 10 + 10 + 12 + 10 + 10 + 6 + 12 + 6 + 10 + 8 = 132 hours

**Item-to-day mapping:**

| Item | Days | Hours |
|------|------|-------|
| 1: Heavy Connectivity Coarsening | Days 1-3 | 28 |
| 2: Multi-pass FM at intermediate uncoarsening levels | Days 4-5 | 20 |
| 3: Spectral bisection at coarsest level | Days 6-8 | 32 |
| 4: Cross-corpus re-bench + production-default decisions + test-bound tightening | Days 9-10 | 16 |
| 5: ND wall-time profile + tightening | Day 11 | 12 |
| 6: `make wall-check` Pres_Poisson ND baseline line | Day 12 | 4 (allotted 6, 2 hr buffer) |
| 7: Tests + docs + retrospective | Days 13-14 | 18 (allotted 16; +2 hr soak buffer) |

**Headline gates (must pass on Day 14):**

- Pres_Poisson ND/AMD ≤ 0.85× literal target met (Sprint 24 baseline 0.952× default / 0.942× best opt-in; Sprint 25 closes the remaining 7-9pp via items 1-3 combined)
- All Sprint 24 nnz_L rows bit-identical or improve under Sprint 25's production defaults
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` asserts the tightened bound (target ≤ 0.85× pinned with 2pp noise margin)
- `make wall-check` exits 0 against Day 12's expanded baseline (now includes Pres_Poisson ND)
- `make format && make lint && make test && make sanitize && make wall-check` clean on Day 14

**Risk flags:**

- Item 3 (32 hrs across Days 6-8) is the load-bearing novel infrastructure: spectral bisection reuses the Sprint 20-21 Lanczos eigensolver, which has been validated on numeric eigenvalue problems but not on graph Laplacians at this scale.  Risk mitigations: the 60/40 balance-tolerance fallback to GGGP is the safety net (Day 7 task 3); edge-case handling for disconnected graphs + Lanczos non-convergence is Day 8 task 1.  If spectral fundamentally doesn't work on the corpus, item 3's 32 hrs become a smaller-than-expected contribution and the headline gate depends on items 1 + 2 + Sprint 24 env vars.
- Item 1 HCC (28 hrs across Days 1-3) is the highest-confidence delivery — algorithmic substitution with a clear reference (Karypis-Kumar 1998 §5) and a known fallback (`SPARSE_ND_COARSENING=heavy_edge` keeps Sprint 22 behavior).  Risk: HCC's per-fixture wins may not aggregate enough to close the 0.85× gap alone; that's expected — items 2 + 3 are the multipliers.
- Item 2 multi-pass intermediate FM (20 hrs across Days 4-5) is the smallest-risk algorithmic item: the `SPARSE_FM_FINEST_PASSES` pattern (Sprint 23 Day 11) is established; this just generalises it.  Risk: 2-3 passes at intermediate levels may not produce measurable cut tightening — that's a no-op outcome (default stays 1) which doesn't move the headline gate.
- Items 5-6 (16 hrs across Days 11-12) are independent of items 1-4 and can be reshuffled if the algorithmic work over-runs.  Item 5's ND wall-time profile may surface a real algorithmic cost that requires more than 12 hrs to fix; if so, document the 21 % drift as variance and proceed to Day 12 with the relaxed threshold.
- The 132-hour estimate has a 4-hour cushion above the 128-hour PROJECT_PLAN.md figure — same shape as Sprint 24's 4-hr cushion above its 126-hour estimate.  If item 3's spectral bisection over-runs by > 4 hrs, the buffer absorbs it; if > 12 hrs, item 4's Day 9-10 sweep + decisions can be compressed into a single day (lose ~6 hrs of slack but still ship the headline).

**Davis 2006 §7.5.1 external-degree refinement (parking lot):** Per PROJECT_PLAN.md Sprint 25 section's parking-lot note — "N/A under (c) revert; resurrect if a future sprint reintroduces approximate-degree."  Sprint 25 doesn't reintroduce approximate-degree, so external-degree stays parked.  No active item in this sprint.

**Conventions inherited from Sprint 24:**

- Each day's commit message follows the Sprint 24 pattern: `Sprint 25 Day N: <short theme>` with a body documenting the changes + cited decision docs + quality-gate results.
- Decision docs land in `docs/planning/EPIC_2/SPRINT_25/` (matching Sprint 24's pattern); each is cited from at least one of: a commit message, an `algorithm.md` cross-reference, a test-file inline comment, or the Sprint 25 retrospective.
- Bench captures land at `docs/planning/EPIC_2/SPRINT_25/bench_*.{csv,txt}` (CSV from `bench_reorder.c`'s native output; human-readable `.txt` rendered via `column -t -s,` for visual diff against Sprint 24's captures).
- Retrospective is a single Day-14 pass (no Day-13 stub) per Sprint 24 retrospective lesson.
