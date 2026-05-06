# Sprint 24 Day 7 — ND tuning re-check + production-default decision

## Decision

1. **Keep `SPARSE_ND_SEP_LIFT_STRATEGY` default at `smaller_weight`** (Sprint 22 Day 4 behaviour, unchanged).  Day 6's literal flip-rule isn't met (Pres_Poisson balanced_boundary lands at 0.953× — neutral, not the ≤ 0.90× "clear win" the rule requires).  The corpus-aggregate evidence in favour of `balanced_boundary` (8-38 percentage-point nnz_L wins on nos4 / bcsstk14 / Kuu) is recorded as documented advisory for non-Pres_Poisson workloads; the env var stays available for opt-in.
2. **Tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd`** from `nnz_nd ≤ 1.00× nnz_amd` to `nnz_nd ≤ 0.96× nnz_amd`.  Default-path measurement is bit-stable at 0.952× (deterministic partitioner; same value across Sprint 23 Day 11 → Sprint 24 Day 7).  Bound has 0.8 percentage-point safety margin (~20 307 nnz cushion).
3. **No production-default change for `SPARSE_ND_COARSEN_FLOOR_RATIO`** either — Day 5 already locked in default 100 (unchanged) with ratio=200 as the env-gated Pres_Poisson advisory.

## Background

Sprint 24 Day 7 was originally PLAN.md Day 10 ("ND Fill-Quality Follow-Up — Tuning & Determinism Re-Check").  Per Day 1's (c)-revert + Day 5/Day 6's pull-forward of items 5(a)/(b), Day 7 inherits two follow-up tasks from PLAN.md Day 10:

- Re-run all 39 `tests/test_graph.c` partition tests under the chosen Day 5 + Day 6 defaults; verify determinism.
- Tighten the `test_nd_pres_poisson_fill_with_leaf_amd` fixture-pin to reflect the new ratio.

…plus the PLAN.md Day-9 task-5 carry-over (production-default for `SPARSE_ND_SEP_LIFT_STRATEGY`) that `nd_sep_strategy_decision.md` "Day 7 plan reference" deferred to today.

## Partition-test verification

`tests/test_graph.c` covers 39 tests / 1 460 assertions, including the two determinism contracts (`test_partition_determinism_10x10_grid`, `test_partition_determinism_two_cliques`) and the smaller-weight tie-break pin (`test_edge_to_vertex_separator_smaller_side`).  Re-ran the full suite under all four env-var combinations:

| Env-var setting                                                                          | Tests | Failed | Assertions |
|------------------------------------------------------------------------------------------|------:|-------:|-----------:|
| (default — neither var set)                                                              |    39 |      0 |      1 460 |
| `SPARSE_ND_COARSEN_FLOOR_RATIO=200`                                                      |    39 |      0 |      1 460 |
| `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`                                          |    39 |      0 |      1 460 |
| `SPARSE_ND_COARSEN_FLOOR_RATIO=200` + `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`    |    39 |      0 |      1 460 |

Determinism contracts pass bit-identically across all four runs.  `test_edge_to_vertex_separator_smaller_side` passes under `balanced_boundary` because the 5×6-grid fixture has tied boundary counts (5 vs 5), so `bb_side` ties to 0 (the same side `smaller_weight`'s tied-weight tie-break picks); both strategies converge to the same lift on this fixture.

`test_partition_pres_poisson_smoke` reports `sep=204` under all four settings — that's a single top-level partition, and the four env-var combos don't move the top-level cut on this fixture.  The Day 5 / Day 6 nnz_L deltas come from the *recursive* ND pipeline, not the single-level partition the smoke test exercises.

## Pres_Poisson fixture-pin tightening

Sprint 23 Day 11 set the bound at `nnz_nd ≤ 1.00× nnz_amd` (5pp margin above the 0.952× measured ratio).  Sprint 24 Day 7 tightens to `≤ 0.96× nnz_amd` (0.8pp margin):

| Quantity                  | Value                                                             |
|---------------------------|-------------------------------------------------------------------|
| AMD nnz(L) (constant)     | 2 668 793                                                         |
| ND nnz(L) (default-path)  | 2 541 734                                                         |
| Measured ratio            | 0.952×                                                            |
| Old bound (Sprint 23)     | `nnz_nd ≤ 2 668 793` (1.00× — 127 059 nnz of headroom = 5pp)      |
| **New bound (Sprint 24)** | `nnz_nd ≤ 2 562 041` (0.96× — 20 307 nnz of headroom = 0.8pp)     |

The partitioner is contractually deterministic (`test_partition_determinism_*`), so Pres_Poisson's nnz_nd is bit-stable across runs — the 0.8pp margin is for future commits that perturb the FM tie-breaks, not for run-to-run noise.  If a future commit changes Pres_Poisson's default-path nnz_nd by more than 20 307 nnz (0.76 % drift), the test fires and the change has to be evaluated explicitly.

The PLAN.md Day-10 task 3 alternative — tightening to `≤ 0.85×` — would put the test in failing-by-construction territory (current 0.952× ≫ 0.85×).  `nd_sep_strategy_decision.md` documents the 0.85× target as Sprint-25 territory; the test bound mirrors that scope.

## Production-default decision for `SPARSE_ND_SEP_LIFT_STRATEGY`

PLAN.md Day 9 task 5 (literal rule): "flip `SPARSE_ND_SEP_LIFT_STRATEGY` to balanced_boundary if it's a clear win on Pres_Poisson (≤ 0.90×) without regressing the smaller fixtures past 5 percentage points; else keep it off."

Day 6 measurement (from `sep_strategy_sweep.txt`):

| Fixture       | Default (smaller_weight)  | balanced_boundary alone | Δpp Pres_Poisson rule? |
|---------------|--------------------------:|------------------------:|------------------------|
| nos4          | 1.520×                    | 1.251×                  | n/a (smaller fixture; -27 pp)  |
| bcsstk14      | 1.129×                    | 1.048×                  | n/a (smaller fixture; -8 pp)   |
| Kuu           | 2.275×                    | 1.415×                  | n/a (smaller fixture; -86 pp)  |
| Pres_Poisson  | 0.952×                    | 0.953×                  | **+0.1 pp — *not* a clear win**|

Pres_Poisson is essentially neutral (+0.1pp drift, in the noise band).  Per the literal rule, **keep `balanced_boundary` off-by-default**.

### Counter-argument considered: corpus-aggregate flip

The Day 6 decision doc flagged a corpus-aggregate counter-argument: balanced_boundary helps every smaller fixture by 8-38 percentage points and is neutral on Pres_Poisson.  If the headline metric were a fixture-aggregated nnz_L sum, balanced_boundary wins clearly.

Why I'm rejecting the corpus-aggregate flip on Day 7:

1. **Sprint 24's stated headline is Pres_Poisson**, not a corpus-aggregate.  PROJECT_PLAN.md item 5 + PLAN.md Day 9 task 5 both name Pres_Poisson as the production-default flip criterion.  Re-deciding the criterion mid-sprint based on it being unmet would change the goalposts.
2. **Blast radius.**  The default-path semantics of `graph_edge_separator_to_vertex_separator` are observable to every ND caller.  Three of four fixtures see *different* permutations under balanced_boundary; production callers that have downstream code paths assuming the Sprint 22 Day 4 ordering would silently shift.  No internal caller is known to depend on this, but the change is non-local.
3. **Test naming.**  `test_edge_to_vertex_separator_smaller_side` is the behaviour pin for the default strategy.  The test's assertions happen to hold under `balanced_boundary` for the chosen fixture (boundary tie + weight tie both resolve to side 0), but the test's *name* + comment reflect the smaller_weight semantics.  Flipping the default would silently make the test misleading rather than failing — worse than a clean breakage.
4. **Sprint 23 lessons-learned echo.**  `fix_decision_day1.md` cites the Sprint 23 retrospective's "What surprised us" lesson: algorithmic features whose justification rests on circumstantial evidence shouldn't ship by default.  Balanced_boundary's smaller-fixture wins are real but circumstantial — the headline target (Pres_Poisson) doesn't validate them.  Same logic applies.
5. **Future re-evaluation cost is low.**  `SPARSE_ND_SEP_LIFT_STRATEGY` ships now; Sprint 25 (or any later sprint) can flip the default after expanding the fixture corpus or after benchmarking a real workload's mix.  The env var is the contract that lets us defer the decision without losing the implementation.

### What changes documentation-wise on Day 13

`docs/algorithm.md`'s ND subsection (Day 13's task) should:
- Describe both `SPARSE_ND_COARSEN_FLOOR_RATIO` (default 100; ratio=200 advisory for Pres_Poisson) and `SPARSE_ND_SEP_LIFT_STRATEGY` (default `smaller_weight`; `balanced_boundary` advisory for non-Pres_Poisson workloads where deeper coarsening + balanced-boundary lift improves cut quality).
- Note the corpus-aggregate evidence behind balanced_boundary (Kuu -38 % nnz_L) so callers in similar domains know to flip the env var.
- Cite Days 5-7's decision docs as the per-fixture sweep + flip-rule rationale.

## Wall-time spot-check

Day 7's only code change is a test-bound tightening — no kernel touched.  The default-path Pres_Poisson ND wall-time is unchanged from Day 6: re-running `build/test_reorder_nd` measured 38.58 s (within run-to-run noise of Day 6's 37.5 s default-path measurement).  bcsstk14 qg-AMD wall-time is also untouched (separator-extraction is ND-only).  `make wall-check` should pass cleanly against the Day 4 baselines (130 ms / 8 000 ms).

## Day 8 plan reference

Day 8 (originally PLAN.md Day 11 — "ND Fill-Quality Follow-Up — Close & Document"):

1. Tighten `tests/test_reorder_nd.c::test_nd_10x10_grid_matches_or_beats_amd_fill`'s bound from Sprint 23's `≤ 1.21× nnz_amd` to whatever Days 5-6's combined effect achieves on the 100-vertex fixture (the 10×10 grid's small size limits how much `SPARSE_ND_COARSEN_FLOOR_RATIO` and `SPARSE_ND_SEP_LIFT_STRATEGY` can move it; expect 1.10×-1.20× under default-path).
2. Update `docs/algorithm.md`'s ND subsection per the "What changes documentation-wise on Day 13" section above (originally scheduled for Day 11 in the re-shuffled plan).
3. Run the full `tests/test_reorder_nd.c` + `tests/test_graph.c` under the four env-var combinations to confirm no test regresses across the matrix.
4. Capture the Day-8 wall-time on Pres_Poisson ND under the new defaults.

## Quality-gate notes

- `make format-check`: pending re-run on the Day 7 commit (only one ASSERT_TRUE expression changed + comment edited; expect clean).
- `make lint`: pending re-run; the `(long long)nnz_nd * 100 <= (long long)nnz_amd * 96` pattern matches the existing `* 100 <= * 125` form on line 412 (10×10 grid bound) — already lint-clean.
- `make test`: ran `build/test_reorder_nd` post-edit; `test_nd_pres_poisson_fill_with_leaf_amd` passes under the new 0.96× bound (measured 0.952×, 0.8pp margin).  Full suite re-run pending.
- `make wall-check`: no kernel touched on Day 7; expect pass against the Day 4 baselines.
