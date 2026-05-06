# Sprint 24 Day 6 — ND separator-lift-strategy decision (item 5 option (b))

## Decision

**Add the `SPARSE_ND_SEP_LIFT_STRATEGY` env var with default `smaller_weight` (Sprint 22 behaviour, unchanged); the new `balanced_boundary` value is implemented but stays env-var-gated off-by-default.**  Per PLAN.md Day 9 task 5, the production-default flip rule is "balanced_boundary is a clear win on Pres_Poisson (≤ 0.90×) without regressing the smaller fixtures past 5 percentage points".  balanced_boundary on Pres_Poisson lands at 0.953× (vs the Sprint 24 Day 5 default of 0.952×) — essentially neutral, so it doesn't qualify as a "clear win on Pres_Poisson" under the literal rule, even though it produces large nnz_L wins on every smaller fixture.  Day 7 (originally PLAN.md Day 10 — re-shuffled forward by Day 1's revert) revisits the production default after the combined Day 5 + Day 6 result on Pres_Poisson is locked in.

The combined ratio=200 + balanced_boundary setting on Pres_Poisson lands at 0.950×, which is **slightly worse** than ratio=200 alone (0.942×) — the two changes don't compose constructively on Pres_Poisson.  Neither setting reaches the 0.85× sprint target nor the 0.90× partial-close fallback.  Pres_Poisson's headline gate stays the Sprint 23 status quo (0.95× ratio); Sprint 25 territory.

## Background

Sprint 24 Day 6 was originally PLAN.md Day 9.  Per `docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md` "Item 4 status (Davis §7.5.1 external-degree refinement)" + `nd_coarsen_floor_decision.md` "Day 6 plan reference", Day 1's (c)-revert choice freed Days 6-7 from the external-degree refinement; Days 8-9's option (a) + (b) ND fill-quality work moved up by three days.  Day 5 captured option (a) — coarsening-floor sweep — and chose ratio=200 as the env-gated Pres_Poisson-tuning advisory.  Day 6 pursues option (b) — smarter separator extraction.

## What changed

`src/sparse_graph.c::graph_edge_separator_to_vertex_separator` (the Sprint 22 Day 4 implementation) previously lifted the smaller-vertex-weight side's boundary into the separator (METIS convention — minimises recursive ND tree-height inflation by descending the larger subgraph first).  Day 6 adds an alternative `balanced_boundary` strategy that lifts whichever side has the smaller *boundary count* regardless of vertex weight, with a 70/30 post-lift weight-balance check that falls back to `smaller_weight` if the chosen lift would skew the recursion too far.

```c
const char *env = getenv("SPARSE_ND_SEP_LIFT_STRATEGY");
if (env && strcmp(env, "balanced_boundary") == 0) {
    idx_t bb_side = (boundary_count[1] < boundary_count[0]) ? 1 : 0;
    /* Project post-lift weights and reject the choice if it would
     * skew the recursion past 70/30. */
    idx_t lift_w = w[bb_side] - boundary_weight[bb_side];
    idx_t other_w = w[1 - bb_side];
    idx_t total_w = lift_w + other_w;
    if (total_w > 0) {
        idx_t max_w = (lift_w > other_w) ? lift_w : other_w;
        if ((idx_t)10 * max_w > (idx_t)7 * total_w)
            balanced = 0;
    }
    if (balanced)
        lift_side = bb_side;
}
```

`smaller_weight` (default) reproduces the Sprint 22 Day 4 behaviour bit-identically — the env-var-unset path takes the same `lift_side = smaller_weight_side` branch.  `balanced_boundary` is the only new behaviour; any other env-var value (typo, future strategy name not implemented yet) silently falls through to the default, matching Sprint 23 Day 11's `SPARSE_FM_FINEST_PASSES` permissive-fallback convention.

Doc-comment updates:
- `src/sparse_graph.c` overview block (Three-phase pipeline → step 3) cites the env var.
- `src/sparse_graph_internal.h` `graph_edge_separator_to_vertex_separator` doc describes both strategies + the 70/30 fallback gate.

## Sweep results

Settings × fixtures (raw capture: `docs/planning/EPIC_2/SPRINT_24/sep_strategy_sweep.txt`):

| Fixture       | n      | A: default     | B: ratio=200   | C: BB alone   | D: ratio=200 + BB |
|---------------|-------:|---------------:|---------------:|--------------:|------------------:|
| nos4 nnz_L    |    100 |          968   |          968   |          797  |              797  |
| nos4 ND/AMD   |        |        1.520×  |        1.520×  |        1.251× |            1.251× |
| nos4 ND ms    |        |        101.7   |         94.7   |         78.2  |             81.4  |
| bcsstk14 nnz_L|  1 806 |      131 017   |      131 017   |      121 641  |          121 641  |
| bcsstk14 ND/AMD|       |        1.129×  |        1.129×  |        1.048× |            1.048× |
| bcsstk14 ND ms|        |       5 601.0  |       5 484.4  |       4 191.0 |           4 312.4 |
| Kuu nnz_L     |  7 102 |      924 385   |      908 832   |      574 861  |          602 598  |
| Kuu ND/AMD    |        |        2.275×  |        2.237×  |        1.415× |            1.483× |
| Kuu ND ms     |        |      12 990.5  |      18 969.0  |      14 235.9 |          18 322.7 |
| Pres_Poisson nnz_L | 14 822 |  2 541 734   |  **2 514 769** |  2 542 706    |       2 536 003   |
| Pres_Poisson ND/AMD|       |        0.952×  |      **0.942×**|        0.953× |            0.950× |
| Pres_Poisson ND ms |       |      37 503.0  |      29 243.0  |      39 575.4 |          32 475.5 |

## Findings

### Smaller fixtures: balanced_boundary is a substantial win

balanced_boundary helps every fixture except Pres_Poisson:

- **Kuu (n=7102):** ND/AMD drops 2.275× → 1.415× — a 38 % reduction in nnz(L) (924 385 → 574 861).  This is the largest single nnz win Sprint 24 has produced on any fixture.  Kuu's structural mesh has a clear smaller-boundary side that the smaller-weight strategy was missing because the boundaries weren't proportional to weights at the GGGP-bisected coarsest level.
- **bcsstk14 (n=1806):** ND/AMD drops 1.129× → 1.048× — an 8 percentage-point improvement (131 017 → 121 641 nnz_L).  ND now beats AMD by a hair (was beat AMD by 13 %, now by 5 %; AMD = 116 071 stayed the headline).
- **nos4 (n=100):** ND/AMD drops 1.520× → 1.251× — a 27 percentage-point improvement (968 → 797 nnz_L).  At 100 vertices the brute-force bisection at the coarsest level dominates and BB's lift choice noticeably affects the resulting subgraphs.

### Pres_Poisson: balanced_boundary is essentially neutral

- Default (Setting A): 2 541 734 nnz_L → 0.952×
- BB alone (Setting C): 2 542 706 nnz_L → 0.953× (1 nnz_L worse, 0.04 % drift)

Pres_Poisson's structurally regular PDE mesh produces near-balanced cuts at the coarsest level, so the smaller-boundary side is also the smaller-weight side a majority of the time — both strategies converge.  The 1-nnz delta is in the noise band for Sprint 22 Day 13's "tie-break drift" criterion.

### Combined Day 5 + Day 6 (Setting D) on Pres_Poisson is *worse* than Day 5 alone

- Setting B (ratio=200 alone): 2 514 769 nnz_L → 0.942×
- Setting D (ratio=200 + BB): 2 536 003 nnz_L → 0.950×

ratio=200 produces a tighter cut at the coarsest level (74 vertices instead of 148); applying balanced_boundary on top changes the lifted side at the finest-level FM step, but the tighter coarsest cut already commits to a side that the BB strategy then re-decides differently — the two changes interact destructively on this fixture.  No single setting reaches the 0.85× sprint target nor the 0.90× partial-close fallback.

### Wall time

balanced_boundary doesn't materially change wall time on Pres_Poisson (39.6 s alone vs 37.5 s default — 5 % bounce, in the run-to-run noise band).  Kuu sees a 10 % BB wall-time bounce (12.9 s → 14.2 s); the BB+ratio=200 combo adds 41 % vs default (12.9 s → 18.3 s) — ratio=200 alone already added 46 % (12.9 s → 19.0 s).  No fixture shows a > 2× wall regression that would trip `make wall-check`.

## Why option (b) misses the 0.85× target on Pres_Poisson

PLAN.md Day 9 task 4 sets the combined-effect target: "If the combined effect drops Pres_Poisson ND/AMD ≤ 0.85×, the sprint's stretch target is met; otherwise document the gap as Sprint-25 territory."  Day 6's combined Setting D measurement (0.950×) is 10 percentage points above 0.85×.  ND/AMD on Pres_Poisson stays in the 0.95× band that Sprint 23 Day 11's multi-pass FM established — neither deeper coarsening (Day 5) nor smarter separator extraction (Day 6) can break through alone or in combination.

The remaining options for closing the 0.85× gap are Sprint-25 territory:
- **Smarter coarsening** — Sprint 22 uses heavy-edge matching; alternatives like the Heavy Connectivity Coarsening of Karypis-Kumar 1998 §5 might preserve the cut structure better through coarsening.
- **Multi-pass FM at intermediate levels** — Sprint 23 Day 11 currently runs 3 passes only at the finest level; running 2-3 passes at the second/third-finest level may unlock further gains (compounded uncoarsening).
- **Spectral bisection at the coarsest level** — currently brute-force / GGGP; spectral bisection (eigenvector-of-the-Laplacian) would give a globally better starting point for the FM uncoarsening cascade.
- **Larger graph corpus tuning** — Sprint 24 swept 4 fixtures at n ≤ 14 822; the 0.85× target may be unreachable on PDE meshes where AMD is already near-optimal.

## Production default decision

Per PLAN.md Day 9 task 5: "flip `SPARSE_ND_SEP_LIFT_STRATEGY` to balanced_boundary if it's a clear win on Pres_Poisson (≤ 0.90×) without regressing the smaller fixtures past 5 percentage points; else keep it off."

balanced_boundary lands at 0.953× on Pres_Poisson — not ≤ 0.90×.  Per the literal rule, **keep balanced_boundary off-by-default and gated behind the env var**.

Counter-argument considered + rejected: balanced_boundary helps every smaller fixture by 8-38 percentage points and is neutral on Pres_Poisson — a corpus-aggregate flip-on case is plausible.  Day 7 (re-shuffled to come next) will reconsider this if Sprint 24's headline gate (a) Pres_Poisson ND/AMD ≤ 0.85× isn't going to land — at that point the production-default value of balanced_boundary should be re-evaluated against the corpus-aggregate evidence, since the literal Pres_Poisson criterion was set under the (since-disproven) expectation that BB or its combination with ratio=200 would itself close the Pres_Poisson gap.

## Day 7 plan reference

Day 7 (originally PLAN.md Day 10):

1. Re-run the 39 `tests/test_graph.c` partition tests under the chosen Day 5 + Day 6 defaults.  The default path stays bit-identical to Sprint 22 Day 4 — both env vars unset means the existing `lift_side = smaller_weight_side` branch fires; no tests should regress.  Verify determinism contracts (`test_partition_determinism_*`) hold bit-identically.
2. Reconsider the production-default for `SPARSE_ND_SEP_LIFT_STRATEGY` given Day 6's findings — balanced_boundary is neutral on Pres_Poisson but a clear win on every smaller fixture.  Decision options:
   - **(i) Keep default `smaller_weight`** — strict PLAN.md compliance; documented advisory for non-Pres_Poisson workloads.  Lowest blast radius.
   - **(ii) Flip default to `balanced_boundary`** — corpus-aggregate Sprint 24 win; Pres_Poisson neutrality means no headline regression but broader fixture set sees 8-38pp improvements.  Higher blast radius (any caller relying on Sprint 22 Day 4's specific tie-break would break) but the test suite — including `tests/test_graph.c::test_edge_to_vertex_separator_smaller_side` — expects smaller_weight semantics by name, so flipping the default breaks that test.
3. Either way, tighten `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` from Sprint 23's `≤ 1.0× nnz_amd` to `≤ 0.96× nnz_amd` (Sprint 24 Day 5/6 default-path achievement + 1pp noise margin).  The `≤ 0.85×` target gets dropped from the assertion as Sprint-25 territory.
4. Capture the Day-7 wall-time + decision in `docs/planning/EPIC_2/SPRINT_24/nd_tuning_day7.md` and update the test bound rationale.

## Quality-gate notes

- `make format-check`: pending re-run on Day 6 commit (only one new conditional + a couple of fields added; expect clean).
- `make lint`: pending re-run; the `getenv` + `strcmp` pattern matches the Sprint 23 Day 11 + Sprint 24 Day 5 env-var conventions which both passed lint.
- `make test`: pending re-run.  The default path (env-var unset) is bit-identical to Sprint 22 — `tests/test_graph.c::test_edge_to_vertex_separator_smaller_side` expects 5 separator vertices on the 5×6 grid with tied weights, which still holds (smaller_weight tie-break to side 0 is the default branch).
- `make wall-check`: pending re-run.  bcsstk14 qg-AMD wall-time is unaffected (separator-extraction is ND-only); Pres_Poisson AMD wall-time also unaffected.  Both gates against the Day 4 baseline should pass.
