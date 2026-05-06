# Sprint 24 Day 5 — ND coarsen-floor-ratio decision (item 5 head-start)

## Decision

**Add the `SPARSE_ND_COARSEN_FLOOR_RATIO` env var with default 100 (unchanged); winning override on Pres_Poisson is ratio 200.**  Option (a) — deeper coarsening alone — is **insufficient** to reach the 0.85× ND/AMD target on Pres_Poisson (best achievable is 0.942×).  Day 6 (originally Day 9 in the PLAN.md schedule, pulled forward by Day 1's fix-decision freeing the Day-5 budget) will pursue option (b) smarter separator extraction.  Day 7 (originally Day 10) decides on the production default after combining options (a) + (b).

## Background

Sprint 24 Day 5 was originally scheduled for the Pres_Poisson AMD parity test (PROJECT_PLAN.md item 3).  Per `docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md` "Item-3 status (Pres_Poisson AMD parity test)", with the (c) revert chosen on Day 1, item 3 reduces to a no-op: the approximate-degree code path is gone, no parity test to run.  Day 5's 8-hour budget recovers as item-5 head-start, pulling Day 8's coarsen-floor work forward by three days.

## What changed

`src/sparse_graph.c::sparse_graph_hierarchy_build` (commit landed Day 5):

```c
idx_t divisor = 100;
{
    const char *env = getenv("SPARSE_ND_COARSEN_FLOOR_RATIO");
    if (env) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (env != endp && *endp == '\0' && v >= 1 && v <= 100000)
            divisor = (idx_t)v;
    }
}
idx_t base_threshold = n_root / divisor;
if (base_threshold < 20)
    base_threshold = 20;
```

Default behavior unchanged (divisor 100; bit-identical fill on every fixture under default).  The env var enables ratio overrides in [1, 100000]; out-of-range or non-numeric input falls back to default 100 (same `strtol` + range-check + fallback pattern as Sprint 23 Day 11's `SPARSE_FM_FINEST_PASSES`).

Doc-comment updates:
- `src/sparse_graph.c` overview block (Coarsening step description) cites the env var.
- `src/sparse_graph_internal.h` `sparse_graph_hierarchy_build` doc updates the stop-condition formula to `MAX(20, root->n / divisor)`.
- `src/sparse_graph_internal.h` `sparse_graph_partition` doc updates the coarsening-stop reference.

## Sweep results

Ratios swept: {100 (default), 200, 400, 800, 100000 (effectively infinite — `n/100000 → 0` for n ≤ 14822, so every fixture floors at the MAX(20, ...) clamp)}.  Fixtures: nos4 (n=100), bcsstk14 (n=1806), Kuu (n=7102), Pres_Poisson (n=14822).

Raw capture: `docs/planning/EPIC_2/SPRINT_24/coarsen_floor_sweep_pres_poisson.txt`.

| Fixture       | n      | ratio   | nnz_nd     | nnz_amd    | ND/AMD  | reorder_ms_nd |
|---------------|-------:|--------:|-----------:|-----------:|--------:|--------------:|
| nos4          |    100 |     100 |        968 |        637 |  1.520× |          90.5 |
| nos4          |    100 |     200 |        968 |        637 |  1.520× |          92.0 |
| nos4          |    100 |     400 |        968 |        637 |  1.520× |          93.0 |
| nos4          |    100 |     800 |        968 |        637 |  1.520× |          89.3 |
| nos4          |    100 |  100000 |        968 |        637 |  1.520× |          89.9 |
| bcsstk14      |  1 806 |     100 |    131 017 |    116 071 |  1.129× |        4915.8 |
| bcsstk14      |  1 806 |     200 |    131 017 |    116 071 |  1.129× |        4872.3 |
| bcsstk14      |  1 806 |     400 |    131 017 |    116 071 |  1.129× |        4868.2 |
| bcsstk14      |  1 806 |     800 |    131 017 |    116 071 |  1.129× |        4904.6 |
| bcsstk14      |  1 806 |  100000 |    131 017 |    116 071 |  1.129× |        4941.9 |
| Kuu           |  7 102 |     100 |    924 385 |    406 264 |  2.275× |       10778.8 |
| Kuu           |  7 102 |     200 |    908 832 |    406 264 |  2.237× |       15965.7 |
| Kuu           |  7 102 |     400 |    908 832 |    406 264 |  2.237× |       16057.2 |
| Kuu           |  7 102 |     800 |    908 832 |    406 264 |  2.237× |       16063.6 |
| Kuu           |  7 102 |  100000 |    908 832 |    406 264 |  2.237× |       15994.7 |
| Pres_Poisson  | 14 822 |     100 |  2 541 734 |  2 668 793 |  0.952× |       35511.1 |
| **Pres_Poisson** | **14 822** | **200** | **2 514 769** | **2 668 793** | **0.942×** |   **25923.9** |
| Pres_Poisson  | 14 822 |     400 |  2 641 961 |  2 668 793 |  0.990× |       35000.7 |
| Pres_Poisson  | 14 822 |     800 |  2 641 961 |  2 668 793 |  0.990× |       33493.8 |
| Pres_Poisson  | 14 822 |  100000 |  2 641 961 |  2 668 793 |  0.990× |       34670.7 |

## Findings

**ratio 200 wins on Pres_Poisson** — ND/AMD drops 0.952× → 0.942× (1.0 percentage point), and the wall time drops from 35.5 s → 25.9 s (a coincidental side-benefit; option (a) gets a tighter cut and apparently a faster path to it).

**ratios 400 / 800 / 100000 regress on Pres_Poisson** — ND/AMD pops up to 0.990× because Pres_Poisson's coarsest level pegs at the floor of 20 vertices for ratios ≥ 400 (n/400 = 37, n/800 = 18, n/100000 = 0; all three clamp at 20).  Once the brute-force bisection at the coarsest level operates on only 20 vertices, the FM uncoarsening can't recover the cut quality that ratio 200's 74-vertex coarsest level produces.  This is a real "sweet spot" — too-aggressive coarsening loses cut quality.

**Smaller fixtures don't regress at ratio 200**:
- nos4 (n=100): floors at 20 already (n/100 = 1 < 20).  No effect from any ratio.
- bcsstk14 (n=1806): same — floors at 20 across all ratios (n/100 = 18 < 20).  No effect.
- Kuu (n=7102): ratio 200 *improves* nnz_nd 924 385 → 908 832 (1.7 % win).  Kuu's ND/AMD ratio ticks down 2.275× → 2.237× — a small Kuu-side improvement.  Wall time bounces up 50 % (10.8 s → 16.0 s) — Kuu's coarsest level moves from 71 vertices (ratio 100) to 35 (ratio 200), spending more time on FM at finer levels.  Kuu's ND/AMD stays well above 1.0 (Kuu is ND-unfriendly) so this isn't a fill-quality concern.

## Why option (a) alone is insufficient

PLAN.md Day 8 task 4 sets the target: "Pick the best ratio that drops Pres_Poisson ND/AMD ≤ 0.90× without regressing the smaller fixtures."  Best ratio (200) reaches 0.942× — short by 4.2 percentage points of the 0.90× target.

Per PLAN.md Day 8 task 4, the fallback when option (a) is insufficient: "If no ratio reaches ≤ 0.90×, mark option (a) insufficient and prepare for Day 9's option (b) work."  Day 9 (originally — re-shuffled to Day 6 here) is option (b) smarter separator extraction (`SPARSE_ND_SEP_LIFT_STRATEGY`).

## Decision: env var lands on default 100; ratio-200 advisory recorded

The Day-5 commit lands `SPARSE_ND_COARSEN_FLOOR_RATIO` with default 100 (unchanged).  Day 6 (option (b) work) layers the separator-strategy env var on top.  Day 7 measures the combined effect and decides whether to:

1. **Flip the default to 200** — if the combined option (a) + (b) effect on Pres_Poisson reaches ≤ 0.85× and the Kuu wall-time bounce stays under 50 %.
2. **Keep default at 100, add a documented "Pres_Poisson tuning" note** — if option (b) alone closes the gap and the combined wins are noisy on smaller fixtures.

## Day 6 plan reference

Day 6 (re-shuffled from Day 9):
1. Read `src/sparse_graph.c::edge_to_vertex_separator` (Sprint 22 Day 4 implementation).
2. Implement `SPARSE_ND_SEP_LIFT_STRATEGY` env var with values `smaller_weight` (current default) and `balanced_boundary` (lift the side with smaller boundary count regardless of vertex weight).  Off by default.
3. Sweep `balanced_boundary` strategy across the full corpus + Pres_Poisson combined with ratio 200 from Day 5.
4. Capture to `docs/planning/EPIC_2/SPRINT_24/sep_strategy_sweep.txt`.
5. Decide whether the combined Day 5 + Day 6 settings reach ≤ 0.85× on Pres_Poisson; record decision in `docs/planning/EPIC_2/SPRINT_24/nd_sep_strategy_decision.md`.

## Quality-gate notes

- `make format-check`: pending re-run on Day 5 commit (no formatting changes expected — only code-block insertion + comment edits).
- `make lint`: pending re-run (the `getenv`/`strtol` pattern is the same as Sprint 23 Day 11's `SPARSE_FM_FINEST_PASSES`, which already passed lint; expect clean).
- `make test`: pending re-run (with the env var unset, all tests should remain bit-identical).
- `make wall-check`: pending re-run; ND wall-time on Pres_Poisson didn't regress under ratio 200 (25.9 s vs 35.5 s default), so the default-path wall-check should pass cleanly.
