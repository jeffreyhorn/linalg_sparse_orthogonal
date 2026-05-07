# Sprint 25 Day 10 — `SPARSE_ND_COARSENING` Production-Default Decision

## Decision

**Production default stays `SPARSE_ND_COARSENING=heavy_edge`** (Sprint
22 baseline; bit-identical to current master).  HCC ships behind the
env var as advisory.

**Recommended advisory combination for Pres_Poisson workloads:**

```
SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200
```

(Day 9 setting 13: closes Pres_Poisson ND/AMD from 0.952× → 0.922×.)

## Why the default flip was attempted and reverted

Day 9's combined-effect sweep (`bench_day9_combinations.txt`)
identified setting 13 (HCC + Sprint 24's ratio=200) as the
Pres_Poisson winner at 0.9218× — a 3pp tightening vs default 0.9524×.
Per-fixture profile of setting 13 looked corpus-safe:

| fixture | default | setting 13 | Δ |
|---|---|---|---|
| nos4 | 1.520× | 1.414× | -10.6pp WIN |
| bcsstk04 | 1.178× | 1.180× | +0.2pp noise |
| Kuu | 2.275× | 2.099× | -17.6pp WIN |
| bcsstk14 | 1.129× | 1.123× | -0.6pp small win |
| s3rmt3m3 | 1.009× | 1.019× | +1.0pp small noise |
| Pres_Poisson | 0.952× | 0.922× | -3.0pp WIN |

Day 10 attempted the default flip:
- `parse_coarsening_strategy()` default changed from `COARSENING_HEAVY_EDGE` → `COARSENING_HCC`
- `divisor` default changed from 100 → 200

Two test failures surfaced under the new defaults:

1. **`test_hierarchy_build_5x5_grid`** (fixable):  the 5×5 grid
   coarsens to 14 vertices under HCC vs 13 under HEM (consistent
   with Sprint 25 Day 2's diagnostic).  The test asserts `n <= 13`
   — would need relaxation to `n <= 14` for HCC default.  Solvable
   with a one-line bound update.

2. **`test_partition_bcsstk14_smoke`** (blocker):  under HCC's
   coarsening, the multilevel pipeline's separator extraction on
   bcsstk14 produces `sep = 0` — a degenerate empty separator.
   Day 10 isolation testing confirmed the sep=0 outcome is caused
   by HCC alone (independent of `SPARSE_ND_COARSEN_FLOOR_RATIO`):

   ```
   HCC alone (ratio=100):    bcsstk14 sep=0  ← BLOCKER
   HEM + ratio=200:          bcsstk14 sep=97 ✓
   HCC + ratio=200:          bcsstk14 sep=0  ← BLOCKER (HCC-driven)
   HEM + ratio=100 (default):bcsstk14 sep=97 ✓
   ```

   The HCC matching choice on bcsstk14 produces a multilevel coarse
   graph whose 2-way bisection assigns essentially all vertices to
   one side, leaving no boundary vertices to lift into the
   separator.  `sparse_reorder_nd` (the recursive ND public API)
   handles `sep = 0` by falling through to natural ordering on the
   degenerate subtree, which is why `bench_reorder.c`'s
   ND-via-`sparse_reorder_nd` measurements still produce a valid
   (if mediocre) `nnz_L` of 130 358 on bcsstk14 under HCC — the
   degenerate single-level partition just degrades the recursion
   structure rather than failing outright.

   But `test_partition_bcsstk14_smoke` exercises
   `sparse_graph_partition` (the single-level partitioner) directly
   and asserts `sep > 0` as a correctness contract.  This test
   captures a real correctness gap in HCC's interaction with
   bcsstk14-class fixtures — the gap exists regardless of whether
   the env-var default is flipped or not, but flipping the default
   would make the gap a CI-failing regression on every PR.

   Resolving this gap requires either:
   - Investigating HCC's matching pattern on bcsstk14 to identify
     why it produces a degenerate coarse-level partition (likely a
     specific structural property of bcsstk14 that interacts with
     HCC's `min(deg(u), deg(v))` weighting).  Sprint 26 territory.
   - Adding a fall-back in `sparse_graph_partition` for the sep=0
     case (e.g. detect → re-bisect with a different strategy).
     Cross-cutting fix; also Sprint 26 territory.

   Given the blocker, the HCC default flip is reverted.  HCC ships
   behind the env var as advisory only.

## PLAN.md flip rule application

PLAN.md Sprint 25 item 1 + Day 10 task 1 spec the flip rule:

> Flip `SPARSE_ND_COARSENING` to `hcc` if the corpus-wide HCC win
> is clear (≥ 1pp Pres_Poisson tightening + no smaller-fixture
> regression).

Application:

| sub-rule | HCC alone | HCC + ratio=200 |
|---|---|---|
| ≥ 1pp Pres_Poisson tightening | ✓ (1.5pp) | ✓ (3.0pp) |
| No smaller-fixture regression past 5pp band | ✗ (Kuu +19.5pp under HCC alone) | ✓ (Kuu -17.6pp, others ≤ 1pp noise) |
| **No new test failures under defaults** | **✗ (bcsstk14 sep=0)** | **✗ (bcsstk14 sep=0)** |

The PLAN.md flip rule didn't anticipate the bcsstk14 sep=0 finding
(it focused on per-fixture nnz_L deltas).  Day 10 surfaced this as
a hidden third sub-rule: any default flip must also preserve the
existing test contracts.  HCC's interaction with bcsstk14 fails
that test, blocking the flip independent of the per-fixture
nnz_L deltas.

## Per-fixture advisory (Sprint 25 user guidance)

| workload | recommended setting | expected fill quality |
|---|---|---|
| Pres_Poisson (2D PDE meshes) | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` | ND/AMD ~0.92× (vs default 0.95×) |
| Kuu / irregular SPD | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200 SPARSE_FM_INTERMEDIATE_PASSES=3 SPARSE_ND_COARSEST_BISECTION=spectral SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (full set) | ND/AMD ~1.31× (vs default 2.28×; -97pp) |
| bcsstk14 / similar SPD | (default) `heavy_edge` | bcsstk14 sep=0 risk on HCC; stay default |
| nos4 / small grids | (default) — env vars are no-ops at this size | unchanged |
| s3rmt3m3 | `SPARSE_FM_INTERMEDIATE_PASSES=2` (Day 5 advisory) | ND/AMD ~1.02× (within noise) |
| bcsstk04 | (default) | bench is small enough that env vars are no-ops |

## What didn't ship

Three of three Sprint 25 algorithmic axes failed to flip their
defaults:

1. `SPARSE_ND_COARSENING` (this doc) — blocked by bcsstk14 sep=0.
2. `SPARSE_FM_INTERMEDIATE_PASSES` — Day 5's `intermediate_fm_decision.md`
   already documented the no-flip outcome (Pres_Poisson ≤ 1pp rule
   not met by passes=2; passes=3 regresses Pres_Poisson).
3. `SPARSE_ND_COARSEST_BISECTION` — see
   `spectral_bisection_decision.md` (spectral alone barely moves
   Pres_Poisson; default stays gggp).

The Sprint 25 stretch target ≤ 0.85× on Pres_Poisson misses.  Best
achievable via env-var combinations is 0.922× (setting 13).
Sprint 26 inherits the residual gap per `nd_tuning_day8.md`
"Sprint 26 routing" candidates.

## Production-default decision tree (for future reference)

For any future "flip default to X" decision, Day 10's experience
suggests this validation sequence:

1. **Per-fixture nnz_L delta vs current default** — must satisfy
   the PLAN's flip rule (≥ 1pp headline win + no > 5pp small-
   fixture regression).
2. **Per-fixture wall_time delta vs current default** — must not
   regress wall_time past ~50 % (Sprint 24 Day 5's bar for
   ratio=200's Kuu wall +50 % stand-alone).  Sprint 25 Day 9
   measured setting 13's bcsstk14 + Kuu wall = better than default.
3. **Existing test contracts** (Day 10's added gate) — the new
   default must not regress ANY test in `make test`.  This is the
   gate HCC + ratio=200 fails on bcsstk14 sep=0.
4. **Edge cases on synthetic fixtures** (n=1, n=2, disconnected,
   star, path) — must produce structurally valid output per the
   Sprint 25 Day 8 spectral test pattern.

If all four sub-gates pass, flip the default.  If any fail,
document the gap + ship behind the env var as advisory.

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 10
- `docs/planning/EPIC_2/SPRINT_25/headline_summary.md` — Day 9's
  Day-10-input recommendations (now superseded by Day 10's
  bcsstk14 finding)
- `docs/planning/EPIC_2/SPRINT_25/bench_day9_combinations.txt` —
  the corpus sweep data this decision is based on
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day3.md` — Day 3's
  HCC escalation (anticipated the Kuu regression but not the
  bcsstk14 sep=0 issue)
- `docs/planning/EPIC_2/SPRINT_25/hcc_design.md` — Day 1's HCC
  algorithm contract; Sprint 26 follow-up should investigate why
  HCC's `min(deg(u), deg(v))` weighting on bcsstk14 produces a
  degenerate coarse-level partition.
- `tests/test_graph.c::test_partition_bcsstk14_smoke` — line 1798's
  `ASSERT_TRUE(sep > 0)` is the contract HCC fails on bcsstk14.
