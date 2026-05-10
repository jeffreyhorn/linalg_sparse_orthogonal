# Sprint 27 Day 13 Cross-Corpus Re-Bench — Headline Summary

## Methodology

Sprint 27 PLAN.md Day 13 task 1: ≤24 representative combinations × 6 corpus fixtures × 2 metrics (nnz_L + ND wall) = ≤288 measurements.  Captured in `bench_day13_combinations.csv` + `.txt`.  Each combination run via `build/bench_reorder --only <fixture> --skip-factor` with the named env vars exported (or `--nd-threshold 256` for settings 17 + 18).

Default Sprint 27 production state: `SPARSE_ND_COARSENING=hcc` (Day 2 flip), `nd_base_threshold=128` (Day 3 flip).  All other axes opt-in via env var.

## Headline: Sprint 27 default is the Pres_Poisson best

**Setting 2 (Sprint 27 default; HCC + t=128) is the lowest Pres_Poisson nnz_L across all 24 combinations** — every algorithmic-axis advisory and every combination either matches or regresses Pres_Poisson.

### Pres_Poisson top 5 (lowest ND/AMD ratio)

| Setting | Description | Pres_Poisson ratio |
|---:|---|---:|
| **2** | **Sprint 27 default (HCC + t=128)** | **0.9226×** |
| 19 | HEM + t=96 + annealing | 0.9271× |
| 15 | spectral + thick-restart | 0.9333× |
| 7 | annealing-linear | 0.9424× |
| 9 | root-spectral | 0.9437× |

### Pres_Poisson worst 3 (catastrophic regress)

| Setting | Description | Pres_Poisson ratio |
|---:|---|---:|
| 21 | spectral + fixed_k×hybrid | 1.6481× |
| 23 | fixed_k×degree + annealing | 1.9898× |
| 5 | fixed_k×degree | 2.0883× |

The fixed_k×degree variants don't combine well with anything — they regress ND nnz_L to **above AMD** (>1.0× — ND becomes worse than the alternative ordering).  This is the most striking finding: stacking advisory env vars OFTEN MAKES THINGS WORSE on Pres_Poisson.

## Corpus-wide best (geometric mean of 6 fixture ratios)

| Setting | Description | Geomean | Pres_Poisson | Kuu |
|---:|---|---:|---:|---:|
| **17** | **t=256 Kuu opt-in (--nd-threshold 256)** | **1.1156** | 0.9475× | 1.7716× |
| 2 | Sprint 27 default | 1.1551 | 0.9226× | 1.8822× |
| 7 | annealing-linear | 1.1588 | 0.9424× | 1.8830× |
| 12 | thick-restart-gauss_noise | 1.1715 | 0.9890× | 1.9020× |
| 8 | annealing-cosine | 1.1719 | 0.9505× | 2.0058× |

**Setting 17 (`--nd-threshold 256`) wins corpus-wide by virtue of its huge Kuu improvement (-5.6pp vs default).**  But it loses Pres_Poisson by 2.5pp.  This is a clear bimodal-class-vs-FE-mesh tradeoff — no single setting wins on both fronts.

## Kuu best (lowest ND/AMD ratio)

| Setting | Description | Kuu ratio | vs default |
|---:|---|---:|---:|
| **18** | **t=256 + fixed_k×hybrid** | **1.2170×** | **−35.3%** |
| 3 | fixed_k×hybrid | 1.2286× | −34.7% |
| 20 | HEM + fixed_k×hybrid | 1.2392× | −34.2% |
| 16 | fixed_k×hybrid + annealing | 1.2410× | −34.1% |
| 21 | spectral + fixed_k×hybrid | 1.3323× | −29.2% |

Kuu opt-in via setting 18 is the largest single fixture improvement Sprint 27 produced (1.882× → 1.217× = **−35.3% nnz_L**).  Combines `--nd-threshold 256` (Day-3 advisory) with `fixed_k×hybrid` (Day-4 advisory).  Documented in `nd_base_threshold_decision.md` + `per_vertex_fixed_k_decision.md`; ships as advisory recipe.

## Sprint 27 Default (Setting 2) Per-Fixture

| Fixture | n | AMD | ND | ratio | wall |
|---|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 1.000× | 0 ms |
| bcsstk04 | 132 | 3 143 | 3 722 | 1.184× | 184 ms |
| Kuu | 7 102 | 406 264 | 764 664 | 1.882× | 8 341 ms |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 1.124× | 700 ms |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 1.028× | 6 713 ms |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | **0.923×** | 10 157 ms |

## Sprint 27 Headline: Literal 0.85× Target — UNMET (5th Consecutive Sprint)

**Pres_Poisson: 0.923× of AMD — 7.3pp from the literal 0.85× target.**

Sprint 27's algorithmic-axis explorations:
- Item 4 (annealing FM): regressed Pres_Poisson 2.2-3.1pp under all 3 schedules (Day 7).
- Item 5 (root-spectral): regressed Pres_Poisson 2.3pp (Day 9).
- Item 4+5 combined: regressed Pres_Poisson 2.4pp (Day 9).
- Item 6 (thick-restart): regressed Pres_Poisson 4.7-11.5pp under all 3 perturbations (Days 11-12).

**Day-13 cross-corpus matrix CONFIRMS the empirical conclusion: no combination of Sprint 27's advisory axes lands Pres_Poisson ≤ 0.85×.**  The closest combinations cluster around 0.93-0.95× and represent slight further regress from the default 0.923×.

The 5-sprint trajectory:
| Sprint | Pres_Poisson best opt-in | Δ vs literal 0.85× |
|---|---:|---:|
| 22 | 1.063× | +21.3pp |
| 23 | 0.952× | +10.2pp |
| 24 | 0.942× | +9.2pp |
| 25 | 0.922× | +7.2pp |
| 26 | 0.9217× | +7.2pp |
| **27** | **0.923×** | **+7.3pp** |

Improvement has flatlined at ~0.92× across 4 sprints despite 24-combination matrix exploration.  Combined with Sprint 27's per-axis regressions, this is strong empirical evidence that **the multilevel pipeline + leaf-AMD reaches near-optimal cuts on Pres_Poisson that pipeline-level interventions cannot improve**.

## Production-Default Decisions (Day 13 Verdict)

| Axis | Sprint 27 verdict | Status |
|---|---|---|
| `SPARSE_ND_COARSENING` | **Flipped Day 2: heavy_edge → hcc** (Kuu-safe; CV-detection-and-HEM-fall-through) | Production |
| `sparse_reorder_nd_base_threshold` | **Flipped Day 3: 96 → 128** (relaxed flip rule + corpus-wide wall improvement) | Production |
| `SPARSE_FM_FINEST_STRATEGY` | Stays `baseline` — annealing + thick_restart both regress Pres_Poisson | Advisory |
| `SPARSE_ND_ROOT_BISECT` | Stays `multilevel` — spectral regresses Pres_Poisson 2.3pp | Advisory |
| `SPARSE_ND_SEP_LIFT_STRATEGY` | Stays default — fixed_k variants are bimodal-class-specific | Advisory |

**Sprint 27 ships 2 default flips (Days 2 + 3) plus 4 advisory env-var-gated paths.**  No further default flips warranted by Day-13's matrix.

## Per-Fixture-Class Advisory Recipes (Day 13 Validated)

These ship as opt-in recipes documented in `docs/algorithm.md` (Day 14):

| Fixture class | Recipe | Win |
|---|---|---|
| **Bimodal-degree (Kuu)** | `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k SPARSE_ND_SEP_LIFT_WEIGHT=hybrid build/bench_reorder --nd-threshold 256` | **−35.3 % nnz_L** (Setting 18) |
| **Tiny irregular (bcsstk04)** | `SPARSE_ND_ROOT_BISECT=spectral` | −1.3 % nnz_L + 23× wall speedup |
| Mid-irregular (bcsstk14) | `SPARSE_FM_FINEST_STRATEGY=annealing` | −0.7 % nnz_L |
| Mid-irregular (s3rmt3m3) | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=random_flip` | −1.0 % nnz_L |

## Test Bound Tightening (Day 13 Task 4)

`tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd`:
- **Tightened: 0.96× → 0.94×** (Sprint 27 default 0.923× + 2pp noise margin = 0.943× → round to 0.94×).
- Pin the Sprint 27 Day 3 production default; reject any future commit that regresses the headline ratio.

`tests/test_reorder_nd.c::test_nd_10x10_grid_matches_or_beats_amd_fill` and `test_nd_bcsstk14_fill_vs_amd`: not tightened; Day 13 measurements show their bounds are still sane.

## Sprint 27 Closure Status

**Closing items 1-7 (algorithmic axes + cross-corpus re-bench):**
- Item 1 (HCC Kuu-safe matching variant): SHIPPED + flipped to default (Day 2).
- Item 2 (`nd_base_threshold` relaxed re-sweep): SHIPPED + flipped to t=128 default (Day 3).
- Item 3 (fixed-K per-vertex selection mode): SHIPPED as advisory (Day 4).
- Item 4 (annealing FM): SHIPPED as advisory (Days 5-7).
- Item 5 (root-level spectral): SHIPPED as advisory (Days 7-9).
- Item 6 (thick-restart FM): SHIPPED as advisory (Days 10-12).
- Item 7 (cross-corpus re-bench + decisions): COMPLETED today.

**Headline target outcome:** UNMET (0.85× literal target; 5th consecutive sprint).  Routed to Sprint 28+ for non-pipeline-level interventions per PROJECT_PLAN.md follow-up.

**Pres_Poisson ND wall improvement (cumulative across Sprint 27):**
| Sprint | Pres_Poisson ND wall |
|---|---:|
| Sprint 25 | ~38 100 ms |
| Sprint 26 (t=96 default flip Day 5) | ~12 200 ms (-67.9 %) |
| **Sprint 27 (HCC + t=128 default flips)** | **~10 100 ms** (-73.5 % vs Sprint 25) |

Even though the literal nnz_L target is unmet, **Sprint 27 ships a meaningful ~73 % cumulative reduction in Pres_Poisson ND wall** vs Sprint 25 baseline.

## Files Generated Day 13

- `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations.csv` — 24 settings × 6 fixtures (raw)
- `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations.txt` — same data, grouped by setting
- `docs/planning/EPIC_2/SPRINT_27/headline_summary.md` — this document
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound tightened 0.96× → 0.94×
