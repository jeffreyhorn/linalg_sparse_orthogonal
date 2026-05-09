# Thick-Restart FM at the Finest Level — Flip-or-Stay Decision (Sprint 27 Day 12)

## Background

Sprint 27 Day 10 designed thick-restart with global-best-tracking + per-pass perturbation; Day 11 implemented it across three perturbation modes (`random_flip`, `boundary_shuffle`, `gauss_noise`) and captured a corpus interim sweep.  Sprint 27 PLAN.md Day 12 task 1 specifies the flip rule:

> Thick-restart ships advisory unless it both lands the headline (Pres_Poisson ≤ 0.85× of AMD nnz_L) AND has cleaner flip-rule application than items 4-5.

This document captures Day 12's flip-rule application + the headline status update.

## Reproducer

```
SPARSE_FM_FINEST_STRATEGY=thick_restart build/bench_reorder --only <fixture> --skip-factor
SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=boundary_shuffle ...
SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=gauss_noise ...
```

Default coarsening: HCC + Kuu-safe (Sprint 27 Day 2); `nd_base_threshold = 128` (Sprint 27 Day 3).

## Sprint 27 Day 12 Sweep Table

(Captured in `thick_restart_sweep.txt`; ND nnz_L only.)

| Fixture | n | AMD | default | random_flip | boundary_shuffle | gauss_noise |
|---|---:|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 637 | 637 | 637 |
| **bcsstk04** | 132 | 3 143 | 3 722 | **3 677** | 4 471 | **3 677** |
| Kuu | 7 102 | 406 264 | 764 664 | 903 195 | 875 152 | 772 566 |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 130 733 | 138 061 | 132 599 |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | **482 731** | 489 901 | 488 049 |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | 2 578 552 | 2 745 603 | 2 639 539 |

### Per-fixture nnz_L deltas vs Sprint 27 Day 9 default (%)

| Fixture | random_flip | boundary_shuffle | gauss_noise |
|---|---:|---:|---:|
| nos4 | 0.0 % | 0.0 % | 0.0 % |
| bcsstk04 | **−1.2 % win** | +20.1 % | **−1.2 % win** |
| Kuu | **+18.1 %** | **+14.4 %** | +1.0 % |
| bcsstk14 | +0.2 % | +5.9 % | +1.7 % |
| s3rmt3m3 | **−1.0 % win** | +0.4 % | +0.04 % |
| **Pres_Poisson** | **+4.7 %** | **+11.5 %** | **+7.2 %** |

### Per-fixture ND/AMD ratios under thick-restart vs default

| Fixture | default ratio | random_flip | boundary_shuffle | gauss_noise |
|---|---:|---:|---:|---:|
| Pres_Poisson | 0.923× | 0.966× | 1.029× | 0.989× |
| Kuu | 1.882× | 2.223× | 2.154× | 1.902× |
| bcsstk14 | 1.124× | 1.126× | 1.190× | 1.142× |

## Flip-Rule Application

Sprint 27 PLAN.md Day 12 task 1:
- **(a)** Pres_Poisson ≤ 0.85× of AMD nnz_L AND
- **(b)** Cleaner flip-rule application than items 4-5 (annealing / root-spectral)

| Perturbation | Pres_Poisson ratio | (a) gate | (b) gate |
|---|---|---|---|
| random_flip | 0.966× ✗ | FAIL | n/a (a) failed) |
| boundary_shuffle | 1.029× ✗ (worse than AMD!) | FAIL | n/a |
| gauss_noise | 0.989× ✗ | FAIL | n/a |

**No perturbation lands Pres_Poisson ≤ 0.85×.**  Gate (a) fails decisively for all three (margins of −11.6 to −17.9pp from the literal target).  No flip.

Note that gate (a) is also strictly worse than items 4 + 5:
- Item 4 (annealing-best linear): Pres_Poisson 0.943× = +9.3pp from target.
- Item 5 (root-spectral): Pres_Poisson 0.944× = +9.4pp from target.
- Item 6 (thick-restart-best gauss_noise): Pres_Poisson 0.989× = +13.9pp from target.

Gate (b) (cleaner flip-rule than items 4-5) is moot since gate (a) already fails harder.

## Decision: Stay At Default `baseline`; Ship Thick-Restart As Advisory

The Day-10 hypothesis ("baseline FM converges to a suboptimal local minimum on Pres_Poisson; thick-restart's perturbation lets the FM walk explore different optima") fails empirically — perturbing the partition state breaks the carefully-constructed cut, and FM can't recover within the pass budget (default 3 passes per Sprint 23 Day 11).

Same root cause as Day 7's annealing rejection: Pres_Poisson's regular FE-mesh structure has a *cohesive* near-global cut that the multilevel pipeline reaches efficiently, and stochastic perturbation only disrupts that without escape benefit.  Three independent attempts (annealing, root-spectral, thick-restart) all confirm the empirical conclusion.

### Per-fixture-class advisory

| Fixture class | Thick-restart verdict | Default behaviour |
|---|---|---|
| Tiny (nos4) | Bit-stable | Stay default |
| **Small irregular (bcsstk04)** | **−1.2 % under random_flip OR gauss_noise** | Stay default; **advisory opt-in** |
| Mid-irregular (bcsstk14) | Slight regress | Stay default |
| **s3rmt3m3** | **−1.0 % under random_flip** | Stay default; **advisory opt-in** |
| Bimodal (Kuu) | Heavy regress (random_flip + boundary_shuffle); near-neutral under gauss_noise | Stay default |
| Regular FE-mesh (Pres_Poisson) | Heavy regress under all 3 | Stay default |

Workloads dominated by small-to-mid irregular SPDs (bcsstk04 + s3rmt3m3 class) can opt in to `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=random_flip` for −1.0 to −1.2 % nnz_L on those fixtures.  Documented in `docs/algorithm.md` Sprint 27 closure subsection.

## Sprint 27 Headline: 0.85× Target — UNMET (5th Consecutive Sprint)

Sprint 23 (1.063× → 0.952×): missed by 10.2pp.
Sprint 24 (best opt-in 0.942×): missed by 9.2pp.
Sprint 25 (best opt-in 0.922×): missed by 7.2pp.
Sprint 26 (best opt-in 0.9217× unchanged): missed by 7.2pp.
**Sprint 27 (default 0.923×, all 3 algorithmic-axis attempts missed): missed by 7.3pp.**

Sprint 27's structural-pipeline-level interventions all produced regressions on Pres_Poisson:
- Item 4 (annealing FM): +2.2 to +3.1 % regress.
- Item 5 (root-spectral): +2.3 % regress.
- Item 4+5 combined: +2.4 % regress.
- Item 6 (thick-restart): +4.7 to +11.5 % regress.

The empirical conclusion across Sprints 23-27 is consistent: **Pres_Poisson under the multilevel pipeline + leaf-AMD reaches near-optimal cuts that none of the FM-cascade or spectral-bisection-style interventions can improve.**  Sprint 28+ pivots to non-pipeline-level interventions per the PROJECT_PLAN.md follow-up routing (e.g. domain-decomposition recursion at the geometric level, METIS-style coarsening with multiple matchings per level, or a fundamentally different ordering algorithm).

## Sprint 27 Items 4 + 5 + 6 Summary Table

| Item | Mechanism | Pres_Poisson best | Verdict |
|---|---|---:|---|
| 4 (annealing) | per-pop accept-with-prob `exp(g/T)` | 0.943× | Advisory: bcsstk14-class wins (−0.7 %) |
| 5 (root-spectral) | full-graph Lanczos + Fiedler median | 0.944× | Advisory: bcsstk04-class wins (−1.3 % + 23× wall speedup) |
| 6 (thick-restart) | global-best + perturbation per pass | 0.989× | Advisory: bcsstk04 / s3rmt3m3 wins (−1.0 to −1.2 %) |

All three ship as advisory; default unchanged.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_27/thick_restart_sweep.txt` — canonical 3-perturbation × 6-fixture corpus sweep (promoted from Day 11's `interim_day11.txt`)
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_decision.md` — this document

## Files NOT Modified

- `src/sparse_graph.c` — `parse_finest_fm_strategy()` default stays `FINEST_FM_BASELINE` (no flip)
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound stays `≤ 0.96×` (Day-3 default 0.923× well within bound; Day 13's tightening pass takes the post-items-4-6 default ratio + 2pp)
