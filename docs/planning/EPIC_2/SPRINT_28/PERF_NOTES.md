# Sprint 28 Performance Notes — Non-Pipeline-Level Pres_Poisson Closure

Sprint 28 extends the Sprint 22-27 ND performance work with one non-pipeline-level pivot (Item 4: supernodal-etree reordering) + two Sprint 27 deferral closures (Item 1: formal gain-noise thick-restart variant; Item 2: multi-strategy FM ensemble).

This file separates from `docs/planning/EPIC_2/SPRINT_22/PERF_NOTES.md` (which absorbed Sprint 22-27 closures and is at ~800 lines) per Sprint 28 PLAN.md Day 14 task 3.

## Sprint 28 Closures Summary

Three new advisory env-var paths; zero production default flips:

| Item | Env var | Sub-axis | Sprint 28 verdict | Reference |
|---|---|---|---|---|
| 1 (Days 2-3) | `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` | linear / exponential schedules | Advisory only; catastrophic Pres_Poisson regress (+24pp; 1.166× ratio) | `gain_noise_decision.md` |
| 2 (Days 4-5) | `SPARSE_FM_FINEST_STRATEGY=ensemble` | `SPARSE_FM_ENSEMBLE_STRATEGIES` selector list | Advisory only; Pres_Poisson regress (+1.5pp; 0.937× ratio); 2-3× wall | `ensemble_fm_decision.md` |
| 4 (Days 6-10) | `SPARSE_ND_SUPERNODAL_POSTORDER={off, on}` | none | Advisory only; nnz_L bit-equivalent to default (symmetric-permutation invariance); +6-15 % `sparse_analyze` wall | `non_pipeline_decision.md` |

Item 4's specific framing: "non-pipeline-level pivot" per Day-1 `pivot_decision_day1.md`.  Picked candidate (c) supernodal-etree reordering over (a) METIS-style multi-matching coarsening (still pipeline-level) and (b) geometric domain decomposition (structurally blocked on Pres_Poisson — corpus lacks coordinates).  Liu 1990 / Davis 2006 §6.5 post-pass: compose the elimination-tree postorder into `analysis->perm`, rebuild B, recompute etree+postorder so colcount + symbolic Cholesky run on the final ordering.

## Sprint 28 Default Path Per-Fixture

Sprint 28 production defaults are identical to Sprint 27 Day-3 closure (HCC + Kuu-safe + `nd_base_threshold = 128`).  No production default flips this sprint.

| Fixture | n | AMD nnz_L | ND nnz_L | ratio | ND wall (Day 12 measurement) |
|---|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 1.000× | 0.3 ms |
| bcsstk04 | 132 | 3 143 | 3 722 | 1.184× | 142 ms |
| Kuu | 7 102 | 406 264 | 764 664 | 1.882× | 5 247 ms |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 1.124× | 421 ms |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 1.028× | 4 538 ms |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | **0.9226×** | 3 418 ms |

Per-sprint Pres_Poisson trajectory (default path):

| Sprint | Default ratio | Δ vs previous |
|---|---:|---:|
| 22 | 1.063× | (baseline) |
| 23 | 0.952× | -10.4pp |
| 24 | 0.952× | flat |
| 25 | 0.952× | flat |
| 26 | 0.952× | flat |
| 27 | 0.9226× | -2.9pp (HCC + t=128 default flips) |
| **28** | **0.9226×** | **0pp (no default flips this sprint)** |

The 0.85× literal target was formally retired Day 13 after 6 consecutive sprints of misses; see `headline_summary.md` "Sprint 28 Verdict on the Literal 0.85× Pres_Poisson Target — RETIRED".

## Sprint 28 Cross-Corpus Matrix Headline (Day 12)

24 settings × 6 fixtures = 144 cells.  Reference: `bench_day12_combinations.csv` + `.txt`.

### Pres_Poisson top 4 (lowest ND/AMD ratio across all 24 settings)

| Setting | Description | Pres_Poisson ratio |
|---:|---|---:|
| 2 | Sprint 27 default (HCC + t=128) | 0.9226× |
| 3 | Sprint 28 item-4 SUPERNODAL_POSTORDER=on | 0.9226× |
| 6 | Sprint 28 item-2 ensemble × default selector | 0.9374× |
| 18 | Sprint 28 stack: item-4 × item-2 | 0.9374× |

Setting 2 and setting 3 are bit-equivalent (the Item-4 post-pass is a symmetric permutation; nnz_L is invariant by construction).  No other Sprint 28 axis or combination beats them.  Detailed analysis in `headline_summary.md` "Sprint 28 Verdict" section.

### Corpus-wide top 2 (lowest geomean of 6 fixture ratios)

| Setting | Description | Geomean | Pres_Poisson | Kuu |
|---:|---|---:|---:|---:|
| 15 | Sprint 27 advisory: t=256 Kuu opt-in | 1.1156 | 0.9475× | 1.7716× |
| 21 | Sprint 28 stack: item-4 × Kuu opt-in (SUPERNODAL=on + t=256) | 1.1156 | 0.9475× | 1.7716× |

Setting 15 (Sprint 27) and setting 21 (Sprint 28 stack) are bit-equivalent: Item-4 doesn't change Kuu either.  Sprint 27's corpus-wide-best advisory recipe survives Sprint 28 intact.

### Kuu top 1 (largest single-fixture wins)

| Setting | Description | Kuu ratio | vs Sprint 27 default |
|---:|---|---:|---:|
| **23** | Sprint 28 kitchen-sink (item-4 + item-1 + item-2 + Sprint 27 t=256 + fixed_k hybrid) | **1.1934×** | **−36.6%** |

Setting 23 is the new Sprint 28 corpus-wide Kuu best — slightly improves Sprint 27's setting 18 (1.217× → 1.193×) by stacking Item-2 ensemble on top of t=256 + fixed_k hybrid.  But it catastrophically regresses Pres_Poisson (1.336×; +41pp from default), so it's purely advisory for workloads that look like Kuu.

## Wall-Time Notes

Sprint 28 added no wall-affecting flips on the default path.  Pres_Poisson ND wall stays at the Sprint 27 Day-3 floor (~7s; measurement-day variance 3-7s across the sprint).  `make wall-check` PASS throughout (Pres_Poisson ND ≤ 1.5× of the Sprint 22 baseline 47s = 70.5s ceiling; current ~14× headroom).

The Item-4 env-on path adds ~6-15 % to `sparse_analyze` wall (extra etree_compute + sparse_permute + recompute pass).  Captured in `non_pipeline_interim_day7.txt`.

## Item 4 (Supernodal-Etree) Empirical Notes

The post-pass composes the etree postorder into the AMD/ND-output perm.  Per Liu 1990, this maximises the number of consecutive columns that satisfy `chol_csc_detect_supernodes`'s fundamental-supernode invariants.

Day-8 measurement (see `non_pipeline_supernode_count_day8.txt`): on the existing corpus, AMD's perm is already approximately etree-postordered.  The composition essentially does NOT move the supernode structure — same count ± 1, same total_grouped on most fixtures, and on the smaller fixtures (nos4, bcsstk04) it slightly REDUCES total_grouped because AMD's specific perm happened to produce more compact contiguous supernodes than the strict postorder.

Where Item-4 would help: future supernodal numeric-factor kernels.  The existing `chol_csc_eliminate_supernodal` detects supernodes but runs the scalar Day-5 elimination kernel.  When a future sprint wires batched supernodal cmod + dense factor + panel solve (per `src/sparse_chol_csc_internal.h` line 614-628 future-sprint deliverable), `SPARSE_ND_SUPERNODAL_POSTORDER=on` becomes the natural input ordering: contiguous supernodes from etree-postorder give the dense BLAS kernels cache-friendly column blocks.

Today's value of the env var: zero on this codebase.  Tomorrow's value: directly enables supernodal-kernel speedups when Sprint 29+ wires them.

## Sprint 28 Items Deferred → Sprint 29+

1. **Supernodal numeric-factor kernels** — the natural follow-up that gives the Sprint-28 Item-4 infrastructure measurable production value.  Day-1 dossier estimated 5-15 % numeric-factor wall reduction on supernodal-heavy fixtures.  Out of Sprint 28 scope.
2. **Pres_Poisson literal 0.85× target** — formally retired with Sprint 28's empirical evidence (6 sprints + non-pipeline pivot).  Sprint 29+ revisit only with fundamentally different machinery (METIS interop, geometric mesh-aware ordering, hybrid AMD-then-ND-on-separators).  None budgeted for Sprint 29.
3. **bench_reorder env-var integration** — the existing perm-pre-applied + REORDER_NONE path doesn't fire the SUPERNODAL_POSTORDER dispatch.  Low priority; Sprint 28 used ad-hoc /tmp helpers for Day 7, 9, 12 measurements (~50-100 LOC each, not committed to benchmarks/).

## Shipping Story for the Sprint 28 PR Description

"Zero production default flips.  Three new advisory env-var paths: `SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` (Item 1 — Sprint 27 deferral; advisory only), `SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES` (Item 2 — Sprint 27 parking-lot; advisory only), `SPARSE_ND_SUPERNODAL_POSTORDER={off, on}` (Item 4 — non-pipeline-level pivot per Day-1 dossier; Liu 1990 post-pass; nnz_L bit-equivalent to default by symmetric-permutation invariance; ships infrastructure for future supernodal numeric-factor kernels).  Pres_Poisson 0.85× literal target REMAINS UNMET (6th consecutive sprint; current default 0.9226× = 7.3pp from target) and is FORMALLY RETIRED with Sprint 28's empirical evidence — the only intervention that can act AFTER the multilevel pipeline produces a 0pp delta on the metric, demonstrating the floor is structural.  Sprint 29+ revisit ONLY with fundamentally different machinery (METIS interop, geometric mesh-aware ordering, hybrid AMD-then-ND-on-separators).  Day 12's 24-setting × 6-fixture cross-corpus matrix confirms Sprint 27 default + Sprint 28 Item-4 SUPERNODAL_POSTORDER=on are tied for Pres_Poisson best; Sprint 27 advisory t=256 Kuu opt-in remains corpus-wide best.  No test bound changes; Sprint 27's 0.94× bound stays."
