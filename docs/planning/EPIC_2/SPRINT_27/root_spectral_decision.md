# Root-Level Spectral Bisection — Flip-or-Stay Decision (Sprint 27 Day 9)

## Background

Sprint 27 Day 7 designed root-level spectral bisection (`SPARSE_ND_ROOT_BISECT={multilevel (default), spectral}`); Day 8 implemented it and captured a corpus Lanczos timing pass.  Sprint 27 PLAN.md Day 9 task 2 specifies the flip rule:

> Flip `SPARSE_ND_ROOT_BISECT` default to `spectral` if (a) Pres_Poisson lands ≤ 0.85× of AMD nnz_L AND (b) no smaller-fixture regress past 5pp AND (c) wall stays under per-key threshold.

This document captures Day 9's flip-rule application + the spectral × annealing combination test (Day 9 task 3) + the headline status update.

## Reproducer

```
SPARSE_ND_ROOT_BISECT=spectral build/bench_reorder --only <fixture> --skip-factor

# Combination:
SPARSE_ND_ROOT_BISECT=spectral SPARSE_FM_FINEST_STRATEGY=annealing build/bench_reorder ...
```

Default coarsening: HCC + Kuu-safe (Sprint 27 Day 2); `nd_base_threshold = 128` (Sprint 27 Day 3).

## Sprint 27 Day 9 Sweep — Spectral Alone (from Day 8's `root_spectral_lanczos_timing.txt`)

| Fixture | n | AMD | multilevel default | root-spectral | Δ nnz % | multilevel ms | spectral ms | Δ wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 637 | 0.0 % | 0.2 | 0.2 | 0 |
| **bcsstk04** | 132 | 3 143 | 3 722 | **3 672** | **−1.3 % WIN** | 105.4 | **4.5** | **−100.9** |
| Kuu | 7 102 | 406 264 | 764 664 | 892 672 | **+16.7 %** | 4 596.3 | 4 667.7 | +71.4 |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 138 693 | +6.3 % | 358.5 | 1 018.9 | +660.4 |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 506 077 | +3.7 % | 3 783.9 | 3 241.5 | −542.4 |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | **2 518 428** | **+2.3 %** | 5 958.3 | 8 856.9 | **+2 898.6** |

### Per-fixture nnz_L deltas vs Sprint 27 Day 3 default

| Fixture | Δ nnz % | ND/AMD ratio under spectral |
|---|---:|---|
| nos4 | 0.0 % | 1.270× |
| **bcsstk04** | **−1.3 %** | **1.168× (was 1.184×)** |
| Kuu | +16.7 % | 2.198× (was 1.882×) |
| bcsstk14 | +6.3 % | 1.195× (was 1.124×) |
| s3rmt3m3 | +3.7 % | 1.066× (was 1.028×) |
| **Pres_Poisson** | **+2.3 %** | **0.944× (was 0.923×)** |

## Combination Test: Spectral + Annealing on Pres_Poisson

Captured in `root_spectral_combo_sweep.txt`.

| Setting | Pres_Poisson nnz_L | ratio | vs default |
|---|---:|---:|---:|
| Sprint 27 Day 6 default (multilevel + baseline FM) | 2 462 201 | 0.923× | (ref) |
| spectral alone | 2 518 428 | 0.944× | +2.3pp |
| annealing alone (Day 7 reference) | 2 538 898 | 0.951× | +3.1pp |
| **spectral + annealing exponential** | **2 526 747** | **0.947×** | **+2.6pp** |
| spectral + annealing linear | 2 526 747 | 0.947× | +2.6pp |
| spectral + annealing cosine | 2 526 747 | 0.947× | +2.6pp |

**The combination does not close to 0.85×.**  All three combination schedules land at the *same* nnz_L (2 526 747) because under spectral root-bisection, the partition tree diverges from baseline; the annealing schedule's per-pass FM jitter at downstream levels gets washed out by the spectral path's larger structural shape change.

## Flip-Rule Application — Spectral Alone

Sprint 27 PLAN.md Day 9 task 2:
- **(a)** Pres_Poisson ≤ 0.85× of AMD nnz_L → 0.944× ✗ FAIL.
- **(b)** No smaller-fixture regress past 5pp → Kuu +16.7 % ✗; bcsstk14 +6.3 % ✗.
- **(c)** Wall under 1.5× ceiling → 8.9 s vs 70.5 s ✓ PASS.

(a) and (b) both fail decisively.  **No flip.**

## Flip-Rule Application — Spectral + Annealing Combination

If the combination had landed Pres_Poisson ≤ 0.85×, Sprint 27 PLAN.md Day 9 task 3 specifies it would ship as a **combined advisory recipe** (default-flip outcomes individual; combined ships in `docs/algorithm.md`).  Empirically combination lands 0.947× (FAIL).  **No advisory recipe.**

## Decision: Stay At Default `multilevel`; Ship Spectral As Advisory

The Day-7 hypothesis ("Fiedler at the root captures geometric structure the multilevel pipeline loses") is empirically wrong on this corpus.  The multilevel pipeline's iterative FM refinement at every uncoarsening level reaches near-optimal cuts that the median-bisect-on-Fiedler doesn't beat.

### Where root-spectral helps

- **bcsstk04 (n=132)**: small irregular fixture; **−1.3 % nnz_L + 23× wall speedup** (105.4 ms → 4.5 ms).  The multilevel pipeline's coarsening overhead dominated wall on this small-n fixture; spectral skips it cleanly.  Per-fixture-class advisory: workloads dominated by small (n ≤ ~200) irregular fixtures benefit from `SPARSE_ND_ROOT_BISECT=spectral`.

### Where root-spectral hurts

- **Kuu (n=7102, CV=0.425)**: bimodal-degree distribution doesn't map to a Fiedler-friendly structure; +16.7 % nnz_L worst case in the corpus.
- **bcsstk14 / s3rmt3m3**: mid-size irregular SPDs; +3-6 % regress.
- **Pres_Poisson**: the headline FE-mesh fixture has clean Fiedler structure, but the multilevel pipeline's FM cascade reaches a tighter cut than median-bisect-on-Fiedler.  +2.3 % regress (0.923× → 0.944×) — actively pushes AWAY from the 0.85× target.

## Headline Status After Day 9

- **Pres_Poisson default unchanged: 0.923× of AMD.**
- 7.3pp gap remains to the literal 0.85× target.
- **Item 4 (annealing FM): missed** (Day 7 advisory; +3.1pp regress).
- **Item 5 (root-spectral): missed** (Day 9 advisory; +2.3pp regress).
- **Item 4 + 5 combined: missed** (+2.6pp regress).
- Sprint 27's structural-pipeline-level interventions for Pres_Poisson all fall short.
- **Item 6 (thick-restart FM, Days 10-12) is now the LAST 0.85× candidate.**  If item 6 also misses, Sprint 27 closes with the literal target unmet — **fifth consecutive sprint** (Sprints 23, 24, 25, 26, 27).
- Sprint 28+ routing: pivot away from FM-cascade-and-spectral-bisection-style interventions; explore non-pipeline-level alternatives per the PROJECT_PLAN.md follow-up.

## Per-Fixture-Class Advisory (Spectral)

| Fixture class | Spectral verdict | Default behaviour |
|---|---|---|
| Tiny irregular (n ≤ ~200; bcsstk04) | **−1.3 % win + 23× wall speedup** | Stay default; **advisory opt-in** |
| Small mesh (nos4) | Bit-stable | Stay default |
| Mid-size irregular (bcsstk14, s3rmt3m3) | Regress 3-6 % | Stay default |
| Bimodal-degree (Kuu) | Regress 16.7 % | Stay default |
| Regular FE-mesh (Pres_Poisson) | Regress 2.3 % | Stay default |

Workloads dominated by small irregular SPDs (bcsstk04's class) get a meaningful wall speedup + slight fill win via `SPARSE_ND_ROOT_BISECT=spectral`.  Documented in `docs/algorithm.md` Sprint 27 closure subsection.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_27/root_spectral_lanczos_timing.txt` — corpus timing pass (Day 8; reused by Day 9)
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_combo_sweep.txt` — spectral × annealing combination sweep on Pres_Poisson
- `docs/planning/EPIC_2/SPRINT_27/root_spectral_decision.md` — this document

## Files NOT Modified

- `src/sparse_reorder_nd.c` — `SPARSE_ND_ROOT_BISECT` default stays `multilevel` (no flip)
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound stays `≤ 0.96×` (Day-3 default 0.923× well within bound; tightening waits for items 6 + Day 13)
