# Multi-Strategy FM Ensemble — Flip-or-Stay Decision (Sprint 28 Day 5)

## Background

Sprint 28 Day 4 implemented `SPARSE_FM_FINEST_STRATEGY=ensemble`:
runs K sub-strategies (default `baseline,fifo,annealing`) per
finest-level FM pass on cloned partition state, scores each by edge
cut, picks the lowest-cut winner.  Day-5 task 1 sweeps the full
Sprint 27 corpus under 4 selector-list variants + strategy-win-counts
under the default; task 2 (this doc) applies the flip-rule and
records the verdict.

## Day-5 Flip Rule

PLAN.md Day 5 task 2:
- **(a)** Pres_Poisson improves ≥ 1pp vs Sprint 27 default 0.923× of AMD nnz_L
- **(b)** No fixture regresses past 5pp
- **(c)** Wall stays under 2× of Sprint 27 default (≈ 20 s Pres_Poisson)

## Sprint 28 Day 5 Sweep Table

Captured in `ensemble_fm_sweep.txt`; ND nnz_L + wall(ms) per setting
per fixture.  Default coarsening: HCC + Kuu-safe (Sprint 27 Day 2);
`nd_base_threshold = 128` (Sprint 27 Day 3).

| Fixture | n | AMD | baseline (S1) | ensemble b,f,a (S2) | ensemble b,a (S3) | ensemble f,a (S4) | ensemble b,f,a,tr (S5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 637 | 637 | 637 | 637 |
| bcsstk04 | 132 | 3 143 | 3 722 | 3 722 | 3 722 | 3 722 | 3 722 |
| Kuu | 7 102 | 406 264 | 764 664 | 791 858 | 765 132 | 807 487 | 791 858 |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 130 462 | 130 422 | 131 240 | 130 462 |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | **483 750** | 487 832 | 487 949 | **483 750** |
| **Pres_Poisson** | **14 822** | **2 668 793** | **2 462 201** | 2 501 627 | 2 514 282 | 2 510 458 | 2 501 627 |

(S5 includes `thick_restart` in the selector list; thick_restart is
silently skipped by the Day-4 parser — see ensemble_fm_design.md
"thick_restart joins iff explicitly requested" + Day-4 implementation
note that thick_restart's multi-pass anchor semantics don't compose
with single-pass-per-strategy ensemble.  S5 therefore degenerates to
S2 (`baseline,fifo,annealing`) — confirmed by S5 ≡ S2 on every
fixture's nnz_L.)

### Per-fixture nnz_L deltas vs Sprint 27 default (%)

| Fixture | S2 (b,f,a) | S3 (b,a) | S4 (f,a) | S5 (b,f,a,tr → b,f,a) |
|---|---:|---:|---:|---:|
| nos4 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| bcsstk04 | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Kuu | **+3.6 %** | 0.06 % | **+5.6 %** | **+3.6 %** |
| bcsstk14 | +0.03 % | 0.0 % | +0.6 % | +0.03 % |
| s3rmt3m3 | **−0.8 % win** | 0.0 % | +0.02 % | **−0.8 % win** |
| **Pres_Poisson** | **+1.6 %** | **+2.1 %** | **+2.0 %** | **+1.6 %** |

### Per-fixture wall (ms)

| Fixture | baseline (S1) | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|
| nos4 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| bcsstk04 | 109.0 | 98.7 | 117.0 | 94.6 | 87.6 |
| Kuu | 4 171.5 | 3 207.0 | 4 302.5 | 3 382.0 | 3 476.3 |
| bcsstk14 | 292.9 | 737.3 | 332.1 | 718.1 | 720.9 |
| s3rmt3m3 | 3 091.9 | 2 295.9 | 3 393.6 | 2 296.5 | 2 136.8 |
| Pres_Poisson | 4 646.3 | 5 387.4 | 5 425.7 | 4 850.4 | 5 275.0 |

`make wall-check` ceiling: Pres_Poisson ND ≤ 70.5 s (1.5× of 47 s
baseline).  All ensemble settings well under ceiling — 5.4 s vs 70.5 s.

### Strategy-win counts under default selector (S2 = b,f,a)

| Fixture | baseline_wins | fifo_wins | annealing_wins | total |
|---|---:|---:|---:|---:|
| nos4 | 0 | 0 | 0 | 0 (ensemble did not fire on n=100) |
| bcsstk04 | 3 | 0 | 0 | 3 |
| Kuu | 162 (93.1 %) | 12 (6.9 %) | 0 (0.0 %) | 174 |
| bcsstk14 | 72 (93.5 %) | 5 (6.5 %) | 0 (0.0 %) | 77 |
| s3rmt3m3 | 135 (95.7 %) | 6 (4.3 %) | 0 (0.0 %) | 141 |
| Pres_Poisson | 333 (91.2 %) | 31 (8.5 %) | 1 (0.3 %) | 365 |

**Headline win-count finding: annealing essentially never wins
(1 / 365 on Pres_Poisson; 0 / N on every other fixture).**  baseline
dominates every fixture (91-96 %); fifo wins a small fraction (4-9 %)
on irregular SPDs.

## Flip-Rule Application

| Selector | Pres_Poisson Δ | (a) ≥ 1pp ↓ | (b) max regress | (c) wall ≤ 2× | Verdict |
|---|---:|---|---:|---|---|
| S2 (b,f,a, default) | **+1.6 %** | **FAIL** | Kuu +3.6 % | PASS (5.4 s) | NO FLIP |
| S3 (b,a) | **+2.1 %** | **FAIL** | Pres_Poisson +2.1 % | PASS (5.4 s) | NO FLIP |
| S4 (f,a) | **+2.0 %** | **FAIL** | Kuu +5.6 % | PASS (4.9 s) | NO FLIP |
| S5 (b,f,a,tr → b,f,a) | **+1.6 %** | **FAIL** | Kuu +3.6 % | PASS (5.3 s) | NO FLIP |

**Every selector variant fails flip-rule gate (a) on Pres_Poisson.**
Gate (a) requires ≥ 1pp improvement; the ensemble REGRESSES
Pres_Poisson by 1.6-2.1pp under every selector variant.  Gate (b) is
also breached on S4 (Kuu +5.6 % > 5pp ceiling).

## Decision: STAY At Default; Ship Ensemble As Advisory

**No default flip.**  `SPARSE_FM_FINEST_STRATEGY` stays at `baseline`
(Sprint 27 default).  `SPARSE_FM_ENSEMBLE_STRATEGIES` stays at the
parser default `baseline,fifo,annealing` when ensemble is opt-in
(matches PLAN.md Day-5 expectation).

Ship ensemble as **advisory** for s3rmt3m3-class workloads:

| Fixture class | Recipe | Win |
|---|---|---|
| **Mid-irregular SPD (s3rmt3m3)** | `SPARSE_FM_FINEST_STRATEGY=ensemble` (default selector) | **−0.8 % nnz_L** + −26 % wall vs Sprint 27 default |

The s3rmt3m3 win is the only positive nnz_L delta in the matrix.
Notably, ensemble's wall on s3rmt3m3 (2 296 ms) is FASTER than
Sprint 27 default (3 092 ms) — the per-strategy partitions converge
quickly + the ensemble's FM walk pattern (running baseline + fifo +
annealing on a clone instead of the same FM 3 passes sequentially)
happens to produce a better-quality starting state for subsequent
ND recursion.

## Why The Ensemble Fails To Move Pres_Poisson

Two reasons emerge from the strategy-win-counts + nnz_L deltas:

1. **Annealing essentially never wins.**  In 365 total partition
   calls on Pres_Poisson under the default selector, annealing won
   only 1 pass (0.3 %).  On every other fixture, annealing won 0
   passes.  The annealing strategy's stochastic acceptance disrupts
   FM's local-greedy descent without producing a competitively low
   cut.  Including annealing in the selector pays 1/K of the FM
   cost for ~0 benefit.

2. **Edge-cut pick-best ≠ nnz_L pick-best on Pres_Poisson.**  When
   FIFO wins by edge cut (8.5 % of passes), the resulting partition
   has a smaller edge cut but produces a DIFFERENT vertex separator
   via Sprint 22's smaller-side lift.  The different separator
   doesn't necessarily preserve the AMD-friendly elimination
   structure inside each side, so the final ND nnz_L can be HIGHER
   despite the lower edge cut.

This is consistent with Sprint 27 retrospective lesson #1:
**Empirical evidence > algorithmic intuition.**  The ensemble's
mechanism (explore K landscapes + pick best) is sound at the edge-
cut level but doesn't translate to pick-best-by-nnz_L on the headline
fixture.  Across Sprints 22-28, **fourth** Sprint-28 confirmation
that FM-cascade-and-FM-bucket-tweaks cannot improve Pres_Poisson
under the multilevel pipeline + leaf-AMD.

## Sprint 28 Headline Status

The literal 0.85× Pres_Poisson target stays at Sprint 27 default
0.923× — Sprint 28 Day 5 verdict matches the Day-1 pivot decision's
framing.  Sprint 28 Item 5 (Days 12-13) will formally retire the
literal target via `headline_summary.md` empirical-floor
calibration.  Day 5 (this doc) is the second of the 4 Sprint-28
data points confirming the retirement:

1. **Sprint 28 Day 1** — `pivot_decision_day1.md` retirement decision
   (a priori from 5-sprint evidence)
2. Sprint 28 Day 3 — `gain_noise_decision.md` formal gain-noise
   variant regresses Pres_Poisson +26.3pp
3. **Sprint 28 Day 5** (this doc) — ensemble regresses Pres_Poisson
   +1.6-2.1pp across all selector variants
4. Sprint 28 Day 10 — `non_pipeline_decision.md` (forthcoming) —
   supernodal-etree reordering verdict

## Per-Fixture-Class Advisory Recipes (Day 13 will validate)

Day 13's `headline_summary.md` extends the advisory list with this
Sprint 28 Day 5 entry:

| Fixture class | Recipe | Win | Sprint |
|---|---|---|---|
| Bimodal-degree (Kuu) | `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k SPARSE_ND_SEP_LIFT_WEIGHT=hybrid --nd-threshold 256` | −35.3 % | 27 |
| Tiny irregular (bcsstk04) | `SPARSE_ND_ROOT_BISECT=spectral` | −1.3 % + 23× wall | 27 |
| Small irregular (bcsstk04 alt) | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` | −1.7 % | 28 |
| Mid-irregular (bcsstk14) | `SPARSE_FM_FINEST_STRATEGY=annealing` | −0.7 % | 27 |
| **Mid-irregular (s3rmt3m3 alt)** | `SPARSE_FM_FINEST_STRATEGY=ensemble` (default selector) | **−0.8 % nnz_L + −26 % wall** | **28** |
| Mid-irregular (s3rmt3m3) | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=random_flip` | −1.0 % | 27 |

`docs/algorithm.md` ND subsection updates with the new advisory at
Sprint 28 Day 14 (Item 7).

## Files Generated

- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_sweep.txt` — canonical
  5-setting × 6-fixture corpus capture
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_sweep_debug.txt` —
  strategy-win counts under default selector
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_decision.md` — this
  document

## Files NOT Modified

- `src/sparse_graph.c` — `parse_fm_finest_strategy()` default stays
  at `baseline` (no flip)
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` —
  bound stays `≤ 0.94×` (Sprint 27 Day 13 tightening); literal 0.85×
  retirement is staged for Sprint 28 Item 5 Day 13

## References

- `docs/planning/EPIC_2/SPRINT_28/PLAN.md` Day 5
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md` Sprint 28
  Day 3 architecture + env-var design
- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_interim_day4.txt`
  Sprint 28 Day 4 implementation interim measurements
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md` Sprint 28
  Day 3 — the precedent flip-or-stay decision pattern
- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` Sprint 28
  Day 1 retirement decision
- `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred"
  + lesson #1 ("Empirical evidence > algorithmic intuition")
- `docs/planning/EPIC_2/SPRINT_27/headline_summary.md` Sprint 27
  24-setting × 6-fixture verdict
