# Formal Gain-Noise Thick-Restart FM — Flip-or-Stay Decision (Sprint 28 Day 3)

## Background

Sprint 27 Day 11 simplified the `gauss_noise` thick-restart variant
to "random-flip with k drawn proportional to a half-Gaussian" under
implementation-time pressure (`thick_restart_design.md` Day-10
deviation note).  The formal Day-10 design called for adding a
Gaussian noise term to the gain-bucket pick step in `graph_refine_fm`
— `noisy_gain = gain + sigma_k * |max_gain| * randn()` — rather than
perturbing the post-pass partition state.

Sprint 28 Day 2 implemented the formal variant behind
`SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` with a new
`SPARSE_FM_GAIN_NOISE_SCHEDULE={linear (default), exponential, cosine}`
sub-axis.  Day 3 (this doc) runs the full corpus sweep + records the
flip-or-stay verdict.

## Sprint 28 PLAN.md Day 3 task 2 flip rule

> flip-or-stay decision (advisory unless formal variant beats Sprint
> 27 default on Pres_Poisson AND has cleaner flip-rule application
> than Sprint 27's simplified `gauss_noise`)

Two gates:
- **(a)** Pres_Poisson improves ≥ 1pp vs Sprint 27 default 0.923× of
  AMD nnz_L
- **(b)** No fixture regresses past 5pp from Sprint 27 default

## Sprint 28 Day 3 Sweep Table

Captured in `gain_noise_formal_sweep.txt`; ND nnz_L + wall (ms) per
mode per fixture.  Default coarsening: HCC + Kuu-safe (Sprint 27
Day 2); `nd_base_threshold = 128` (Sprint 27 Day 3).

| Fixture | n | AMD | baseline | gnf-linear | gnf-exponential | gauss_noise (Sprint 27 simplified) |
|---|---:|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 637 | 637 | 637 |
| bcsstk04 | 132 | 3 143 | 3 722 | **3 660** | **3 678** | **3 677** |
| Kuu | 7 102 | 406 264 | 764 664 | 1 279 823 | 1 245 143 | 772 566 |
| bcsstk14 | 1 806 | 116 071 | 130 422 | 146 045 | 148 268 | 132 599 |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 526 211 | 521 918 | 488 049 |
| **Pres_Poisson** | **14 822** | **2 668 793** | **2 462 201** | 3 110 704 | 3 000 680 | 2 639 539 |

### Per-fixture nnz_L deltas vs Sprint 27 default (%)

| Fixture | gnf-linear | gnf-exponential | gauss_noise (reference) |
|---|---:|---:|---:|
| nos4 | 0.0 % | 0.0 % | 0.0 % |
| bcsstk04 | **−1.7 % win** | **−1.2 % win** | **−1.2 % win** |
| Kuu | **+67.4 %** | **+62.9 %** | +1.0 % |
| bcsstk14 | **+12.0 %** | **+13.7 %** | +1.7 % |
| s3rmt3m3 | **+7.9 %** | **+7.0 %** | +0.04 % |
| **Pres_Poisson** | **+26.3 %** | **+21.9 %** | **+7.2 %** |

### Per-fixture ND/AMD ratios

| Fixture | baseline | gnf-linear | gnf-exponential | gauss_noise |
|---|---:|---:|---:|---:|
| Pres_Poisson | 0.923× | 1.166× | 1.124× | 0.989× |
| Kuu | 1.882× | 3.151× | 3.065× | 1.902× |
| bcsstk14 | 1.124× | 1.258× | 1.277× | 1.142× |
| bcsstk04 | 1.184× | 1.164× | 1.170× | 1.170× |
| s3rmt3m3 | 1.028× | 1.109× | 1.100× | 1.029× |

### Wall (ms) per mode

| Fixture | baseline | gnf-linear | gnf-exponential | gauss_noise |
|---|---:|---:|---:|---:|
| nos4 | 0.7 | 0.4 | 0.4 | 0.5 |
| bcsstk04 | 180.2 | 177.5 | 175.6 | 172.4 |
| Kuu | 7 969.8 | 4 893.4 | **2 385.1** | 6 611.0 |
| bcsstk14 | 718.2 | 1 260.7 | 999.2 | 467.6 |
| s3rmt3m3 | 6 572.4 | 4 097.6 | 3 626.0 | 4 874.2 |
| Pres_Poisson | 9 717.9 | 11 989.1 | 11 111.6 | 10 796.6 |

`make wall-check` Pres_Poisson ND ceiling (1.5× of 47 s baseline =
70.5 s): all modes pass; gnf modes add 14-23% wall on Pres_Poisson but
stay well under ceiling.

## Flip-Rule Application

| Mode | Pres_Poisson ratio | (a) gate | (b) gate |
|---|---|---|---|
| gnf-linear (default) | 1.166× ✗ (+24.3pp worse than default) | FAIL | n/a (a failed) |
| gnf-exponential | 1.124× ✗ (+20.1pp worse) | FAIL | n/a (a failed) |

**Neither schedule lands Pres_Poisson within 1pp of Sprint 27 default.**
Both modes regress decisively: gnf-linear by +24.3pp, gnf-exponential
by +20.1pp.  Gate (a) fails for both; gate (b) is moot.

Comparison vs Sprint 27 simplified `gauss_noise` (which regressed
Pres_Poisson +7.2pp): the formal variant is **strictly worse** than
the Day-11 simplification it was supposed to replace.  Sprint 27 Day 11
made a fortunate trade-off — the random-flip-with-half-Gaussian-k
approximation produces less Pres_Poisson disruption than the formal
gain-bucket-comparator perturbation does.

## Decision: STAY At Default; Ship gain_noise_formal As Advisory

**No default flip.**  `SPARSE_FM_FINEST_STRATEGY` stays at `baseline`
(Sprint 27 default).  `SPARSE_FM_THICK_RESTART_PERTURB` stays at
`random_flip` (Sprint 27 default) when thick-restart is opt-in.

Ship `gain_noise_formal` as **advisory** for bcsstk04-class workloads:

| Fixture class | Recipe | Win |
|---|---|---|
| **Small irregular (bcsstk04)** | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal SPARSE_FM_GAIN_NOISE_SCHEDULE=linear` (default) | **−1.7 % nnz_L** |

The bcsstk04 win is the only positive outcome; all 4 other
non-trivial fixtures regress, three of them severely (Kuu, bcsstk14,
Pres_Poisson).

The exponential schedule offers no advantage over linear on the
headline fixture (both regress decisively; exponential by 4pp less
but still ~+20pp from default).  Linear stays default.

## Why The Formal Variant Loses To The Simplified gauss_noise

Sprint 27 simplified `gauss_noise` perturbs **partition state**
between FM passes (random-flip ~n/50 vertices with k drawn from a
half-Gaussian), leaving the gain-bucket walk inside each pass
unperturbed.  The formal gain_noise_formal perturbs the **gain-bucket
comparator** during each pass, which:

1. Disrupts FM's local-greedy pop order from the very first step,
   not just at pass boundaries — Pres_Poisson's regular FE-mesh
   structure has a flat near-optimal cut landscape that benefits
   from FM's greedy descent; pop-order disruption breaks the
   convergence path.
2. Compounds across passes — each pass's noise adds new
   perturbation, so even with sigma_k decay the early passes carry
   damaging noise into the global-best anchor.
3. Doesn't add escape benefit on regular fixtures — Pres_Poisson's
   FM landscape (Sprint 27 200+ measurements) doesn't have a
   "missing" structural cut that bucket-noise would discover.

This is consistent with Sprint 27 retrospective lesson #1:
**Empirical evidence > algorithmic intuition.**  The Day-10 design's
"perturb the comparator, not the state" intent had clean theoretical
motivation but loses empirically to the Day-11 partition-state
simplification.  Three independent attempts (annealing, thick-restart
random-flip / boundary-shuffle / gauss_noise / gain_noise_formal) all
regress Pres_Poisson; the conclusion is increasingly robust:
**FM-cascade-and-FM-bucket-tweaks cannot improve Pres_Poisson under
the multilevel pipeline + leaf-AMD.**

## Sprint 28 Headline Status

The literal 0.85× Pres_Poisson target stays at Sprint 27 default
0.923× — Sprint 28 Day 3 verdict matches the Day-1 pivot decision's
framing: the formal gain-noise variant ships as advisory, the
literal target is retired with empirical-floor calibration documented
in Day 13's `headline_summary.md` (Sprint 28 Item 5).

This is the **third** Sprint-28 confirmation of the 5-sprint
conclusion (`pivot_decision_day1.md` already documented the
retirement decision; Day 2 + Day 3 add data points):

1. Sprint 27 Day-13's 24-setting × 6-fixture matrix established no
   pipeline-level intervention beats default on Pres_Poisson.
2. Sprint 28 Day 1 pivot-decision study found candidates (a) METIS-
   multi-matching, (b) geometric-DD, (c) supernodal-etree all
   either fit the pipeline-level rejected pattern OR can't move
   nnz_L by the 7.3pp required magnitude.
3. **Sprint 28 Day 3 (this doc): formal gain-noise variant regresses
   Pres_Poisson +26.3pp** — the Day-10 design's algorithmic intent
   is implemented faithfully but the empirical outcome confirms
   bucket-noise perturbation does NOT help on regular FE meshes.

## Per-Fixture-Class Advisory Recipes (Day 13 will validate)

The Sprint 27 advisory list (per `headline_summary.md` Sprint 27
Day 13) extends with one new entry from Sprint 28 Day 3:

| Fixture class | Recipe | Win | Sprint |
|---|---|---|---|
| Bimodal-degree (Kuu) | `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k SPARSE_ND_SEP_LIFT_WEIGHT=hybrid --nd-threshold 256` | −35.3 % | 27 |
| Tiny irregular (bcsstk04) | `SPARSE_ND_ROOT_BISECT=spectral` | −1.3 % + 23× wall speedup | 27 |
| **Small irregular (bcsstk04 alt)** | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal` | **−1.7 % nnz_L** | **28** |
| Mid-irregular (bcsstk14) | `SPARSE_FM_FINEST_STRATEGY=annealing` | −0.7 % | 27 |
| Mid-irregular (s3rmt3m3) | `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=random_flip` | −1.0 % | 27 |

`docs/algorithm.md` ND subsection updates with the new advisory at
Sprint 28 Day 14 (Item 7).

## Files Generated

- `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_sweep.txt` —
  canonical 4-mode × 6-fixture corpus capture (promoted from Day-2
  interim)
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md` — this
  document

## Files NOT Modified

- `src/sparse_graph.c` — `parse_fm_thick_restart_perturb()` default
  stays `RANDOM_FLIP` (no flip)
- `parse_fm_gain_noise_schedule()` default stays `LINEAR` (no flip;
  linear ≥ exponential on the corpus already)
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` —
  bound stays `≤ 0.94×` (Sprint 27 Day 13 tightening); literal 0.85×
  retirement happens at Day 13 via `headline_summary.md`

## References

- `docs/planning/EPIC_2/SPRINT_28/PLAN.md` Day 3 task 1 + task 2
- `docs/planning/EPIC_2/SPRINT_28/gain_noise_formal_interim_day2.txt`
  — Day-2 quick-look measurements (now superseded by Day-3 canonical)
- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` — Day-1
  empirical-floor retirement decision
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md` Day-10
  deviation note — the original "formal vs simplified" framing
- `docs/planning/EPIC_2/SPRINT_27/thick_restart_decision.md` Sprint
  27 Day 11 simplified gauss_noise verdict
- `docs/planning/EPIC_2/SPRINT_27/headline_summary.md` Sprint 27
  Day 13 24-setting × 6-fixture verdict
- `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred"
  + lesson #1 ("Empirical evidence > algorithmic intuition")
