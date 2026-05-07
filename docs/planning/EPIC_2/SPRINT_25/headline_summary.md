# Sprint 25 Day 9 — Headline Summary

## Verdict

| Headline gate | result |
|---|---|
| Pres_Poisson ND/AMD ≤ 0.85× (literal target) | **MISS** — best combination achieves 0.9218× |
| Pres_Poisson ND/AMD ≤ 0.90× (partial close) | **MISS** — best combination achieves 0.9218× |
| Pres_Poisson ND/AMD < Sprint 24 baseline (0.952×) | **PASS** — best 0.9218× = -3pp tightening |
| Smaller-fixture corpus safety (no > 5pp regression) | **PASS** — worst regression is s3rmt3m3 +1.0pp |

Per PLAN.md Day 9 task 4, **the remaining gap routes to Sprint 26**.
This is the third sprint in a row to miss the Pres_Poisson literal
target (Sprint 22 PLAN's 0.5× → Sprint 23 PLAN's 0.7× → Sprint 24
PLAN's 0.85× → Sprint 25 PLAN's 0.85×, all unmet).

## Best combination

**Setting 13: `SPARSE_ND_COARSENING=hcc` + `SPARSE_ND_COARSEN_FLOOR_RATIO=200`**

Pres_Poisson ND/AMD = **0.9218×** of AMD nnz_L.

| stage | Pres_Poisson ND/AMD | Δ |
|---|---|---|
| Sprint 22 baseline | 1.063× | — |
| Sprint 23 baseline | 0.952× | -11.1pp from Sprint 22 |
| Sprint 24 baseline | 0.952× | unchanged from Sprint 23 |
| Sprint 24 best opt-in (ratio=200 alone) | 0.942× | -1.0pp from S24 baseline |
| Sprint 25 HCC alone (Day 3) | 0.937× | -1.5pp |
| Sprint 25 spectral alone (Day 8) | 0.953× | unchanged |
| Sprint 25 multi-pass FM intermediate (Day 5; default passes=1) | 0.952× | unchanged |
| **Sprint 25 best (HCC + ratio=200)** | **0.922×** | **-3.0pp from S24 baseline** |
| Sprint 25 stretch target (literal) | 0.85× | -10.2pp gap remaining |
| Sprint 25 partial-close target | 0.90× | -2.2pp gap remaining |

Constructive composition: HCC + ratio=200 beats both individual
axes by 1-2pp each.  Other combinations show destructive
interactions — the cut-quality knobs don't compose linearly.

## Per-fixture deltas (setting 13 vs Sprint 24 default)

| fixture | default ND/AMD | setting 13 ND/AMD | Δ | category |
|---|---|---|---|---|
| nos4 | 1.520× | 1.414× | -10.6pp | win |
| bcsstk04 | 1.178× | 1.180× | +0.2pp | noise |
| Kuu | 2.275× | 2.099× | -17.6pp | win |
| bcsstk14 | 1.129× | 1.123× | -0.6pp | small win |
| s3rmt3m3 | 1.009× | 1.019× | +1.0pp | small noise (within 5pp band) |
| **Pres_Poisson** | **0.952×** | **0.922×** | **-3.0pp** | **headline win** |

Setting 13 wins on 4 of 6 fixtures (nos4, Kuu, bcsstk14, Pres_Poisson)
+ noise on bcsstk04 + small s3rmt3m3 noise.  No fixture regresses
past the 5pp band.  **Setting 13 is the Day 10 production-default-
flip candidate.**

## Alternative best: corpus-wide setting

**Setting 15: full_set (HCC + INTERMEDIATE=3 + spectral + ratio=200 + balanced_boundary)**

Pres_Poisson ND/AMD = 0.9470× (-0.5pp; not the headline winner)
but produces dramatic wins on the smaller fixtures:

| fixture | default | setting 15 | Δ |
|---|---|---|---|
| nos4 | 1.520× | 1.256× | -26.4pp |
| Kuu | 2.275× | 1.309× | **-97pp (huge win)** |
| bcsstk14 | 1.129× | 1.037× | -9.2pp |

Setting 15 is the **best non-Pres_Poisson** combination and the
candidate to consider as a parallel default-flip if Day 10 prefers
corpus-aggregate fill quality over the Pres_Poisson headline.

ND wall_time on Pres_Poisson under setting 15: ~1 600 ms (vs
default 37 365 ms — **23× speedup** because spectral cuts close to
the FM optimum + intermediate-FM-3 polishes faster).

## Trade-off matrix for Day 10

Day 10's production-default decision needs to balance three axes:

1. **Pres_Poisson nnz_L** (the headline gate): setting 13 wins
   (0.922×); setting 15 mediocre (0.947×).
2. **Corpus-wide nnz_L** (broader fill quality): setting 15
   wins (Kuu -97pp, nos4 -26pp); setting 13 medium (Kuu -18pp,
   nos4 -11pp).
3. **Pres_Poisson wall_time**: setting 15 wins (23× speedup
   via spectral); setting 13 unchanged (HCC + ratio=200 don't
   touch wall).

Recommended Day 10 routing (preliminary; final on Day 10):

- **Default flip to setting 13** (HCC + ratio=200) for the
  Pres_Poisson headline win + clean corpus profile.
- **Document setting 15** as advisory for callers who want the
  Kuu/nos4/bcsstk14 wins + Pres_Poisson wall speedup at the
  cost of marginal Pres_Poisson nnz_L.
- **Document HCC alone** as advisory for callers who want a
  cleaner partial improvement without the divisor-200 Kuu wall
  cost (Sprint 24 Day 5 measured Kuu wall +50 % under
  ratio=200; setting 13 inherits this).

## Sprint 26 routing for the residual gap

The 0.85× literal target requires algorithmic work outside
Sprint 25's scope.  Per `nd_tuning_day8.md`, Sprint 26 candidates:

1. **Multi-pass FM at the FINEST level beyond 3 passes**.  Sprint
   23 Day 11 found pass=5 hit the same 0.952× as pass=3.  Sprint
   26 could try (a) annealing acceptance rule, (b) different
   bucket-FM tie-break, (c) thick-restart-style FM with global
   rollback.  This is the most-promising candidate because the
   finest-level FM is the cut-determining stage on regular
   structured fixtures (Day 8 hypothesis).

2. **Direct geometric cut on regular grids**.  Detect the
   Pres_Poisson grid structure (vertex degrees clustered at 4 in
   interior + 3 on edges + 2 on corners) and substitute a
   geometric median-row-or-column cut.  Workload-specific but
   Pres_Poisson is the headline.

3. **Per-vertex separator scoring**.  Sprint 22 Day 4's
   `smaller_weight` lift + Sprint 24 Day 6's `balanced_boundary`
   lift both choose a side-then-lift; an alternative would
   choose vertices individually based on a "separator-suitability"
   score (similar to AMD pivot selection).

The Sprint 25 sweep's strongest evidence for routing is that
three independent algorithmic axes (HCC, multi-pass FM
intermediate, spectral bisection) ALL fail to move Pres_Poisson
significantly when applied at coarsest / intermediate levels —
the finest-level FM dominates downstream.  Sprint 26 must
intervene at the FINEST level (axis 1 above) or pre-empt the
multilevel pipeline entirely (axis 2).

## Day 10 inputs

Day 10 picks the production default + lands the per-item decision
docs based on this Day-9 sweep:

- `coarsening_decision.md` — `SPARSE_ND_COARSENING` flip
  decision (HCC default flip is now justified given setting 13's
  Pres_Poisson + corpus wins; Day 3's "Kuu regresses under HCC
  alone" finding is overridden by setting 13's "Kuu IMPROVES
  under HCC + ratio=200").
- `intermediate_fm_decision.md` — already written Day 5; passes=1
  stays default.  Day 9 confirms: settings 03/04 (interFM alone)
  regress Pres_Poisson; default-flip stays out.
- `spectral_bisection_decision.md` — `SPARSE_ND_COARSEST_BISECTION`
  flip decision.  Day 8's "spectral alone barely moves Pres_Poisson"
  finding confirms; setting 05 (spectral alone) is essentially
  default.  Spectral ships behind the env var as advisory for the
  ND wall_time benefit (23× on Pres_Poisson).

## Quality verification

- 96/96 measurements captured (16 settings × 6 fixtures).
- `bench_day9_combinations.csv` is the canonical record;
  `bench_day9_combinations.txt` is the human-readable view.
- All quality gates clean post-sweep (Day 9 sweep is doc-only;
  no source changes).

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 9
- `docs/planning/EPIC_2/SPRINT_25/bench_day9_combinations.{csv,txt}` —
  the 96-measurement sweep this verdict is based on
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day3.md` — Day 3 HCC
  escalation
- `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` —
  Day 5 multi-pass FM intermediate decision
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day8.md` — Day 8
  spectral escalation + Sprint 26 routing candidates
- Sprint 24 `RETROSPECTIVE.md` "What didn't go well" #1 — the
  historical pattern of Pres_Poisson literal targets missing.
