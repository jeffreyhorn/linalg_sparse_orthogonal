# Sprint 25 Day 5 — `SPARSE_FM_INTERMEDIATE_PASSES` Production-Default Decision

## Decision

**Production default stays at `SPARSE_FM_INTERMEDIATE_PASSES=1`**
(Sprint 23 Day 11 behavior; intermediate uncoarsening levels run
single-pass FM).  Neither `=2` nor `=3` meets the PLAN.md Day 5
flip rule on the headline fixture.

Two corpus-specific advisory env-var settings are documented for
opt-in use:

- **Kuu (irregular SPD, n = 7 102)**: set
  `SPARSE_FM_INTERMEDIATE_PASSES=3` for a 10.2 % nnz_L drop
  (924 385 → 829 774; ND/AMD 2.275× → 2.043×).  Wall cost: +3.7 %.
  This is the strongest per-fixture win Sprint 25 has produced
  so far (across HCC + multi-pass intermediate FM combined).
- **bcsstk14 (irregular SPD, n = 1 806)**: set
  `SPARSE_FM_INTERMEDIATE_PASSES=2` for a 1.9 % nnz_L drop
  (131 017 → 128 501; ND/AMD 1.129× → 1.107×).  Wall cost: +4.7 %.
  passes=3 is marginally better on nnz_L (-2.0 % vs -1.9 %) but
  saves -11.4 % wall on bcsstk14, so passes=3 is also a
  reasonable advisory if other Kuu-class fixtures aren't in the
  workload mix.

## Flip-rule application

PLAN.md Sprint 25 item 2 + Day 5 task 2 spec the flip rule:

> Flip default if the chosen value tightens Pres_Poisson by
> ≥ 1pp AND no smaller fixture regresses past 5pp.

Sweep results vs the rule:

| setting   | Pres_Poisson Δ      | smaller-fixture worst | flip? |
|-----------|---------------------|-----------------------|-------|
| passes=2  | -0.04pp (< 1pp)     | Kuu +6.6pp (> 5pp)    | ✗ ✗   |
| passes=3  | +1.5pp (regression) | Kuu -23.2pp (win)     | ✗     |

`passes=2` fails both halves of the rule (Pres_Poisson barely
moves AND Kuu regresses past the 5pp band).  `passes=3` fails the
Pres_Poisson half (regresses Pres_Poisson by 1.5pp) — even though
its smaller-fixture profile is the strongest in the sweep (Kuu
-23.2pp; bcsstk14 -2.3pp), it makes the headline gate worse.

Default stays `passes=1`.  Item 2 contributes ~0pp to the
Pres_Poisson 0.85× headline gate.

## Implications for the Sprint 25 headline gate

After Day 3's HCC contribution (~1.5pp on Pres_Poisson) and
Day 5's multi-pass FM intermediate contribution (~0pp), the
remaining gap to the 0.85× literal target is:

- Sprint 24 baseline: 0.952×
- After HCC alone (Day 3): 0.937× (-1.5pp from Sprint 24)
- After HCC + multi-pass FM intermediate (Day 5): ~0.937× (no
  additional movement; passes=2 doesn't compose
  constructively with HCC at this fixture)
- Remaining gap to 0.85×: ~8.7pp

**Days 6-8 (spectral bisection) must close ~8.7pp on Pres_Poisson
essentially alone**.  This is the strongest concentration of the
headline-gate burden onto a single Sprint 25 deliverable; the
fallback path (partial close to ≤ 0.90× → Sprint 26 routing) is
the documented graceful-degradation per Sprint 24's
RETROSPECTIVE.md "What didn't go well" #1 ("0.85× stretch target
is the second sprint in a row to miss its literal goal").

## Per-fixture advisory

| fixture | recommended setting | nnz_L Δ | ND/AMD ratio | rationale |
|---|---|---|---|---|
| nos4 | (default) `passes=1` | 0 | 1.520× = | small fixture; floor pegs at MAX(20, 1) — no movement |
| bcsstk04 | (default) `passes=1` | 0 | 1.178× = | small fixture; no movement |
| Kuu | **`passes=3`** | -10.2 % (924 385 → 829 774) | 2.275× → 2.043× (-23.2pp) | strongest single-fixture win in Sprint 25 to date; +3.7 % wall cost |
| bcsstk14 | **`passes=2`** or `passes=3` | -1.9 % / -2.0 % | 1.129× → 1.107× / 1.106× | both work; passes=2 +4.7 % wall, passes=3 -11.4 % wall |
| s3rmt3m3 | (default) `passes=1` | +1.7 % regression at passes=2/3 | 1.009× → 1.026× | passes ≥ 2 hurts this fixture; stay default |
| Pres_Poisson | (default) `passes=1` | -0.04 % at passes=2; +1.6 % regress at passes=3 | 0.952× = | item 2 contributes ~0pp to headline; default optimal |

## Why multi-pass at intermediate levels doesn't help Pres_Poisson

Sprint 23 Day 11's multi-pass at the FINEST level dropped
Pres_Poisson from 1.026× (1 pass) → 0.952× (3 passes) — a 7pp
improvement.  Why does the same trick at the SECOND-finest level
contribute ~0pp?

Hypothesis (informed by Sprint 24 retrospective + Day 2's HCC
finding that downstream FM washes out coarsening choices):

The finest-level FM operates on the actual graph — its refinements
directly produce the cut that ND uses.  The second-finest level's
FM operates on a coarsened graph; any refinement it produces gets
projected to the finer level via cmap and then re-refined by the
finest-level FM.  If the finest-level FM converges to a local
optimum independent of starting state (a property of FM on dense-
enough graphs after enough passes), then the second-finest-level
refinement's effort is wasted — the finest-level FM throws away
its work.

Pres_Poisson is "regular enough" (uniform 2D grid structure) that
3-pass finest-level FM converges to essentially the same cut
regardless of the starting partition the second-finest level
hands off.  Kuu is "irregular enough" that the finest-level FM
doesn't fully converge in 3 passes; the second-finest level's
refinement gives the finest level a better starting point that
*does* propagate to a measurably better final cut.  bcsstk14 sits
between the two extremes.

This pattern (FM at the finest level dominates; intermediate-
level refinement only helps when the finest level is under-
converged) suggests:

1. **Production-default `passes=1` is correct for Pres_Poisson-
   shaped workloads.**  No flip-rule revision warranted.
2. **Per-fixture advisory `passes=3` is correct for
   Kuu-shaped workloads.**  Documented above.
3. **Days 6-8 spectral bisection** is the only remaining axis
   that intervenes BEFORE the finest-level FM has a chance to
   wash everything out — spectral bisection at the COARSEST level
   produces a cut that propagates ALL the way through every
   uncoarsening level + finest-level FM.  This is consistent
   with Day 3 escalation's "spectral is now THE primary headline-
   mover" finding.

## Related decisions

- Sprint 24 Day 5 (`SPARSE_ND_COARSEN_FLOOR_RATIO`) — same
  decision pattern: env-var-gated alternative; default unchanged
  for corpus-wide robustness; per-fixture advisory recorded for
  opt-in.
- Sprint 24 Day 6 (`SPARSE_ND_SEP_LIFT_STRATEGY`) — same pattern.
- Sprint 25 Day 3 (`SPARSE_ND_COARSENING`) — same pattern; HCC
  default stays `heavy_edge` because of Kuu's regression at HCC.
  Day 3's `nd_tuning_day3.md` has the full HCC flip-rule
  application.

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 5 — task list +
  flip rule
- `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_sweep.txt` —
  the per-fixture × per-setting capture this decision is based on
- `docs/planning/EPIC_2/SPRINT_25/nd_tuning_day3.md` — Day-3
  HCC escalation finding that this decision builds on
- `src/sparse_graph.c::graph_uncoarsen` — Sprint 25 Day 4
  implementation: per-level pass count selection routes
  level == 1 + level == 2 to `intermediate_passes`; level == 0
  to `finest_passes`; level ≥ 3 to single-pass
