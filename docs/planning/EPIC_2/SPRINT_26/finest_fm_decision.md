# Sprint 26 Day 8 — FINEST FM (FIFO Variant) Decision

## Decision

**Production default stays `SPARSE_FM_FINEST_STRATEGY=baseline`**
(Sprint 23 LIFO-on-insertion-order behaviour).  FIFO ships behind
the env var as advisory only; **the Pres_Poisson 0.85× literal
target still misses by 7.2pp** (best opt-in 0.9217× = Sprint 25
setting 13 unchanged).  Day 9-10 (geometric grid-cut) is the next
0.85× attempt; if it falls short, Day 11-12 (per-vertex separator
scoring) is the third-and-last Sprint 26 candidate.

The Day 4 profile-driven hypothesis ("LIFO's deterministic tie-
break trapped FM in local minima") is **falsified on Pres_Poisson**:
FIFO finds different cuts but they are **worse** on this fixture,
both alone and in combination with Sprint 25's best opt-ins.

## Sweep results

8-setting × 6-fixture cross-corpus capture (`finest_fm_sweep.txt`).
ND nnz_L per fixture, with AMD baseline shown in the header for
ND/AMD ratio computation:

```
                  AMD nnz_L:    637      3 143      406 264     116 071     474 609   2 668 793
setting           |   nos4   bcsstk04        Kuu    bcsstk14   s3rmt3m3   Pres_Poisson  Pres ND/AMD
------------------|--------+----------+-----------+----------+----------+-------------+-----------
01 baseline       |   809      3 722    881 177    129 292    483 195   2 536 427    0.9504×
02 fifo alone     |   777      3 722    825 396    133 330    483 558   2 615 031    0.9799× ← +3pp regress
03 setting 13     |   887      3 717    796 898    125 032    482 669   2 459 934    0.9217× ★ best Pres
04 fifo + s13     |   808      3 700    752 431    132 699    484 257   2 583 021    0.9679× ← +1.7pp regress vs s13
05 s13 + interFM=3|   887      3 717    797 235    125 397    483 013   2 530 851    0.9483×
06 fifo + s13 + i3|   808      3 700    587 118    129 465    483 961   2 594 593    0.9722×
07 s15-ish        |   762      3 717    623 413    124 059    477 859   2 552 546    0.9565×
08 fifo + s15-ish |   748      3 700    610 509    120 650    481 479   2 488 715    0.9326×
```

(Settings legend: setting 13 = `SPARSE_ND_COARSENING=hcc` +
`SPARSE_ND_COARSEN_FLOOR_RATIO=200`; setting 15-ish = setting 13 +
`SPARSE_ND_COARSEST_BISECTION=spectral` +
`SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`.  All settings run
with Sprint 26 Day 5's `nd_base_threshold = 96`.)

## Pres_Poisson ranking

| rank | setting | Pres_Poisson ND/AMD | Δ vs baseline |
|---|---|---|---|
| 1 | **03 setting 13** | **0.9217×** | -2.9pp WIN |
| 2 | 08 fifo + setting 15-ish | 0.9326× | -1.8pp |
| 3 | 05 setting 13 + interFM=3 | 0.9483× | -0.2pp |
| 4 | 01 baseline | 0.9504× | — |
| 5 | 07 setting 15-ish | 0.9565× | +0.6pp regress |
| 6 | 04 fifo + setting 13 | 0.9679× | +1.8pp regress |
| 7 | 06 fifo + setting 13 + interFM=3 | 0.9722× | +2.2pp regress |
| 8 | 02 fifo alone | 0.9799× | +3.0pp regress |

**Setting 13 (HCC + ratio=200) is still the Pres_Poisson winner,
unchanged from Sprint 25 Day 9.**  FIFO regresses Pres_Poisson in
EVERY combination tested (+1.8 to +3.0pp).

## Per-fixture winners

| fixture | best setting | best ND nnz_L | ND/AMD | Δ vs Sprint 25 best |
|---|---|---|---|---|
| nos4 (n=100) | **08 fifo + s15-ish** | 748 | 1.174× | -7.5pp vs Day-5 baseline; better than Sprint 25 setting 15's 1.256× |
| bcsstk04 | (noise band) | 3 700 | 1.177× | flat ≈ 1.178× across all settings |
| Kuu | 06 fifo + s13 + interFM=3 | **587 118** | 1.445× | -41pp from default; but Sprint 25 setting 15's 1.309× still better |
| bcsstk14 | 08 fifo + s15-ish | 120 650 | 1.040× | -3pp from Sprint 25 setting 15's 1.037× (essentially equal) |
| s3rmt3m3 | 07 s15-ish | 477 859 | 1.007× | small win |
| Pres_Poisson | 03 setting 13 | 2 459 934 | 0.9217× | unchanged from Sprint 25 |

**FIFO contributes** to wins on **nos4 (-2pp from setting 15-ish
without FIFO)**, **bcsstk14 (-3pp)**, and **Kuu in combination with
interFM=3 (-2pp vs setting 5)**.  But on Pres_Poisson — the
headline — FIFO is a regression in every combination.

## Flip-rule application

PLAN.md Day 8 task 3 specifies the flip rule:
> Flip default if (a) Pres_Poisson tightens by ≥ 1pp AND (b) no
> smaller fixture regresses past 5pp.

| candidate default | (a) Pres_Poisson Δ | (b) max smaller-fixture regress | flip-rule |
|---|---|---|---|
| `fifo` | +3.0pp regress | n/a (gate (a) fails) | FAIL |
| `setting 13` | -2.9pp ✓ | (Day 3 unblocked HCC bcsstk14 sep=0; Kuu HCC-alone +8.5pp regress prevented Sprint 25 default-flip; with ratio=200 the Kuu profile is recovered) | already opt-in; not the FIFO axis |

**FIFO does not satisfy the flip rule.**  Default stays at
`baseline`.

The Sprint 26 Day 5 `nd_base_threshold = 96` flip already
contributed Pres_Poisson 0.9524× → 0.9504× (-0.2pp) on the default
path; that's the only Sprint 26 default-path Pres_Poisson tightening
to date.  The 7.2pp gap to 0.85× remains.

## Why FIFO falsified the Day 4 hypothesis

Day 4's per-recursion-depth profile drove the FIFO selection
rationale: "169 small-subgraph FM calls at depths 6-9 each get a
different exploration direction under FIFO, breaking the LIFO
saturation at 3 passes."

The empirical finding contradicts this on Pres_Poisson:

1. **LIFO's local minima are good minima.**  The Sprint 25 Day 5
   saturation at 0.952× (passes ≥ 5 → no improvement) suggested LIFO
   was stuck.  But it's stuck at a *high-quality* local minimum.
   FIFO's different exploration direction lands at a *different*
   local minimum that's slightly *worse* on this fixture.
2. **The FM landscape is regular on Pres_Poisson.**  Pres_Poisson is
   a 2D Poisson-grid fixture; its FM landscape on the coarsened
   levels has many equivalent good cuts (the regular-grid structure
   means most planar cuts have similar gain profiles).  Tie-break
   choice (FIFO vs LIFO) picks different members of the equivalent
   class but doesn't move beyond it.  Sub-axis (b) bucket-tie-break
   exploration is fundamentally limited by the cut equivalence class
   the coarsening + bisection landed in.
3. **The 169 small-subgraph FM calls compound randomly, not
   constructively.**  Day 4's hypothesis assumed 169 different
   exploration directions would produce 169 marginal improvements.
   In practice, half help + half hurt + the net is noise on
   Pres_Poisson.

This negative result is informative for Sprint 26 Item 5's design:
**any tie-break-only intervention will fail on Pres_Poisson**.  The
Pres_Poisson 0.85× target requires either (a) a different *cut
selection* (geometric grid-cut, Item 6) or (b) a different
*acceptance criterion* (annealing over local-minimum thresholds,
sub-axis (a) — but rejected at Day 6 design for cost reasons).

## What FIFO does help

Despite the Pres_Poisson regression, FIFO contributes to small wins
on:

- **nos4**: setting 08 (fifo + setting 15-ish) lands 748 nnz_L,
  better than Sprint 25 setting 15's 1.256× × 656 ≈ 824 → 748 is
  -9pp tighter.  Small fixture; cumulative win of all opt-ins.
- **bcsstk14**: setting 08 lands 120 650, vs Sprint 25 setting 15's
  1.037× × 116 071 ≈ 120 350 — essentially flat.
- **Kuu in combination with interFM=3**: setting 06 lands Kuu
  587 118 (1.445×), -2pp from setting 05 alone.  But Sprint 25's
  setting 15 (1.309×) is still better; setting 06 is not advisory.

These are all **opt-in wins for setting 08 (fifo + setting 15-ish)**
on small / irregular fixtures.  Setting 08 is **not advisory for
Pres_Poisson** (regresses to 0.9326× vs setting 13's 0.9217×).

## Per-fixture advisory (Sprint 26 user guidance)

Updated per-fixture advisory list (extends Sprint 25 advisory):

| workload | recommended setting | Pres_Poisson | corpus |
|---|---|---|---|
| Pres_Poisson | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` (setting 13; **unchanged from Sprint 25**) | **0.9217×** | flat-to-noise on others |
| nos4 / bcsstk14 / s3rmt3m3 (small / structural-mechanics) | setting 08: setting 13 + spectral + balanced_boundary + `SPARSE_FM_FINEST_STRATEGY=fifo` | 0.9326× (small regress) | nos4 -9pp, bcsstk14 ~flat, s3rmt3m3 -1pp; FIFO contribution = -2 to -3pp on the small fixtures |
| Kuu (irregular SPD) | full setting (Sprint 25 setting 15: hcc + ratio=200 + spectral + balanced_boundary + INTERMEDIATE=3) | 0.9483× | Kuu 1.309× best |
| Default (no env vars) | baseline | 0.9504× | (Day 5 nd_base_threshold=96 flip default) |

FIFO is recommended only as part of setting 08 for non-Pres_Poisson
workloads.  Pres_Poisson workloads should NOT enable FIFO.

## Escalation to Day 9-10 (geometric grid-cut)

Per PLAN.md Day 8 task 5, FIFO failing the Pres_Poisson flip rule
escalates the 0.85× target to Day 9-10's geometric grid-cut
intervention.  The reasoning chain:

1. **Sprint 25 Day 9** ran 16 settings × 6 fixtures = 96 measurements
   on the algorithmic axes available then (HCC, intermediate-FM,
   spectral) and concluded none move Pres_Poisson individually.
2. **Sprint 26 Day 4** profile-driven analysis pointed at FINEST FM
   tie-break as the next-most-likely lever.
3. **Sprint 26 Day 8** (this) found FINEST FM tie-break also doesn't
   move Pres_Poisson.

The remaining Sprint 26 candidates:

- **Item 6 (geometric grid-cut)**: detect Pres_Poisson's regular 2D
  grid structure + substitute a geometric median-row-or-column cut
  for the multilevel pipeline.  Pres_Poisson-specific by design;
  irregular fixtures must reject the heuristic cleanly.
- **Item 7 (per-vertex separator scoring)**: alternative to
  Sprint 22's smaller-side lift + Sprint 24's balanced_boundary lift.
  Score boundary vertices individually; picks may differ enough to
  produce a structurally different separator.

Neither is profile-driven; both are workload-specific bets.  If
Item 6 lands the 0.85× close on Pres_Poisson, Sprint 26 ships its
headline.  If neither lands, **Pres_Poisson 0.85× becomes the fourth
sprint in a row to miss the literal target** (Sprint 22-25 + Sprint
26), and Sprint 27+ inherits with much narrower remaining
candidates.

## Sprint 27 inputs (if items 6-7 also fall short)

Per Sprint 25 RETROSPECTIVE.md "Sprint 26 inputs" #1, the three
sub-axes were:

- (a) annealing acceptance — REJECTED at Day 6 design (+20-50% wall);
  Day 5's wall improvement makes this affordable now.
- (b) bucket-tie-break — IMPLEMENTED Day 7-8; Pres_Poisson FAIL.
- (c) thick-restart-style FM with global rollback — REJECTED at Day 6
  design (2-3× wall); Day 5's wall improvement reduces but doesn't
  eliminate the cost concern.

Sprint 27+ candidates if Sprint 26 items 6-7 also fall short:

1. **Sub-axis (a) annealing**: now-affordable cost; could land in
   Sprint 27 with a 4-day budget (Days 1-2 design + impl, Days 3-4
   sweep + decide).
2. **Multi-strategy ensemble**: run baseline + FIFO + (future
   annealing) in parallel; pick best cut per partition call.
   Doubles wall but explores 2× the FM landscape.
3. **Pre-empt the multilevel pipeline at large n**: Sprint 26 Day 5
   raised `nd_base_threshold = 96` to 96; bumping further to 256-512
   would push more work to leaf-AMD on Pres_Poisson + skip the
   coarsening + FM cascade entirely.  Trade-off vs leaf-AMD fill
   quality at large n.

## What ships in Sprint 26 Day 8

- `finest_fm_decision.md` (this doc) — production-default decision
  (no flip), per-fixture sweep table, flip-rule application,
  per-fixture advisory updates, Sprint-27 inputs.
- `finest_fm_sweep.txt` — raw 8-setting × 6-fixture capture.
- No source changes (default stays at baseline; FIFO continues to
  ship as opt-in via Day 7's `SPARSE_FM_FINEST_STRATEGY=fifo`).
- All quality checks clean.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 8 + Day 9 + Day 10
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md` — Day 6
  sub-axis selection rationale (now partially falsified by Day 8
  empirical)
- `docs/planning/EPIC_2/SPRINT_26/per_recursion_profile_day4.md` —
  Day 4 profile + hypothesis (also falsified on Pres_Poisson)
- `docs/planning/EPIC_2/SPRINT_26/bench_day7_finest_fm_quicklook.txt`
  — Day 7 quick-look that flagged this escalation
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_sweep.txt` — Day 8 raw
  sweep (8 settings × 6 fixtures)
- `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Sprint 26 inputs"
  #1 — the three-sub-axis menu Day 6-8 down-selected from
- `src/sparse_graph.c::graph_uncoarsen` — Day 6/7 dispatch site
  (unchanged today)
- `src/sparse_graph.c::fm_bucket_pop_max_tail` — Day 7 FIFO pop
  (continues to ship as the runtime backing for `=fifo`)
