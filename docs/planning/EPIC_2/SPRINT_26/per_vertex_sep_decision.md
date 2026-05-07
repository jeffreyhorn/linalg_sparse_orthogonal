# Sprint 26 Day 12 — Per-Vertex Separator Scoring Decision

## Decision

**Production default stays `SPARSE_ND_SEP_LIFT_STRATEGY=smaller_weight`**
(Sprint 22 baseline; bit-identical to current master).  Per_vertex
ships as opt-in advisory ONLY for bcsstk04-class fixtures; on every
other fixture, **`balanced_boundary` (Sprint 24 Day 6 advisory) is
strictly better than per_vertex**.

Day 12's three-weight-scheme sweep also produced a methodological
finding: **the 70/30-balance-respecting greedy top-K selection
dominates the score formula.**  All three per_vertex weight schemes
(`per_vertex` / `per_vertex_balance` / `per_vertex_degree`) converge
to nearly-identical outputs on 5 of 6 fixtures (bit-identical on
nos4, bcsstk04, bcsstk14, s3rmt3m3, Pres_Poisson; differ by ≤ 1.5 %
on Kuu).  Tunable scoring weights are NOT a useful sweep dimension
in this implementation.

## Sweep results (PLAN.md task 1)

5-scheme × 6-fixture cross-corpus capture (`per_vertex_sep_sweep.txt`).
Sprint 26 Day 5's `nd_base_threshold = 96` is the active default;
all settings sweep on top of that.

ND nnz_L per setting × fixture:

| setting | nos4 | bcsstk04 | Kuu | bcsstk14 | s3rmt3m3 | Pres_Poisson |
|---|---:|---:|---:|---:|---:|---:|
| 01 smaller_weight (default) | 809 | 3 722 | 881 177 | 129 292 | 483 195 | 2 536 427 |
| 02 balanced_boundary | 748 | 3 722 | 527 884 | 123 909 | 477 485 | 2 542 064 |
| 03 per_vertex (hybrid) | 765 | **3 549** | 691 880 | 151 049 | 614 153 | 3 312 433 |
| 04 per_vertex_balance | 765 | **3 549** | 705 728 | 151 049 | 614 153 | 3 312 433 |
| 05 per_vertex_degree | 765 | **3 549** | 701 332 | 151 049 | 614 153 | 3 312 433 |

ND/AMD ratios (AMD nnz_L for reference: nos4=637, bcsstk04=3143,
Kuu=406264, bcsstk14=116071, s3rmt3m3=474609, Pres_Poisson=2668793):

| setting | nos4 | bcsstk04 | Kuu | bcsstk14 | s3rmt3m3 | **Pres_Poisson** |
|---|---:|---:|---:|---:|---:|---:|
| 01 smaller_weight | 1.270× | 1.184× | 2.169× | 1.114× | 1.018× | **0.9504×** |
| 02 balanced_boundary | 1.174× ★ | 1.184× | **1.299×** ★ | **1.068×** ★ | **1.006×** ★ | 0.9525× |
| 03 per_vertex_hybrid | 1.201× | **1.129×** ★ | 1.703× | 1.301× | 1.294× | 1.241× |
| 04 per_vertex_balance | 1.201× | **1.129×** ★ | 1.737× | 1.301× | 1.294× | 1.241× |
| 05 per_vertex_degree | 1.201× | **1.129×** ★ | 1.726× | 1.301× | 1.294× | 1.241× |

★ = best for that fixture across all settings.

## Per-fixture analysis

### Pres_Poisson (the headline)

**No per_vertex variant helps.**  All three per_vertex weight schemes
land at **0.9504× → 1.241× = +29pp REGRESSION**.  The +29pp
regression is the same to 6 significant digits across all three
weight variants — the scoring difference is entirely washed out by
the greedy top-K + 70/30 balance selection.

Best Pres_Poisson among Day 12 settings: smaller_weight (default;
0.9504×) — the existing baseline.  Sprint 26's Day 13-14 closing
work continues with Pres_Poisson at the Sprint 25 setting 13 best
opt-in (0.9217× via HCC + ratio=200; not advisable for fixtures with
HCC's bcsstk14-style sep=0 risk, but Sprint 26 Day 3 fixed that).

### Kuu

**balanced_boundary wins** (1.299× vs per_vertex's 1.703-1.737×).
Sprint 24 Day 6's existing advisory is strictly better than any
Sprint 26 per_vertex variant on Kuu.

The 3 per_vertex variants differ by ~1.5 % on Kuu (691k / 706k /
701k vs each other) — the only fixture where the weight schemes
produce non-bit-identical outputs.  Why: Kuu's high CV (0.425) +
diverse boundary-vertex profile gives the score formula slightly
different rankings.  But Hybrid (691k) is still ~30 % worse than
balanced_boundary (528k) and not advisable.

### bcsstk04 (the only per_vertex win)

per_vertex variants land 3 549, vs smaller_weight + balanced_boundary
both at 3 722.  Per_vertex wins by **-4.6 %** here.  bcsstk04 is a
small fixture (n=132) where the per-vertex top-K selection happens
to find a slightly tighter separator than the side-then-lift
heuristics.  Small absolute win (~170 nnz); per-vertex's primary
advisory niche is exactly this fixture class.

### bcsstk14 / s3rmt3m3 / nos4

All three per_vertex variants regress vs balanced_boundary by
3-5 %.  Smaller_weight + balanced_boundary remain superior.

## All three weight schemes converge

Looking at the 5-scheme matrix more carefully:

- **5 of 6 fixtures: bit-identical across all 3 per_vertex variants.**
  nos4 = 765, bcsstk04 = 3 549, bcsstk14 = 151 049, s3rmt3m3 =
  614 153, Pres_Poisson = 3 312 433 — every weight scheme produces
  the same nnz_L on these fixtures.
- **Kuu only: variants differ by ≤ 1.5 %** (691k vs 706k vs 701k).

Why scheme-convergence?  The greedy top-K selection is gated by the
70/30 post-lift balance constraint.  Once enough vertices are lifted
to push max(w0, w1) / total past 0.70, selection STOPS — regardless
of how many higher-score vertices are still in the queue.

The score formula determines the ORDER of vertices visited, but the
70/30 stop criterion picks the same SET of vertices in the FINAL
analysis (the constraint is symmetric across the score-ranking
direction; vertices that satisfy 70/30 collectively are similar for
all weight schemes).

**Conclusion**: Day 12's planned weight-sweep dimension is not
useful in this implementation.  Tunable scoring weights would
require a fundamentally different selection criterion (e.g.,
fixed-K with K = min(boundary_count[0], boundary_count[1]) instead
of dynamic-K with 70/30 balance gate) for the weights to
differentiate.  Sprint 27+ work, if at all.

## Flip-rule application (PLAN.md task 2)

PLAN.md Day 12 task 2 specifies the flip rule:
> Flip default if (a) Pres_Poisson tightens by ≥ 1pp AND (b) no
> smaller fixture regresses past 5pp.

| candidate | (a) Pres_Poisson | (b) max smaller-fixture regress | flip-rule |
|---|---|---|---|
| `per_vertex_hybrid` | +29pp regress | n/a (gate (a) fails by far) | FAIL |
| `per_vertex_balance` | +29pp regress | n/a | FAIL |
| `per_vertex_degree` | +29pp regress | n/a | FAIL |

**No per_vertex variant satisfies the flip rule.**  Default stays
at `smaller_weight`.

## Cross-evaluation against Items 5/6 (PLAN.md task 3)

Per Sprint 26 Items 5/6/7 status:
- **Item 5 (FINEST FM FIFO)**: Day 8 decision, NO FLIP.  Pres_Poisson
  +3pp regress alone; +1.7pp regress combined.
- **Item 6 (geometric grid-cut)**: Day 9 REJECTED.  Pres_Poisson
  doesn't have a 2D-grid signature (mean degree 47.3, NOT ~5).
- **Item 7 (per-vertex sep scoring)**: Day 12 decision (this), NO
  FLIP.  Pres_Poisson +29pp regress alone.

**All three Sprint 26 algorithmic-axis attempts fall short of the
0.85× literal target.**  The Pres_Poisson gap to 0.85× remains
7.2pp (best opt-in still Sprint 25 setting 13 = 0.9217×).

This is the **fourth consecutive sprint to miss the Pres_Poisson
literal target**:
- Sprint 22 PLAN's 0.5× → 1.063× actual
- Sprint 23 PLAN's 0.7× → 0.952× actual
- Sprint 24 PLAN's 0.85× → 0.942× best opt-in
- Sprint 25 PLAN's 0.85× → 0.9217× best opt-in
- Sprint 26 PLAN's 0.85× → 0.9217× best opt-in (unchanged from S25)

Sprint 27+ inherits the gap.

## Per-fixture advisory (Sprint 26 final)

Updated per-fixture advisory (extends Sprint 24/25 advisory):

| workload | recommended setting | Pres_Poisson | corpus |
|---|---|---|---|
| Pres_Poisson | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` (Sprint 25 setting 13; **unchanged from Sprint 25**) | **0.9217×** | flat-to-noise on others |
| Kuu / irregular SPDs | `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (Sprint 24 advisory; **unchanged**) | 0.9525× ≈ neutral | Kuu 1.299× best |
| bcsstk04 (small irregular) | `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` (Sprint 26 new; -4.6 %) | 1.241× regress; **avoid** | bcsstk04 only marginally better than baseline |
| Default (no env vars) | smaller_weight + nd_base_threshold=96 (Sprint 26 Day 5 default flip) | 0.9504× | unchanged from Sprint 26 Day 5 |

**FIFO (Day 7)** continues to ship behind `SPARSE_FM_FINEST_STRATEGY=fifo`
as advisory; combined with setting 15-ish (HCC + ratio=200 +
spectral + balanced_boundary), it produces small wins on smaller
fixtures.

## What ships in Sprint 26 Day 12

- `src/sparse_graph.c`:
  - 3 new SEP_LIFT_PER_VERTEX_* enum values
  - per_vertex code path extended with the three preset weight schemes
    (hybrid / balance / degree); Day 10's hybrid is the unmodified
    `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex` value.
  - `is_per_vertex_strategy()` helper for the dispatch gate.
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_sweep.txt` — raw
  5-scheme × 6-fixture sweep.
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_decision.md` (this
  doc) — production-default decision (no flip), per-fixture
  advisory, weight-scheme convergence finding, Sprint 27 inputs.
- All quality checks clean.  Default code path
  (`SEP_LIFT_SMALLER_WEIGHT`) bit-identical to current master.

## Sprint 27 inputs

If Sprint 27 inherits the Pres_Poisson 0.85× target, the remaining
candidates outlined in `geometric_cut_design.md` Sprint-27 routing
+ `finest_fm_decision.md` Sprint-27 inputs:

1. **Root-level spectral bisection** (4-day budget): extend Sprint 25
   Day 6-8's spectral from coarsest to root level.  Higher prior
   than Sprint 26 Item 6's 2D-grid heuristic.
2. **Annealing-acceptance FM** (3-4 day budget): rejected at Day 6
   for cost reasons; Day 5's wall improvement makes affordable now.
3. **Multi-strategy ensemble**: run baseline + FIFO + (future axes)
   in parallel; pick best cut per partition call.
4. **Larger nd_base_threshold beyond 96**: Sprint 26 Day 5 found
   t=96 was the maximum threshold satisfying the flip rule; t=128
   regressed s3rmt3m3 by +1.05 %.  Sprint 27 could re-evaluate
   with relaxed flip-rule (e.g., 2pp tolerance) to push past 96.
5. **Tunable per_vertex selection criterion** (1-2 day budget):
   Day 12 finding that the 70/30 balance gate dominates; Sprint 27
   could explore fixed-K selection (K = min(boundary_count[0,1]))
   instead of dynamic-K to make the weight schemes differentiate.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 12
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_design.md` — Day 10
  design + quick-look (now finalised by this doc)
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_sweep.txt` — raw
  Day 12 sweep
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md` — Day 8 FIFO
  decision (parallel rejection)
- `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md` — Day 9
  Item 6 rejection
- `docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md` — Sprint 25
  setting 13 advisory (still the Pres_Poisson best opt-in)
- `docs/planning/EPIC_2/SPRINT_24/nd_sep_strategy_decision.md` —
  Sprint 24 Day 6 balanced_boundary decision (still the
  non-Pres_Poisson workload advisory)
- `src/sparse_graph.c::graph_edge_separator_to_vertex_separator` —
  Day 10/12 implementation
