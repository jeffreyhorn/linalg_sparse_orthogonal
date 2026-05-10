# Annealing FM at the Finest Level — Flip-or-Stay Decision (Sprint 27 Day 7)

## Background

Sprint 27 Day 5 wired the dispatch skeleton for `SPARSE_FM_FINEST_STRATEGY=annealing`; Day 6 implemented the acceptance-probability overlay (`P = exp(g/T)`) in `graph_refine_fm` plus the three temperature schedules behind `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential (default), cosine}`.  Sprint 27 PLAN.md Day 7 task 2 specifies the flip rule:

> Flip `SPARSE_FM_FINEST_STRATEGY` default to `annealing` (with the best-performing schedule) if (a) Pres_Poisson lands ≤ 0.85× of AMD nnz_L AND (b) no smaller-fixture regress past 5pp.

This document captures Day 7's flip-rule application against the Day-6 corpus sweep + the decision (advisory or default).

## Reproducer

```
SPARSE_FM_FINEST_STRATEGY=annealing build/bench_reorder --only <fixture> --skip-factor
SPARSE_FM_FINEST_STRATEGY=annealing SPARSE_FM_ANNEALING_SCHEDULE=linear build/bench_reorder ...
SPARSE_FM_FINEST_STRATEGY=annealing SPARSE_FM_ANNEALING_SCHEDULE=cosine build/bench_reorder ...
```

Default coarsening: HCC + Kuu-safe (Sprint 27 Day 2 flip); `nd_base_threshold = 128` (Sprint 27 Day 3 flip).

## Sprint 27 Day 7 Sweep Table

(Captured in `annealing_fm_sweep.txt`; ND nnz_L only — wall-time trends consistent with the schedule choice.)

| Fixture | n | AMD | default | exponential | linear | cosine |
|---|---:|---:|---:|---:|---:|---:|
| nos4 | 100 | 637 | 637 | 637 | 637 | 637 |
| bcsstk04 | 132 | 3 143 | 3 722 | 3 722 | 3 722 | 3 722 |
| Kuu | 7 102 | 406 264 | 764 664 | 815 158 | 765 132 | 815 158 |
| **bcsstk14** | 1 806 | 116 071 | 130 422 | **129 530** | 130 111 | **129 530** |
| s3rmt3m3 | 5 357 | 474 609 | 487 832 | 487 832 | 487 832 | 487 832 |
| **Pres_Poisson** | 14 822 | 2 668 793 | **2 462 201** | 2 538 898 | 2 515 202 | 2 535 749 |

### Per-fixture nnz_L deltas vs Day-3 default (%)

| Fixture | exponential | linear | cosine |
|---|---:|---:|---:|
| nos4 | 0.0 % | 0.0 % | 0.0 % |
| bcsstk04 | 0.0 % | 0.0 % | 0.0 % |
| Kuu | **+6.6 %** | +0.06 % | **+6.6 %** |
| bcsstk14 | -0.7 % win | -0.2 % win | -0.7 % win |
| s3rmt3m3 | 0.0 % | 0.0 % | 0.0 % |
| **Pres_Poisson** | **+3.1 %** | **+2.2 %** | **+3.0 %** |

### Per-fixture ND/AMD ratio under default vs annealing-best

| Fixture | default | annealing best | annealing schedule |
|---|---:|---:|---|
| Kuu | 1.882× | 1.883× | linear (0% delta) |
| bcsstk14 | 1.124× | **1.116×** | exponential or cosine |
| Pres_Poisson | **0.923×** | 0.943× | linear (least bad) |

## Flip-Rule Application

Flip rule:
- **(a)** Pres_Poisson lands ≤ 0.85× of AMD nnz_L.
- **(b)** No smaller-fixture regress past 5pp.

| Schedule | (a) Pres_Poisson ratio | (b) max regress | flip-rule |
|---|---|---|---|
| exponential | 0.951× ✗ | Kuu +6.6 % ✗ | FAIL (both) |
| linear | 0.943× ✗ | Kuu +0.06 % ✓ | FAIL ((a)) |
| cosine | 0.951× ✗ | Kuu +6.6 % ✗ | FAIL (both) |

**No schedule satisfies the flip rule.**  All three regress Pres_Poisson (gate (a) fails by 9-10pp on the literal 0.85× target; the regress alone is 2.2-3.1pp from the Day-3 default).  Two of three also regress Kuu past the 5pp budget (gate (b)).

Linear schedule is the "least bad" option (smallest Pres_Poisson regress + smallest Kuu impact) but still fails the headline target.

## Decision: Stay At Default `baseline`; Ship Annealing As Advisory

Annealing **does not move Pres_Poisson toward the 0.85× target**; it actively regresses it.  The hypothesised mechanism — annealing's stochastic acceptance lets the FM walk explore cuts that baseline-greedy can't reach — fails empirically because Pres_Poisson's regular FE-mesh structure has a single global-best cut that baseline FM happens to find efficiently.  Annealing's random rejections cause the trajectory to MISS some of baseline's saved-best path, landing on a final cut that's strictly worse.

Where annealing helps: bcsstk14 (irregular structural-mechanics fixture; multiple near-optimal cuts; annealing's stochastic exploration lands on a tighter cut than baseline's saved best).  Win is small (-0.7% nnz_L; 1.124× → 1.116× of AMD) but real.

Where annealing hurts: Pres_Poisson (regular FE-mesh; single global-best cut; annealing trajectory misses it) and Kuu (high-CV bimodal-degree; annealing's stochasticity disrupts the careful coarsening + lift sequence Sprint 27 Day 2's CV-fall-through tuned to land at 1.882×).

### Why this isn't surprising in retrospect

Sprint 26 Day 6 originally rejected annealing on cost grounds (20-50 % wall expansion); Sprint 27 Day 5 revisited under the post-Sprint-27-Day-3 wall budget and found cost is acceptable.  But the Day-6 measurement reveals a deeper issue: **FM's rollback-to-best-cut floor combined with annealing's stochastic acceptance means annealing's trajectory subtractively deviates from baseline's**.  The hope that annealing would *escape* local minima neglected that baseline already finds the global minimum on regular structures.

The right framing of when annealing helps: fixtures where baseline FM converges to a *suboptimal* local minimum.  bcsstk14 fits this; Pres_Poisson and Kuu don't.

## Per-Fixture-Class Advisory

| Fixture class | Annealing verdict | Default behaviour |
|---|---|---|
| Tiny (n ≤ 200; nos4, bcsstk04) | Bit-stable (no FM passes) | Stay default |
| Regular FE-meshes (Pres_Poisson, s3rmt3m3) | Regress / neutral | Stay default |
| **Irregular structural mechanics (bcsstk14-class)** | **Slight win (-0.7 %)** | **Stay default; advisory opt-in** |
| Bimodal-degree (Kuu) | Regress | Stay default |

Workloads dominated by irregular structural-mechanics SPDs (bcsstk14's class — non-regular degree distribution, multiple near-optimal cuts) get a small win via `SPARSE_FM_FINEST_STRATEGY=annealing` (default exponential schedule).  Documented in `docs/algorithm.md` Sprint 27 closure subsection.

## Item 5 Routes Up

The 0.85× literal target gap remains 7.3pp (default 0.923×; target 0.85×).  Sprint 27 PLAN.md item 5 (root-level spectral bisection, Days 7-9) is the next 0.85× candidate.  Item 5's mechanism — full-graph Fiedler-vector projection + median bisection — bypasses the multilevel pipeline entirely on inputs where Lanczos is fast enough.  This is structurally different from annealing FM (which acts at the finest level *of* the multilevel pipeline) and may reach cuts the multilevel pipeline can't find.

If item 5 also misses the target, item 6 (thick-restart FM, Days 10-12) fires as a tertiary candidate.  If all three fall short, Sprint 28+ pivots to non-pipeline-level interventions per the PROJECT_PLAN.md follow-up routing.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_sweep.txt` — canonical 3-schedule × 6-fixture corpus sweep
- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_decision.md` — this document

## Files NOT Modified

- `src/sparse_graph.c` — `parse_finest_fm_strategy()` default stays `FINEST_FM_BASELINE` (no flip)
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound stays `≤ 0.96×` (no tightening; Day-3 default 0.923× is well within bound; Day 13 may revisit after items 5-6)

## Headline Status After Day 7 Item-4 Close

- **Pres_Poisson default unchanged: 0.923× of AMD** (no annealing regress under default).
- bcsstk14 advisory opt-in via `SPARSE_FM_FINEST_STRATEGY=annealing` (-0.7 % nnz_L).
- 7.3pp gap remains to literal 0.85× target → item 5 (root-level spectral) carries the remaining weight in Days 7-9 design + Days 8-9 implementation.
