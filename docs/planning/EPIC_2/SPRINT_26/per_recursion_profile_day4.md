# Sprint 26 Day 4 — Per-Recursion-Level Partition Profile Analysis

## Headline finding

**Cost concentrates at depths 6-9 (the smaller-subgraph end of the
recursion), not at the root partition.**  This INVALIDATES the
PLAN.md Day 4 task 5 hypothesis ("cost is concentrated at depths 0-2
(the largest subgraphs); leaf-AMD splice region near depth
log2(n / nd_base_threshold) is essentially free").

Per-depth breakdown averaged across 5 Pres_Poisson runs:

| depths | calls | avg total ms | % of partition |
|---|---|---|---|
| 0-2 (root + first two splits) | 7 | 818 | **1.9 %** |
| 3-5 (intermediate) | 56 | 4 422 | 10.2 % |
| **6-9 (near-leaf, small subgraphs)** | **238** | **38 116** | **87.9 %** |

The hypothesis was inverted: depths 0-2 are *cheapest* (1.9 % of
partition cost, despite being on the largest subgraphs), and
depths 6-9 dominate (87.9 %, on subgraphs of n ≈ 50-300 each).

## Per-depth detail (5-run average)

| depth | calls | avg ms | avg ms/call | run-to-run variance | % of partition |
|---|---|---|---|---|---|
| 0 | 1 | 285 | 285 | 11 % | 0.7 % |
| 1 | 2 | 274 | 137 | 13 % | 0.6 % |
| 2 | 4 | 260 | 65 | 11 % | 0.6 % |
| 3 | 8 | 1 254 | 157 | 12 % | 2.9 % |
| 4 | 16 | 1 195 | 75 | 11 % | 2.8 % |
| 5 | 32 | 1 973 | 62 | 13 % | 4.6 % |
| 6 | 64 | 8 033 | 126 | 12 % | 18.7 % |
| **7** | **107** | **16 877** | **158** | **13 %** | **39.3 %** ← peak |
| 8 | 62 | 11 010 | 178 | 12 % | 25.6 % |
| 9 | 5 | 2 196 | 439 | 14 % | 5.1 % |

**Total partition (5-run average):** 43 356 ms = 99.5 % of wall
(matches Sprint 25 Day 11's per-phase finding).

Call counts are deterministic across runs (same recursion structure
on Pres_Poisson every time).  Per-depth wall-time variance tracks
the aggregate 16 % within-run variance Sprint 25 Day 11
characterised as macOS arm64 thermal management on the partition
phase.

## Why depths 6-9 dominate despite per-call cost dropping

Per-call cost decreases with depth as expected (smaller subgraphs
cost less): 285 ms at depth 0, 65 ms at depth 2, 62 ms at depth 5.
But CALL COUNT grows roughly geometrically until a peak, then drops
as the recursion reaches the base-threshold cutoff:

```
call counts:    1, 2, 4, 8, 16, 32, 64, 107, 62, 5
                    geometric growth     ↑ peak  drop-off
```

The peak at depth 7 (107 calls) reflects two things: (a) the
recursion's binary-tree structure × Pres_Poisson's ND favourable
partitioning (cuts produce two roughly-balanced halves that both
recurse), and (b) the separator vertices peeling off at each level
slowing the "halving" relative to a pure binary tree (so depth 7
has more calls than the pure-binary 128 = 2^7 would predict — wait,
107 < 128, actually it's slightly LESS, so the separator-peel
effect REDUCES the per-depth call count).

The drop-off at depth 8-9 (62, 5 calls) reflects the
`sparse_reorder_nd_base_threshold = 32` cutoff: subgraphs of
n ≤ 32 go to leaf-AMD splice instead of recursing, so the deepest
recursive partition calls are on subgraphs of n ~ 33-100.

**Avg per-call cost at depths 6-9 (126-439 ms) is HIGHER than at
depths 2-5 (62-157 ms)** despite smaller subgraph sizes.  This is
the surprising part.  Hypothesis: partition cost scales not with
n but with the multilevel pipeline's overhead floor — coarsening +
GGGP-or-brute-force bisection + uncoarsening + FM at every level.
For tiny subgraphs (n ~ 50-300), the constant-factor overhead per
multilevel-pipeline call dominates; the coarsening levels don't
amortize their setup.  This is a structural property of the
multilevel partitioner, not specific to FM passes.

(Day 5's `nd_base_threshold` re-sweep will measure whether raising
the threshold past 32 reduces this overhead floor.)

## Implications for Item 5 (FINEST FM) sub-axis selection

PLAN.md's three sub-axis candidates from Sprint 25 RETROSPECTIVE.md
"Sprint 26 inputs" #1:

- **(a) Annealing acceptance**: accept worsening moves with
  decreasing probability over passes (analogous to simulated
  annealing) to escape local minima.
- **(b) Different bucket-tie-break**: currently FIFO within bucket;
  try LIFO or random with seeded RNG.
- **(c) Thick-restart-style FM with global rollback**: track best
  cut across all passes; allow each pass to re-explore from that
  anchor with random perturbation.

### Cost analysis vs the per-depth profile

The FINEST FM is invoked once per partition call, at the finest
uncoarsening level of each multilevel pipeline run.  Cost
multipliers:

- **Sub-axis (a) annealing**: same per-pass cost as baseline; no
  per-call wall expansion.  Possibly +20-50 % wall via more passes
  if annealing benefits from a longer schedule.  Manageable.
- **Sub-axis (b) bucket-tie-break**: zero runtime cost — same
  number of operations, just different order.  Free.  The cleanest
  experiment to try first.
- **Sub-axis (c) thick-restart with global rollback**: 2-3× per-pass
  cost (each pass re-runs from the saved anchor).  At depths 6-9
  with 87.9 % of cost concentrating, this would expand total wall
  to ~80-120 s.  **Likely too expensive** given Day 11's wall-check
  baseline of 47 055 ms with a 1.5× = 70 583 ms ceiling — sub-axis
  (c) at 2× would push past the gate.

### Recommendation: sub-axis (b) bucket-tie-break

**Pick sub-axis (b) bucket-tie-break for Day 6's FINEST FM
implementation.**  Reasoning:

1. **Cost-free runtime**: same per-call work as baseline; the
   wall-check 1.5× ceiling is preserved.  Days 6-8 can experiment
   freely without risking the gate.
2. **High leverage at depths 6-9**: with 169 partition calls in
   that range each invoking the FINEST FM, a non-FIFO tie-break
   gets compounded across many independent FM invocations.  If FIFO
   consistently picks the same wrong tie-break per pass, switching
   to LIFO or seeded random gives the FM cascade 169 different
   exploration directions — much more entropy than annealing's
   single per-call schedule could produce.
3. **Sprint 25 Day 5's saturation finding ("passes ≥ 5 saturate at
   0.952×") suggests the FIFO order is the local-minimum trap**:
   each successive pass picks the same tie-break and converges to
   the same cut, regardless of how many passes run.  If true, a
   different tie-break unlocks new cuts that more passes alone
   couldn't find.

### Day 6 design inputs

Sub-axis (b) implementation plan for Day 6:

- Add `SPARSE_FM_FINEST_TIEBREAK={fifo,lifo,random}` env var (or
  fold into `SPARSE_FM_FINEST_STRATEGY={baseline,bucket_tiebreak}`
  per PLAN.md).
- Default `fifo` (Sprint 23 baseline) preserves bit-identical
  behavior when off.
- `lifo`: pop from the bucket linked list's tail instead of head.
  Trivial code change in `src/sparse_graph_fm_buckets.h`'s pop
  routine.
- `random`: seeded RNG selects which bucket entry to pop.  Slightly
  more code; needs to thread the seed through.
- Day 7 implements one of the two; Day 8 sweeps + decides which to
  ship + whether to flip default.

### What about sub-axes (a) and (c)?

If sub-axis (b) saturates without closing Pres_Poisson to ≤ 0.85×,
sub-axis (a) annealing is the next candidate (manageable cost,
moderate complexity).  Sub-axis (c) thick-restart is the
expensive-fall-back; recommended only if (a) and (b) both fall
short and the wall budget can be expanded.

### Day 5 (`nd_base_threshold`) interaction

Day 5's `nd_base_threshold` sweep raises the threshold from 32 →
{48, 64, 96, 128}, which would shift the cost concentration:
fewer depth-7-9 partition calls (because more subgraphs become
leaf-AMD), more depth-6-7 calls on slightly larger subgraphs, and
more leaf-AMD work.  The Day-5 finding may modify Day-6's sub-axis
weighting — if depth 7's 39.3 % share collapses under
threshold = 64, the design pressure shifts back toward depths 0-5.
But the sub-axis (b) bucket-tie-break recommendation is robust:
zero runtime cost means it's a clean experiment regardless of
threshold flip.

## What we learned from Day 4

1. **Cost is at the SMALLER subgraphs, not the root.**  Future ND
   wall-time work should target the depth-6-9 partition calls where
   87.9 % of wall lives.

2. **Per-call cost has a constant-factor floor.**  Tiny subgraphs
   (n ~ 50-300) cost 60-200 ms each in the multilevel pipeline —
   most of which is pipeline-setup overhead (coarsen + bisect +
   uncoarsen) rather than per-vertex work.  Day 5 may shift this
   by raising `nd_base_threshold`; future work could also bypass
   the multilevel path at small n (direct GGGP without coarsening).

3. **Item 5 sub-axis (b) bucket-tie-break is the highest-leverage,
   lowest-risk experiment**.  Day 6 picks it.

4. **The PLAN's hypothesis was wrong**, in a useful way.  The
   per-depth profile let Day 4 invert the hypothesis cleanly +
   re-target Item 5's sub-axis selection.  Without per-depth
   instrumentation, Day 6 would have built around the wrong
   assumption.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 4 + Day 5 + Day 6
- `docs/planning/EPIC_2/SPRINT_26/profile_day4_per_depth.txt` —
  the 5-run capture this analysis is based on
- `docs/planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt` —
  Sprint 25 Day 11's per-phase profile (the cumulative-only version
  this Day 4 work extends)
- `docs/planning/EPIC_2/SPRINT_25/RETROSPECTIVE.md` "Sprint 26
  inputs" #1 — the three sub-axis candidates Day 4 just down-selected
- `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` —
  the saturation-at-3-passes finding that motivates the bucket-tie-
  break hypothesis
- `src/sparse_reorder_nd.c` — Day 4's per-depth instrumentation
- `src/sparse_graph_fm_buckets.h` — Day 6's bucket-tie-break
  implementation target
