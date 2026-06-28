# Sprint 93 Day 8: Post-Landing Audit & Rerank

## Purpose

Re-rank the remaining runtime and threading work after the Day 7 ND landing
so Sprint 93's second implementation center is chosen from live post-landing
evidence rather than from the original Day 3 runtime map alone.

## Main Result

The Day 7 landing closed the strongest first Sprint 93 contradiction:

- the reviewed ND owner no longer pays the same recursion-side heap churn and
  separator-emission scan cost on every non-leaf frame
- the first touched reviewed runtime seam is no longer the highest-value
  remaining Sprint 93 target
- the broad "runtime" problem now re-reads more as control-model sharpness
  plus later proof/evidence follow-through than as another immediate
  recursion-side algorithmic batch

That changes the ranked remaining runtime map to:

- strongest first target now:
  - runtime-control cleanup centered on the ND policy/env and override
    plumbing in `src/sparse_reorder_nd.c`
- strongest second target now:
  - proof-surface rebalancing only after the touched runtime-control seam is
    bounded cleanly
- strongest third target now:
  - bounded benchmark and runtime-evidence follow-through after the runtime
    model itself is sharper
- strongest support-only but real target now:
  - maintainer and public wording only where later control cleanup or runtime
    evidence truly changes the maintained contract reading

## Why The Rerank Changed

Day 7 materially changed the runtime reading in one important way:

- `src/sparse_reorder_nd.c` now uses one packed scratch layout for side-0,
  side-1, and separator vertices
- the ND driver no longer allocates separate side buffers at each non-leaf
  recursion frame
- the ND driver no longer performs a separate full post-recursion scan over
  `part[]` just to emit separators
- the touched recursion-side overhead reduction landed without reopening:
  - graph-policy tuning
  - threshold tuning
  - proof-owner widening
  - threading or support-surface claims

That means the strongest remaining contradiction is no longer "does the
reviewed ND lane still have obvious repeated recursion-side work?" It is now
"does the touched runtime/threading control model still expose more internal
override and compatibility complexity than the bounded Sprint 93 runtime story
needs?"

## Strongest Remaining Contradiction

The strongest remaining contradiction is now runtime-control sharpness:

- `src/sparse_reorder_nd.c` still carries the main compatibility env parsing
  and override orchestration for the ND runtime lane
- the touched runtime story still depends on a wide set of internal
  policy/override seams:
  - ND profile override
  - ND base-threshold hook
  - graph coarsening override
  - coarsest-bisection override
  - separator-lift override
  - related compatibility env normalization
- that now outranks proof rebalancing because the next proof/evidence pass
  should validate a cleaner touched runtime model rather than preserve a looser
  one

This is now the highest-value next move because:

- it stays inside Sprint 93's planned runtime-control cleanup lane
- it sharpens the repo's bounded threading/runtime interpretation instead of
  widening claims
- it is a more coherent next step than moving benchmark or proof topology work
  ahead of a still-loose touched control seam

## Exact Day 9 Design Center

The exact Day 9 design center is now fixed to:

- `src/sparse_reorder_nd.c`

The strongest directly forced support-only follow-through, only if the Day 9
contract truly forces movement, is:

- `src/sparse_graph_internal.h`
- `src/sparse_reorder_nd_internal.h`
- `tests/test_reorder_nd.c`
- `tests/test_graph.c`
- `benchmarks/bench_reorder.c`

## Explicit Non-Needs After Day 7

Sprint 93 no longer needs:

- a second immediate recursion-side ND runtime reduction batch
- broad graph/FM policy rewriting as the next center
- proof-surface rebalancing before the touched runtime-control seam is bounded
- benchmark/reporting widening before the touched runtime model is sharper
- support-surface wording churn detached from real runtime-control movement

## Exit State

- The strongest remaining Sprint 93 seam is now explicit after the first ND
  runtime landing.
- The second implementation center stays code-owned and is fixed to
  runtime-control cleanup on the touched ND owner.
- Day 9 can now define one exact bounded runtime-control cleanup contract from
  the live post-Day-7 tree.
