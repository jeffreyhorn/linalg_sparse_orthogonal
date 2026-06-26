# Sprint 93 Day 6: ND Runtime Reduction Design

## Purpose

Define the bounded implementation contract for the touched ND recursive runtime
seam so Sprint 93 can land one real reviewed-runtime reduction without
reopening broad graph policy, threading, or benchmark-governance work.

## Main Result

The exact Sprint 93 first implementation center is now fixed to:

- `src/sparse_reorder_nd.c`

The exact runtime reduction target is now fixed to:

- remove avoidable recursion-side work inside the ND driver before widening
  any graph-policy or threading story
- prioritize repeated non-leaf overhead that is paid across the reviewed ND
  recursion:
  - temporary side-set collection and repeated full-array passes
  - avoidable heap churn around partition-side bookkeeping
  - recursion-local work that does not change the final permutation or policy
    reading

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- Sprint 93 should first reduce repeated ND driver overhead, not redesign ND
  policy
- it should preserve the current tuned threshold and current policy surface
  unless the touched recursion-side reduction proves impossible otherwise
- it should read success as a smaller reviewed ND runtime cost with the same
  ordering semantics, not as a broader graph-quality or threading claim

## Preserved Invariants

The first landing now preserves these invariants explicitly:

- permutation contract remains `perm[new_i] = old_i`
- current ND policy/env interpretation remains unchanged
- leaf-vs-non-leaf routing at the current threshold remains unchanged
- separator-last ordering remains unchanged
- touched reviewed proof owners remain deterministic under repeated runs

## Exact Forced Owners

The strongest directly forced proof and evidence owners are now fixed to:

- `tests/test_reorder_nd.c`
- `benchmarks/bench_reorder.c`

The strongest adjacent owners remain support-only unless the first landing
truly forces movement:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `tests/test_graph.c`

## Deferred From The First Runtime Batch

The first batch now explicitly defers:

- leaf-AMD semantics as a product-level redesign
- FM/coarsening policy changes in `src/sparse_graph.c`
- broad thread-local override cleanup
- new public runtime knobs
- detached benchmark-only tuning

## Exit State

- Sprint 93 has one explicit ND runtime-reduction implementation contract.
- Day 7 can land the touched recursion-side runtime batch without reopening
  broad graph policy, runtime-control, or benchmark-governance work.
