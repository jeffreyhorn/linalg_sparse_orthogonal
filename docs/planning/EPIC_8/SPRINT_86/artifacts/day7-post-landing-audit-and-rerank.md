# Sprint 86 Day 7: Post-Landing Runtime Audit and Rerank

## Purpose

Re-rank the remaining Sprint 86 contradiction map after the Day 6 ND
runtime-reduction landing.

## Main Result

The Day 6 landing closed the strongest first Sprint 86 contradiction:

- `src/sparse_reorder_nd.c` no longer stands out as the clear next landing
  center
- the repo now has one real bounded ND runtime/scalability seam landed
- a second immediate algorithm-first ND batch is not the highest-value next
  move

The strongest remaining Sprint 86 seam is now reviewed-surface
concentration.

## Exact Next Center

The exact Day 8 design center is now fixed to:

- `tests/test_reorder_nd.c`

The key post-Day-6 runtime reading is explicit:

- reviewed `test_reorder_nd` improved from `283.53 sec` to `138.68 sec`
- reviewed CMake total improved from `404.15 sec` to `234.05 sec`
- despite that win, `test_reorder_nd` still accounts for roughly `59%` of the
  reviewed CMake total

That means the remaining contradiction is no longer primarily unresolved ND
policy. It is concentrated reviewed proof ownership.

## Post-Day-6 Hotspot Context

Post-Day-6 live hotspot map:

- `tests/test_reorder_nd.c` = `2288` lines
- `tests/test_graph.c` = `2925` lines
- `tests/test_reorder.c` = `1082` lines
- `src/sparse_reorder_nd.c` = `771` lines
- `src/sparse_graph.c` = `841` lines
- `src/sparse_graph_coarsen.c` = `659` lines
- `src/sparse_graph_bisect.c` = `528` lines
- `src/sparse_graph_refine.c` = `602` lines
- `src/sparse_graph_separator.c` = `297` lines
- `benchmarks/bench_reorder.c` = `322` lines
- `benchmarks/bench_fillin.c` = `178` lines

The useful distinction is no longer raw size alone. It is that the first
algorithmic/runtime contradiction has already been reduced in code, while the
reviewed proof-owner concentration still has not.

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_graph.c`
- `tests/test_reorder.c`
- `docs/maintainer_guide.md`
- `README.md`

Current reading:

- `tests/test_graph.c` and `tests/test_reorder.c` remain adjacent proof owners
  but do not become the next landing center unless the Day 8 design truly
  forces shared fixture or rerun movement
- `docs/maintainer_guide.md` and `README.md` remain truthful and should stay
  deferred unless the next batch changes reviewed rerun or maintenance
  guidance

## Preserved Non-Touch Map

The useful Day 7 clarification is explicit now:

- no second immediate ND-policy retuning batch as the next center
- no graph-pipeline rewrite before the proof-owner concentration is designed
- no early benchmark/comparison batch before the reviewed proof surface is
  rebalanced
- no CI/reviewed-path wording movement before a real reviewed-surface seam
  lands

## Strongest Clarification

Sprint 86's next contradiction center is no longer “do more ND algorithm work
because the first runtime batch succeeded.”

It is also not “jump straight to benchmark or CI follow-through because the
runtime curve already moved.”

It is the remaining reviewed proof-owner concentration on the reordered ND
lane.

That fixes the ordering:

- next seam = proof-surface rebalancing
- later seam = benchmark/comparison follow-through
- later seam = CI/reviewed-path alignment
- later only if newly justified = another algorithmic ND or graph-family
  runtime batch

## Validation

This was a docs-only rerank day, so no build/test rerun was required.

The rerank was grounded in direct rereads and live post-Day-6 measurement of:

- `tests/test_reorder_nd.c`
- `tests/test_graph.c`
- `tests/test_reorder.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_fillin.c`

## Exit State

- Sprint 86 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one proof-surface design lane centered on
  `tests/test_reorder_nd.c`.
- Benchmark/comparison follow-through and CI/reviewed-path alignment remain
  clearly separated from the real next implementation move.
