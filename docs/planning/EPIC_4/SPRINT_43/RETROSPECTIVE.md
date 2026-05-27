# Sprint 43 Retrospective

**Sprint:** 43 — Graph / ND Subsystem Decomposition Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 43 baseline and graph-subsystem scope captured before implementation
- [x] `src/sparse_graph.c` seam inventory refreshed against live code
- [x] bounded Phase-1 graph module boundary design completed
- [x] build/include strategy for multi-file graph wiring completed
- [x] graph ownership / construction extraction landed in live code
- [x] hierarchy / coarsening extraction landed in live code
- [x] residual coarsening/hierarchy audit completed
- [x] bounded coarse-bisection extraction landed in live code
- [x] post-extraction runtime/glue reconciliation completed
- [x] focused graph seam-test design completed
- [x] focused graph seam tests landed
- [x] full validation sweep completed
- [x] Sprint 43 closeout and handoff completed from the measured baseline

## What Went Well

1. **Sprint 43 delivered a real subsystem split, not just monolith cleanup.**
   The sprint moved the graph / ND implementation from one dominant hotspot
   into explicit Phase-1 subsystem files:
   - `src/sparse_graph_core.c`
   - `src/sparse_graph_coarsen.c`
   - `src/sparse_graph_bisect.c`
   - narrowed residual `src/sparse_graph.c`

2. **The extraction order was well chosen.** Sprint 43 avoided trying to split
   the hardest graph regions first. The stable seams were taken in the right
   order:
   - ownership / construction
   - hierarchy / coarsening
   - coarse bisection
   This kept the sprint structural and behavior-preserving instead of turning
   into FM or separator-lifting churn.

3. **The Phase-1 boundary held.** The sprint consistently resisted scope drift:
   - no FM refinement extraction
   - no separator-lifting extraction
   - no fake “finish the whole graph subsystem” push
   - no broad runtime-strategy rewrite
   That made the residual monolith a deliberate later-phase queue rather than
   an uncontrolled leftover.

4. **The new module seams are now protected by direct tests.** Day 12 added
   the right graph-specific seam protections instead of broad new algorithm
   tests:
   - successful `sparse_graph_subgraph(...)`
   - forced `gggp` dispatch on a small graph
   - forced `brute` fallback to GGGP on an oversized graph

5. **The sprint closed from a measured maintained baseline.** Day 13 validated
   both the normal code-change floor and the strongest local reviewed path:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   It also revalidated the direct graph / ND surfaces after the split.

6. **The existing Epic 4 contracts survived the refactor cleanly.**
   Sprint 43 preserved:
   - Sprint 40 validation truthfulness anchors
   - Sprint 41 shared-helper / bounded-refactor discipline
   - Sprint 42 compatibility-preserving internal-first style
   That matters because graph decomposition could easily have drifted into
   broader architectural churn. It did not.

## What Didn't Go Well

1. **The residual monolith is still large and non-trivial.** Sprint 43
   correctly narrowed `src/sparse_graph.c`, but it still owns the most complex
   remaining graph behavior:
   - FM refinement
   - uncoarsening/orchestration glue
   - separator lifting
   - deeper runtime-strategy interactions
   This is expected, but it means Sprint 43 is clearly Phase 1, not the end of
   graph decomposition.

2. **The graph subsystem remains harder to validate than a small local slice.**
   Even bounded extraction batches required confidence across:
   - `test_graph`
   - `test_graph_fm_buckets`
   - `test_reorder_nd`
   - `test_reorder_amd_qg`
   - the full reviewed wrapper paths
   The sprint handled that correctly, but graph/ND refactors continue to have a
   higher validation burden than ordinary local cleanup.

3. **Sprint 43 deliberately deferred the most entangled graph logic.** That
   was the correct scope choice, but it means later graph phases still need to
   tackle the parts of the subsystem where ownership and orchestration are most
   interwoven.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 43 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Sprint 43 artifact package

| Metric | Sprint 43 close state |
|---|---:|
| total artifact files under `SPRINT_43/artifacts/` | `15` |
| implementation-focused artifacts (Days 5-10, 12) | `7` |
| validation / closeout artifacts (Days 13-14) | `2` |

### Graph decomposition outputs

| Metric | Sprint 43 close state |
|---|---:|
| new graph implementation modules added | `3` |
| maintained build surfaces updated for graph split | `2` |
| direct graph/ND reruns in Day 13 | `4` |
| focused seam regressions added on Day 12 | `4` |

Notes:

- new graph implementation modules:
  - `src/sparse_graph_core.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
- maintained build surfaces:
  - `Makefile`
  - `CMakeLists.txt`
- Day 12 focused seam regressions:
  - `test_graph_subgraph_argument_validation`
  - `test_graph_subgraph_path_slice`
  - `test_bisect_forced_gggp_small_graph`
  - `test_bisect_forced_brute_large_graph_falls_back_to_gggp`

## Residual Deferred Debt

Sprint 43 was explicitly Phase 1 of graph / ND decomposition. The main open
work it intentionally hands forward is:

- FM refinement extraction
- separator lifting extraction
- deeper runtime-strategy simplification
- further reduction of the residual orchestration layer in `src/sparse_graph.c`

Not carried forward as unresolved Sprint 43 debt:

- missing extracted ownership / construction seam
- missing extracted hierarchy / coarsening seam
- missing extracted coarse-bisection seam
- missing focused graph seam tests
- missing measured validation closeout
- missing build/include wiring for the new graph modules

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-graph-module-boundary-design.md](./artifacts/day3-graph-module-boundary-design.md)
- [day4-build-and-include-strategy-design.md](./artifacts/day4-build-and-include-strategy-design.md)
- [day5-graph-ownership-construction-extraction-batch1.md](./artifacts/day5-graph-ownership-construction-extraction-batch1.md)
- [day6-hierarchy-coarsening-extraction-batch1.md](./artifacts/day6-hierarchy-coarsening-extraction-batch1.md)
- [day8-hierarchy-coarsening-extraction-batch2.md](./artifacts/day8-hierarchy-coarsening-extraction-batch2.md)
- [day9-coarse-bisection-extraction-batch1.md](./artifacts/day9-coarse-bisection-extraction-batch1.md)
- [day10-runtime-strategy-and-glue-reconciliation.md](./artifacts/day10-runtime-strategy-and-glue-reconciliation.md)
- [day12-focused-graph-seam-tests.md](./artifacts/day12-focused-graph-seam-tests.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 43 achieved its goal:

- Epic 4 now has a first real graph / ND subsystem decomposition
- graph ownership / construction no longer lives only in the original monolith
- hierarchy / coarsening no longer lives only in the original monolith
- coarse bisection no longer lives only in the original monolith
- the remaining `src/sparse_graph.c` is narrower and more honest about what it
  still owns
- the new Phase-1 boundaries are pinned by direct seam tests
- the sprint closed from a measured maintained validation baseline

Later graph phases can now work from explicit subsystem files instead of
reopening whether the first decomposition seam exists at all.
