# Sprint 43 Day 14: Closeout and Handoff

## Summary

Sprint 43 closes with a validated Phase-1 decomposition of the graph / ND
subsystem.

The sprint now hands off:

- an extracted graph ownership / construction seam
- an extracted hierarchy / coarsening seam
- an extracted coarse-bisection seam
- updated shared graph-internal wiring
- focused seam tests for the extracted boundaries
- a measured Day 13 validation baseline

This is a real structural handoff, not just a collection of local cleanups.

## What Sprint 43 Accomplished

### 1. Extracted graph ownership / construction

The graph creation / ownership slice now lives in:

- `src/sparse_graph_core.c`

This removed the core graph object lifecycle from exclusive dependence on the
original monolith.

### 2. Extracted hierarchy / coarsening

The multilevel hierarchy and coarsening seam now lives in:

- `src/sparse_graph_coarsen.c`

This gives later graph work a real coarsening-focused subsystem file rather
than forcing every follow-on change back through `src/sparse_graph.c`.

### 3. Extracted coarse bisection

The coarse-level bisection seam now lives in:

- `src/sparse_graph_bisect.c`

This includes the bounded coarse-dispatch and supporting coarse-level split
logic chosen during Sprint 43 design.

### 4. Preserved a coherent residual orchestration layer

The remaining `src/sparse_graph.c` is no longer “the whole graph subsystem.”
It is now much more clearly the residual layer for:

- FM refinement
- uncoarsening/orchestration glue
- separator lifting / final projection
- later runtime-strategy cleanup

That is a better handoff point for later graph phases.

### 5. Added focused seam protection

Sprint 43 also added direct seam tests for:

- successful `sparse_graph_subgraph(...)`
- explicit forced-`gggp` coarse-bisection dispatch
- explicit oversized forced-`brute` fallback to GGGP

Those tests now protect the new extracted boundaries from silent ownership drift
in later work.

## Final Validated Baseline

Sprint 43 closes from the Day 13 measured baseline:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Direct graph / ND reruns also passed:

- `./build/test_graph`
- `./build/test_graph_fm_buckets`
- `./build/test_reorder_nd`
- `./build/test_reorder_amd_qg`

## Residual Queue for Later Graph Phases

Sprint 43 intentionally does **not** finish the full graph decomposition.

The remaining later-phase queue is still:

- FM refinement extraction
- separator lifting extraction
- deeper runtime-strategy simplification

These were deliberate Sprint 43 deferrals, not regressions or newly discovered
cleanup debt.

## `PROJECT_PLAN.md` Check

Sprint 43 did not surface any new deferred work beyond the later graph-phase
queue already present in Epic 4 planning.

No `PROJECT_PLAN.md` update was needed at closeout.

## Bottom Line

Sprint 43 leaves behind the first real decomposition of the graph / ND
subsystem:

- ownership / construction extracted
- hierarchy / coarsening extracted
- coarse bisection extracted
- residual monolith narrowed to later-phase concerns
- focused seam tests added
- validated local reviewed baseline preserved

That is the correct Phase-1 handoff for the later Epic 4 graph work.
