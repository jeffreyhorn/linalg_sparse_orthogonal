# Sprint 86 Day 1: Scope and Reviewed Runtime Baseline

## Purpose

Turn the Sprint 86 project-plan section and the Sprint 85 validated closeout
into one bounded reviewed-runtime execution package before any reorder- or
runtime-aware code lands.

## Starting Truth

Sprint 86 begins from a validated Sprint 85 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 85 already moved the strongest prior contradiction:

- one bounded iterative-source ownership cleanup landed
- one bounded direct-family ownership move landed
- one bounded giant-test registration cleanup landed

That means Sprint 86 can start from the next real Epic 8 contradiction center:

- the current reviewed runtime and reorder / ND scalability ceiling on the
  highest-value touched proof-owner and implementation lanes

## Sprint 86 Workstreams

The highest-value Sprint 86 package is now fixed explicitly around:

- reviewed runtime audit
- algorithm / proof runtime design
- ND runtime reduction
- proof-surface rebalancing
- benchmark / comparison follow-through
- CI / reviewed-path alignment
- validation and closeout

## Strongest Runtime Starting Point

The validated Sprint 85 close state already fixed the strongest runtime
starting truth:

- reviewed CMake `Total Test time (real)` = `404.15 sec`
- reviewed `test_reorder_nd` time = `283.53 sec`

Sprint 86 therefore does not begin from a generic whole-suite slowdown. It
begins from one dominant reviewed long pole concentrated on the reorder / ND
proof lane.

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 86 surfaces:

- reorder reviewed proof owners:
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder.c`
  - `tests/test_reorder_amd_qg.c`
- reorder and ND implementation owners:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
- measurement and support surfaces:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Preserved Fence

Sprint 86 is explicitly bounded against:

- reopening Sprint 85’s source-decomposition package as the first
  implementation center
- weakening correctness proof quality to buy runtime wins
- benchmark-governance or example-governance drift into correctness ownership
- broad package/platform maturity widening
- support-surface churn detached from a real landed runtime seam

## Day 1 Result

Sprint 86 now starts from one precise reviewed-runtime execution package
rather than from a generic “make tests faster” bucket. The strongest likely
touch surfaces, preserved non-goals, and validated/runtime baseline are fixed
in writing before the validation/proof recheck begins.
