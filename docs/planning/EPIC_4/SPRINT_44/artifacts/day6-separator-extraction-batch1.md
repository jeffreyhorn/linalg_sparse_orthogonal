# Sprint 44 Day 6 Artifact: Separator Extraction Batch 1

## Purpose

Land the bounded Sprint 44 Phase-2 separator extraction by moving separator
policy, per-vertex separator scoring, and the final edge-to-vertex separator
conversion out of the residual graph orchestration file and into its own
implementation unit.

## 1. Extracted Separator Module

Day 6 created:

- `src/sparse_graph_separator.c`

The module now owns:

- separator-lift strategy enums
- separator weight enums
- separator env-var parsers
- per-vertex separator scoring helpers
- `graph_edge_separator_to_vertex_separator(...)`

This matches the Day 4 separator design boundary directly.

## 2. Residual `src/sparse_graph.c` Boundary After the Move

The residual graph file no longer contains separator-local policy code.

It now retains the intended orchestration seam:

- `graph_uncoarsen(...)`
- top-level partition orchestration
- retry / fallback glue

It no longer owns:

- separator strategy enums
- separator weight enums
- separator env-var parsers
- per-vertex separator scoring helpers
- final separator lifting/conversion implementation

## 3. Internal Interface Shape

The shared graph contract stayed intentionally narrow.

Day 6 did **not** expose separator-local enums or parser helpers in
`src/sparse_graph_internal.h`.

The shared internal seam remains the one behavior entry point that other graph
phases actually need:

- `graph_edge_separator_to_vertex_separator(...)`

Interpretation:

- the extraction moved ownership by file without creating broad new shared
  internal surface area

## 4. Build Wiring

The new separator module was added to both maintained build surfaces:

- `Makefile`
- `CMakeLists.txt`

No special-case logic or graph-only build path was needed.

## 5. Scope That Intentionally Stayed Deferred

Day 6 did **not** move:

- `graph_uncoarsen(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- sep=`0` retry / fallback composition

This preserves the planned Sprint 44 mid-sprint order:

1. Day 5: FM extraction
2. Day 6: separator extraction
3. Day 7: residual runtime/orchestration audit
4. Day 8: residual runtime/orchestration cleanup

## 6. Validation

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also re-covered the graph/ND surfaces most
relevant to the extraction:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

## Bottom Line

Day 6 delivered the second real Sprint 44 Phase-2 graph extraction:

- `src/sparse_graph_separator.c` now owns separator policy and lifting
- `src/sparse_graph.c` is materially narrower and more honestly scoped
- the shared internal contract stayed minimal
- the build wiring remained routine
- the full validation gate passed cleanly
