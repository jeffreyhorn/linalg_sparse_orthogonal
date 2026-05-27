# Sprint 43 Day 4 Artifact: Build and Include Strategy Design

## Purpose

Define how Sprint 43 Phase-1 graph extraction should update the maintained
build surfaces and internal include boundaries before code movement begins.

## Build-System Strategy

### Current build shape

Both maintained build surfaces already model the library through explicit source
lists:

- `Makefile`
  - `LIB_SRCS = ...`
- `CMakeLists.txt`
  - `add_library(sparse_lu_ortho STATIC ...)`

### Phase-1 rule

The Sprint 43 extraction batches should update those explicit source lists in
lockstep by:

- adding `src/sparse_graph_core.c`
- adding `src/sparse_graph_coarsen.c`
- adding `src/sparse_graph_bisect.c`
- retaining the remaining `src/sparse_graph.c`

### What Sprint 43 should not do

- no globbing or wildcard source discovery
- no graph-only sublibrary split
- no generated source registration
- no special test-target family for graph extraction

## Shared Header Strategy

### Keep as the main shared graph contract surface

Phase 1 should keep:

- `src/sparse_graph_internal.h`

as the main shared internal graph header.

It should remain the declaration home for:

- `sparse_graph_t`
- `sparse_graph_hierarchy_t`
- graph construction / free / subgraph declarations
- coarsening declarations
- hierarchy declarations
- coarse-bisection declarations
- top-level partition declarations

### Why this is the right Phase-1 choice

- it is already the current graph/ND shared internal contract surface
- it is already consumed by:
  - `src/sparse_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
- it avoids premature header-tree fragmentation while the file split is still
  settling

## FM Header Strategy

Keep:

- `src/sparse_graph_fm_buckets.h`

as a separate narrow header for the FM bucket-array seam.

Phase-1 rule:

- do not merge FM bucket declarations into `src/sparse_graph_internal.h`
- do not make the new extracted Phase-1 modules depend on FM bucket internals
- keep FM bucket tests consuming the narrow FM header directly

This preserves the explicit defer boundary around the later FM extraction work.

## Shared vs Local Declaration Rule

### Put declarations in `src/sparse_graph_internal.h` when:

they are needed by more than one stable consumer across:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- remaining `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- graph-focused tests already using internal graph APIs

### Keep declarations translation-unit local when:

- they are helper-only comparators or tiny support structs
- they are parser enums/helpers used by just one file
- they are FM-only thread-local controls
- they are separator-lift-only scoring helpers
- they are one-off local scoring/shuffle helpers with no stable external
  consumer

## Include and Dependency Risks

### Known high-signal dependency edges

- `src/sparse_reorder_nd.c` -> `src/sparse_graph_internal.h`
- `tests/test_graph.c` -> `src/sparse_graph_internal.h`
- `tests/test_reorder_nd.c` -> `src/sparse_graph_internal.h`
- `tests/test_graph_fm_buckets.c` -> `src/sparse_graph_fm_buckets.h`

### Phase-1 implication

- extracted modules should include the shared graph-internal header
- ND and graph tests should keep consuming the same shared graph contract
  surface
- FM bucket tests should stay isolated on the narrow FM header

### Practical risk checklist

- avoid moving declarations into implementation files if ND/tests still need
  them
- avoid pushing FM-only declarations into the shared graph contract surface
- avoid exposing local-only parser or helper types across files without a real
  stable need

## Test Wiring Strategy

The existing graph-focused test targets are already sufficient as the Phase-1
build topology:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

Phase-1 rule:

- do not create new graph-only test target classes
- extend existing graph-focused binaries when later seam tests are needed
- keep build-system target topology unchanged unless a future sprint has a
  stronger reason to alter it

## Wiring Checklist for Day 5+

When code extraction begins, each batch should follow this order:

1. move code into the target implementation unit
2. add the new source file to `Makefile` and `CMakeLists.txt`
3. add shared declarations to `src/sparse_graph_internal.h` only if they have
   multiple stable consumers
4. keep helper-only or phase-local declarations translation-unit local
5. keep FM bucket declarations isolated in `src/sparse_graph_fm_buckets.h`
6. leave graph test target topology unchanged unless the touched batch proves a
   real need

## Day 4 Bottom Line

Sprint 43 Phase 1 does not have a build-model problem. It has a controlled
source-list and declaration-placement problem.

The stable strategy is:

- explicit source-list expansion in both build systems
- `src/sparse_graph_internal.h` remains the shared graph contract surface
- `src/sparse_graph_fm_buckets.h` remains a narrow FM-only seam
- shared declarations only when they have multiple stable consumers
- local helpers stay local

That is enough wiring discipline for the Phase-1 extraction batches to proceed
without reopening architecture or build-topology scope.
