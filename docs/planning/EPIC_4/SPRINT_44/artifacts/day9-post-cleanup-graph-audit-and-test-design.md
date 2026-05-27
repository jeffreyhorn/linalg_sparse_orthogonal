# Sprint 44 Day 9 Artifact: Post-Cleanup Graph Audit and Test Design

## Purpose

Audit the live graph subsystem after the Sprint 44 Phase-2 cleanup so the new
module boundaries are reviewed before the sprint shifts into large-test
maintainability work, then define the smallest useful Day 10 graph-test batch.

## 1. Live Ownership Split

The current graph decomposition is now:

- `src/sparse_graph_core.c`
  - graph construction / ownership
  - induced subgraph creation
- `src/sparse_graph_coarsen.c`
  - HEM/HCC coarsening
  - hierarchy lifecycle
  - forced-HEM override seam
- `src/sparse_graph_bisect.c`
  - coarsest bisection
  - brute/GGGP/spectral support
- `src/sparse_graph_refine.c`
  - FM runtime state
  - FM parser helpers
  - cut-weight evaluation
  - gain buckets
  - `graph_refine_fm(...)`
- `src/sparse_graph_separator.c`
  - separator policy parsers
  - per-vertex separator scoring helpers
  - `graph_edge_separator_to_vertex_separator(...)`
- residual `src/sparse_graph.c`
  - `graph_uncoarsen(...)`
  - top-level partition composition
  - retry/fallback glue

Measured file sizes confirm the residual orchestration layer is now much
smaller than the combined extracted ownership:

- residual `src/sparse_graph.c` = `801` lines
- extracted ownership total:
  - `core = 264`
  - `coarsen = 597`
  - `bisect = 521`
  - `refine = 619`
  - `separator = 311`

## 2. What Is Already Well-Protected

The current test surface already gives good protection to several major seams.

### Core / subgraph

- `test_graph_subgraph_argument_validation`
- `test_graph_subgraph_path_slice`

### Bisection dispatch / fallback

- forced `gggp` dispatch on a small graph
- forced `brute` request falling back to `gggp` on an oversized graph
- spectral fallback coverage on star/small/disconnected fixtures

### FM behavior and FM runtime strategy plumbing

- `test_fm_reduces_checkerboard_cut`
- `test_fm_optimal_partition_no_regress`
- `test_fm_null_args`
- `test_fm_intermediate_passes_smoke`
- FIFO finest-strategy smoke
- thick-restart gain-noise smoke
- ensemble corpus-safety + determinism

### End-to-end ND integration

- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

Interpretation:

- Day 10 does not need another FM-private or bisection-private unit-test wave
- the Phase-2 split already inherited strong behavior-level coverage there

## 3. Strongest Remaining Gap

The main remaining direct gap is the separator-lifting seam.

After extraction, `src/sparse_graph_separator.c` owns:

- separator strategy parsing
- separator weight parsing
- per-vertex boundary scoring
- edge-to-vertex separator conversion

But direct helper-level coverage in `tests/test_graph.c` is still limited to:

- default smaller-side lifting
- null-argument handling

There is broader separator-policy evidence elsewhere:

- `test_per_vertex_fixed_k_differs_from_dynamic_k`
- separator-weight differentiation in `tests/test_reorder_nd.c`

Still, the extracted separator module would benefit from one additional direct
behavior-level contract test that does not depend on its private local helpers.

## 4. What Should Not Be Tested Directly

The residual orchestration layer after Day 8 is smaller, but it should still be
protected through end-to-end behavior rather than through private helper tests.

Avoid on Day 10:

- direct tests of static parser helpers
- tests that pin `graph_uncoarsen(...)` buffer choreography
- tests that assert exact intermediate FM runtime state
- tests that depend on local implementation-only comments or section layout

Reason:

- those would freeze the internal cleanup shape rather than the public
  behavior-level contract Sprint 44 is trying to preserve

## 5. Concrete Day 10 Batch

### Target 1: direct separator-policy contract

File:

- `tests/test_graph.c`

Best shape:

- add one small direct test for `graph_edge_separator_to_vertex_separator(...)`
  under a non-default separator policy
- strongest candidate:
  - `balanced_boundary` on a crafted graph/partition where the policy choice is
    observably different from plain smaller-side lifting while still preserving
    the partition invariant

Why:

- directly protects the extracted separator module
- stays behavior-level
- avoids exposing private parsing/scoring helpers

### Target 2: compact post-split orchestration smoke

File:

- `tests/test_graph.c`

Best shape:

- add one small end-to-end `sparse_graph_partition(...)` smoke under a stable
  non-default configuration that composes the extracted FM and separator
  modules through the residual orchestration path

Example design direction:

- modest grid or mesh fixture
- one non-default FM strategy and/or separator-lift strategy
- assert only structural behavior:
  - valid partition
  - reasonable separator count / balance band
  - no degenerate regression

Why:

- protects the module interaction boundary after Day 8 cleanup
- avoids overfitting to private orchestration details

## 6. Bottom Line

Day 9 shows that Sprint 44’s graph work is now in the right state:

- FM, bisection, and ND integration already have good behavior-level coverage
- the extracted separator policy seam is the clearest remaining direct gap
- residual orchestration should stay protected end-to-end, not via private
  helper tests
- Day 10 can therefore stay small and high-signal instead of turning into a
  second exploratory graph test sweep
