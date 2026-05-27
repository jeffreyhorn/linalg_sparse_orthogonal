# Sprint 43 Day 2 Artifact: Monolith Seam Refresh Inventory

## Purpose

Refresh the live seam inventory inside `src/sparse_graph.c` so Sprint 43's
Phase-1 extraction order is based on the current code, not only on the sprint
plan headings.

## Current Monolith Size

- `src/sparse_graph.c` = `3555` lines

This still matches Sprint 40's hotspot baseline and keeps graph decomposition
as the highest-value source-level structural refactor target.

## Live Seam Map

The current monolith reduces cleanly to seven main regions:

1. **Graph construction / ownership**
   - `sparse_graph_from_sparse(...)`
   - `sparse_graph_free(...)`
   - `sparse_graph_subgraph(...)`
2. **Hierarchy and coarse-graph lifecycle**
   - `sparse_graph_hierarchy_t`
   - `sparse_graph_hierarchy_build(...)`
   - `sparse_graph_hierarchy_free(...)`
3. **Matching / coarsening**
   - `graph_coarsen_with_strategy(...)`
   - `graph_coarsen_heavy_edge_matching(...)`
   - `graph_coarsen_hcc(...)`
   - coarsening strategy parsing/support helpers
4. **Coarse bisection**
   - `bisect_brute_force(...)`
   - `bisect_gggp(...)`
   - `graph_build_laplacian(...)`
   - `graph_bisect_coarsest_spectral(...)`
   - `graph_bisect_coarsest(...)`
5. **FM refinement**
   - FM bucket-array implementation
   - annealing / thick-restart / ensemble controls
   - `graph_refine_fm(...)`
6. **Separator lifting / final partition projection**
   - separator-lift strategy parsing and scoring
   - `graph_edge_separator_to_vertex_separator(...)`
7. **Top-level orchestration / runtime strategy glue**
   - `graph_uncoarsen(...)`
   - `partition_once(...)`
   - `sparse_graph_partition(...)`

## Stable Phase-1 Extraction Candidates

### Tier 1: extract now

The strongest stable Phase-1 seams are:

- graph ownership / construction
- hierarchy / coarsening
- coarse bisection

These seams are strong because they already have:

- coherent local helper families
- bounded ownership/state transitions
- less direct coupling to the later FM and separator phases than the rest of
  the file

### Tier 2: keep in remaining orchestration layer for now

- top-level orchestration
- cross-cutting runtime strategy glue

These should stay mostly local during Phase 1 except where a parser/helper is
obviously owned by a newly extracted module.

### Tier 3: explicit later-phase defer

- FM refinement
- separator lifting

These are real subsystem seams, but they are still too entangled with each
other and with the top-level flow to be safe first-wave extraction targets.

## Cross-Cutting State and Coupling Risks

The main cross-cutting state that currently travels through too many phases:

- thread-local FM strategy controls
- parsed strategy-selection behavior across coarsening, bisection, FM, and
  separator lifting
- hierarchy-level `cmap` ownership and coarse/fine graph transitions
- finest-level retry and orchestration logic

Implication:

- Sprint 43 should extract modules by stable ownership seams first
- broader strategy unification remains a later concern

## Current Header Surface

The current graph-internal header surface is:

- `src/sparse_graph_internal.h`
  - graph representation
  - constructor/free/subgraph API
  - hierarchy API
  - coarsening API
  - coarse-bisection API
  - top-level partition API
- `src/sparse_graph_fm_buckets.h`
  - FM bucket-array API

Day 2 implication:

- the repo already has a partial shared internal graph surface
- Day 3 / Day 4 should decide whether Phase-1 extraction keeps expanding
  `sparse_graph_internal.h` or introduces narrower dedicated internal headers

## Focused Test Surface at Start of Sprint 43

The current graph-focused regression surface is concentrated in:

- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

This is enough baseline surface to support Phase-1 extraction, but later sprint
days should still add seam-specific checks where new module boundaries justify
them.

## Initial Extraction Order

Day 2's recommended Sprint 43 Phase-1 landing order:

1. graph ownership / construction
2. hierarchy / coarsening
3. coarse bisection
4. top-level orchestration reconciliation after the above

Explicit defer order:

1. FM refinement extraction
2. separator lifting extraction
3. deeper runtime strategy simplification

## Day 2 Bottom Line

The graph monolith is large, but it is no longer structurally ambiguous once
the live seams are named directly. Sprint 43 Phase 1 should center on three
stable module targets:

- graph ownership / construction
- hierarchy / coarsening
- coarse bisection

while leaving FM refinement, separator lifting, and the broader runtime-glue
surface explicitly in the later-phase defer class.
