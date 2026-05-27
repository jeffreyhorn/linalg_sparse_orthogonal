# Sprint 43 Day 3 Artifact: Graph Module Boundary Design

## Purpose

Define the concrete Phase-1 file split for the graph / ND subsystem before any
code extraction begins.

## Phase-1 File Layout

The target Phase-1 decomposition is:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- remaining `src/sparse_graph.c`

This is intentionally a small first-wave split. It extracts the strongest
stable seams without pretending Sprint 43 will erase the entire graph monolith
in one pass.

## Owned Responsibilities by File

### 1. `src/sparse_graph_core.c`

Owns:

- `sparse_graph_from_sparse(...)`
- `sparse_graph_free(...)`
- `sparse_graph_subgraph(...)`
- graph representation construction/teardown invariants

Why this seam is stable:

- it is already a bounded ownership surface
- it has low direct dependence on FM and separator-lifting internals
- it provides a clean foundation for the rest of the subsystem

### 2. `src/sparse_graph_coarsen.c`

Owns:

- `graph_coarsen_with_strategy(...)`
- `graph_coarsen_heavy_edge_matching(...)`
- `graph_coarsen_hcc(...)`
- `parse_coarsening_strategy(...)`
- `sparse_graph_hierarchy_build(...)`
- `sparse_graph_hierarchy_free(...)`
- hierarchy/coarse-graph ownership transitions

Why this seam is stable:

- matching/coarsening and hierarchy lifecycle already form one algorithm/ownership
  family
- `cmap` and coarse-graph ownership naturally belong here
- the seam is largely independent of FM and separator-lifting internals

### 3. `src/sparse_graph_bisect.c`

Owns:

- `bisect_brute_force(...)`
- `bisect_gggp(...)`
- `graph_build_laplacian(...)`
- `graph_bisect_coarsest_spectral(...)`
- `graph_bisect_coarsest(...)`
- `parse_coarsest_bisect_strategy(...)`

Why this seam is stable:

- coarse-level partition initialization is already a bounded algorithm family
- spectral support logic belongs with coarse bisection, not with FM or separator
  lifting
- the parser is tightly owned by this algorithm family

### 4. Remaining `src/sparse_graph.c`

Retains in Phase 1:

- FM bucket-array implementation and FM thread-local controls
- `graph_refine_fm(...)`
- `graph_uncoarsen(...)`
- separator-lifting helpers
- `graph_edge_separator_to_vertex_separator(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- retry/fallback orchestration and other cross-phase runtime glue

Why it remains:

- FM refinement and separator lifting are explicit later-phase subsystem targets
- uncoarsening and public partition orchestration still depend on both of those
  later seams
- keeping them together preserves scope discipline for Sprint 43 Phase 1

## Shared Header Strategy

Phase 1 should keep `src/sparse_graph_internal.h` as the main shared internal
contract surface.

It should own declarations needed across the extracted files for:

- `sparse_graph_t`
- `sparse_graph_hierarchy_t`
- graph construction / free / subgraph API
- coarsening API
- hierarchy API
- coarse-bisection API
- top-level partition API

Phase 1 should **not** introduce a broad new header tree unless Day 4 finds a
real build/include need for it.

## Translation-Unit-Local Boundaries

Keep local to individual implementation files in Phase 1:

- helper-only comparator structs
- local scoring/helper enums not required cross-file
- FM-only thread-local controls
- separator-lifting-only scoring helpers
- other support routines with no stable cross-file consumer

This keeps Phase 1 from overexposing unstable internals just because the file
count increases.

## Parser Ownership Rule

Move with extracted module when the parser is tightly local to that seam:

- `parse_coarsening_strategy(...)` -> `sparse_graph_coarsen.c`
- `parse_coarsest_bisect_strategy(...)` -> `sparse_graph_bisect.c`

Keep in the remaining monolith when the parser is still tied to later-phase
orchestration:

- FM strategy parsers
- separator-lift strategy parsers

This preserves real ownership instead of forcing artificial symmetry.

## Keep-in-Monolith vs Extract-Now

### Extract now

- graph ownership / construction
- hierarchy / coarsening
- coarse bisection

### Keep in monolith for Phase 1

- FM refinement
- separator lifting
- top-level uncoarsening / public partition orchestration
- broader runtime strategy glue spanning multiple later seams

## Naming Direction

Chosen names:

- `sparse_graph_core.c`
- `sparse_graph_coarsen.c`
- `sparse_graph_bisect.c`

These names are:

- behavior-oriented
- stable after the sprint
- narrow enough to leave room for later FM/separator extraction

Rejected naming directions:

- sprint-history names
- generic “helpers” names
- premature Phase-2 naming that assumes later seams are already extracted

## Day 3 Bottom Line

Sprint 43 Phase 1 now has a concrete decomposition target:

- `sparse_graph_core.c`
- `sparse_graph_coarsen.c`
- `sparse_graph_bisect.c`
- smaller remaining `sparse_graph.c` for FM, separator lifting, and
  orchestration

That is the right first-wave layout: it extracts the three strongest stable
seams and leaves the more entangled FM/separator/orchestration work explicitly
for later phases.
