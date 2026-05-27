# Sprint 44 Day 2 Artifact: Residual Graph Seam Refresh Inventory

## Purpose

Refresh the internal seam inventory for the residual `src/sparse_graph.c`
after Sprint 43 so Sprint 44's implementation order is grounded in the live
Phase-2 file rather than in the older full-monolith picture.

## Starting Point

Sprint 43 already extracted:

- graph ownership / construction into `src/sparse_graph_core.c`
- hierarchy / coarsening into `src/sparse_graph_coarsen.c`
- coarse bisection into `src/sparse_graph_bisect.c`

That means the residual `src/sparse_graph.c` is no longer a whole-subsystem
hotspot. It is a narrowed Phase-2 file, and Day 2's job is to identify the
remaining stable seams inside that narrowed file.

## Live Residual Graph Shape

The current size picture is:

- `src/sparse_graph.c` = `2153` lines
- `src/sparse_graph_core.c` = `264`
- `src/sparse_graph_coarsen.c` = `597`
- `src/sparse_graph_bisect.c` = `521`

The residual file now maps cleanly to five implementation classes:

1. FM refinement core
2. uncoarsening and finest-level strategy orchestration
3. separator lifting and separator-policy selection
4. top-level partition orchestration
5. retry / fallback glue

## Residual Seam Classification

### 1. FM refinement core

Main live seam:

- `compute_cut_weight(...)`
- gain-bucket implementation
- FM thread-local strategy controls
- `parse_fm_anneal_schedule(...)`
- `graph_refine_fm(...)`

Classification:

- **stable extract-now seam**

Why:

- the region is already highly cohesive
- the public/internal contract surface is small
- it already has a direct focused test sibling:
  - `tests/test_graph_fm_buckets.c`
- the internal graph header already treats `graph_refine_fm(...)` as a stable
  cross-module behavior seam

### 2. Uncoarsening and finest-level strategy orchestration

Main live seam:

- `graph_uncoarsen(...)`
- finest-pass env parsing
- annealing / thick-restart / ensemble setup
- restart/anchor/strategy dispatch glue

Classification:

- **still coupled; leave in the residual orchestration layer for now**

Why:

- it composes:
  - extracted coarsening hierarchy
  - extracted coarse bisection
  - residual FM refinement
  - residual separator lifting
- it is the bridge that becomes simpler only after the FM and separator moves
  land

### 3. Separator lifting and separator-policy selection

Main live seam:

- separator strategy enums
- `parse_sep_lift_strategy(...)`
- `parse_sep_lift_weight(...)`
- `is_per_vertex_strategy(...)`
- `per_vertex_score_cmp_desc(...)`
- `graph_edge_separator_to_vertex_separator(...)`

Classification:

- **stable extract-now seam**

Why:

- the separator region is already contiguous and behavior-cohesive
- the parser/config logic is separator-owned, not general orchestration-owned
- the internal graph header already exposes only the main behavior seam:
  - `graph_edge_separator_to_vertex_separator(...)`

### 4. Top-level partition orchestration

Main live seam:

- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

Classification:

- **keep residual until FM and separator extraction finish**

Why:

- this is not a self-contained subsystem
- it is the composition layer that sequences:
  - hierarchy build
  - coarsest split
  - FM
  - uncoarsening
  - separator lifting
  - retry/fallback policy

### 5. Retry / fallback glue

Main live seam:

- `graph_partition_should_retry_with_forced_hem(...)`
- forced-HEM override composition in `sparse_graph_partition(...)`

Classification:

- **orchestration-owned residual glue**

Why:

- the override implementation lives in the coarsening module
- the retry decision belongs to the top-level partition composition seam
- this logic should become smaller after the FM and separator moves, not before

## Stable Extract-Now vs Still-Coupled Split

### Stable extract-now seams

- FM refinement core
- separator lifting and separator-policy selection

### Still-coupled seams

- `graph_uncoarsen(...)`
- top-level partition orchestration
- retry / fallback glue

Interpretation:

- Sprint 44 should move the real owned algorithm slices first
- only then should it simplify the composition layer that remains

## Runtime / Config Parsing Ownership Split

The current parser/config logic is **not** one generic seam. It splits by
subsystem:

### FM-owned parsing

- `parse_fm_anneal_schedule(...)`
- finest-level strategy env handling inside `graph_uncoarsen(...)`
- thick-restart / ensemble / gain-noise setup

### Separator-owned parsing

- `parse_sep_lift_strategy(...)`
- `parse_sep_lift_weight(...)`

### Orchestration-owned parsing / policy

- sep=0 retry decision
- forced-HEM override composition

Interpretation:

- do not start with a generic parser file
- let parser ownership follow subsystem ownership

## Shared Declaration Implications

The shared internal graph header currently exposes only these residual
behavior seams:

- `graph_refine_fm(...)`
- `graph_uncoarsen(...)`
- `graph_edge_separator_to_vertex_separator(...)`
- `sparse_graph_partition(...)`

Everything else in the residual file is still translation-unit local.

Day 2 implication:

- Day 3 should promote only the minimum declarations needed for a real FM file
- Day 4 should do the same for separator lifting
- the orchestration helpers should stay local unless a later move proves
  otherwise

## Correct Phase-2 Extraction Order

The strongest Day 2 implementation order is now explicit:

1. FM boundary design
2. separator/runtime/test design
3. FM extraction
4. separator extraction
5. runtime/orchestration cleanup after those moves land

## Bottom Line

Sprint 44 no longer needs to "find" the remaining graph seams. They are now
clear in the live file:

- FM is the first extract-now seam
- separator lifting is the second extract-now seam
- `graph_uncoarsen(...)` plus partition/retry composition is the true residual
  glue layer
- parser ownership should follow FM, separator, and orchestration seams rather
  than being forced into a generic runtime-config bucket
