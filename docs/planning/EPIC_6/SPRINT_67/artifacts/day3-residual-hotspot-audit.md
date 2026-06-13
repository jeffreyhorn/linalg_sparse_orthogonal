# Sprint 67 Day 3: Residual Hotspot Audit

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Audit the current remaining large-source ownership hotspots from the live repo
state before any Sprint 67 decomposition work lands.

## Ranked Hotspot Order

The live hotspot order is now:

1. graph/reorder orchestration residuals
2. CSC/analysis residuals
3. iterative/eigensolver residuals
4. public coordination-header truth follow-through

This means Sprint 67 is not starting from a generic "all large files are
equal" maintainability problem. It is starting from a smaller ranked seam map.

## Strongest First Target: Graph/Reorder

The strongest first target is:

- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Why this lane ranks first:

- it still carries the densest remaining sprint-history commentary in permanent
  implementation files
- top-level orchestration, retry/fallback glue, parser/runtime policy, and
  durable subsystem ownership are still not separated cleanly enough
- the proof surface is already strong and naturally bounded:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`

The strongest exact first seam is now:

- residual uncoarsening / orchestration in `src/sparse_graph.c`
- residual root-policy / profiling / fallback orchestration in
  `src/sparse_reorder_nd.c`

## Strongest Second Target: CSC/Analysis

The strongest second target is:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

Why it ranks second:

- these files are still large and still carry residual chronology and
  coordination burden
- but they already read more like owned backend/analysis surfaces than the
  graph/reorder orchestration files do
- the proof burden is broader because the touched behavior fans into:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`

So CSC/analysis remains a real Sprint 67 lane, but it is not the best first
landing.

## Later or Narrower Target: Iterative/Eigensolver Residuals

The remaining iterative/eigensolver hotspots are:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Why they rank below graph and CSC:

- both files are large, but their ownership blur is weaker than in the
  graph/reorder lane
- the chronology burden is lighter and more localized
- the proof burden is substantial:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`

This makes them valid later or narrower Sprint 67 targets, not the first
landing center.

## Strongest Current Contradictions

The strongest remaining contradictions are:

- durable algorithm explanation mixed with sprint-history narration
- runtime/env-policy parsing mixed with top-level orchestration
- fallback and retry logic mixed with family-local ownership
- already-extracted subsystem boundaries still explained through old sprint
  archaeology in permanent files

These contradictions are strongest in:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`

So the real target is clearer ownership and less permanent chronology, not raw
line-count reduction for its own sake.

## Measured Hotspots

The measured Day 3 hotspot sizes remain:

- `src/sparse_graph.c` = `801`
- `src/sparse_graph_coarsen.c` = `641`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_graph_refine.c` = `629`
- `src/sparse_reorder_nd.c` = `743`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_analysis.c` = `1020`
- `src/sparse_chol_csc.c` = `1532`
- `src/sparse_ldlt_csc.c` = `2130`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_eigs.c` = `1534`

## Exit State

Sprint 67 Day 3 closes with one explicit ranked target order:

1. graph/reorder decomposition first
2. CSC/analysis residual decomposition second
3. iterative/eigensolver residuals later or narrower only if still justified

That fixes the Day 4 job too:

- turn the graph/reorder lane into the exact first landing fence instead of
  keeping it as a generic hotspot bucket
