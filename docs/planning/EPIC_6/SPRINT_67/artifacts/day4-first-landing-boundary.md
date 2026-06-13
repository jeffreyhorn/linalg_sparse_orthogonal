# Sprint 67 Day 4: First-Landing Boundary

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Convert the Day 3 hotspot ranking into one exact first implementation fence so
Sprint 67 starts from a bounded graph/reorder landing instead of a generic
cleanup set.

## Exact First Landing

The exact first landing is now fixed to:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

Why this is the right first batch:

- these two files still carry the strongest remaining combination of:
  - top-level orchestration
  - retry/fallback glue
  - runtime/env-policy handling
  - stale sprint-history commentary in permanent implementation surfaces

So the first Sprint 67 landing should shrink the remaining orchestration shells,
not spread evenly across every large graph-related file.

## Support Context, Not First-Batch Center

The following already-extracted graph subsystem files stay out of the first
landing unless the design proves a truly necessary support edit:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`

Why they stay out:

- they already read as narrower subsystem owners than `src/sparse_graph.c`
- their chronology burden is lower than the remaining orchestration shells
- widening into them immediately would blur whether Sprint 67 is still doing a
  bounded ownership extraction or a broad graph rewrite

## Likely Proof Home

The strongest proof surfaces for the first landing are now:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Likely support only if needed:

- `src/sparse_graph_internal.h`
- `tests/test_integration.c`

This keeps the first landing family-local by default and leaves cross-family
proof as optional rather than assumed.

## Explicit Non-Touch Set

The following remain outside the first landing fence:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- packaging/platform/build churn
- public coordination headers unless the landed graph/reorder design proves
  they truly need moving

## Ranked Order After Day 4

Sprint 67 now has one explicit implementation order:

1. first landing:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
2. likely proof home:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
3. support only if needed:
   - `src/sparse_graph_internal.h`
   - `tests/test_integration.c`
4. later/deferred:
   - CSC/analysis residual decomposition
   - iterative/eigensolver residual decomposition

## Exit State

Sprint 67 Day 4 closes with one exact first landing boundary:

- remaining graph/reorder orchestration first
- already-extracted graph subsystems held stable unless needed
- CSC/iterative explicitly deferred from the first batch

That gives Day 5 one exact job:

- define the ownership and extraction contract for the bounded
  `src/sparse_graph.c` / `src/sparse_reorder_nd.c` landing
