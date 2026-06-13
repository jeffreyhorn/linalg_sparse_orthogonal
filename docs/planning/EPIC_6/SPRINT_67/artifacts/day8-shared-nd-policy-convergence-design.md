# Sprint 67 Day 8: Shared ND Policy Convergence Design

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Define the bounded convergence design for the strongest remaining Sprint 67
maintainability seam: duplicated ND compatibility-policy normalization across
`src/sparse_reorder_nd.c` and `src/sparse_analysis.c`.

## Exact Design Target

The strongest remaining seam is now fixed to:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`

The exact target is not broad CSC backend work. It is:

- one internal owner for ND compatibility parsing and default policy
  normalization
- two consumers:
  - the public repeated-run analysis path
  - the direct `sparse_reorder_nd(...)` path

So the next landing is a shared policy-ownership convergence batch rather than
another graph-only extraction or a backend rewrite.

## Current Duplication To Converge

The duplicated policy surface currently covers:

- root-bisect mode
- coarsening mode
- coarsest-bisection mode
- root-bisect max-n
- coarsen floor ratio
- coarsening CV fallthrough
- separator-lift strategy
- separator-lift weight

The design implication is explicit:

- keep `src/sparse_analysis.c` as the owner of typed analysis-option
  resolution
- keep `src/sparse_reorder_nd.c` as the owner of the direct ND reorder entry
- move compatibility/default-policy normalization behind one internal helper
  seam instead of leaving it duplicated in both places

## Preserved Compatibility Contract

The convergence batch must preserve:

- zero-init-safe `sparse_analysis_reorder_opts_t` behavior
- typed analysis values overriding compatibility env vars exactly as shipped
- direct `sparse_reorder_nd(...)` continuing to honor the compatibility path
  when no typed analysis layer is involved
- no public change to the meaning of:
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`

That means the next landing is an ownership convergence batch, not a new
option-model or public-API redesign.

## Exact Day 9-10 File Fence

Required implementation surfaces:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`

Likely support only if the landed helper needs it:

- `src/sparse_reorder_nd_internal.h`

Likely proof home:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

Header/docs follow-through only if the landed code truly moves the wording:

- `include/sparse_analysis.h`

This keeps the second lane bounded and avoids widening into the backend-heavy
CSC files that Day 7 explicitly did not rerank to the top.

## Explicit Non-Widening Fence

The shared ND policy convergence batch should not widen into:

- `src/sparse_graph.c`
- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public API redesign
- packaging/platform/build churn

So the next batch remains a maintainability convergence landing, not a broader
analysis/CSC rewrite.

## Exit State

Sprint 67 Day 8 closes with one exact second-lane design:

1. strongest target:
   - shared ND compatibility/default-policy normalization
2. required code surfaces:
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
3. likely proof home:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`
4. support only if needed:
   - `src/sparse_reorder_nd_internal.h`
5. header/docs follow-through only if wording actually moves:
   - `include/sparse_analysis.h`

That gives Day 9 one exact job:

- land the bounded shared ND policy convergence batch without widening into CSC
  backend implementation files or public API redesign
