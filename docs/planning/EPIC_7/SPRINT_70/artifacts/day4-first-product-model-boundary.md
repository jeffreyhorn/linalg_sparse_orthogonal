# Sprint 70 Day 4: First Product-Model Boundary

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Convert the Day 3 product-model hotspot ranking into one exact first Epic 7
implementation fence so later product-model work starts from a bounded direct-
workflow convergence lane instead of a generic matrix-model rewrite.

## Exact First Boundary

The exact first Epic 7 product-model boundary is now fixed to:

- `README.md`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Likely support only if needed:

- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `tests/test_integration.c`

Why this is the right first batch:

- it attacks the strongest user-facing workflow seam directly
- it clarifies one-shot versus repeated-run ownership before broader storage
  or backend work
- it has a lower proof burden than rewriting the generic matrix API first
- it gives later compressed-path and matrix-state work a cleaner public center

So the first Epic 7 product-model landing should shrink ambiguity around:

- copy-first one-shot direct usage
- factor/workspace ownership
- same-pattern repeated-run ownership
- preserved-original and publication expectations

## Support Context, Not First-Batch Center

The following remain important but stay outside the first batch center:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`

Why they stay out:

- the generic matrix API seam is broader and higher-proof-burden than the
  first workflow-ownership landing
- the compressed conversion/writeback seam is real, but widening into it first
  would blur product-model convergence into backend rewrite
- first clarifying the public direct-workflow center gives those later seams a
  cleaner ownership target

## Explicit Non-Touch Set

The following remain outside the first product-model fence:

- broad logical/physical accessor redesign
- generic sparse arithmetic redesign
- permutation-accessor redesign
- full CSC/CSR publication or writeback redesign
- capability-surface widening (`idx_t`, scalar type, unsymmetric eigensolver
  scope)
- packaging/platform/install contract work
- benchmark-governance redesign

## Ranked Order After Day 4

Sprint 70 now has one explicit implementation order for Epic 7 product-model
work:

1. first boundary:
   - public direct-workflow ownership convergence
2. likely support only if needed:
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
   - `tests/test_integration.c`
3. next support seam:
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
4. later/deferred:
   - CSC/CSR conversion/writeback convergence
   - broad matrix-state API redesign
   - capability-adjacent widening

## Exit State

Sprint 70 Day 4 closes with one exact first product-model boundary:

- direct-workflow ownership first
- generic matrix-state redesign held as support or later work
- compressed backend publication convergence explicitly deferred behind the
  first workflow ownership lane

That gives later Sprint 70 and Sprint 72 planning one exact job:

- land the direct-solver product-story convergence first, then widen only
  where the bounded ownership changes prove it is necessary
