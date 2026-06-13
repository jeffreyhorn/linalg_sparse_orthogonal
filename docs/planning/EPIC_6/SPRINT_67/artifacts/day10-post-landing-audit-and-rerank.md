# Sprint 67 Day 10: Post-Landing Audit And Rerank

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Audit the post-Day-9 branch state and fix the next real Sprint 67 target in
writing rather than widening automatically from the shared ND policy lane.

## Audit Inputs

Re-read and compared:

- `docs/planning/EPIC_6/SPRINT_67/artifacts/day8-shared-nd-policy-convergence-design.md`
- `docs/planning/EPIC_6/SPRINT_67/artifacts/day9-shared-nd-policy-convergence-batch.md`
- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

Proof homes rechecked:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`

## Main Rerank

### The Day 9 batch closed the strongest ND-policy seam

The shared ND compatibility/default-policy contradiction is no longer the
highest-value maintainability problem:

- `src/sparse_reorder_nd.c` now owns the ND compatibility/default baseline
- `src/sparse_analysis.c` now consumes that baseline instead of carrying a
  second copy

The remaining `supernodal_postorder` compatibility parser in
`src/sparse_analysis.c` is intentionally smaller and separate from the Day 8
ND-policy target, so another immediate ND-policy batch would now be weaker than
the next available seam.

### The queue has shifted to the large-n analysis → CSC handoff

The strongest remaining ownership blur now sits in the large-`n` explicit
analysis lifecycle handoff across:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

The issue is not generic “big file” pressure.  It is the split ownership across
multiple partially parallel internal surfaces:

- `factor_cholesky_with_analysis_csc(...)`
- `factor_ldlt_with_analysis_csc(...)`
- `chol_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- CSC writeback/publication paths

So the next real maintainability win is now analysis-to-CSC orchestration
coherence on the large-`n` direct-family lane.

## Exact Next Target

### Cholesky owns the best next bounded landing

Cholesky is the better next landing than LDL^T because:

- `factor_cholesky_with_analysis_csc(...)` is the simpler large-`n` analysis
  handoff in `src/sparse_analysis.c`
- `src/sparse_chol_csc.c` already co-locates the analysis-aware conversion and
  the CSC writeback/publication seam
- LDL^T still carries extra Bunch-Kaufman-specific ownership:
  - `D`
  - `D_offdiag`
  - `pivot_size`
  - composed permutation state
  - resolved-analysis preparation

That makes Cholesky the better next bounded batch, while LDL^T remains a later
follow-through surface.

### Fixed Day 11 fence

Strongest next batch:

- large-`n` Cholesky analysis/CSC handoff coherence

Required code surfaces:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`

Likely proof home:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Support only if the landing truly needs it:

- `src/sparse_chol_csc_internal.h`

Still likely deferred:

- `src/sparse_ldlt_csc.c`
- `tests/test_ldlt_csc.c`
- `include/sparse_analysis.h`

## Non-Widening Fence

The next landing should not widen into:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- packaging/platform/build churn
- public API redesign

So Sprint 67 now transitions from the ND lane to the CSC/analysis lane while
still staying inside a bounded maintainability follow-through batch.

## Exit State

Sprint 67 Day 10 closes with one explicit rerank:

1. the shared ND policy lane is no longer the strongest remaining seam
2. the strongest next seam is the large-`n` analysis-to-CSC direct-family handoff
3. Cholesky owns the best next bounded landing in that lane
4. Day 11 should target:
   - `src/sparse_analysis.c`
   - `src/sparse_chol_csc.c`
   - likely proof in `tests/test_integration.c` and `tests/test_chol_csc.c`
