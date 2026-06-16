# Sprint 72 Day 4: Product-Model Surface Audit II and First Landing Boundary

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Refine the Day 3 ownership ranking and freeze the first bounded Sprint 72
convergence fence before implementation design begins.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day3-product-model-surface-audit.md`
- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

## Day 4 Boundary Conclusions

### 1. The strongest first Sprint 72 fence is the public direct-workflow seam, not the deeper compressed publication seam

The Day 4 rerank confirms the best first bounded lane is:

- direct one-shot workflow centered on `SparseMatrix`
- plus the repeated-run lifecycle handoff that already exists beside it

That lane has the strongest combination of:

- public direct-workflow pain
- implementation leverage
- bounded cleanup payoff
- acceptable first-pass compatibility risk

The compressed-path publication/writeback seam remains real, but it is not the
right first landing because it widens quickly into family-specific internals
across CSC Cholesky, CSC LDL^T, and CSR LU.

### 2. The generic matrix-state seam is support context for the first lane, not a separate rewrite target

The mixed logical/physical/permuted-state contract remains the strongest
second contradiction.

But the rerank shows it should be treated as:

- support context for direct-workflow clarification
- not a standalone matrix-model rewrite program

So the first batch can touch the matrix-state shell only where it clarifies:

- one-shot direct-workflow ownership
- repeated-run handoff boundaries
- visible factor-state expectations

It should not widen into generic arithmetic, permutation-accessor, or broad
logical-versus-physical API redesign.

### 3. The first-batch landing surfaces are now explicit

Required first landing:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`

Likely support only if the first landing forces it:

- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `tests/test_integration.c`
- `tests/test_sparse_matrix.c`

Deferred or explicitly later:

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- deeper family-local proof-owner expansion
- capability-surface work
- packaging/platform/install surfaces

### 4. The strongest non-goal fence is now explicit

Sprint 72 Day 4 fixes the first-lane non-goals:

- no repo-wide `SparseMatrix` rewrite
- no capability or type widening hidden inside ownership cleanup
- no broad family-by-family redesign without a ranked center
- no compressed-path publication overhaul as the first move
- no factor/workspace abstraction campaign detached from the direct workflow

## Exit State

Sprint 72 Day 4 closes with:

1. one explicit first convergence boundary around the public direct-workflow
   seam
2. one fixed support-only map for examples and proof follow-through
3. one explicit deferred map for deeper compressed-path cleanup
4. one clear starting point for Day 5 implementation design
