# Sprint 72 Day 3: Product-Model Surface Audit I

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Audit the live direct-workflow and matrix-state ownership seams so Sprint 72
can target a real first convergence boundary instead of treating the full
linked-list public model as one undifferentiated problem.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `README.md`
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
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`

## Day 3 Product-Model Conclusions

### 1. The strongest exact Sprint 72 seam is still the one-shot direct workflow centered on copied `SparseMatrix` mutation

The current one-shot direct-solver story still leads callers toward:

- building or loading a `SparseMatrix`
- copying that matrix when they need to preserve the coefficient view
- factoring the copy in place through a family-local entry point
- solving through the factored copy
- relying on the matrix object itself to keep carrying permutation and
  factored-state compatibility

This is still a real strength for smaller or occasional usage because it
remains low-ceremony and easy to explain.

It is also still the strongest Day 3 ceiling because the public center of
gravity remains:

- copy-first
- mutation-heavy
- matrix-state-sensitive
- centered on a linked-list compatibility shell even when the strongest
  numeric work now clearly happens in CSC or CSR working formats

### 2. The second strongest seam is the mixed logical/physical/permuted-state contract of the generic matrix API

`SparseMatrix` still tries to carry too many roles simultaneously:

- mutable construction and edit surface
- generic arithmetic surface
- permutation owner
- factored-state carrier
- interoperability shell
- storage-inspection surface for advanced callers

That shows up in the public API through:

- separate logical versus physical element accessors
- public permutation-array accessors
- `sparse_reset_perms(...)`
- warnings around arithmetic or mutation on non-identity-permutation matrices
- factored-state markers and state checks

This is powerful, but it also means part of the generic matrix API is only
safe or meaningful if the caller already knows what logical or physical state
the matrix is currently in.

### 3. The third strongest seam is compressed direct-path work that still converts out of and publishes back into `SparseMatrix`

The strongest direct numeric paths now clearly live in compressed working
formats:

- CSC Cholesky
- CSC LDL^T
- CSR LU

But the current product story still routes through:

- linked-list input state
- conversion into CSC or CSR working formats
- compressed numeric factorization and solve
- publication or writeback back into a linked-list-facing result surface

The strongest exact examples are:

- `chol_csc_from_sparse_with_analysis(...)`
- `chol_csc_writeback_to_sparse(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_writeback_to_ldlt(...)`
- `lu_csr_from_sparse(...)`
- `lu_csr_to_sparse(...)`

So the compressed paths are already the real implementation centers, but they
still do not read like the dominant public product model.

### 4. The shared repeated-run lifecycle is already a convergence step, but it still sits beside the one-shot family surfaces rather than replacing them as the dominant product center

The shared repeated-run lifecycle already aligns better with long-term
factor/workspace ownership:

- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

It already owns:

- symbolic and permutation reuse
- cross-family repeated-run direct workflow
- a cleaner factor/workspace split than the one-shot family-local paths

But it still reads as a parallel advanced surface rather than the dominant
product model because:

- the one-shot family headers remain the easier public front door
- family-local owned-factor types still coexist beside shared
  `sparse_factors_t`
- same-pattern and identity-permutation preconditions remain fairly visible at
  the caller boundary

### 5. The broad Sprint 72 product-model problem now reduces to four ranked seams

The ranked Day 3 product-model seams are now explicit:

1. copy-first, in-place one-shot direct workflow centered on `SparseMatrix`
2. mixed logical/physical/permuted-state semantics across the generic matrix
   API
3. compressed direct-path conversion and publication/writeback ownership split
4. shared repeated-run lifecycle versus family-local one-shot and owned-factor
   parallel surfaces

Lower but still real support context:

- compile-time threshold and backend-selector spill around the public direct
  workflow
- public permutation-array exposure that reinforces how much state still lives
  on the matrix object
- interoperability and Matrix Market flows that keep `SparseMatrix` as the
  universal public shell

## Exit State

Sprint 72 Day 3 closes with:

1. one explicit current-state product-model hotspot map
2. one ranked ownership-contradiction list for the first Sprint 72 design pass
3. one clear starting point for Day 4 boundary work: the copied-matrix
   one-shot direct workflow, with the generic matrix-state seam immediately
   behind it
