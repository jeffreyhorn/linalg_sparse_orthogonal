# Sprint 70 Day 3: Product-Model Gap Inventory I

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Audit the strongest remaining linked-list/product-model and conversion-heavy
workflow seams in the live library so Epic 7 can target real ownership and
usability ceilings rather than broad “matrix model” rhetoric.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`
- `README.md`
- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- `src/sparse_matrix.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`

## Day 3 Product-Model Conclusions

### 1. The strongest exact product-model seam is still the one-shot direct workflow centered on copied `SparseMatrix` mutation

The public direct-solver surface still leads callers toward:

- building or loading a `SparseMatrix`
- making a `sparse_copy()` when they need to preserve the coefficient view
- factoring the copy in place
- solving through the factored copy
- relying on the matrix object itself to carry permutation/factored-state
  compatibility

This remains a real strength for small or occasional usage because it is low
ceremony and easy to explain.

It is also still the strongest Day 3 ceiling because the public center of
gravity remains:

- copy-first
- mutation-heavy
- matrix-state-sensitive
- centered on a general linked-list container even when the numeric backend is
  no longer really centered there

### 2. The second strongest seam is the mixed logical/physical/permuted-state contract of the generic matrix API

`SparseMatrix` still tries to carry too many roles simultaneously:

- mutable construction/edit surface
- generic arithmetic surface
- permutation owner
- factored-state carrier
- interoperability shell
- storage-inspection surface for advanced callers

That shows up in the public API through:

- distinct logical vs physical accessors
- public permutation-array accessors
- `sparse_reset_perms(...)`
- many warnings that operations should not be used on non-identity-permutation
  matrices
- factored-state markers and state checks

This is powerful, but it also means much of the matrix API is only safe or
meaningful if the caller already knows which matrix state they are currently
in.

### 3. The third strongest seam is compressed backend work that still converts out of and publishes back into `SparseMatrix`

The fastest direct numeric paths now clearly live in compressed working
formats:

- CSC Cholesky
- CSC LDL^T
- CSR LU

But the public product story still routes through:

- linked-list input state
- conversion into CSC/CSR working formats
- compressed numeric elimination and solve
- writeback or publication back into a linked-list-facing result surface

The most explicit exact examples are:

- `lu_csr_from_sparse(...)` / `lu_csr_to_sparse(...)`
- `chol_csc_from_sparse_with_analysis(...)`
- `chol_csc_writeback_to_sparse(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_writeback_to_ldlt(...)`

So the compressed paths are real implementation centers, but they still do not
look like the library’s public product center.

### 4. The shared repeated-run direct lifecycle is already a major convergence step, but it is still a parallel surface rather than the single dominant product model

The shared repeated-run lifecycle is clearly better aligned with long-term
factor/workspace ownership:

- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

It already owns:

- symbolic/permutation reuse
- cross-family repeated-run direct workflow
- a cleaner factor/workspace split than the one-shot family surfaces

But it still reads as a parallel advanced surface rather than the single
dominant product model because:

- one-shot family headers remain the main front door
- family-local owned-factor types still coexist beside shared
  `sparse_factors_t`
- identity-permutation and same-pattern preconditions are still caller-visible
  in a fairly detailed way

### 5. The broad Epic 7 product-model problem now reduces to four ranked seams

The current ranked Day 3 product-model seams are:

1. copy-first, in-place one-shot direct workflow centered on `SparseMatrix`
2. mixed logical/physical/permuted-state semantics across the generic matrix
   API
3. compressed backend conversion/writeback ownership split
4. shared repeated-run lifecycle versus family-local one-shot and owned-factor
   parallel surfaces

Lower but still real support context:

- compile-time tuning and threshold knobs on public surfaces
- public permutation-array accessors that expose how much state still lives on
  the matrix object
- interop and Matrix Market flow that still reinforce `SparseMatrix` as the
  universal public shell

## Day 3 Exit State

Sprint 70 now has one explicit first product-model hotspot map:

- the broad Epic 7 product-model concern is reduced to four concrete seams
- the strongest exact target is the copied-matrix, in-place one-shot direct
  workflow
- the next step is to rerank those seams against user cost, performance cost,
  compatibility burden, and proof burden before fixing the first true Epic 7
  product-model boundary
