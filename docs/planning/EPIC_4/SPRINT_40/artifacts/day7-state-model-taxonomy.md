# Sprint 40 Day 7: State-Model Taxonomy

## Objective

Reduce the Day 5 and Day 6 lifecycle inventories into a stable taxonomy that
classifies APIs by lifecycle role rather than by file. This taxonomy is the
bridge between raw inventory and the later Epic 4 handle-model design.

## Inputs

This taxonomy is derived directly from:

- `docs/planning/EPIC_4/SPRINT_40/artifacts/day5-lifecycle-inventory-lu-cholesky-ldlt.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day6-lifecycle-inventory-qr-svd-analysis-iterative-eigs.md`

## Taxonomy Overview

The current Epic 4 state model is best understood as five major API classes:

1. matrix-mutating factor builders
2. original-matrix consumers with separate result handles
3. analysis / factor handle pipelines
4. read-only operator consumers
5. bridge / mixed-boundary surfaces

This classification is more useful than “direct vs iterative” or
“by-header-name” because it maps directly to lifecycle obligations:

- what owns persistent factor state
- what mutates matrix state
- what requires original matrix eligibility
- what returns partial progress on cancellation
- what still mixes matrix-centric and explicit-handle designs

## Class 1: Matrix-Mutating Factor Builders

### Definition

APIs that:

- mutate `SparseMatrix` in place during factorization
- turn that same matrix object into the post-factor solve handle
- store permutation / factor-state telemetry on the matrix itself
- create the strongest cancellation-sensitive matrix lifecycle risk

### Members

- `sparse_lu_factor`
- `sparse_lu_factor_opts`
- `sparse_cholesky_factor`
- `sparse_cholesky_factor_opts`

### Shared contract

- callers often need `sparse_copy()` first if original coefficients must be
  preserved
- solve reads factor state from the same matrix object
- cancellation can leave the matrix in a non-original state almost immediately

### Why this class matters

This is the clearest “legacy overloaded `SparseMatrix` lifecycle” class in the
current codebase and the strongest candidate for later explicit-handle
internalization.

## Class 2: Original-Matrix Consumers With Separate Result Handles

### Definition

APIs that:

- keep the input matrix read-only
- write persistent factor/result state into a dedicated output object
- still require the caller to provide the original / identity-permutation
  matrix view

### Members

- `sparse_ldlt_factor`
- `sparse_ldlt_factor_opts`
- `sparse_qr_factor`
- `sparse_qr_factor_opts`
- `sparse_qr_solve_minnorm`
- `sparse_svd_compute`
- `sparse_svd_partial`
- `sparse_svd_rank`
- `sparse_pinv`
- `sparse_svd_lowrank`
- `sparse_svd_lowrank_sparse`
- `sparse_ilu_factor`
- `sparse_ilut_factor`
- `sparse_ic_factor`

### Shared contract

- input matrix is not reused as the result/factor handle
- cleanup is explicit through handle-specific free routines
- matrix eligibility rules are strict:
  - original/unfactored state
  - identity permutations
  - subsystem-specific shape/symmetry rules

### Why this class matters

This class already reflects much of the architecture Epic 4 wants, but it
still pushes substantial lifecycle reasoning onto the caller via preconditions.

## Class 3: Analysis / Factor Handle Pipelines

### Definition

APIs that:

- split symbolic and numeric work across multiple explicit handles
- support staged workflows rather than one-shot factor-and-solve calls
- already expose a multi-object lifecycle to the caller

### Members

- `sparse_analyze`
- `sparse_analysis_free`
- `sparse_factor_numeric`
- `sparse_factor_solve`
- `sparse_factor_free`
- `sparse_refactor_numeric`

### Shared contract

- input matrix must still be original / identity-permuted / unfactored
- symbolic analysis and numeric factors are separate caller-visible objects
- refactorization reuses prior symbolic work

### Why this class matters

This is the strongest public bridge toward the future Epic 4 architecture:

- callers already think in terms of handles
- lifecycle phases are already explicit
- but internal factor representation still partially depends on a
  `SparseMatrix *F`, so the subsystem is not fully free of the old model

## Class 4: Read-Only Operator Consumers

### Definition

APIs that:

- treat the matrix primarily as an operator
- do not create a persistent factor handle themselves
- keep mutable numerical state in iterates, result buffers, workspaces, and
  optional callback/preconditioner contexts

### Members

Iterative solvers:

- `sparse_solve_cg`
- `sparse_solve_gmres`
- `sparse_solve_minres`
- `sparse_solve_bicgstab`
- block variants
- matrix-free variants

Eigensolvers:

- `sparse_eigs_sym`

### Shared contract

- matrix is read-only
- lifecycle emphasis is on:
  - solver options
  - initial guess
  - result buffers
  - convergence / partial progress
  - preconditioner or shift-invert context

### Why this class matters

This class should not be forced into the same handle model as direct
factorization families. Its state is iterative and transient, not matrix-owned.

## Class 5: Bridge / Mixed-Boundary Surfaces

### Definition

APIs or structs that do not fit cleanly into one model because they combine:

- explicit public handles
- internal matrix-centric payloads
- or read-only operator semantics with external factor/preconditioner
  dependencies

### Primary members

#### `sparse_factors_t`

This is the strongest bridge object:

- public handle is explicit
- but it still stores `SparseMatrix *F`
- plus LDLT-specific side arrays

This means it is handle-oriented at the API layer but still partly
matrix-centric in representation.

#### Iterative + preconditioner workflows

Iterative solvers alone are clean operator consumers, but realistic workflows
often compose them with:

- `sparse_ilu_t`
- `sparse_ldlt_t`
- `sparse_ilu_precond`
- `sparse_ic_precond`
- external factor contexts

That makes the end-to-end lifecycle mixed even when the solver entry point is
not.

#### Eigensolver shift-invert / LOBPCG composition

The eigensolver entry point is read-only on `A`, but internally or externally
it composes with:

- LDLT factorization for shift-invert
- preconditioner callbacks for LOBPCG
- optional refinement via repeated LDLT solves

So its public class is “operator consumer,” but its real dependency graph
touches the handle-building classes heavily.

## Grouped API-Role Map

### Matrix-mutating direct factor builders

- LU factorization path
- Cholesky factorization path

### Separate-handle factor / decomposition builders

- LDLT factorization path
- QR factorization path
- SVD family
- ILU / ILUT / IC builders

### Multi-phase symbolic/numeric pipelines

- analysis / factor / refactor / solve path

### Read-only operator consumers

- iterative solvers
- eigensolvers

### Composition / bridge surfaces

- `sparse_factors_t`
- solver + preconditioner workflows
- eigensolver + shift-invert / LOBPCG workflows

## Clean Boundaries vs Mixed Boundaries

### Cleanest boundaries already present

The cleanest lifecycle boundaries today are:

- LDLT factor handle
- QR factor handle
- SVD result handle
- iterative solver operator-consumer model

These already communicate a fairly direct ownership story.

### Most mixed / transitional boundaries

The strongest mixed-boundary hotspots are:

- LU / Cholesky overloading `SparseMatrix`
- `sparse_factors_t` wrapping `SparseMatrix *F`
- analysis pipeline mixing explicit public handles with matrix-centric internal
  storage
- iterative/eigensolver workflows that are clean at the entry point but depend
  on strict-lifecycle preconditioner/factor contexts in realistic usage

## Strongest Design Pressure Points

The taxonomy now makes the later handle-model pressure points explicit:

### 1. Move away from matrix-as-factor-handle

This pressure comes almost entirely from the Class 1 LU / Cholesky family.

### 2. Reduce caller-facing matrix-eligibility burden

This pressure comes from Class 2 and Class 3:

- QR
- SVD
- analysis
- ILU / ILUT / IC

These are already handle-based but still require careful original-state
reasoning from callers.

### 3. Preserve operator-consumer semantics where appropriate

This pressure comes from Class 4:

- iterative solvers
- eigensolvers

These should not be collapsed into a factor-handle story they do not need.

### 4. Normalize bridge objects before large implementation refactors

This pressure comes from Class 5:

- `sparse_factors_t`
- composed solver/preconditioner workflows
- eigensolver composition paths

## Day 7 Conclusions

1. The current codebase is not one lifecycle model; it is at least five.
2. Explicit handles already exist across much of the API surface, so Epic 4 is
   not starting from zero.
3. The two biggest architecture problems are now distinct:
   - matrix-as-factor-handle overloading
   - strict and inconsistent original-matrix eligibility rules
4. The analysis pipeline is the clearest bridge subsystem between the current
   architecture and the future Epic 4 handle model.
5. This taxonomy is now stable enough to drive Day 8’s precondition/mutation/
   cancellation contract map and Day 9’s first handle-model design note.
