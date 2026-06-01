# Sprint 50 Day 4 Artifact: Lifecycle Precedent Inventory

## Purpose

Map the lifecycle precedents Sprint 50 can safely reuse, separate those from
the direct-solver-specific structural seams that should remain private, and fix
the precedent set that the Day 5 gap analysis should measure against.

## Precedent Classes

### 1. Direct public lifecycle precedent

The strongest direct-solver precedent is already public in
`include/sparse_analysis.h`:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

This precedent is domain-native for Sprint 50 because it already expresses the
actual repeated direct workflow:

1. analyze once
2. factor numerically
3. solve
4. refactor with the same sparsity pattern
5. solve again
6. free explicit owned state

### 2. Generic public repeated-run handle precedent

Epic 4 added a second precedent class:

- iterative handles
  - `sparse_iter_handle_t`
  - `sparse_iter_handle_init(...)`
  - `sparse_iter_handle_prepare_*`
  - `sparse_solve_*_with_handle(...)`
  - `sparse_iter_handle_free(...)`
- eigensolver handles
  - `sparse_eigs_handle_t`
  - `sparse_eigs_handle_init(...)`
  - `sparse_eigs_handle_prepare(...)`
  - `sparse_eigs_sym_with_handle(...)`
  - `sparse_eigs_handle_free(...)`

These do not provide the direct-solver domain story, but they do provide the
generic public-contract shape:

- zero-init or init helper
- explicit prepare step
- repeated run step
- free safe on zeroed state
- one-shot public APIs remain first-class
- reuse preserves capacity/setup, not old numerical state

### 3. Direct structural implementation precedents

The direct implementation side already contains structural seams that later API
exposure can route through:

- `src/sparse_analysis.c`
  - permutation-aware working-copy construction
  - factor-type dispatch
  - factor payload ownership transfer
  - solve dispatch
  - safe refactor overwrite semantics
- `src/sparse_chol_csc_internal.h`
  - `chol_csc_from_sparse_with_analysis(...)`
  - analysis-driven CSC preallocation path
- `src/sparse_ldlt_csc_internal.h`
  - `ldlt_csc_from_sparse_with_analysis(...)`
  - pre-pass / pre-permuted indefinite workflow scaffolding

These are valuable implementation precedents, but they are not themselves good
public API shapes.

## Borrow vs Keep Direct-Specific

### Borrow from existing precedents

Sprint 50 should borrow the following:

- explicit owned lifecycle objects
- zero-init / init helper safety
- explicit prepare/analyze-before-run discipline
- one-shot API preservation as a compatibility rule
- reuse preserves setup/capacity, not old numerical state
- explicit free safe on empty/zeroed state

These rules are already stable across:

- `sparse_analysis.h`
- iterative handles
- eigensolver handles
- factor-object APIs such as LDL^T and QR

### Keep direct-solver-specific

Sprint 50 should keep the following direct-solver-specific:

- symbolic-analysis semantics
- same-pattern numeric refactor contract
- factor-type differences between LU, Cholesky, and LDL^T
- permutation and reordered-copy interaction
- CSC/native backend dispatch and telemetry details
- mutable-`SparseMatrix` compatibility realities in LU and Cholesky

These are not generic repeated-run concerns. They are the direct-solver domain
constraints the final public lifecycle model has to accommodate.

## Key Day 4 Mapping

### `sparse_analysis.h` as primary design anchor

This is the best direct design anchor because it already:

- uses direct-solver terminology
- includes factor and refactor explicitly
- exposes the same-pattern repeated-run contract
- teaches explicit free ownership
- routes naturally across LU, Cholesky, and LDL^T factor types

### Sprint 49 handles as secondary public-contract anchor

The iterative/eigensolver handles are the best secondary precedent because they
already fix:

- prepare / run / free public wording
- compatibility framing for one-shot wrappers
- reuse semantics around capacity rather than old iteration state

### CSC/internal seams as implementation-only precedent

The CSC and direct dispatch seams should inform later implementation planning,
but not public naming or public shape:

- `chol_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- factor payload transfer helpers in `src/sparse_analysis.c`

Those are internal ways to realize the lifecycle, not the lifecycle contract
callers should see.

## Highest-Value Day 4 Conclusions

### 1. Sprint 50 already has the precedent set it needs

It does not need a new public lifecycle model from scratch. It already has:

- a direct repeated-run precedent (`sparse_analysis.h`)
- a generic public repeated-run-handle precedent (Sprint 49)
- internal structural seams that can support later implementation

### 2. The direct-solver side should be analysis-centric first, handle-centric second

The direct-solver domain is already organized around analyze / factor /
refactor. Any later lifecycle exposure should compose around that truth rather
than replacing it with a purely opaque-workspace-centered story.

### 3. Day 5 can now analyze gaps against a bounded and coherent precedent set

The Day 4 precedent inventory leaves Day 5 with a much narrower question:

- what is still missing between the current one-shot compatibility surfaces and
  the already-existing analysis/refactor direct lifecycle, given the generic
  public-contract rules Sprint 49 proved out?
