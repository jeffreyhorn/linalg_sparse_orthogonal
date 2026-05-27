# Sprint 42 Day 11 Artifact: Compatibility Bridge & Focused-Test Design

## Scope

Day 11 defines the compatibility and focused-test design that follows the Day
10 contract-normalization landing:

- how current one-shot APIs remain the public compatibility wrappers
- how `sparse_factors_t` continues as the preserve-and-evolve bridge
- which lifecycle misuse/copy-before-use tests are already strong enough
- which small Day 12 additions are still worth landing

This is a design day. It does not change public API shape or add tests yet.

## Compatibility Wrapper Plan

### 1. One-shot APIs remain the compatibility front door

Sprint 42 still keeps the existing public families as the caller-facing
compatibility layer:

- direct matrix-mutating one-shot wrappers:
  - LU
  - Cholesky
- separate-handle families:
  - LDLT
  - QR
  - SVD
- analyze-once compatibility bridge:
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`

Design rule:

- later internal-handle work should continue to land underneath these public
  surfaces first
- later public-handle work should only happen after the internal seams are
  already stable

### 2. Compatibility shims should stay thin

The correct later Epic 4 direction is:

- public wrappers stay stable
- internal ownership/payload logic keeps moving behind private seams
- wrappers should adapt into the newer internals, not duplicate lifecycle logic

Interpretation:

- Sprint 42 is not creating a parallel public API family
- compatibility is preserved through adapter thinning, not surface multiplication

## `sparse_factors_t` Preserve-and-Evolve Role

### What remains preserved now

After Days 9-10, `sparse_factors_t` still preserves:

- installed public struct shape
- current factor-type dispatch role
- current analyze-once solve contract
- caller-facing ownership/free pattern

### What can keep evolving later

The bridge can still evolve internally through:

- helper-based payload assembly
- helper-based LDLT solve-view reconstruction
- success-only output commit
- later internal payload substitutions behind the same public fields

Design rule:

- treat `sparse_factors_t` as the public compatibility shell until later Epic 4
  phases deliberately choose a new public bridge story
- do not force a public bridge redesign just because the private lifecycle
  model improves

## Copy-Before-Use Compatibility Rule

### Stable caller rule

Sprint 42 still depends on one explicit caller expectation:

- if the original coefficient matrix must remain available for later reuse, the
  caller should work from a fresh `sparse_copy(...)` before direct
  matrix-mutating factorization

This remains important for:

- LU
- Cholesky
- any workflow that later needs the original matrix for:
  - QR
  - SVD
  - analyze-once
  - ILU / ILUT / IC

### Why this remains compatible

This is already the documented contract in the public headers and tutorial.
Sprint 42 does not remove it; it only makes the underlying lifecycle seams
cleaner and more explicit.

## Current Test Coverage Assessment

### Already strong enough

The following areas already have useful Sprint 42-level coverage:

- cancel-path coverage for:
  - LU
  - Cholesky
  - LDLT
  - QR
  - iterative/eigensolver families
- analyze-once success/refactor coverage in `tests/test_etree.c`
- Day 10 regression:
  - failed `sparse_factor_numeric(...)` preserves old factors
- factored-matrix misuse coverage already present for:
  - ILU
  - ILUT

Interpretation:

- Day 12 does not need to reopen these areas broadly

### Highest-value remaining gaps

The remaining worthwhile gaps are explicit misuse-rejection tests.

#### 1. Analyze-once misuse tightening

Target file:

- `tests/test_etree.c`

Planned additions:

- `sparse_analyze(...)` rejects already-factored matrices
- `sparse_factor_numeric(...)` rejects matrices with non-identity row/col state

Why this matters:

- these are the core analyze-once bridge preconditions
- the existing test home already owns this API family

#### 2. QR copy-before-use misuse tightening

Target file:

- `tests/test_qr.c`

Planned additions:

- factor or reorder a matrix first
- then assert QR rejects the reused matrix with `SPARSE_ERR_BADARG`

Why this matters:

- the public contract already says QR requires original/unreordered state
- Day 12 should pin that operator-facing rule directly in tests

#### 3. SVD copy-before-use misuse tightening

Target file:

- `tests/test_svd.c`

Planned additions:

- factor or reorder a matrix first
- then assert SVD rejects the reused matrix with `SPARSE_ERR_BADARG`

Why this matters:

- this is the same caller lifecycle rule as QR
- it is a high-signal compatibility expectation, not a niche edge case

## Day 12 Batch Boundary

Day 12 should **not** widen into:

- broad new lifecycle-test framework work
- public-handle tests for APIs that do not yet exist
- large benchmark/example misuse coverage
- broad documentation cleanup
- general QR/SVD redesign work

The right Day 12 batch is:

- small
- rejection/contract focused
- tightly aligned with touched Sprint 42 lifecycle seams

## Day 11 Outcome

Sprint 42 now has an explicit compatibility and focused-test design:

- current one-shot APIs remain the public compatibility wrappers
- `sparse_factors_t` remains the preserve-and-evolve bridge
- copy-before-use stays an explicit compatibility rule for matrix-mutating
  workflows
- Day 12 now has a concrete bounded landing set:
  - `tests/test_etree.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
