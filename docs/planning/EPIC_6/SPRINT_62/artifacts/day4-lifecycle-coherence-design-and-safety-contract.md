# Sprint 62 Day 4: Lifecycle Coherence Design and Safety Contract

Date: 2026-06-10
Branch: `sprint-62`


## Purpose

Define the bounded Sprint 62 hardening model for direct one-shot usability and
the exact preserved compatibility rules before code changes begin, so the
first implementation batch lands against a real public/internal ownership
contract.

## Design Decision

### 1. The explicit repeated-run direct lifecycle remains the canonical reuse contract

Sprint 62 should preserve the current ownership split exactly:

- one-shot wrappers remain first-class/default direct entry points
- the explicit repeated-run direct contract remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- internal reuse of lifecycle plumbing by a one-shot wrapper does not make the
  wrapper the same public workflow

This means Sprint 62 is a coherence sprint, not a public direct-workflow
merger.

### 2. Sprint 62 should reduce surprise by clarifying mutation and state publication, not by hiding them

The main safety rule is:

- do not silently copy inside one-shot wrappers just to mask mutation
- do not change family-local ownership models
- do tighten preconditions, invalidation, cleanup, and user-facing wording so
  the actual mutation/state behavior is easier to understand and harder to
  misuse

Preserved family-specific model:

- LU:
  - caller-owned matrix remains the one-shot factor container
- Cholesky:
  - caller-owned copied matrix remains the in-place factor container
- LDL^T:
  - family-local owned `sparse_ldlt_t` remains the one-shot result object
- QR:
  - original unfactored/unreordered matrix expectation remains explicit

This is intentionally not a “make all direct solvers behave the same” sprint.

## Ranked Implementation Direction

### 1. LU is the strongest first implementation target

LU is the best first batch because it mixes the most behavior into one public
story:

- one-shot in-place mutation
- reorder-before-factor control flow
- wrapper/lifecycle crossover through the shared lifecycle-compatible fast path
- cancellation and compatibility-mirror nuances

Recommended first hardening focus:

- make the LU one-shot wrapper easier to reason about when:
  - it stays in the one-shot path
  - it routes through the lifecycle fast path
  - it exits early around reorder or cancellation-sensitive transitions

### 2. Cholesky is the strongest second target

Cholesky still matters, but mostly for:

- mutation surprise
- backend clarity
- cancellation caveats

It belongs in the Sprint 62 contract, but should stay outside the first exact
touched-file fence so LU can land as one coherent batch first.

### 3. LDL^T remains a follow-through target

LDL^T should stay in the design and later regression queue, but not define the
first landing batch. Its main remaining gap is coherence with the shared
lifecycle story, not the highest-severity one-shot mutation risk.

### 4. QR remains a contrast surface

QR should mostly remain a comparison surface for shared caller expectations,
not the defining Sprint 62 code target.

## Ownership Split

### Public wrapper behavior should own

- clearer precondition and mutation wording
- clearer one-shot versus explicit lifecycle positioning
- default-wrapper behavior normalization where the implementation already
  promises equivalence

### Internal factor-state hardening should own

- reorder metadata invalidation/retention discipline
- compatibility-mirror cleanup around early exits
- wrapper/lifecycle fast-path coherence

### Lifecycle helper plumbing should own

- shared helper use only where it reduces ambiguity without widening API
- alignment between one-shot wrappers and explicit lifecycle publish/free rules

### Docs/examples should own

- copy-discipline guidance
- one-shot versus repeated-run workflow choice
- family-specific mutation caveats only where callers actually need them

## Compatibility Contract

### 1. What Sprint 62 preserves

- one-shot wrappers remain available as the default/simple entry points
- the explicit repeated-run direct lifecycle remains the only public
  analyze-once / factor-many workflow
- no new top-level direct lifecycle object
- no broad API removal or renaming
- no hidden broadening of repeated-run support boundaries
- no silent semantic promise that previously mutating one-shot paths are now
  bit-identical or copy-preserving

### 2. What Sprint 62 may clarify or tighten

- when one-shot wrappers are the right default
- when the explicit lifecycle should replace them
- which state becomes invalid on reorder/cancel/factor transitions
- which wrapper behaviors already align with explicit lifecycle outputs on the
  supported path

### 3. What Sprint 62 explicitly does not justify

- merging one-shot and explicit lifecycle APIs into one public surface
- automatic hidden copying to “fix” mutation surprise
- broad backend-policy work
- repeated-run support widening outside the existing lifecycle fence

## Day 5-7 Landing Fence

### First implementation target

- LU wrapper/lifecycle coherence only

### Likely first touched surfaces

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`
- possibly bounded docs follow-through later, not first

### Keep out of the first batch

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `include/sparse_ldlt.h`
- `src/sparse_ldlt.c`
- `include/sparse_qr.h`
- `src/sparse_qr.c`
- broad `sparse_analysis.h` redesign
- packaging/platform work
- configuration-surface work

## Day 4 Exit State

Sprint 62 now has a concrete direct-usability contract:

- the public lifecycle boundary is preserved
- the mutation/ownership model is preserved
- the first hardening target is fixed to LU
- the first landing batch is clearly smaller than the whole direct solver set
- Day 5 can now define the exact touched-file plan against this safety fence
