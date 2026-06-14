# Sprint 68 Day 8: Differential/Oracle Coverage Design

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Turn the Day 7 rerank into one exact second-layer numerical assurance contract
so the next Sprint 68 batch strengthens the hardest retained large-`n`
CSC-backed Cholesky public path without widening into unrelated proof or
implementation work.

## Chosen Assurance Owner

The Day 9 batch should be owned by:

- `tests/test_integration.c`

Why this is the right owner:

- it already owns the public one-shot versus explicit repeated-run contract
- it already contains the large-`n` CSC-backed Cholesky public-path parity lane
- it can absorb one stronger oracle batch without dragging in
  implementation-detail ownership

The current family-local support context remains:

- `tests/test_chol_csc.c`

But that file is support only if the final oracle shape truly requires it.

## Chosen Oracle Shape

The strongest additive proof is a staged public-path parity oracle:

1. build one large-`n` SPD baseline on the CSC side
2. confirm one-shot Cholesky and the explicit repeated-run path agree
3. refactor to a same-pattern second SPD matrix and confirm they still agree
4. refactor to a same-pattern third SPD matrix and confirm they still agree
5. keep a fixed exact-solution oracle so every stage checks both:
   - public-path parity
   - numerical correctness against an external-style target vector

Why this shape is stronger than the current split state:

- the current repo already proves one-shot-vs-analysis parity
- it already proves same-pattern repeated-run parity
- what is missing is one continuous public-path oracle story across multiple
  large-`n` CSC-backed stages in the same owner

## Tolerance and Failure Contract

The Day 9 batch should use the following explicit contract:

- stay on the CSC side:
  - `n >= SPARSE_CSC_THRESHOLD`
- each public one-shot solve and explicit repeated-run solve must match the
  fixed exact solution to:
  - `1e-12`
- each one-shot / repeated-run solve pair must match each other to:
  - `1e-12`
- when path-publication state is observed, assert CSC-side routing explicitly

What the batch is not:

- not a failure-preservation batch
- not a family-local kernel/residual batch
- not a benchmark/throughput batch
- not an implementation or backend-design batch

## Exact Day 9 File Fence

Required likely implementation surface:

- `tests/test_integration.c`

Support only if the final oracle shape truly needs it:

- `tests/test_chol_csc.c`

Explicit non-touch set:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- implementation `src/` files
- benchmark/docs truth surfaces

## Exit State

Sprint 68 Day 8 closes with one exact Day 9 oracle contract:

1. owner:
   - `tests/test_integration.c`
2. likely support only if needed:
   - `tests/test_chol_csc.c`
3. proof shape:
   - large-`n` CSC-backed Cholesky public-path staged parity across multiple
     same-pattern SPD states
4. oracle/tolerance contract:
   - exact-solution agreement at `1e-12`
   - one-shot vs explicit repeated-run agreement at `1e-12`
   - explicit CSC-side routing assertion when publication state is observed
5. explicit non-touch set:
   - other giant tests
   - implementation files
   - benchmark/docs truth surfaces

That gives Day 9 one exact job:

- land one bounded large-`n` CSC-backed Cholesky public-path oracle/parity
  batch in `tests/test_integration.c`
