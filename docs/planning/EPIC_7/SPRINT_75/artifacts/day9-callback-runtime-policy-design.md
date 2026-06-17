# Sprint 75 Day 9 Artifact: Callback / Runtime Policy Design

Date: 2026-06-17
Branch: sprint-75

## Purpose

Define the bounded Cholesky CSC callback/runtime follow-through batch after
the Day 7 kernel landing.

## Main Result

The next batch should target:

- bounded CSC orchestration-level callback/cancel parity

It should not claim:

- linked-list and CSC per-column callback parity
- mid-supernode rollback or cancellation semantics the backend does not
  actually implement

## Corrected Ownership Split

The true first public runtime owners are:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`

Reason:

- `sparse_cholesky_factor_opts(...)` owns backend selection
- it publishes `used_csc_path`
- it owns reordered-working-copy versus no-reorder mutation semantics
- it already owns the linked-list callback/cancel entry point

The CSC storage/runtime support seams move only if needed:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

## Day 10 Target

Required Day 10 touch set:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

Support only if the implementation truly forces them:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `docs/maintainer_guide.md`

## Preserved Runtime Truth

Day 10 must preserve:

- linked-list backend callback semantics unchanged
- `used_csc_path` publication unchanged
- CSC cancel points only at explicit orchestration checkpoints before
  publish-back commits the factor shell into the caller matrix
- `SPARSE_ERR_BACKEND_CONTRACT` remains a narrow backend-helper/callback
  failure
- no fake claim of per-column CSC callback parity

## Explicit Non-Centers

Not the Day 10 center:

- `src/sparse_dense.c`
- eigensolver backend/runtime parity
- QR backend/runtime follow-through
- SVD backend/runtime follow-through
- broad README or platform/install wording

## Exit State

Day 9 closes with:

- one corrected public runtime owner split
- one bounded CSC orchestration-level parity design
- one exact Day 10 touch set
- one explicit preserved parity checklist
