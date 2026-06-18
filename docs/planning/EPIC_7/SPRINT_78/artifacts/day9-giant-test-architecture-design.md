# Sprint 78 Day 9 - Giant-Test Architecture Design

Date: 2026-06-17  
Branch: sprint-78

## Purpose
Define the bounded proof-architecture batch for the strongest remaining permanent giant-test surface before any Day 10 edits begin.

## Main Result
Sprint 78 now has one explicit giant-test architecture contract:

- required Day 10 center:
  - `tests/test_chol_csc.c`
- family-local helper support only if the batch truly forces it:
  - `tests/test_chol_csc_supernodal_helpers.h`
- proof-owner or policy support only if truly forced:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
  - `docs/maintainer_guide.md`

## Why This Is the Right Day 10 Center
The best Day 10 move is not to split the entire Cholesky CSC proof surface into many files.

It is to reduce the strongest mixed proof-role review pressure inside
`tests/test_chol_csc.c`.

The strongest bounded payoff is therefore:

- tighter local helper ownership
- clearer internal test-block boundaries
- less mixed registration/comment chronology in one giant owner

## Exact Day 10 Architecture Center
The strongest likely Day 10 seam is the family-local supernodal/writeback/dispatch proof cluster inside `tests/test_chol_csc.c`, because that is where:

- family-local helper density is highest
- backend-contract proof is mixed with broader lifecycle proof
- section-local helper and registration pressure is strongest

`tests/test_chol_csc_supernodal_helpers.h` is the right support-only companion because it is already a narrow family-local helper seam rather than a shared global harness.

`tests/test_framework.h` is explicitly not a Day 10 center because widening the shared test harness would turn a family-local maintainability batch into broader test-infrastructure churn.

## Preserved Proof Checklist
Day 10 must preserve:

- current behavior coverage
- current proof-owner reading:
  - `tests/test_chol_csc.c` remains the Cholesky CSC family-local owner
- current platform-sensitive proof interpretation
- the Day 2 validation contract:
  - touched code/test batch runs `make format`, `make lint`, `make test`
  - substantial maintainability batch also runs `make quality-review-full`

## Exact Non-Touch Set
Day 10 explicitly should not widen into:

- unrelated source files
- unrelated giant tests by default
- broad proof-taxonomy rewrites across all solver families
- workflow, platform, benchmark, or packaging claims
- shared harness redesign
- public API or header changes

## Exit State
- The Day 10 giant-test batch is explicitly bounded before edits begin.
- `tests/test_chol_csc.c` is fixed as the required center.
- `tests/test_chol_csc_supernodal_helpers.h` is the only likely support seam.
