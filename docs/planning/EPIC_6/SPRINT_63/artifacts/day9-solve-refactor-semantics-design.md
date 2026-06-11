# Sprint 63 Day 9: Solve/Refactor Semantics Design

Date: 2026-06-10
Branch: sprint-63

## Purpose

Convert the Day 8 rerank into one exact Day 10 implementation fence for the
remaining shared direct lifecycle semantics lane.

## Reviewed Surfaces

Primary design surfaces:

- `src/sparse_analysis.c`
- `tests/test_integration.c`

Adjacent public-story surfaces rechecked:

- `include/sparse_analysis.h`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

## Main Design Result

The remaining Sprint 63 semantics queue is now reduced to one exact question:

- how should the shared direct lifecycle prove and, if needed, tighten the
  large-`n` CSC-backed Cholesky factor/refactor retention contract?

This is narrower than a general lifecycle redesign because the core mechanism is
already mostly correct on the landed branch:

- `sparse_factor_numeric(...)` builds a temporary `new_factors` object and only
  swaps it into the caller `factors` object after success
- `sparse_refactor_numeric(...)` validates existing factors, factors into a
  temporary, and preserves old factors on error

The missing strength is not the broad algorithm. The missing strength is the
explicit large-`n` CSC-backed Cholesky proof and, only if necessary, the
smallest semantics follow-through needed to support it.

## What Is Already Strong Enough

The current branch already proves:

- zeroed-factor solve rejection
- mismatched-analysis solve rejection with preserved factors
- zeroed-factor refactor acceptance as a first factorization
- mismatched-existing-factor refactor rejection
- old-factor preservation on refactor failure
- nnz-drift rejection with preserved old factors
- large-`n` Cholesky same-pattern success parity against one-shot factorization
  at `n = 120`, already on the CSC side of `SPARSE_CSC_THRESHOLD`

That means Day 10 should not try to redesign `sparse_factor_numeric(...)` or
`sparse_refactor_numeric(...)` wholesale.

## Remaining Asymmetry

The live proof asymmetry is:

- linked-list Cholesky failure-preserve proof exists on the public lifecycle
  path (`n = 40`)
- large-`n` LDL^T failure-preserve proof exists on the public lifecycle path
  (`n = 150`)
- large-`n` Cholesky success parity exists on the public lifecycle path
  (`n = 120`)
- but large-`n` CSC-backed Cholesky failure-preserve semantics are not yet
  pinned with the same explicitness

This is the right Day 10 seam because it is:

- public repeated-run direct lifecycle behavior
- CSC-sensitive
- already close to fully covered
- small enough to land without widening into unrelated direct-family work

## Exact Day 10 Target

The next implementation batch should land one bounded CSC-backed public
lifecycle semantics slice:

1. prove that large-`n` CSC-backed Cholesky refactor failure preserves the old
   usable factors on the public lifecycle path
2. prove that the same large-`n` lane rejects gross structure drift while still
   preserving old usable factors
3. only if the proof exposes a real gap, make the smallest
   `src/sparse_analysis.c` follow-through needed to keep factor/refactor swap
   semantics uniform and explicit

## Exact Touched-File Fence

Required:

- `tests/test_integration.c`

Likely:

- `src/sparse_analysis.c`

Likely header truth follow-through only if semantics actually move:

- `include/sparse_analysis.h`

Optional only if the proof burden forces it:

- `tests/test_chol_csc.c`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

## Intended Proof Shape

The proof should stay in `tests/test_integration.c`, not widen into a new
harness.

Best shape:

- start from the existing large-`n` Cholesky public lifecycle lane (`n = 120`)
- build a usable baseline factor and solve
- then refactor on:
  - a same-pattern but no-longer-SPD matrix, or
  - a gross-structure-drift matrix,
  whichever closes the strongest missing CSC-backed retention fact first
- prove the failing refactor returns the expected error
- prove a later solve with the old factors still succeeds and matches the
  pre-failure solution

That keeps the proof aligned with the strongest public contract:

- reuse preserves symbolic/permutation setup
- failed refactor does not silently destroy the previous usable numeric factor

## Explicit Non-Goals

Day 10 should not widen into:

- LU wrapper follow-through
- LDL^T symmetry cleanup
- QR comparison work
- benchmark-governance or packaging/platform work
- broad docs/example cleanup unless the landed semantics actually require a
  wording correction

## Exit State

Sprint 63 Day 9 closes with one exact implementation fence:

- the shared direct lifecycle mechanism is already mostly right
- the missing strength is explicit large-`n` CSC-backed Cholesky
  failure-preserve proof, with code follow-through only if the proof exposes a
  real semantics gap
- Day 10 can now land a bounded `tests/test_integration.c`-first batch instead
  of reopening a general lifecycle redesign
