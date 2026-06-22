# Sprint 82 Day 10 - Benchmark, Differential, and Runtime Alignment Design

Date: 2026-06-19  
Branch: sprint-82

## Purpose

Fix the exact proof, benchmark, and runtime-alignment follow-through required
after the Day 6 and Day 9 backend landings without widening Sprint 82 into a
generic docs cleanup pass or another implementation batch.

## Main Result

Sprint 82 now has one exact Day 11 follow-through contract:

- required surface:
  - `docs/maintainer_guide.md`
- strongest support-only wording if the batch truly needs it:
  - `README.md`
  - `include/sparse_ldlt.h`
- lower-value support-only surfaces that do not currently need movement:
  - `benchmarks/README.md`
  - `benchmarks/bench_refactor_csc.c`
  - `tests/test_ldlt.c`

## Strongest Remaining Contradiction

The strongest current contradiction is narrow and explicit:

- `docs/maintainer_guide.md` still describes the backend-aware performance
  surface as if:
  - the first backend-aware lane is local only to CSC supernodal Cholesky
  - broader LDL^T backend-aware follow-through is still deferred
- that reading is now stale after Day 9:
  - Cholesky still owns the first backend-aware landing
  - but LDL^T now also owns a bounded optional dense-factor runtime seam on the
    CSC supernodal lane
  - the maintained interpretation should therefore widen from
    Cholesky-only to the bounded direct-family backend-aware surface that now
    exists in both lanes

## Differential and Benchmark Result

The proof and benchmark sides are already in the right owners:

- `tests/test_ldlt.c` already owns:
  - builtin env-selection proof
  - accelerate env-selection proof
  - solver-visible forced-CSC correctness through the widened selector seam
- `benchmarks/bench_refactor_csc.c` already stays correctly bounded:
  - repeated-run throughput/proof owner
  - not a differential-oracle or runtime-selector policy owner
- no additional proof-code expansion is required
- no benchmark binary or benchmark-output change is required

## Support-Surface Reading

The support-only wording lane is narrower than a generic cleanup pass:

- `README.md` already remains broadly truthful because it does not currently
  overclaim a wider LDL^T backend story
- `include/sparse_ldlt.h` already remains truthful because the Day 9 selector
  widened an internal dense-factor seam, not the public LDL^T backend enum or
  callback contract
- `benchmarks/README.md` does not need edits because benchmark reporting did
  not widen and still should not be reinterpreted as the owner of runtime
  selection truth

## Preserved Fence

- no more proof-code expansion
- no benchmark CSV or reporting churn
- no README/tutorial/examples sweep
- no package/platform or shared-library claim widening
- no reopening of QR or SVD backend work

## Exit State

- Sprint 82 now knows the exact Day 11 touch set.
- The strongest required follow-through is narrowed to the authoritative
  maintainer-policy reading of the widened backend surface.
- Day 11 can stay bounded instead of turning into a generic support-surface
  cleanup pass.
