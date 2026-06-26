# Sprint 92 Day 8: Solver Adoption Follow-Through Design

## Purpose

Freeze one exact Day 9 adoption contract so Sprint 92 converges the strongest
remaining LDLT dense-backend seam onto the widened shared dense owner without
turning the second batch into a broad direct-family rewrite.

## Main Result

Sprint 92 now has one exact second implementation contract:

- required Day 9 center:
  - `src/sparse_ldlt_csc.c`
- directly forced support-only follow-through only if the Day 9 contract truly
  needs them:
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
- strongest later surfaces only if LDLT adoption exposes a real shared seam
  that truly needs them:
  - `benchmarks/bench_refactor_csc.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Exact Day 9 Target

The exact Day 9 target is now explicit:

- stop treating LDLT dense backend selection as a family-local
  Accelerate-only side path
- converge LDLT onto the widened shared builtin-vs-portable backend reading
  already landed on the Cholesky side
- preserve builtin fallback truth and keep the backend contract bounded

In practical terms, Day 9 should center on:

- `src/sparse_ldlt_csc.c` runtime/backend selection
- adoption of the shared backend reading rather than a second parallel
  backend story
- only the proof-owner and internal-header changes truly needed to validate
  that adoption

## Strongest Clarification

The strongest Day 8 clarification is now explicit:

- Day 9 should not become a generic LDLT numeric rewrite
- Day 9 should not widen to QR adoption
- Day 9 should not shift early to benchmark/reporting work
- Day 9 should not reopen package/install/workflow wording detached from a
  real adoption movement

## Deferred Behind Day 9

The following remain later unless the LDLT adoption batch truly forces them:

- QR/backend convergence
- benchmark and observability widening in `bench_refactor_csc.c`
- README / install / maintainer wording
- broader build/package/workflow follow-through

## Exit State

- The second Sprint 92 implementation center is fixed.
- Day 9 will stay code-owned and bounded to LDLT backend-adoption convergence.
- Later proof/benchmark/support work remains sequenced behind a real landed
  adoption batch.
