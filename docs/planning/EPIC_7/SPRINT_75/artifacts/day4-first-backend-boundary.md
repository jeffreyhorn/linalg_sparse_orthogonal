# Sprint 75 Day 4 Artifact: First Backend Boundary

Date: 2026-06-17
Branch: sprint-75

## Purpose

Freeze the first Sprint 75 backend/policy fence so the next design pass starts
from one bounded implementation lane rather than from a generic backend or
performance-architecture backlog.

## Main Result

Sprint 75 now has one explicit first landing boundary:

- required first landing:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if the first landing forces it:
  - `include/sparse_cholesky.h`
  - `benchmarks/bench_chol_csc.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
- strongest deferred second lane:
  - `include/sparse_eigs.h`
  - `src/sparse_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `tests/test_eigs.c`

## Why This Is the Right First Fence

The CSC supernodal Cholesky lane remains the best first landing because it
already has the full bounded backend-aware architecture shape:

- one concrete dense-kernel owner
- one real shipped runtime descriptor
- one maintained benchmark-side proof surface
- one family-local regression owner
- one existing documented truthfulness contract

That gives Sprint 75 the strongest combination of:

- runtime leverage
- low compatibility risk
- manageable proof cost
- bounded payoff without overstating the product surface

## Support Surface Reading

The support surfaces are bounded rather than assumed:

- `include/sparse_cholesky.h`
  - move only if the first batch changes local callback, publish-back, or
    dense-kernel descriptor truth
- `benchmarks/bench_chol_csc.c`
  - move only if the first batch changes what must be made benchmark-visible
- `tests/test_chol_csc.c`
  - move only if the first batch changes correctness, fallback, or backend
    contract proof
- `tests/test_integration.c`
  - move only if a public-path guarantee actually changes
- `docs/maintainer_guide.md`
  - move only if the bounded backend contract itself becomes clearer in a way
    the policy surface should capture

## Explicit Deferred Set

The Day 4 deferred set is now fixed:

- eigensolver backend/runtime parity:
  - `include/sparse_eigs.h`
  - `src/sparse_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `tests/test_eigs.c`
- QR backend/performance follow-through:
  - `include/sparse_qr.h`
  - `src/sparse_qr.c`
  - `tests/test_qr.c`
- SVD backend/performance follow-through:
  - `include/sparse_svd.h`
  - `src/sparse_svd.c`
  - `tests/test_svd.c`
  - `benchmarks/bench_svd.c`
- broad docs/governance spill:
  - `README.md`
  - `benchmarks/README.md`

## Non-Goal Fence

The first Sprint 75 batch explicitly does not include:

- a broad backend abstraction-layer rewrite
- any fake optional-backend or shared-library maturity claim
- benchmark-threshold portability claims
- a repo-wide callback/cancellation uniformity campaign
- broad benchmark-governance or public-surface cleanup

## Day 5 Implication

The Day 5 design pass should therefore start from:

- exact first implementation center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if truly forced:
  - `include/sparse_cholesky.h`
  - `benchmarks/bench_chol_csc.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
- explicitly not next:
  - eigs lane
  - QR lane
  - SVD lane
  - broad benchmark refresh
  - broad public docs cleanup
