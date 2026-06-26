# Sprint 92 Day 7: Post-Landing Audit & Rerank

## Purpose

Re-rank the remaining backend-maturity work after the Day 6 landing so Sprint
92's second implementation center is chosen from live post-landing evidence
rather than from the original Day 3 hotspot map alone.

## Main Result

The Day 6 landing closed the strongest first Sprint 92 contradiction:

- the shared dense owner no longer lacks a real bounded optional portable
  backend seam
- Cholesky no longer depends only on a narrower family-local acceleration
  pocket to expose optional dense-kernel acceleration
- the backend runtime contract now has one shared visible naming surface:
  - `builtin`
  - `accelerate`
  - `blas-lapack`

That changes the ranked remaining backend map to:

- strongest first target now:
  - direct-family backend adoption convergence centered on
    `src/sparse_ldlt_csc.c`
- strongest second target now:
  - QR and later dense-consumer adoption only after LDLT stops lagging the
    widened shared dense seam
- strongest third target now:
  - bounded benchmark/proof observability once the strongest direct-family
    adopters actually share the widened seam
- strongest support-only but real target now:
  - build/package/support wording only where later observability work truly
    changes the maintained backend contract

## Why The Rerank Changed

Day 6 materially changed the backend reading in one important way:

- `src/sparse_dense.c` now owns one widened optional external
  BLAS/LAPACK-class dense-kernel seam
- builtin fallback remains authoritative and always available
- `tests/test_chol_csc.c` now proves the shared seam through the Cholesky-side
  environment contract
- the build surfaces already carry the minimum forced non-Apple `dlopen`
  linkage follow-through

That means the strongest remaining contradiction is no longer "does the repo
have a bounded portable backend seam at all?" It is now "do the strongest
direct-family dense consumers actually converge on that shared seam?"

## Strongest Remaining Contradiction

The strongest remaining contradiction is now direct-family backend adoption
convergence:

- `src/sparse_ldlt_csc.c` still carries its own bounded family-local
  Accelerate-only dense-factor selection seam
- the LDLT dense backend contract still reads narrower than the new Cholesky
  shared-seam contract
- that duplication now outranks benchmark wording, package wording, or QR
  adoption because it still limits backend maturity on one of the strongest
  direct consumers

This is now the highest-value next move because:

- it stays inside Sprint 92's intended direct-family adoption lane
- it sharpens backend coherence instead of widening claims
- it is a more immediate maturity win than pushing benchmark or doc work ahead
  of a still-split implementation story

## Exact Day 8 Design Center

The exact Day 8 design center is now fixed to:

- `src/sparse_ldlt_csc.c`

The strongest support-only follow-through, only if the Day 8 contract truly
forces movement, is:

- `src/sparse_ldlt_csc_internal.h`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- `benchmarks/bench_refactor_csc.c`

## Explicit Non-Needs After Day 6

Sprint 92 no longer needs:

- a second immediate shared-dense-owner implementation batch
- benchmark or reporting widening before the strongest direct-family adoption
  seam lands
- QR/backend adoption as the next center
- build/package wording churn detached from real adoption movement

## Exit State

- The strongest remaining Sprint 92 seam is now explicit after the first
  backend landing.
- The second implementation center stays code-owned and is fixed to LDLT
  backend-adoption convergence.
- Day 8 can now define one exact bounded LDLT adoption contract from the live
  post-Day-6 tree.
