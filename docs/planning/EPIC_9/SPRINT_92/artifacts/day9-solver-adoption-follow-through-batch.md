# Sprint 92 Day 9: Solver Adoption Follow-Through Batch

## Purpose

Land the bounded Sprint 92 LDLT backend-adoption batch so the strongest
remaining direct-family dense consumer converges onto the widened shared
builtin-vs-portable backend seam without reopening broader family or
support-surface work.

## Main Result

Sprint 92 Day 9 landed one bounded LDLT backend-adoption batch:

- required implementation center:
  - `src/sparse_ldlt_csc.c`
- directly forced follow-through:
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_ldlt.c`

The landed result is:

- LDLT no longer depends on a family-local Accelerate-only dense backend side
  path
- LDLT now reads the same bounded backend contract already established on the
  shared dense owner and Cholesky side:
  - `builtin`
  - `accelerate`
  - `blas-lapack`
- builtin remains the authoritative correctness and fallback path when no
  optional external provider is available or when the requested provider does
  not match the live platform/provider state
- the public proof surface now covers the new `external` environment contract
  end-to-end through the retained LDLT proof owner

## Kept Boundary

The Day 9 batch stayed inside the Day 8 fence:

- no generic LDLT numeric rewrite
- no QR adoption batch
- no benchmark/reporting widening
- no README / install / maintainer wording changes
- no build/package/workflow follow-through

## Validation

The required implementation-day queue passed cleanly:

- `make format`
- `make lint`
- `make test`

One local lint-style preprocessor-guard issue surfaced while widening the
LDLT-side external-backend seam. It was corrected, and the full validation
queue then passed again from the top.

## Strongest Outcome

The strongest Day 9 outcome is that Sprint 92 no longer carries one widened
shared dense backend seam on the Cholesky side and one narrower family-local
backend seam on the LDLT side:

- the strongest direct-family dense adopters now share one bounded backend
  reading
- LDLT backend naming and fallback behavior now matches the post-Day-6 shared
  dense contract
- later QR adoption, benchmark observability, and support-surface wording can
  now build on a more coherent direct-family backend story

## Exit State

- Sprint 92 has completed its bounded LDLT backend-adoption batch.
- The strongest remaining backend question is no longer whether LDLT still
  lags the shared seam; it is where the next highest-value post-adoption
  follow-through should land.
- Day 10 can now rerank the remaining backend, proof, benchmark, and support
  surfaces from the live post-Day-9 tree.
