# Sprint 92 Day 11: Observability and Build Alignment Batch

## Purpose

Land the bounded Sprint 92 observability batch so the retained repeated-run
LDLT benchmark exposes backend request, actual selected backend, and fallback
behavior for the widened dense backend seam without widening to QR, broad
reporting rewrites, or package claims.

## Main Result

Sprint 92 Day 11 landed one bounded observability batch:

- required implementation center:
  - `benchmarks/bench_refactor_csc.c`
- directly forced support-only follow-through:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`

The landed result is:

- `bench_refactor_csc` CSV rows now include:
  - `ldlt_dense_backend_request`
  - `ldlt_dense_backend_selected`
  - `ldlt_dense_backend_fallback`
- default SPD / Cholesky rows keep those fields as `n/a`
- `--indefinite-kkt` now makes the widened LDLT backend seam observable from
  the retained repeated-run benchmark owner
- fallback is explicitly visible when an external-capable request normalizes
  back to builtin

## Kept Boundary

The Day 11 batch stayed inside the Day 10 fence:

- no QR adoption work
- no LDLT correctness-test widening
- no canonical report script rewrite
- no Makefile or CMake follow-through
- no README or install-surface wording changes
- no portable-performance or platform-symmetry overclaim

## Validation

The required implementation-day queue passed cleanly:

- `make format`
- `make lint`
- `make test`

Focused observability reruns also passed:

- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
- `SPARSE_LDLT_DENSE_BACKEND=external ./build/bench_refactor_csc --indefinite-kkt --repeat 1`

Observed live output on this machine:

- default request:
  - `ldlt_dense_backend_request=builtin`
  - `ldlt_dense_backend_selected=builtin`
  - `ldlt_dense_backend_fallback=no`
- explicit external request:
  - `ldlt_dense_backend_request=external`
  - `ldlt_dense_backend_selected=accelerate`
  - `ldlt_dense_backend_fallback=no`

## Strongest Outcome

The strongest Day 11 outcome is that Sprint 92's widened LDLT backend seam is
now benchmark-visible rather than only test-visible:

- the retained LDLT proof owner still covers correctness and env-contract
  behavior
- the retained repeated-run benchmark owner now exposes the same backend story
  as observable workflow evidence
- benchmark-side fallback visibility no longer depends on reading internal code
  or test-only helper names

## Exit State

- Sprint 92 has completed its bounded observability batch.
- The retained LDLT repeated-run benchmark now exposes backend request,
  backend selection, and fallback state in its CSV output.
- Day 12 can now freeze the final owner map and validation queue from a live
  post-Day-11 tree.
