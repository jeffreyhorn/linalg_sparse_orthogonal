# Sprint 84 Day 1: Scope and Assurance Baseline

## Purpose

Turn the Sprint 84 project-plan section and the Sprint 83 validated closeout
into one bounded numerical-assurance execution package before any assurance-
aware code lands.

## Starting Truth

Sprint 84 begins from a validated Sprint 83 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 83 already moved the strongest prior contradiction:

- the highest-value shared matrix-shell scalar-owner seam was widened
- the shared scalar/index vocabulary owner was reconciled
- one bounded QR public-header widening landed without overstating the shipped
  real-only scalar contract

That means Sprint 84 can start from the next real Epic 8 contradiction center:

- the current external-differential, seeded-property, and failure-path
  assurance ceiling on the highest-value touched shared/direct lanes

## Sprint 84 Workstreams

The highest-value Sprint 84 package is now fixed explicitly around:

- differential-proof audit
- oracle / property / failure-path architecture design
- first maintained direct-family external differential batch
- deterministic seeded property expansion
- failure-path numerical proof
- focused policy / CI / support-surface alignment only where implementation
  truly moves the contract
- validation and closeout

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 84 surfaces:

- shared/public and family-level contract surfaces:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- implementation and family-local assurance seams:
  - `src/sparse_matrix.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- strongest proof, benchmark, and reporting surfaces:
  - `tests/test_sparse_matrix.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Preserved Fence

Sprint 84 is explicitly bounded against:

- reopening Sprint 83's capability-surface owner fence as the first
  implementation center
- repo-wide maintained external-proof promises
- benchmark-governance drift into correctness ownership
- broad oracle dependency stories for untouched families
- generic package/platform maturity widening
- support-surface churn detached from a real landed assurance seam

## Day 1 Result

Sprint 84 now starts from one precise numerical-assurance execution package
rather than from a generic “more testing” bucket. The strongest likely touch
surfaces, preserved non-goals, and validated baseline are fixed in writing
before the validation/proof recheck begins.
