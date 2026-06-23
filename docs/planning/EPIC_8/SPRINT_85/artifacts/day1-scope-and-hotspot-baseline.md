# Sprint 85 Day 1: Scope and Hotspot Baseline

## Purpose

Turn the Sprint 85 project-plan section and the Sprint 84 validated closeout
into one bounded maintainability execution package before any hotspot-aware
code lands.

## Starting Truth

Sprint 85 begins from a validated Sprint 84 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 84 already moved the strongest prior contradiction:

- one bounded maintained external differential lane landed on the
  direct-family SPD Cholesky CSC path
- one deeper deterministic large-`n` seeded-property owner landed
- one stronger shared retry-after-failure lifecycle proof owner landed

That means Sprint 85 can start from the next real Epic 8 contradiction center:

- the current implementation and giant-test maintainability ceiling on the
  highest-value touched source and proof-owner lanes

## Sprint 85 Workstreams

The highest-value Sprint 85 package is now fixed explicitly around:

- hotspot rerank
- decomposition / ownership architecture design
- iterative-source cleanup
- direct-family hotspot cleanup
- giant-test architecture cleanup
- focused proof/docs alignment only where implementation truly moves the
  contract
- validation and closeout

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 85 surfaces:

- shared/public and family-level contract surfaces:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- implementation hotspots:
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
- strongest proof, review, and support surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Preserved Fence

Sprint 85 is explicitly bounded against:

- reopening Sprint 84's bounded assurance-owner fence as the first
  implementation center
- repo-wide architectural cleanup claims detached from touched seams
- proof dilution from moving helpers without preserving owner boundaries
- benchmark-governance or example-governance drift into correctness ownership
- broad package/platform maturity widening
- support-surface churn detached from a real landed hotspot seam

## Day 1 Result

Sprint 85 now starts from one precise maintainability execution package rather
than from a generic “refactor more code” bucket. The strongest likely touch
surfaces, preserved non-goals, and validated baseline are fixed in writing
before the validation/proof recheck begins.
