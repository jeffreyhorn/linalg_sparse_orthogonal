# Sprint 83 Day 1: Scope and Capability Baseline

## Purpose

Turn the Sprint 83 project-plan section and the Sprint 82 validated closeout
into one bounded capability-surface execution package before any capability-
aware code lands.

## Starting Truth

Sprint 83 begins from a validated Sprint 82 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 82 already moved the strongest prior contradiction:

- the builtin scalar dense/backend ceiling was reduced on the highest-value
  Cholesky and LDL^T CSC lanes
- optional acceleration now sits behind one bounded backend/runtime contract
  rather than another generic backend bucket

That means Sprint 83 can start from the next real Epic 8 contradiction center:

- current real-only and compile-time-bounded capability surface on the
  highest-value public seams

## Sprint 83 Workstreams

The highest-value Sprint 83 package is now fixed explicitly around:

- capability re-rank
- scalar / index architecture design
- first scalar-surface expansion on the highest-value public seams
- touched-path index / ABI follow-through
- one bounded algorithm-surface widening lane
- focused regression / docs / package alignment only where implementation
  truly moves the contract
- validation and closeout

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 83 surfaces:

- shared/public contract surfaces:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- shared implementation and family-local widening seams:
  - `src/sparse_types.c`
  - `src/sparse_matrix.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
- strongest proof and reporting surfaces:
  - `tests/test_sparse_matrix.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Preserved Fence

Sprint 83 is explicitly bounded against:

- reopening Sprint 82's dense/backend ABI fence as the first implementation
  center
- repo-wide complex-number promises
- broad mixed-precision frameworks
- generic package/platform maturity widening
- premature algorithm-family widening before the shared scalar/index contract is
  explicit
- benchmark-governance drift
- support-surface churn detached from a real landed capability seam

## Day 1 Result

Sprint 83 now starts from one precise capability-surface execution package
rather than from a generic “capability expansion” bucket. The strongest likely
touch surfaces, preserved non-goals, and validated baseline are fixed in
writing before the validation/proof recheck begins.
