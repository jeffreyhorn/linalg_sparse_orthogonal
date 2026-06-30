# Post-Epic-9 Handoff

## Purpose

Carry forward the explicit residual work and non-claims from Epic 9 without
weakening the validated Sprint 99 baseline.

This handoff is based on:

- [Sprint 99 final residual queue](./SPRINT_99/artifacts/day9-final-residual-queue.md)
- [Sprint 99 closeout evidence package](./SPRINT_99/artifacts/day12-closeout-evidence-package.md)
- [Epic 9 retrospective](./EPIC_9_RETROSPECTIVE.md)

## Validated Baseline

Sprint 99 closed with:

- `make quality-review-full` passed
- Makefile/CMake test-count parity: 54/54
- full CTest: 54 passed, 0 failed
- `bash tests/test_install.sh`: 14 passed, 0 failed
- `bash tests/test_cmake_install.sh`: 16 passed, 0 failed, 0 skipped
- `make examples`: 12 example binaries built
- representative examples executed successfully:
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- `make bench-reorder-sprint86` passed
- `make bench-canonical-report` passed

## Carry-Forward Queue

| Item | Owner surface | Next-step expectation |
|---|---|---|
| broader LDLT CSC Matrix Market or indefinite corpus comparison | `tests/test_ldlt_csc.c`, `tests/ldlt_external_dense_reference.py`, future fixtures | design reference architecture before adding fixtures |
| iterative solver external comparison architecture | iterative tests and future reference scripts | define convergence, preconditioning, restart, and residual semantics first |
| eigensolver/LOBPCG external comparison architecture | eigensolver tests and future reference scripts | define fixture, tolerance, cluster, and runtime bounds first |
| QR/SVD external comparison architecture | `tests/test_qr.c`, `tests/test_svd.c`, future references | define family-specific reference and tolerance ownership first |
| generated reorder/fill report target if repeated captures justify it | `benchmarks/bench_reorder.c`, scripts, benchmark docs | preserve `nnz_L` as primary fill field and avoid portable timing thresholds |
| continued large-source extraction | largest source owners such as QR, eigs, LU CSR, LDLT, matrix, SVD | design family-local extraction boundaries and run focused plus full C validation |
| continued giant-test extraction | `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_integration.c`, and adjacent owners | split with registration parity checks and focused tests |
| lower-level chronology cleanup where useful | lower-level tests, implementation comments, selected docs | avoid compatibility-breaking renames without migration plans |

## Non-claims To Preserve

- full compressed-first replacement of the linked-list shell
- broad complex support
- broad mixed-precision maturity
- broad backend-neutral acceleration maturity
- shared-library-first package contract
- dynamic ABI guarantee
- symmetric Linux/macOS/Windows reviewed parity
- Windows Makefile parity or install-validation lane
- portable timing superiority or universal reorder/fill superiority
- every-solver-family external correctness comparison

## Validation Rules For Future Work

- If `.c` or `.h` files change, run the required full quality chain:
  `make format && make lint && make test`.
- If build, CMake, install/export, workflow, script, benchmark, or package
  surfaces change, run the focused proof commands that own that surface.
- If comparison or benchmark claims change, write the claim boundary before
  implementation and keep timing fields out of portable product claims.
- If platform wording changes, re-check Linux/macOS/Windows scope statements
  and Windows expected-count assertions.
- If docs-only carry-forward work changes only planning artifacts, run
  `git diff --check` and a trailing-whitespace scan on touched docs.

## Recommended Starting Points

1. Start any external comparison expansion with a one-page proof architecture:
   fixture ownership, oracle behavior, tolerance model, runtime cost, and
   claim language.
2. Start any source/test extraction with a family-local owner map and focused
   validation list.
3. Start any package maturity work by deciding whether the existing
   static-first proof remains authoritative or whether a new package contract
   is being created.
4. Start any benchmark/reporting expansion by deciding whether repeated manual
   captures justify a generated target.
5. Keep the Day 12 supported and unsupported closeout language available during
   review.
