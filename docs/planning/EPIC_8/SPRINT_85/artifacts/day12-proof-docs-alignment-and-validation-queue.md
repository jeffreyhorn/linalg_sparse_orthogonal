# Sprint 85 Day 12: Proof / Docs Alignment & Validation Queue Freeze

## Purpose

Reconcile the touched Sprint 85 proof-owner and support surfaces against the
landed cleanup boundaries and freeze the exact Day 13 validation queue.

## Main Result

No new support-only edit is needed before the full validation sweep.

The final Sprint 85 touched-surface truth map is now explicit:

- adopted cleanup centers:
  - `src/sparse_iterative.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_chol_csc.c`
- retained proof owners, not Sprint 85 adopted cleanup centers:
  - `tests/test_iterative.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_qr.c`
- support-only surfaces that do not need new movement before Day 13:
  - `docs/maintainer_guide.md`
  - `README.md`

## Final Proof-Owner Map

The final Sprint 85 proof-owner split is now fixed:

- bounded iterative-source cleanup proof owner:
  - `tests/test_iterative.c`
- bounded direct-family and giant-test cleanup proof owner:
  - `tests/test_chol_csc.c`
- shared lifecycle/public-behavior proof owner:
  - `tests/test_integration.c`
- retained adjacent family proof owners:
  - `tests/test_ldlt.c`
  - `tests/test_qr.c`

The maintained reviewed examples and benchmark follow-ons relevant to the
Sprint 85 close baseline remain:

- `example_analysis`
- `example_basic_solve`
- `bench_refactor_csc`
- `bench_svd`

## Support-Surface Boundary

The support-surface reading stayed fixed:

- `docs/maintainer_guide.md` and `README.md` do not need another Sprint 85
  wording change before validation
- canonical benchmark/reporting remains command/script-owned through
  `make bench-canonical-report`
- install/export proof remains script-owned through:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

Sprint 85 did not reopen package, install/export, benchmark ownership, or
runtime-surface mechanics, so those surfaces remain out of the Day 13
validation core except for the retained canonical reporting command.

## Frozen Day 13 Queue

The exact Day 13 queue is now fixed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- focused reviewed proof owners:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- benchmark/reporting follow-ons:
  - `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
  - `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `make bench-canonical-report`

## Strongest Clarification

The useful Day 12 clarification is explicit now:

- Sprint 85 does not need another proof-owner or docs batch before validation
- the only authoritative correctness owners for the landed cleanup package are
  the retained reviewed proof-owner tests and examples already fixed above
- install/export proof remains real repo coverage, but it is not part of the
  Sprint 85 close queue because Sprint 85 did not touch those mechanics

## Exit State

- no support-only drift remains before Day 13
- the final validation queue is explicit and unambiguous
- Day 13 can execute from a fixed touched-surface truth map rather than
  re-deciding Sprint 85 scope
