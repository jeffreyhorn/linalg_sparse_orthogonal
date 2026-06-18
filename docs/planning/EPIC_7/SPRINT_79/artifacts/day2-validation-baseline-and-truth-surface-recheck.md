# Sprint 79 Day 2 - Validation Baseline and Truth-Surface Recheck

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Reconfirm the Sprint 79 implementation-day validation contract and the live proof-surface split across reviewed CMake proof owners, representative examples, canonical report command surfaces, and install/export proof owners before any final assurance work lands.

## Main Result
Sprint 79's implementation-day validation and truth-surface contract is now explicit before any final oracle, property, lifecycle, or integration batch lands.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity remains the main truthfulness anchor:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The Sprint 79 authority split is now fixed explicitly:
- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial assurance or integration batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split
The reviewed CMake tree currently owns the key Sprint 79 proof surfaces most likely to be stressed by final assurance and integration work:
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/test_fuzz`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

The canonical benchmark/reporting lane remains command- and script-owned rather than reviewed-binary-owned:
- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters consumed by that command:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

Maintained install/package proof remains script-owned:
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## High-Signal Ownership Reading
The strongest current proof and truth-surface ownership split is already stable:
- `tests/test_chol_csc.c` owns family-local large-`n` Cholesky CSC handoff, publish-back, and backend-contract proof
- `tests/test_integration.c` owns public one-shot vs explicit repeated-run lifecycle parity plus callback/cancel truth
- `tests/test_fuzz.c` remains bounded seeded generative follow-through for retained lifecycle/property pressure
- `benchmarks/README.md` and `docs/maintainer_guide.md` already keep the canonical benchmark/reporting and threshold-free reading explicit
- `INSTALL.md` plus the install scripts already keep the local install/export proof split explicit

## Sprint 79 Rerun Set
The highest-signal rerun set is now fixed around the likely touched final-closeout seams:
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/test_fuzz`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Interpretation
The useful Day 2 conclusion is now explicit:
- Sprint 79 does not need a new validation policy.
- It should preserve the reviewed/test-owned/script-owned split already established across Sprint 71 through Sprint 78.
- Final assurance work should therefore target the highest-value residual proof or truth gaps, not reopen stable ownership boundaries.

## Exit State
- The reviewed parity anchor is current.
- The live proof split across reviewed binaries, command-owned benchmark reporting, and script-owned install/export proof is fixed in writing.
- Sprint 79 can now move into an assurance-gap audit from a precise Day 2 validation baseline.
