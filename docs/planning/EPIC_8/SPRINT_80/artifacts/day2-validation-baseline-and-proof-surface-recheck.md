# Sprint 80 Day 2 - Validation Baseline and Proof-Surface Recheck

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Reconfirm the Sprint 80 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, canonical benchmark/report command surfaces, and install/export proof
owners before any Epic 8 baseline or contract batch lands.

## Main Result
Sprint 80's implementation-day validation and truth-surface contract is now
explicit before any Epic 8 baseline, comparison-contract, or integration work
lands.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity remains the main truthfulness anchor:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

The Sprint 80 authority split is now fixed explicitly:
- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial baseline, comparison-contract, or integration batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split
The reviewed CMake tree currently owns the strongest early-Epic-8 proof
surfaces:
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

The canonical benchmark/reporting lane remains command- and script-owned rather
than reviewed-binary-owned:
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
The strongest current proof and truth-surface split is already stable:
- reviewed CMake proof-owner binaries and representative examples remain the
  main executable truth surfaces
- canonical benchmark reporting remains command/script owned
- install/export proof remains script owned

The useful Day 2 conclusion is now explicit:
- Sprint 80 does not need a new validation policy.
- It should preserve the reviewed-binary / command-owned / script-owned split
  already established by Epic 7.
- Early Epic 8 work should therefore target baseline, contract, and ranking
  contradictions rather than reopening stable proof ownership.

## Sprint 80 Rerun Set
The highest-signal rerun set is now fixed around the likely touched Sprint 80
baseline and contract seams:
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

## Exit State
- The reviewed parity anchor is current.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- Sprint 80 can now move into the competitive gap inventory and external
  comparison contract work from a precise Day 2 validation baseline.
