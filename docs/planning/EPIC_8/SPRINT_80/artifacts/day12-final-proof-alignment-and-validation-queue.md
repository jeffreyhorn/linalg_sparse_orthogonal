# Sprint 80 Day 12: Final Proof Alignment and Validation Queue

## Purpose

Fix the exact proof-owner map and authoritative Day 13 validation queue for the
Sprint 80 baseline-and-contract package.

## Live Anchor Recheck

The strongest reviewed parity anchor remains explicit:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The highest-signal reviewed proof-owner binaries most relevant to Sprint 80 are
present and stable:

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

The maintained canonical benchmark/reporting lane remains command- and
script-owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

The maintained install/export proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Focused Proof Ownership

Sprint 80 remains a baseline-and-contract sprint, so the strongest proof split
is still:

- reviewed CMake tests and representative examples:
  - executable truth for the core solver, lifecycle, and regression surfaces
- canonical benchmark/report command surface:
  - performance/reporting ownership only
  - not oracle or pass/fail timing-gate ownership
- install/export scripts:
  - local direct proof for the static-first package surface

No extra focused regression or support-surface change is required before Day
13.

## Authoritative Day 13 Validation Queue

Run the following:

1. `make format`
2. `make lint`
3. `make test`
4. `make quality-review-full`
5. `ctest -N --test-dir build/quality-review-cmake`
6. `./build/quality-review-cmake/test_chol_csc`
7. `./build/quality-review-cmake/test_ldlt_csc`
8. `./build/quality-review-cmake/test_ldlt`
9. `./build/quality-review-cmake/test_iterative`
10. `./build/quality-review-cmake/test_qr`
11. `./build/quality-review-cmake/test_integration`
12. `./build/quality-review-cmake/test_reorder_nd`
13. `./build/quality-review-cmake/test_fuzz`
14. `./build/quality-review-cmake/test_eigs`
15. `./build/quality-review-cmake/example_analysis`
16. `./build/quality-review-cmake/example_basic_solve`
17. `make bench-canonical-report`
18. `bash tests/test_install.sh`
19. `bash tests/test_cmake_install.sh`

## Retained Closeout Anchors

Day 13 and Day 14 should retain at minimum:

- reviewed CMake parity count
- Makefile/CMake parity status
- full reviewed CTest pass count
- total reviewed runtime
- canonical benchmark/report command success
- install/export script pass counts
- any representative retained outputs needed to prove the baseline stayed
  stable

## Day 12 Exit State

Sprint 80 now has one explicit final proof-owner map and one authoritative Day
13 validation queue. No validation ambiguity remains around the refreshed
baseline package.
