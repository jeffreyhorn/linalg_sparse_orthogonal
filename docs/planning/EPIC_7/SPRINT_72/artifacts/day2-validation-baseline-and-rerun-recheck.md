# Sprint 72 Day 2: Validation Baseline and Rerun Recheck

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Reconfirm the Sprint 72 implementation-day validation contract and fix the
highest-signal rerun set before any ownership convergence work lands.

## Main Result

Sprint 72 now has an explicit code-day validation split and rerun set tied to
the live ownership-risk surface.

## Baseline Anchors

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Authority Split

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial architecture or ownership-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split

The Day 2 recheck confirms the live local split:

- reviewed CMake tree owns the key proof-owner tests and representative
  examples
- root `build/` owns the maintained benchmark binaries
- maintained install/package proof remains script-owned

## High-Signal Sprint 72 Rerun Set

- direct-workflow and ownership-boundary proof:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
- direct CSC-family proof owners:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
- likely support family proofs:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

## Exit State

Sprint 72 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one fixed local proof-surface map
4. one exact rerun set for the strongest likely Sprint 72 ownership lanes
