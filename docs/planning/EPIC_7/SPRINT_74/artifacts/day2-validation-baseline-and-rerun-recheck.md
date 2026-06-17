# Sprint 74 Day 2: Validation Baseline and Rerun Recheck

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Reconfirm the Sprint 74 implementation-day validation contract and fix the
highest-signal rerun set before any capability-modernization work lands.

## Main Result

Sprint 74 now has an explicit code-day validation split and rerun set tied to
the live capability-risk surface.

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
- substantial architecture or capability-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split

The Day 2 recheck confirms the live local split:

- the reviewed CMake tree currently owns the key matrix, integration,
  iterative, and eigensolver proof-owner tests, representative examples, and
  maintained capability benchmark binaries
- maintained install/package proof remains script-owned
- the root `build/` tree is not currently carrying the usual maintained
  capability benchmark binaries, so Sprint 74 should not assume that split
  until a later landing explicitly materializes them again

## High-Signal Sprint 74 Rerun Set

- matrix and direct-workflow proof owners:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
- scalar/callback and algorithm-breadth proof owners:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained capability benchmark/reporting surfaces currently materialized in
  the reviewed tree:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_iterative_reuse`
  - `./build/quality-review-cmake/bench_eigs_reuse`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

## Exit State

Sprint 74 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one truthful live proof-surface map
4. one exact rerun set for the strongest likely Sprint 74 capability lanes
