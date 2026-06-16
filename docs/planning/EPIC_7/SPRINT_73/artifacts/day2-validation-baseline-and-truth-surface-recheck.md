# Sprint 73 Day 2: Validation Baseline and Truth-Surface Recheck

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Reconfirm the Sprint 73 implementation-day validation contract and fix the
highest-signal rerun set before any configuration-modernization work lands.

## Main Result

Sprint 73 now has an explicit code-day validation split and rerun set tied to
the live residual-control risk surface.

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
- substantial architecture or precedence-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split

The Day 2 recheck confirms the live local split:

- the reviewed CMake tree currently owns the key graph/reorder proof-owner
  tests, representative examples, and reorder benchmark binaries
- maintained install/package proof remains script-owned
- the root `build/` tree is not currently carrying the usual maintained
  benchmark binaries, so Sprint 73 should not assume that split until a later
  landing explicitly materializes them again

## High-Signal Sprint 73 Rerun Set

- graph/FM proof owners:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
- reorder/precedence proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
- compatibility/support proof owners:
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_framework_optin`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained reorder/reporting surfaces currently materialized in the reviewed
  tree:
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

## Exit State

Sprint 73 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one truthful live proof-surface map
4. one exact rerun set for the strongest likely Sprint 73 configuration lanes
