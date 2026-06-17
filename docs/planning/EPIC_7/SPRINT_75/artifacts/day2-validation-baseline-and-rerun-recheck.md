# Sprint 75 Day 2: Validation Baseline and Rerun Recheck

Date: 2026-06-17
Branch: `sprint-75`

## Purpose

Reconfirm the Sprint 75 implementation-day validation contract and the live
proof-surface split before any backend-aware landing work begins.

## Strongest Reviewed Baseline

Sprint 75 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Validation Authority Split

The Sprint 75 validation contract is now explicit:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial backend or architecture batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Proof-Surface Split

The Day 2 recheck fixes the current local proof split:

### Reviewed CMake tree

The reviewed CMake tree currently owns the key Sprint 75 backend/performance
proof surfaces:

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `./build/quality-review-cmake/bench_svd`

### Install/package proof

Maintained install/package proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## High-Signal Sprint 75 Rerun Set

The strongest likely rerun set for Sprint 75 is:

### Backend-aware solver proof owners

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_eigs`

### Representative adoption surfaces

- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

### Maintained benchmark/reporting surfaces

- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `./build/quality-review-cmake/bench_svd`

### Maintained install/package proof

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Truthfulness Reading

Sprint 75 must preserve this reading:

- `make quality-review-full` remains the strongest local reviewed baseline
- benchmark binaries remain proof/reporting surfaces, not portable timing gates
- install/package regressions remain maintained proof surfaces, but do not by
  themselves widen the reviewed platform contract
- callback/runtime follow-through must preserve the current family-local
  cancellation and observability truth

## Exit State

Sprint 75 Day 2 closes with:

1. one explicit implementation-day validation contract
2. one fixed live proof-surface split across reviewed tests, examples,
   benchmarks, and install scripts
3. one high-signal rerun set for later backend-aware landings
