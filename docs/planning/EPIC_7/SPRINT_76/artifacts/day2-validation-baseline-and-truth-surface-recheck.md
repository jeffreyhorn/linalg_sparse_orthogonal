# Sprint 76 Day 2: Validation Baseline and Truth-Surface Recheck

Date: 2026-06-17
Branch: `sprint-76`

## Purpose

Reconfirm the Sprint 76 implementation-day validation contract and the live
truth-surface split across reviewed benchmark binaries, report-generation
workflow entry points, representative examples, and install/package proof
before any benchmark-governance or reporting landing work begins.

## Strongest Reviewed Baseline

Sprint 76 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Validation Authority Split

The Sprint 76 validation contract is now explicit:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial benchmark, workflow, or governance batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

## Live Truth-Surface Split

The Day 2 recheck fixes the current local proof and workflow split.

### Reviewed CMake tree

The reviewed CMake tree currently owns the key Sprint 76 benchmark-governance
proof surfaces:

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_iterative_reuse`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `./build/quality-review-cmake/bench_reorder`
- `./build/quality-review-cmake/bench_amd_qg`

### Canonical report-generation workflow

The threshold-free canonical reporting surface remains workflow and
script-owned rather than reviewed-binary owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`

The current workflow still consumes the root `build/` canonical emitters:

- `build/bench_refactor_csc`
- `build/bench_chol_csc`
- `build/bench_iterative_reuse`
- `build/bench_eigs_reuse`

### Install/package proof

Maintained install/package proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## High-Signal Sprint 76 Rerun Set

The strongest likely rerun set for Sprint 76 is:

### Maintained benchmark-governance proof owners

- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`

### Representative adoption surfaces

- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`

### Maintained benchmark/reporting surfaces

- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_iterative_reuse`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `./build/quality-review-cmake/bench_reorder`
- `./build/quality-review-cmake/bench_amd_qg`
- `make bench-canonical-report`

### Maintained install/package proof

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Truthfulness Reading

Sprint 76 must preserve this reading:

- `make quality-review-full` remains the strongest local reviewed baseline
- `make bench-canonical-report` remains the threshold-free maintained
  reporting surface
- reviewed benchmark binaries remain proof and reporting surfaces, not
  portable timing gates
- narrower exploratory or thresholded lanes must not silently widen into the
  canonical proof contract
- install/package regressions remain maintained proof surfaces, but do not by
  themselves widen the reviewed platform contract

## Exit State

Sprint 76 Day 2 closes with:

1. one explicit implementation-day validation contract
2. one fixed live truth-surface split across reviewed binaries, canonical
   reporting workflow ownership, and install/package proof
3. one high-signal rerun set for later governance and reporting landings
