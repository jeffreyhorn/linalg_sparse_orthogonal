# Sprint 66 Day 13: Full Validation Sweep

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Run the full Sprint 66 validation sweep from the landed packaging/install/workflow
state and reconfirm the touched install/package proof surfaces before the final
closeout day.

## Core Validation

The full validation sweep passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real)` = `558.62 sec`

## Focused Install and Package Proof

Focused install/package proof reruns also passed:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained proof points:

- Make install/uninstall path passed
- CMake install/export/find-package path passed
- both scripts reported installed `pkg-config` version `2.2.0`
- `tests/test_install.sh` passed `11 / 11`
- `tests/test_cmake_install.sh` passed `13 / 13`

This confirms the tightened version-source-of-truth rule remains stable on both
maintained local install/package proof surfaces.

## Canonical Benchmark Report

The canonical maintained performance snapshot also reran cleanly:

- `make bench-canonical-report`

Generated report set:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/manifest.txt`

Representative retained rows:

- `bench_refactor_csc,proof,nos4.mtx,chol_spd,...,1.87,8.24e-16,7.06e-16`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,...,scalar,supernodal,builtin,...,0.83,0.92,...`
- `bench_iterative_reuse,proof,cg-tridiag-300,iter_handle_reuse,cg,...,1.07,...`
- `bench_eigs_reuse,proof,growm-nos4-k5,eigs_handle_reuse,lanczos_growm,...,1.08,...`

This confirms the normalized canonical maintained benchmark surface remains
intact after the Sprint 66 packaging/platform closeout work.

## Notes

One non-blocking note remained unchanged from the reviewed path:

- `test_reorder_nd` was still the dominant reviewed CMake test at `363.69 sec`
  out of the `558.62 sec` total

That is a known existing runtime characteristic, not a new Sprint 66 regression.

## Exit State

Sprint 66 Day 13 closes with:

- one revalidated reviewed baseline
- one revalidated install/package proof pair under the tightened version contract
- one revalidated canonical benchmark snapshot surface
- one clean Day 14 closeout starting point
