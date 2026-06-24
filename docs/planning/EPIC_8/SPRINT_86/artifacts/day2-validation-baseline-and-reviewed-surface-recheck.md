# Sprint 86 Day 2: Validation Baseline and Reviewed-Surface Recheck

## Purpose

Refresh the implementation-day validation contract and the live reviewed
proof-owner split before Sprint 86 changes any reorder, ND, or reviewed-runtime
surface.

## Reviewed Validation Contract

Sprint 86 continues to inherit the strongest local reviewed baseline:

- `make quality-review-full`

The code-day and docs-day split is now fixed explicitly:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial runtime, proof-surface, or reviewed-path batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

Reviewed CMake parity remains the primary truthfulness anchor:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Reviewed Proof Owners

The reviewed CMake tree currently owns the strongest early-Sprint-86 proof
and runtime surfaces:

- reorder and ND reviewed proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- reviewed benchmark follow-on binaries:
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_fillin`

## Canonical Reporting and Install Ownership

Canonical benchmark reporting remains command- and script-owned rather than
reviewed-binary-owned:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- root `build/` canonical emitters:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

Maintained install/package proof remains script-owned:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Day 2 Result

Sprint 86 now has one explicit validation and reviewed-surface contract before
the runtime long-pole audit begins:

- reviewed CMake reorder/ND proof-owner tests and representative examples
  remain the main executable truth surfaces
- reviewed benchmark binaries remain benchmark-side measurability surfaces
- canonical benchmark reporting remains command/script owned
- install/export proof remains script owned

The highest-signal rerun set is now fixed around:

- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/test_reorder`
- `./build/quality-review-cmake/test_reorder_amd_qg`
- `./build/quality-review-cmake/test_graph`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_reorder`
- `./build/quality-review-cmake/bench_fillin`
- `make bench-canonical-report`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
