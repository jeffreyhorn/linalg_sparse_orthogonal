# Sprint 65 Day 11: Local/CI-Friendly Regression Checks

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Add one bounded threshold-free reporting surface for the canonical maintained
benchmark lane without creating a noisy new timing gate or widening CI claims
past the current reviewed truthfulness contract.

## Landed Scope

This batch intentionally stays limited to:

- `Makefile`
- `scripts/bench_canonical_report.sh`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

It intentionally does not widen into:

- benchmark `*.c` sources
- solver implementation files
- public headers
- CI workflow expansion
- new timing thresholds or machine-class baselines

## Landed Reporting Surface

The new bounded reporting target is:

- `make bench-canonical-report`

It now:

- runs the four canonical maintained benchmark binaries:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- writes one CSV per benchmark under:
  - `build/bench-reports/canonical/`
- writes one `manifest.txt` with the exact fixture/command mapping

The selected report contract is intentionally narrow:

- `bench_refactor_csc`:
  - `tests/data/suitesparse/nos4.mtx --repeat 1`
- `bench_chol_csc`:
  - `tests/data/suitesparse/nos4.mtx --repeat 1`
- `bench_iterative_reuse`:
  - default corpus
- `bench_eigs_reuse`:
  - default corpus

## Why This Batch Matters

Interpretation:

- the canonical maintained surface is now easy to capture as a stable artifact
  set
- the report stays threshold-free, so it supports branch-to-branch comparison
  without pretending single-run timings are a portable pass/fail truth
- the Sprint 65 benchmark-governance model is now clearer:
  - `bench-canonical-report` = threshold-free reporting
  - `bench-fast` = bounded runtime lane
  - `wall-check` = narrow thresholded historical gate

## Direct Report Check

The direct Day 11 report-target check was:

- `make bench-canonical-report`

It wrote:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/manifest.txt`

Representative retained rows are:

- `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,0.488,0.214,0.142,0.010,0.006,1.51,8.24e-16,7.06e-16`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,100,594,scalar,supernodal,builtin,0.379,0.453,0.426,0.008,0.005,0.005,0.84,0.89,7.06e-16,5.89e-16,5.89e-16`
- `bench_iterative_reuse,proof,cg-tridiag-300,iter_handle_reuse,cg,300,400,59.2050,56.8200,1.04,17,17,5.192e-11,5.192e-11,1,1,OK,OK`
- `bench_eigs_reuse,proof,growm-nos4-k5,eigs_handle_reuse,lanczos_growm,100,5,40,2.2300,2.1590,1.03,115,115,5,5,4.326e-14,4.326e-14,100,100,0.000e+00,0.000e+00,lanczos_growm,OK,OK`

## Validation

No `*.c` / `*.h` files changed in Day 11, but this still altered the live
benchmark-governance execution surface, so the stronger reviewed validation
path remained the correct closeout gate:

- `make quality-review-full`

The Day 11 report target also ran directly before the reviewed gate:

- `make bench-canonical-report`

All passed. The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 580.39 sec`

One non-blocking note remains explicit: the reviewed CMake path still spent
most of its wall time in `test_reorder_nd`, but the full reviewed path
completed cleanly and passed all parity gates.

## Day 11 Exit State

Sprint 65 now has:

- one threshold-free canonical report target for the maintained benchmark
  surface
- one cleaner split between canonical reporting, bounded runtime checks, and
  the existing narrow thresholded gate
- one smaller and more maintainable reporting story to carry into Day 12 docs
  alignment
