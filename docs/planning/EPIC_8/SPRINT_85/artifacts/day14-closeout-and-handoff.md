# Sprint 85 Day 14: Closeout and Handoff

## Purpose

Close Sprint 85 from the validated Day 13 baseline and leave one explicit
handoff queue for Sprint 86 and the later Epic 8 implementation sprints.

## Closeout State

Sprint 85 now closes as one coherent Epic 8 maintainability-modernization
package across:

- hotspot rerank
- bounded decomposition / ownership architecture contract
- Day 6 bounded iterative-source cleanup
- Day 9 bounded direct-family hotspot cleanup
- Day 11 bounded giant-test architecture cleanup
- validated Day 13 close baseline

The preserved fence stayed intact:

- Sprint 85 reduced maintainability cost on touched hotspot owners instead of
  reopening Sprint 84 assurance widening
- the first cleanup stayed source-owned inside `src/sparse_iterative.c`
- the direct-family cleanup stayed bounded to the embedded dense LDL^T /
  backend seam rehomed to the LDL^T CSC owner
- the giant-test cleanup stayed inside the retained Cholesky CSC proof owner
- adjacent proof-owner tests remained retained validation surfaces rather than
  becoming redistribution targets
- benchmarks and examples still did not become correctness owners
- package, install, export, runtime-package, and reviewed-Windows claims were
  not widened beyond the untouched mechanics

## Project-Plan Recheck

`docs/planning/EPIC_8/PROJECT_PLAN.md` does not need a Sprint 85 correction.

The landed Sprint 85 package still supports the intended Epic 8 execution
order:

1. Sprint 86: reviewed runtime convergence and reordering-scalability work
   after the highest remaining maintainability hotspots were reduced
2. later bounded follow-through on adjacent large sources and proof hotspots
   only where the refreshed hotspot map justifies more extraction
3. later package/platform/runtime maturity only where touched mechanics
   justify broader support claims

## Validated Baseline

Sprint 85 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 404.15 sec`
- `./build/quality-review-cmake/test_iterative` -> `80 / 80`
- `./build/quality-review-cmake/test_chol_csc` -> `151 / 151`
- `./build/quality-review-cmake/test_integration` -> `56 / 56`
- `./build/quality-review-cmake/test_ldlt` -> `87 / 87`
- `./build/quality-review-cmake/test_qr` -> `73 / 73`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

This means Sprint 85 hands off from one measured maintainability baseline
rather than from decomposition intent alone.

## Handoff Queue

The ranked carry-forward queue from Sprint 85 is now fixed explicitly:

1. reviewed runtime convergence and reordering-scalability work after the
   highest remaining maintainability hotspots were reduced
2. later bounded follow-through on adjacent large sources and proof hotspots
   only where the refreshed hotspot map justifies more extraction
3. later package/platform/runtime maturity only where touched mechanics
   justify broader support claims

## Bottom Line

Sprint 85 achieved its purpose: the project now has one clearer iterative
source hotspot, one better-owned direct-family dense LDL^T seam under the LDL^T
CSC owner instead of the Cholesky CSC hotspot, one less costly Cholesky CSC
giant-test registration block, and one validated close baseline with the
maintainability work kept bounded to what the sprint actually touched. Sprint
86 can now move to reviewed runtime and reordering-scalability work on top of
cleaner hotspot ownership instead of reopening the same decomposition questions
first.
