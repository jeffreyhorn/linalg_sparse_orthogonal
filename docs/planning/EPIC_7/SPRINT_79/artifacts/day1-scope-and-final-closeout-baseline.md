# Sprint 79 Day 1 - Scope and Final Assurance/Closeout Baseline

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Establish a precise Sprint 79 starting baseline for numerical assurance expansion, final cross-surface integration, and Epic 7 closeout using the live tree, the Sprint 78 handoff, and the current permanent proof-owner and truth-surface map rather than another broad planning reset.

## Main Result
Sprint 79 now starts from a precise assurance and final-closeout queue, not from another generic review pass and not from another broad product, capability, backend, benchmark, packaging, or platform batch.

The strongest local reviewed baseline is still:
- `make quality-review-full`

Reviewed CMake parity was re-materialized live and remains explicit:
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`

The highest-value Sprint 79 pressure is now clearly narrowed to:
- assurance-gap rerank
- bounded oracle/property/lifecycle regression expansion
- final cross-surface integration sweep
- full validation sweep
- Epic 7 summary and residual finalization
- retrospective and handoff

## Preserved Fence
Sprint 79 must preserve the current Epic 7 assurance and truthfulness fence:
- no fake algorithm-breadth, platform-breadth, or packaging-breadth claim widening
- no broad subsystem redesign disguised as final assurance work
- no threshold-driven benchmark-governance reopening
- no public-surface rewrite when narrower proof-owner or truth-surface fixes suffice
- no closeout summary that outruns the maintained validation and proof evidence

## Strongest Likely Sprint 79 Touch Surfaces
Proof-owner and assurance hotspots from the live tree:
- `tests/test_chol_csc.c` = `4724`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_iterative.c` = `2841`
- `tests/test_ldlt.c` = `2798`
- `tests/test_integration.c` = `2543`
- `tests/test_reorder_nd.c` = `2287`
- `tests/test_fuzz.c` = `651`

Support and final-truth surfaces from the live tree:
- `README.md` = `1045`
- `INSTALL.md` = `265`
- `docs/maintainer_guide.md` = `685`
- `benchmarks/README.md` = `393`
- `docs/tutorial.md` = `473`
- `examples/README.md` = `166`
- `include/sparse_iterative.h` = `773`
- `include/sparse_eigs.h` = `651`
- `include/sparse_ldlt.h` = `335`
- `include/sparse_cholesky.h` = `227`

Workflow, install, and reporting proof surfaces:
- `tests/test_install.sh` = `172`
- `tests/test_cmake_install.sh` = `146`
- `scripts/bench_canonical_report.sh` = `101`
- `Makefile` = `899`
- `CMakeLists.txt` = `413`
- `.github/workflows/ci.yml` = `223`
- `.github/workflows/macos-ci.yml` = `117`
- `.github/workflows/windows-ci.yml` = `63`

## High-Signal Live Reading
The live tree already fixes several important Sprint 79 truths:
- Sprint 78's carry-forward queue still names the strongest residual implementation and family-local proof seams, especially:
  - `src/sparse_iterative.c`
  - `tests/test_ldlt_csc.c`
  - `src/sparse_chol_csc.c`
  - `tests/test_qr.c`
- The public and support surfaces already name the strongest lifecycle, progress/cancel, benchmark-reporting, and install/export truths, so Sprint 79 should fix only real remaining contradictions rather than reopen broad wording churn.
- The benchmark and install surfaces already have stable permanent owners:
  - canonical benchmark reporting = `scripts/bench_canonical_report.sh` + `Makefile`
  - local install/export proof = `tests/test_install.sh` + `tests/test_cmake_install.sh`
- The strongest residual assurance pressure sits in proof-owner tests and lifecycle/oracle surfaces, not in a new broad documentation or workflow campaign.

## Interpretation
The strongest Day 1 narrowing is now explicit:
- Sprint 79 is not “polish everything left in Epic 7.”
- It is one bounded final assurance and integration sprint.
- The highest-value early work is therefore:
  - rerank the remaining oracle, lifecycle, and property gaps
  - land only the most valuable residual proof expansion
  - perform one final cross-surface truth sweep
  - close Epic 7 from maintained reviewed evidence

## Exit State
- The maintained reviewed baseline is rechecked.
- The reviewed CMake parity anchor is explicit and current.
- The live assurance, integration, support, reporting, and install/workflow surfaces are fixed in writing.
- Sprint 79 can now move into an assurance-gap audit from a precise Day 1 baseline.
