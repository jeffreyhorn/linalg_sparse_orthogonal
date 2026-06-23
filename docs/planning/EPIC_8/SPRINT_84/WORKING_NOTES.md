# Sprint 84 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 84 baseline for Epic 8 by grounding the sprint in
the validated Sprint 83 close state, the live Sprint 84 project-plan section,
and the current oracle, property, proof-owner, benchmark, and support-surface
seams rather than another generic “add more tests” restart.

### Actions
- Re-read the Sprint 84 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 84 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_84/PLAN.md`.
- Re-read the strongest Sprint 83 closeout context:
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_83/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 84
  touch surfaces across shared/public headers, implementation owners,
  direct-family proof-owner tests, iterative/eigs proof owners, benchmark
  surfaces, and support surfaces.
- Opened Sprint 84 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 84 starts from the same strongest local reviewed baseline Sprint 83
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 84 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 84 is not a generic “add more tests” sprint. Its highest value is one
  bounded assurance-modernization package centered on:
  - differential-proof audit
  - oracle / property / failure-path architecture design
  - first maintained direct-family external differential batch
  - deterministic seeded property expansion
  - failure-path numerical proof
  - focused policy / CI / support-surface alignment only where implementation
    truly moves the assurance contract
  - validation and closeout
- The strongest likely Sprint 84 assurance, proof, and support surfaces are
  explicit from the live tree:
  - `include/sparse_types.h` = `313`
  - `include/sparse_matrix.h` = `622`
  - `include/sparse_qr.h` = `392`
  - `include/sparse_svd.h` = `257`
  - `include/sparse_cholesky.h` = `227`
  - `include/sparse_ldlt.h` = `335`
  - `src/sparse_matrix.c` = `1297`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_chol_csc.c` = `1841`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
  - `tests/test_sparse_matrix.c` = `1136`
  - `tests/test_qr.c` = `3234`
  - `tests/test_svd.c` = `2766`
  - `tests/test_chol_csc.c` = `4787`
  - `tests/test_ldlt.c` = `2921`
  - `tests/test_iterative.c` = `2841`
  - `tests/test_eigs.c` = `1560`
  - `tests/test_integration.c` = `2973`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `716`
- The strongest Day 1 clarification is now fixed:
  - Sprint 84 should not reopen Sprint 83’s capability-surface owner fence as
    its first implementation center
  - Sprint 84 should not inflate support-surface or CI claims before one
    bounded maintained assurance seam truly lands
  - it should first reduce the current external-differential, seeded-property,
    and failure-path assurance ceiling on the highest-value touched lanes only
- The preserved Sprint 84 non-goal pressure is explicit before Day 2:
  - no generic capability widening restart
  - no repo-wide claim that every family now has maintained external proof
  - no benchmark-governance drift into correctness ownership
  - no broad oracle dependency story for untouched families
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed assurance seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared/public, implementation, proof-owner, benchmark, and
  support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 84 no longer starts from generic Epic 8 assurance prose.
- The baseline, differential rerank, oracle/property/failure-path design,
  first direct-family differential landing, seeded property expansion,
  failure-path proof, focused policy/CI alignment, and validation workstreams
  are fixed in writing.
- The strongest likely Sprint 84 touch surfaces are explicit before the
  validation/proof recheck begins.
