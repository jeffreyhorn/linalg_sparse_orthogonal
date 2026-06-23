# Sprint 85 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 85 baseline for Epic 8 by grounding the sprint in
the validated Sprint 84 close state, the live Sprint 85 project-plan section,
and the current source, giant-test, benchmark, and support-surface hotspots
rather than another generic “refactor more code” restart.

### Actions
- Re-read the Sprint 85 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 85 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the strongest Sprint 84 closeout context:
  - `docs/planning/EPIC_8/SPRINT_84/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_84/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 85
  touch surfaces across shared/public headers, implementation owners,
  direct-family and iterative proof-owner tests, and support surfaces.
- Opened Sprint 85 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 85 starts from the same strongest local reviewed baseline Sprint 84
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 85 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 85 is not a generic “cleanup everything” sprint. Its highest value is
  one bounded maintainability-modernization package centered on:
  - hotspot rerank
  - decomposition / ownership architecture design
  - iterative-source cleanup
  - direct-family hotspot cleanup
  - giant-test architecture cleanup
  - proof/docs alignment
  - validation and closeout
- The strongest likely Sprint 85 implementation, proof, and support surfaces
  are explicit from the live tree:
  - shared/public contract surfaces:
    - `include/sparse_types.h` = `313`
    - `include/sparse_matrix.h` = `622`
    - `include/sparse_qr.h` = `392`
    - `include/sparse_svd.h` = `257`
    - `include/sparse_cholesky.h` = `227`
    - `include/sparse_ldlt.h` = `335`
  - strongest source hotspots:
    - `src/sparse_iterative.c` = `1985`
    - `src/sparse_chol_csc.c` = `1841`
    - `src/sparse_qr.c` = `1563`
    - `src/sparse_ldlt.c` = `1535`
  - strongest giant proof-owner tests:
    - `tests/test_chol_csc.c` = `4964`
    - `tests/test_qr.c` = `3234`
    - `tests/test_integration.c` = `3197`
    - `tests/test_ldlt.c` = `2921`
    - `tests/test_iterative.c` = `2841`
  - strongest support surfaces:
    - `README.md` = `1050`
    - `docs/maintainer_guide.md` = `726`
- The strongest Day 1 clarification is now fixed:
  - Sprint 85 should not reopen Sprint 84’s bounded assurance-owner fence as
    its first implementation center
  - Sprint 85 should first reduce review and reasoning cost on the largest
    remaining implementation and giant-test hotspots
  - it should preserve proof ownership while decomposing the touched seams
    rather than diffusing ownership across more files without a stronger
    contract
- The preserved Sprint 85 non-goal pressure is explicit before Day 2:
  - no generic assurance widening restart
  - no repo-wide architectural cleanup claim
  - no proof dilution from helper movement detached from owner boundaries
  - no benchmark-governance or example-governance drift into correctness
    ownership
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed hotspot seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared/public, implementation, proof-owner, benchmark, and
  support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 85 no longer starts from generic Epic 8 maintainability prose.
- The hotspot rerank, decomposition design, iterative cleanup, direct-family
  cleanup, giant-test cleanup, proof/docs alignment, and validation workstreams
  are fixed in writing.
- The strongest likely Sprint 85 touch surfaces are explicit before the
  validation/proof recheck begins.
