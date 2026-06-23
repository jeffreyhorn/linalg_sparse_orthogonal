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

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live proof-owner
split before Sprint 85 decomposes any large source or giant-test hotspot.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the strongest recent validation/proof-surface template from
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day2-validation-baseline-and-proof-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed proof-owner binaries for the
  early Sprint 85 lanes:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the script-owned support-proof surfaces:
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 85 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial source-hotspot, giant-test, or support-policy batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently owns the strongest early-Sprint-85 proof
  surfaces:
  - iterative, direct-family, and giant-test owners:
    - `./build/quality-review-cmake/test_iterative`
    - `./build/quality-review-cmake/test_chol_csc`
    - `./build/quality-review-cmake/test_integration`
    - `./build/quality-review-cmake/test_ldlt`
    - `./build/quality-review-cmake/test_qr`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
  - reviewed benchmark follow-on binaries:
    - `./build/quality-review-cmake/bench_refactor_csc`
    - `./build/quality-review-cmake/bench_svd`
- Canonical benchmark reporting remains command- and script-owned rather than
  reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - reviewed CMake proof-owner tests and representative examples remain the
    main executable truth surfaces for Sprint 85 hotspot work
  - reviewed benchmark binaries remain measurability surfaces, not the
    canonical reporting owner
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed proof-owner tests,
  representative examples, and reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, and `tests/test_cmake_install.sh`.

### Day 2 Exit State
- Sprint 85 now has one explicit implementation-day validation contract before
  the hotspot rerank begins.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/package proof is fixed in writing.
- The highest-signal rerun set is explicit before the first hotspot-priority
  rerank.
