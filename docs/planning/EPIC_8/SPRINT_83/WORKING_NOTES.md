# Sprint 83 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 83 baseline for Epic 8 by grounding the sprint in
the validated Sprint 82 close state, the live Sprint 83 project-plan section,
and the current capability, index-width, proof-owner, benchmark, and
support-surface seams rather than another generic capability restart.

### Actions
- Re-read the Sprint 83 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 83 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Re-read the strongest Sprint 82 closeout context:
  - `docs/planning/EPIC_8/SPRINT_82/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_82/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 83
  touch surfaces across shared types, matrix/public seams, solver-family
  headers, implementation owners, proof-owner tests, and support surfaces.
- Opened Sprint 83 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 83 starts from the same strongest local reviewed baseline Sprint 82
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 83 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 83 is not a generic “add more math types” sprint. Its highest value
  is one bounded capability-surface modernization package centered on:
  - capability re-rank
  - scalar / index architecture design
  - first scalar-surface expansion on the highest-value public seams
  - touched-path index / ABI follow-through
  - one bounded algorithm-surface widening lane
  - focused regression / docs / package alignment only where the
    implementation truly moves the contract
- The strongest likely Sprint 83 capability, proof, and support surfaces are
  explicit from the live tree:
  - `include/sparse_types.h` = `310`
  - `include/sparse_matrix.h` = `614`
  - `include/sparse_qr.h` = `385`
  - `include/sparse_svd.h` = `257`
  - `include/sparse_cholesky.h` = `227`
  - `include/sparse_ldlt.h` = `335`
  - `src/sparse_types.c` = `54`
  - `src/sparse_matrix.c` = `1295`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_chol_csc.c` = `1841`
  - `src/sparse_ldlt.c` = `1535`
  - `tests/test_sparse_matrix.c` = `1104`
  - `tests/test_qr.c` = `3197`
  - `tests/test_svd.c` = `2766`
  - `tests/test_chol_csc.c` = `4787`
  - `tests/test_ldlt.c` = `2921`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `710`
- The strongest Day 1 clarification is now fixed:
  - Sprint 83 should not reopen Sprint 82's dense/backend ABI fence as its
    first implementation center
  - Sprint 83 should not claim repo-wide complex, mixed-precision, or package
    maturity before one bounded seam truly lands
  - it should first reduce the current real-only and compile-time-bounded
    capability ceiling on the highest-value public seams only
- The preserved Sprint 83 non-goal pressure is explicit before Day 2:
  - no broad backend-framework reopening
  - no repo-wide complex-number promise
  - no broad mixed-precision framework
  - no generic package/platform maturity claim widening
  - no algorithm-family widening before the shared scalar/index contract is
    explicit
  - no benchmark-governance drift or support-surface churn detached from a real
    landed capability seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared-type, matrix/public, solver-family, proof-owner,
  benchmark, and support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 83 no longer starts from generic Epic 8 capability prose.
- The baseline, capability rerank, scalar/index design, first public-seam
  widening, touched-path width/ABI follow-through, bounded algorithm widening,
  focused proof, and package-alignment workstreams are fixed in writing.
- The strongest likely Sprint 83 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Reconfirm the Sprint 83 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, benchmark/report command surfaces, and install/export proof owners
before any capability-surface batch lands.

### Actions
- Re-read the Sprint 83 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest reviewed proof-owner binaries and representative
  examples most likely to matter early in Sprint 83:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the strongest reviewed benchmark follow-on binaries most likely to
  matter:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- Rechecked the maintained canonical report command surface with:
  - `make -n bench-canonical-report`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 83 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 83 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial capability, width/ABI, algorithm-surface, or package/runtime
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Sprint-83 proof
  surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- The canonical benchmark/reporting lane remains command- and script-owned
  rather than reviewed-binary-owned:
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
- The strongest current proof and truth-surface split is now fixed for Sprint
  83's first lane:
  - reviewed CMake proof-owner tests and representative examples remain the
    main executable truth surfaces
  - reviewed benchmark binaries remain benchmark-side measurability surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 83 rerun set is now fixed around the likely touched
  capability surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Sprint 83.
- Rechecked the strongest reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 83 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the capability re-rank audit
  begins.
