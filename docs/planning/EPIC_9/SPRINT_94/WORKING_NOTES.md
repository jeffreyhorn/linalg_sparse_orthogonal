# Sprint 94 Working Notes

## Day 1 - Scope and Capability Baseline

### Goal
Turn the Sprint 94 project-plan section and the Sprint 93 validated closeout
into one bounded capability-surface modernization execution package before any
capability rerank, scalar/index contract design, or implementation lands.

### Actions
- Re-read the Sprint 94 contract in
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 94 day-by-day plan in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the closest prior closeout and handoff surfaces:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_9/SPRINT_93/RETROSPECTIVE.md`
- Re-read the closest prior Epic 9 capability-target surface:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day5-competitive-target-and-product-contract-design.md`
- Reconfirmed that the strongest local reviewed entry point still begins at:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest likely Sprint 94 touch surfaces by line count and
  owner role:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `include/sparse_types.h`
  - `include/sparse_dense.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_eigs.h`
  - `include/sparse_iterative.h`
  - `include/sparse_matrix.h`
  - `src/sparse_types.c`
  - `src/sparse_dense.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`
  - `src/sparse_matrix.c`
  - `tests/test_dense.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_svd.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
- Wrote the Day 1 scope artifact and authoritative-input list.

### Findings
- Sprint 94 begins from a validated Sprint 93 close state, not from another
  generic runtime or backend reset:
  - strongest local reviewed baseline remains `make quality-review-full`
  - reviewed CMake parity was re-materialized live and remains explicit:
    - `ctest -N --test-dir build/quality-review-cmake` = `53`
    - Makefile/CMake parity = `53 vs 53`
- Sprint 91-93 already moved the strongest prior Epic 9 contradictions that
  had to land before capability widening:
  - compressed-first public and direct-workflow entry seams are materially
    clearer
  - the shared dense owner now has one bounded builtin-vs-portable backend
    seam
  - the strongest reviewed runtime long pole is smaller and more explicitly
    measured
- That means Sprint 94 can start from the next real Epic 9 contradiction
  center:
  - bounded capability breadth beyond the current real-only scalar contract,
    compile-time-selected index-width contract, and narrow solver-family
    reading on the touched public seams
- The highest-value Sprint 94 package is now fixed explicitly around:
  - capability rerank
  - scalar/index architecture design
  - scalar-family widening design and batch
  - index/ABI maturity follow-through
  - solver-family breadth follow-through
  - proof/docs/package alignment
- The live tree currently points most strongly at these Sprint 94 surfaces:
  - strongest public scalar/index contract owners:
    - `include/sparse_types.h` = `313`
    - `include/sparse_matrix.h` = `622`
    - `include/sparse_dense.h` = `197`
    - `include/sparse_qr.h` = `392`
    - `include/sparse_svd.h` = `257`
    - `include/sparse_eigs.h` = `651`
    - `include/sparse_iterative.h` = `773`
  - strongest shared implementation owners likely to matter:
    - `src/sparse_dense.c` = `955`
    - `src/sparse_qr.c` = `1563`
    - `src/sparse_svd.c` = `1319`
    - `src/sparse_eigs.c` = `1534`
    - `src/sparse_iterative.c` = `1854`
    - `src/sparse_matrix.c` = `1297`
    - `src/sparse_types.c` = `54`
  - strongest proof-owner tests likely to matter:
    - `tests/test_dense.c` = `647`
    - `tests/test_qr.c` = `3234`
    - `tests/test_svd.c` = `2766`
    - `tests/test_eigs.c` = `1560`
    - `tests/test_iterative.c` = `2841`
    - `tests/test_sparse_matrix.c` = `1233`
    - `tests/test_integration.c` = `3421`
  - strongest benchmark and capability-evidence owners:
    - `benchmarks/bench_svd.c` = `180`
    - `benchmarks/bench_eigs.c` = `958`
    - `benchmarks/bench_iterative_reuse.c` = `395`
  - strongest support, build, and workflow surfaces if capability work forces
    follow-through:
    - `README.md` = `1136`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `738`
    - `Makefile` = `928`
    - `CMakeLists.txt` = `425`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
- Sprint 94 is explicitly bounded against:
  - reopening the Sprint 93 runtime lane as the first owner again
  - promising broad full-library complex or mixed-precision maturity before
    one bounded widened capability seam is actually proved
  - treating compile-time index-width configurability alone as equivalent to
    finished touched-path ABI maturity
  - widening into broad package/platform claims before the touched scalar and
    index seams are materially stronger
  - drifting into generic solver-family expansion detached from one real
    scalar/index contract improvement

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the strongest likely capability, proof, benchmark, and support
  surfaces by live file size and owner role.

### Day 1 Exit State
- Sprint 94 now starts from one precise capability-surface modernization
  execution package rather than from a generic "add more numerical features"
  bucket.
- The strongest likely touch surfaces, preserved non-goals, and maintained
  reviewed starting truth are fixed in writing before the validation and
  maintained-surface recheck begins.
- Day 2 can now freeze the authoritative reviewed, benchmark, install/export,
  and support-surface truth split without reopening the Day 1 scope question.

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, benchmark, install/export, example, and workflow truth split before
Sprint 94 begins capability-focused implementation work on the touched
scalar-, index-, and solver-family seams.

### Actions
- Re-read the Sprint 94 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day2-validation-baseline-and-maintained-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained Sprint 94
  truth surfaces:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so Sprint 94 does
  not overclaim reviewed capability breadth, install/export parity, or
  cross-platform package maturity while touching scalar and index seams.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 94 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for
  capability- and ABI-focused work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial capability-contract, proof-owner, benchmark, or
    support-surface batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 94
  code lands:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest reviewed executable truth owners for Sprint 94's capability
  lane are now fixed around:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Canonical benchmark reporting remains command- and script-owned rather than
  reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/export proof remains script- and fixture-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- Workflow truth remains intentionally layered rather than flattened:
  - Linux remains the strongest reviewed source of truth through the enforced
    reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and
    supplemental coverage lanes
  - macOS remains a narrower reviewed Apple Clang lane plus supplemental
    static-first install confidence
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim reviewed Makefile parity or separate reviewed install-validation
    parity
- The highest-signal rerun set is now fixed for the rest of Sprint 94:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 94 should read scalar-, index-, and solver-family-touching changes
    against the reviewed scalar-sensitive proof owners
  - canonical reporting remains a bounded command/script-owned evidence
    surface, not a reviewed capability-parity claim

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the presence of the strongest reviewed scalar-sensitive, example,
  install/export, and workflow owner surfaces.

### Day 2 Exit State
- Sprint 94 now has one explicit validation and maintained-surface contract
  before capability implementation begins.
- Reviewed scalar-sensitive binaries remain the main executable truth anchor.
- Canonical benchmark reporting remains command/script owned.
- Install/export proof remains script owned.
- Workflow lanes remain layered support evidence rather than broad
  cross-platform capability-parity claims.
