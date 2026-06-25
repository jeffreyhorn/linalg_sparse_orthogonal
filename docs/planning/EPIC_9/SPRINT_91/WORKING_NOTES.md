# Sprint 91 Working Notes

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, install/export, reporting, example, and workflow truth split before
Sprint 91 begins compressed-first implementation work on the direct workflow
surface.

### Actions
- Re-read the Sprint 91 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day2-validation-baseline-and-maintained-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained Sprint 91
  truth surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so Sprint 91 does
  not overclaim reviewed parity or install/export breadth while touching the
  direct product model.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 91 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for
  compressed-first product work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial product-contract, proof-owner, or support-surface batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 91
  code lands:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest reviewed executable truth owners for Sprint 91’s direct-product
  lane are now fixed around:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
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
  - Linux remains the strongest reviewed source of truth
  - macOS remains a narrower reviewed Apple Clang lane plus supplemental
    static-first install confidence
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim reviewed Makefile or separate reviewed install-validation parity
- The highest-signal rerun set is now fixed for the rest of Sprint 91:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 91 should read direct-workflow product changes against the direct
    workflow proof owners, not against the reorder/runtime owners that drove
    earlier epic work
  - install/export proof and benchmark reporting remain bounded maintained
    surfaces, not generic product-parity claims
  - later Sprint 91 audit, design, and implementation days should stay
    disciplined about what is executable truth, what is script-owned proof,
    and what is only supplemental workflow evidence

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the representative reviewed binaries/examples and the maintained
  install/export, consumer, reporting, and workflow-owner surfaces.

### Day 2 Exit State
- Sprint 91 now has one explicit validation and maintained-surface contract
  before compressed-first implementation work begins.
- The reviewed direct-workflow truth owners, canonical reporting owner,
  install/export proof owners, and workflow-side support evidence are fixed in
  writing.
- Later Day 3-Day 11 audit, design, implementation, and follow-through work no
  longer needs to guess which surfaces are authoritative.

## Day 3 - Remaining Linked-List-First Cost Audit

### Goal
Reduce Sprint 91's broad compressed-first product-model problem to one ranked
live contradiction map centered on the remaining linked-list-first
construction, import/export, publication, and lifecycle costs.

### Actions
- Re-read the Sprint 91 Day 3 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the closest prior structural audit artifact:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day3-product-performance-capability-gap-audit.md`
- Re-scanned the live tree against the strongest Sprint 91 contradiction class:
  - linked-list-first construction
  - linked-list-first import/export and publication
  - shell-centric direct-workflow entry paths
  - lifecycle ambiguity on mutated vs solve-ready states
- Re-anchored the audit directly on the current product and implementation
  owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `include/sparse_analysis.h`
  - `include/sparse_csr.h`
- Rechecked the strongest direct-workflow proof owners likely to matter later:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
- Captured the live hotspot map for those strongest owner surfaces.
- Wrote the Day 3 audit artifact and fixed the ranked shell-cost order in
  writing.

### Findings
- Sprint 91's broad compressed-first problem is now reduced to one ranked live
  map of the highest-value linked-list-first costs:
  - strongest first target:
    - compressed-first construction and import entry points on the public
      matrix shell
  - strongest second target:
    - shell-centric publication and export round-trips that still keep the
      linked-list shell as the default conceptual owner
  - strongest third target:
    - one-shot direct-workflow entry paths that still read as shell-first even
      when the repeated-run direct lifecycle already exists
  - strongest fourth target:
    - lifecycle ambiguity on mutated vs solve-ready shell state and on where
      the long-lived direct-workflow owner really lives
  - strongest support-only but real target:
    - README, maintainer, and public-header wording that still teaches the
      shell as conceptual center instead of bounded mutable compatibility
      surface
- The strongest current contradiction is still the public construction and
  ownership reading:
  - `README.md` still opens by describing the project as an orthogonal
    linked-list sparse matrix library
  - `include/sparse_matrix.h` still describes the public API as the orthogonal
    linked-list sparse matrix shell
  - the same header still keeps the shell as the public mutable sparse
    construction and one-shot direct-workflow compatibility owner
  - `src/sparse_matrix.c` remains a major shell, mutation, and utility owner
- The strongest current owner surfaces are now explicit from the live tree:
  - `include/sparse_matrix.h` = `622`
  - `src/sparse_matrix.c` = `1297`
  - `include/sparse_analysis.h` = `499`
  - `include/sparse_csr.h` = `109`
  - `README.md` = `1113`
  - `tests/test_sparse_matrix.c` = `1136`
  - `tests/test_integration.c` = `3197`
  - `tests/test_chol_csc.c` = `4987`
  - `tests/test_ldlt_csc.c` = `3680`
- The fix-now vs compatibility-only split is now explicit:
  - Sprint 91 should drive:
    - shell-first construction/import costs
    - shell-centric publication/export reading
    - one-shot vs repeated-run direct ownership ambiguity
  - Sprint 91 should keep compatibility-only for now:
    - broad shell removal
    - family-wide direct-API rewrites
    - fake fully compressed-first ownership claims
    - backend, runtime, capability, or packaging widening under Day 3
- The strongest Day 3 clarification is now fixed:
  - Sprint 91 does not begin with generic direct-family cleanup
  - it begins with one ranked shell-cost map
  - the best first implementation center is compressed-first construction and
    import on the public matrix-shell story
  - publication/export and lifecycle tightening follow after that

### Validation
- Re-read the Sprint 91 Day 3 plan contract.
- Re-read the closest prior Sprint 90 structural audit.
- Re-scanned the live product-model owners and strongest direct-workflow proof
  owners.
- Captured the live hotspot map for the strongest likely Sprint 91 surfaces.

### Day 3 Exit State
- Sprint 91 now has one ranked live shell-cost contradiction map grounded in
  the current post-Sprint-90 tree.
- The first compressed-first implementation center is fixed to construction and
  import entry points on the public matrix-shell story.
- Day 4 can freeze the first implementation boundary without reopening the
  ranked shell-cost order.
