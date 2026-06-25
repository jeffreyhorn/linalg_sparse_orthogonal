# Sprint 90 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 90 baseline for Epic 9 by grounding the sprint in
the validated Epic 8 close state, the live Sprint 90 project-plan section, and
the strongest current product-model, comparison, support-surface, build, and
hotspot signals rather than another generic "new epic" reset.

### Actions
- Re-read the Sprint 90 section of `docs/planning/EPIC_9/PROJECT_PLAN.md` and
  the full Sprint 90 day-by-day plan in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the strongest Epic 8 closeout context:
  - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md`
  - `docs/planning/EPIC_8/SPRINT_89/artifacts/day14-closeout-and-handoff.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor from the Day 1 parity rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw line-count hotspot map for the strongest likely Sprint
  90 touch surfaces across the Epic 9 planning surfaces, prior Epic 8 review
  inputs, support/build/workflow owners, and the highest-value residual code
  and proof hotspots that the Epic 9 review is likely to discuss.
- Opened Sprint 90 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 90 starts from the same strongest local reviewed baseline Epic 8
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 90 review or
  planning work widens:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 90 is not a generic "start the next epic" sprint. Its highest value
  is one bounded Epic 9 baseline package centered on:
  - baseline recheck
  - target-state freeze
  - product-model audit
  - comparison/measurement contract
  - non-goal and risk fence
  - review/todo/project-plan package
- The validated Epic 8 close state already fixed the strongest handoff truth
  entering Sprint 90:
  - the project now has a real bounded external SPD comparison lane
  - the package/install/export contract is sharper and explicitly static-first
  - the front-door adoption/support split is materially cleaner
  - the residual queue is smaller and better calibrated than at Epic 8 start
- The strongest current contradiction entering Epic 9 is no longer basic
  engineering hygiene. It is the structural product gap between:
  - a highly disciplined and well-proven sparse library
  - and a truly state-of-the-art sparse numerical product
- The strongest likely Sprint 90 review and planning surfaces are explicit
  from the live tree:
  - Epic 9 planning and prior-epic interpretation owners:
    - `docs/planning/EPIC_9/PROJECT_PLAN.md` = `350`
    - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md` = `292`
    - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md` = `501`
    - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md` = `361`
  - strongest support, package, and workflow owners:
    - `README.md` = `1113`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `727`
    - `Makefile` = `908`
    - `CMakeLists.txt` = `416`
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
    - `scripts/bench_canonical_report.sh` = `101`
  - strongest residual implementation and proof hotspots likely to matter in
    the Epic 9 review:
    - `src/sparse_matrix.c` = `1297`
    - `src/sparse_dense.c` = `862`
    - `src/sparse_iterative.c` = `1854`
    - `src/sparse_ldlt_csc.c` = `2694`
    - `tests/test_reorder_nd.c` = `2340`
    - `tests/test_graph.c` = `2925`
    - `tests/test_chol_csc.c` = `4987`
    - `tests/test_ldlt_csc.c` = `3680`
- The strongest Day 1 clarification is now fixed:
  - Sprint 90 should begin with one evidence-first baseline and target audit
    rather than with assumptions inherited loosely from Epic 8
  - the Epic 9 planning package should be grounded in live post-Epic-8 truth
    surfaces, not just in prior planning prose
  - anti-sprawl and anti-overclaim work belongs at the start of the epic, not
    later after implementation has already widened
- The preserved Sprint 90 non-goal pressure is explicit before Day 2:
  - no generic "review everything" prose detached from repo evidence
  - no speculative implementation batch before the target state and ranking are
    fixed
  - no fake platform, package, or capability broadening in the review wording
  - no benchmark or workflow reinterpretation that outruns the maintained
    proof-owner surfaces
  - no Epic 9 summary writing before the review, todo, and project-plan
    package is evidence-backed

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live planning/support/hotspot map from direct line-count
  measurement.

### Day 1 Exit State
- Sprint 90 no longer starts from generic Epic 9 ambition prose.
- The baseline recheck, target-state freeze, product-model audit,
  comparison/measurement contract, non-goal/risk fence, and review/todo/plan
  workstreams are fixed in writing.
- The strongest likely touch surfaces and preserved non-goals are explicit
  before the validation and maintained-surface recheck begins.

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the strongest implementation-day validation contract and the live
reviewed, install/export, reporting, example, and workflow ownership split so
Sprint 90 can write Epic 9 review findings against the real maintained truth
surfaces rather than against flattened assumptions.

### Actions
- Re-read the Sprint 90 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_8/SPRINT_89/artifacts/day2-validation-baseline-and-maintained-cross-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Re-materialized the reviewed CMake parity tree and ownership surface with:
  - `make quality-review-cmake-compile`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained support
  surfaces Sprint 90 must treat as authoritative:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so the Sprint 90
  review package does not overclaim reviewed parity or install/export
  coverage.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 90 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for Epic 9
  planning work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial review, comparison, final-calibration, or closeout-support
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 90
  review or plan claims widen:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- The strongest shared executable truth owners entering Sprint 90 remain:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
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
    reviewed Makefile compile-quality, reviewed CMake parity, and dead-code
    lanes
  - macOS remains a narrower reviewed Apple Clang lane plus supplemental
    static-first Make install/`pkg-config` confidence
  - Windows remains a reviewed CMake-first consumer subset, keeps its
    `ctest -N` expectation at `50`, and does not claim reviewed Makefile or
    separate reviewed install-validation parity
- The highest-signal rerun set is now fixed for the rest of Sprint 90:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 90 review findings must read against layered maintained truth
    surfaces rather than against one fake flattened "all platforms/all
    workflows/all reports are equivalent" story
  - install/export proof and benchmark reporting remain bounded maintained
    surfaces, not generic CI or reviewed-binary claims
  - later Epic 9 review and plan language should stay disciplined about what
    is executable truth, what is local maintained proof, and what is only
    supplemental workflow evidence

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Re-ran `make quality-review-cmake-compile`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the representative reviewed binaries/examples and the maintained
  install/export, consumer, reporting, and workflow-owner surfaces.

### Day 2 Exit State
- Sprint 90 now has one explicit validation and maintained-surface contract
  before Epic 9 review writing begins.
- The reviewed executable truth owners, canonical reporting owners,
  install/export proof owners, and workflow-side support evidence are fixed in
  writing.
- Later Day 3-Day 10 audit, design, review, and plan work no longer needs to
  guess which surfaces are authoritative.

## Day 3 - Product, Performance, and Capability Gap Audit

### Goal
Reduce the broad post-Epic-8 structural problem to one ranked live
contradiction map across the public product model, dense/backend maturity, and
capability breadth so Epic 9 can choose real implementation centers rather
than another generic modernization story.

### Actions
- Re-read the Sprint 90 Day 3 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the strongest Epic 9 baseline review and closure-order package:
  - `docs/planning/EPIC_9/reviews/review-codex-2026-06-24.md`
  - `docs/planning/EPIC_9/reviews/todo-codex-2026-06-24.md`
- Re-scanned the live tree against the strongest structural concern classes:
  - linked-list-first public/product model
  - dense/backend maturity ceiling
  - capability breadth and ABI/index maturity
- Re-anchored the audit directly on the current product and implementation
  surfaces:
  - `README.md`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
  - `src/sparse_dense.c`
  - `src/sparse_matrix.c`
  - `src/sparse_ldlt_csc.c`
- Rechecked the live wording around:
  - the public orthogonal linked-list shell
  - bounded compressed-first helpers
  - real-only `sparse_scalar_t`
  - optional Accelerate-backed dense/direct slices
  - optional OpenMP/runtime lanes
- Wrote the Day 3 ranked contradiction artifact and fixed the strongest owner
  surfaces in writing.

### Findings
- Sprint 90's broad structural problem is now reduced to one ranked live map:
  - strongest first target:
    - compressed-first product-model convergence centered on the public
      matrix-shell and direct-workflow ownership story
  - strongest second target:
    - portable dense/backend maturity beyond the current builtin scalar core
      plus bounded optional Accelerate slices
  - strongest third target:
    - capability-surface widening beyond the current real-only scalar and
      intentionally bounded solver/eigensolver breadth
  - strongest fourth target:
    - runtime/threading and ABI/index follow-through only where they sharpen
      the first three structural lanes
  - strongest support-only but real target:
    - public/support wording that still truthfully reflects the narrower
      current product, backend, and capability reading
- The strongest current contradiction is now explicit:
  - `README.md` still introduces the library as an orthogonal linked-list
    sparse matrix library
  - `include/sparse_matrix.h` still presents the matrix shell as the library's
    mutable sparse construction and one-shot direct-workflow compatibility
    shell
  - the same header explicitly says compressed-first helpers may exist
    internally, but the public ownership model still stays with that
    compatibility shell
  - `src/sparse_matrix.c` remains a major shell/mutation/utility owner
- The strongest second contradiction is backend maturity:
  - `src/sparse_dense.c` still owns scalar builtin dense helpers
  - `src/sparse_ldlt_csc.c` still exposes only a bounded optional Accelerate
    lane rather than a portable backend story
  - the repo now has backend architecture, but not yet backend maturity
- The strongest third contradiction is capability breadth:
  - `include/sparse_types.h` still binds `sparse_scalar_t` to real-only
    `double`
  - the same header explicitly says this does not imply broad numeric
    genericity, complex support, or wider precision maturity today
  - `README.md` still keeps iterative/eigensolver breadth intentionally
    bounded
  - wider indices are real, but still read as a compile-time lane rather than
    as deeply matured product-wide support
- The strongest owner surfaces are now fixed:
  - product model:
    - `README.md`
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - backend/performance:
    - `src/sparse_dense.c`
    - `src/sparse_ldlt_csc.c`
    - `README.md`
  - capability surface:
    - `include/sparse_types.h`
    - `README.md`
    - later touched iterative/eigs/direct public headers if widening truly
      moves there
- The useful fix-now vs bounded non-claim split is now explicit:
  - fix-now Epic 9 structural contradictions:
    - linked-list-first public/product ownership
    - bounded dense/backend maturity
    - bounded capability breadth
  - bounded non-claims for now:
    - broad complex support
    - broad mixed precision
    - symmetric platform/package maturity
    - broad best-in-class runtime or ordering claims
  - contradictions already materially improved by Epic 8:
    - package/install/export truth
    - front-door usability and support layering
    - bounded external SPD comparison
    - shared scalar/index ownership consistency on touched seams
    - reviewed runtime concentration relative to the Sprint 85 close state
- The strongest Day 3 clarification is now fixed:
  - Epic 9 should not begin with another generic modernization lane
  - it should begin with a compressed-first product-model center
  - portable backend maturity follows next
  - capability widening follows after that
  - runtime/threading and ABI/index maturity remain real, but explicitly
    later unless Day 5 target-state design proves otherwise

### Validation
- Re-read the Sprint 90 Day 3 plan contract.
- Re-read the Epic 9 review and todo package.
- Rechecked the live product-model, backend, and scalar/index wording in the
  touched public and implementation surfaces.
- Recorded the ranked contradiction map and strongest owner surfaces in the
  Day 3 artifact.

### Day 3 Exit State
- Sprint 90 now has one ranked live product/performance/capability
  contradiction map grounded in the current tree.
- The first structural Epic 9 center is fixed to compressed-first
  product-model convergence.
- Portable backend maturity, capability widening, and runtime/ABI follow-through
  are explicitly ordered behind that first center before Day 4 widens into
  maintainability, coherence, and workflow duplication.

## Day 4 - Maintainability, Coherence, and Workflow Audit

### Goal
Reduce the remaining maintainability, documentation-coherence, and
build/workflow-duplication problem to one ranked live contradiction map so
Epic 9 can choose bounded structural cleanup lanes rather than another generic
cleanup story.

### Actions
- Re-read the Sprint 90 Day 4 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the Epic 9 review findings around maintainability, chronology
  residue, and build/workflow duplication.
- Re-measured the highest-signal implementation, proof, support, build, and
  workflow owners directly from the live tree.
- Re-scanned the live tree against the remaining Epic 9 concern classes:
  - large mixed-role implementation hotspots
  - giant-test and proof-owner concentration
  - sprint-era chronology in permanent docs/code/tests
  - duplicated build/package/workflow topology
- Rechecked direct chronology residue across public/support surfaces, headers,
  benchmarks, workflows, and the retained sprint-named tests.
- Rechecked the main duplicated topology owners:
  - `Makefile`
  - `CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Wrote the Day 4 ranked audit artifact and fixed the highest-friction
  support-surface map in writing.

### Findings
- Sprint 90's remaining maintainability/coherence/duplication problem is now
  reduced to one ranked live map:
  - strongest first target:
    - large mixed-role implementation hotspot reduction
  - strongest second target:
    - giant-test and proof-owner concentration cleanup
  - strongest third target:
    - sprint-era chronology removal from permanent product, maintainer,
      header, benchmark, and test surfaces
  - strongest fourth target:
    - build/package/workflow topology convergence where hand-maintained
      duplication is still high
  - strongest support-only but real target:
    - public/support surface simplification where structural cleanup truly
      changes naming or navigation expectations
- The strongest current contradiction is now explicit:
  - the largest implementation hotspots remain large and mixed-role:
    - `src/sparse_ldlt_csc.c` = `2694`
    - `src/sparse_iterative.c` = `1854`
    - `src/sparse_lu_csr.c` = `1665`
    - `src/sparse_qr.c` = `1563`
    - `src/sparse_ldlt.c` = `1535`
    - `src/sparse_eigs.c` = `1534`
    - `src/sparse_svd.c` = `1319`
    - `src/sparse_matrix.c` = `1297`
    - `src/sparse_chol_csc.c` = `1279`
  - these files still combine algorithm detail, lifecycle policy, helper
    logic, and family-local orchestration in the same retained owners
- The strongest second contradiction is proof concentration:
  - `tests/test_chol_csc.c` = `4987`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3234`
  - `tests/test_integration.c` = `3197`
  - `tests/test_etree.c` = `2962`
  - `tests/test_graph.c` = `2925`
  - `tests/test_ldlt.c` = `2921`
  - `tests/test_iterative.c` = `2841`
  - `tests/test_reorder_nd.c` = `2340`
- The strongest third contradiction is chronology residue:
  - `README.md` still contains many sprint-era historical note blocks
  - `docs/maintainer_guide.md` still contains many "after Sprint X" anchors
  - `INSTALL.md`, `benchmarks/README.md`, several headers, and several
    benchmark files still carry sprint/day chronology in permanent surfaces
  - sprint-named retained proof owners still exist:
    - `tests/test_sprint20_integration.c`
    - `tests/test_sprint19_integration.c`
    - `tests/test_sprint11_integration.c`
    - `tests/test_sprint8_integration.c`
    - `tests/test_sprint6_integration.c`
    - `tests/test_sprint5_integration.c`
- The strongest fourth contradiction is duplicated topology:
  - `Makefile` = `908`
  - `CMakeLists.txt` = `416`
  - `.github/workflows/ci.yml` = `223`
  - `.github/workflows/macos-ci.yml` = `104`
  - `.github/workflows/windows-ci.yml` = `63`
  - `Makefile` still hand-enumerates library, test, benchmark, and workflow
    topology
  - `CMakeLists.txt` still duplicates the library/test/benchmark/example
    topology separately
  - all three workflows still carry intentionally asymmetric product truth,
    which is honest but increases long-term maintenance cost
- The highest-friction public/support owners are now fixed:
  - `README.md` = `1113`
  - `docs/maintainer_guide.md` = `727`
  - `docs/tutorial.md` = `473`
  - `benchmarks/README.md` = `399`
  - `INSTALL.md` = `315`
- The strongest Day 4 clarification is now fixed:
  - the maintainability/coherence problem is real, but it is not one
    undifferentiated bucket
  - the best first cleanup lane is the largest mixed-role implementation
    concentration
  - giant-test concentration follows next
  - chronology cleanup follows after that
  - build/package/workflow convergence follows after that
  - support surfaces stay support-only unless structural cleanup truly changes
    owner, naming, or navigation expectations

### Validation
- Re-read the Sprint 90 Day 4 plan contract.
- Re-read the Epic 9 review findings around maintainability, chronology, and
  duplication.
- Re-measured the highest-signal implementation, proof, support, build, and
  workflow owners from the live tree.
- Rechecked direct sprint-era chronology residue across permanent repo
  surfaces.
- Recorded the ranked contradiction map and high-friction support surfaces in
  the Day 4 artifact.

### Day 4 Exit State
- Sprint 90 now has one ranked live
  maintainability/coherence/duplication map grounded in the current tree.
- The strongest residual source hotspots, proof hotspots, chronology residue,
  and duplicated topology surfaces are explicit.
- Day 5 can now define the Epic 9 target state against the full contradiction
  inventory instead of against only the structural product/backend/capability
  side.
