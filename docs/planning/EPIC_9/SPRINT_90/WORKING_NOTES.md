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

## Day 5 - Competitive Target and Product Contract Design

### Goal
Define what Epic 9 is actually trying to make the library become so later
implementation, comparison, and closeout work can aim at one stable bounded
target rather than at a vague "state of the art" label.

### Actions
- Re-read the Sprint 90 Day 5 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the Sprint 90 Day 3 and Day 4 audit artifacts against the live Epic
  9 project-plan intent.
- Re-read the Epic 9 project plan's sprint ordering so the target state stays
  consistent with the later compressed-first, backend, capability,
  runtime, narrative, package, and comparison lanes.
- Compared the plausible target-state readings explicitly:
  - high-end research sparse toolkit
  - production-grade sparse numerical library
  - bounded state-of-the-art sparse linear algebra library
- Fixed the success-marker order and the first-pass claim fence in writing.
- Wrote the Day 5 target-state artifact and recorded the ownership split for
  the first, second, and third implementation centers implied by that target.

### Findings
- Sprint 90 now has one explicit Epic 9 target-state reading:
  - bounded state-of-the-art sparse linear algebra library
- The target is explicitly *not*:
  - a generic research playground with unconstrained surface sprawl
  - a fully productized broad-platform/broad-backend industrial sparse stack
  - a benchmark-supremacy claim machine
- The strongest success-marker order is now fixed:
  - first:
    - compressed-first product-model convergence on the highest-value public
      direct and interop workflows
  - second:
    - portable dense/backend maturity beyond the current scalar builtin core
      and bounded Darwin-only acceleration seams
  - third:
    - one real capability-breadth widening beyond the current bounded
      real-only scalar and family-local limits
  - fourth:
    - cleaner runtime/threading, proof, and reviewed-surface maturity where
      it materially strengthens the first three lanes
  - fifth:
    - lower chronology/duplication/maintenance drag across permanent product
      and maintainer surfaces
- The Day 5 product reading is now explicit:
  - Epic 9 should aim to make this repo a serious modern sparse numerical
    library that is:
    - unusually well validated
    - structurally closer to compressed-first compute reality
    - backend-aware in a portable way
    - broader in capability on at least one high-value lane
    - cleaner in permanent product narrative and build topology
- The first implementation-center owners implied by the target are now fixed:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- The second implementation-center owners implied by the target are now fixed:
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - later touched direct-family consumers
- The third implementation-center owners implied by the target are now fixed:
  - `include/sparse_types.h`
  - later touched iterative/eigs/direct public headers and implementation
    seams
- The most important success markers are now explicit:
  - high-value workflows no longer read as linked-list-first by default
  - the library has at least one serious portable accelerated dense/backend
    lane beyond the builtin scalar baseline
  - the repo ships one real additional high-value capability lane
  - runtime concentration and runtime-control complexity are reduced where
    they materially affect credibility
  - package/build truth remains honest and cleaner without fake symmetry
  - the largest mixed-role source/proof owners, chronology residue, and
    duplicated topology surfaces are smaller and easier to navigate
- The first-pass claim fence is now fixed:
  - valid end-state claims may include:
    - more compressed-first
    - more competitive backend ceiling
    - broader than the Epic 8 capability surface
    - better calibrated and better compared
    - cleaner and easier to maintain
  - invalid end-state claims remain:
    - fully generic sparse numerical platform
    - broad complex and mixed-precision maturity
    - symmetric cross-platform product parity
    - broad shared-library product maturity
    - universal best-in-class runtime or reorder quality
- The strongest Day 5 clarification is now fixed:
  - Epic 9 should aim higher than "research-grade but careful"
  - it should not aim so high that it starts lying about product maturity
  - the right target is a bounded state-of-the-art sparse linear algebra
    library with explicit non-claims

### Validation
- Re-read the Sprint 90 Day 5 plan contract.
- Re-read the Day 3 and Day 4 audit artifacts.
- Re-read the Epic 9 project-plan sprint ordering.
- Recorded the target-state reading, success markers, ownership split, and
  first-pass claim fence in the Day 5 artifact.

### Day 5 Exit State
- Sprint 90 now has one bounded Epic 9 target-state contract.
- "State of the art" now has a concrete local meaning for this repo.
- Later Day 6-Day 10 comparison, risk-fence, review, todo, and plan work can
  reference one stable success-marker and claim-fence package.

## Day 6 - Comparison and Measurement Contract Design

### Goal
Freeze the external correctness, runtime, package-shape, and workflow
comparison model for Epic 9 so later sprint work can widen evidence
intentionally instead of opportunistically.

### Actions
- Re-read the Sprint 90 Day 6 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the Day 5 target-state contract and the Day 2 maintained-surface
  ownership map.
- Re-read the closest prior bounded comparison-contract pattern:
  - `docs/planning/EPIC_8/SPRINT_89/artifacts/day8-external-comparison-sweep-design.md`
- Re-read the Epic 9 review's external-comparison and benchmark finding so the
  contract responds to the real credibility gap rather than to generic
  benchmarking pressure.
- Fixed the evidence classes explicitly:
  - maintained correctness
  - maintained package shape
  - bounded runtime-reference
  - advisory but not maintained ecosystem comparison
- Wrote the Day 6 artifact and fixed the claim strength for each evidence
  class in writing.

### Findings
- Sprint 90 now has one explicit Epic 9 comparison-and-measurement contract:
  - maintained correctness comparison lane:
    - bounded external SPD differential proof
  - maintained package-shape comparison lane:
    - local install/export and downstream-consumer proof
  - bounded runtime-reference lane:
    - touched reorder/runtime comparison slices plus canonical reporting
  - advisory but not maintained comparison lanes:
    - broader ecosystem, broader platform, and broader solver-family
      comparison
- The strongest evidence-class order is now fixed:
  - first:
    - correctness evidence strong enough to support bounded product claims
  - second:
    - package-shape and installed-consumer evidence strong enough to support
      bounded product-shape claims
  - third:
    - bounded runtime-reference evidence strong enough to support
      calibration, not superiority
  - fourth:
    - advisory ecosystem comparison strong enough to inform prioritization,
      not to carry maintained product claims
- The exact maintained comparison protocol is now fixed:
  - correctness:
    - `./build/quality-review-cmake/test_chol_csc`
    - retained external-dense-reference readings for `nos4` and `bcsstk04`
    - interpreted through `max|x-x_ref|` plus retained residual strength
  - package shape:
    - `bash tests/test_install.sh`
    - `bash tests/test_cmake_install.sh`
    - interpreted through pass/fail/skip totals plus agreement with the
      maintained static-first consumer/export contract
  - runtime reference:
    - `make bench-reorder-sprint86`
    - `make bench-canonical-report`
    - interpreted as bounded touched-runtime evidence and reproducible
      reporting, not as broad timing proof
- The widening order is now explicit:
  - correctness-only widening may land later where deterministic and
    dependency-bounded
  - runtime/fill/reference widening may land later where still bounded and
    reproducible
  - broader ecosystem comparison remains advisory until a later explicit
    maintained contract widens it
- The claim fence by evidence class is now fixed:
  - strong enough for bounded product claims:
    - maintained SPD external correctness lane
    - maintained install/export and downstream-consumer package-shape lane
  - strong enough for calibration/support claims only:
    - bounded reorder/runtime reference slices
    - canonical benchmark-report generation
    - asymmetric macOS/Windows workflow evidence
  - not strong enough for maintained product claims today:
    - broader ecosystem comparisons not yet under maintained proof
    - ad hoc local timing wins
    - platform-specific speed anecdotes
    - workflow completion outside the retained reviewed/script-owned lanes
- The runtime evidence fence is now explicit:
  - Epic 9 may use runtime evidence to claim bounded touched-lane improvement
    and better calibration
  - Epic 9 may not use runtime evidence to claim universal speed leadership,
    cross-platform timing parity, or benchmark pass/fail guarantees
- The package/workflow comparison reading is now explicit:
  - package-shape truth stays with:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `examples/cmake_example/CMakeLists.txt`
  - Linux remains the strongest reviewed truth lane
  - macOS remains narrower reviewed truth plus supplemental install
    confidence
  - Windows remains the reviewed CMake-first consumer subset only
- The strongest Day 6 clarification is now fixed:
  - Epic 9 needs stronger comparison depth, but not every comparison class
    should be promoted equally
  - correctness and package-shape proof are the maintained claim-bearing lanes
  - runtime evidence remains intentionally bounded and calibration-oriented
  - broader ecosystem comparison remains advisory until a later deliberate
    contract widens it

### Validation
- Re-read the Sprint 90 Day 6 plan contract.
- Re-read the Day 5 target-state contract and Day 2 maintained-surface map.
- Re-read the prior bounded comparison-design artifact from Sprint 89.
- Re-read the Epic 9 review finding on comparison-story limits.
- Recorded the comparison protocol, widening order, and claim fence in the
  Day 6 artifact.

### Day 6 Exit State
- Sprint 90 now has one explicit comparison-and-measurement contract.
- Correctness, package-shape, runtime-reference, and advisory ecosystem
  evidence classes are clearly separated.
- Later Day 7-Day 10 risk-fence, review, todo, and plan work can reference
  one stable evidence-class and claim-strength package.

## Day 7 - Non-goal and Risk Fence

### Goal
Fix the anti-sprawl contract for Epic 9 before the final review, todo, and
project-plan package is written, so later sprint planning cannot widen the
epic into fake product claims or undisciplined rewrite work.

### Actions
- Re-read the Sprint 90 Day 7 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the Day 5 target-state contract and the Day 6
  comparison-and-measurement contract.
- Re-read the Epic 9 todo package's Stage 1 anti-sprawl guidance so the Day 7
  fence stays aligned with the intended epic closure order.
- Fixed the strongest invalid widenings explicitly:
  - fake full compressed rewrite claims
  - fake platform symmetry
  - fake broad complex/mixed-precision claims
  - fake benchmark supremacy claims
  - generic rewrite-without-proof modernization
- Recorded the strongest execution risks:
  - branch-wide sprawl
  - build/package drift
  - proof-owner overload
  - runtime/benchmark overclaim
  - claim inflation from partial wins
- Fixed the strongest carry-forward boundary areas that may still remain after
  Epic 9.
- Wrote the Day 7 artifact and recorded one stable "will do / will not do"
  boundary in working notes.

### Findings
- Sprint 90 now has one explicit Epic 9 non-goal and risk fence:
  - Epic 9 *will* pursue:
    - bounded compressed-first product-model convergence
    - bounded portable backend maturity
    - one real capability-breadth widening
    - bounded runtime/proof/product-coherence improvement
    - bounded build/package/workflow convergence
    - broader but still disciplined comparison depth
  - Epic 9 *will not* pursue:
    - a blanket full-library compressed rewrite
    - fake platform symmetry
    - fake broad complex or mixed-precision maturity
    - fake benchmark supremacy
    - generic rewrite-without-proof modernization
- The strongest invalid widenings are now fixed:
  - no blanket "rewrite the whole library in compressed form" story
  - no product claim that the linked-list shell disappears entirely
  - no broad complex-scalar claim without real implemented and tested support
  - no broad mixed-precision claim without real implemented and tested support
  - no claim of symmetric Linux/macOS/Windows reviewed parity
  - no claim of broad shared-library-first package maturity unless later
    work truly lands and proves it
  - no claim of universal best-in-class runtime, fill quality, or solver
    performance from bounded touched benchmark slices
  - no generic modernization rewrite detached from the ranked contradiction map
    and maintained proof ownership
- The strongest execution risks are now explicit:
  - branch-wide sprawl
  - build/package drift
  - proof-owner overload
  - runtime/benchmark overclaim
  - claim inflation from partial wins
- The stable execution boundary is now fixed:
  - Epic 9 will:
    - move the public product reading closer to compressed-first compute
      reality
    - raise the backend ceiling with at least one bounded portable lane
    - widen capability on at least one high-value real lane
    - reduce the strongest runtime/control, maintenance, chronology, and
      duplication costs where they materially affect product credibility
    - broaden comparison evidence in a bounded, reviewable,
      claim-disciplined way
  - Epic 9 will not:
    - pretend the repo becomes a fully generalized industrial sparse stack in
      one epic
    - collapse maintained proof into ad hoc benchmark or workflow claims
    - reopen every historical rough edge at once
    - treat every large file or every sprint-era comment as equally urgent
    - trade truthfulness for ambition in packaging, backend, runtime, or
      capability wording
- The strongest carry-forward boundary areas are now fixed:
  - broader external correctness comparison beyond the bounded maintained lanes
  - broader ecosystem runtime and reorder/fill comparison depth
  - broad complex and mixed-precision maturity if Epic 9 lands only one
    bounded capability widening
  - full symmetric cross-platform package and workflow maturity
  - complete elimination of all large mixed-role source and proof owners
  - full removal of all historical chronology from every permanent surface
- The strongest Day 7 clarification is now fixed:
  - Epic 9 should be ambitious, but only inside a hard truthfulness fence
  - it may move the repo materially closer to a state-of-the-art sparse
    numerical product
  - it may not broaden claims faster than proof, implementation, and
    ownership maturity

### Validation
- Re-read the Sprint 90 Day 7 plan contract.
- Re-read the Day 5 target-state and Day 6 comparison contracts.
- Re-read the Epic 9 todo package's anti-sprawl guidance.
- Recorded the invalid widenings, execution risks, carry-forward boundaries,
  and stable will-do/will-not-do fence in the Day 7 artifact.

### Day 7 Exit State
- Sprint 90 now has one stable anti-sprawl contract.
- The strongest invalid widenings, execution risks, and carry-forward
  boundary areas are explicit.
- Later Day 8-Day 10 review, todo, and plan writing can use one durable
  "will do / will not do" scope boundary instead of re-deciding epic scope ad
  hoc.

## Day 8 - Review Structure and Evidence Freeze

### Goal
Turn the Day 2-Day 7 audit, target, comparison, and risk outputs into one
final review structure so the Epic 9 review can be drafted from a frozen
evidence package rather than from another open-ended repo-wide reinterpretation.

### Actions
- Re-read the Sprint 90 Day 8 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the prior review structure in:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
  - `docs/planning/EPIC_9/reviews/review-codex-2026-06-24.md`
- Re-read the current Day 3-Day 7 Sprint 90 artifacts so the review outline,
  evidence map, and checklist stay aligned with the frozen contradiction,
  target, comparison, and anti-sprawl contracts.
- Fixed the final review outline explicitly:
  - `Scope`
  - `Baseline`
  - `Executive Verdict`
  - `What The Project Does Unusually Well`
  - `Repository Snapshot`
  - `Findings`
  - `Category Assessment`
  - `Bottom-Line Gap Summary`
- Mapped the evidence sources each major review section must cite.
- Fixed the Day 9 review-writing checklist so the review stays evidence-backed
  and does not drift into generic sparse-library commentary.

### Findings
- Sprint 90 now has one exact Epic 9 review-writing structure:
  - `Scope`
  - `Baseline`
  - `Executive Verdict`
  - `What The Project Does Unusually Well`
  - `Repository Snapshot`
  - `Findings`
  - `Category Assessment`
  - `Bottom-Line Gap Summary`
- The strongest Day 9 writing rule is now fixed:
  - the review must distinguish:
    - actual engineering debt
    - bounded non-claims
    - Epic 8 improvements that already materially moved the repo
- The evidence map is now explicit:
  - `Scope`:
    - Epic 9 project-plan surface plus the live post-Epic-8 tree
  - `Baseline`:
    - Day 2 maintained-surface contract
  - `Executive Verdict`:
    - Day 3 and Day 4 contradiction maps
    - Day 5 target-state contract
    - Day 6 comparison contract
    - Day 7 anti-sprawl fence
  - `Strengths`:
    - validated reviewed baseline
    - bounded external SPD comparison
    - install/export proof
    - materially cleaner front door and support split
    - broad proof ownership
  - `Repository Snapshot`:
    - Day 4 implementation, proof, and support/build/workflow measurements
  - `Findings`:
    - Day 3 product/backend/capability contradictions
    - Day 4 maintainability/coherence/duplication contradictions
    - Day 6 comparison-depth contradiction
  - `Category Assessment`:
    - synthesized from the Day 3-Day 6 contract stack
  - `Bottom-Line Gap Summary`:
    - the frozen ranked closure order from the Sprint 90 artifacts
- The review-writing checklist is now fixed:
  - cite the maintained baseline before making gap claims
  - lead with the verdict, not changelog recap
  - preserve the strongest strengths explicitly
  - keep the findings ranked
  - distinguish real debt from bounded non-claims
  - distinguish current gaps from Epic 8 issues already materially reduced
  - keep comparison ambition disciplined by the Day 6 evidence fence
  - keep product ambition disciplined by the Day 5 target-state contract
  - avoid generic sparse-library commentary detached from repo evidence
- The strongest Day 8 clarification is now fixed:
  - the final Epic 9 review should not be a fresh brainstorm
  - it should be a structured verdict written from a frozen evidence package
  - it may be blunt, but it may not drift away from the Day 2-Day 7 contract
    stack

### Validation
- Re-read the Sprint 90 Day 8 plan contract.
- Re-read the prior Epic review structures.
- Re-read the Day 3-Day 7 Sprint 90 artifacts.
- Recorded the final outline, evidence map, and review-writing checklist in
  the Day 8 artifact.

### Day 8 Exit State
- Sprint 90 now has one frozen review structure and evidence map.
- Day 9 can draft the Epic 9 review from a stable package without reopening
  scope, ranking, or claim-fence questions.
- Later todo and project-plan drafting can inherit the same frozen ranked gap
  set.

## Day 9 - Epic 9 Review Draft

### Goal
Write the full repo-wide Epic 9 review from the frozen Day 8 structure and
evidence package so later todo and project-plan drafting can inherit one
stable ranked finding set.

### Actions
- Re-read the Sprint 90 Day 8 review structure and evidence-freeze artifact.
- Re-read the prior Epic 9 review draft from `2026-06-24` as the closest
  structural baseline.
- Drafted the dated Sprint 90 review at:
  - `docs/planning/EPIC_9/reviews/review-codex-2026-06-25.md`
- Preserved the frozen review structure exactly:
  - `Scope`
  - `Baseline`
  - `Executive Verdict`
  - `What The Project Does Unusually Well`
  - `Repository Snapshot`
  - `Findings`
  - `Category Assessment`
  - `Bottom-Line Gap Summary`
- Kept the frozen writing rules explicit:
  - cite the maintained baseline before making gap claims
  - preserve the strongest post-Epic-8 strengths
  - keep the findings ranked
  - distinguish actual engineering debt from bounded non-claims
  - distinguish current gaps from Epic 8 problems already materially reduced
- Wrote the Day 9 artifact to record the draft state and the preserved finding
  set.

### Findings
- Sprint 90 now has a complete dated Epic 9 review draft at:
  - `docs/planning/EPIC_9/reviews/review-codex-2026-06-25.md`
- The draft preserves the frozen Day 8 structure exactly:
  - `Scope`
  - `Baseline`
  - `Executive Verdict`
  - `What The Project Does Unusually Well`
  - `Repository Snapshot`
  - `Findings`
  - `Category Assessment`
  - `Bottom-Line Gap Summary`
- The draft explicitly covers all requested evaluation dimensions:
  - code efficiency
  - maintainability
  - usability
  - documentation quality
  - cross-surface coherence
  - test coverage
  - state-of-the-art sparse linear algebra readiness
- The review draft now carries the frozen ranked contradiction set:
  1. linked-list-first public/core product model
  2. dense/backend ceiling too low for a top-tier performance claim
  3. capability surface still too narrow
  4. concurrency/runtime story still partial and family-local
  5. maintainability hotspots still too concentrated
  6. sprint-era chronology still leaks into permanent surfaces
  7. build/package/workflow topology still too manual and duplicated
  8. test coverage is strong but still fragmented and operationally heavy
  9. usability is improved but still spread across too many large surfaces
  10. comparison depth is still too bounded for a top competitive claim
- The strongest Day 9 clarification is now fixed:
  - the review draft does not reopen scope
  - it uses the frozen Day 2-Day 8 package
  - it keeps the verdict blunt without widening into fake claims or generic
    sparse-library commentary

### Validation
- Re-read the Sprint 90 Day 8 review-freeze artifact.
- Re-read the prior Epic 9 review draft from `2026-06-24`.
- Drafted the new dated review and checked that it preserves the frozen
  section structure and ranked finding set.
- Recorded the draft state in the Day 9 artifact.

### Day 9 Exit State
- Sprint 90 now has one complete Epic 9 review draft grounded in the frozen
  evidence package.
- Day 10 can convert that ranked review directly into the staged gap-closure
  todo without reopening the repo-wide verdict.

## Day 10 - Epic 9 Gap-Closure Todo Draft

### Goal
Convert the frozen Day 9 review into one ordered Epic 9 closure sequence so
later project-plan drafting can inherit a stable staged execution order.

### Actions
- Re-read the Sprint 90 Day 10 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the full dated Epic 9 review draft from `2026-06-25`.
- Re-read the prior Epic 9 todo draft from `2026-06-24` as the closest
  structural baseline.
- Drafted the dated Epic 9 todo at:
  - `docs/planning/EPIC_9/reviews/todo-codex-2026-06-25.md`
- Preserved the staged closure order implied by the frozen review:
  - target and claim fence first
  - structural product/backend ceilings before broader widening
  - coherence and duplication after major architecture moves
  - comparison-depth widening later
  - final closeout from a measured end-state
- Wrote the Day 10 artifact to record the draft state and preserved stage
  sequence.

### Findings
- Sprint 90 now has a complete dated Epic 9 todo draft at:
  - `docs/planning/EPIC_9/reviews/todo-codex-2026-06-25.md`
- The draft preserves the frozen ranked gap order from the review and converts
  it into one staged execution sequence:
  - Stage 1:
    - hold the Epic 9 target and claim fence
  - Stage 2:
    - heal the core product and compute model
  - Stage 3:
    - raise the dense/backend and runtime ceiling
  - Stage 4:
    - expand the capability envelope
  - Stage 5:
    - heal documentation, naming, and product coherence
  - Stage 6:
    - reduce maintainability concentration
  - Stage 7:
    - reduce duplication in build/package/workflow surfaces
  - Stage 8:
    - broaden comparison and assurance evidence
  - Stage 9:
    - final convergence and closeout
- The drafted todo now explicitly maps the review's contradiction classes to
  remediation stages:
  - compressed-first product-model convergence
  - backend and runtime ceiling
  - capability breadth
  - chronology/coherence cleanup
  - maintainability hotspot reduction
  - build/package/workflow convergence
  - broader comparison depth
- The strongest Day 10 clarification is now fixed:
  - the todo is not just a summary of problems
  - it is the ordered execution sequence the later project plan must inherit
  - it keeps the epic narrow enough to stay truthful while still ambitious

### Validation
- Re-read the Sprint 90 Day 10 plan contract.
- Re-read the full dated review draft from `2026-06-25`.
- Re-read the prior todo draft from `2026-06-24`.
- Drafted the new dated todo and checked that it preserves the frozen ranked
  gap order and staged execution logic.
- Recorded the draft state in the Day 10 artifact.

### Day 10 Exit State
- Sprint 90 now has one complete Epic 9 gap-closure todo draft.
- Day 11 can map that staged sequence directly onto the 10-sprint Epic 9
  project plan without reopening the ranked closure order.

## Day 11 - Project Plan Draft

### Goal
Turn the frozen Day 9 review and Day 10 todo into one complete 10-sprint Epic 9
project plan with bounded hours and stage-ordered sprint sequencing.

### Actions
- Re-read the Sprint 90 Day 11 contract in
  `docs/planning/EPIC_9/SPRINT_90/PLAN.md`.
- Re-read the frozen dated Epic 9 review from `2026-06-25`.
- Re-read the dated Epic 9 todo from `2026-06-25`.
- Re-read the existing `docs/planning/EPIC_9/PROJECT_PLAN.md` draft.
- Updated the project-plan introduction so it now points at the frozen
  `2026-06-25` review and todo package instead of the earlier provisional
  review-only reading.
- Reordered the late-epic sprint sequence so it now follows the Day 10 stage
  order:
  - maintainability reduction before build/package/workflow duplication
- Wrote the Day 11 artifact to record the completed project-plan draft state.

### Findings
- Sprint 90 now has an updated full Epic 9 project plan at:
  - `docs/planning/EPIC_9/PROJECT_PLAN.md`
- The project plan now explicitly inherits the frozen `2026-06-25` review and
  todo package.
- The staged closure order from the todo now maps cleanly onto Sprints 90-99:
  - Sprint 90:
    - baseline, target, comparison, and risk fence
  - Sprint 91:
    - compressed-first product convergence
  - Sprints 92-93:
    - backend ceiling and runtime/threading convergence
  - Sprint 94:
    - capability widening
  - Sprint 95:
    - public narrative and workflow coherence
  - Sprint 96:
    - large-source and giant-test maintainability reduction
  - Sprint 97:
    - build/package/workflow convergence
  - Sprint 98:
    - broader assurance and comparison architecture
  - Sprint 99:
    - final integration, calibration, and closeout
- The strongest Day 11 clarification is now fixed:
  - the plan is no longer only directionally aligned with the review
  - it now follows the same ranked gap order and stage order as the frozen
    todo
  - maintainability now lands before duplication reduction, matching the Day
    10 closure sequence

### Validation
- Re-read the Sprint 90 Day 11 plan contract.
- Re-read the full dated review and todo package from `2026-06-25`.
- Re-read the current Epic 9 project-plan draft.
- Checked that each sprint remains within the `168`-hour cap.
- Checked that the late-epic sprint order now matches the Day 10 stage order.
- Recorded the completed draft state in the Day 11 artifact.

### Day 11 Exit State
- Sprint 90 now has one complete 10-sprint Epic 9 project-plan draft aligned
  to the frozen review and todo package.
- Day 12 can perform cross-document alignment from one coherent review/todo/plan
  package without reopening sprint ordering.
