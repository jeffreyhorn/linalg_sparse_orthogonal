# Sprint 93 Working Notes

## Day 1 - Scope and Runtime Baseline

### Goal
Turn the Sprint 93 project-plan section and the Sprint 92 validated closeout
into one bounded runtime-scalability, threading, and ND-convergence execution
package before any runtime audit, contract design, or implementation lands.

### Actions
- Re-read the Sprint 93 contract in
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 93 day-by-day plan in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the closest prior closeout and handoff surfaces:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_9/SPRINT_92/RETROSPECTIVE.md`
- Re-read the closest prior Epic 9 planning and runtime-contract surfaces:
  - `docs/planning/EPIC_9/SPRINT_91/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day6-comparison-and-measurement-contract-design.md`
- Reconfirmed that the strongest local reviewed entry point still begins at:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest likely Sprint 93 touch surfaces by line count and
  owner role:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_amd_qg.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_threads.c`
  - `tests/test_omp.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Wrote the Day 1 scope artifact and authoritative-input list.

### Findings
- Sprint 93 begins from a validated Sprint 92 close state, not from another
  generic backend or benchmark reset:
  - strongest local reviewed baseline remains `make quality-review-full`
  - reviewed CMake parity was re-materialized live and remains explicit:
    - `ctest -N --test-dir build/quality-review-cmake` = `53`
    - Makefile/CMake parity = `53 vs 53`
- Sprint 92 already moved the strongest prior backend contradiction:
  - the shared dense owner now has one bounded builtin-vs-portable backend
    seam
  - LDLT now converges onto the same bounded backend reading instead of a
    separate family-local acceleration pocket
  - benchmark-side backend observability is explicit enough that Sprint 93
    does not need to reopen backend truthfulness first
- That means Sprint 93 can start from the next real Epic 9 contradiction
  center:
  - reviewed runtime concentration, threading/runtime contract sharpness, and
    ND-convergence follow-through
- The highest-value Sprint 93 package is now fixed explicitly around:
  - reviewed runtime audit
  - threading/runtime contract design
  - ND runtime reduction design and batch
  - runtime-control cleanup
  - proof-surface rebalancing
  - runtime evidence follow-through
- The live tree currently points most strongly at these Sprint 93 surfaces:
  - strongest runtime and reordering implementation owners:
    - `src/sparse_reorder_nd.c` = `771`
    - `src/sparse_graph.c` = `841`
    - `src/sparse_graph_refine.c` = `602`
    - `src/sparse_reorder_amd_qg.c` = `611`
  - strongest proof-owner tests likely to matter:
    - `tests/test_reorder_nd.c` = `2340`
    - `tests/test_graph.c` = `2925`
    - `tests/test_threads.c` = `690`
    - `tests/test_omp.c` = `451`
  - strongest benchmark and runtime-evidence owners:
    - `benchmarks/bench_reorder.c` = `338`
    - `benchmarks/bench_amd_qg.c` = `332`
    - `benchmarks/bench_iterative_reuse.c` = `395`
    - `scripts/bench_canonical_report.sh` = `101`
  - strongest support, build, and workflow surfaces if runtime work forces
    follow-through:
    - `README.md` = `1136`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `730`
    - `Makefile` = `928`
    - `CMakeLists.txt` = `425`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
- Sprint 93 is explicitly bounded against:
  - reopening the Sprint 92 backend lane as the first owner again
  - promising broad runtime scalability beyond the maintained reviewed lane
    before touched runtime proof improves
  - widening into capability-surface or packaging-product work before the
    runtime concentration seam is reduced
  - treating benchmark timing alone as stronger than reviewed executable truth
    or maintained workflow/proof-owner surfaces

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the strongest likely runtime, proof, benchmark, and workflow
  surfaces by live file size and owner role.

### Day 1 Exit State
- Sprint 93 now starts from one precise runtime-scalability, threading, and
  ND-convergence execution package rather than from a generic "speed up graph
  and reorder" bucket.
- The strongest likely touch surfaces, preserved non-goals, and maintained
  reviewed starting truth are fixed in writing before the validation and
  maintained-surface recheck begins.
- Day 2 can now freeze the authoritative reviewed, benchmark, install/export,
  and workflow truth split without reopening the Day 1 scope question.

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, benchmark, install/export, example, and workflow truth split before
Sprint 93 begins runtime-, threading-, and ND-focused implementation work on
the graph and reorder surfaces.

### Actions
- Re-read the Sprint 93 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day2-validation-baseline-and-maintained-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained Sprint 93
  truth surfaces:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_threads`
  - `./build/quality-review-cmake/test_omp`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so Sprint 93 does
  not overclaim reviewed runtime parity, threading breadth, or install/export
  coverage while touching graph and reorder concentration.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 93 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for
  runtime- and ND-convergence work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial runtime-contract, proof-owner, benchmark, or support-surface
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 93
  code lands:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest reviewed executable truth owners for Sprint 93’s runtime lane
  are now fixed around:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_threads`
  - `./build/quality-review-cmake/test_omp`
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
  - macOS remains a narrower reviewed Apple Clang lane plus a supplemental
    static-first install/`pkg-config` confidence lane
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim reviewed Makefile parity or separate reviewed install-validation
    parity
- The highest-signal rerun set is now fixed for the rest of Sprint 93:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_threads`
  - `./build/quality-review-cmake/test_omp`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 93 should read runtime- and threading-focused changes against the
    reviewed graph, reorder, and concurrency proof owners
  - canonical reporting remains a bounded command/script-owned evidence
    surface, not a reviewed-binary runtime-parity claim

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the presence of the strongest reviewed runtime, example,
  install/export, and workflow owner surfaces.

### Day 2 Exit State
- Sprint 93 now has one explicit validation and maintained-surface contract
  before runtime implementation begins.
- Reviewed graph/runtime binaries remain the main executable truth anchor.
- Canonical benchmark reporting remains command/script owned.
- Install/export proof remains script owned.
- Workflow lanes remain layered support evidence rather than broad
  cross-platform runtime-parity claims.

## Day 3 - Reviewed Runtime Audit

### Goal
Reduce Sprint 93's broad runtime-scalability and threading problem to one
ranked live contradiction map centered on the reviewed ND long pole, the
highest-value graph/reorder owners, and the remaining runtime-control and
proof concentration seams.

### Actions
- Re-read the Sprint 93 Day 3 plan target in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the closest prior audit pattern:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day3-dense-hotspot-profiling-audit.md`
- Re-read the strongest current reviewed runtime owner:
  - `src/sparse_reorder_nd.c`
- Re-read the strongest current reviewed ND proof owner:
  - `tests/test_reorder_nd.c`
- Re-read the strongest concurrency proof surfaces:
  - `tests/test_threads.c`
  - `tests/test_omp.c`
- Re-read the strongest bounded runtime-evidence surface:
  - `benchmarks/bench_reorder.c`
- Re-searched the live runtime/control seams across source, test, benchmark,
  README, and maintainer surfaces:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - ND base-threshold controls
  - graph and FM thread-local override owners
  - reviewed Windows/macOS/Linux runtime-support wording
- Wrote the ranked Day 3 runtime audit artifact.

### Findings
- Sprint 93's broad runtime problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - the ND recursive driver and its graph-partition pipeline concentrated in
      `src/sparse_reorder_nd.c`, `src/sparse_graph.c`, and
      `src/sparse_graph_refine.c`
  - strongest second target:
    - runtime-control and thread-local override complexity across the ND and
      graph pipeline, where tuning and profile hooks are real but now too
      diffuse to read as one clean runtime model
  - strongest third target:
    - proof concentration in `tests/test_reorder_nd.c` and `tests/test_graph.c`,
      where major runtime truth still sits inside large single-binary owners
  - strongest fourth target:
    - bounded benchmark and evidence follow-through so touched runtime changes
      remain measurable against the maintained reorder lane
  - strongest support-only but real target:
    - public and maintainer wording that still needs to stay truthful about
      bounded threading maturity and reviewed runtime claims
- The strongest current contradiction is still the reviewed ND long pole:
  - `src/sparse_reorder_nd.c` remains the recursive entry point and still owns
    threshold, profile, leaf, and recursion-side runtime behavior
  - the current tree still centers the reviewed runtime hotspot on
    `test_reorder_nd`, not on broad library execution or on backend work
  - `benchmarks/bench_reorder.c` already carries a bounded touched rerun lane
    (`--sprint86-slice`, `--nd-threshold`, `--skip-factor`) that matches this
    ownership story rather than widening it
- The strongest second contradiction is runtime-control complexity:
  - `src/sparse_reorder_nd.c` still exposes profile env and override hooks
  - `src/sparse_graph.c` and `src/sparse_graph_internal.h` still carry a
    growing set of thread-local FM / coarsening / separator override seams
  - that control surface is useful for diagnosis and bounded tuning, but it
    now reads as a real cleanup target before the repo can claim a sharper
    runtime/threading model
- The strongest third contradiction is proof concentration rather than proof
  absence:
  - `tests/test_reorder_nd.c` remains a giant owner containing fixture loading,
    runtime-control coverage, and major reviewed runtime proof
  - `tests/test_graph.c` remains the adjacent giant owner for partition and FM
    behavior
  - `tests/test_threads.c` and `tests/test_omp.c` provide concurrency proof,
    but they do not yet rebalance the cost concentration centered on the ND and
    graph review owners
- The strongest fourth contradiction is evidence follow-through:
  - `benchmarks/bench_reorder.c` is already the bounded runtime evidence owner
  - `benchmarks/bench_amd_qg.c` and `benchmarks/bench_iterative_reuse.c`
    remain adjacent measurement surfaces, but they are clearly second-tier
    relative to the reviewed ND lane
  - canonical reporting remains real, but it should not become the first
    implementation owner
- Sprint 93's fix-now vs deferred split is now clearer:
  - should drive Sprint 93 implementation:
    - ND recursive runtime seam
    - runtime-control cleanup on touched ND/graph owners
    - proof-surface rebalancing only where it reduces reviewed-runtime cost
  - remains later or bounded non-claim territory:
    - fake broad multithreading maturity
    - generic graph/reorder rewrite everywhere at once
    - broad benchmark-superiority claims
    - capability or packaging work outside the touched runtime lane

### Validation
- Re-read the current ND recursive owner in `src/sparse_reorder_nd.c`.
- Re-read the current ND reviewed proof owner in `tests/test_reorder_nd.c`.
- Re-read concurrency proof owners in `tests/test_threads.c` and
  `tests/test_omp.c`.
- Re-read the bounded runtime evidence owner in `benchmarks/bench_reorder.c`.
- Re-searched the live runtime/profile/override seams across the touched tree.

### Day 3 Exit State
- Sprint 93 now has one ranked live reviewed-runtime contradiction map grounded
  in the current post-Sprint-92 tree.
- The strongest first Sprint 93 implementation center is fixed to the ND
  recursive runtime seam and its adjacent graph-partition owner surfaces.
- Day 4 can freeze the runtime/threading contract without reopening the
  ranked runtime order.

## Day 4 - Threading and Runtime Contract Design

### Goal
Separate remaining Sprint 93 debt into algorithmic runtime concentration,
runtime-control complexity, and proof-topology cost so the first
implementation boundary can stay bounded to the highest-value reviewed seam.

### Actions
- Re-read the Sprint 93 Day 4 plan target in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the Day 3 ranked runtime audit:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day3-reviewed-runtime-audit.md`
- Re-read the closest contract-design pattern:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day5-portable-backend-abi-and-runtime-contract-design.md`
- Re-read the Sprint 93 section of the Epic 9 project plan so the design
  stays inside the explicit runtime/threading/non-claim fence.
- Reconciled the ranked Day 3 runtime owners against the project-plan item
  split:
  - ND runtime reduction batch
  - runtime-control cleanup
  - proof-surface rebalancing
  - runtime evidence follow-through
- Wrote the Day 4 runtime/threading contract artifact.

### Findings
- Sprint 93 now has one explicit runtime/threading contract:
  - algorithmic runtime debt:
    - means repeated work, recursion-side cost, or graph-partition cost on the
      touched reviewed ND lane
    - remains the strongest first-class implementation target
  - runtime-control debt:
    - means profile env vars, threshold knobs, or thread-local FM/coarsening
      overrides that are still useful but too diffuse to read as one clean
      runtime model
    - remains a real Sprint 93 target, but sequenced behind the first
      algorithmic seam unless directly forced
  - proof-topology debt:
    - means reviewed runtime cost caused by giant binary owners or repeated
      heavy fixture/proof concentration rather than by the algorithm itself
    - remains real Sprint 93 work, but only where rebalancing reduces cost
      without weakening correctness trust
- The strongest Day 4 clarification is now explicit:
  - Sprint 93 should not treat all remaining runtime debt as a concurrency
    problem
  - it should not treat every thread-local override as equally urgent
  - it should first improve the touched reviewed ND runtime seam, then tighten
    the runtime-control story only where the same seam still depends on it
- The preserved non-claim fence is now fixed more sharply:
  - no fake broad scaling victory
  - no fake repo-wide threading maturity claim
  - no broad cross-platform runtime parity claim
  - no benchmark-superiority claim detached from the reviewed proof owners
- The strongest direct-owner reading is now explicit:
  - first-center implementation owners:
    - `src/sparse_reorder_nd.c`
    - `src/sparse_graph.c`
    - `src/sparse_graph_refine.c`
  - second-center runtime-control owners if truly forced:
    - `src/sparse_graph_internal.h`
    - `src/sparse_reorder_nd_internal.h`
    - adjacent profile / override test coverage in `tests/test_reorder_nd.c`
      and `tests/test_graph.c`
  - later proof-only or support-only owners unless the first landing forces
    movement:
    - `tests/test_threads.c`
    - `tests/test_omp.c`
    - `benchmarks/bench_reorder.c`
    - `README.md`
    - `docs/maintainer_guide.md`

### Validation
- Re-read the Day 3 runtime audit against the Sprint 93 project-plan contract.
- Re-read the closest prior contract-design artifact for bounded-seam format.
- Reconfirmed that Sprint 93's runtime/threading lane stays inside the Epic 9
  non-goal fence.

### Day 4 Exit State
- Sprint 93 now has one explicit threading/runtime contract before the first
  implementation fence is frozen.
- Algorithmic runtime debt, runtime-control debt, and proof-topology debt are
  separated in writing.
- Day 5 can freeze one bounded first landing without reopening generic
  threading or benchmark claims.

## Day 5 - First Implementation Boundary

### Goal
Fix one bounded first implementation fence so Sprint 93 starts with the
highest-value ND runtime seam instead of generic graph, threading, or
benchmark churn.

### Actions
- Re-read the Sprint 93 Day 5 plan target in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the Day 3 runtime audit:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day3-reviewed-runtime-audit.md`
- Re-read the Day 4 runtime/threading contract:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day4-threading-and-runtime-contract-design.md`
- Re-read the closest prior first-boundary pattern:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day4-first-implementation-boundary.md`
- Reconciled the ranked runtime seam against the Day 4 debt split:
  - algorithmic ND runtime debt first
  - runtime-control cleanup second
  - proof-topology and evidence follow-through later unless forced
- Wrote the Day 5 boundary artifact and updated working notes.

### Findings
- Sprint 93 now has one explicit first implementation fence:
  - required first landing:
    - `src/sparse_reorder_nd.c`
    - the matching touched recursion-side and leaf/runtime seam behind the
      reviewed ND owner
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `src/sparse_graph.c`
    - `src/sparse_graph_refine.c`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `benchmarks/bench_reorder.c`
  - explicitly later unless the first landing truly forces movement:
    - `src/sparse_graph_internal.h`
    - `src/sparse_reorder_nd_internal.h`
    - `tests/test_threads.c`
    - `tests/test_omp.c`
    - `benchmarks/bench_amd_qg.c`
    - `benchmarks/bench_iterative_reuse.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
    - `Makefile`
    - `CMakeLists.txt`
    - install/export and workflow surfaces
- The strongest Day 5 clarification is now explicit:
  - Sprint 93 should start by improving the ND recursive runtime seam
  - it should not begin by widening every graph-partition, threading, or
    runtime-control owner at once
  - it should not reopen proof naming, workflow wording, or broad benchmark
    interpretation in the first batch unless the touched runtime seam itself
    truly forces it
- The first batch now explicitly defers:
  - broad graph/reorder rewrites
  - generic multithreading everywhere
  - runtime-control cleanup detached from the touched ND seam
  - proof-surface restructuring detached from real reviewed-runtime savings
  - benchmark/reporting widening detached from the first runtime landing
  - public support-surface wording churn detached from the touched seam

### Validation
- Re-read the Day 3 and Day 4 artifacts against the Sprint 93 project-plan
  contract.
- Re-read the closest prior boundary-freeze artifact for bounded-seam format.
- Reconfirmed that the first landing stays inside the Day 4 runtime/threading
  non-claim fence.

### Day 5 Exit State
- Sprint 93 now has one explicit first implementation boundary.
- The first code landing is fixed to the ND recursive runtime owner with only
  the strongest adjacent graph and proof surfaces as directly forced
  follow-through.
- Day 6 can define the runtime-reduction implementation contract without
  reopening the ranked first-center choice.
