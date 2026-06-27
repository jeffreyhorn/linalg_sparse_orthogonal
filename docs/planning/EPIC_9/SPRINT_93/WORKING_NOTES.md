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

## Day 6 - ND Runtime Reduction Design

### Goal
Define the bounded implementation contract for the touched ND recursive runtime
seam so Sprint 93 can land one real reviewed-runtime reduction without
reopening broad graph policy, threading, or benchmark-governance work.

### Actions
- Re-read the Sprint 93 Day 6 plan target in
  `docs/planning/EPIC_9/SPRINT_93/PLAN.md`.
- Re-read the Day 5 first-boundary fence:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day5-first-implementation-boundary.md`
- Re-read the touched first-center owner:
  - `src/sparse_reorder_nd.c`
- Re-read the strongest bounded runtime evidence owner:
  - `benchmarks/bench_reorder.c`
- Reconciled the live ND recursive owner against the Day 4 debt split so the
  first code landing stays inside:
  - algorithmic runtime debt
  - directly forced proof/evidence owners only
  - explicit non-claim fence
- Wrote the Day 6 runtime-reduction design artifact.

### Findings
- The exact Sprint 93 first implementation center is now fixed to:
  - `src/sparse_reorder_nd.c`
- The exact runtime reduction target is now fixed to:
  - remove avoidable recursion-side work inside the ND driver before widening
    any graph-policy or threading story
  - prioritize repeated non-leaf overhead that is paid across the reviewed ND
    recursion:
    - temporary side-set collection and repeated full-array passes
    - avoidable heap churn around partition-side bookkeeping
    - recursion-local work that does not change the final permutation or
      policy reading
- The first landing is explicitly not targeting:
  - leaf-AMD semantics as a product-level redesign
  - FM/coarsening policy changes in `src/sparse_graph.c`
  - broad thread-local override cleanup
  - new public runtime knobs
  - detached benchmark-only tuning
- The preserved invariants are now fixed:
  - permutation contract must remain `perm[new_i] = old_i`
  - current ND policy/env interpretation must remain unchanged
  - leaf-vs-non-leaf routing at the current threshold must remain unchanged
  - separator-last ordering must remain unchanged
  - touched reviewed proof owners must stay deterministic under repeated runs
- The strongest directly forced proof and evidence owners are now fixed to:
  - `tests/test_reorder_nd.c`
  - `benchmarks/bench_reorder.c`
- The strongest adjacent owners remain support-only unless the first landing
  truly forces movement:
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `tests/test_graph.c`
- The strongest Day 6 clarification is now explicit:
  - Sprint 93 should first reduce repeated ND driver overhead, not redesign
    ND policy
  - it should preserve the current tuned threshold and current policy surface
    unless the touched recursion-side reduction proves impossible otherwise
  - it should read success as a smaller reviewed ND runtime cost with the same
    ordering semantics, not as a broader graph-quality or threading claim

### Validation
- Re-read the Day 5 boundary fence against the live ND owner.
- Re-read the bounded runtime evidence owner in `benchmarks/bench_reorder.c`.
- Reconfirmed that the first runtime landing stays inside the Day 4
  algorithmic/runtime-control/proof-topology split.

### Day 6 Exit State
- Sprint 93 now has one explicit ND runtime-reduction implementation contract.
- Day 7 can land the touched recursion-side runtime batch without reopening
  broad graph policy, runtime-control, or benchmark-governance work.

## Day 7 - ND Runtime Reduction Batch

### Goal
Land one bounded recursion-side runtime reduction inside the reviewed ND owner
without changing ND policy semantics, widening graph-policy work, or reopening
proof topology beyond directly forced validation.

### Actions
- Re-read the Day 6 runtime-reduction contract:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day6-nd-runtime-reduction-design.md`
- Re-read the touched first-center owner:
  - `src/sparse_reorder_nd.c`
- Reworked the per-recursion partition-side bookkeeping in `nd_recurse`:
  - added `nd_collect_partition_vertices(...)`
  - replaced separate `vs0` / `vs1` heap allocations with one `scratch`
    buffer sized to the current subgraph
  - packed side 0, side 1, and separator vertices in one scratch layout
  - reused that packed layout for both recursive side calls and separator
    emission
- Verified that the landing stayed inside the Day 5 fence:
  - no policy/env changes
  - no threshold changes
  - no graph-policy changes
  - no proof-owner widening beyond validation of the existing touched owners
- Ran the required validation queue:
  - `make format`
  - `make lint`
  - `make test`
- Wrote the Day 7 implementation artifact.

### Findings
- The Day 7 landing stayed inside the exact Day 6 contract:
  - the only code owner touched was `src/sparse_reorder_nd.c`
  - no directly forced edits were needed in:
    - `src/sparse_graph.c`
    - `src/sparse_graph_refine.c`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `benchmarks/bench_reorder.c`
- The landed runtime reduction is now explicit:
  - ND no longer allocates two separate side arrays (`vs0`, `vs1`) per
    non-leaf recursion frame
  - ND no longer performs a separate full post-recursion scan over `part[]`
    to emit separators
  - one `scratch` buffer now carries:
    - side 0 vertices
    - side 1 vertices
    - separator vertices
  - the recursive side calls and final separator-last emission both consume
    that same packed layout
- The preserved semantics remained unchanged:
  - per-side vertex order is still ascending because packing still walks
    `part[]` left-to-right
  - `perm[new_i] = old_i` stays unchanged
  - separator-last behavior stays unchanged
  - current threshold, policy, and env/control interpretation stay unchanged
- Validation passed cleanly after the landing:
  - `make format`
  - `make lint`
  - `make test`
- The strongest Day 7 clarification is now explicit:
  - Sprint 93's first code batch reduced recursion-side overhead without
    widening into graph-policy or threading cleanup
  - proof and benchmark owners stayed validation-only, which keeps the next
    rerank honest before any further widening

### Validation
- `make format`
- `make lint`
- `make test`

### Day 7 Exit State
- Sprint 93 now has one landed ND recursion-side runtime reduction batch.
- The first implementation seam reduced heap churn and post-partition scan
  cost without changing ND ordering semantics.
- Day 8 can now rerank the remaining runtime debt from the post-landing tree
  instead of from the pre-landing design state.

## Day 8 - Post-Landing Audit and Rerank

### Goal
Re-rank the remaining runtime and threading work after the Day 7 ND landing
so Sprint 93's second implementation center is chosen from live post-landing
evidence rather than from the original Day 3 runtime map alone.

### Actions
- Re-read the Day 7 landing artifact:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day7-nd-runtime-reduction-batch.md`
- Re-read the closest prior post-landing rerank patterns:
  - `docs/planning/EPIC_9/SPRINT_91/artifacts/day7-post-landing-audit-and-rerank.md`
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day7-post-landing-audit-and-rerank.md`
- Re-read the Sprint 93 plan around the post-landing rerank and Day 9 control
  design steps:
  - `docs/planning/EPIC_9/SPRINT_93/PLAN.md`
- Re-read the strongest remaining runtime-control and support-only owners from
  the live tree:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`
- Reconciled the live post-Day-7 state against the Sprint 93 project-plan
  order so the rerank reflects:
  - landed algorithmic runtime reduction already done
  - runtime-control cleanup next if still required
  - proof-surface rebalancing and runtime evidence only after that
- Wrote the Day 8 rerank artifact and fixed the exact Day 9 design center in
  writing.

### Findings
- The Day 7 landing closed the strongest first Sprint 93 contradiction:
  - the reviewed ND owner no longer pays the same recursion-side heap churn
    and separator-emission scan cost at each non-leaf recursion frame
  - a second immediate recursion-side runtime batch is no longer the
    highest-value remaining Sprint 93 move
- The ranked remaining runtime map is now:
  - strongest first target:
    - runtime-control cleanup centered on the ND policy/env and override
      plumbing in `src/sparse_reorder_nd.c`
  - strongest second target:
    - proof-surface rebalancing only after the touched runtime-control seam
      is bounded cleanly
  - strongest third target:
    - bounded benchmark and runtime-evidence follow-through after the runtime
      model itself is sharper
  - strongest support-only but real target:
    - maintainer and public wording only where later control cleanup or
      runtime evidence truly changes the maintained contract reading
- The strongest remaining contradiction is now runtime-control sharpness:
  - `src/sparse_reorder_nd.c` still carries the main compatibility env
    parsing and override orchestration for the ND runtime lane
  - the touched runtime story still depends on a wide set of internal
    policy/override seams:
    - ND profile override
    - ND base-threshold hook
    - graph coarsening override
    - coarsest-bisection override
    - separator-lift override
    - related compatibility env normalization
  - that now outranks proof rebalancing because the next proof/evidence pass
    should validate a cleaner touched runtime model rather than preserve a
    looser one
- The exact Day 9 design center is now fixed to:
  - `src/sparse_reorder_nd.c`
- The strongest directly forced support-only follow-through, only if the Day 9
  contract truly forces movement, is now fixed to:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`
- The strongest Day 8 clarification is now explicit:
  - Sprint 93 should not widen into proof-topology or benchmark/reporting work
    before the touched runtime-control seam is bounded
  - it should not treat generic threading cleanup as stronger than the
    touched ND runtime model itself
  - it should sharpen the control model first, then validate and report from
    that cleaner touched state

### Validation
- Re-read the Day 7 landing artifact against the Sprint 93 Day 8 rerank step
  in `PLAN.md`.
- Re-read the strongest remaining runtime-control and support-only owners from
  the live tree.
- Reconfirmed that the live post-Day-7 state moves Sprint 93 from algorithmic
  runtime reduction into runtime-control cleanup rather than into a second
  immediate recursion-side batch.

### Day 8 Exit State
- The strongest remaining Sprint 93 seam is now explicit after the first ND
  runtime landing.
- The second implementation center stays code-owned and is fixed to
  runtime-control cleanup on the touched ND owner.
- Day 9 can now define one exact bounded runtime-control cleanup contract from
  the live post-Day-7 tree.

## Day 9 - Runtime-Control Cleanup Design

### Goal
Freeze one exact Day 10 cleanup contract so Sprint 93 can sharpen the touched
ND runtime/threading control model without turning the second batch into a
broad graph-policy rewrite, proof-topology pass, or generic threading sweep.

### Actions
- Re-read the Day 8 rerank artifact:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day8-post-landing-audit-and-rerank.md`
- Re-read the Sprint 93 plan around the Day 9 and Day 10 control-cleanup
  steps:
  - `docs/planning/EPIC_9/SPRINT_93/PLAN.md`
- Re-read the touched runtime-control owner and its strongest staging seams:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph_internal.h`
- Re-read the main live control cluster in `src/sparse_reorder_nd.c`:
  - compatibility env/default policy normalization
  - ND profile override
  - ND base-threshold hook
  - graph override begin/end staging inside `sparse_reorder_nd_with_policy(...)`
- Reconciled the live control seam against the Sprint 93 contract so the next
  batch stays centered on:
  - control-model sharpness
  - preserved runtime-policy results
  - deferred proof/evidence widening
- Wrote the Day 9 design artifact and fixed the exact Day 10 cleanup center in
  writing.

### Findings
- Sprint 93 now has one exact second implementation contract:
  - required Day 10 center:
    - `src/sparse_reorder_nd.c`
  - directly forced support-only follow-through only if the Day 10 batch truly
    needs them:
    - `src/sparse_reorder_nd_internal.h`
    - `src/sparse_graph_internal.h`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `benchmarks/bench_reorder.c`
  - strongest later surfaces only if runtime-control cleanup exposes a real
    maintained-contract mismatch:
    - `tests/test_threads.c`
    - `tests/test_omp.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The exact Day 10 target is now explicit:
  - stop treating ND runtime-control as a loose cluster of compatibility
    parsing plus stacked override begin/end calls
  - keep the touched cleanup centered on `src/sparse_reorder_nd.c`
  - preserve the current runtime-policy results while making the control seam
    smaller and sharper
- In practical terms, the Day 10 batch is now fixed around:
  - consolidating the ND compatibility/default policy normalization path
  - tightening the override-staging seam around:
    - coarsening override
    - coarsen-floor-ratio override
    - coarsening-CV-fallthrough override
    - coarsest-bisection override
    - separator-lift override
  - preserving the current typed-policy and compatibility semantics:
    - `sparse_reorder_nd_default_policy()` remains the baseline owner
    - typed policy still wins where the shipped contract says it should win
    - current env names and accepted values stay intact
  - preserving the touched benchmark/test-only hooks unless a strictly smaller
    owner can preserve them cleanly:
    - ND profile override
    - ND base-threshold hook
- The strongest Day 9 clarification is now explicit:
  - Day 10 should not become a generic graph-policy redesign
  - Day 10 should not widen into FM/coarsening algorithm changes
  - Day 10 should not reopen proof-topology or benchmark/reporting work ahead
    of the touched control-model cleanup
  - Day 10 should not widen into public or maintainer wording detached from a
    real touched runtime-control movement

### Validation
- Re-read the Day 8 rerank against the Sprint 93 Day 9 and Day 10 plan steps.
- Re-read the strongest live control-seam owners in
  `src/sparse_reorder_nd.c`, `src/sparse_reorder_nd_internal.h`, and
  `src/sparse_graph_internal.h`.
- Reconfirmed that the smallest truthful Day 10 batch is control-model cleanup
  on the touched ND owner rather than proof-topology, benchmark, or generic
  threading widening.

### Day 9 Exit State
- The second Sprint 93 implementation contract is explicit before code moves.
- Day 10 now has one exact bounded center:
  - `src/sparse_reorder_nd.c`
- Later proof, benchmark, and support work remains clearly sequenced behind a
  real landed runtime-control improvement.

## Day 10 - Runtime-Control Cleanup Batch

### Goal
Land one bounded runtime-control cleanup inside the touched ND owner without
changing current runtime-policy results, widening into graph-policy work, or
pulling proof and benchmark follow-through forward before they are needed.

### Actions
- Re-read the Day 9 cleanup contract:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day9-runtime-control-cleanup-design.md`
- Re-read the touched runtime-control owner:
  - `src/sparse_reorder_nd.c`
- Split the touched ND control seam into smaller local owners:
  - added a baseline default-policy constructor:
    - `nd_default_policy_baseline()`
  - added a compatibility-override application seam:
    - `nd_apply_compat_policy_overrides(...)`
  - added a scoped graph-override staging owner:
    - `nd_graph_override_scope_begin(...)`
    - `nd_graph_override_scope_end(...)`
- Rewired `sparse_reorder_nd_default_policy()` to read as:
  - baseline defaults first
  - compatibility overrides second
- Rewired `sparse_reorder_nd_with_policy(...)` to route the touched graph
  override cluster through the new scoped helper instead of manually spelling
  each begin/end call in the main path
- Verified that the landing stayed inside the Day 9 fence:
  - no env-name changes
  - no typed-policy precedence changes
  - no threshold-hook changes
  - no graph-policy algorithm changes
  - no proof or benchmark widening
- Ran the required validation queue:
  - `make format`
  - `make lint`
  - `make test`
- Wrote the Day 10 implementation artifact.

### Findings
- The Day 10 landing stayed inside the exact Day 9 contract:
  - the only code owner touched was `src/sparse_reorder_nd.c`
  - no directly forced edits were needed in:
    - `src/sparse_reorder_nd_internal.h`
    - `src/sparse_graph_internal.h`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `benchmarks/bench_reorder.c`
- The landed cleanup is now explicit:
  - ND default policy construction is split into:
    - one baseline owner with the shipped default values
    - one compatibility-override application seam
  - the graph override begin/end stack is now grouped behind one scoped
    helper:
    - `nd_graph_override_scope_begin(...)`
    - `nd_graph_override_scope_end(...)`
  - `sparse_reorder_nd_with_policy(...)` now applies the touched
    graph-policy override cluster through that one scoped seam instead of
    manually spelling each begin/end call in the main execution path
- The preserved semantics remained unchanged:
  - `sparse_reorder_nd_default_policy()` still returns the same effective
    compatibility-default policy surface
  - current env names and accepted values stay unchanged
  - typed-policy precedence remains unchanged
  - current benchmark/test-only hooks stay intact:
    - ND profile override
    - ND base-threshold hook
  - ND ordering semantics and runtime-policy results stay unchanged
- Validation passed cleanly after the landing:
  - `make format`
  - `make lint`
  - `make test`
- The strongest Day 10 clarification is now explicit:
  - Sprint 93's second code batch sharpened the touched ND control seam
    without widening into proof-topology, benchmark/reporting, or public
    runtime-claim work
  - the remaining queue can now move to proof and runtime-evidence work from
    a cleaner control-model state

### Validation
- `make format`
- `make lint`
- `make test`

### Day 10 Exit State
- Sprint 93 now has one landed ND runtime-control cleanup batch.
- The touched ND control model is smaller and sharper without changing current
  runtime-policy behavior.
- Day 11 can now design the remaining proof-surface rebalancing and bounded
  runtime-evidence follow-through from that cleaner touched control seam.

## Day 11 - Proof-Surface Rebalancing and Runtime Evidence Design

### Goal
Freeze one exact Day 12 evidence contract so Sprint 93 can close the remaining
runtime gap from the cleaner Day 10 control seam without reopening broad
proof, benchmark-governance, or support-surface churn.

### Actions
- Re-read the Day 10 cleanup artifact:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day10-runtime-control-cleanup-batch.md`
- Re-read the Sprint 93 plan around the Day 11 and Day 12 proof/evidence
  steps:
  - `docs/planning/EPIC_9/SPRINT_93/PLAN.md`
- Re-read the retained runtime-evidence owner:
  - `benchmarks/bench_reorder.c`
- Re-read the strongest adjacent proof and reporting owners from the live
  tree:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- Reconciled the live post-Day-10 state against the remaining Sprint 93 queue
  so the next batch stays centered on:
  - bounded runtime evidence
  - no speculative proof-owner movement
  - no broad reporting or support churn
- Wrote the Day 11 design artifact and fixed the exact Day 12 center in
  writing.

### Findings
- Sprint 93 now has one exact Day 12 follow-through contract:
  - required Day 12 center:
    - `benchmarks/bench_reorder.c`
  - directly forced support-only follow-through only if the Day 12 batch truly
    needs them:
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `scripts/bench_canonical_report.sh`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - retained later surfaces unless the evidence batch exposes a real contract
    mismatch:
    - `README.md`
    - `INSTALL.md`
    - `tests/test_threads.c`
    - `tests/test_omp.c`
- The exact Day 12 center is now explicit:
  - keep the remaining Sprint 93 gap evidence-owned rather than proof-owned
  - use the retained reorder benchmark owner:
    - `bench_reorder --sprint86-slice`
  - expose the bounded runtime evidence needed for the touched ND lane after
    the Day 7 runtime reduction and Day 10 control cleanup
- The strongest reason for that choice is now explicit:
  - the touched proof owners already passed cleanly after the Day 10 landing:
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
  - no new correctness contradiction surfaced from the runtime or
    control-model batches
  - the remaining gap is not baseline proof trust anymore
  - the remaining gap is bounded runtime evidence shape:
    - what Sprint 93 wants to keep reporting about the touched ND lane
    - how the Sprint 86 slice should read after the Day 7 recursion-side
      reduction
    - whether the touched benchmark lane needs a smaller, cleaner emitted
      shape before closeout
- The proof-topology call is now explicit:
  - a Day 12 proof-owner rebalance is not currently required
  - `tests/test_reorder_nd.c` remains heavy, but the Day 10 landing did not
    force new proof movement or expose weakened trust
  - proof-owner movement should land only if the evidence batch shows a real
    mismatch that the current reviewed proof surfaces cannot explain or
    validate
- The strongest Day 11 clarification is now explicit:
  - Day 12 should not become another `src/sparse_reorder_nd.c`
    implementation batch
  - Day 12 should not widen into generic proof splitting just because the
    runtime owner is large
  - Day 12 should not widen canonical reporting beyond the touched reorder
    lane unless the bounded evidence contract truly forces it

### Validation
- Re-read the Day 10 cleanup artifact against the Sprint 93 Day 11 and Day 12
  plan steps.
- Re-read the retained runtime-evidence owner in `benchmarks/bench_reorder.c`
  and the strongest adjacent proof/reporting owners from the live tree.
- Reconfirmed that the smallest truthful Day 12 batch is bounded
  runtime-evidence follow-through on the touched reorder benchmark lane rather
  than proof-owner rebalancing.

### Day 11 Exit State
- The remaining Sprint 93 gap is explicit before the final follow-through
  batch.
- Day 12 now has one exact bounded center:
  - `benchmarks/bench_reorder.c`
- Proof-owner movement and support wording remain sequenced behind real
  evidence changes.
