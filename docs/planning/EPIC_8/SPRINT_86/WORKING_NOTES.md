# Sprint 86 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 86 baseline for Epic 8 by grounding the sprint in
the validated Sprint 85 close state, the live Sprint 86 project-plan section,
and the current reviewed-runtime, reorder, nested-dissection, benchmark, and
support-surface hotspots rather than another generic “optimize tests” reset.

### Actions
- Re-read the Sprint 86 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 86 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the strongest Sprint 85 closeout context:
  - `docs/planning/EPIC_8/SPRINT_85/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_85/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 86
  touch surfaces across reorder proof owners, reorder/graph implementation
  owners, benchmark surfaces, and support surfaces.
- Opened Sprint 86 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 86 starts from the same strongest local reviewed baseline Sprint 85
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 86 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 86 is not a generic “speed up tests” sprint. Its highest value is one
  bounded reviewed-runtime and reordering-scalability package centered on:
  - reviewed runtime audit
  - algorithm / proof runtime design
  - ND runtime reduction
  - proof-surface rebalancing
  - benchmark / comparison follow-through
  - CI / reviewed-path alignment
  - validation and closeout
- The validated Sprint 85 close baseline already fixed the strongest runtime
  contradiction entering Sprint 86:
  - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - reviewed `test_reorder_nd` time = `283.53 sec`
  - the strongest runtime long pole is therefore concentrated on the reorder /
    ND reviewed proof lane, not on a generic whole-suite slowdown
- The strongest likely Sprint 86 implementation, proof, and support surfaces
  are explicit from the live tree:
  - strongest reviewed proof and runtime owner:
    - `tests/test_reorder_nd.c` = `2287`
  - adjacent reorder proof owners:
    - `tests/test_reorder.c` = `1082`
    - `tests/test_reorder_amd_qg.c` = `273`
  - strongest reorder and ND implementation owners:
    - `src/sparse_graph.c` = `841`
    - `src/sparse_reorder_nd.c` = `757`
    - `src/sparse_graph_coarsen.c` = `659`
    - `src/sparse_reorder_amd_qg.c` = `611`
    - `src/sparse_graph_refine.c` = `602`
    - `src/sparse_graph_bisect.c` = `528`
    - `src/sparse_reorder.c` = `419`
    - `src/sparse_graph_separator.c` = `297`
  - strongest measurement and support surfaces:
    - `benchmarks/bench_reorder.c` = `321`
    - `benchmarks/bench_fillin.c` = `178`
    - `README.md` = `1050`
    - `docs/maintainer_guide.md` = `726`
- The strongest Day 1 clarification is now fixed:
  - Sprint 86 should not reopen Sprint 85’s source-decomposition package as
    its first implementation center
  - Sprint 86 should first reduce reviewed runtime concentration and improve
    reorder / ND scalability on the strongest current long pole
  - it should preserve correctness-proof quality while deciding how much of
    the fix is algorithmic, fixture-organization, or reviewed-path
    architecture
- The preserved Sprint 86 non-goal pressure is explicit before Day 2:
  - no generic maintainability decomposition restart
  - no weakening of correctness proof quality to buy runtime wins
  - no benchmark-governance or example-governance drift into correctness
    ownership
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed runtime seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Carried forward the validated Sprint 85 close runtime anchors:
  - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - reviewed `test_reorder_nd` time = `283.53 sec`
- Captured the live reorder, ND, benchmark, and support-surface hotspot map
  from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 86 no longer starts from generic Epic 8 runtime prose.
- The reviewed runtime audit, algorithm/proof design, ND runtime reduction,
  proof-surface rebalancing, benchmark follow-through, CI alignment, and
  validation workstreams are fixed in writing.
- The strongest likely Sprint 86 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Reviewed-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live reviewed
proof-owner split before Sprint 86 changes any reorder, ND, or reviewed-runtime
surface.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the strongest recent validation/proof-surface template from
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day2-validation-baseline-and-proof-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed proof-owner and runtime
  binaries for the early Sprint 86 lanes:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_fillin`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the script-owned support-proof surfaces:
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 86 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial runtime, proof-surface, or reviewed-path batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently owns the strongest early-Sprint-86 proof
  and runtime surfaces:
  - reorder and ND reviewed proof owners:
    - `./build/quality-review-cmake/test_reorder_nd`
    - `./build/quality-review-cmake/test_reorder`
    - `./build/quality-review-cmake/test_reorder_amd_qg`
    - `./build/quality-review-cmake/test_graph`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
  - reviewed benchmark follow-on binaries:
    - `./build/quality-review-cmake/bench_reorder`
    - `./build/quality-review-cmake/bench_fillin`
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
  - reviewed CMake reorder/ND proof-owner tests and representative examples
    remain the main executable truth surfaces for Sprint 86
  - reviewed benchmark binaries remain runtime-side measurability surfaces,
    not the canonical reporting owner
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed reorder/ND proof-owner
  tests, representative examples, and reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, and `tests/test_cmake_install.sh`.

### Day 2 Exit State
- Sprint 86 now has one explicit validation and reviewed-surface contract
  before the runtime long-pole audit begins.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/package proof is fixed in writing.
- The highest-signal rerun set is explicit before the first runtime-cause
  rerank.

## Day 3 - Reviewed Runtime Long-Pole Audit

### Goal
Reduce Sprint 86's broad reviewed-runtime problem to one ranked live cause map
so the sprint can choose one bounded ND/reorder runtime lane instead of
another generic performance bucket.

### Actions
- Re-read the Day 3 runtime-audit expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the strongest recent rerank template from
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day3-hotspot-rerank-audit.md`.
- Re-read the validated Sprint 85 close runtime anchor from
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day13-full-validation-sweep.md`
  and the Sprint 85 handoff from
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day14-closeout-and-handoff.md`.
- Refreshed the live reorder/ND hotspot map from direct `wc -l` measurement:
  - `tests/test_reorder_nd.c` = `2287`
  - `tests/test_reorder.c` = `1082`
  - `tests/test_reorder_amd_qg.c` = `273`
  - `tests/test_graph.c` = `2925`
  - `src/sparse_reorder_nd.c` = `757`
  - `src/sparse_reorder.c` = `419`
  - `src/sparse_reorder_amd_qg.c` = `611`
  - `src/sparse_graph.c` = `841`
  - `src/sparse_graph_bisect.c` = `528`
  - `src/sparse_graph_coarsen.c` = `659`
  - `src/sparse_graph_refine.c` = `602`
  - `src/sparse_graph_separator.c` = `297`
  - `benchmarks/bench_reorder.c` = `321`
  - `benchmarks/bench_fillin.c` = `178`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `726`
- Re-scanned the strongest runtime and proof concentration inside:
  - `tests/test_reorder_nd.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `benchmarks/bench_reorder.c`
- Reconfirmed the carried runtime anchor from the validated Sprint 85 close:
  - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - reviewed `test_reorder_nd` time = `283.53 sec`

### Findings
- Sprint 86's broad runtime problem is now reduced to one ranked live cause
  map:
  - strongest first target:
    - bounded ND runtime reduction centered on `tests/test_reorder_nd.c`,
      `src/sparse_reorder_nd.c`, and the multilevel graph pipeline it drives
  - strongest second target:
    - proof-surface concentration rebalancing across `tests/test_reorder_nd.c`
      and adjacent reorder/graph proof owners where repeated heavy fixture work
      is avoidable without weakening correctness ownership
  - strongest third target:
    - bounded graph-pipeline follow-through in `src/sparse_graph.c`,
      `src/sparse_graph_coarsen.c`, `src/sparse_graph_bisect.c`, and
      `src/sparse_graph_refine.c`
  - strongest fourth target:
    - benchmark/comparison follow-through in `benchmarks/bench_reorder.c` and
      `benchmarks/bench_fillin.c` after a real landed runtime seam exists
  - strongest support-only but real target:
    - maintainer/docs wording only where the landed runtime seam changes proof,
      rerun, or reviewed-path expectations
- The strongest current contradiction is now explicit:
  - the validated Sprint 85 close already fixed the reviewed long pole to
    `test_reorder_nd` at `283.53 sec` out of `404.15 sec`
  - the live tree shows that this is not just a large-test-file problem
  - `tests/test_reorder_nd.c` concentrates many large-fixture and env-policy
    proofs while the underlying algorithmic work is split across
    `src/sparse_reorder_nd.c` and the `src/sparse_graph*.c` pipeline
  - that means the first Sprint 86 move should be one bounded ND runtime lane,
    not generic test trimming or benchmark-driven retuning
- The strongest second-tier contradictions are also clear:
  - proof-surface concentration is real:
    - `tests/test_reorder_nd.c` = `2287`
    - `tests/test_graph.c` = `2925`
    - `tests/test_reorder.c` = `1082`
  - algorithmic/policy concentration is real:
    - `src/sparse_graph.c` = `841`
    - `src/sparse_reorder_nd.c` = `757`
    - `src/sparse_graph_coarsen.c` = `659`
    - `src/sparse_reorder_amd_qg.c` = `611`
    - `src/sparse_graph_refine.c` = `602`
    - `src/sparse_graph_bisect.c` = `528`
  - benchmark surfaces remain informative but secondary:
    - `benchmarks/bench_reorder.c` = `321`
    - `benchmarks/bench_fillin.c` = `178`
- The Sprint 80/Sprint 85 carry-forward reading is now fixed:
  - Sprint 80 already fenced the performance contract so Sprint 86 does not
    need to reopen generic performance governance
  - Sprint 85 already handed Sprint 86 a reviewed-runtime-first queue rather
    than another maintainability-first decomposition sprint
  - the first Sprint 86 landing must preserve correctness ownership while
    reducing reviewed runtime on the ND lane

### Validation
- Re-read the Sprint 85 validated runtime close and handoff artifacts.
- Re-scanned the live reorder, ND, graph-pipeline, benchmark, and support
  hotspot map from direct `wc -l` measurement.
- Re-read the high-signal runtime and proof concentration surfaces in
  `tests/test_reorder_nd.c`, the reorder/graph implementation owners, and the
  reorder benchmark lane.

### Day 3 Exit State
- Sprint 86 now has one ranked live reviewed-runtime contradiction map grounded
  in the current tree and validated Sprint 85 runtime anchors.
- The first implementation center is fixed to one bounded ND runtime reduction
  lane.
- Later proof-surface rebalancing, graph-pipeline follow-through, benchmark
  comparisons, and support-only wording are explicitly ordered behind that
  first lane.

## Day 4 - First Runtime and Scalability Boundary Freeze

### Goal
Fix the first bounded Sprint 86 runtime/scalability implementation fence so the
next design pass can define one real ND runtime contract instead of another
broad optimization rewrite.

### Actions
- Re-read the Day 4 boundary-freeze expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the Sprint 86 project-plan section in
  `docs/planning/EPIC_8/PROJECT_PLAN.md`.
- Re-read the Day 3 reviewed-runtime rerank artifact from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day3-reviewed-runtime-long-pole-audit.md`.
- Reconciled the Day 3 ranking against the Sprint 80 performance-contract
  carry-forward and the Sprint 85 runtime-first handoff.
- Fixed the first implementation fence by separating:
  - required first landing center
  - directly forced support-only proof and graph-path surfaces
  - explicitly deferred proof, benchmark, CI, and support spillover

### Findings
- Sprint 86 now has one explicit first implementation fence:
  - required first landing:
    - `src/sparse_reorder_nd.c`
  - support only if the first landing truly forces it:
    - `src/sparse_graph.c`
    - `src/sparse_graph_coarsen.c`
    - `src/sparse_graph_bisect.c`
    - `src/sparse_graph_refine.c`
    - `src/sparse_graph_separator.c`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `tests/test_reorder.c`
    - `docs/maintainer_guide.md`
    - `README.md`
  - explicitly deferred from the first landing:
    - `src/sparse_reorder.c`
    - `src/sparse_reorder_amd_qg.c`
    - `tests/test_reorder_amd_qg.c`
    - `benchmarks/bench_reorder.c`
    - `benchmarks/bench_fillin.c`
    - proof-surface rebalancing as a first-batch center
    - benchmark/comparison follow-through as a first-batch center
    - CI/reviewed-path alignment as a first-batch center
    - install/package/runtime-surface widening
    - generic maintainability decomposition restart
- The strongest Day 4 clarification is now explicit:
  - the best first Sprint 86 move is one bounded ND orchestration/runtime
    reduction inside `src/sparse_reorder_nd.c`
  - graph-pipeline source movement remains allowed only where that first seam
    truly forces it
  - reorder/graph proof-owner tests stay support-only unless the runtime
    landing changes their contract or requires tightly scoped proof updates
  - benchmark and canonical-reporting surfaces remain outside the first
    implementation center
  - CI/reviewed-path alignment remains later work after a real landed runtime
    seam exists
- The preserved first-batch non-goal fence is fixed now:
  - no weakening of correctness proof quality to buy runtime wins
  - no broad graph/reorder family rewrite detached from the ND lane
  - no generic maintainability decomposition restart
  - no benchmark-governance or example-governance drift into correctness
    ownership
  - no support-surface churn detached from a real landed runtime seam
  - no package/platform maturity claim widening

### Validation
- Re-read the Sprint 86 project-plan section and Day 4 plan expectations.
- Re-read the Day 3 reviewed-runtime rerank artifact.
- Reconciled the fixed first-batch fence against Sprint 80's performance
  contract and Sprint 85's close handoff.

### Day 4 Exit State
- Sprint 86 now has one bounded first runtime/scalability landing center.
- Day 5 can design one ND runtime architecture contract inside that fence.
- Later proof-surface rebalancing, graph-pipeline spillover, benchmark
  comparisons, CI alignment, and broader support movement are held back until
  later lanes.

## Day 5 - Algorithm and Proof Runtime Architecture Design

### Goal
Define the bounded runtime/scalability contract that Sprint 86 will actually
land on the first ND runtime-reduction lane.

### Actions
- Re-read the Day 5 runtime-design expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the Day 4 boundary artifact from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day4-first-runtime-scalability-boundary.md`.
- Re-read the Day 3 reviewed-runtime rerank artifact from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day3-reviewed-runtime-long-pole-audit.md`.
- Re-scanned the ownership seams across:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
- Fixed the first-batch ownership split between:
  - ND runtime-reduction owner
  - retained proof-owner tests
  - graph-pipeline follow-through owners only if forced
  - benchmark/comparison and CI/reviewed-path surfaces explicitly later

### Findings
- Sprint 86 now has one explicit first implementation contract:
  - required implementation center:
    - `src/sparse_reorder_nd.c`
  - support only if the first batch truly forces it:
    - `src/sparse_graph.c`
    - `src/sparse_graph_coarsen.c`
    - `src/sparse_graph_bisect.c`
    - `src/sparse_graph_refine.c`
    - `src/sparse_graph_separator.c`
    - `tests/test_reorder_nd.c`
    - `tests/test_graph.c`
    - `tests/test_reorder.c`
    - `docs/maintainer_guide.md`
    - `README.md`
- The Day 5 ownership split is now fixed:
  - ND runtime-reduction owner:
    - `src/sparse_reorder_nd.c`
  - retained reviewed proof owner after the runtime landing:
    - `tests/test_reorder_nd.c`
  - graph-pipeline follow-through owners only if the runtime seam truly forces
    algorithmic spillover:
    - `src/sparse_graph.c`
    - `src/sparse_graph_coarsen.c`
    - `src/sparse_graph_bisect.c`
    - `src/sparse_graph_refine.c`
    - `src/sparse_graph_separator.c`
  - retained graph-family proof owner only if the first batch truly changes
    graph-path behavior:
    - `tests/test_graph.c`
  - retained public reorder proof owner only if the first batch changes
    top-level reorder behavior outside the ND-focused reviewed lane:
    - `tests/test_reorder.c`
  - benchmark/comparison evidence owners, but not first-batch owners:
    - `benchmarks/bench_reorder.c`
    - `benchmarks/bench_fillin.c`
  - support-surface wording owners only if implementation truly changes the
    maintainer rerun or reviewed-path reading:
    - `docs/maintainer_guide.md`
    - `README.md`
- The strongest Day 5 clarification is explicit now:
  - the first landing should stay runtime-owned inside `src/sparse_reorder_nd.c`
  - it should reduce reviewed runtime concentration by changing one bounded ND
    orchestration/policy seam rather than redistributing work across many new
    owners
  - it should preserve `tests/test_reorder_nd.c` as the primary reviewed proof
    owner instead of turning Day 6 into a proof-surface rebalance batch
  - it should keep graph-pipeline movement support-only unless the touched ND
    seam genuinely exposes one graph-local bottleneck that must move in the
    same batch
  - it should keep benchmarks informative rather than authoritative
  - it should keep CI/reviewed-path alignment explicitly later, after a real
    runtime seam lands
- The preserved first-batch fence is explicit:
  - no weakening of correctness proof quality to buy runtime wins
  - no broad graph/reorder family rewrite detached from the ND lane
  - no proof-surface rebalancing folded into the first batch unless the ND
    runtime seam truly forces it
  - no benchmark/reporting or example drift into correctness ownership
  - no generic maintainability decomposition restart
  - no public docs or package/runtime churn detached from the landed runtime
    seam

### Validation
- Re-read the Day 5 plan expectations and the Day 3/Day 4 Sprint 86 artifacts.
- Re-scanned the live ND/reorder and graph-pipeline ownership seams.
- Reconciled the first-batch ownership split against Sprint 80's performance
  fence and Sprint 85's clearer owner map.

### Day 5 Exit State
- Sprint 86 now has one bounded ND runtime architecture contract.
- Ownership between the first ND runtime lane, retained reviewed proof owner,
  graph-pipeline spillover, and later benchmark/CI follow-through is fixed
  before Day 6 begins.
- Proof-surface rebalancing, benchmark evidence, CI alignment, and broader
  support spillover remain explicitly outside the first batch.

## Day 6 - ND Runtime Reduction Batch

### Goal
Land one bounded ND runtime-reduction batch inside `src/sparse_reorder_nd.c`
that moves the authoritative reviewed-runtime long pole without widening into
proof-surface rebalancing or graph-family rewrite.

### Actions
- Re-read the Day 6 runtime-batch expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the Day 5 runtime-design artifact from
  `docs/planning/EPIC_8/SPRINT_86/artifacts/day5-algorithm-proof-runtime-architecture-design.md`.
- Re-profiled the current ND long pole with focused runtime instrumentation:
  - `SPARSE_ND_PROFILE=1 ./build/test_reorder_nd`
- Measured the current policy seam with bounded threshold sweeps using:
  - `./build/bench_reorder --skip-factor --nd-threshold <n> ...`
- Tried two leaf-glue-oriented `src/sparse_reorder_nd.c` experiments and
  discarded both after validation because they did not improve the
  authoritative reviewed path.
- Landed the kept ND policy flip by raising
  `sparse_reorder_nd_base_threshold` from `128` to `160` and aligned the
  touched local-history comments in:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph.c`
  - `benchmarks/bench_reorder.c`
- Applied the only forced proof-owner follow-through in
  `tests/test_reorder_nd.c`:
  - retained the Pres_Poisson fill gate with the updated current ratio
  - switched the fixed-`k` differentiation fixture from `bcsstk04` to
    `bcsstk14` because `bcsstk04` becomes a pure leaf-AMD case at the new
    default threshold
- Revalidated the code-day gates and the authoritative reviewed path.

### Findings
- Sprint 86's first implementation landing stayed inside the Day 5 fence:
  - required implementation center:
    - `src/sparse_reorder_nd.c`
  - directly forced support follow-through actually needed:
    - `src/sparse_reorder_nd_internal.h`
    - `src/sparse_graph.c`
    - `benchmarks/bench_reorder.c`
    - `tests/test_reorder_nd.c`
  - not needed in the batch:
    - `src/sparse_graph_coarsen.c`
    - `src/sparse_graph_bisect.c`
    - `src/sparse_graph_refine.c`
    - `src/sparse_graph_separator.c`
    - `tests/test_graph.c`
    - `tests/test_reorder.c`
    - `docs/maintainer_guide.md`
    - `README.md`
- The key Day 6 runtime clarification is now explicit:
  - the real current long pole is not leaf-AMD glue
  - `SPARSE_ND_PROFILE=1 ./build/test_reorder_nd` showed the current ND cost
    is dominated by partition work:
    - `partition = 23022.473 ms`
    - `leaf_amd = 155.773 ms`
    - `subgraph = 55.253 ms`
    - `total = 23482.393 ms`
  - the kept win therefore came from the ND orchestration/policy seam
    instead of deeper leaf-side surgery
- The bounded threshold re-sweep fixed the kept landing:
  - headline Pres_Poisson sweep:
    - `t=128`: `nnz(L)=2462201`, `reorder wall=7371.8 ms`
    - `t=160`: `nnz(L)=2474435`, `reorder wall=5015.2 ms`
    - `t=192`: `nnz(L)=2499686`, `reorder wall=4687.5 ms`
  - retained default:
    - `128 -> 160`
  - reason:
    - `160` materially reduces the current reviewed-runtime hotspot while
      preserving the current fill-quality proof contract
    - `192` buys comparatively little extra runtime on Pres_Poisson while
      pushing fill higher there and was left opt-in
- The multi-fixture threshold evidence stayed inside current proof tolerances:
  - `nos4`:
    - unchanged at `nnz(L)=637`
  - `bcsstk04`:
    - `3722 -> 3143`
    - `135.2 ms -> 2.5 ms`
  - `Kuu`:
    - `764664 -> 753755`
    - `5972.7 ms -> 2964.4 ms`
  - `bcsstk14`:
    - `130422 -> 132634`
    - `464.6 ms -> 377.5 ms`
  - `s3rmt3m3`:
    - `487832 -> 484890`
    - `4896.7 ms -> 3423.9 ms`
  - `Pres_Poisson`:
    - `2462201 -> 2474435`
    - `7371.8 ms -> 5015.2 ms`
- The only proof-owner follow-through that the kept runtime seam truly forced
  was inside `tests/test_reorder_nd.c`:
  - the Pres_Poisson ratio commentary now matches the current default path:
    - `0.923 -> 0.927`
  - the fixed-`k` differentiation seam now uses `bcsstk14`, which still
    crosses the partitioner at the new threshold and differentiates clearly:
    - `hybrid=284058`
    - `balance=195336`
    - `degree=267391`
- The authoritative reviewed-path win is now explicit relative to the Sprint
  85 close anchor:
  - reviewed `test_reorder_nd`:
    - `283.53 sec -> 138.68 sec`
  - reviewed CMake total real time:
    - `404.15 sec -> 234.05 sec`
  - Makefile/CMake parity remained exact:
    - `53 vs 53`

### Validation
- Re-ran:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- Reconfirmed reviewed parity:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
- Captured the final authoritative runtime anchors on the kept revision:
  - reviewed `test_reorder_nd` = `138.68 sec`
  - reviewed CMake `Total Test time (real)` = `234.05 sec`

### Day 6 Exit State
- Sprint 86 now has one landed bounded ND runtime-reduction batch.
- The first real win came from the ND threshold/policy seam rather than a
  deeper graph rewrite or proof-surface redistribution.
- The reviewed-runtime long pole moved materially while correctness proof
  quality and reviewed parity stayed intact.

## Day 7 - Post-Landing Runtime Audit and Rerank

### Goal
Re-rank the remaining Sprint 86 contradiction map after the Day 6 ND
runtime-reduction landing so the next batch follows the actual remaining
reviewed-path pressure rather than the original pre-landing ordering.

### Actions
- Re-read the Day 7 rerank expectations from
  `docs/planning/EPIC_8/SPRINT_86/PLAN.md`.
- Re-read the Day 6 landing record from:
  - `docs/planning/EPIC_8/SPRINT_86/WORKING_NOTES.md`
  - `docs/planning/EPIC_8/SPRINT_86/artifacts/day6-nd-runtime-reduction-batch.md`
- Re-read the validated Sprint 85 close runtime anchor and compared it against
  the Day 6 reviewed close:
  - Sprint 85 close:
    - reviewed `test_reorder_nd` = `283.53 sec`
    - reviewed CMake `Total Test time (real)` = `404.15 sec`
  - post-Day-6 reviewed close:
    - reviewed `test_reorder_nd` = `138.68 sec`
    - reviewed CMake `Total Test time (real)` = `234.05 sec`
- Refreshed the live post-Day-6 hotspot map from direct `wc -l` measurement:
  - `tests/test_reorder_nd.c` = `2288`
  - `tests/test_graph.c` = `2925`
  - `tests/test_reorder.c` = `1082`
  - `src/sparse_reorder_nd.c` = `771`
  - `src/sparse_graph.c` = `841`
  - `src/sparse_graph_coarsen.c` = `659`
  - `src/sparse_graph_bisect.c` = `528`
  - `src/sparse_graph_refine.c` = `602`
  - `src/sparse_graph_separator.c` = `297`
  - `benchmarks/bench_reorder.c` = `322`
  - `benchmarks/bench_fillin.c` = `178`
  - `docs/maintainer_guide.md` = `726`
  - `README.md` = `1050`
- Reconciled the post-Day-6 runtime reading against the Sprint 86 queue:
  - proof-surface rebalancing
  - benchmark/comparison follow-through
  - CI/reviewed-path alignment

### Findings
- The Day 6 landing closed the strongest first Sprint 86 contradiction:
  - `src/sparse_reorder_nd.c` no longer stands out as the clear next landing
    center
  - the repo now has one real bounded ND runtime/scalability seam landed
  - a second immediate algorithm-first ND batch is not the highest-value next
    move
- The strongest remaining Sprint 86 seam is now reviewed-surface
  concentration:
  - reviewed `test_reorder_nd` still dominates the reviewed path even after
    the Day 6 win:
    - `138.68 sec` out of `234.05 sec`
    - roughly `59%` of the reviewed CMake total
  - that remaining pressure now reads more like proof concentration than
    unresolved ND threshold policy
- The exact Day 8 design center is now fixed to:
  - `tests/test_reorder_nd.c`
- The strongest support-only follow-through is now:
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- The useful Day 7 clarification is explicit now:
  - no second immediate ND-policy retuning batch as the next center
  - no graph-pipeline rewrite before the proof-owner concentration is designed
  - no early benchmark/comparison batch before the reviewed proof surface is
    rebalanced
  - no CI/reviewed-path wording movement before a real reviewed-surface seam
    lands
- The remaining ordering is now fixed:
  - next seam:
    - proof-surface rebalancing centered on `tests/test_reorder_nd.c`
  - later seam:
    - benchmark/comparison follow-through
  - later seam:
    - CI/reviewed-path alignment
  - still deferred unless newly justified:
    - another algorithmic ND or graph-family runtime landing

### Validation
- This was a docs-only rerank day, so no build/test rerun was required.
- The rerank was grounded in direct rereads of the Day 6 landing records, the
  validated Sprint 85 close baseline, and the live post-Day-6 hotspot map.

### Day 7 Exit State
- Sprint 86 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one proof-surface design lane centered on
  `tests/test_reorder_nd.c`.
- Benchmark/comparison follow-through and CI/reviewed-path alignment remain
  clearly separated from the real next implementation move.
