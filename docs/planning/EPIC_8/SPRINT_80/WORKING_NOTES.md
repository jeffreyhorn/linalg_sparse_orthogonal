# Sprint 80 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 80 baseline for Epic 8 by grounding the sprint in
the validated Epic 7 end state, the live Epic 8 review/todo/project-plan
package, and the current permanent validation, package, benchmark, and policy
surfaces rather than another generic planning reset.

### Actions
- Re-read the Sprint 80 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 80 day-by-day plan in `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Epic 7 close state in
  `docs/planning/EPIC_7/EPIC_7_RETROSPECTIVE.md`.
- Re-read the strongest direct closeout handoff artifact from Epic 7:
  `docs/planning/EPIC_7/SPRINT_79/artifacts/day14-closeout-and-handoff.md`.
- Re-read the Epic 8 opening review package:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`
- Rechecked the maintained reviewed wrapper surface with
  `make -n quality-review-full`.
- Reconfirmed the reviewed CMake parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 80
  touch surfaces across support/policy, package/install/export, workflow, and
  review-package surfaces.
- Opened Sprint 80 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 80 starts from the same strongest local reviewed baseline Epic 7
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 80 work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Sprint 80 is not another broad architecture or feature sprint. Its highest
  value is to make Epic 8 executable from one fresh, truthful, measurable
  starting point.
- The highest-value Sprint 80 workstreams are now fixed explicitly:
  - baseline recheck
  - competitive gap inventory
  - external oracle contract
  - performance / benchmark contract
  - non-goal and risk fence
  - review-package and closeout documentation
- The strongest likely Sprint 80 support, policy, and product-story surfaces
  are explicit from the live tree:
  - `README.md` = `1050`
  - `INSTALL.md` = `265`
  - `docs/maintainer_guide.md` = `698`
  - `benchmarks/README.md` = `393`
  - `Makefile` = `899`
  - `CMakeLists.txt` = `413`
  - `scripts/bench_canonical_report.sh` = `101`
  - `tests/test_install.sh` = `172`
  - `tests/test_cmake_install.sh` = `146`
  - `.github/workflows/ci.yml` = `223`
  - `.github/workflows/macos-ci.yml` = `117`
  - `.github/workflows/windows-ci.yml` = `63`
- The strongest likely Sprint 80 review-package surfaces are also explicit:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md` = `464`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md` = `339`
  - `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`
- The strongest Day 1 clarification is now fixed:
  - Sprint 80 should not start implementation work.
  - It should first lock one current baseline, one contract for external
    comparison, one benchmark/performance reading, and one risk fence for the
    rest of Epic 8.
- The preserved Epic 8 non-goal pressure is explicit before Day 2:
  - no fake state-of-the-art claim inflation
  - no broad subsystem redesign hidden inside baseline work
  - no external dependency sprawl without an explicit contract
  - no benchmark-threshold or platform-parity claim widening detached from
    maintained proof

### Validation
- Rechecked `make -n quality-review-full`.
- Reconfirmed the reviewed parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live support/policy, package/install/export, workflow, and
  review-package hotspot maps from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 80 no longer starts from generic Epic 8 planning prose.
- The baseline, gap-inventory, external-oracle, benchmark-contract, and
  non-goal/risk-fence workstreams are fixed in writing.
- The strongest likely Sprint 80 touch surfaces are explicit before the
  validation-baseline recheck begins.

## Day 2 - Validation Baseline

### Goal
Reconfirm the Sprint 80 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, canonical benchmark/report command surfaces, and install/export proof
owners before any Epic 8 baseline or contract batch lands.

### Actions
- Re-read the Sprint 80 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-materialized the reviewed CMake parity tree with
  `make quality-review-cmake-compile`.
- Reconfirmed the reviewed CMake parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner tests and representative examples
  most likely to matter early in Epic 8:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical report command surface with
  `make -n bench-canonical-report`.
- Reconfirmed the root `build/` canonical benchmark emitters consumed by that
  report path:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`

### Findings
- Sprint 80 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 80 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial baseline, comparison-contract, or integration batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Epic-8 proof
  surfaces:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
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
- The strongest early-Epic-8 proof and truth-surface ownership split is
  already stable:
  - reviewed CMake proof owners and representative examples remain the main
    executable truth surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 80 rerun set is now fixed around the likely touched
  baseline and contract seams:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Ran `make quality-review-cmake-compile`.
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Epic 8.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 80 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the competitive gap inventory
  and external-oracle design work begins.

## Day 3 - Competitive Gap Inventory

### Goal
Reduce Epic 8's broad “state-of-the-art gap” problem to one ranked live
contradiction map grounded in the current tree so later Sprint 80 design work
can choose bounded contracts rather than restating the whole review package.

### Actions
- Re-read the Epic 8 review in
  `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`.
- Re-read the Epic 8 closure sequence in
  `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`.
- Re-read the Sprint 80 Day 1 and Day 2 working-notes baseline.
- Rechecked the strongest current hotspot and product surfaces through the live
  tree and refreshed raw `wc -l` measurements for the main first-tier
  candidates:
  - `src/sparse_matrix.c`
  - `src/sparse_dense.c`
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
  - `README.md`
  - `benchmarks/README.md`
  - `CMakeLists.txt`
  - `Makefile`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
- Reconciled the live hotspot map against the Epic 7 residual journal and the
  Epic 8 review findings so previously deferred seams were not inherited
  blindly.

### Findings
- Epic 8's broad improvement problem is now reduced to one ranked live
  contradiction map instead of one generic modernization bucket:
  - strongest first target:
    - linked-list-first product/storage ceiling
  - strongest second target:
    - builtin dense/backend performance ceiling
  - strongest third target:
    - bounded capability surface
  - strongest fourth target:
    - external differential/numerical-assurance ceiling
  - strongest fifth target:
    - maintainability concentration in large source and giant-test hotspots
  - strongest sixth target:
    - reviewed runtime concentration around reordering/ND proof
  - strongest seventh target:
    - static-first and asymmetric package/platform maturity
  - strongest support-only but real target:
    - front-door usability and policy density
- The strongest current contradiction center is still the public/core
  storage-workflow model, not package wording or one more hotspot split:
  - `README.md` still opens by identifying the library as an orthogonal
    linked-list sparse matrix library
  - `include/sparse_matrix.h` still keeps the mutable linked-list shell as the
    public conceptual center
  - `src/sparse_matrix.c` still centers mutation around pointer-heavy row/col
    walk and slab-node allocation
- The dense/backend lane remains the strongest second contradiction rather than
  the first:
  - `src/sparse_dense.c` is still the clearest hard performance ceiling
  - the backend-aware architecture from Epic 7 is real, but it still terminates
    in a builtin scalar dense-kernel surface
  - that means the repo now has backend-aware seams without backend maturity
- The capability surface remains the strongest third contradiction rather than
  a support-only concern:
  - `include/sparse_types.h` still fixes the public scalar contract to
    real-only and the index-width story to compile-time selection
  - the eigensolver and iterative breadth remains intentionally narrower than a
    best-in-class scientific sparse stack
- External differential proof is stronger than it was before Epic 7, but it is
  still the strongest assurance-side contradiction:
  - the repo has strong internal proof ownership
  - it still lacks a maintained external numerical-oracle lane strong enough to
    support a serious competitive claim
- The strongest maintainability concentration is real but now clearly reads as
  the fifth target rather than the first:
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_chol_csc.c` = `1564`
  - `src/sparse_qr.c` = `1563`
  - `tests/test_chol_csc.c` = `4724`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_integration.c` = `2689`
- The reviewed runtime concentration remains explicit but lower-order than the
  product, backend, capability, and oracle ceilings:
  - `tests/test_reorder_nd.c` = `2287`
  - the long-pole problem is operationally real, but it is not the first Epic
    8 execution center
- Package/platform asymmetry remains real but is still later than the four
  structural ceiling lanes above it:
  - `CMakeLists.txt` still keeps the maintained package surface static-only
  - Windows and macOS workflow evidence remain intentionally asymmetric
- The strongest Day 3 clarification is now explicit:
  - Sprint 81 should begin with the product/storage ceiling, not with a broad
    package or docs campaign
  - the dense/backend lane should stay second, not be collapsed into the first
    storage batch
  - capability expansion should stay third, after the first two structural
    ceilings move
  - maintainability, reviewed runtime, and platform/package convergence remain
    important, but they are not the first contradiction center

### Validation
- Re-read the Epic 8 review and gap-closure todo directly.
- Reconciled the Day 3 ranking against the Sprint 80 Day 1 and Day 2 baseline.
- Refreshed the live raw `wc -l` hotspot map for the main first-tier source,
  proof, package, and workflow candidates.

### Day 3 Exit State
- Epic 8 no longer reads as one generic “modernize everything” queue.
- The live contradiction ranking is explicit enough to drive the later
  external-oracle and performance-contract design work.
- The first implementation center is fixed to the product/storage ceiling
  unless a later Sprint 80 audit finds a higher-value contradiction.

## Day 4 - External Oracle Candidate Audit

### Goal
Identify the strongest realistic external correctness and performance reference
classes for Epic 8 and reduce the broad “compare against the ecosystem”
pressure to one ranked maintained-vs-advisory candidate map.

### Actions
- Re-read the Sprint 80 Day 4 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Day 3 contradiction ranking so the candidate audit stayed
  subordinate to the first Epic 8 execution order.
- Re-scanned the live tree for existing references to external corpora or
  libraries across:
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `include/`
  - `src/`
  - `tests/`
  - `benchmarks/`
  - `CMakeLists.txt`
  - `Makefile`
  - `.github/workflows/`
- Reconfirmed that the tree already uses real SuiteSparse fixture data heavily
  but does not currently maintain external solver-library linkage or CI-backed
  comparison proof.
- Rechecked the shipped fixture inventory in `tests/data/` and
  `tests/data/suitesparse/` so the external-oracle conversation stayed grounded
  in the actual corpus already available locally.

### Findings
- The live tree already has strong corpus realism but not strong maintained
  external solver-oracle proof:
  - SuiteSparse matrices are used widely across tests, examples, and
    benchmarks
  - the tree does not currently maintain CHOLMOD, UMFPACK, SuperLU, MUMPS,
    MKL, OpenBLAS, or LAPACK linkage as a reviewed proof surface
- That means the strongest Day 4 distinction is now explicit:
  - matrix corpus realism is already present
  - external numerical-oracle realism is the missing piece
- The strongest realistic external candidate classes are now ranked as follows:
  - strongest maintained correctness candidate:
    - SuiteSparse-family direct references, especially CHOLMOD-class SPD
      Cholesky comparison and a narrower unsymmetric/direct counterpart where
      packaging burden stays tolerable
  - strongest maintained performance-reference candidate:
    - BLAS/LAPACK-class dense-kernel references for bounded backend
      calibration, not for broad product proof
  - strongest advisory but likely not first maintained candidate:
    - METIS-class nested-dissection / graph-quality comparison
  - strongest exploratory-only candidates:
    - broad external sparse solver families such as SuperLU, MUMPS, PARDISO,
      or wider Eigen-style comparison layers
- Candidate suitability now reads as:
  - CHOLMOD-class SPD differential proof:
    - high correctness value
    - good alignment to the library's strongest CSC Cholesky lane
    - moderate packaging burden
    - realistic first maintained candidate if kept bounded
  - bounded SuiteSparse-family unsymmetric/direct comparison:
    - meaningful value
    - higher packaging and API-shape burden than SPD Cholesky
    - likely second maintained candidate, not the first
  - BLAS/LAPACK dense-kernel calibration:
    - high performance-reference value
    - useful for backend calibration
    - should be read as advisory/performance support rather than product-wide
      correctness proof
  - METIS-style graph/reordering comparison:
    - real algorithmic interest
    - weaker first-sprint payoff than direct-solver and dense-kernel
      references
    - better as advisory or later support context
  - wide external solver matrix:
    - too much dependency and platform burden for the first maintained Epic 8
      contract
- The strongest Day 4 clarification is now explicit:
  - Epic 8 should not try to compare against every major sparse package
  - it should begin with one bounded maintained direct-solver correctness lane
    and one bounded dense-kernel performance-reference lane
  - broader ecosystem comparison belongs in advisory or later-stage context,
    not in the first maintained contract

### Validation
- Re-scanned the live tree for external-library and external-corpus references.
- Rechecked the shipped `tests/data/suitesparse/` fixture inventory directly.
- Reconciled the candidate ranking against the Day 3 contradiction order so the
  external-oracle lane remained fourth in the epic ordering rather than
  accidentally becoming first.

### Day 4 Exit State
- The external-oracle conversation is now reduced to one ranked maintained vs
  advisory candidate map.
- The first realistic maintained candidate reads as bounded SuiteSparse-family
  direct-solver correctness comparison, with BLAS/LAPACK-class dense-kernel
  calibration as the strongest performance-reference support lane.
- Day 5 can now freeze the external-oracle contract from a real candidate set
  instead of from generic ecosystem aspirations.

## Day 5 - External Oracle Contract

### Goal
Freeze the bounded Epic 8 external-oracle contract from the ranked Day 4
candidate set so later sprints know which comparison lanes are maintained,
which are advisory, and which are explicitly outside the first contract.

### Actions
- Re-read the Sprint 80 Day 5 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Day 4 candidate audit and the Stage 1 external-reference section
  in `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`.
- Rechecked the Day 3 contradiction ranking so the contract stayed subordinate
  to the Epic 8 execution order:
  - product/storage first
  - dense/backend second
  - capability third
  - external differential/oracle fourth
- Separated the candidate set into:
  - maintained correctness references
  - maintained performance-reference support
  - advisory comparison context
  - explicitly non-maintained / non-contract surfaces
- Fixed the preserved external-comparison non-goal fence so the contract does
  not widen into dependency sprawl or fake portability claims.

### Findings
- The first maintained Epic 8 external-oracle contract is now fixed explicitly:
  - maintained correctness reference lane:
    - bounded SuiteSparse-family direct-solver comparison centered first on the
      CHOLMOD-class SPD Cholesky lane
  - maintained performance-reference support lane:
    - BLAS/LAPACK-class dense-kernel calibration for backend-aware performance
      work
- The contract intentionally does **not** promote those two lanes to the same
  kind of proof:
  - CHOLMOD-class comparison is the strongest first maintained correctness
    candidate
  - BLAS/LAPACK-class comparison is performance-reference support, not broad
    product correctness proof
- The strongest second-tier but not first-contract maintained candidate is now
  explicit:
  - a narrower unsymmetric/direct external comparison may become valid later,
    but it is not part of the first maintained Sprint 80 contract
- The strongest advisory comparison context is now explicit:
  - METIS-class graph/reordering comparison remains useful advisory context
  - it is not part of the first maintained external-oracle contract
- The strongest explicitly non-contract candidates are also fixed:
  - broad external sparse solver families such as SuperLU, MUMPS, PARDISO, or
    wide comparison-layer ecosystems remain exploratory only
- The preserved external-comparison non-goal fence is now fixed directly:
  - no mandatory heavyweight external stack for normal builds
  - no fake cross-platform proof parity from locally convenient dependencies
  - no “compare to everything” benchmark theater
  - no broad correctness claim inflation from performance-reference lanes
  - no broad dependency matrix that outruns the maintained CI/package surface
- The strongest Day 5 clarification is now explicit:
  - Epic 8 should begin with one bounded maintained direct-solver correctness
    comparison lane and one bounded dense-kernel calibration lane
  - broader ecosystem comparison belongs later, and only if earlier structural
    ceilings move successfully

### Validation
- Re-read the Day 4 ranked candidate audit directly.
- Re-read the Epic 8 todo's external-reference stage directly.
- Reconciled the contract against the Day 3 execution order and the preserved
  non-goal fence.

### Day 5 Exit State
- Epic 8 now has one explicit external-oracle contract instead of one generic
  ecosystem-comparison aspiration.
- Maintained, advisory, and non-contract external lanes are fixed in writing.
- Later Sprint 80 work can now build the benchmark/performance contract and the
  risk fence against one stable comparison reading.

## Day 6 - Performance and Benchmark Contract

### Goal
Freeze the benchmark and performance-governance interpretation Epic 8 backend
and runtime work must preserve so later comparison or acceleration work does
not widen into fake timing gates or broader product claims.

### Actions
- Re-read the Sprint 80 Day 6 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Day 5 external-oracle contract so the benchmark contract stayed
  aligned with the maintained vs advisory comparison split.
- Re-read the strongest benchmark-governance surfaces directly:
  - `benchmarks/README.md`
  - `README.md`
  - `Makefile`
  - `scripts/bench_canonical_report.sh`
- Rechecked the current maintained benchmark taxonomy and command surface:
  - canonical maintained benchmark face
  - threshold-free canonical reporting bundle
  - bounded runtime lane
  - narrow thresholded regression gate
  - exploratory/context-only benchmark lanes
- Reconciled the benchmark-reading contract against the Day 3 contradiction
  order so performance governance stayed a supporting contract rather than the
  first execution center.

### Findings
- The benchmark/performance contract Epic 8 must preserve is now fixed
  explicitly:
  - canonical maintained benchmark face:
    - `bench_refactor_csc`
    - `bench_chol_csc`
    - `bench_iterative_reuse`
    - `bench_eigs_reuse`
  - threshold-free canonical reporting surface:
    - `make bench-canonical-report`
    - `scripts/bench_canonical_report.sh`
  - bounded runtime lane:
    - `bench-fast`
  - narrow thresholded regression gate:
    - `wall-check`
  - exploratory/context-only lanes:
    - broader bench surfaces outside the compact maintained face
- The strongest current benchmark-governance reading is now explicit:
  - canonical report bundles are artifact-friendly longitudinal snapshots
  - they are intentionally not pass/fail timing gates
  - benchmark binaries own row semantics and path measurability
  - tests remain the owners of regression/oracle/property truth
- The Day 5 external-comparison contract composes with the benchmark contract
  as follows:
  - maintained CHOLMOD-class correctness comparison belongs in test or
    differential-proof surfaces first, not in canonical benchmark pass/fail
    interpretation
  - BLAS/LAPACK-class dense-kernel calibration belongs as backend-aware
    performance-reference support and may widen benchmark fields or advisory
    comparison output, but it does not change the threshold-free reading of the
    canonical bundle
- The strongest preserved benchmark non-goal fence is now fixed:
  - no portable timing verdicts from single-run benchmark numbers
  - no historical-diff pass/fail gate hidden inside canonical reporting
  - no broad benchmark-surface widening before the structural ceilings move
  - no confusion between benchmark proof and test-owned correctness proof
- The strongest Day 6 clarification is now explicit:
  - Epic 8 backend and runtime work may strengthen benchmark measurability
  - it must do so without turning the canonical benchmark/report surface into a
    noisy timing gate or broader product-claim surface

### Validation
- Re-read `benchmarks/README.md`, `README.md`, `Makefile`, and
  `scripts/bench_canonical_report.sh` directly.
- Reconciled the benchmark contract against the Day 5 external-oracle
  contract.
- Rechecked that the current command surfaces still present canonical reporting
  as threshold-free and `bench-fast` / `wall-check` as distinct bounded lanes.

### Day 6 Exit State
- Epic 8 now has one explicit benchmark/performance contract.
- Backend-aware comparison work can now move later without reopening benchmark
  governance every time.
- The canonical reporting surface, runtime lane, and narrow threshold gate are
  fixed in writing before the later risk-fence work begins.

## Day 7 - Non-goal and Risk Fence

### Goal
Fix the strongest Epic 8 non-goals and execution risks in writing so later
implementation sprints can work against one stable claim fence instead of
reopening what the epic is allowed to promise.

### Actions
- Re-read the Sprint 80 Day 7 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Day 3 competitive gap inventory, Day 5 external-oracle
  contract, and Day 6 benchmark contract so the fence stayed anchored to the
  live contradiction order and the maintained comparison/governance split.
- Re-read the strongest current support and policy surfaces directly:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `CMakeLists.txt`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-ranked the main Epic 8 execution risks surfaced by Days 1-6:
  - over-broad architecture churn
  - fake state-of-the-art claim inflation
  - external dependency sprawl
  - platform-claim inflation
  - benchmark-governance drift
- Separated the live claim space into:
  - approved Epic 8 target claims
  - explicitly deferred claims
  - explicitly prohibited marketing-style interpretations

### Findings
- Epic 8 now has one explicit approved-claim set:
  - it may try to move from a linked-list-first public product toward a more
    compressed-first workflow without erasing the bounded mutable shell
  - it may raise the dense/backend ceiling while keeping the builtin fallback
  - it may narrow selected capability ceilings through bounded, proof-backed
    widening
  - it may add one bounded maintained external differential lane and stronger
    benchmark measurability without turning canonical reporting into a timing
    gate
  - it may improve maintainability and package/platform confidence without
    overstating current reviewed parity
- The strongest deferred claims are now fixed explicitly:
  - broad external sparse-solver comparison
  - unsymmetric direct external-oracle work
  - broad graph/reordering comparison beyond advisory context
  - broad scalar-family or complex-support claims
  - broad platform/install parity claims
  - shared-library or dynamic-ABI maturity claims
  - portable benchmark-threshold claims
  - whole-repo storage-model replacement in one batch
- The strongest prohibited interpretations are now fixed directly:
  - Epic 8 is not a generic rewrite of the whole library
  - Epic 8 is not automatic state-of-the-art parity with mature sparse
    packages
  - Epic 8 does not prove broad backend maturity before real acceleration
    lands
  - Epic 8 does not prove broad cross-platform install/export parity before
    maintained reviewed evidence exists
  - Epic 8 does not prove shared-library or dynamic-ABI maturity while the
    package surface remains static-first
  - Epic 8 does not convert canonical benchmark reports into pass/fail timing
    gates
- The strongest current execution risks are now ranked with mitigation
  ordering:
  - first:
    - over-broad architecture churn
  - second:
    - fake state-of-the-art claim inflation
  - third:
    - external dependency sprawl
  - fourth:
    - platform-claim inflation
  - fifth:
    - benchmark-governance drift
- The strongest Day 7 clarification is now explicit:
  - Epic 8 should spend implementation budget against the structural ceilings
    first
  - later package/platform, benchmark, and comparison improvements must stay
    inside one already-fixed truthfulness fence rather than widening the
    product claim surface opportunistically

### Validation
- Re-read the Day 3, Day 5, and Day 6 Sprint 80 artifacts directly.
- Re-read the strongest current support/policy/workflow/package surfaces
  directly.
- Reconciled the Day 7 fence against the Epic 8 review verdict and the
  gap-closure todo ordering.

### Day 7 Exit State
- Sprint 80 now has one explicit Epic 8 non-goal and risk fence.
- Approved, deferred, and prohibited claims are separated in writing.
- Later Sprint 80 review/todo refinement can build against one stable
  execution fence instead of reopening the target state.

## Day 8 - Review Package Refinement

### Goal
Strengthen the top-level Epic 8 review so it now reads from the refreshed
Sprint 80 baseline and contract surfaces rather than only from the opening
post-Epic-7 review pass.

### Actions
- Re-read the Sprint 80 Day 8 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Epic 8 review against the landed Sprint 80 Day 2-7 artifacts:
  - validation baseline
  - competitive gap inventory
  - external-oracle candidate audit
  - external-oracle contract
  - benchmark/performance contract
  - non-goal and risk fence
- Re-scanned the review for sections that still needed stronger grounding in:
  - the refreshed reviewed baseline
  - the bounded external-oracle contract
  - the benchmark/performance reading
  - the explicit Epic 8 claim fence
- Tightened only the sections that materially benefited from the new Sprint 80
  contract package instead of rewriting the whole review.

### Findings
- The Epic 8 review still remains the authoritative finding set for the epic:
  - strong engineering rigor
  - not yet state of the art
  - storage/product ceiling first
  - dense/backend ceiling second
  - capability ceiling third
- The review now reads from the refreshed Sprint 80 baseline more directly:
  - `make quality-review-full` remains the strongest local reviewed baseline
  - reviewed CMake parity remains explicit at `53`
- The review now reads from the external-oracle contract more directly:
  - bounded CHOLMOD-class SPD Cholesky comparison is the first maintained
    external correctness target
  - BLAS/LAPACK-class references remain performance-reference support, not a
    broad maintained correctness contract
- The review now reads from the benchmark/performance contract more directly:
  - canonical reporting remains threshold-free
  - `bench-fast` remains the bounded runtime lane
  - `wall-check` remains the narrow thresholded regression gate
- The review now reads from the non-goal fence more directly:
  - Epic 8 is not a generic whole-library rewrite
  - Epic 8 is not allowed to imply fake platform parity, shared-library
    maturity, or broad capability genericity before those surfaces really move
- The strongest Day 8 clarification is now explicit:
  - Sprint 80 did not change the Epic 8 verdict
  - it made the review’s execution fence and interpretation contract more
    explicit

### Validation
- Re-read the Epic 8 review directly against the Day 2-7 Sprint 80 artifact
  set.
- Rechecked that the review findings still support the current Epic 8 sprint
  ordering.
- Rechecked that no review section now contradicts the external-oracle,
  benchmark, or non-goal contracts fixed earlier in Sprint 80.

### Day 8 Exit State
- The Epic 8 review still stands as the authoritative finding set.
- Its baseline, comparison, benchmark, and non-goal reading are now tightened
  to match the landed Sprint 80 contract package.
- Day 9 can refine the closure-sequence todo against one stable review surface.

## Day 9 - Todo and Closure Sequence Refinement

### Goal
Tighten the Epic 8 gap-closure todo so it now reads from the refreshed Sprint
80 baseline and contract package without quietly reopening any of the Day 7
non-goals.

### Actions
- Re-read the Sprint 80 Day 9 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Epic 8 todo against the landed Sprint 80 Day 3-8 findings:
  - competitive gap inventory
  - external-oracle candidate audit
  - external-oracle contract
  - benchmark/performance contract
  - non-goal and risk fence
  - review refinement
- Re-scanned the todo for wording that still read too broad around:
  - external comparison
  - benchmark/governance interpretation
  - cross-platform proof growth
  - shared-library or packaging assumptions
  - final comparison/calibration scope
- Tightened only the sections that materially benefited from the new Sprint 80
  contract package instead of resequencing the whole todo.

### Findings
- The Epic 8 todo still closes the major review findings in the right order:
  - baseline and contract first
  - storage/workflow ceiling second
  - dense/backend ceiling third
  - capability breadth fourth
  - assurance expansion fifth
  - maintainability cleanup sixth
  - runtime concentration seventh
  - packaging/platform convergence eighth
  - usability simplification ninth
  - final comparison and closeout tenth
- The todo now reads from the external-oracle contract more directly:
  - bounded CHOLMOD-class SPD direct-solver comparison first
  - BLAS/LAPACK-class references as performance-reference support
- The todo now reads from the benchmark contract more directly:
  - canonical reporting remains threshold-free
  - benchmark measurability work must not silently reopen portable timing-gate
    interpretations
- The todo now reads from the non-goal fence more directly:
  - cross-platform proof growth stays bounded and evidence-led
  - shared-library work remains optional and proof-backed rather than assumed
  - final external comparison remains bounded rather than “compare against
    everything” theater
- The strongest Day 9 clarification is now explicit:
  - Sprint 80 did not change the major Epic 8 execution order
  - it removed a few remaining broad readings that could have reopened
    policy-contradicting work later

### Validation
- Re-read the Epic 8 todo directly against the Day 3-8 Sprint 80 artifact set.
- Rechecked that the closure sequence still matches the current Epic 8 review
  verdict and sprint ordering.
- Rechecked that no todo section now quietly reopens disallowed benchmark,
  platform, package, or comparison claims.

### Day 9 Exit State
- The Epic 8 todo still stands as the closure-sequence owner for the review.
- Its external, benchmark, platform, and final-comparison wording are now
  tightened to match the landed Sprint 80 contract package.
- Day 10 can now recheck the 10-sprint project plan against one stable review
  and todo pair.

## Day 10 - Project-Plan Contract Recheck

### Goal
Reconcile the 10-sprint Epic 8 project plan against the landed Sprint 80
baseline and contract package so later implementation sprints do not inherit an
ordering or dependency mistake.

### Actions
- Re-read the Sprint 80 Day 10 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read `docs/planning/EPIC_8/PROJECT_PLAN.md` against the landed Sprint 80
  Day 2-9 package:
  - validation baseline
  - competitive gap inventory
  - external-oracle contract
  - benchmark/performance contract
  - non-goal and risk fence
  - review refinement
  - todo refinement
- Rechecked the major sprint ordering and dependency chain across Sprints 80-89
  for contradictions around:
  - storage/workflow sequencing
  - backend timing
  - capability timing
  - assurance timing
  - package/platform timing
  - final comparison scope
- Re-scanned the later-sprint wording for any assumptions that now overreached
  the Sprint 80 claim fence.

### Findings
- No `PROJECT_PLAN.md` edit is actually needed.
- The current 10-sprint Epic 8 plan still matches the landed Sprint 80
  findings:
  - Sprint 80 freezes the baseline and contracts first
  - Sprint 81 still correctly opens implementation with the
    storage/workflow ceiling
  - Sprint 82 still correctly follows with the dense/backend ceiling
  - Sprint 83 still correctly treats capability expansion as downstream of the
    first structural and backend moves
  - Sprint 84 still correctly keeps assurance expansion bounded by the
    external-oracle contract
  - Sprint 85 and Sprint 86 still correctly place maintainability and runtime
    work after the earlier product/backend/assurance moves
  - Sprint 87 still correctly treats packaging/platform work as a later
    convergence lane, not an opening implementation center
  - Sprint 88 still correctly keeps usability simplification late, after the
    product/package truth is clearer
  - Sprint 89 still correctly treats final comparison as bounded
    evidence-based calibration rather than a pre-earned claim
- The strongest Day 10 clarification is now explicit:
  - Sprint 80 refined how the Epic 8 plan should be read
  - it did not reveal an ordering flaw or dependency contradiction that
    requires rewriting the plan itself

### Validation
- Re-read `docs/planning/EPIC_8/PROJECT_PLAN.md` directly against the Sprint
  80 review/todo/contract package.
- Rechecked that no sprint now depends on fake platform parity, assumed shared
  backend maturity, broad external dependency sprawl, or whole-library rewrite
  pressure.
- Rechecked that the time-bounded 10-sprint ordering still tracks the current
  ranked contradiction map.

### Day 10 Exit State
- The Epic 8 project plan still stands without revision.
- Sprint 80 has now fixed the interpretation contract around the plan as well
  as around the review and todo.
- Day 11 can proceed to the support-surface truth sweep without any project
  plan correction.

## Day 11 - Support-Surface Truth Sweep

### Goal
Check whether the main support and policy surfaces already read truthfully
against the landed Sprint 80 baseline and contract package so the sprint does
not drift into unnecessary docs churn.

### Actions
- Re-read the Sprint 80 Day 11 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the highest-signal support and policy surfaces against the landed
  Sprint 80 Day 2-10 package:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
- Rechecked whether any contradiction had appeared around:
  - linked-list-first product wording
  - static-first packaging wording
  - reviewed-platform asymmetry wording
  - canonical benchmark/report interpretation
  - future shared-library or capability-genericity overclaim risk
- Captured the touched-surface raw `wc -l` map for the rechecked support set.

### Findings
- No bounded support-surface correction is actually needed.
- The current support surfaces already reconcile cleanly with the Sprint 80
  contract package:
  - `README.md` still keeps the product, benchmark, and platform claims
    bounded
  - `INSTALL.md` still keeps the static-first install/export contract and the
    narrower reviewed-platform confidence split explicit
  - `docs/maintainer_guide.md` still remains the authoritative packaging,
    benchmark, and platform policy owner
  - `benchmarks/README.md` still keeps canonical reporting threshold-free and
    separated from timing-gate claims
  - `.github/workflows/windows-ci.yml` still keeps Windows as the reviewed
    CMake-consumer subset only
  - `.github/workflows/macos-ci.yml` still keeps macOS install verification as
    supplemental confidence rather than reviewed install/export parity
- The strongest Day 11 clarification is now explicit:
  - Sprint 80 changed the interpretation contract around these surfaces
  - it did not create a new contradiction that justifies support-surface churn
- Final touched-surface counts from the recheck are:
  - `README.md` = `1050`
  - `INSTALL.md` = `265`
  - `docs/maintainer_guide.md` = `698`
  - `benchmarks/README.md` = `393`
  - `.github/workflows/windows-ci.yml` = `63`
  - `.github/workflows/macos-ci.yml` = `117`

### Validation
- Re-read the main support and workflow surfaces directly.
- Rechecked that those surfaces still match the Day 5 external-oracle
  contract, Day 6 benchmark contract, Day 7 non-goal fence, and Day 10 plan
  interpretation.
- Reconfirmed that no support surface now widens the claim fence beyond the
  maintained proof.

### Day 11 Exit State
- Sprint 80 now has an explicit no-op truth-sweep record.
- The main support surfaces remain truthful without edits.
- Day 12 can move on to final proof-owner alignment and the validation queue.

## Day 12 - Final Proof Alignment and Validation Queue

### Goal
Fix the exact proof-owner map and authoritative Day 13 validation queue for the
Sprint 80 baseline-and-contract package so the sprint closes from one explicit
measured rerun set.

### Actions
- Re-read the Sprint 80 Day 12 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the highest-signal reviewed proof-owner binaries and
  representative examples most relevant to Sprint 80:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Reconfirmed the command-owned canonical reporting lane and the maintained
  root `build/` emitters:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- Reconfirmed the maintained install/export proof owners:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Findings
- The strongest reviewed parity anchor remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest Sprint 80 proof split is still fixed cleanly:
  - reviewed CMake tests and representative examples own executable
    regression/lifecycle truth
  - canonical benchmark reporting remains command- and script-owned
  - install/export proof remains script-owned
- No extra focused regression or support-surface follow-through is required
  before Day 13.
- The authoritative Day 13 validation queue is now fixed explicitly around:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 12 clarification is now explicit:
  - Sprint 80’s close validation should not rely only on the inherited Epic 7
    baseline
  - it should rerun the refreshed reviewed, benchmark/report, and install
    proof surfaces directly

### Validation
- Re-ran `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner binaries and representative
  examples.
- Rechecked the canonical report command surface, root `build/` benchmark
  emitters, and the maintained install/export proof scripts.

### Day 12 Exit State
- Sprint 80 now has one explicit final proof-owner map.
- The Day 13 validation queue is fixed in writing.
- Day 13 can execute from one stable rerun set without ambiguity.

## Day 13 - Full Validation Sweep

### Goal
Execute the full Sprint 80 validation queue and retain one explicit measured
baseline for the Epic 8 review-and-contract package.

### Actions
- Ran the full local code-day validation baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- Reconfirmed reviewed CMake parity directly:
  - `ctest -N --test-dir build/quality-review-cmake`
- Re-ran the highest-signal reviewed proof-owner binaries and representative
  examples before the install/export scripts reset the root build tree:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
- Re-ran the representative examples and retained their established residual
  anchors.
- Re-ran the maintained canonical reporting command:
  - `make bench-canonical-report`
- Re-ran the maintained install/export proof:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Findings
- The full validation queue completed cleanly from the final rerun state:
  - `make format` passed
  - `make lint` passed
  - `make test` passed
  - `make quality-review-full` passed
- The maintained reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 642.39 sec`
- The focused Sprint 80 follow-ons all passed:
  - `test_chol_csc` -> `147 / 147`
  - `test_ldlt_csc` -> `96 / 96`
  - `test_ldlt` -> `84 / 84`
  - `test_iterative` -> `80 / 80`
  - `test_qr` -> `72 / 72`
  - `test_integration` -> `51 / 51`
  - `test_reorder_nd` -> `35 / 35`
  - `test_fuzz` -> `26 / 26`
  - `test_eigs` -> `31 / 31`
  - `tests/test_install.sh` -> `11 / 11`
  - `tests/test_cmake_install.sh` -> `13 / 13`
- The maintained canonical report bundle also passed on the final rerun and
  wrote:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`
- Representative retained outputs stayed clean:
  - `test_fuzz` retained `large-n LDLT CSC lifecycle property: 3/3 passed`
  - `test_chol_csc` retained `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
  - `test_reorder_nd` retained `Pres_Poisson ND/AMD = 0.923`
  - `test_reorder_nd` retained `bcsstk14 ND/AMD = 1.124`
  - `example_analysis` residual stayed `4.44e-16`
  - `example_basic_solve` residual stayed `0.00e+00`
  - canonical `bench_refactor_csc nos4` retained `speedup_refactor = 1.71`
  - canonical `bench_chol_csc nos4` retained
    `csc_supernodal_panel_solver = batched_panel`
  - canonical `bench_iterative_reuse` retained `cg 1.00x`, `gmres 1.05x`,
    `minres 1.06x`
  - canonical `bench_eigs_reuse` retained `growm 1.00x`,
    `thick_restart 1.07x`, `lobpcg 1.01x`
  - installed `pkg-config` version remained `2.2.0`
- One transient note is recorded explicitly:
  - the first standalone `make bench-canonical-report` rerun hit a
    non-reproducing missing-output error at the `bench_eigs_reuse.csv` write
    step
  - an immediate clean rerun from the same tree succeeded without source
    edits, so the final Day 13 close state is the successful rerun rather than
    the transient intermediate failure
- One non-blocking runtime note also remains explicit:
  - reviewed CMake `test_reorder_nd` still dominated runtime at `486.38 sec`
    out of the `642.39 sec` total

### Validation
- Completed the full Day 13 validation queue.
- Reconfirmed the reviewed parity anchor and Makefile/CMake parity.
- Reconfirmed the focused proof-owner reruns, canonical report bundle, and
  install/export proof scripts.

### Day 13 Exit State
- Sprint 80 now has one explicit full-validation baseline for the Epic 8
  baseline-and-contract package.
- The review, todo, and project-plan surfaces remain supported by a fresh
  measured rerun set.
- Day 14 can close from this validated baseline.

## Day 14 - Closeout and Handoff

### Goal
Close Sprint 80 from the validated Day 13 baseline and leave one explicit
handoff queue for Sprint 81 and the later Epic 8 implementation sprints.

### Actions
- Re-read the Sprint 80 project-plan section and the Sprint 80 Day 14 plan
  expectations in `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the landed Sprint 80 artifacts, working-notes findings, and the Day
  13 validation baseline.
- Rechecked `docs/planning/EPIC_8/PROJECT_PLAN.md` against the final Sprint 80
  package to confirm whether any project-plan correction was required.
- Wrote the Day 14 closeout/handoff artifact and finalized the closeout state
  in working notes.

### Findings
- Sprint 80 now closes as one coherent Epic 8 baseline-and-contract package
  across:
  - refreshed reviewed and install/export baseline
  - ranked live competitive gap inventory
  - bounded external-oracle contract
  - bounded benchmark/performance contract
  - explicit non-goal and risk fence
  - review/todo/project-plan reconciliation
  - validated Day 13 close baseline
- The preserved fence stayed intact:
  - no implementation sprint started early under ambiguous storage or backend
    assumptions
  - no fake state-of-the-art, platform-parity, or shared-library maturity
    claim was introduced
  - no canonical benchmark reporting surface was turned into a timing gate
  - no broad external dependency matrix was smuggled into the maintained
    contract
- `docs/planning/EPIC_8/PROJECT_PLAN.md` does not need a Sprint 80 correction:
  - Sprint 81 still starts with the linked-list-first product/storage ceiling
  - Sprint 82 still remains the dense/backend ceiling lane
  - Sprint 83 and later still keep the same bounded order for capability,
    assurance, maintainability, runtime, package/platform, usability, and
    final comparison work
- The ranked handoff queue is now fixed explicitly:
  1. Sprint 81: compressed-first product/storage modernization
  2. Sprint 82: bounded optional dense-backend acceleration
  3. Sprint 83: capability-breadth widening on the highest-value seams
  4. later Epic 8 lanes only after those stronger first contradictions move

### Validation
- No new code or docs correction was required beyond the closeout record.
- Sprint 80 closes from the Day 13 validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 642.39 sec`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh` = `11 / 11`
  - `bash tests/test_cmake_install.sh` = `13 / 13`

### Day 14 Exit State
- Sprint 80 now hands off one truthful starting contract for Epic 8
  implementation work.
- Sprint 81 can begin without reopening the baseline, oracle, benchmark, or
  non-goal interpretation package.
- The branch closes cleanly from the Day 13 validated state.
