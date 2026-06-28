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

## Day 3 - Capability Re-rank Audit

### Goal
Reduce Sprint 94's broad capability-surface modernization problem to one
ranked live contradiction map centered on the strongest scalar-contract,
index-width maturity, and solver-family breadth seams.

### Actions
- Re-read the Sprint 94 Day 3 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the closest prior audit pattern:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day3-dense-hotspot-profiling-audit.md`
- Re-read the strongest current public scalar and index contract owners:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
- Re-read the strongest current shared dense-scalar public helper owner:
  - `include/sparse_dense.h`
- Re-read the strongest current widened solver-family public owners:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_qr.h`
- Re-searched the live capability and constraint seams across source, test,
  benchmark, README, maintainer, and Epic 9 review/todo surfaces:
  - `sparse_scalar_t`
  - `SPARSE_SCALAR_BITS`
  - `SPARSE_IDX_BITS`
  - `idx_t`
  - `real-only`
  - `complex`
  - `mixed precision`
- Wrote the ranked Day 3 capability audit artifact.

### Findings
- Sprint 94's broad capability problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - the public scalar contract centered on `include/sparse_types.h` and the
      shared matrix-shell helper seam in `include/sparse_matrix.h`, where the
      repo now advertises one bounded scalar-preparation seam but still ships
      a real-only `double` contract
  - strongest second target:
    - the touched 64-bit and ABI-maturity seam, where `SPARSE_IDX_BITS` and
      `idx_t` are real product features but touched consumer interpretation,
      formatting, and width-aware proof reading are still the main remaining
      maturity question rather than raw typedef availability
  - strongest third target:
    - solver-family breadth concentrated in iterative, eigensolver, QR, SVD,
      and adjacent dense-helper owners, where caller-facing scalar naming has
      widened but implementation semantics still read as bounded real-only
      support
  - strongest fourth target:
    - proof and benchmark follow-through so any widened scalar or index claim
      is anchored to maintained executable truth
  - strongest support-only but real target:
    - public and maintainer wording that still needs to stay truthful about
      bounded capability breadth and explicit non-claims
- The strongest current contradiction is still the public scalar-contract
  ceiling:
  - `include/sparse_types.h` explicitly binds `sparse_scalar_t` to `double`
    and states that the current shipped contract remains real-only
  - `include/sparse_matrix.h` routes shared matrix-shell helper paths through
    `sparse_scalar_t`, but explicitly says this does not imply broad generic
    numeric or complex support
  - `README.md` and `docs/maintainer_guide.md` repeat the same bounded
    interpretation, which means the public capability story is intentionally
    prepared for widening but still narrower than a state-of-the-art
    competitive scalar surface
- The strongest second contradiction is index-width maturity rather than
  missing index configurability:
  - `SPARSE_IDX_BITS` and `idx_t` are already first-class public contract
    surfaces
  - Sprint 91-93 materially improved touched formatting and width-aware
    printing on some reviewed lanes, but the remaining challenge is now
    touched-path maturity and ABI clarity rather than "add 64-bit support"
  - that makes index-width work a real Sprint 94 target, but it reads after
    the first scalar-contract widening seam rather than before it
- The strongest third contradiction is solver-family breadth concentration:
  - `include/sparse_iterative.h`, `include/sparse_eigs.h`, and
    `include/sparse_qr.h` already route many caller-owned buffers through
    `sparse_scalar_t`
  - `include/sparse_dense.h` still exposes a `double`-backed dense matrix
    type and `double` dense kernels, which keeps the deeper implementation
    story materially real-only
  - the strongest implementation owners tied to that bounded breadth remain:
    - `src/sparse_dense.c`
    - `src/sparse_qr.c`
    - `src/sparse_svd.c`
    - `src/sparse_eigs.c`
    - `src/sparse_iterative.c`
    - `src/sparse_matrix.c`
  - that makes solver-family widening real Sprint 94 work, but explicitly as
    a bounded follow-through on the scalar/index contract rather than as a
    generic family-wide rewrite
- The strongest fourth contradiction is proof and evidence follow-through:
  - the repo already has reviewed scalar-sensitive proof owners:
    - `tests/test_dense.c`
    - `tests/test_qr.c`
    - `tests/test_svd.c`
    - `tests/test_eigs.c`
    - `tests/test_iterative.c`
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
  - benchmark and evidence owners exist, but they remain second-tier relative
    to the reviewed executable proof surfaces:
    - `benchmarks/bench_svd.c`
    - `benchmarks/bench_eigs.c`
    - `benchmarks/bench_iterative_reuse.c`
- Sprint 94's fix-now vs deferred split is now clearer:
  - should drive Sprint 94 implementation:
    - one scalar-contract widening seam
    - touched 64-bit and ABI maturity on that seam
    - one bounded solver-family breadth follow-through
    - proof/docs/package alignment for the widened claim
  - remains later or bounded non-claim territory:
    - fake full-library complex support
    - fake broad mixed-precision maturity
    - generic templated rewrite across every family
    - broad package or platform symmetry claims

### Validation
- Re-read the public scalar and index contract in `include/sparse_types.h`.
- Re-read the shared matrix-shell owner in `include/sparse_matrix.h`.
- Re-read the strongest widened solver-family public owners in
  `include/sparse_dense.h`, `include/sparse_iterative.h`,
  `include/sparse_eigs.h`, and `include/sparse_qr.h`.
- Re-searched the live capability, non-claim, and width-contract seams across
  the touched tree and the dated Epic 9 review/todo package.

### Day 3 Exit State
- Sprint 94 now has one ranked live capability contradiction map grounded in
  the current post-Sprint-93 tree.
- The strongest first Sprint 94 implementation center is fixed to the public
  scalar-contract seam, with index-width maturity second and solver-family
  breadth third.
- Day 4 can freeze the scalar/index capability contract without reopening the
  ranked capability order.

## Day 4 - Scalar and Index Capability Contract Design

### Goal
Separate remaining Sprint 94 debt into scalar-contract widening,
index/ABI-maturity follow-through, and solver-family breadth so the first
implementation boundary can stay bounded to the highest-value capability seam.

### Actions
- Re-read the Sprint 94 Day 4 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 3 ranked capability audit:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day3-capability-rerank-audit.md`
- Re-read the Sprint 94 section of the Epic 9 project plan so the design
  stays inside the explicit capability/non-claim fence.
- Re-read the closest contract-design pattern:
  - `docs/planning/EPIC_9/SPRINT_92/artifacts/day5-portable-backend-abi-and-runtime-contract-design.md`
- Re-read the current maintained scalar/index interpretation in
  `docs/maintainer_guide.md`.
- Reconciled the ranked Day 3 capability owners against the project-plan item
  split:
  - scalar/index architecture design
  - scalar-family widening batch
  - index/ABI maturity batch
  - solver-family breadth batch
  - proof/docs/package alignment
- Wrote the Day 4 scalar/index capability contract artifact.

### Findings
- Sprint 94 now has one explicit scalar/index capability contract:
  - public scalar-contract debt:
    - means the repo has already widened naming and some caller-facing buffer
      seams through `sparse_scalar_t`, but still ships a real-only `double`
      product truth
    - remains the strongest first-class implementation target
  - index/ABI-maturity debt:
    - means compile-time `SPARSE_IDX_BITS` and `idx_t` are real public
      features, but touched-path formatting, width-aware proof, and consumer
      interpretation are not yet as mature as the contract wording invites
    - remains real Sprint 94 work, but sequenced behind the first scalar seam
      unless directly forced
  - solver-family breadth debt:
    - means some solver-family public buffers already route through
      `sparse_scalar_t`, while deeper dense/solver owners still read as
      bounded real-only implementations
    - remains real Sprint 94 work, but only where a first scalar/index
      widening actually needs it
- The strongest Day 4 clarification is now explicit:
  - Sprint 94 should not treat all remaining capability debt as one generic
    "support more numeric types" problem
  - it should not treat wider-index maturity as equivalent to broad scalar
    widening
  - it should first widen one bounded scalar-contract seam, then tighten
    touched index/ABI maturity and solver-family breadth only where the same
    widened seam still depends on them
- The preserved non-claim fence is now fixed more sharply:
  - no fake full-library complex support claim
  - no fake broad mixed-precision maturity claim
  - no fake templated-everywhere product story
  - no broad package/platform symmetry claim detached from the touched scalar
    and index seams
- The strongest direct-owner reading is now explicit:
  - first-center public and implementation owners:
    - `include/sparse_types.h`
    - `include/sparse_matrix.h`
    - the matching strongest shared implementation seam behind that public
      scalar owner
  - second-center index/ABI owners if truly forced:
    - touched width-aware public headers
    - touched formatting, allocation, and consumer-proof owners
  - third-center solver-family owners only if the widened scalar/index seam
    truly requires them:
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_qr.h`
    - `include/sparse_dense.h`
    - the matching implementation and proof owners
  - later proof-only or support-only owners unless the first landing forces
    movement:
    - `benchmarks/bench_svd.c`
    - `benchmarks/bench_eigs.c`
    - `benchmarks/bench_iterative_reuse.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The useful bounded product interpretation is now fixed:
  - builtin reviewed defaults stay authoritative
  - public wording may widen one bounded scalar lane without claiming broad
    complex or mixed-precision maturity
  - touched 64-bit maturity must read as stronger ABI/consumer correctness on
    touched paths, not as a claim that every product surface is fully 64-bit
    battle-hardened

### Validation
- Re-read the ranked Day 3 capability audit.
- Re-read the Sprint 94 project-plan contract.
- Re-read the closest prior contract-design pattern.
- Re-read the current maintained scalar/index interpretation in
  `docs/maintainer_guide.md`.

### Day 4 Exit State
- Sprint 94 now has one explicit scalar/index capability contract before code
  movement.
- The strongest first Sprint 94 implementation center remains fixed to the
  bounded public scalar-contract seam.
- Day 5 can freeze the first implementation boundary without reopening the
  ranked capability order.

## Day 5 - First Implementation Boundary

### Goal
Fix one bounded first implementation fence so Sprint 94 starts with the
highest-value scalar-contract seam instead of generic capability churn.

### Actions
- Re-read the Sprint 94 Day 5 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 3 ranked capability audit:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day3-capability-rerank-audit.md`
- Re-read the Day 4 scalar/index capability contract:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day4-scalar-and-index-capability-contract-design.md`
- Re-read the closest first-boundary patterns:
  - `docs/planning/EPIC_9/SPRINT_93/artifacts/day5-first-implementation-boundary.md`
  - `docs/planning/EPIC_9/SPRINT_91/artifacts/day4-first-implementation-boundary.md`
- Reconciled the ranked Day 3 and Day 4 owners against the required first
  scalar landing and the explicitly later index-, solver-family-, proof-, and
  support-surface queues.
- Wrote the Day 5 first-implementation boundary artifact.

### Findings
- Sprint 94 now has one explicit first implementation fence:
  - required first landing:
    - `include/sparse_types.h`
    - the matching strongest shared scalar implementation seam behind that
      public type owner
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
    - `docs/maintainer_guide.md`
  - explicitly later unless the first landing truly forces movement:
    - touched 64-bit and ABI maturity beyond the first scalar seam
    - `include/sparse_dense.h`
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_qr.h`
    - the matching solver-family implementation owners
    - `tests/test_dense.c`
    - `tests/test_qr.c`
    - `tests/test_svd.c`
    - `tests/test_eigs.c`
    - `tests/test_iterative.c`
    - `benchmarks/bench_svd.c`
    - `benchmarks/bench_eigs.c`
    - `benchmarks/bench_iterative_reuse.c`
    - `README.md`
    - `INSTALL.md`
    - `Makefile`
    - `CMakeLists.txt`
- The strongest Day 5 clarification is now explicit:
  - Sprint 94 should start by widening the bounded public scalar-contract seam
  - it should not begin by widening every solver-family owner at once
  - it should not reopen broad docs, package wording, or benchmark
    interpretation in the first batch unless the touched scalar seam itself
    truly forces it
- The first batch is now explicitly deferred away from:
  - fake full-library complex support
  - broad mixed-precision maturity
  - generic family-wide numeric rewriting
  - wider-index cleanup detached from the touched scalar seam
  - solver-family breadth expansion detached from the first capability landing
  - benchmark/reporting widening detached from reviewed proof owners
  - support-surface wording churn detached from the first scalar landing

### Validation
- Re-read the ranked Day 3 capability audit.
- Re-read the Day 4 scalar/index contract.
- Re-read the closest prior first-boundary patterns.

### Day 5 Exit State
- Sprint 94 now has one explicit first implementation boundary.
- The first code landing is fixed to the bounded public scalar-contract seam
  with only the strongest matrix-shell public/support owners as directly
  forced follow-through.
- Day 6 can define the scalar widening implementation contract without
  reopening the ranked first-center choice.

## Day 6 - Scalar-Family Widening Design

### Goal
Define the bounded implementation contract for Sprint 94's first scalar-family
widening seam so Day 7 can land one real capability step without widening
into broad solver-family, index, or support churn.

### Actions
- Re-read the Sprint 94 Day 6 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 5 first-implementation boundary:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day5-first-implementation-boundary.md`
- Re-read the strongest first-center public scalar owner:
  - `include/sparse_types.h`
- Re-read the strongest directly forced shared matrix-shell owner:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- Re-read the strongest directly forced proof owners:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- Reconciled the live scalar aliasing story against the still-`double`
  matrix-shell implementation seam:
  - public alias already widened to `sparse_scalar_t`
  - matrix-shell storage and helper implementation still materially read as
    `double`
- Wrote the Day 6 scalar-widening design artifact.

### Findings
- Sprint 94 now has one explicit scalar-widening implementation contract:
  - exact first implementation center:
    - `include/sparse_types.h`
    - `src/sparse_matrix.c`
  - exact widening target:
    - finish the first public-to-implementation scalar seam on the shared
      matrix-shell helper path so the public `sparse_scalar_t` contract stops
      reading as a naming-only preparation layer on this owner
    - keep the widening bounded to the matrix-shell helper, arithmetic, and
      storage seam rather than reopening every solver-family implementation
      owner at once
- The preserved invariants are now fixed:
  - the default reviewed build remains real-only and continues to use `double`
    semantics unless the widened contract explicitly says otherwise
  - public API clarity must improve, not blur; callers should be able to see
    that the widened seam is real on the touched owner without inferring fake
    broad numeric genericity elsewhere
  - touched width and ABI interpretation remains unchanged in Day 7; wider
    index maturity is still a later batch unless the scalar landing truly
    forces it
  - touched proof owners stay deterministic and auditable
- The strongest directly forced follow-through is now fixed to:
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- The strongest explicitly deferred work is also fixed:
  - `include/sparse_dense.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_qr.h`
  - the matching solver-family implementation owners
  - `docs/maintainer_guide.md` and broader public/support wording unless the
    first scalar landing truly changes the contract reading
- The useful bounded interpretation is now explicit:
  - Day 7 should make the shared matrix-shell scalar seam real enough to
    support one credible capability widening step
  - it should not be a broad complex-support claim
  - it should not become a generic family-wide numeric rewrite

### Validation
- Re-read the Day 5 first implementation boundary.
- Re-read the first-center public and shared implementation owners.
- Re-read the strongest directly forced proof owners.

### Day 6 Exit State
- Sprint 94 now has one exact Day 7 implementation center fixed to
  `include/sparse_types.h` plus the shared matrix-shell implementation seam.
- The widened scalar target and preserved invariants are explicit before code
  moves.
- Later index/ABI and solver-family work remains clearly sequenced behind the
  first scalar landing.

## Day 7 - Scalar-Family Widening Batch

### Goal
Land one real bounded scalar widening step on the shared matrix-shell owner so
the public `sparse_scalar_t` contract stops reading like naming-only
preparation on this touched seam.

### Actions
- Re-read the Sprint 94 Day 7 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 6 scalar widening contract:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day6-scalar-family-widening-design.md`
- Re-read the touched public scalar contract owner:
  - `include/sparse_types.h`
- Re-read the strongest shared matrix-shell implementation seam:
  - `src/sparse_matrix.c`
  - `src/sparse_matrix_internal.h`
- Re-read the strongest directly forced proof owner:
  - `tests/test_sparse_matrix.c`
- Reconciled the remaining `double`-specialized spots against the widened
  matrix-shell helper contract:
  - internal node storage still read as `double`
  - builder triplet storage still read as `double`
  - Matrix Market pattern-load default still read as `double`
- Landed the bounded scalar widening batch:
  - widened the touched matrix-shell storage/build seam to
    `sparse_scalar_t`
  - tightened the public scalar contract wording so the touched matrix-shell
    owner now truthfully claims that the storage/build seam follows the alias
  - added one focused transpose proof so the builder path is exercised through
    the public scalar alias test
- Wrote the Day 7 implementation artifact.

### Findings
- Sprint 94's first scalar landing is now real on the touched owner:
  - `src/sparse_matrix_internal.h`
    - `Node.value` now uses `sparse_scalar_t`
  - `src/sparse_matrix.c`
    - `make_node(...)` now takes `sparse_scalar_t`
    - `SparseBuildEntry.value` now uses `sparse_scalar_t`
    - builder-side overwrite selection now keeps `sparse_scalar_t`
    - Matrix Market pattern defaults now enter through `sparse_scalar_t`
  - `include/sparse_types.h`
    - the public scalar contract wording now explicitly says the touched
      matrix-shell storage/build seam follows the alias
- The Day 7 landing stayed bounded:
  - directly forced proof follow-through:
    - `tests/test_sparse_matrix.c`
  - not needed:
    - `include/sparse_matrix.h`
    - `tests/test_integration.c`
    - `docs/maintainer_guide.md`
- The kept proof win is explicit:
  - `test_matrix_public_scalar_alias` now exercises a transpose/build path in
    addition to insert/matvec/norminf/scale, so the widened builder seam is
    covered from the public matrix-shell surface
- The preserved non-claims stayed intact:
  - no broad solver-family numeric widening
  - no wider-index/ABI reopening
  - no fake complex or mixed-precision product claim

### Validation
- `make format`
- `make lint`
- `make test`

### Day 7 Exit State
- Sprint 94 now has one real bounded scalar widening landing on the shared
  matrix-shell owner.
- The public `sparse_scalar_t` contract no longer reads like naming-only
  preparation on this touched seam.
- Day 8 can rerank the remaining capability contradictions from a validated
  post-landing baseline instead of from a still-`double` matrix-shell owner.

## Day 8 - Post-Landing Audit and Re-rank

### Goal
Re-rank the remaining Sprint 94 capability contradictions from the validated
Day 7 state so the next implementation move stays bounded to the strongest
remaining maturity seam rather than widening into generic solver-family churn.

### Actions
- Re-read the Sprint 94 Day 8 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 3 capability rerank and Day 4 scalar/index contract:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day3-capability-rerank-audit.md`
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day4-scalar-and-index-capability-contract-design.md`
- Re-read the Day 7 landing result:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day7-scalar-family-widening-batch.md`
- Re-scanned the strongest remaining touched owner surfaces after the Day 7
  landing:
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
  - `src/sparse_matrix.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`
  - the later solver-family public headers and implementation owners
- Reconciled what Day 7 materially closed versus what still remains open
  across scalar breadth, index/ABI maturity, solver-family breadth, and proof
  or public-surface clarity.
- Wrote the Day 8 rerank artifact.

### Findings
- The Day 7 scalar landing materially closed the strongest first Sprint 94
  contradiction:
  - the shared matrix-shell public helper seam now has a real matching
    storage/build owner through `sparse_scalar_t`
  - the touched matrix-shell scalar story no longer reads like naming-only
    preparation
  - a second immediate scalar landing is not the highest-value next move
- The strongest remaining contradiction is now the touched 64-bit and ABI
  maturity seam:
  - `SPARSE_IDX_BITS`, `idx_t`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` are
    already first-class contract surfaces
  - the strongest remaining work is now touched-path maturity and consumer
    interpretation, especially on the matrix-shell I/O and debug/inspection
    surfaces in `src/sparse_matrix.c`
  - this is a tighter, more defensible next batch than widening deeper
    solver-family numeric owners immediately
- Solver-family breadth is still real, but now clearly third:
  - later public scalar-buffer owners in iterative, eigensolver, QR, SVD, and
    dense-family seams still read as bounded real-only implementations
  - reopening them next would widen claim scope faster than the current
    bounded scalar/index contract intends
- Proof and public-surface clarity remain support lanes rather than the next
  implementation center:
  - focused proof should follow the touched index/ABI batch where it is
    directly forced
  - broader maintainer or product wording still remains later unless the
    touched index path truly changes the contract reading

### Validation
- Re-read the Day 3 rerank from the live post-Day-7 tree.
- Re-read the Day 4 scalar/index contract.
- Re-read the Day 7 landed batch.
- Re-scanned the strongest touched matrix-shell index/ABI owner surfaces.

### Day 8 Exit State
- Sprint 94's remaining contradiction order is now fixed from the validated
  post-Day-7 tree:
  - strongest next target:
    - touched 64-bit and ABI maturity
  - strongest later target:
    - bounded solver-family breadth only where the widened scalar/index
      contract truly needs it
  - strongest support-only later target:
    - proof/docs/package wording only where the touched capability claim
      truly moves
- The exact Day 9 design center is now fixed to:
  - `src/sparse_matrix.c`
- The strongest directly forced support-only follow-through, only if the Day 9
  contract truly needs it, is now fixed to:
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`

## Day 9 - Index and ABI Maturity Design

### Goal
Define the bounded touched-path 64-bit and ABI maturity contract so Day 10 can
land one real matrix-shell index-width improvement without reopening broader
solver-family or package surfaces.

### Actions
- Re-read the Sprint 94 Day 9 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 8 rerank:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day8-post-landing-audit-and-rerank.md`
- Re-read the strongest touched owner surfaces on the matrix-shell family:
  - `src/sparse_matrix.c`
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`
- Reconciled the touched index/ABI maturity problem against the live owner
  split:
  - public width contract already exists through `SPARSE_IDX_BITS`,
    `idx_t`, `SPARSE_PRIDX`, `SPARSE_SCNIDX`, and `sparse_idx_bits()`
  - the strongest remaining batch is not a typedef change
  - the strongest remaining batch is touched-path maturity on Matrix Market
    save/load plus debug and inspection printing paths inside the matrix-shell
    owner
- Wrote the Day 9 design artifact.

### Findings
- Sprint 94 now has one exact touched index/ABI implementation contract:
  - required Day 10 center:
    - `src/sparse_matrix.c`
  - exact maturity target:
    - strengthen the matrix-shell save/load and debug/inspection paths so the
      touched consumer and diagnostic surfaces read as width-aware and ABI-safe
      under both `SPARSE_IDX_BITS=32` and `SPARSE_IDX_BITS=64`
    - keep the batch bounded to formatting, parsing, and width-aware
      interpretation on this owner rather than widening into broader package,
      binary, or solver-family claims
- The preserved invariants are now fixed:
  - `SPARSE_IDX_BITS`, `idx_t`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` remain
    the authoritative public width contract
  - the default reviewed build remains unchanged unless the touched batch
    explicitly says otherwise
  - no new dynamic-ABI or broad binary-compatibility claim should be inferred
  - touched proof owners stay deterministic and focused on the matrix-shell
    consumer and diagnostic seam
- The strongest directly forced follow-through is now fixed to:
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`
- The strongest explicitly deferred work is also fixed:
  - later solver-family breadth follow-through
  - broader maintainer, README, INSTALL, or package wording
  - broader benchmark/reporting interpretation
  - broad binary, shared-library, or platform-symmetry claim widening

### Validation
- Re-read the Day 8 rerank from the validated post-Day-7 tree.
- Re-read the touched matrix-shell save/load and debug/inspection owners.
- Re-read the strongest directly forced proof owners.

### Day 9 Exit State
- Sprint 94 now has one exact Day 10 implementation center fixed to
  `src/sparse_matrix.c`.
- The next batch is explicitly bounded to touched matrix-shell index and ABI
  maturity on save/load and diagnostic surfaces.
- Later solver-family and support-surface work remains clearly sequenced
  behind this touched index-width batch.

## Day 10 - Index and ABI Maturity Batch

### Goal
Land one bounded matrix-shell index and ABI maturity step on the touched
save/load and diagnostic surfaces so the current `idx_t` width contract reads
as more trustworthy to consumers and maintainers on this owner.

### Actions
- Re-read the Sprint 94 Day 10 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 9 touched index/ABI contract:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day9-index-and-abi-maturity-design.md`
- Re-read the required owner:
  - `src/sparse_matrix.c`
- Re-read the strongest directly forced proof surfaces:
  - `tests/test_sparse_io.c`
  - `tests/test_sparse_matrix.c`
- Landed the bounded matrix-shell index/ABI batch:
  - added checked stream-print helpers for matrix-shell save and diagnostic
    surfaces so write failures now fail closed with `SPARSE_ERR_IO`
  - tightened Matrix Market parsing so malformed negative dimensions,
    rectangular symmetric headers, zero coordinates, and out-of-range
    coordinates are rejected as `SPARSE_ERR_PARSE` instead of being accepted
    or silently dropped
  - kept the batch strictly inside the matrix-shell owner and the directly
    forced proof surfaces
- Added focused proof follow-through:
  - malformed Matrix Market parse rejection coverage in `tests/test_sparse_io.c`
  - width-aware diagnostic output smoke coverage in
    `tests/test_sparse_matrix.c`
- Ran the full implementation-day validation queue to completion after fixing
  one local brace regression, one `-Wformat-nonliteral` strict-warning issue
  inside the checked print helper, and one cppcheck redundancy note on the new
  bounds guard.
- Wrote the Day 10 implementation artifact.

### Findings
- Sprint 94 now has one real touched matrix-shell index/ABI maturity landing:
  - `src/sparse_matrix.c`
    - save and diagnostic output paths now treat stream write failures as real
      `SPARSE_ERR_IO` results instead of ignoring them
    - Matrix Market header and coordinate parsing now rejects malformed
      dimension and coordinate cases more truthfully as parse failures
    - the touched width-aware read/write story is sharper without changing the
      public width contract itself
- The Day 10 landing stayed inside the Day 9 fence:
  - directly forced proof follow-through:
    - `tests/test_sparse_io.c`
    - `tests/test_sparse_matrix.c`
  - not needed:
    - `include/sparse_matrix.h`
    - broader README / INSTALL / maintainer wording
    - later solver-family owners
- The kept maturity win is explicit:
  - the matrix-shell consumer and diagnostic seam now better matches the
    existing `SPARSE_IDX_BITS` / `idx_t` / format-macro contract
  - malformed Matrix Market inputs are no longer silently tolerated on the
    touched cases covered by the batch
- The preserved non-claims stayed intact:
  - no typedef or ABI-policy rewrite
  - no broad solver-family widening
  - no new shared-library or broad binary-compatibility claim

### Validation
- `make format`
- `make lint`
- `make test`

### Day 10 Exit State
- Sprint 94 now has one validated bounded index/ABI maturity landing on the
  matrix-shell save/load and diagnostic owner.
- The touched consumer and diagnostic surfaces are materially sharper without
  widening beyond the Sprint 94 scalar/index contract.
- Day 11 can now decide whether any remaining solver-family breadth follow-
  through is still justified from this validated post-Day-10 baseline.

## Day 11 - Solver-Family Breadth and Alignment Design

### Goal
Decide whether Sprint 94 still needs one bounded solver-family implementation
landing after the validated Day 7 scalar and Day 10 index/ABI batches, or
whether the truthful final move is support-only alignment from the post-Day-10
baseline.

### Actions
- Re-read the Sprint 94 Day 11 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the validated Day 8 post-landing rerank and Day 10 index/ABI batch:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day8-post-landing-audit-and-rerank.md`
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day10-index-and-abi-maturity-batch.md`
- Re-read the strongest widened scalar/index contract owners:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- Re-read the strongest already-real bounded solver-family public scalar seams:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_qr.h`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_qr.c`
- Re-read the strongest support-only interpretation owners:
  - `README.md`
  - `docs/maintainer_guide.md`
- Wrote the Day 11 breadth/alignment design artifact.

### Findings
- Sprint 94 does not need another solver-family implementation landing to make
  the current bounded capability claim truthful:
  - the Day 7 matrix-shell scalar landing made the highest-value shared public-
    to-implementation scalar seam real
  - the Day 10 matrix-shell index/ABI landing made the touched width-aware
    consumer and diagnostic seam materially sharper
  - the iterative, eigensolver, and QR public scalar seams were already real
    and proof-owned before Day 11
- The strongest remaining contradiction is now support-surface interpretation,
  not missing implementation:
  - maintainer wording should reflect that the current bounded scalar claim now
    includes the real matrix-shell storage/build seam plus the already-real
    iterative/eigs/QR public seams
  - public wording should stay explicit that SVD, dense helpers, and broad
    complex or mixed-precision maturity remain deferred
- Sprint 94's final Day 12 center is therefore support-only:
  - required Day 12 center:
    - `docs/maintainer_guide.md`
  - directly forced support-only follow-through only if the wording truly
    needs it:
    - `README.md`
  - not reopened unless a wording contradiction forces it:
    - solver-family implementation owners
    - proof binaries or package/install surfaces

### Validation
- Re-read the Day 8 rerank from the validated post-Day-7 tree.
- Re-read the Day 10 validated matrix-shell index/ABI batch.
- Re-read the current public scalar and support-surface contract owners.

### Day 11 Exit State
- Sprint 94's final implementation batch is explicitly retired by evidence.
- Day 12 is now fixed as a support-only alignment pass centered on
  `docs/maintainer_guide.md`, with `README.md` only if a real wording
  contradiction remains.
- The bounded Sprint 94 capability claim stays truthful without reopening
  broader solver-family numeric owners.

## Day 12 - Solver-Family Breadth and Support Alignment Batch

### Goal
Land the final support-only follow-through required by the validated Sprint 94
capability contract so the maintainer and public wording match the actual
scalar/index landing set without reopening broader solver-family owners.

### Actions
- Re-read the Sprint 94 Day 12 plan target in
  `docs/planning/EPIC_9/SPRINT_94/PLAN.md`.
- Re-read the Day 11 alignment design:
  - `docs/planning/EPIC_9/SPRINT_94/artifacts/day11-solver-family-breadth-and-alignment-design.md`
- Re-read the required center and only directly forced support surface:
  - `docs/maintainer_guide.md`
  - `README.md`
- Landed the bounded support-only alignment batch:
  - tightened `docs/maintainer_guide.md` so the maintained scalar reading now
    explicitly includes the real shared matrix-shell storage/build seam and the
    already-real iterative/eigs/QR public seams
  - updated the maintained proof-owner map so it reflects the validated Sprint
    94 baseline, including the touched Matrix Market parse-width proof in
    `tests/test_sparse_io.c`
  - tightened `README.md` so the bounded real-only scalar limitation matches
    the now-real matrix-shell and already-real iterative/eigs/QR public seams
- Kept the batch support-only:
  - no solver-family implementation owners reopened
  - no proof binaries or package/install surfaces changed
- Wrote the Day 12 alignment artifact.

### Findings
- Sprint 94's final landing set is now aligned across code, proof ownership,
  maintainer interpretation, and public non-claims.
- The Day 12 support batch stayed inside the Day 11 fence:
  - required center:
    - `docs/maintainer_guide.md`
  - directly forced support-only follow-through:
    - `README.md`
  - not needed:
    - solver-family implementation owners
    - focused capability tests
    - package/install/export wording
- The kept non-claims stayed explicit:
  - no broad complex support
  - no broad mixed-precision maturity
  - no dense/SVD family-wide scalar widening claim
  - no broader package or platform reinterpretation

### Validation
- Re-read the Day 11 design from the validated post-Day-10 baseline.
- Re-read the touched maintainer and public wording after the support-only
  edits.

### Day 12 Exit State
- Sprint 94's final capability claim is now supported by aligned code, proof,
  maintainer wording, and public wording.
- The final Sprint 94 landing set is explicit without reopening any broader
  solver-family implementation owner.
- Day 13 can now run the full close validation sweep from one fixed landing
  set.
