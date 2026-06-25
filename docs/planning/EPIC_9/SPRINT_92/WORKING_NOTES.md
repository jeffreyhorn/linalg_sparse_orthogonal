# Sprint 92 Working Notes

## Day 1 - Scope and Backend Baseline

### Goal
Turn the Sprint 92 project-plan section and the Sprint 91 validated closeout
into one bounded portable dense backend and kernel-maturity execution package
before any hotspot profiling, backend design, or implementation lands.

### Actions
- Re-read the Sprint 92 contract in
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 92 day-by-day plan in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the closest prior closeout and handoff surfaces:
  - `docs/planning/EPIC_9/SPRINT_91/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_9/SPRINT_91/RETROSPECTIVE.md`
- Re-read the closest prior Epic 9 planning baseline:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day1-scope-and-epic9-baseline.md`
- Reconfirmed that the strongest local reviewed entry point still begins at:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest likely Sprint 92 touch surfaces by line count and
  ownership role:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_dense.c`
  - `tests/test_qr.c`
- Wrote the Day 1 scope artifact and authoritative-input list.

### Findings
- Sprint 92 begins from a validated Sprint 91 close state, not from another
  generic direct-family cleanup reset:
  - strongest local reviewed baseline remains `make quality-review-full`
  - reviewed CMake parity was re-materialized live and remains explicit:
    - `ctest -N --test-dir build/quality-review-cmake` = `53`
    - Makefile/CMake parity = `53 vs 53`
- Sprint 91 already moved the strongest prior product-model contradiction:
  - compressed CSR/CSC inputs now have first-class public constructor-style
    entry paths
  - the public one-shot vs repeated-run direct story is sharper
  - constructor-built direct workflows now have explicit integration proof
- That means Sprint 92 can start from the next real Epic 9 contradiction
  center:
  - the current dense-kernel and optional-backend ceiling on the strongest
    direct-family workloads
- The highest-value Sprint 92 package is now fixed explicitly around:
  - dense hotspot profiling
  - backend ABI and runtime-selection design
  - portable backend integration
  - solver adoption follow-through
  - benchmark and proof observability
  - build/package alignment
- The live tree currently points most strongly at these Sprint 92 surfaces:
  - strongest dense/backend implementation owners:
    - `src/sparse_dense.c` = `862`
    - `src/sparse_ldlt_csc.c` = `2694`
    - `src/sparse_chol_csc.c` = `1279`
    - `src/sparse_qr.c` = `1563`
  - strongest touched benchmark and measurement owners:
    - `benchmarks/bench_chol_csc.c` = `423`
    - `benchmarks/bench_refactor_csc.c` = `611`
    - `benchmarks/bench_svd.c` = `180`
  - strongest proof-owner tests likely to matter:
    - `tests/test_chol_csc.c` = `4987`
    - `tests/test_ldlt_csc.c` = `3680`
    - `tests/test_dense.c` = `584`
    - `tests/test_qr.c` = `3234`
  - strongest support and package wording surfaces if backend work forces
    follow-through:
    - `README.md` = `1136`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `727`
    - `Makefile` = `908`
    - `CMakeLists.txt` = `416`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
- Sprint 92 is explicitly bounded against:
  - treating optional acceleration as stronger than builtin fallback truth
  - promising broad platform symmetry before a maintained portable backend lane
    exists
  - widening into runtime/threading, capability-surface, or packaging-product
    work before the backend seam is fixed
  - treating benchmark evidence as stronger than solver correctness, install,
    or reviewed proof-owner surfaces

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the strongest likely backend, benchmark, proof, and support
  surfaces by live file size and owner role.

### Day 1 Exit State
- Sprint 92 now starts from one precise portable dense backend and
  kernel-maturity execution package rather than from a generic "speed up
  direct solvers" bucket.
- The strongest likely touch surfaces, preserved non-goals, and maintained
  reviewed starting truth are fixed in writing before the validation and
  proof-owner recheck begins.
- Day 2 can now freeze the authoritative reviewed, benchmark, install/export,
  and workflow truth split without reopening the Day 1 scope question.

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, benchmark, install/export, example, and workflow truth split before
Sprint 92 begins backend-focused implementation work on the dense-kernel and
direct-family surfaces.

### Actions
- Re-read the Sprint 92 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_9/SPRINT_91/artifacts/day2-validation-baseline-and-maintained-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained Sprint 92
  truth surfaces:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so Sprint 92 does
  not overclaim reviewed parity, backend breadth, or install/export coverage
  while touching the dense-kernel ceiling.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 92 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for
  backend-maturity work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial backend-contract, proof-owner, benchmark, or support-surface
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 92
  code lands:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest reviewed executable truth owners for Sprint 92’s dense/backend
  lane are now fixed around:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
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
    reviewed Makefile compile-quality path, reviewed CMake parity path, and
    dead-code path
  - macOS remains a narrower reviewed Apple Clang lane plus a supplemental
    static-first install/`pkg-config` confidence lane
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim reviewed Makefile parity or separate reviewed install-validation
    parity
- The highest-signal rerun set is now fixed for the rest of Sprint 92:
  - `./build/quality-review-cmake/test_dense`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 92 should read backend-maturity changes against dense-kernel and
    direct-family proof owners, not against the product-model owners that
    drove Sprint 91
  - benchmark reporting remains a bounded command/script-owned evidence
    surface, not a reviewed-binary parity claim
  - install/export proof and workflow evidence remain bounded maintained
    support surfaces, not broad backend-platform maturity claims

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the representative reviewed binaries/examples and the maintained
  install/export, consumer, reporting, and workflow-owner surfaces.

### Day 2 Exit State
- Sprint 92 now has one explicit validation and maintained-surface contract
  before backend implementation begins.
- The reviewed dense/direct truth owners, canonical reporting owner,
  install/export proof owners, and workflow-side support evidence are fixed in
  writing.
- Later Day 3-Day 11 audit, design, implementation, and follow-through work no
  longer needs to guess which surfaces are authoritative.

## Day 3 - Dense Hotspot Profiling Audit

### Goal
Reduce Sprint 92's broad portable-backend problem to one ranked live
contradiction map centered on the strongest builtin dense-kernel hotspots, the
highest-value direct-family consumers, and the narrow current optional-backend
story.

### Actions
- Re-read the Sprint 92 Day 3 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the closest prior Epic 9 structural audit:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day3-product-performance-capability-gap-audit.md`
- Re-scanned the live tree against the strongest Sprint 92 contradiction
  class:
  - builtin dense-kernel hotspots
  - strongest direct-family dense consumers
  - Darwin-only or bounded acceleration seams
  - runtime, allocation, and observability costs around dense work
- Re-anchored the audit directly on the current dense and direct-family owners:
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `include/sparse_ldlt.h`
  - `include/sparse_matrix.h`
- Rechecked the strongest benchmark and proof owners likely to matter later:
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `tests/test_dense.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
- Captured the live hotspot map and current backend-contract evidence from
  those owner surfaces.
- Wrote the Day 3 audit artifact and fixed the ranked backend-hotspot order in
  writing.

### Findings
- Sprint 92's broad backend problem is now reduced to one ranked live map of
  the highest-value portable-kernel opportunities:
  - strongest first target:
    - the generic dense-kernel owner in `src/sparse_dense.c`, where the
      builtin scalar GEMM/GEMV/factor/solve primitives still define the
      broadest performance ceiling
  - strongest second target:
    - the direct-family adoption seam currently concentrated in
      `src/sparse_chol_csc.c` and `src/sparse_ldlt_csc.c`, where backend
      dispatch is real but still narrow and family-local
  - strongest third target:
    - QR and adjacent dense consumers that still read as builtin-only and do
      not yet share the strongest bounded backend seam
  - strongest fourth target:
    - benchmark and observability follow-through so any widened backend path
      is measurable and fallback-visible
  - strongest support-only but real target:
    - build/package/support wording that still truthfully reflects a builtin-
      first default plus bounded optional acceleration
- The strongest current contradiction is still the backend-maturity ceiling in
  the shared dense owner:
  - `src/sparse_dense.c` still owns the generic dense GEMM/GEMV and Cholesky
    dense factor/solve primitives in self-contained scalar C
  - the only current optional accelerated lane exposed there is the
    Apple-only Accelerate probe for the Cholesky supernodal dense-kernel
    descriptor
  - that lane is environment-selected and bounded by backend-contract limits,
    not a broader portable backend story
- The strongest direct-family dense adoption surfaces are now explicit:
  - `src/sparse_chol_csc.c` already consumes `chol_csc_supernodal_dense_kernels`
    and therefore has the cleanest immediate backend-adoption seam
  - `src/sparse_ldlt_csc.c` carries its own bounded optional Accelerate seam,
    but it is still family-local rather than converged with the shared dense
    owner
  - `src/sparse_qr.c` remains a large dense consumer candidate, but it reads
    more like a later adopter than the first backend-integration center
- The strongest current owner surfaces are now explicit from the live tree:
  - `src/sparse_dense.c` = `862`
  - `src/sparse_ldlt_csc.c` = `2694`
  - `src/sparse_chol_csc.c` = `1279`
  - `src/sparse_qr.c` = `1563`
  - `benchmarks/bench_chol_csc.c` = `423`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `tests/test_dense.c` = `584`
  - `tests/test_chol_csc.c` = `4987`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3234`
- The fix-now vs later split is now explicit:
  - Sprint 92 should drive:
    - the shared dense-kernel backend seam
    - the strongest direct-family backend adopters
    - backend observability, fallback proof, and bounded package wording only
      where the implementation truly moves
  - Sprint 92 should keep later for now:
    - broad runtime/threading work
    - fake cross-platform backend symmetry
    - full-family dense-kernel convergence in every solver owner
    - capability-surface widening beyond backend maturity
- The strongest Day 3 clarification is now fixed:
  - Sprint 92 does not begin with another generic direct-family speed pass
  - it begins with one ranked backend-hotspot map
  - the best first implementation center is the shared dense-kernel owner and
    the strongest immediate Cholesky/LDL^T adoption seam
  - QR follow-through, benchmark/reporting widening, and support-surface
    wording remain real Sprint 92 work, but only after the first backend seam
    is fixed

### Validation
- Re-read the Sprint 92 Day 3 plan contract.
- Re-read the closest prior Sprint 90 structural audit.
- Re-scanned the live dense/backend owners and strongest benchmark/proof
  surfaces.
- Captured the live hotspot map and current backend-contract evidence from the
  strongest likely Sprint 92 surfaces.

### Day 3 Exit State
- Sprint 92 now has one ranked live backend-hotspot contradiction map grounded
  in the current post-Sprint-91 tree.
- The first backend implementation center is fixed to the shared dense-kernel
  owner and its strongest direct-family adoption seam.
- Day 4 can freeze the first implementation boundary without reopening the
  ranked backend-hotspot order.

## Day 4 - First Implementation Boundary

### Goal
Fix one bounded first implementation fence so Sprint 92 starts with the
highest-value backend seam instead of generic dense or direct-family churn.

### Actions
- Re-read the Sprint 92 Day 4 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 3 backend-hotspot audit against the Sprint 92 project-plan
  contract.
- Decided the required first landing center:
  - `src/sparse_dense.c`
  - the matching shared dense-kernel descriptor and optional-backend seam
- Decided which adjacent surfaces are directly forced support-only follow-
  through and which are explicitly later.
- Wrote the Day 4 boundary artifact and updated working notes with the frozen
  first-batch fence.

### Findings
- Sprint 92 now has one explicit first implementation fence:
  - required first landing:
    - `src/sparse_dense.c`
    - the matching shared dense-kernel descriptor and optional-backend seam
      consumed by the strongest existing direct-family owner
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `src/sparse_chol_csc.c`
    - `include/sparse_chol_csc_internal.h`
    - `tests/test_dense.c`
    - `tests/test_chol_csc.c`
    - `benchmarks/bench_chol_csc.c`
  - explicitly later unless the first landing truly forces movement:
    - `src/sparse_ldlt_csc.c`
    - `include/sparse_ldlt.h`
    - `tests/test_ldlt_csc.c`
    - `src/sparse_qr.c`
    - `tests/test_qr.c`
    - `benchmarks/bench_refactor_csc.c`
    - `benchmarks/bench_svd.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
    - `Makefile`
    - `CMakeLists.txt`
    - install/export and workflow surfaces
- The useful Day 4 clarification is now explicit:
  - Sprint 92 should start by improving the shared dense-kernel seam
  - it should not begin by widening every dense consumer at once
  - it should not reopen QR, package wording, runtime/threading, or fake
    cross-platform symmetry in the first batch unless the shared seam itself
    truly forces it
- The first batch now explicitly defers:
  - broad dense rewrite
  - family-wide direct-solver backend convergence as a first-batch center
  - QR/backend adoption as a first-batch center
  - benchmark/reporting widening detached from a real backend seam
  - build/package/workflow wording churn detached from the first code landing
  - runtime/threading or capability-surface widening

### Validation
- Re-read the Day 3 backend-hotspot artifact.
- Re-read the Sprint 92 Day 4 plan contract.
- Rechecked the strongest likely first-batch dense owner and its immediate
  Cholesky-side adopters against the later-adopter and support-only surfaces.

### Day 4 Exit State
- Sprint 92 now has one explicit first implementation boundary.
- The first code landing is fixed to the shared dense-kernel owner and the
  strongest immediate Cholesky-side adoption seam.
- Day 5 can define the backend ABI/runtime contract without reopening the
  ranked first-center choice.

## Day 5 - Portable Backend ABI and Runtime Contract Design

### Goal
Define the bounded builtin-vs-portable backend contract so Day 6 can widen the
shared dense-kernel seam without breaking builtin-default truth or reopening
the Day 4 boundary.

### Actions
- Re-read the Sprint 92 Day 5 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 4 first-implementation boundary artifact.
- Re-read the current shared dense-kernel and strongest adopter interfaces:
  - `include/sparse_dense.h`
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_internal.h`
  - `benchmarks/bench_backend_compare_helpers.h`
- Rechecked the current bounded backend evidence:
  - shared dense owner exposes only the Cholesky supernodal descriptor seam
  - runtime selection is environment-based and currently Apple Accelerate only
  - proof and benchmark surfaces already know how to inspect backend names
- Wrote the Day 5 architecture artifact and fixed the Day 6 implementation
  center in writing.

### Findings
- Sprint 92 now has one explicit builtin-vs-portable backend contract:
  - builtin dense kernels:
    - remain the default, self-contained, always-available product truth
    - continue to define correctness and fallback semantics for every caller
    - must stay usable even when no optional backend is present
  - optional portable backend lane:
    - should widen the shared dense-kernel descriptor/runtime-selection seam
      rather than creating another family-local acceleration pocket
    - should remain optional and capability-gated
    - should fail closed to builtin kernels when unavailable or unsupported
  - runtime or compile-time selection:
    - stays bounded to the shared dense owner
    - should present one explicit backend name / descriptor contract to direct
      consumers and proof surfaces
    - should not turn Sprint 92 into a broad public configuration-product
      rewrite
- The exact Day 6 implementation center is now fixed to:
  - `src/sparse_dense.c`
  - directly forced follow-through only if needed in:
    - `src/sparse_chol_csc.c`
    - `src/sparse_chol_csc_internal.h`
    - `tests/test_dense.c`
    - `tests/test_chol_csc.c`
    - `benchmarks/bench_chol_csc.c`
- The strongest Day 5 clarification is now explicit:
  - Sprint 92 should not try to solve every backend problem at once
  - the first landing should widen the shared descriptor/runtime-selection seam
    around the Cholesky dense kernel path
  - LDL^T, QR, broader benchmark/reporting follow-through, and package wording
    stay later unless the shared seam truly forces them
- The contract also fixes what Day 6 should not become:
  - a broad public API redesign in `include/sparse_dense.h`
  - a fake platform-symmetry claim
  - a QR or LDL^T adoption batch as the first code center
  - a build/package/workflow batch detached from the shared dense seam

### Validation
- Re-read the Sprint 92 Day 5 plan contract.
- Re-read the Day 4 boundary artifact.
- Re-read the current shared dense-kernel and Cholesky-side internal
  interfaces.
- Rechecked the benchmark/proof surfaces that already observe backend names.

### Day 5 Exit State
- Sprint 92 now has one explicit builtin-vs-portable backend contract before
  code moves.
- Day 6 implementation is fixed to the shared dense-kernel seam with only the
  strongest Cholesky-side adopter as directly forced follow-through.
- Later LDL^T, QR, benchmark, and package work stays sequenced behind the
  first backend landing.

## Day 6 - Portable Backend Integration Batch

### Goal
Land the first bounded portable-backend integration batch in the shared dense
owner, keeping builtin kernels authoritative while widening the runtime-probed
optional external backend seam for the strongest current direct-family adopter.

### Actions
- Implemented the Day 5 first landing in:
  - `src/sparse_dense.c`
- Widened the shared dense-kernel runtime probe from the prior bounded
  Apple-only seam into one optional external BLAS/LAPACK-class provider seam
  with builtin fallback still authoritative.
- Added provider-name and environment-selection support for:
  - builtin
  - Accelerate when present on Apple
  - external BLAS/LAPACK-class providers when present through runtime loading
- Kept the first adopter bounded to the existing Cholesky supernodal dense
  descriptor contract through:
  - `src/sparse_chol_csc_internal.h`
- Added the directly forced proof follow-through in:
  - `tests/test_chol_csc.c`
- Added the directly forced build/link follow-through for non-Apple POSIX
  `dlopen` resolution in:
  - `Makefile`
  - `CMakeLists.txt`
- Ran the required implementation-day validation queue:
  - `make format`
  - `make lint`
  - `make test`

### Findings
- Sprint 92 Day 6 landed as one bounded shared-dense-owner batch rather than a
  broad dense-family rewrite:
  - `src/sparse_dense.c` now owns one wider optional external
    BLAS/LAPACK-class dense-kernel seam
  - builtin kernels remain the default and authoritative fallback
  - the first adopted consumer remains the existing Cholesky supernodal dense
    backend descriptor path
- The backend-name/runtime contract is now sharper without reopening the
  public product contract:
  - builtin remains `builtin`
  - optional Apple provider remains visible as `accelerate`
  - optional non-Apple or generic external provider is surfaced as
    `blas-lapack`
  - invalid or unavailable external requests still fail closed to builtin
- The directly forced support-only movement stayed bounded:
  - no LDL^T adoption movement
  - no QR adoption movement
  - no benchmark/reporting widening
  - no README / install / maintainer wording follow-through
  - no install/export or workflow-surface movement
- The only validation interruption was one local lint correction in the new
  preprocessor style inside `src/sparse_dense.c`; after that fix, the full
  queue passed from the top.

### Validation
- `make format`
- `make lint`
- `make test`

### Day 6 Exit State
- Sprint 92 now has one landed optional portable dense-backend seam in the
  shared dense owner with builtin fallback still authoritative.
- The first adoption stays bounded to the Cholesky supernodal dense path and
  its proof/build follow-through.
- Day 7 can now rerank from a real landed backend seam instead of from the Day
  5 design contract alone.

## Day 7 - Post-Landing Audit & Rerank

### Goal
Re-rank the remaining backend-maturity work after the Day 6 landing so Sprint
92's second implementation center is chosen from the live post-landing tree
rather than from the original hotspot audit alone.

### Actions
- Re-read the Sprint 92 Day 7 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 3 hotspot audit and the Day 6 backend-batch artifact.
- Rechecked the live backend-owner split after Day 6 across:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_internal.h`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
- Rechecked the remaining backend-name and environment-contract surfaces to
  distinguish:
  - shared-seam work already landed
  - residual family-local backend seams
  - later proof/benchmark/build wording follow-through
- Wrote the Day 7 rerank artifact and fixed the exact Day 8 design center in
  writing.

### Findings
- The Day 6 landing closed the strongest first Sprint 92 contradiction:
  - the shared dense owner no longer lacks a real bounded optional portable
    backend seam
  - Cholesky no longer depends only on a narrower family-local acceleration
    pocket to expose optional dense-kernel acceleration
  - the backend runtime contract is now visible through one shared
    `builtin` / `accelerate` / `blas-lapack` naming surface
- That changes the ranked remaining backend map to:
  - strongest first target now:
    - direct-family backend adoption convergence centered on
      `src/sparse_ldlt_csc.c`
  - strongest second target now:
    - QR and later dense-consumer adoption only after the LDLT seam is
      reconciled with the shared dense owner
  - strongest third target now:
    - bounded benchmark/proof observability follow-through once the strongest
      direct-family adopters actually share the widened seam
  - strongest support-only but real target now:
    - build/package/support wording only where later observability work truly
      changes the maintained backend contract
- The strongest remaining contradiction is now explicit:
  - `src/sparse_ldlt_csc.c` still owns a bounded family-local
    Accelerate-only dense-factor selection seam
  - that means the repo now has one widened shared dense seam on the
    Cholesky side and one still narrower family-local seam on the LDLT side
  - this duplication now outranks benchmark wording or build/package wording
    because it still limits backend maturity on one of the strongest direct
    consumers
- The useful rerank change is therefore:
  - Sprint 92 should stay code-owned for its second implementation center
  - it should not shift yet to benchmark/proof owners
  - it should not widen to QR before LDLT stops lagging the new shared dense
    contract
- The strongest later-but-not-next seam is now explicit too:
  - `src/sparse_qr.c` still reads as a later adopter rather than the next
    forced backend consumer
  - `benchmarks/bench_refactor_csc.c` remains real Sprint 92 work, but only
    after the strongest direct-family adoption seam actually lands

### Validation
- Re-read the Sprint 92 Day 7 plan contract.
- Re-read the Day 3 hotspot audit and Day 6 landed-batch artifact.
- Rechecked the live backend-owner and proof-owner split across the shared
  dense owner, Cholesky, LDLT, and benchmark surfaces.

### Day 7 Exit State
- The post-Day-6 contradiction map is now explicit.
- Sprint 92's second implementation center stays code-owned and is fixed to
  the LDLT backend-adoption seam.
- Day 8 can now define one exact bounded LDLT adoption contract from the live
  post-Day-6 tree.

## Day 8 - Solver Adoption Follow-Through Design

### Goal
Define one exact bounded Day 9 adoption contract that converges the strongest
remaining LDLT dense-backend seam onto the widened shared dense owner without
reopening broad direct-family or benchmark work.

### Actions
- Re-read the Sprint 92 Day 8 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 7 rerank and the Day 5 backend contract.
- Re-read the live LDLT backend-selection and contract surfaces:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
- Rechecked which adjacent surfaces are truly forced by LDLT backend
  convergence versus which remain later:
  - proof-owner tests
  - benchmark owners
  - public/support wording
- Wrote the Day 8 design artifact and fixed the exact Day 9 center in
  writing.

### Findings
- Sprint 92 now has one exact second implementation contract:
  - required Day 9 center:
    - `src/sparse_ldlt_csc.c`
  - directly forced support-only follow-through only if the Day 9 contract
    truly needs them:
    - `src/sparse_ldlt_csc_internal.h`
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
  - strongest later surfaces only if LDLT adoption exposes a real shared seam
    that needs them:
    - `benchmarks/bench_refactor_csc.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The exact Day 9 adoption target is now explicit:
  - stop treating LDLT dense backend selection as a family-local
    Accelerate-only side path
  - converge LDLT onto the widened shared builtin-vs-portable backend reading
    already landed on the Cholesky side
  - preserve builtin fallback truth and avoid widening claims beyond the
    landed shared seam
- The strongest Day 8 clarification is now explicit:
  - Day 9 should not become a generic LDLT numeric rewrite
  - it should not widen to QR
  - it should not shift early to benchmark/reporting work
  - it should not reopen package or workflow wording detached from a real
    adoption movement

### Validation
- Re-read the Sprint 92 Day 8 plan contract.
- Re-read the Day 7 rerank artifact and the Day 5 backend contract.
- Rechecked the live LDLT backend-selection, internal contract, and proof
  surfaces.

### Day 8 Exit State
- Sprint 92 now has one exact bounded LDLT backend-adoption contract.
- Day 9 is fixed to `src/sparse_ldlt_csc.c` with only tightly bounded test and
  internal-header follow-through if truly forced.
- Later benchmark, support-surface, and QR work remains sequenced behind real
  LDLT adoption movement.

## Day 9 - Solver Adoption Follow-Through Batch

### Goal
Land the bounded LDLT backend-adoption batch so the strongest remaining
direct-family dense consumer converges onto the widened shared builtin-vs-
portable backend seam without reopening broader family or support-surface
work.

### Actions
- Re-read the Sprint 92 Day 9 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 8 adoption design and the Day 6 shared-backend landing.
- Updated the live LDLT backend-selection seam in:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
- Added retained proof-owner follow-through in:
  - `tests/test_ldlt.c`
- Ran the full implementation-day validation queue:
  - `make format`
  - `make lint`
  - `make test`
- Wrote the Day 9 batch artifact after validation passed.

### Findings
- Sprint 92 Day 9 landed one bounded LDLT backend-adoption batch:
  - required implementation center:
    - `src/sparse_ldlt_csc.c`
  - directly forced follow-through:
    - `src/sparse_ldlt_csc_internal.h`
    - `tests/test_ldlt.c`
- The live LDLT backend reading now converges onto the widened shared contract:
  - builtin remains the default and authoritative fallback path
  - LDLT can now consume the bounded optional external backend lane under the
    same runtime contract already visible on the Cholesky side
  - the retained backend names are now:
    - `builtin`
    - `accelerate`
    - `blas-lapack`
- The retained proof-owner follow-through stayed bounded:
  - `tests/test_ldlt.c` now proves the `external` env-contract path end-to-end
    without widening to broader LDLT CSC or benchmark owners
- The strongest Day 9 coherence win is now explicit:
  - Sprint 92 no longer carries one widened direct-family backend story on
    the Cholesky side and one narrower family-local backend story on the
    LDLT side
  - the strongest direct-family dense adopters now share one bounded
    builtin-vs-portable backend reading
- The Day 8 boundary held:
  - no generic LDLT numeric rewrite
  - no QR adoption
  - no benchmark/reporting widening
  - no README / install / maintainer wording changes

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- One local lint-style preprocessor-guard issue surfaced while widening the
  LDLT-side external-backend seam; it was corrected, and the full validation
  queue then passed again from the top.

### Day 9 Exit State
- Sprint 92 has completed its bounded LDLT backend-adoption batch.
- LDLT now shares the widened shared dense backend contract rather than
  lagging on a family-local Accelerate-only side path.
- Day 10 can now rerank the remaining backend, proof, benchmark, and support
  surfaces from the live post-Day-9 tree.

## Day 10 - Observability and Proof Design

### Goal
Define one exact bounded Day 11 observability center so Sprint 92 can expose
backend selection, fallback behavior, and bounded repeated-run evidence for the
widened direct-family dense backend surface without reopening broad solver,
proof, or package work.

### Actions
- Re-read the Sprint 92 Day 10 contract in
  `docs/planning/EPIC_9/SPRINT_92/PLAN.md`.
- Re-read the Day 9 landing and the Day 8 adoption design.
- Re-read the live benchmark and adjacent proof/support surfaces:
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/README.md`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- Rechecked whether the remaining evidence gap is benchmark-owned, proof-owned,
  or one truly bounded combined follow-through.
- Wrote the Day 10 design artifact and fixed the exact Day 11 center in
  writing.

### Findings
- Sprint 92 now has one exact Day 11 observability contract:
  - required Day 11 center:
    - `benchmarks/bench_refactor_csc.c`
  - directly forced support-only follow-through only if the Day 11 contract
    truly needs them:
    - `benchmarks/README.md`
    - `README.md`
    - `docs/maintainer_guide.md`
  - retained adjacent proof owners only if the benchmark lane exposes a real
    contradiction they must cover:
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
- The strongest Day 10 rerank is now explicit:
  - the retained LDLT proof owner already proves the widened env contract
    cleanly enough
  - the remaining Sprint 92 evidence gap is benchmark-side observability, not
    baseline LDLT correctness
  - the repeated-run LDLT benchmark owner still does not visibly report which
    dense backend actually ran or whether a requested external path fell back
    to builtin
- Sprint 92's Day 11 reporting shape is now frozen around:
  - selected dense backend visibility
  - builtin fallback visibility when external acceleration is requested but not
    selected
  - bounded repeated-run LDLT evidence through
    `bench_refactor_csc --indefinite-kkt`
- The useful boundary call is now explicit:
  - Day 11 should stay benchmark-owned
  - Day 11 should not reopen QR adoption
  - Day 11 should not widen canonical reporting or package/build surfaces
    unless the benchmark contract truly changes them
  - Day 11 should not try to prove portable external-backend superiority

### Validation
- Re-read the Sprint 92 Day 10 plan contract.
- Re-read the Day 8 design and Day 9 landed-batch artifact.
- Rechecked the live benchmark, proof, and support-owner split around the
  widened LDLT backend seam.

### Day 10 Exit State
- Sprint 92 now has one exact bounded observability center.
- Day 11 is fixed to `benchmarks/bench_refactor_csc.c`.
- Support-only wording and build/package movement remain sequenced behind real
  output or command-contract changes.
