# Sprint 82 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 82 baseline for Epic 8 by grounding the sprint in
the validated Sprint 81 close state, the live Sprint 82 project-plan section,
and the current permanent validation, dense-helper, benchmark, and proof-owner
surfaces rather than another generic backend start.

### Actions
- Re-read the Sprint 82 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 82 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_82/PLAN.md`.
- Re-read the strongest Sprint 81 closeout context:
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_81/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 82
  touch surfaces across dense-helper owners, direct-family consumers,
  proof-owner tests, and support surfaces.
- Opened Sprint 82 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 82 starts from the same strongest local reviewed baseline Sprint 81
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 82 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 82 is not a broad “performance cleanup” sprint. Its highest value is
  one bounded dense/backend modernization package centered on:
  - dense hotspot profiling
  - backend ABI design
  - first optional accelerated dense-kernel integration
  - solver adoption follow-through
  - focused benchmark and differential proof
  - packaging/runtime alignment only where the implementation truly moves the
    contract
- The strongest likely Sprint 82 dense/backend and proof surfaces are explicit
  from the live tree:
  - `include/sparse_cholesky.h` = `227`
  - `include/sparse_ldlt.h` = `335`
  - `include/sparse_qr.h` = `385`
  - `include/sparse_svd.h` = `257`
  - `src/sparse_dense.c` = `633`
  - `src/sparse_chol_csc_supernodal.c` = `484`
  - `src/sparse_chol_csc.c` = `1564`
  - `src/sparse_ldlt_csc_supernodal.c` = `392`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
  - `tests/test_chol_csc.c` = `4724`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_qr.c` = `3197`
  - `tests/test_integration.c` = `2973`
  - `benchmarks/bench_chol_csc.c` = `423`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `benchmarks/README.md` = `393`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `698`
- The strongest Day 1 clarification is now fixed:
  - Sprint 82 should not reopen Sprint 80's oracle and benchmark contract
    package
  - Sprint 82 should not reopen Sprint 81's product/storage contradiction
    center
  - it should first reduce the builtin scalar dense/backend ceiling on the
    highest-value solver seams only
- The preserved Sprint 82 non-goal pressure is explicit before Day 2:
  - no capability-surface widening
  - no broad package/platform reopening
  - no fake platform or shared-library maturity claim
  - no benchmark-threshold inflation
  - no mandatory heavyweight optional-backend dependency for the default build

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live dense-helper, proof-owner, benchmark, and support-surface
  hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 82 no longer starts from generic Epic 8 planning prose.
- The baseline, dense-hotspot profiling, backend ABI design, first optional
  integration, solver adoption, focused proof, and packaging/runtime-alignment
  workstreams are fixed in writing.
- The strongest likely Sprint 82 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Reconfirm the Sprint 82 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, canonical benchmark/report command surfaces, and install/export proof
owners before any dense/backend modernization batch lands.

### Actions
- Re-read the Sprint 82 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_82/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest reviewed proof-owner binaries and representative
  examples most likely to matter early in Sprint 82:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the strongest reviewed benchmark follow-on binaries most likely to
  matter:
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- Rechecked the maintained canonical report command surface with:
  - `make -n bench-canonical-report`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 82 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 82 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial backend, solver-adoption, or package/runtime batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Sprint-82 proof
  surfaces:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
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
- The strongest current proof and truth-surface split is now fixed for Sprint
  82's first lane:
  - reviewed CMake proof-owner tests and representative examples remain the
    main executable truth surfaces
  - reviewed benchmark binaries remain benchmark-side measurability surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 82 rerun set is now fixed around the likely touched
  backend/workflow seams:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Sprint 82.
- Rechecked the strongest reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 82 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the dense-hotspot audit
  begins.

## Day 3 - Dense Hotspot Profiling Audit

### Goal
Reduce Sprint 82's broad backend problem to one ranked live contradiction map
grounded in the current dense-helper, direct-family, benchmark, and proof-owner
surfaces so later boundary and ABI work can choose one bounded accelerated
backend seam instead of another generic performance bucket.

### Actions
- Re-read the Sprint 82 Day 3 plan expectations in
  `docs/planning/EPIC_8/SPRINT_82/PLAN.md`.
- Re-read the Sprint 80 contradiction map in
  `docs/planning/EPIC_8/SPRINT_80/artifacts/day3-live-competitive-gap-inventory.md`.
- Rechecked the strongest likely Sprint 82 dense/backend surfaces and current
  live `wc -l` measurements:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
- Re-read the current dense-helper contract in `include/sparse_dense.h` and
  `src/sparse_dense.c`, especially:
  - builtin dense GEMM/GEMV ownership
  - `chol_dense_factor`
  - `chol_dense_solve_lower`
  - `chol_dense_solve_panel`
  - the current `chol_dense_kernels_t` builtin descriptor and test override
- Re-read the strongest supernodal dense consumer in
  `src/sparse_chol_csc_supernodal.c`, especially:
  - extract / eliminate-diag / eliminate-panel / writeback orchestration
  - dense-kernel descriptor consumption
  - missing-kernel backend-contract error boundary
- Re-read the strongest second-tier direct-family backend surface in
  `src/sparse_ldlt.c`, especially:
  - public backend selection
  - auto / linked-list / CSC dispatch contract
  - current backend observability semantics
- Rechecked the densest QR and SVD local-workspace paths through targeted
  searches in `src/sparse_qr.c` and `src/sparse_svd.c`.

### Findings
- Sprint 82's broad dense/backend problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - Cholesky CSC dense-kernel descriptor and supernodal consumer lane
  - strongest second target:
    - LDL^T backend/runtime parity and supernodal dense-kernel follow-through
  - strongest third target:
    - QR and SVD dense-workspace ceiling
  - strongest support-only but real target:
    - benchmark/runtime measurability and package/runtime interpretation
- The strongest current contradiction center is now explicit:
  - `src/sparse_dense.c` still owns only a builtin scalar dense-kernel surface
    for the highest-value Cholesky inner kernels
  - the current backend-aware seam is narrow:
    - one `chol_dense_kernels_t` builtin descriptor
    - one test-only override path
    - no maintained optional accelerated runtime path yet
  - `src/sparse_chol_csc_supernodal.c` is already the clearest direct-family
    consumer because it runs:
    - dense diagonal factor
    - batched panel solve
    - backend-contract failure handling
  - that makes the highest-value Sprint 82 first move the Cholesky dense-kernel
    descriptor/runtime-selection seam itself, not package wording or broader
    solver-family adoption first
- The strongest second contradiction is also explicit now:
  - `src/sparse_ldlt.c` already has solver-level backend selection and
    observability semantics
  - but it still reads as the strongest second lane rather than the first
    backend descriptor owner because the Cholesky CSC supernodal path is the
    most mature dense-kernel consumer seam today
- The strongest third contradiction remains lower-order:
  - `src/sparse_qr.c` and `src/sparse_svd.c` still carry meaningful dense
    workspace cost
  - but they do not read like the best first bounded accelerated-backend
    landing center compared with the supernodal Cholesky lane
- The strongest support-tier backend surfaces are now explicit too:
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- The useful Day 3 clarification is now fixed:
  - Sprint 82 should begin with the dense-kernel descriptor and Cholesky CSC
    supernodal consumer lane
  - LDL^T backend/runtime parity should remain the strongest second
    implementation lane
  - QR/SVD dense-workspace follow-through remains real, but it is not the first
    contradiction center

### Validation
- Re-read the dense-helper contract in `include/sparse_dense.h` and
  `src/sparse_dense.c`.
- Re-read the highest-value Cholesky CSC supernodal dense consumer in
  `src/sparse_chol_csc_supernodal.c`.
- Re-read the LDL^T backend dispatch surface in `src/sparse_ldlt.c`.
- Rechecked the strongest dense/backend hotspot map via targeted searches and
  direct `wc -l` measurement.

### Day 3 Exit State
- Sprint 82's backend problem is now reduced to one ranked live contradiction
  map.
- The strongest first implementation center is fixed to the Cholesky
  dense-kernel descriptor/runtime lane before boundary design begins.
- Lower-value QR/SVD and support-surface spillover work is separated from the
  first lane.
