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
