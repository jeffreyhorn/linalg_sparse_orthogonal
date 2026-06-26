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
