# Sprint 81 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 81 baseline for Epic 8 by grounding the sprint in
the validated Sprint 80 close state, the live Epic 8 project-plan section, and
the current permanent validation, product/storage, benchmark, and proof-owner
surfaces rather than another generic implementation start.

### Actions
- Re-read the Sprint 81 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 81 day-by-day plan in `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the strongest Sprint 80 closeout context:
  - `docs/planning/EPIC_8/SPRINT_80/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_80/RETROSPECTIVE.md`
- Restored the Epic 8 planning tree from `origin/sprint-80` because the
  current `master` branch did not carry `docs/planning/EPIC_8/`.
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 81
  touch surfaces across product/storage, direct-workflow, proof-owner, and
  support surfaces.
- Opened Sprint 81 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 81 starts from the same strongest local reviewed baseline Sprint 80
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 81 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 81 is not a broad “storage rewrite” sprint. Its highest value is one
  bounded product/storage modernization package centered on:
  - storage/conversion audit
  - compressed-first architecture design
  - construction/import landing
  - repeated-run workflow convergence
  - focused proof and benchmark follow-through
  - docs/examples/header alignment only where the implementation truly moves
    the contract
- The strongest likely Sprint 81 product/storage and proof surfaces are
  explicit from the live tree:
  - `include/sparse_matrix.h` = `610`
  - `src/sparse_matrix.c` = `1125`
  - `src/sparse_cholesky.c` = `615`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_qr.c` = `1563`
  - `tests/test_sparse_matrix.c` = `1071`
  - `tests/test_integration.c` = `2689`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/README.md` = `393`
  - `README.md` = `1050`
  - `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`
- The strongest Day 1 clarification is now fixed:
  - Sprint 81 should not reopen Sprint 80’s baseline, oracle, benchmark, or
    non-goal contract package
  - it should first reduce the linked-list-first product/storage ceiling on the
    highest-value seams only
- The preserved Sprint 81 non-goal pressure is explicit before Day 2:
  - no backend/performance lane spill
  - no capability-surface widening
  - no broad package/platform reopening
  - no generic whole-library workflow rewrite
  - no broad public API redesign hidden inside storage cleanup

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live product/storage, proof-owner, and support-surface hotspot
  map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 81 no longer starts from generic Epic 8 planning prose.
- The baseline, storage/conversion audit, compressed-first design,
  construction/import landing, workflow-convergence, and focused proof
  follow-through workstreams are fixed in writing.
- The strongest likely Sprint 81 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Reconfirm the Sprint 81 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, canonical benchmark/report command surfaces, and install/export proof
owners before any product/storage modernization batch lands.

### Actions
- Re-read the Sprint 81 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest reviewed proof-owner binaries and representative
  examples most likely to matter early in Sprint 81:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical report command surface with:
  - `make -n bench-canonical-report`
- Reconfirmed the root `build/` canonical benchmark emitters consumed by that
  report path:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 81 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 81 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial storage/workflow or architecture batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Sprint-81 proof
  surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
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
- The strongest current proof and truth-surface split is now fixed for Sprint
  81’s first lane:
  - reviewed CMake proof-owner binaries and representative examples remain the
    main executable truth surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 81 rerun set is now fixed around the likely touched
  storage/workflow seams:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Sprint 81.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 81 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the storage/conversion
  hotspot audit begins.

## Day 3 - Storage / Conversion Hotspot Audit

### Goal
Reduce Sprint 81's broad product/storage problem to one ranked live
contradiction map grounded in the current tree so later boundary and design
work can choose one bounded compressed-first seam instead of another generic
"storage modernization" bucket.

### Actions
- Re-read the Sprint 81 Day 3 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the Sprint 80 contradiction map in
  `docs/planning/EPIC_8/SPRINT_80/artifacts/day3-live-competitive-gap-inventory.md`.
- Rechecked the strongest likely Sprint 81 product/storage surfaces and current
  live `wc -l` measurements:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `src/sparse_cholesky.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
  - `src/sparse_analysis.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
  - `examples/example_analysis.c`
- Re-read the public matrix-shell contract in `include/sparse_matrix.h`,
  especially the linked-list shell framing and the repeated-run direct-workflow
  split against `sparse_analysis.h`.
- Re-read the main shell lifecycle and mutation core in `src/sparse_matrix.c`,
  especially:
  - pool allocation
  - shell-buffer lifecycle
  - `sparse_create`
  - `sparse_copy`
  - linked-list mutation/publication helpers
- Re-read the repeated-run factorization handoff in `src/sparse_analysis.c`,
  especially `sparse_factor_numeric(...)` and its small-problem
  `build_permuted_copy(...)` routes for Cholesky, LU, and LDL^T.
- Rechecked the strongest one-shot direct wrapper surface in
  `src/sparse_cholesky.c` to confirm whether it remains the first contradiction
  center or only support-tier context.

### Findings
- Sprint 81's broad storage/workflow problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - public mutable matrix-shell and mutation/publication center
  - strongest second target:
    - repeated-run direct-workflow factor path that still rebuilds linked-list
      permuted copies on the small-problem lane
  - strongest third target:
    - family-local one-shot direct wrappers that still keep the linked-list
      shell as the visible compatibility center
  - strongest support-only but real target:
    - proof and benchmark surfaces that currently normalize the linked-list
      shell rather than a compressed-first reading
- The strongest current contradiction center is now explicit:
  - `include/sparse_matrix.h` still describes the public matrix API as the
    orthogonal linked-list shell
  - `src/sparse_matrix.c` still concentrates the mutable construction,
    insertion, copy, transpose, and shell lifecycle around pointer-heavy row
    and column walks plus slab-node allocation
  - that makes the highest-value Sprint 81 first move the matrix-shell
    construction/import/publication seam itself, not a later wrapper-only
    cleanup
- The strongest second contradiction is also explicit now:
  - `src/sparse_analysis.c` already owns the repeated-run direct path, but its
    small-problem factorization lanes still drop back through
    `build_permuted_copy(...)` into linked-list-first shells before factoring
  - that means Sprint 81 should treat repeated-run workflow convergence as a
    likely second batch, not the first boundary center
- The strongest third contradiction remains lower-order:
  - `src/sparse_cholesky.c`, `src/sparse_ldlt.c`, and related one-shot direct
    wrappers still keep the linked-list shell as the compatibility owner
  - but those surfaces now read as support/follow-through context rather than
    the first batch center
- The strongest proof-tier context is now fixed too:
  - `tests/test_sparse_matrix.c` is the family-local shell/lifecycle proof owner
  - `tests/test_integration.c` is the public repeated-run and cross-workflow
    proof owner
  - `benchmarks/bench_refactor_csc.c` is the strongest benchmark-side
    measurability surface for repeated-run direct workflows
- The useful Day 3 clarification is now explicit:
  - Sprint 81 should not start by rewriting every direct-family wrapper
  - it should first narrow the matrix-shell construction/import/publication
    contradiction
  - the repeated-run direct path should remain the strongest second seam after
    the first landing, not before it

### Validation
- Re-read the strongest product/storage and repeated-run workflow surfaces
  directly.
- Rechecked the live `wc -l` hotspot map against the Sprint 80 contradiction
  order.
- Reconfirmed that the first contradiction center is still storage/product
  rather than backend, capability, package, or support-only drift.

### Day 3 Exit State
- Sprint 81 now has one ranked live storage/workflow contradiction map.
- The first implementation center is fixed to the public matrix-shell
  construction/import/publication seam unless Day 4 boundary work finds a
  stronger contradiction.
- The repeated-run direct path is fixed as the strongest likely second seam,
  not the first landing center.
