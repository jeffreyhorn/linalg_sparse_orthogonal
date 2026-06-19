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

## Day 4 - First Storage Boundary Freeze

### Goal
Fix the first bounded Sprint 81 implementation fence so the coming
compressed-first landing moves one coherent product/storage seam rather than
sprawling into repeated-run workflow, direct-family wrapper, or broad support
surface churn.

### Actions
- Re-read the Sprint 81 Day 4 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the Day 3 storage/workflow contradiction map in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day3-storage-conversion-hotspot-audit.md`.
- Rechecked the public matrix-shell owner surfaces:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- Rechecked the strongest second-tier repeated-run workflow owner surface:
  - `src/sparse_analysis.c`
- Rechecked the strongest likely support-only proof and measurement surfaces:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
- Rechecked compact support-surface context only for possible wording follow-through:
  - `README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`

### Findings
- Sprint 81 now has one explicit first implementation fence instead of a
  generic storage-modernization backlog:
  - required first landing:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - support only if the first landing truly forces it:
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
    - `benchmarks/bench_refactor_csc.c`
    - `README.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - explicitly deferred from the first landing:
    - `src/sparse_analysis.c`
    - `src/sparse_cholesky.c`
    - `src/sparse_ldlt.c`
    - `src/sparse_qr.c`
    - broad direct-family wrapper cleanup
    - repeated-run workflow convergence as a first-batch center
- The strongest Day 4 clarification is now fixed:
  - the best first Sprint 81 move is the public matrix-shell construction,
    import, and publication owner
  - the repeated-run direct path remains the strongest second seam, not the
    first implementation center
  - tests and benchmark surfaces are proof/measurement support only unless the
    first landing truly changes behavior on those seams
- The preserved first-batch non-goal fence is explicit now:
  - no broad API redesign
  - no backend or capability reopening
  - no generic whole-library workflow rewrite
  - no hidden escalation into repeated-run architecture cleanup in the first
    batch
  - no broad support-surface churn without an implementation-forced reason

### Validation
- Re-read the first-tier and second-tier storage/workflow owner surfaces
  directly.
- Rechecked the strongest proof-owner and benchmark context against the Day 3
  ranking.
- Reconfirmed that the public matrix-shell seam remains the highest-value first
  landing center.

### Day 4 Exit State
- Sprint 81 now has one explicit first implementation boundary.
- The first batch is fixed to the public matrix-shell owner surfaces.
- Day 5 can define one bounded compressed-first implementation contract without
  reopening the storage/workflow ranking.

## Day 5 - Compressed-First Architecture Design

### Goal
Define the bounded Sprint 81 implementation contract so the first landing
reduces linked-list-first construction/import tax without widening into the
repeated-run direct path, broad wrapper cleanup, or general API redesign.

### Actions
- Re-read the Sprint 81 Day 5 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the fixed Day 4 boundary in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day4-first-storage-boundary.md`.
- Rechecked the public matrix-shell ownership surface in:
  - `include/sparse_matrix.h`
- Rechecked the first implementation-center logic and helper clustering in:
  - `src/sparse_matrix.c`
  - `src/sparse_matrix_internal.h`
- Rechecked the strongest second-tier repeated-run workflow seam in:
  - `src/sparse_analysis.c`
- Rechecked the strongest support-only proof and benchmark context:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`

### Findings
- Sprint 81 now has one explicit first implementation contract:
  - required implementation center:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - support only if the first batch truly forces it:
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
    - `benchmarks/bench_refactor_csc.c`
    - `README.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
- The Day 5 ownership split is now fixed:
  - compressed-first construction/import owner:
    - `src/sparse_matrix.c`
    - specifically the shell lifecycle plus Matrix Market load/build paths
  - linked-list compatibility shell owner:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
    - retained as the mutable compatibility shell, not the only permanent
      product reading
  - conversion/publication owner:
    - `src/sparse_matrix.c`
    - especially copy, transpose, save/export, and shell publication paths
  - repeated-run workflow reuse owner, but not in the first batch:
    - `src/sparse_analysis.c`
- The strongest Day 5 compatibility reading is now explicit:
  - the first landing should preserve the existing public `SparseMatrix`
    compatibility shell for callers
  - it should reduce linked-list-first tax by making construction/import and
    publication read more like a bounded compressed-first seam internally
  - it should not promise that repeated-run direct workflows are converged in
    the same batch
- The preserved first-batch non-goal fence is explicit too:
  - no broad public API redesign
  - no repo-wide compressed-format rewrite
  - no reopening of direct-family wrapper cleanup
  - no hidden escalation into `src/sparse_analysis.c`
  - no forced docs/examples/header churn unless the implementation truly moves
    the contract

### Validation
- Re-read the public matrix-shell construction/import/publication surface
  directly.
- Rechecked the small-problem repeated-run direct fallback in
  `sparse_factor_numeric(...)` to confirm it remains the strongest second seam.
- Reconfirmed that the first landing can stay inside the public matrix-shell
  owner without reopening the Day 4 boundary.

### Day 5 Exit State
- Sprint 81 now has one explicit compressed-first implementation contract.
- Ownership between the matrix-shell first landing and later repeated-run
  workflow convergence is clear.
- Day 6 can land one bounded construction/import batch without reopening
  design questions.

## Day 6 - Construction / Import Batch 1

### Goal
Land the first bounded compressed-first construction/import seam inside the
public matrix-shell owner without widening into repeated-run workflow
convergence or broader direct-family wrapper cleanup.

### Actions
- Implemented a bounded bulk-build helper in:
  - `src/sparse_matrix.c`
- Re-routed the strongest construction/import/publication paths through that
  helper:
  - `sparse_copy(...)`
  - `sparse_transpose(...)`
  - `sparse_load_mm(...)`
- Tightened the public matrix-shell contract wording in:
  - `include/sparse_matrix.h`
- Added focused proof for duplicate Matrix Market coordinate overwrite/removal
  semantics in:
  - `tests/test_sparse_matrix.c`
- Preserved the repeated-run direct seam untouched in:
  - `src/sparse_analysis.c`

### Findings
- The Day 6 landing stayed inside the Day 5 fence:
  - required implementation center:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - forced proof follow-through:
    - `tests/test_sparse_matrix.c`
- The main Day 6 result is now explicit:
  - the matrix shell has one bounded compressed-first internal build seam
  - `sparse_copy(...)`, `sparse_transpose(...)`, and `sparse_load_mm(...)`
    no longer rebuild matrices through repeated `sparse_insert(...)` row/column
    search walks
  - Matrix Market import still preserves the visible last-write-wins
    duplicate-entry contract, including zero-as-removal behavior
  - the public `SparseMatrix` compatibility shell remains intact for callers
- The preserved fence stayed intact:
  - no public API redesign
  - no repo-wide compressed-format rewrite
  - no hidden escalation into `src/sparse_analysis.c`
  - no repeated-run workflow convergence hidden inside the first batch

### Validation
- Ran:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- All passed.
- Reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 486.07 sec`
- The focused new proof also passed:
  - `test_sparse_matrix` retained the new
    `test_load_mm_duplicate_last_write_wins`

### Day 6 Exit State
- Sprint 81 now has one landed compressed-first construction/import seam.
- The public matrix-shell owner is narrower and more deliberate.
- Day 7 can audit the post-landing rerank without reopening the first-batch
  contract.

## Day 7 - Post-Landing Audit and Rerank

### Goal
Re-rank the strongest remaining Sprint 81 product/storage contradiction after
the Day 6 construction/import landing so the sprint does not blindly repeat
another same-family matrix-shell batch without evidence.

### Actions
- Re-read the Sprint 81 Day 7 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the landed Day 6 implementation artifact in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day6-construction-import-batch1.md`.
- Rechecked the strongest remaining repeated-run workflow owner seam in:
  - `src/sparse_analysis.c`
- Rechecked the strongest support proof and measurement surfaces for that seam:
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
- Rechecked the strongest support-only contract/context surfaces:
  - `include/sparse_analysis.h`
  - `README.md`
  - `docs/maintainer_guide.md`

### Findings
- The Day 6 landing closed the strongest first implementation contradiction:
  - `include/sparse_matrix.h` and `src/sparse_matrix.c` no longer read like
    the strongest remaining Sprint 81 seam
  - a second immediate matrix-shell construction/import batch is not the
    highest-value next move
- The strongest remaining seam has now shifted to repeated-run workflow
  convergence:
  - required next landing center:
    - `src/sparse_analysis.c`
  - strongest support-only proof and benchmark follow-through:
    - `tests/test_integration.c`
    - `benchmarks/bench_refactor_csc.c`
  - support-only contract wording if the next batch truly forces it:
    - `include/sparse_analysis.h`
    - `README.md`
    - `docs/maintainer_guide.md`
- The strongest Day 7 clarification is now explicit:
  - the next contradiction is not public shell construction/import anymore
  - it is the smaller-problem repeated-run direct path that still falls back
    through `build_permuted_copy(...)` inside `sparse_factor_numeric(...)`
  - publication/writeback follow-through and support-surface alignment remain
    real, but they are weaker than the repeated-run convergence seam
- Later deferred work stays fixed too:
  - another broad `src/sparse_matrix.c` cleanup pass
  - direct-family wrapper cleanup in `src/sparse_cholesky.c`,
    `src/sparse_ldlt.c`, or `src/sparse_qr.c`
  - broader docs/examples churn without an implementation-forced reason

### Validation
- Re-read the landed Day 6 seam against the remaining direct-workflow owner.
- Reconfirmed that the small-problem repeated-run path still goes through
  `build_permuted_copy(...)` for Cholesky, LU, and LDL^T numeric factoring.
- Rechecked the strongest proof-owner and benchmark context for the repeated-run
  lane.

### Day 7 Exit State
- Sprint 81 now has one explicit strongest remaining seam.
- Day 8 is fixed to the repeated-run workflow convergence design center.
- The support-only follow-through map is explicit before the next design pass.

## Day 8 - Workflow Convergence Design

### Goal
Define one bounded repeated-run workflow convergence contract for Sprint 81 so
the next implementation batch reduces one-shot versus repeated-run ambiguity on
the highest-value direct-workflow seam without widening into another matrix
shell pass, broad solver-family rewrite, or support-surface churn.

### Actions
- Re-read the Sprint 81 Day 8 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the Day 7 rerank in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day7-post-landing-audit-and-rerank.md`.
- Re-read the strongest remaining repeated-run owner surface in:
  - `src/sparse_analysis.c`
  - especially `build_permuted_copy(...)`, `factor_cholesky_with_analysis_csc`,
    `factor_ldlt_with_analysis_csc`, and `sparse_factor_numeric(...)`
- Re-read the strongest proof-owner and benchmark follow-through context:
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
- Re-read likely support-only contract/context surfaces only for forced
  follow-through:
  - `include/sparse_analysis.h`
  - `README.md`
  - `docs/maintainer_guide.md`

### Findings
- Sprint 81 now has one exact second implementation contract:
  - required Day 9 center:
    - `src/sparse_analysis.c`
  - strongest proof/measurement follow-through only if the implementation
    truly forces it:
    - `tests/test_integration.c`
    - `benchmarks/bench_refactor_csc.c`
  - support-only wording only if the implementation truly changes the public
    reading:
    - `include/sparse_analysis.h`
    - `README.md`
    - `docs/maintainer_guide.md`
- The exact Day 9 seam is now fixed:
  - reduce the smaller-problem repeated-run direct ambiguity inside
    `sparse_factor_numeric(...)`
  - specifically the Cholesky and LDL^T branches that still fall back through
    `build_permuted_copy(...)` before factoring
  - keep the batch centered on working-copy preparation and repeated-run
    factor publication, not on another public matrix-shell rewrite
- The strongest useful Day 8 clarification is explicit now:
  - LU also still uses `build_permuted_copy(...)`, but it is not the best next
    landing center
  - widening the batch to LU would turn Sprint 81 Day 9 into a broader
    solver-family architecture rewrite instead of one bounded repeated-run
    convergence pass
  - the highest-value next move is therefore Cholesky plus LDL^T convergence
    first, because those lanes already have stronger analysis-backed CSC-aware
    structure and stronger public repeated-run proof/benchmark context
- The preserved second-batch fence is explicit too:
  - no reopening of the Day 6 matrix-shell construction/import batch
  - no broad direct-family wrapper cleanup in `src/sparse_cholesky.c`,
    `src/sparse_ldlt.c`, or `src/sparse_qr.c`
  - no generic repeated-run architecture rewrite
  - no backend, capability, package, or workflow-lane spill
  - no support-surface churn unless the implementation truly forces it

### Validation
- Re-read the live repeated-run owner surface in `src/sparse_analysis.c`.
- Reconfirmed that the small-problem repeated-run Cholesky, LU, and LDL^T
  branches still go through `build_permuted_copy(...)`, but that Cholesky and
  LDL^T are the stronger bounded next seam.
- Rechecked the strongest public repeated-run proof owner in
  `tests/test_integration.c` and the strongest benchmark-side measurement owner
  in `benchmarks/bench_refactor_csc.c`.

### Day 8 Exit State
- Sprint 81 now has one explicit repeated-run workflow convergence contract.
- The exact Day 9 touch set is fixed to `src/sparse_analysis.c`, with tests,
  benchmark, and wording surfaces support-only unless forced.
- Day 9 can land one bounded workflow-convergence batch without reopening
  matrix-shell, wrapper, or support-surface drift.

## Day 9 - Workflow Convergence Batch

### Goal
Land one bounded repeated-run workflow convergence batch so the strongest
remaining Sprint 81 seam stops treating smaller Cholesky and LDL^T repeated-run
numeric factoring as a linked-list-first fallback path.

### Actions
- Re-read the Sprint 81 Day 9 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the Day 8 contract in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day8-workflow-convergence-design.md`.
- Update the repeated-run factoring owner in `src/sparse_analysis.c`:
  - keep Cholesky repeated-run numeric factoring on the
    analysis-backed CSC-aware path for all problem sizes
  - keep LDL^T repeated-run numeric factoring on the same bounded CSC-aware
    path for all problem sizes
  - preserve the symmetric-direct-family repeated-run input guard so failed
    public refactors do not silently replace old factors
- Extend the strongest public repeated-run proof owner in
  `tests/test_integration.c` with focused below-threshold same-pattern
  convergence tests for Cholesky and LDL^T.
- Reconcile the strongest benchmark-side support wording in
  `benchmarks/bench_refactor_csc.c` so it matches the landed shared repeated-run
  path.
- Run the required validation set for a substantial `*.c` implementation batch:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- Re-run the reviewed CMake `ctest` surface directly to capture the exact
  retained timing anchor.

### Findings
- The bounded Day 9 workflow-convergence batch landed in:
  - `src/sparse_analysis.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
- The main implementation result is now explicit:
  - repeated-run public Cholesky factoring no longer drops through the
    smaller-problem linked-list `build_permuted_copy(...)` fallback
  - repeated-run public LDL^T factoring no longer drops through the same
    smaller-problem linked-list fallback
  - the shared repeated-run owner now keeps those lanes on the
    analysis-backed CSC-aware path for all problem sizes
- The important Day 9 safeguard was preserved too:
  - symmetric direct repeated-run inputs still reject non-symmetric matrices
    before old factors are replaced
  - that keeps the public failure-preserves-old-factors reading intact for the
    Cholesky / LDL^T analysis path
- Focused public proof landed exactly where Day 8 said it should:
  - `test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_cholesky`
  - `test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_ldlt`
  - `./build/quality-review-cmake/test_integration` retained `53 / 53`
- The benchmark surface stayed support-only:
  - `benchmarks/bench_refactor_csc.c` only needed wording follow-through so the
    comment no longer describes the old linked-list-side cost structure as the
    shared repeated-run path
- The Day 8 preserved fence held:
  - no LU widening
  - no `src/sparse_matrix.c` reopening
  - no wrapper-family cleanup in `src/sparse_cholesky.c`, `src/sparse_ldlt.c`,
    or `src/sparse_qr.c`
  - no support-surface churn in headers or docs

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `make quality-review-full` passed.
- Reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 424.67 sec`
- Focused public repeated-run proof also passed:
  - `./build/quality-review-cmake/test_integration` = `53 / 53`

### Day 9 Exit State
- Sprint 81 has now closed its strongest remaining repeated-run convergence
  contradiction.
- The public matrix-shell and repeated-run direct lanes now read more
  consistently as a bounded compressed-first modernization path.
- The next rerank can now judge whether follow-through pressure shifts to proof,
  benchmark measurability, or residual support-surface drift.

## Day 10 - Proof and Benchmark Follow-Through Design

### Goal
Fix the exact proof, benchmark, header, and support-surface follow-through
required after the Day 6 and Day 9 implementation batches without widening
Sprint 81 into generic docs churn or another implementation pass.

### Actions
- Re-read the Sprint 81 Day 10 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the landed implementation and validation notes from:
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day6-construction-import-batch1.md`
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day9-workflow-convergence-batch.md`
- Re-read the strongest likely follow-through surfaces:
  - `include/sparse_analysis.h`
  - `README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Reconfirm whether any extra proof code or benchmark logic is still missing
  after the Day 9 landing.

### Findings
- Sprint 81 now has one exact Day 11 follow-through contract:
  - required surface:
    - `include/sparse_analysis.h`
  - strongest support-only wording if the batch truly needs it:
    - `README.md`
    - `docs/maintainer_guide.md`
  - lower-value support-only surfaces that do not currently need movement:
    - `benchmarks/README.md`
    - `examples/README.md`
- The strongest current contradiction is narrow and explicit:
  - `include/sparse_analysis.h` still describes the shared Cholesky CSC
    repeated-run path as a larger-problem-only reuse lane
  - that is now stale after Day 9, because the shared repeated-run Cholesky and
    LDL^T paths both stay on the analysis-backed CSC-aware route for all
    problem sizes
- The proof and benchmark side is already in the right place:
  - `tests/test_integration.c` already owns the new below-threshold same-pattern
    Cholesky and LDL^T parity proofs
  - `benchmarks/bench_refactor_csc.c` only needed the Day 9 comment correction
  - no additional benchmark binary or proof-code follow-through is required
- The support-only docs lane is narrower than a generic cleanup pass:
  - `README.md` already stays broadly truthful, but may benefit from a bounded
    wording refresh if the public repeated-run contract reads too
    large-`n`-centric after the header fix
  - `docs/maintainer_guide.md` is the strongest policy-side support surface if
    the Day 11 wording batch needs one authoritative ownership refresh
  - `benchmarks/README.md` and `examples/README.md` already reconcile cleanly
    with the landed batch and do not currently justify edits
- The preserved Day 10 fence is explicit:
  - no more proof-code expansion
  - no more benchmark logic changes
  - no generic README/tutorial/examples sweep
  - no reopening of `src/sparse_matrix.c` or `src/sparse_analysis.c`

### Validation
- Re-read the landed implementation owner and proof surfaces.
- Reconfirmed that the only real stale public-contract wording is in
  `include/sparse_analysis.h`.
- Reconfirmed that proof ownership and benchmark measurability are already
  closed after the Day 9 landing.

### Day 10 Exit State
- Sprint 81 now knows the exact Day 11 touch set.
- The strongest required follow-through is narrowed to the public repeated-run
  header contract, with README and maintainer wording support-only.
- Day 11 can stay bounded instead of turning into a generic support-surface
  cleanup pass.

## Day 11 - Docs / Examples / Header Alignment Batch

### Goal
Land the bounded follow-through from Day 10 so the public repeated-run direct
contract reads truthfully after the Day 9 workflow-convergence batch without
spreading into a generic README, maintainer, benchmark, or example cleanup.

### Actions
- Re-read the Sprint 81 Day 11 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the Day 10 design in
  `docs/planning/EPIC_8/SPRINT_81/artifacts/day10-proof-and-benchmark-follow-through-design.md`.
- Update the required public repeated-run header surface in
  `include/sparse_analysis.h`.
- Reconfirm whether support-only surfaces truly need movement:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Run the required code-day validation set because a public `*.h` surface
  changed:
  - `make format`
  - `make lint`
  - `make test`

### Findings
- The bounded Day 11 follow-through batch landed only in:
  - `include/sparse_analysis.h`
- The required contract correction is now explicit:
  - the `sparse_factor_numeric(...)` public header block no longer describes
    the shared Cholesky repeated-run CSC-aware path as larger-problem-only
  - it now says directly that the shared Cholesky repeated-run path stays on
    the analysis-backed CSC-aware route for all problem sizes
  - it also makes the residual split clearer:
    - LDL^T remains analysis-backed CSC-aware with its documented
      pivot-prepass-conditioned fallback
    - LU remains the direct family that still delegates through the one-shot
      routine
- No support-only follow-through was actually needed:
  - `README.md` already stayed broadly truthful
  - `docs/maintainer_guide.md` already stayed broadly truthful
  - `benchmarks/README.md` already stayed aligned with the landed proof and
    benchmark ownership split
  - `examples/README.md` already stayed aligned with the landed repeated-run
    adoption split
- The Day 10 preserved fence held:
  - no new proof-code expansion
  - no benchmark logic changes
  - no generic docs/examples sweep
  - no reopening of implementation surfaces

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.

### Day 11 Exit State
- Sprint 81's public repeated-run header contract now matches the landed Day 9
  workflow behavior.
- The docs/examples/header follow-through batch stayed bounded to one required
  surface.
- Day 12 can now focus on final proof alignment instead of support-surface
  drift.

## Day 12 - Final Proof Alignment and Validation Queue

### Goal
Fix the exact Day 13 rerun set and final ownership map for Sprint 81 so the
closeout baseline is taken from one stable measured queue rather than from a
partial implementation or support-surface state.

### Actions
- Re-read the Sprint 81 Day 12 plan expectations in
  `docs/planning/EPIC_8/SPRINT_81/PLAN.md`.
- Re-read the landed Sprint 81 implementation and follow-through artifacts:
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day6-construction-import-batch1.md`
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day9-workflow-convergence-batch.md`
  - `docs/planning/EPIC_8/SPRINT_81/artifacts/day11-docs-examples-header-alignment-batch.md`
- Re-read the strongest proof, benchmark, and support owners:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `benchmarks/bench_refactor_csc.c`
  - `examples/example_analysis.c`
  - `examples/example_basic_solve.c`
  - `include/sparse_analysis.h`
- Recheck the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake`

### Findings
- No new proof code or support-surface edit is actually needed before the full
  sweep.
- The final Sprint 81 proof-owner map is now fixed explicitly:
  - `tests/test_sparse_matrix.c` owns the bounded matrix-shell
    construction/import/publication regression surface from Day 6
  - `tests/test_integration.c` owns the public repeated-run direct parity and
    failure-preservation contract, including the new below-threshold Cholesky
    and LDL^T same-pattern convergence proofs from Day 9
  - `tests/test_chol_csc.c` remains the family-local large-`n`
    analysis-backed CSC Cholesky owner and the publish-back ownership proof
    home
  - `tests/test_ldlt.c` remains the family-local LDL^T backend and
    cross-backend proof owner
  - `benchmarks/bench_refactor_csc.c` remains the benchmark-side retained
    repeated-run throughput/proof surface, not the oracle owner
  - `examples/example_analysis.c` and `examples/example_basic_solve.c` remain
    representative example-side adoption surfaces, not regression owners
- The exact Day 13 validation queue is now fixed around:
  - code-day gate:
    - `make format`
    - `make lint`
    - `make test`
  - strongest reviewed validation baseline:
    - `make quality-review-full`
    - `ctest -N --test-dir build/quality-review-cmake`
  - authoritative focused proof-owner reruns:
    - `./build/quality-review-cmake/test_sparse_matrix`
    - `./build/quality-review-cmake/test_integration`
    - `./build/quality-review-cmake/test_chol_csc`
    - `./build/quality-review-cmake/test_ldlt`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
  - touched benchmark/reporting follow-on:
    - `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- Install/export proof is not part of the Sprint 81 Day 13 queue:
  - Sprint 81 did not touch package, install, or export mechanics
  - the bounded header wording change does not justify reopening the install
    scripts on this sprint's validation queue

### Validation
- Re-read the landed implementation, proof, benchmark, and support surfaces.
- Reconfirmed the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Reconfirmed that the proof-owner map is already coherent and needs no extra
  code or docs follow-through.

### Day 12 Exit State
- Sprint 81 now has one authoritative final proof-owner map.
- The exact Day 13 rerun set is fixed before validation starts.
- Day 13 can execute from one stable measured queue without reopening support
  or implementation drift.
