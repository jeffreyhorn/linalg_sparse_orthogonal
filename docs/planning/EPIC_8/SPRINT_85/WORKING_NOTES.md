# Sprint 85 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 85 baseline for Epic 8 by grounding the sprint in
the validated Sprint 84 close state, the live Sprint 85 project-plan section,
and the current source, giant-test, benchmark, and support-surface hotspots
rather than another generic “refactor more code” restart.

### Actions
- Re-read the Sprint 85 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 85 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the strongest Sprint 84 closeout context:
  - `docs/planning/EPIC_8/SPRINT_84/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_84/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 85
  touch surfaces across shared/public headers, implementation owners,
  direct-family and iterative proof-owner tests, and support surfaces.
- Opened Sprint 85 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 85 starts from the same strongest local reviewed baseline Sprint 84
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 85 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 85 is not a generic “cleanup everything” sprint. Its highest value is
  one bounded maintainability-modernization package centered on:
  - hotspot rerank
  - decomposition / ownership architecture design
  - iterative-source cleanup
  - direct-family hotspot cleanup
  - giant-test architecture cleanup
  - proof/docs alignment
  - validation and closeout
- The strongest likely Sprint 85 implementation, proof, and support surfaces
  are explicit from the live tree:
  - shared/public contract surfaces:
    - `include/sparse_types.h` = `313`
    - `include/sparse_matrix.h` = `622`
    - `include/sparse_qr.h` = `392`
    - `include/sparse_svd.h` = `257`
    - `include/sparse_cholesky.h` = `227`
    - `include/sparse_ldlt.h` = `335`
  - strongest source hotspots:
    - `src/sparse_iterative.c` = `1985`
    - `src/sparse_chol_csc.c` = `1841`
    - `src/sparse_qr.c` = `1563`
    - `src/sparse_ldlt.c` = `1535`
  - strongest giant proof-owner tests:
    - `tests/test_chol_csc.c` = `4964`
    - `tests/test_qr.c` = `3234`
    - `tests/test_integration.c` = `3197`
    - `tests/test_ldlt.c` = `2921`
    - `tests/test_iterative.c` = `2841`
  - strongest support surfaces:
    - `README.md` = `1050`
    - `docs/maintainer_guide.md` = `726`
- The strongest Day 1 clarification is now fixed:
  - Sprint 85 should not reopen Sprint 84’s bounded assurance-owner fence as
    its first implementation center
  - Sprint 85 should first reduce review and reasoning cost on the largest
    remaining implementation and giant-test hotspots
  - it should preserve proof ownership while decomposing the touched seams
    rather than diffusing ownership across more files without a stronger
    contract
- The preserved Sprint 85 non-goal pressure is explicit before Day 2:
  - no generic assurance widening restart
  - no repo-wide architectural cleanup claim
  - no proof dilution from helper movement detached from owner boundaries
  - no benchmark-governance or example-governance drift into correctness
    ownership
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed hotspot seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared/public, implementation, proof-owner, benchmark, and
  support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 85 no longer starts from generic Epic 8 maintainability prose.
- The hotspot rerank, decomposition design, iterative cleanup, direct-family
  cleanup, giant-test cleanup, proof/docs alignment, and validation workstreams
  are fixed in writing.
- The strongest likely Sprint 85 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live proof-owner
split before Sprint 85 decomposes any large source or giant-test hotspot.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the strongest recent validation/proof-surface template from
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day2-validation-baseline-and-proof-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed proof-owner binaries for the
  early Sprint 85 lanes:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the script-owned support-proof surfaces:
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 85 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial source-hotspot, giant-test, or support-policy batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently owns the strongest early-Sprint-85 proof
  surfaces:
  - iterative, direct-family, and giant-test owners:
    - `./build/quality-review-cmake/test_iterative`
    - `./build/quality-review-cmake/test_chol_csc`
    - `./build/quality-review-cmake/test_integration`
    - `./build/quality-review-cmake/test_ldlt`
    - `./build/quality-review-cmake/test_qr`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
  - reviewed benchmark follow-on binaries:
    - `./build/quality-review-cmake/bench_refactor_csc`
    - `./build/quality-review-cmake/bench_svd`
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
  - reviewed CMake proof-owner tests and representative examples remain the
    main executable truth surfaces for Sprint 85 hotspot work
  - reviewed benchmark binaries remain measurability surfaces, not the
    canonical reporting owner
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed proof-owner tests,
  representative examples, and reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, and `tests/test_cmake_install.sh`.

### Day 2 Exit State
- Sprint 85 now has one explicit implementation-day validation contract before
  the hotspot rerank begins.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/package proof is fixed in writing.
- The highest-signal rerun set is explicit before the first hotspot-priority
  rerank.

## Day 3 - Hotspot Rerank

### Goal
Reduce Sprint 85's broad maintainability problem to one ranked live
contradiction map so the sprint can choose one bounded decomposition lane
instead of another generic “split large files” bucket.

### Actions
- Re-read the Day 3 hotspot-rerank expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the strongest recent rerank template from
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day3-differential-proof-audit.md`.
- Re-read the Sprint 84 close handoff in
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day14-closeout-and-handoff.md`
  to preserve the intended Sprint 85 ordering.
- Re-measured the strongest live source, giant-test, and support hotspots with
  direct `wc -l` checks across:
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_eigs.c`
  - `tests/test_chol_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- Rechecked the strongest live proof-owner and helper concentration signals
  with targeted `rg -n` across `Makefile`, `CMakeLists.txt`, `tests/`, `src/`,
  `include/`, `README.md`, and `docs/maintainer_guide.md` for:
  - `test_iterative`
  - `test_chol_csc`
  - `test_integration`
  - `test_ldlt`
  - `test_qr`
  - `test_svd`

### Findings
- Sprint 85's broad maintainability problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - bounded iterative-source cleanup centered on `src/sparse_iterative.c`
  - strongest second target:
    - bounded direct-family source cleanup centered on
      `src/sparse_chol_csc.c`
  - strongest third target:
    - bounded giant-test architecture cleanup centered first on
      `tests/test_chol_csc.c`
  - strongest fourth target:
    - next giant-test/source follow-through on `tests/test_qr.c`,
      `tests/test_integration.c`, `src/sparse_qr.c`, or `src/sparse_ldlt.c`
  - strongest support-only but real target:
    - maintainer/docs wording only where cleanup changes proof-owner or helper
      boundaries
- The strongest current contradiction is not merely file size in isolation:
  - `src/sparse_iterative.c` is the largest implementation hotspot at `1985`
    lines
  - it already sits on a heavily retained proof-owner seam through
    `tests/test_iterative.c`
  - it is therefore the highest-value first decomposition lane because it can
    reduce review and reasoning cost without reopening Sprint 84's bounded
    assurance-owner package
- The strongest second contradiction is direct-family source concentration:
  - `src/sparse_chol_csc.c` remains a very large implementation hotspot at
    `1841` lines
  - its proof-owner counterpart `tests/test_chol_csc.c` is the largest giant
    test in the tree at `4964` lines
  - this makes the Cholesky CSC lane the strongest second maintainability seam
    after the first iterative source move lands
- The strongest third contradiction is giant-test concentration rather than
  source concentration alone:
  - `tests/test_chol_csc.c` = `4964`
  - `tests/test_qr.c` = `3234`
  - `tests/test_integration.c` = `3197`
  - `tests/test_ldlt.c` = `2921`
  - `tests/test_iterative.c` = `2841`
  - giant proof-owner cleanup is real Sprint 85 work, but the current live map
    still favors one implementation hotspot first, not an immediate test-only
    cleanup batch
- The strongest support-only surfaces remain bounded:
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `726`
  - these should move only where landed decomposition actually changes helper,
    owner, or rerun expectations
- The strongest Day 3 clarification is now explicit:
  - Sprint 85 should start with one bounded source-owner decomposition lane,
    not another repo-wide maintainability claim
  - it should preserve proof ownership while extracting helpers and reducing
    local review burden
  - it should keep benchmark, example, and install/package surfaces outside the
    first implementation center
  - it should leave reviewed runtime-convergence and `test_reorder_nd`
    optimization pressure to Sprint 86 rather than widening Day 4 toward
    runtime work

### Validation
- Re-read the Sprint 85 Day 3 expectations and the strongest recent rerank
  template.
- Re-measured the strongest live source, giant-test, and support hotspots with
  direct `wc -l` checks.
- Rechecked proof-owner and helper concentration signals with targeted
  repository `rg -n` inspection.

### Day 3 Exit State
- Sprint 85 now has one ranked live maintainability contradiction map grounded
  in the current tree.
- The first implementation center is fixed to one bounded iterative-source
  cleanup lane.
- Later direct-family source cleanup, giant-test architecture work, and
  support-only wording follow-through are explicitly ordered behind that first
  lane.

## Day 4 - First Maintainability Boundary Freeze

### Goal
Fix the first bounded Sprint 85 maintainability implementation fence so the
next design pass can define one real extraction/decomposition contract instead
of another broad cleanup rewrite.

### Actions
- Re-read the Day 4 boundary-freeze expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 3 hotspot rerank artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day3-hotspot-rerank-audit.md`.
- Re-read the strongest recent boundary template from
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day4-first-assurance-boundary.md`.
- Reconciled the live Day 3 ordering against the Sprint 85 project-plan scope:
  - iterative source cleanup first
  - direct-family source cleanup second
  - giant-test architecture cleanup only where the first source lane truly
    forces it
- Fixed the required first landing center, the support-only touch set, and the
  preserved non-goal fence in writing.

### Findings
- Sprint 85 now has one explicit first implementation fence:
  - required first landing:
    - `src/sparse_iterative.c`
  - support only if the first landing truly forces it:
    - `tests/test_iterative.c`
    - `tests/test_iterative_handle_helpers.h`
    - `tests/test_integration.c`
    - `docs/maintainer_guide.md`
    - `README.md`
  - explicitly deferred from the first landing:
    - `src/sparse_chol_csc.c`
    - `src/sparse_qr.c`
    - `src/sparse_ldlt.c`
    - `src/sparse_eigs.c`
    - `tests/test_chol_csc.c`
    - `tests/test_qr.c`
    - `tests/test_ldlt.c`
    - generic giant-test registration cleanup as a first-batch center
    - reviewed runtime-convergence work
    - benchmark/reporting ownership changes
    - install/package/runtime maturity widening
- The strongest Day 4 clarification is now explicit:
  - the best first Sprint 85 move is one bounded iterative-source cleanup
  - direct-family source cleanup remains the strongest second seam, not part
    of the first batch center
  - giant-test architecture cleanup remains real Sprint 85 work, but only
    after the first source cleanup exposes the actual helper seam that should
    move
  - proof-owner tests and maintainer wording stay support-only unless the
    first source landing truly changes helper boundaries or rerun
    expectations
- The preserved first-batch non-goal fence is fixed now:
  - no broad algorithm rewriting detached from ownership cleanup
  - no proof dilution from moving helpers without preserving owners
  - no repo-wide “cleanup sweep” claim
  - no support-surface churn detached from a real landed hotspot seam
  - no runtime-tuning or `test_reorder_nd` work inside the first lane
  - no reopening Sprint 84's bounded assurance package as part of the first
    Sprint 85 landing

### Validation
- Re-read the Sprint 85 Day 4 boundary expectations, the Day 3 rerank
  artifact, and the strongest recent boundary template.
- Reconciled the required first landing center, support-only touch set, and
  non-goal fence against the current Sprint 85 ordering.

### Day 4 Exit State
- Sprint 85 now has one explicit first maintainability landing boundary.
- Support-only proof-owner and documentation surfaces are clearly separated
  from the batch center.
- Day 5 can design one bounded iterative extraction/decomposition contract
  instead of a broad source/test cleanup rewrite.

## Day 5 - Decomposition / Ownership Architecture Design

### Goal
Define the bounded extraction and ownership contract Sprint 85 will actually
land on the first iterative-source cleanup lane.

### Actions
- Re-read the Day 5 decomposition-design expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 4 boundary artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day4-first-maintainability-boundary.md`.
- Re-read the strongest recent architecture-design template from
  `docs/planning/EPIC_8/SPRINT_84/artifacts/day5-oracle-property-failure-path-architecture-design.md`.
- Reconciled the Day 4 first-batch fence against the current maintainability
  contradictions:
  - mixed-responsibility concentration in `src/sparse_iterative.c`
  - retained reviewed proof ownership in `tests/test_iterative.c`
  - shared lifecycle follow-through in `tests/test_integration.c`
  - later direct-family hotspot work in `src/sparse_chol_csc.c`
- Fixed the Day 5 ownership split, touch fence, and preserved first-batch
  non-goal fence in writing.

### Findings
- Sprint 85 now has one explicit first implementation contract:
  - required implementation center:
    - `src/sparse_iterative.c`
  - support only if the first batch truly forces it:
    - `tests/test_iterative.c`
    - `tests/test_iterative_handle_helpers.h`
    - `tests/test_integration.c`
    - `docs/maintainer_guide.md`
    - `README.md`
- The Day 5 ownership split is now fixed:
  - iterative-source extraction owner:
    - `src/sparse_iterative.c`
  - retained iterative reviewed proof owner after extraction:
    - `tests/test_iterative.c`
  - retained iterative test-helper owner only if extraction truly forces proof
    helper movement:
    - `tests/test_iterative_handle_helpers.h`
  - shared lifecycle/public repeated-run owner, but not first-batch adoption
    owner unless the extraction exposes a true lifecycle contradiction:
    - `tests/test_integration.c`
  - strongest next source-cleanup owner, but not first-batch owner:
    - `src/sparse_chol_csc.c`
  - support-surface wording owners only if implementation truly changes the
    maintainer rerun or owner reading:
    - `docs/maintainer_guide.md`
    - `README.md`
- The strongest Day 5 clarification is explicit now:
  - the first landing should keep the cleanup source-owned inside
    `src/sparse_iterative.c`
  - it should reduce local mixed-responsibility concentration by extracting one
    bounded internal helper seam rather than distributing logic across many new
    owners
  - it should preserve `tests/test_iterative.c` as the reviewed proof owner
    instead of turning the first cleanup into a test-architecture rewrite
  - it should keep `tests/test_integration.c` as a follow-through lifecycle
    owner only if the source extraction actually changes repeated-run or handle
    semantics
  - it should keep direct-family source and giant-test cleanup explicitly
    later, not folded into the first iterative lane
  - it should not widen into public-header, package/runtime, benchmark, or
    example ownership changes
- The preserved first-batch fence is explicit:
  - no broad algorithm rewriting detached from ownership cleanup
  - no proof-owner diffusion by moving helpers without preserving owners
  - no repo-wide “cleanup sweep” claim
  - no public-header or package/runtime churn detached from the landed source
    seam
  - no benchmark/reporting or example drift into correctness ownership
  - no reopening Sprint 84 assurance widening as part of the first Sprint 85
    cleanup batch

### Validation
- Re-read the Sprint 85 Day 5 design expectations, the Day 4 boundary
  artifact, and the strongest recent architecture-design template.
- Reconciled the iterative-source extraction owner, retained proof owners, and
  support-only touch fence against the current Sprint 85 ordering.

### Day 5 Exit State
- Sprint 85 now has one bounded iterative extraction/decomposition contract.
- Ownership between the first iterative source lane, retained proof owners, and
  later direct-family/test follow-through is fixed before Day 6 begins.
- Public-header, runtime, benchmark, example, and broader source/test
  spillover remain explicitly outside the first batch.

## Day 6 - Iterative-Source Cleanup Batch

### Goal
Land one bounded iterative-source cleanup batch inside `src/sparse_iterative.c`
without widening into proof-owner, public-header, or direct-family cleanup.

### Actions
- Re-read the Sprint 85 Day 6 batch expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 5 architecture contract in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day5-decomposition-ownership-architecture-design.md`.
- Audited the highest-repeat frontend boilerplate inside
  `src/sparse_iterative.c`, focusing on:
  - repeated `sparse_iter_result_t` zero/reset setup
  - repeated trivial `n == 0` handling
  - repeated zero-right-hand-side early return handling
- Landed one bounded helper seam in `src/sparse_iterative.c`:
  - `s85_iter_result_reset`
  - `s85_iter_result_mark_converged`
  - `s85_iter_handle_trivial_system`
- Rewired the affected iterative frontends and block-entry points to use the
  shared helper seam while preserving existing public behavior and proof-owner
  coverage.
- Kept the batch source-owned inside `src/sparse_iterative.c`; no test-helper,
  integration, maintainer-guide, or README follow-through was required.

### Findings
- The first Sprint 85 implementation landing stayed inside the Day 5 fence:
  - required implementation center:
    - `src/sparse_iterative.c`
  - strongest support-only follow-through actually needed:
    - none
- The landed seam reduced local mixed-responsibility concentration without
  widening into a broader rewrite:
  - centralized frontend result reset logic
  - centralized trivial empty-system handling
  - centralized zero-right-hand-side converged fast path
- The cleanup remained internal and compatibility-preserving:
  - no public API change
  - no proof-owner migration into `tests/test_iterative.c`
  - no shared lifecycle follow-through needed in `tests/test_integration.c`
  - no documentation/support-surface movement needed in
    `docs/maintainer_guide.md` or `README.md`
- The preserved Day 6 non-goal fence held:
  - no direct-family cleanup folded into the batch
  - no giant-test architecture cleanup folded into the batch
  - no benchmark, example, package, or runtime ownership drift
  - no assurance-surface widening detached from the landed source seam

### Validation
- `make format`
- `make lint`
- `make test`

### Day 6 Exit State
- Sprint 85 now has one real bounded iterative-source cleanup batch landed in
  `src/sparse_iterative.c`.
- The first maintainability move reduced repeated frontend/trivial-case
  boilerplate while preserving the existing proof-owner split.
- Direct-family hotspot cleanup and giant-test architecture cleanup remain the
  strongest later Sprint 85 seams.

## Day 7 - Post-Landing Audit and Rerank

### Goal
Re-rank the remaining Sprint 85 maintainability contradictions after the Day 6
iterative-source cleanup landing.

### Actions
- Re-read the Sprint 85 Day 7 rerank expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 6 landed batch artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day6-iterative-source-cleanup-batch.md`.
- Re-measured the live post-Day-6 hotspot sizes across the strongest remaining
  implementation and proof-owner surfaces:
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
  - `tests/test_chol_csc.c`
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
- Reconciled what contradiction Day 6 actually removed against the preserved
  Sprint 85 ordering:
  - first iterative source cleanup landed
  - strongest next source hotspot still outstanding
  - giant-test architecture cleanup still real, but not yet stronger than the
    remaining direct-family source concentration
- Fixed the exact Day 8 design center and support-only follow-through map in
  writing.

### Findings
- The Day 6 landing closed the strongest first maintainability contradiction:
  - `src/sparse_iterative.c` no longer stands out as the clear first cleanup
    center
  - the repo now has one real bounded iterative-source cleanup seam landed
  - a second immediate iterative-only batch is not the highest-value next move
- The strongest remaining Sprint 85 seam is now bounded direct-family source
  cleanup:
  - `src/sparse_chol_csc.c` = `1841` lines
  - `src/sparse_iterative.c` = `1854` lines after the Day 6 cleanup
  - the useful distinction is no longer raw size alone; it is that the
    iterative first-lane contradiction has already been reduced in code while
    the direct-family hotspot still has not
- Giant-test architecture cleanup remains explicitly later than the next
  source batch:
  - `tests/test_chol_csc.c` remains the strongest proof-owner concentration
  - but it is still support-later work unless the next direct-family source
    cleanup exposes a helper/registration seam that truly forces movement
- The exact Day 8 design center is now fixed to:
  - `src/sparse_chol_csc.c`
- The strongest support-only follow-through is now:
  - `tests/test_chol_csc.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- The preserved post-Day-6 non-touch map is now explicit:
  - no second immediate iterative cleanup batch as the next center
  - no giant-test architecture rewrite before the next source hotspot is
    designed
  - no spillover into `src/sparse_qr.c`, `src/sparse_ldlt.c`, or
    `tests/test_qr.c` as the next center
  - no benchmark, example, package, or runtime ownership drift

### Validation
- This was a docs-only rerank day, so no build/test rerun was required.
- Re-read the Day 6 landed batch and re-measured the live source/test hotspot
  map from the current tree.

### Day 7 Exit State
- Sprint 85 now has one explicit post-Day-6 rerank.
- The second implementation center is fixed to the direct-family source hotspot
  in `src/sparse_chol_csc.c`.
- Giant-test architecture cleanup remains real Sprint 85 work, but only after
  the next direct-family source design is fixed.

## Day 8 - Direct-Family Hotspot Design

### Goal
Define the exact bounded direct-family cleanup seam Sprint 85 will land next on
the `src/sparse_chol_csc.c` hotspot.

### Actions
- Re-read the Sprint 85 Day 8 design expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 7 rerank artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day7-post-landing-audit-and-rerank.md`.
- Re-scanned `src/sparse_chol_csc.c` for the strongest mixed-responsibility
  seams inside the remaining direct-family hotspot:
  - Cholesky CSC conversion/elimination/solve ownership
  - factor/factor+solve orchestration
  - dense LDL^T primitive and backend-selection ownership
  - CSC-to-linked-list writeback/publication ownership
- Reconciled those seams against the current LDL^T family owners in:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
- Fixed the bounded Day 9 extraction contract, support-only touch set, and
  preserved non-goal fence in writing.

### Findings
- Sprint 85 now has one explicit second implementation contract:
  - required implementation center:
    - `src/sparse_chol_csc.c`
  - directly forced support surfaces if the extraction truly needs them:
    - `src/sparse_ldlt_csc.c`
    - `src/sparse_ldlt_csc_internal.h`
    - `tests/test_chol_csc.c`
    - `docs/maintainer_guide.md`
    - `README.md`
- The strongest bounded cleanup seam is not generic Cholesky helper
  redistribution.  It is the embedded dense LDL^T/backend block currently
  owned by `src/sparse_chol_csc.c` even though it belongs to the LDL^T family:
  - `ldlt_dense_factor`
  - `ldlt_dense_factor_selected`
  - `ldlt_dense_factor_backend_name`
  - the associated Accelerate probe / backend-selection helpers
- The Day 8 ownership split is now fixed:
  - Cholesky CSC backend owner after cleanup:
    - `src/sparse_chol_csc.c`
  - LDL^T dense primitive and backend-selection owner after cleanup:
    - `src/sparse_ldlt_csc.c`
  - LDL^T internal declaration owner if needed:
    - `src/sparse_ldlt_csc_internal.h`
  - retained proof owner only if symbol movement truly forces follow-through:
    - `tests/test_chol_csc.c`
  - support-surface wording owners only if helper ownership movement changes
    maintainer guidance:
    - `docs/maintainer_guide.md`
    - `README.md`
- The strongest clarification is explicit now:
  - Day 9 should reduce mixed family ownership inside `src/sparse_chol_csc.c`
    by moving the LDL^T dense/backend seam to the LDL^T CSC owner
  - it should not reopen Cholesky CSC elimination behavior, public contracts,
    or supernodal semantics
  - it should not turn the batch into a broad split of every helper block in
    `src/sparse_chol_csc.c`
  - it should keep giant-test architecture cleanup explicitly later unless the
    symbol move truly forces proof-owner follow-through
- The preserved Day 8 non-goal fence is explicit:
  - no generic family-wide refactor across Cholesky and LDL^T
  - no public-header or install/package/runtime churn
  - no benchmark/example ownership drift
  - no giant-test registration rewrite as part of the source move
  - no reopening Sprint 84 assurance widening

### Validation
- This was a docs-only design day, so no build/test rerun was required.
- Re-read the Day 7 rerank, re-scanned the `src/sparse_chol_csc.c` hotspot,
  and reconciled the candidate extraction seam against the current LDL^T CSC
  owners.

### Day 8 Exit State
- Sprint 85 now has one explicit second implementation contract.
- Day 9 is fixed to one bounded mixed-ownership cleanup centered on moving the
  dense LDL^T/backend seam out of `src/sparse_chol_csc.c`.
- Giant-test architecture cleanup remains clearly separated from the next
  source batch.

## Day 9 - Direct-Family Hotspot Batch

### Goal
Land the bounded mixed-ownership cleanup fixed on Day 8 by moving the dense
LDL^T/backend seam out of `src/sparse_chol_csc.c` and into the LDL^T CSC owner.

### Actions
- Re-read the Sprint 85 Day 9 batch expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 8 direct-family design artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day8-direct-family-hotspot-design.md`.
- Moved the dense LDL^T primitive and bounded backend-selection implementation
  from `src/sparse_chol_csc.c` into `src/sparse_ldlt_csc.c`:
  - `ldlt_dense_factor`
  - `ldlt_dense_factor_selected`
  - `ldlt_dense_factor_backend_name`
  - the associated Accelerate probe and backend-selection helpers
- Moved the internal declarations for that seam from
  `src/sparse_chol_csc_internal.h` into `src/sparse_ldlt_csc_internal.h`.
- Reconciled the directly forced proof-owner include follow-through:
  - added `src/sparse_ldlt_csc_internal.h` to `tests/test_chol_csc.c`
  - replaced the stray `sparse_chol_csc_internal.h` dependency in
    `tests/test_ldlt.c` with `sparse_ldlt_csc_internal.h`
- Preserved the non-goal fence:
  - no Cholesky CSC elimination rewrite
  - no family-wide public/API widening
  - no giant-test registration cleanup
  - no maintainer-guide or README movement
- Ran the full required validation gate for a real source-hotspot batch.

### Findings
- The Day 9 landing stayed inside the Day 8 fence:
  - required implementation center:
    - `src/sparse_chol_csc.c`
  - directly forced support surfaces actually needed:
    - `src/sparse_ldlt_csc.c`
    - `src/sparse_ldlt_csc_internal.h`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
  - not needed in the batch:
    - `docs/maintainer_guide.md`
    - `README.md`
    - giant-test architecture cleanup
    - adjacent source cleanups in `src/sparse_qr.c` or `src/sparse_ldlt.c`
- The real contradiction reduced in code was mixed family ownership:
  - `src/sparse_chol_csc.c` no longer owns the LDL^T dense primitive/backend
    seam
  - `src/sparse_ldlt_csc.c` now owns the LDL^T dense primitive/backend seam
    that its supernodal path and env-contract tests already read as family-local
- The proof-owner follow-through stayed minimal and bounded:
  - no test logic rewrite
  - only internal header include adjustment where the symbol owner moved
- The strongest Day 9 clarification is explicit:
  - this was a real direct-family maintainability reduction, not a generic
    Cholesky refactor
  - the batch reduced the `src/sparse_chol_csc.c` hotspot by removing a
    non-Cholesky ownership block rather than redistributing arbitrary helpers
  - giant-test architecture cleanup remains later Sprint 85 work

### Validation
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- Reviewed parity remained exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
- Reviewed CMake runtime note:
  - `test_reorder_nd` remained the long tail at `254.28 sec`
  - reviewed CMake `Total Test time (real)` = `351.72 sec`

### Day 9 Exit State
- Sprint 85 now has one landed bounded direct-family source cleanup batch.
- The Cholesky CSC hotspot no longer owns the LDL^T dense/backend seam.
- Giant-test architecture cleanup remains the strongest later Sprint 85 seam.

## Day 10 - Giant-Test Architecture Design

### Goal
Fix the exact bounded giant-test architecture seam Sprint 85 should land next
after the Day 9 direct-family cleanup.

### Actions
- Re-read the Sprint 85 Day 10 expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 9 batch artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day9-direct-family-hotspot-batch.md`
  to confirm what maintainability work remains after the source-owner move.
- Re-checked the live giant proof-owner hotspots by line count:
  - `tests/test_chol_csc.c` = `4965`
  - `tests/test_qr.c` = `3234`
  - `tests/test_integration.c` = `3197`
  - `tests/test_ldlt.c` = `2921`
  - `tests/test_iterative.c` = `2841`
- Re-scanned `tests/test_chol_csc.c` for local ownership concentration:
  - dense helper concentration
  - late-file runner/helper groups already present for supernodal, writeback,
    and dispatch coverage
  - the remaining flat `RUN_TEST(...)` concentration in `main()`
- Reconciled the candidate cleanup seam against the Day 9 non-goal fence to
  keep the next batch test-owned and bounded.

### Findings
- The strongest remaining giant-test contradiction is now explicit:
  - required next implementation center:
    - `tests/test_chol_csc.c`
- The highest-value bounded cleanup seam is not generic helper extraction
  across proof owners.  It is the still-concentrated registration layout in
  `tests/test_chol_csc.c`, where `main()` continues to own a long flat block of
  `RUN_TEST(...)` calls even though the file already has local runner-group
  patterns near the end for:
  - supernode detection
  - supernodal postorder
  - supernodal dense operations
  - extract/writeback
  - diagonal factor helpers
  - panel helpers
  - parametrised supernodal cases
  - writeback
  - dispatch
- The exact Day 11 seam is now fixed:
  - reduce the long `main()` registration concentration in
    `tests/test_chol_csc.c`
  - keep cleanup inside that same proof owner by introducing a few additional
    local runner groups that match the pattern already used later in the file
  - treat helper-header, docs, and adjacent proof-owner files as support-only
    surfaces that move only if the local registration split truly forces them
- Directly forced support surfaces if the Day 11 cleanup truly needs them:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `docs/maintainer_guide.md`
  - `README.md`
- Explicitly not the Day 11 target unless the local split truly forces it:
  - `tests/test_qr.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - source files
  - cross-file proof-owner redistribution

### Validation
- This was a docs-only design day, so no build/test rerun was required.
- Re-read the Day 9 landed state, rescanned the giant-test hotspot, and
  fixed the next bounded cleanup seam around local registration concentration
  inside the retained Cholesky CSC proof owner.

### Day 10 Exit State
- Sprint 85 now has one explicit giant-test architecture contract.
- Day 11 is fixed to one bounded `tests/test_chol_csc.c` cleanup centered on
  local runner organization and `main()` registration reduction.
- Proof-owner clarity is preserved by keeping the next cleanup inside the same
  test owner instead of diffusing helpers across files.

## Day 11 - Giant-Test Architecture Batch

### Goal
Land the bounded giant-test cleanup fixed on Day 10 by reducing registration
concentration in `tests/test_chol_csc.c` while keeping proof ownership local to
that file.

### Actions
- Re-read the Sprint 85 Day 11 batch expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the Day 10 giant-test architecture design artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day10-giant-test-architecture-design.md`.
- Re-scanned the late-file runner-group pattern already present in
  `tests/test_chol_csc.c` for supernodal, writeback, and dispatch coverage.
- Added a matching set of local runner groups for the earlier coverage
  families still registered inline in `main()`:
  - alloc / growth
  - conversion round-trips
  - permutations plus fill-factor / norm caching
  - symbolic analysis plus validate / edge hardening
  - workspace plus elimination scaffolding
  - scalar kernel coverage
  - solve / residual / shim coverage
- Replaced the long early flat `RUN_TEST(...)` block in `main()` with those
  local runner-group calls.
- Preserved the Day 10 non-goal fence:
  - no cross-file proof-owner redistribution
  - no test logic rewrite
  - no helper-header churn
  - no docs movement
- Ran the full required validation gate for a giant-test maintainability batch.

### Findings
- The Day 11 landing stayed inside the Day 10 fence:
  - required implementation center:
    - `tests/test_chol_csc.c`
  - directly forced support surfaces actually needed:
    - none
  - not needed in the batch:
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `docs/maintainer_guide.md`
    - `README.md`
    - adjacent proof-owner files
- The real maintainability reduction was local registration concentration:
  - `main()` no longer owns the early giant flat registration block
  - the file now uses one consistent runner-group pattern across both the early
    and late coverage families
  - proof ownership is still fully local to `tests/test_chol_csc.c`
- The strongest Day 11 clarification is explicit:
  - this was a bounded architecture cleanup, not a behavioral or oracle change
  - it reduced giant-test concentration without diffusing logic into new files
  - it keeps later Sprint 85 test-owner work separate from source-owner work

### Validation
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

### Day 11 Exit State
- Sprint 85 now has one landed bounded giant-test architecture cleanup batch.
- `tests/test_chol_csc.c` retains proof ownership, but its registration layout
  now follows one consistent local runner-group structure.
- The strongest remaining Sprint 85 seam is now later proof/docs alignment and
  final validation rather than another immediate Cholesky CSC registration
  cleanup.

## Day 12 - Proof / Docs Alignment & Validation Queue Freeze

### Goal
Reconcile the touched Sprint 85 proof-owner and support surfaces against the
landed cleanup boundaries and fix the final Day 13 validation queue in writing.

### Actions
- Re-read the Sprint 85 Day 12 expectations from
  `docs/planning/EPIC_8/SPRINT_85/PLAN.md`.
- Re-read the landed Day 11 giant-test batch artifact in
  `docs/planning/EPIC_8/SPRINT_85/artifacts/day11-giant-test-architecture-batch.md`.
- Reconciled the landed Sprint 85 implementation package against the touched
  proof-owner and support surfaces:
  - `src/sparse_iterative.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_chol_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_qr.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- Rechecked the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Rechecked the representative reviewed proof-owner and support binaries now
  fixed for Day 13:
  - `test_iterative`
  - `test_chol_csc`
  - `test_integration`
  - `test_ldlt`
  - `test_qr`
  - `example_analysis`
  - `example_basic_solve`
  - `bench_refactor_csc`
  - `bench_svd`
- Rechecked canonical benchmark/reporting and retained install/export proof
  ownership:
  - `make -n bench-canonical-report`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- No new support-only edit is needed before the full sweep.
- The final Sprint 85 proof-owner map is now explicit:
  - `tests/test_iterative.c` owns the retained proof for the bounded
    iterative-source cleanup
  - `tests/test_chol_csc.c` owns the retained proof for both the bounded
    direct-family cleanup follow-through and the bounded giant-test
    architecture cleanup
  - `tests/test_integration.c` remains the shared lifecycle/public-behavior
    owner, but it was not a Sprint 85 adopted cleanup center
  - `tests/test_ldlt.c` and `tests/test_qr.c` remain retained adjacent proof
    owners, not Sprint 85 cleanup centers
  - `docs/maintainer_guide.md` and `README.md` remain support-only surfaces
    with no further alignment change required before Day 13
- The support-surface boundary stayed fixed:
  - benchmark/reporting surfaces remain command/script-owned rather than
    correctness owners
  - install/export proof remains script-owned and out of the Sprint 85 touched
    implementation path
- The exact Day 13 validation queue is now fixed:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - focused reviewed proof owners:
    - `./build/quality-review-cmake/test_iterative`
    - `./build/quality-review-cmake/test_chol_csc`
    - `./build/quality-review-cmake/test_integration`
    - `./build/quality-review-cmake/test_ldlt`
    - `./build/quality-review-cmake/test_qr`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
  - benchmark/reporting follow-ons:
    - `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
    - `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
    - `make bench-canonical-report`

### Validation
- This was a docs-only alignment day, so no build/test rerun was required.
- Rechecked the reviewed parity anchor, representative reviewed binaries, and
  canonical benchmark/install ownership surfaces to freeze the Day 13 queue
  against the actual landed Sprint 85 boundaries.

### Day 12 Exit State
- No support-only drift remains before the full sweep.
- The final Sprint 85 validation queue is explicit and unambiguous.
- Day 13 can execute from a fixed touched-surface truth map instead of
  re-deciding Sprint 85 scope.
