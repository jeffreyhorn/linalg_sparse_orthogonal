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
