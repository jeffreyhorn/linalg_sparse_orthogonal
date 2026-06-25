# Sprint 89 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 89 baseline for Epic 8 by grounding the sprint in
the validated Sprint 88 close state, the live Sprint 89 project-plan section,
and the current end-state review, comparison, proof, reporting, and closeout
hotspots rather than another generic "final polish" reset.

### Actions
- Re-read the Sprint 89 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 89 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the strongest Sprint 88 closeout context:
  - `docs/planning/EPIC_8/SPRINT_88/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_88/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly through the Day 1 parity
  rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw line-count hotspot map for the strongest likely Sprint
  89 touch surfaces across planning/closeout docs, maintained install/export
  proof, benchmark/reporting surfaces, workflows, and the highest-value
  reviewed runtime and graph/reorder owners still likely to matter in a final
  fix batch.
- Opened Sprint 89 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 89 starts from the same strongest local reviewed baseline Sprint 88
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 89 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 89 is not a generic "close things out" sprint. Its highest value is
  one bounded final-integration and Epic 8 closeout package centered on:
  - end-state re-audit
  - external comparison sweep
  - final cross-surface fix batch
  - full validation and reporting sweep
  - residual queue finalization
  - retrospective, handoff, and final project-summary closeout
- The validated Sprint 88 close state already fixed the strongest handoff
  truth entering Sprint 89:
  - the front-door adoption path is clearer
  - example and install guidance now has a cleaner audience split
  - the static-first package and consumer contract remain bounded and
    truthful
  - Epic 8 now has one explicit final sprint queue instead of a broad
    residual bucket
- The strongest current end-state contradiction is no longer a single product,
  runtime, or usability seam. It is the final evidence problem:
  - the project needs one fresh live re-audit against the original Epic 8
    concerns
  - it needs one bounded external comparison package rather than only
    internally generated proof
  - it needs one final calibrated residual queue that distinguishes real carry
    forward work from deliberate non-claims
- The strongest likely Sprint 89 implementation, proof, reporting, and
  closeout surfaces are explicit from the live tree:
  - planning and closeout owners:
    - `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`
    - `docs/planning/EPIC_8/SPRINT_88/RETROSPECTIVE.md` = `267`
    - `docs/planning/EPIC_8/SPRINT_88/artifacts/day14-closeout-and-handoff.md`
      = `73`
  - strongest support, install/export, and reporting owners:
    - `README.md` = `1113`
    - `INSTALL.md` = `315`
    - `docs/maintainer_guide.md` = `727`
    - `tests/test_install.sh` = `195`
    - `tests/test_cmake_install.sh` = `208`
    - `benchmarks/README.md` = `399`
    - `scripts/bench_canonical_report.sh` = `101`
  - strongest workflow/build/package evidence surfaces:
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
    - `CMakeLists.txt` = `416`
    - `Makefile` = `908`
  - strongest reviewed-runtime and reorder/graph proof surfaces still likely
    to matter if the re-audit forces a final batch:
    - `tests/test_reorder_nd.c` = `2340`
    - `tests/test_graph.c` = `2925`
    - `tests/test_reorder.c` = `1082`
    - `tests/test_reorder_amd_qg.c` = `273`
    - `benchmarks/bench_reorder.c` = `338`
    - `benchmarks/bench_fillin.c` = `178`
- The strongest Day 1 clarification is now fixed:
  - Sprint 89 should begin with one evidence-first end-state review rather
    than with a speculative fix batch
  - external comparison belongs before any last-mile implementation widening
  - final closeout writing should come only after the strongest reviewed,
    install/export, and reporting anchors are refreshed from the live tree
- The preserved Sprint 89 non-goal pressure is explicit before Day 2:
  - no broad reopening of earlier sprint scope
  - no speculative optimization or capability widening without fresh evidence
  - no support-surface churn detached from a live end-state contradiction
  - no benchmark or workflow rewriting that outclaims the maintained proof
    owners
  - no Epic 8 summary writing before the final validation/reporting baseline
    is rebuilt

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live final-integration / proof / reporting hotspot map from
  direct line-count measurement.

### Day 1 Exit State
- Sprint 89 no longer starts from generic Epic 8 closeout prose.
- The end-state re-audit, external comparison sweep, final cross-surface fix
  batch, validation/reporting, residual-queue calibration, and closeout
  workstreams are fixed in writing.
- The strongest likely Sprint 89 touch surfaces and preserved non-goals are
  explicit before the validation and maintained cross-surface recheck begins.

## Day 2 - Validation and Maintained Cross-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, install/export, benchmark-reporting, example, and workflow truth
split before Sprint 89 changes any final-integration, comparison, or closeout
surface.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the strongest recent validation/surface templates from:
  - `docs/planning/EPIC_8/SPRINT_88/artifacts/day2-validation-baseline-and-maintained-support-surface-recheck.md`
  - `docs/planning/EPIC_8/SPRINT_87/artifacts/day2-validation-baseline-and-maintained-consumer-surface-recheck.md`
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed representative binaries and
  examples that remain the main executable truth surfaces entering Sprint 89:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the maintained reporting and consumer-proof owners:
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- Re-read the CI, macOS, and Windows workflow surfaces that constrain the
  current reviewed, supplemental, and staged platform truth:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

### Findings
- Sprint 89 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial final-integration, comparison, residual-calibration, or
    closeout-support batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently remains the strongest shared executable
  truth surface entering Sprint 89:
  - reviewed representative proof owners:
    - `./build/quality-review-cmake/test_reorder_nd`
    - `./build/quality-review-cmake/test_reorder`
    - `./build/quality-review-cmake/test_reorder_amd_qg`
    - `./build/quality-review-cmake/test_graph`
  - representative examples:
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
  - `bash tests/test_install.sh` proves the local Unix-side Make
    install/uninstall + `pkg-config` path
  - `bash tests/test_cmake_install.sh` proves the local Unix-side CMake
    install/export + `find_package(Sparse)` path
  - `examples/cmake_example/CMakeLists.txt` remains the representative
    downstream CMake consumer surface used by the CMake install/export proof
- Workflow-side truth remains intentionally layered rather than flattened into
  one broad parity claim:
  - Linux remains the strongest reviewed source of truth through the enforced
    reviewed Makefile, reviewed CMake, and dead-code lanes
  - macOS carries a narrower enforced reviewed Apple Clang lane plus a
    supplemental static-first Make install/`pkg-config` confidence lane
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim a reviewed Makefile or separate reviewed install-validation lane
  - Windows still fixes its reviewed `ctest -N` expectation at `50` and keeps
    staged exclusions explicit in job output
- The strongest Day 2 clarification is now fixed:
  - reviewed CMake binaries remain the main executable truth anchor
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
  - downstream consumer proof remains local and bounded
  - workflow lanes remain support evidence rather than broad cross-platform
    parity claims

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed representative binaries and
  examples.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `examples/cmake_example/CMakeLists.txt`, and the CI/macOS/Windows workflow
  surfaces.

### Day 2 Exit State
- Sprint 89 now has one explicit validation and maintained cross-surface
  contract before the end-state re-audit begins.
- Reviewed CMake binaries remain the main executable truth anchor.
- Canonical benchmark reporting remains command/script owned.
- Install/export proof remains script owned.
- Workflow lanes remain support evidence rather than broad parity claims.

## Day 3 - End-State Re-audit

### Goal
Reduce the full post-Sprint-88 tree to one ranked final contradiction map by
re-checking the original Epic 8 opening review categories against the live
project state instead of inheriting earlier sprint assumptions.

### Actions
- Re-read the Day 3 end-state audit contract from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Epic 8 opening concern list in
  `docs/planning/EPIC_8/PROJECT_PLAN.md`.
- Re-read the original competitive-gap vocabulary from
  `docs/planning/EPIC_8/SPRINT_80/artifacts/day3-live-competitive-gap-inventory.md`.
- Re-audited the live post-Sprint-88 tree against the original Epic 8 concern
  categories:
  - linked-list-first product/storage ceiling
  - builtin dense/backend performance ceiling
  - bounded capability surface
  - limited external differential proof
  - large source / giant-test maintainability concentration
  - reviewed runtime concentration
  - static-first and asymmetric package/platform maturity
  - front-door usability and policy density
- Reconciled those original concerns with the actual outcomes already landed
  in Sprints 81-88.
- Separated:
  - contradictions still strong enough to justify a final fix batch
  - contradictions that now primarily need explicit calibration or
    non-claims
  - contradictions already substantially closed by earlier sprint landings
  - contradictions that belong to the next planning cycle rather than Sprint
    89 implementation scope
- Identified the proof and comparison surfaces most likely to decide the final
  end-state call.

### Findings
- Sprint 89's broad closeout problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - final evidence and external-comparison ceiling
  - strongest second target:
    - reviewed runtime concentration around reorder/ND proof
  - strongest third target:
    - residual package/platform asymmetry calibration rather than broad new
      product work
  - strongest fourth target:
    - residual large-source and giant-test carry-forward calibration
  - strongest support-only but real target:
    - final public non-claims and next-cycle queue wording
- The strongest current contradiction is now explicit:
  - the project no longer lacks a sprint-by-sprint modernization story
  - it no longer lacks strong internal reviewed, install/export, and bounded
    benchmark/reporting proof surfaces
  - but it still does not yet have one final project-level end-state review
    plus external comparison package strong enough to close Epic 8 from the
    same vocabulary that opened it
- The original Epic 8 concern list now re-reads as follows against the live
  tree:
  - linked-list-first product/storage ceiling:
    - materially reduced by Sprint 81
    - still a truthful product characteristic
    - now mostly a calibrated non-claim rather than a Sprint 89 fix center
  - builtin dense/backend performance ceiling:
    - materially improved by Sprint 82's bounded optional backend lane
    - still not a claim of universal backend maturity
    - now mostly a calibration/comparison issue rather than a first Sprint 89
      implementation center
  - bounded capability surface:
    - materially improved by Sprint 83's scalar/index/public-surface widening
    - still intentionally bounded
    - now reads as a residual queue and explicit non-claim issue
  - limited external differential proof:
    - partially improved by Sprint 84's maintained direct-family differential
      lane
    - still the strongest remaining project-level evidence gap because the
      repo lacks one final bounded external comparison sweep across the
      end-state claims
  - large source / giant-test maintainability concentration:
    - meaningfully reduced by Sprint 85
    - still present in some retained hotspots
    - now reads as next-cycle carry-forward more than Sprint 89 first-batch
      work
  - reviewed runtime concentration:
    - materially reduced by Sprint 86
    - still visibly present around `test_reorder_nd`
    - remains the strongest live implementation-side contradiction if a final
      fix batch is truly needed
  - static-first and asymmetric package/platform maturity:
    - sharpened significantly by Sprint 87
    - still intentionally asymmetric
    - now reads primarily as a truthful bounded contract rather than a broad
      reopening target
  - front-door usability and policy density:
    - materially improved by Sprint 88
    - no longer reads like a first-tier open contradiction
- The strongest likely fix-now vs calibrate-only split is now explicit:
  - strongest likely fix-now candidates:
    - any last-mile evidence or touched-surface reconciliation exposed by the
      final external comparison sweep
    - any bounded reorder/ND runtime or proof-surface seam that still reads as
      disproportionately costly in the final validated baseline
  - strongest calibrate-only lanes:
    - linked-list-first public product reading
    - bounded capability surface beyond the widened Epic 8 seams
    - static-first and asymmetric platform/package maturity
    - residual large-source concentration not reopened by the final evidence
      package
- The strongest proof-surface priority map is now fixed:
  - main reviewed executable truth:
    - `make quality-review-full`
    - reviewed CMake parity and representative reviewed binaries
  - main maintained consumer/package truth:
    - `bash tests/test_install.sh`
    - `bash tests/test_cmake_install.sh`
  - main maintained reporting truth:
    - `make bench-canonical-report`
    - `scripts/bench_canonical_report.sh`
  - main bounded external-comparison starting lane:
    - the retained direct-family and touched runtime/package surfaces rather
      than a broad library-wide oracle claim
- The strongest Day 3 clarification is now fixed:
  - Sprint 89 should not start with a blind last-mile fix batch
  - it should first produce one end-state audit and one bounded external
    comparison package
  - only then should it decide whether a final implementation batch is truly
    necessary

### Validation
- Re-read the Sprint 89 Day 3 plan contract.
- Re-read the Epic 8 opening concern list from
  `docs/planning/EPIC_8/PROJECT_PLAN.md`.
- Re-read the original gap vocabulary from
  `docs/planning/EPIC_8/SPRINT_80/artifacts/day3-live-competitive-gap-inventory.md`.
- Re-audited the live post-Sprint-88 tree against the original Epic 8 concern
  categories and reconciled them with the landed Sprint 81-88 outcomes.

### Day 3 Exit State
- Sprint 89 now has one ranked live end-state contradiction map grounded in
  the original Epic 8 opening review vocabulary.
- The strongest remaining problem is fixed to final evidence and external
  comparison, not another generic modernization lane.
- The strongest implementation-side residual seam is now clearly bounded to
  reviewed runtime concentration only if the later evidence package justifies a
  real fix batch.

## Day 4 - Final Integration Boundary Freeze

### Goal
Fix the first bounded Sprint 89 implementation fence so the next design pass
can define one real external-comparison and final-evidence contract instead of
another broad final-cleanup rewrite.

### Actions
- Re-read the Day 4 boundary expectations from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Day 3 rerank and strongest contradiction split from
  `docs/planning/EPIC_8/SPRINT_89/artifacts/day3-end-state-re-audit.md`.
- Reconciled the strongest remaining contradiction with the preserved Sprint 89
  non-goal fence.
- Separated:
  - the required first implementation center
  - directly forced support surfaces only if the first landing truly needs
    them
  - support-only proof, reporting, validation, and closeout surfaces that stay
    later unless the first landing truly changes their obligations
  - explicitly deferred lanes that must not become the first batch center
- Recorded the first final-integration boundary in working notes and the Day 4
  artifact.

### Findings
- Sprint 89 now has one explicit first implementation fence:
  - required first landing:
    - bounded external comparison and end-state evidence package
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `scripts/bench_canonical_report.sh`
    - `benchmarks/README.md`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
    - `benchmarks/bench_reorder.c`
    - `benchmarks/bench_fillin.c`
  - support-only proof, workflow, and closeout surfaces that stay later unless
    the first landing truly forces movement:
    - `make quality-review-full`
    - reviewed representative binaries under `build/quality-review-cmake/`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
    - Sprint 89 retrospective / Epic 8 closeout / final project-summary
      surfaces
  - explicitly deferred from the first landing:
    - final cross-surface fix batch as a first-batch center
    - full validation/reporting sweep as a first-batch center
    - residual-queue finalization as a first-batch center
    - Epic 8 summary writing as a first-batch center
    - broad reopening of product, capability, packaging, or usability lanes
- The strongest Day 4 clarification is now fixed:
  - the best first Sprint 89 move is one bounded external-comparison and
    end-state-evidence lane
  - the first landing should decide how the final correctness, package-shape,
    and bounded performance evidence will be gathered and interpreted before
    any last-mile implementation widening moves
  - install/export proof, benchmark/reporting surfaces, and support docs remain
    directly allowed only if the comparison contract truly forces them to move
  - reviewed runtime reduction, residual-queue calibration, and all final
    closeout writing stay later unless the evidence lane proves they must move
- The preserved Sprint 89 first-batch non-goal fence is explicit now:
  - no blind final-fix batch before external evidence exists
  - no broad reopening of earlier sprint scope
  - no speculative runtime or capability widening without the comparison lane
    justifying it
  - no summary/retrospective writing before the final validated baseline exists
  - no support-surface churn detached from a real evidence-package seam

### Validation
- Re-read the Sprint 89 Day 4 plan contract.
- Re-read the Day 3 end-state rerank.
- Reconciled the required first landing against the preserved non-goal fence.

### Day 4 Exit State
- Sprint 89 now has one bounded first final-integration landing center.
- Day 5 can design one explicit external-comparison and final-evidence
  contract inside that fence.
- Later fix, validation, residual-calibration, and closeout-writing work is
  explicitly held back until the evidence lane is defined.

## Day 5 - Comparison and Fix Architecture Design

### Goal
Define the bounded external-comparison and final-fix contract Sprint 89 will
actually support before any end-state evidence or last-mile implementation work
lands.

### Actions
- Re-read the Day 5 integration-design expectations from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Day 4 boundary fence from
  `docs/planning/EPIC_8/SPRINT_89/artifacts/day4-final-integration-boundary.md`.
- Re-read the bounded external-oracle contract from:
  - `docs/planning/EPIC_8/SPRINT_80/artifacts/day5-external-oracle-contract.md`
  - `docs/planning/EPIC_8/SPRINT_84/artifacts/day5-oracle-property-failure-path-architecture-design.md`
  - `docs/planning/EPIC_8/SPRINT_84/artifacts/day6-direct-family-differential-batch.md`
- Re-read the current maintained external differential helper and bounded
  runtime evidence surfaces:
  - `tests/chol_external_dense_reference.py`
  - `benchmarks/bench_reorder.c`
  - `Makefile` `bench-reorder-sprint86`
- Fixed the exact comparison contract across:
  - correctness signal
  - package-shape signal
  - bounded performance signal
- Fixed how Sprint 89 will interpret comparison outcomes:
  - immediate final fix candidate
  - calibrated non-claim
  - future residual item
- Fixed the ownership split across comparison, final fixes,
  validation/reporting, and closeout writing.
- Defined the objective entry criteria for any final cross-surface fix batch.

### Findings
- Sprint 89 now has one explicit first implementation contract:
  - required implementation center:
    - bounded external comparison and end-state evidence package
  - directly forced support surfaces only if the first batch truly needs them:
    - `tests/test_chol_csc.c`
    - `tests/chol_external_dense_reference.py`
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `benchmarks/bench_reorder.c`
    - `Makefile`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
  - retained later owners unless the first batch truly changes their
    obligations:
    - `scripts/bench_canonical_report.sh`
    - `make quality-review-full`
    - Sprint 89 retrospective
    - Epic 8 closeout notes
    - final project-summary surfaces
- The Day 5 ownership split is now fixed:
  - maintained correctness comparison owner:
    - `tests/test_chol_csc.c`
  - retained external dense reference helper owner:
    - `tests/chol_external_dense_reference.py`
  - maintained package-shape truth owners:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
  - bounded performance-reference support owner:
    - `benchmarks/bench_reorder.c`
  - bounded runtime rerun contract owner if the comparison lane truly needs a
    dedicated local driver:
    - `Makefile` through `make bench-reorder-sprint86`
  - retained canonical reporting owner after the comparison lane:
    - `scripts/bench_canonical_report.sh`
  - support-surface wording owners only if the evidence package truly changes
    how the contract should be read:
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The strongest comparison contract is now explicit:
  - maintained correctness comparison lane:
    - bounded CHOLMOD-class SPD direct-solver comparison through the retained
      external dense reference lane already owned by `tests/test_chol_csc.c`
      and `tests/chol_external_dense_reference.py`
  - maintained package-shape comparison lane:
    - installed consumer and export-surface truth through
      `tests/test_install.sh` and `tests/test_cmake_install.sh`
  - bounded performance-reference support lane:
    - touched reorder/ND runtime evidence through
      `bench_reorder --sprint86-slice --skip-factor`
    - this is support for the final runtime reading, not a broad product
      correctness or benchmark-superiority claim
  - explicitly advisory but not first-contract lanes:
    - METIS-class graph/reordering comparison remains useful advisory context
      only
    - broader sparse-solver ecosystem comparison remains outside the maintained
      Sprint 89 contract
- The strongest outcome interpretation contract is now fixed:
  - immediate final fix candidate:
    - maintained correctness comparison disagreement on the bounded SPD lane
    - package/install/export contract mismatch on the maintained local proof
      surfaces
    - clear touched-lane runtime contradiction on the retained reorder/ND
      evidence surface that stays attributable and bounded
  - calibrated non-claim:
    - comparison confirms the repo remains intentionally bounded rather than
      broad or best-in-class on a lane
    - package/platform asymmetry remains truthful and explicitly supported only
      on the maintained surfaces
  - future residual item:
    - advisory ecosystem gaps that remain real but fall outside the maintained
      Sprint 89 comparison contract
- The strongest final-fix entry contract is now explicit:
  - a final fix batch should land only if the comparison package exposes:
    - a correctness mismatch on the maintained SPD comparison lane
    - a local install/export or consumer-shape contradiction
    - a bounded reorder/ND runtime or proof-surface contradiction still large
      enough to justify one last touched implementation pass
    - or a support-surface wording contradiction made unavoidable by the
      evidence package
  - a final fix batch should not land just because:
    - advisory ecosystem comparisons look broader elsewhere
    - the repo remains intentionally bounded on capability or platform shape
    - a performance result is merely less impressive than an external system
      without contradicting the maintained contract
- The strongest Day 5 clarification is now fixed:
  - Day 6 should not try to compare "everything"
  - it should preserve the Sprint 80 oracle fence and the Sprint 84 direct
    differential lane
  - it should add package-shape truth and bounded runtime-reference support to
    that same final evidence package
  - it should keep canonical reporting and all closeout writing as later lanes
    rather than collapsing them into the first evidence batch

### Validation
- Re-read the Sprint 89 Day 5 plan contract.
- Re-read the Day 4 boundary artifact.
- Re-read the bounded oracle contract and the Sprint 84 maintained external
  differential ownership package.
- Re-read the current external dense reference helper and bounded reorder
  runtime evidence surfaces.

### Day 5 Exit State
- Sprint 89 now has one bounded external-comparison and final-fix architecture
  contract.
- Ownership between correctness comparison, package-shape proof,
  performance-reference support, retained validation/reporting, and later
  closeout writing is fixed before Day 6 begins.
- Any final implementation batch is now gated by objective evidence-entry
  criteria rather than by generic endgame pressure.

## Day 6 - End-State Re-audit Batch

### Goal
Materialize the final architecture, capability, usability, performance, and
packaging review against the live post-Sprint-88 tree so Sprint 89 carries one
explicit category-by-category end-state package rather than only a ranked Day 3
contradiction sketch.

### Actions
- Re-read the Day 6 re-audit-batch expectations from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Day 3 rerank and the Day 5 comparison/fix contract.
- Reconciled the original Epic 8 opening concern list against the actual
  Sprint 81-88 outcomes in one category-by-category review package.
- Classified each major Epic 8 concern as:
  - closed
  - partially closed but calibrated
  - still contradictory and fix-worthy
- Identified the smallest truthful set of final-fix candidates consistent with
  the Day 5 entry contract.
- Recorded the re-audit result in working notes and the Day 6 artifact.

### Findings
- Sprint 89 now has one explicit end-state review package across the original
  Epic 8 concern categories:
  - closed:
    - front-door usability and workflow layering
    - bounded direct-family external differential adoption as a real maintained
      lane
    - package/install/export consumer-contract sharpness at the maintained
      static-first level
  - partially closed but calibrated:
    - linked-list-first product/storage ceiling
    - builtin dense/backend performance ceiling
    - bounded capability surface
    - large source / giant-test maintainability concentration
    - static-first and asymmetric package/platform maturity
  - still contradictory and fix-worthy:
    - final external-comparison package is not yet executed
    - final cross-surface end-state evidence is not yet assembled in one place
    - reviewed runtime concentration remains the strongest live implementation
      seam if later evidence still justifies a last touched batch
- The strongest category-by-category end-state reading is now explicit:
  - architecture and maintainability:
    - materially improved by the compressed-first, maintainability, and proof
      ownership work in Sprints 81 and 85
    - still not a claim that all large sources or giant tests are small
    - now calibrated as acceptable Epic 8 close state with explicit carry
      forward hotspots rather than a Sprint 89 first-batch implementation lane
  - capability surface:
    - materially improved by Sprint 83
    - still intentionally bounded and not a broad complex/mixed-precision
      library claim
    - now an explicit non-claim and residual-queue issue rather than a final
      implementation target
  - numerical assurance and comparison:
    - materially improved by Sprint 84's maintained external SPD lane plus
      seeded-property and failure-path proof
    - still lacks the final bounded project-level external comparison package
      needed for end-state closeout
    - remains the strongest still-open evidence lane
  - runtime and scalability:
    - materially improved by Sprint 86
    - still has a visible reviewed `test_reorder_nd` long pole
    - remains the strongest live implementation-side contradiction only if the
      later evidence package still points at a bounded touched fix
  - packaging and consumer shape:
    - materially improved by Sprint 87
    - now reads as a truthful maintained static-first and asymmetric platform
      contract
    - no longer looks like a broad reopening target unless the comparison lane
      exposes a real mismatch
  - front-door usability and workflow layering:
    - materially improved by Sprint 88
    - now reads as effectively closed for Epic 8 purposes
- The smallest truthful final-fix candidate set is now explicit:
  - strongest likely candidate:
    - any contradiction exposed by the still-unrun final external comparison
      package on the maintained SPD/package/runtime surfaces
  - bounded implementation-side fallback candidate:
    - one touched reorder/ND runtime or proof-surface adjustment if the final
      evidence package still shows a materially disproportionate cost worth
      correcting
  - likely no-op lanes unless forced by evidence:
    - product/storage
    - capability surface
    - packaging/product-matrix semantics
    - front-door usability surfaces
- The strongest opening-review reconciliation is now fixed:
  - Epic 8 did materially move every original concern class
  - but several classes close truthfully only as bounded, calibrated
    improvements rather than as total eliminations
  - Sprint 89 therefore needs a final evidence package and calibrated residual
    queue more than it needs another broad implementation sprint
- The strongest Day 6 clarification is now explicit:
  - the last-mile contradiction list is small
  - the strongest still-open work is evidence and comparison, not a hidden new
    modernization lane
  - the final fix batch should stay empty unless later evidence really forces
    movement

### Validation
- Re-read the Sprint 89 Day 6 plan contract.
- Re-read the Day 3 rerank and Day 5 comparison/fix architecture contract.
- Reconciled the original Epic 8 concern list against the landed Sprint 81-88
  outcomes in one explicit end-state review package.

### Day 6 Exit State
- Sprint 89 now has one category-by-category end-state review package rather
  than only a ranked contradiction list.
- The smallest truthful final-fix candidate set is explicit before the Day 7
  rerank.
- Sprint 89 is now positioned to decide whether the next move is comparison
  first, fix first, or a tightly coupled evidence/fix sequence.

## Day 7 - Post-Re-audit Rerank

### Goal
Convert the Day 6 end-state package into one exact landing order for the
comparison sweep, any possible final fix batch, and the remaining closeout
work.

### Actions
- Re-read the Day 7 rerank contract from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Day 6 end-state review package and the Day 5 comparison/fix
  architecture contract together.
- Re-ranked the still-open Sprint 89 work by:
  - value
  - urgency
  - proof strength
  - dependency order
- Separated:
  - must-land-before-close items
  - comparison-only evidence items
  - residual-queue items
- Fixed the exact Day 8 design center and the expected order between the
  comparison lane and any later final fix batch.

### Findings
- Sprint 89's remaining closeout work is now reduced to one exact landing
  order:
  - strongest first target:
    - external comparison sweep design and execution
  - strongest second target:
    - final cross-surface fix batch only if the comparison package exposes a
      real contradiction
  - strongest third target:
    - full validation and reporting sweep
  - strongest fourth target:
    - residual-queue finalization and calibrated non-claims
  - strongest fifth target:
    - Sprint 89, Epic 8, and final project-summary closeout writing
- The strongest post-re-audit clarification is now explicit:
  - external comparison should come before any implementation widening
  - the current evidence package is already strong enough that a blind fix
    batch would be lower-value and less defensible than executing the
    comparison lane first
  - the likely final fix batch is now small enough that it may collapse to a
    no-op if the comparison package stays aligned with the maintained contract
- The must-land-before-close set is now fixed:
  - bounded external comparison protocol
  - bounded external comparison execution
  - full final validation and reporting sweep
  - calibrated residual queue
  - Sprint 89 and Epic 8 closeout writing from the validated baseline
- The comparison-only evidence set is now fixed:
  - maintained SPD correctness comparison outcome
  - maintained package-shape comparison outcome
  - bounded reorder/ND runtime-reference outcome
  - comparison-backed interpretation of where the repo is intentionally
    bounded versus still contradictory
- The residual-queue set is now fixed more tightly:
  - linked-list-first public product reading as an explicit bounded product
    characteristic
  - intentionally bounded capability surface beyond the widened Epic 8 seams
  - static-first and asymmetric platform/package maturity beyond the
    maintained proof surface
  - retained large-source and giant-test hotspots not reopened by Sprint 89
  - reviewed runtime concentration only if comparison and final validation do
    not justify one last bounded fix
- The strongest fix-vs-no-fix order is now explicit:
  - comparison first
  - then decide whether the final fix batch is:
    - required
    - bounded
    - or explicitly unnecessary
- The exact Day 8 design center is now fixed to:
  - the bounded external comparison protocol and reporting shape

### Validation
- Re-read the Sprint 89 Day 7 plan contract.
- Re-read the Day 6 end-state re-audit package.
- Re-read the Day 5 comparison/fix architecture contract.
- Re-ranked the still-open Sprint 89 work by value, urgency, proof strength,
  and dependency order.

### Day 7 Exit State
- Sprint 89 now has one exact post-re-audit landing order.
- The comparison lane and fix lane are ordered clearly.
- Day 8 can design the explicit external comparison protocol without ambiguity
  about what comes next.

## Day 8 - External Comparison Sweep Design

### Goal
Freeze the exact bounded external comparison protocol and reporting shape so
Day 9 can execute one explicit comparison package instead of assembling ad hoc
evidence from mixed proof and benchmark surfaces.

### Actions
- Re-read the Day 8 comparison-design contract from
  `docs/planning/EPIC_8/SPRINT_89/PLAN.md`.
- Re-read the Day 7 rerank and the Day 5 comparison/fix architecture
  contract.
- Re-read the retained external comparison and runtime-reference owners:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `benchmarks/bench_reorder.c`
  - `Makefile` through `make bench-reorder-sprint86`
- Fixed the exact comparison lanes, commands, fixtures, and output fields
  Sprint 89 will actually treat as maintained Day 9 evidence.
- Fixed the acceptance criteria for a "good enough to close" comparison result
  and the narrow conditions that would still justify a final fix batch.
- Recorded the result in working notes and the Day 8 artifact.

### Findings
- Sprint 89 now has one exact Day 9 comparison-execution contract:
  - required execution owners:
    - `tests/test_chol_csc.c`
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `Makefile` through `make bench-reorder-sprint86`
  - directly forced support-only comparison surfaces only if the execution
    truly exposes a contradiction:
    - `tests/chol_external_dense_reference.py`
    - `benchmarks/bench_reorder.c`
    - `README.md`
    - `INSTALL.md`
    - `docs/maintainer_guide.md`
- The exact comparison protocol is now fixed:
  - maintained correctness lane:
    - execute the retained SPD external differential owner on the reviewed
      build through `./build/quality-review-cmake/test_chol_csc`
    - capture the bounded external-dense-reference outputs for:
      - `nos4`
      - `bcsstk04`
    - interpret the lane through:
      - `max|x-x_ref|`
      - retained in-repo residual strength
  - maintained package-shape lane:
    - execute:
      - `bash tests/test_install.sh`
      - `bash tests/test_cmake_install.sh`
    - capture the exact pass/fail/skip totals and whether the installed
      static-first consumer and export surfaces still match the maintained
      package contract
  - bounded runtime-reference support lane:
    - execute `make bench-reorder-sprint86`
    - capture the bounded touched-corpus slice for:
      - `bcsstk14`
      - `Pres_Poisson`
    - report the emitted:
      - `nnz_L`
      - `reorder_ms`
      - reorder name
    - preserve the bounded reading:
      - branch-local touched-lane evidence only
      - not a portable timing gate
      - not a broad product-superiority claim
- The accepted Day 9 reporting shape is now fixed:
  - correctness agreement:
    - one explicit fixture-by-fixture statement for `nos4` and `bcsstk04`
    - each statement must include the external agreement metric and the
      retained residual reading
  - package/consumer shape alignment:
    - exact totals from the two install/export proof scripts
    - explicit statement of whether the installed package contract still reads
      as maintained and truthful
  - bounded runtime observations:
    - one explicit AMD-vs-ND comparison on the Sprint 86 slice fixtures
    - interpretation framed as touched-runtime evidence, not pass/fail timing
      policy
- The strongest "good enough to close" comparison criteria are now explicit:
  - the maintained SPD external lane shows no correctness mismatch on `nos4`
    or `bcsstk04`
  - both install/export proof scripts pass without exposing a package-shape or
    consumer-contract contradiction
  - the Sprint 86 reorder slice remains interpretable as bounded mixed runtime
    evidence and does not expose one new touched contradiction large enough to
    force a final implementation batch on its own
  - any remaining differences must be classifiable as:
    - bounded and acceptable
    - or explicit residual items
- The strongest forced-spillover rule is now fixed:
  - Day 10 and Day 11 should only move into a real final fix batch if the
    Day 9 comparison execution exposes:
    - an SPD correctness disagreement
    - a local install/export contradiction
    - a touched reorder/ND contradiction still large enough to justify one
      last bounded fix
    - or an unavoidable support-surface wording contradiction created by the
      evidence
  - the comparison package should not force movement just because:
    - external ecosystems are broader elsewhere
    - a fixture shows mixed rather than uniformly dominant runtime behavior
    - the repo remains intentionally bounded on capability, platform, or
      product shape
- The strongest Day 8 clarification is now explicit:
  - Day 9 does not need to invent a new comparison lane
  - it only needs to execute the already-retained SPD, package-shape, and
    bounded runtime owners together and interpret them coherently
  - that keeps the final-fix decision evidence-backed and keeps Epic 8 closeout
    truthful rather than aspirational

### Validation
- Re-read the Sprint 89 Day 8 plan contract.
- Re-read the Day 7 rerank and Day 5 comparison/fix architecture contract.
- Re-read the retained SPD external differential owner, the install/export
  proof owners, and the bounded Sprint 86 runtime-reference owner.

### Day 8 Exit State
- Sprint 89 now has one exact bounded external comparison protocol and
  reporting shape.
- Day 9 can execute the retained correctness, package-shape, and bounded
  runtime evidence lanes without ad hoc framing.
- Any possible final fix batch remains gated by explicit comparison outcomes
  rather than by generic endgame pressure.
