# Sprint 84 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 84 baseline for Epic 8 by grounding the sprint in
the validated Sprint 83 close state, the live Sprint 84 project-plan section,
and the current oracle, property, proof-owner, benchmark, and support-surface
seams rather than another generic “add more tests” restart.

### Actions
- Re-read the Sprint 84 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 84 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_84/PLAN.md`.
- Re-read the strongest Sprint 83 closeout context:
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_83/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 84
  touch surfaces across shared/public headers, implementation owners,
  direct-family proof-owner tests, iterative/eigs proof owners, benchmark
  surfaces, and support surfaces.
- Opened Sprint 84 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 84 starts from the same strongest local reviewed baseline Sprint 83
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 84 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 84 is not a generic “add more tests” sprint. Its highest value is one
  bounded assurance-modernization package centered on:
  - differential-proof audit
  - oracle / property / failure-path architecture design
  - first maintained direct-family external differential batch
  - deterministic seeded property expansion
  - failure-path numerical proof
  - focused policy / CI / support-surface alignment only where implementation
    truly moves the assurance contract
  - validation and closeout
- The strongest likely Sprint 84 assurance, proof, and support surfaces are
  explicit from the live tree:
  - `include/sparse_types.h` = `313`
  - `include/sparse_matrix.h` = `622`
  - `include/sparse_qr.h` = `392`
  - `include/sparse_svd.h` = `257`
  - `include/sparse_cholesky.h` = `227`
  - `include/sparse_ldlt.h` = `335`
  - `src/sparse_matrix.c` = `1297`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_chol_csc.c` = `1841`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
  - `tests/test_sparse_matrix.c` = `1136`
  - `tests/test_qr.c` = `3234`
  - `tests/test_svd.c` = `2766`
  - `tests/test_chol_csc.c` = `4787`
  - `tests/test_ldlt.c` = `2921`
  - `tests/test_iterative.c` = `2841`
  - `tests/test_eigs.c` = `1560`
  - `tests/test_integration.c` = `2973`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `716`
- The strongest Day 1 clarification is now fixed:
  - Sprint 84 should not reopen Sprint 83’s capability-surface owner fence as
    its first implementation center
  - Sprint 84 should not inflate support-surface or CI claims before one
    bounded maintained assurance seam truly lands
  - it should first reduce the current external-differential, seeded-property,
    and failure-path assurance ceiling on the highest-value touched lanes only
- The preserved Sprint 84 non-goal pressure is explicit before Day 2:
  - no generic capability widening restart
  - no repo-wide claim that every family now has maintained external proof
  - no benchmark-governance drift into correctness ownership
  - no broad oracle dependency story for untouched families
  - no package/platform maturity claim widening
  - no support-surface churn detached from a real landed assurance seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared/public, implementation, proof-owner, benchmark, and
  support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 84 no longer starts from generic Epic 8 assurance prose.
- The baseline, differential rerank, oracle/property/failure-path design,
  first direct-family differential landing, seeded property expansion,
  failure-path proof, focused policy/CI alignment, and validation workstreams
  are fixed in writing.
- The strongest likely Sprint 84 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live proof-owner
split before Sprint 84 widens any external differential, seeded-property, or
failure-path assurance seam.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_84/PLAN.md`.
- Re-read the strongest recent validation/proof-surface template from
  `docs/planning/EPIC_8/SPRINT_83/artifacts/day2-validation-baseline-and-proof-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed proof-owner binaries for the
  early Sprint 84 lanes:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_integration`
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
- Sprint 84 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial differential, property, failure-path, or support-policy
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently owns the strongest early-Sprint-84 proof
  surfaces:
  - public/shared and direct-family proof owners:
    - `./build/quality-review-cmake/test_sparse_matrix`
    - `./build/quality-review-cmake/test_qr`
    - `./build/quality-review-cmake/test_svd`
    - `./build/quality-review-cmake/test_chol_csc`
    - `./build/quality-review-cmake/test_ldlt`
    - `./build/quality-review-cmake/test_iterative`
    - `./build/quality-review-cmake/test_eigs`
    - `./build/quality-review-cmake/test_integration`
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
    main executable truth surfaces for early Sprint 84 assurance work
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
- Sprint 84 now has one explicit implementation-day validation contract before
  the differential rerank begins.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/package proof is fixed in writing.
- The highest-signal rerun set is explicit before the first assurance-priority
  rerank.

## Day 3 - Differential-Proof Audit

### Goal
Reduce Sprint 84's broad assurance problem to one ranked live contradiction map
so the sprint can choose one bounded maintained differential lane instead of
another generic “more tests” bucket.

### Actions
- Re-read the Sprint 84 differential-audit expectations from
  `docs/planning/EPIC_8/SPRINT_84/PLAN.md` and the Sprint 84 project-plan
  section in `docs/planning/EPIC_8/PROJECT_PLAN.md`.
- Re-read the bounded Sprint 80 external-oracle contract at
  `docs/planning/EPIC_8/SPRINT_80/artifacts/day5-external-oracle-contract.md`.
- Re-read the Sprint 83 close context so the audit stays downstream of the
  landed capability work rather than reopening it:
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_83/RETROSPECTIVE.md`
- Re-scanned the strongest likely assurance owners and cross-check-friendly
  proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`
- Rechecked the current maintainer reading around proof ownership, oracle
  boundaries, property ownership, cancellation/lifecycle semantics, and
  benchmark ownership in `docs/maintainer_guide.md` and `README.md`.
- Reconciled the live tree against the Sprint 80 oracle fence and the Sprint
  83 capability closeout before fixing the Day 3 rank order.

### Findings
- Sprint 84's broad assurance problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - bounded maintained external differential proof on the core direct-family
      SPD lane centered first on Cholesky CSC
  - strongest second target:
    - deterministic seeded property expansion beyond the current bounded
      lifecycle/property seams
  - strongest third target:
    - failure-path numerical proof on the most fragile cancellation,
      lifecycle-preservation, and residual-accounting seams
  - strongest fourth target:
    - iterative and eigensolver external differential follow-through
  - strongest support-only but real target:
    - CI/docs/support wording that still reflects the narrower current
      assurance reading
- The strongest current contradiction is not the absence of internal proof:
  - `tests/test_chol_csc.c` already owns large SuiteSparse residual checks,
    scalar-vs-batched cross-checks, and path-selection proof
  - `tests/test_ldlt.c` already owns residual, refine, lifecycle, and
    cross-backend proof
  - `tests/test_iterative.c` already owns true-residual, SuiteSparse, and
    direct-solver comparison proof
  - `tests/test_eigs.c` already owns dense cross-checks, SuiteSparse Ritz
    residuals, refinement checks, and SVD-side agreement checks
  - `tests/test_fuzz.c` already owns bounded seeded generative lifecycle
    property follow-through
- The contradiction is that the highest-value maintained external differential
  lane fixed by Sprint 80 still has not landed:
  - Sprint 80 froze the first maintained external-oracle lane as a
    CHOLMOD-class SPD Cholesky comparison
  - the current tree still proves the core direct-family SPD lanes mainly by
    internal residual, cross-path, and generated-property checks
  - benchmark and example surfaces remain intentionally non-oracle surfaces
  - that leaves the strongest first Sprint 84 move explicit:
    - land one bounded maintained external differential lane on the direct SPD
      Cholesky family first
    - treat broader solver-family external comparisons as follow-through only
      if that first lane lands cleanly
- The strongest second contradiction is property breadth:
  - `tests/test_fuzz.c` already covers LU, Cholesky, QR, SVD, and the large-`n`
    direct lifecycle parity lanes
  - deterministic property coverage is still narrower than the current public
    lifecycle and repeated-run assurance surface
  - this makes seeded-property widening real Sprint 84 work, but it reads as
    follow-through after the first maintained external differential lane is
    explicit
- The strongest third contradiction is fragile failure-path numerical proof:
  - `tests/test_integration.c` already owns cancellation and lifecycle
    preservation semantics across direct, QR, iterative, and eigensolver lanes
  - `tests/test_iterative.c` and `tests/test_eigs.c` already pin several true
    residual and refinement invariants
  - the most fragile cancellation/error-path/cross-check guarantees are still
    bounded and family-local rather than widened into one clearer assurance
    package
- The strongest fourth contradiction is iterative/eigs external proof depth:
  - these lanes already have stronger internal residual and direct-comparison
    proof than the direct-family external lane has today
  - Sprint 80's oracle fence does not justify making them the first maintained
    external comparison center ahead of the bounded direct SPD lane
- The useful Day 3 clarification is now explicit:
  - the best first Sprint 84 move is not generic property expansion
  - it is one bounded maintained external differential landing on the direct
    SPD Cholesky lane that Sprint 80 already fenced as first
  - seeded property widening follows next
  - failure-path numerical proof follows after that where the first lanes
    expose the real fragility
  - iterative/eigs external comparisons remain real, but they are explicitly
    later than the first direct-family lane
  - CI/docs/support surfaces stay support-only unless implementation truly
    moves the assurance contract
- The preserved Sprint 84 non-goal pressure remains explicit:
  - no repo-wide claim that every solver now has maintained external proof
  - no benchmark or example drift into correctness ownership
  - no broad dependency story for untouched families
  - no reopening Sprint 83's capability-surface owner work
  - no support-surface churn detached from a real landed proof seam

### Validation
- Re-read the Sprint 84 plan and project-plan differential-audit scope.
- Re-read the bounded Sprint 80 external-oracle contract.
- Re-scanned the strongest direct, iterative, eigensolver, lifecycle, and
  seeded-property proof-owner surfaces in the live tree.
- Rechecked the current maintainer and README proof-ownership reading so the
  rank order stays aligned with the maintained support surfaces.

### Day 3 Exit State
- Sprint 84 no longer has a generic assurance-expansion backlog.
- The first implementation center is fixed to one bounded maintained external
  differential lane on the direct-family SPD Cholesky path.
- Seeded property widening, failure-path proof, iterative/eigs external
  follow-through, and support-surface wording are explicitly ordered behind
  that first lane.

## Day 4 - First Assurance Boundary Freeze

### Goal
Fix the first bounded assurance implementation fence for Sprint 84 so the next
design pass can define one real oracle/property/failure-path contract instead
of another broad proof rewrite.

### Actions
- Re-read the Day 3 differential ranking against the Sprint 84 project-plan
  scope and the bounded Sprint 80 external-oracle contract.
- Re-fixed the required first implementation center around the first
  maintained external differential lane rather than seeded-property or
  failure-path widening.
- Separated the strongest support-only proof, CI, and wording surfaces that
  should move only if the first landing truly forces them.
- Re-fixed the preserved non-goal fence so the first batch cannot widen into:
  - broad oracle dependency stories for untouched families
  - benchmark-governance drift into correctness ownership
  - repo-wide maintained external-proof claims
  - support-surface churn detached from a real landed proof seam
- Recorded the first implementation fence in working notes and a Day 4
  artifact.

### Findings
- Sprint 84 now has one explicit first implementation fence:
  - required first landing:
    - `tests/test_chol_csc.c`
  - support only if the first landing truly forces it:
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `tests/test_framework.h`
    - `tests/test_ldlt.c`
    - `tests/test_fuzz.c`
    - `tests/test_integration.c`
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
    - `README.md`
    - `docs/maintainer_guide.md`
  - explicitly deferred from the first landing:
    - `tests/test_svd.c`
    - `src/sparse_chol_csc.c`
    - `src/sparse_ldlt.c`
    - `src/sparse_iterative.c`
    - `src/sparse_eigs.c`
    - generic seeded-property expansion as a first-batch center
    - broad failure-path numerical-proof widening as a first-batch center
    - iterative/eigs maintained external comparisons
    - benchmark/reporting surfaces as correctness owners
    - package/runtime/dependency-matrix widening
- The useful Day 4 clarification is now explicit:
  - the best first Sprint 84 move is the direct-family SPD external
    differential lane on the Cholesky CSC proof owner
  - seeded-property widening remains the strongest second seam, not the first
    implementation center
  - failure-path numerical proof remains real, but it is explicitly later than
    the first external differential landing unless that landing forces it
  - iterative/eigs external follow-through remains real, but it is explicitly
    later than the first direct-family lane
  - proof and support surfaces stay support-only unless the first landing
    truly changes behavior there
- The preserved first-batch non-goal fence is explicit now:
  - no repo-wide claim that every solver now has maintained external proof
  - no benchmark or example drift into oracle ownership
  - no broad external dependency story for untouched families
  - no seeded-property or failure-path expansion ahead of the first external
    differential contract
  - no reopening Sprint 83's capability-surface owner work
  - no support-surface churn detached from a real landed assurance seam

### Validation
- Re-read the Day 3 differential-audit result and the Day 4 plan boundary
  expectations.
- Re-read the bounded Sprint 80 external-oracle contract so the first landing
  stays inside the already-fixed maintained lane.
- Rechecked the direct-family, seeded-property, failure-path, and
  support-surface owners in the live tree before fixing the fence.

### Day 4 Exit State
- Sprint 84 now has one bounded first assurance landing center.
- Day 5 can design one oracle/property/failure-path contract inside that
  fence.
- Lower-value seeded-property, failure-path, iterative/eigs external, and
  broader support/dependency spillover work is held back until later lanes.

## Day 5 - Oracle / Property / Failure-Path Architecture Design

### Goal
Define the bounded assurance contract that Sprint 84 will actually land on the
first maintained direct-family external differential lane.

### Actions
- Re-read the Day 4 boundary and the bounded Sprint 80 external-oracle
  contract so the Day 5 design stayed inside the already-fixed maintained
  comparison lane.
- Re-scanned the current proof-owner split in:
  - `tests/test_chol_csc.c`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- Fixed the ownership split between:
  - maintained external differential harnesses
  - deterministic seeded property generators
  - failure-path invariant and cancellation checks
  - direct-family vs iterative/eigs adoption boundaries
- Re-fixed the touch fence for tests, support wording, and non-goal surfaces
  before any implementation batch lands.
- Recorded the Day 5 architecture contract in working notes and an artifact.

### Findings
- Sprint 84 now has one explicit first implementation contract:
  - required implementation center:
    - `tests/test_chol_csc.c`
  - support only if the first batch truly forces it:
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `tests/test_framework.h`
    - `tests/test_fuzz.c`
    - `tests/test_integration.c`
    - `tests/test_ldlt.c`
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
    - `README.md`
    - `docs/maintainer_guide.md`
- The Day 5 ownership split is now fixed:
  - maintained external differential harness owner:
    - `tests/test_chol_csc.c`
  - deterministic seeded property owner, but not in the first batch:
    - `tests/test_fuzz.c`
  - public failure-path, cancellation, and lifecycle-preservation owner, but
    not in the first batch unless forced:
    - `tests/test_integration.c`
  - direct-family support comparison owner if the first batch truly forces a
    second family-local seam:
    - `tests/test_ldlt.c`
  - iterative/eigensolver retained proof owners, but not first-batch adoption
    owners:
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
  - support-surface wording owners only if implementation truly moves the
    public assurance reading:
    - `README.md`
    - `docs/maintainer_guide.md`
- The useful Day 5 clarification is explicit now:
  - the first landing should preserve the Sprint 80 oracle fence by keeping
    the maintained external differential lane bounded to the direct-family SPD
    Cholesky path
  - it should keep the first maintained comparison test-owned, fixture-backed,
    and family-local inside `tests/test_chol_csc.c`
  - it should keep deterministic seeded property coverage in `tests/test_fuzz.c`
    as a separate follow-through seam rather than collapsing that work into the
    first external differential batch
  - it should keep cancellation, lifecycle-preservation, and error-path proof
    centered in `tests/test_integration.c` unless the first batch exposes one
    truly local Cholesky-only contradiction
  - it should keep iterative/eigs proof owners unchanged in the first batch
    rather than inflating the first maintained external lane into a repo-wide
    adoption claim
  - it should not turn benchmarks or examples into correctness owners
- The preserved first-batch fence is explicit now:
  - no mandatory heavyweight external stack for normal builds
  - no repo-wide claim that every solver now has maintained external proof
  - no seeded-property widening folded into the first batch unless the direct
    differential contract truly forces it
  - no failure-path/package/platform churn detached from a real landed
    comparison seam
  - no benchmark/reporting drift into oracle ownership
  - no reopening Sprint 83 capability-surface work

### Validation
- Re-read the Day 4 boundary and the bounded Sprint 80 external-oracle
  contract.
- Re-scanned the current direct-family, seeded-property, failure-path, and
  support-surface owners in the live tree.
- Rechecked the current maintainer and README proof-ownership reading so the
  design stays aligned with maintained surfaces and non-goal fences.

### Day 5 Exit State
- Sprint 84 now has one bounded oracle/property/failure-path architecture
  contract.
- Ownership between the first maintained direct-family external harness, the
  seeded-property lane, and the public failure-path lane is fixed before Day 6
  begins.
- Later iterative/eigs adoption and broader support/dependency spillover
  remain explicitly outside the first batch.

## Day 6 - Direct-Family Differential Batch

### Goal
Land the first maintained external differential proof batch on the bounded
direct-family SPD Cholesky lane.

### Actions
- Added a bounded external-process differential harness in
  `tests/test_chol_csc.c` for the Cholesky CSC proof owner.
- Added `tests/chol_external_dense_reference.py` as a pure-stdlib Python dense
  SPD reference helper so the maintained external lane does not require a
  heavyweight Python stack in normal builds.
- Kept the first batch fixture-backed and family-local by comparing against
  SuiteSparse SPD inputs:
  - `tests/data/suitesparse/nos4.mtx`
  - `tests/data/suitesparse/bcsstk04.mtx`
- Preserved the Day 5 fence:
  - no benchmark/example promotion into oracle ownership
  - no seeded-property widening folded into the first batch
  - no iterative/eigs external adoption in the same landing
  - no production `src/` churn for this proof-only batch
- Reconciled the maintainer-policy reading in `docs/maintainer_guide.md` so
  the direct-family external differential owner is explicit and still bounded.
- Validated the landed batch with:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- Recorded the landed batch in working notes and a Day 6 artifact.

### Findings
- Sprint 84 Day 6 landed one bounded maintained direct-family external
  differential batch:
  - required implementation center:
    - `tests/test_chol_csc.c`
  - strongest support-only follow-through that was truly needed:
    - `tests/chol_external_dense_reference.py`
    - `docs/maintainer_guide.md`
  - not needed in the batch:
    - `tests/test_chol_csc_supernodal_helpers.h`
    - `tests/test_framework.h`
    - `tests/test_fuzz.c`
    - `tests/test_integration.c`
    - `tests/test_ldlt.c`
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
    - `README.md`
- The landed differential seam is explicit now:
  - `test_external_dense_reference_nos4_csc`
  - `test_external_dense_reference_bcsstk04_amd_csc`
  - helper-owned dense reference solve via `python3
    tests/chol_external_dense_reference.py`
- The strongest proof stayed bounded:
  - test-owned
  - fixture-backed
  - family-local to the direct-family SPD Cholesky CSC path
  - external-process based without imposing a mandatory SciPy/CHOLMOD runtime
- Representative retained outputs stayed clean:
  - `nos4`: `max|x-x_ref| = 4.690e-13`, `rel_residual = 3.907e-15`
  - `bcsstk04`: `max|x-x_ref| = 3.224e-11`, `rel_residual = 3.010e-16`
- The useful Day 6 clarification is explicit now:
  - Sprint 84's first maintained external differential lane is real and landed
  - it is still not a repo-wide external-proof claim
  - seeded-property and failure-path expansion remain separate follow-through
    seams
  - iterative/eigs external adoption remains later work

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `make quality-review-full` passed.

### Day 6 Exit State
- Sprint 84 now has one landed bounded direct-family maintained external
  differential batch.
- The strongest missing assurance seam is no longer "any maintained external
  proof at all" on the direct-family SPD lane.
- Later sprint work can stay focused on reranking seeded-property expansion,
  failure-path numerical proof, and later-family external follow-through
  instead of reopening whether the first external lane exists.

## Day 7 - Post-Landing Audit and Rerank

### Goal
Re-rank the strongest remaining assurance seam after the Day 6 direct-family
external differential landing.

### Actions
- Re-read the Day 6 implementation batch, proof-owner notes, and the touched
  support surface in:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `docs/maintainer_guide.md`
- Re-scanned the strongest retained second-half assurance owners in the live
  tree:
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `README.md`
- Re-ranked what contradiction actually closed versus what remains strongest
  across:
  - more direct-family external differential work
  - deterministic seeded-property expansion
  - failure-path numerical proof
  - CI/support-surface alignment
- Verified that support-only churn can still be deferred safely after the Day 6
  landing.
- Recorded the rerank and exact Day 8 design center in working notes and a Day
  7 artifact.

### Findings
- The Day 6 landing closed the strongest first assurance contradiction:
  - the repo no longer lacks any maintained external differential lane on the
    highest-value direct-family SPD path
  - `tests/test_chol_csc.c` now owns a real bounded external-process
    differential seam on `nos4` and `bcsstk04`
  - a second immediate direct-family external batch is not the highest-value
    next move
- The strongest remaining Sprint 84 seam is now deterministic seeded-property
  expansion centered on:
  - `tests/test_fuzz.c`
- The strongest support-only follow-through is now:
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- Current reading:
  - `tests/test_fuzz.c` already owns the broadest deterministic property
    generator surface across random SPD, QR, SVD, and large-`n` Cholesky/LDL^T
    lifecycle properties
  - `tests/test_integration.c` already owns the strongest public cancellation,
    failure-path, and lifecycle-preservation invariants, so it stays
    support-only unless the property batch exposes one local contradiction
  - `docs/maintainer_guide.md` is already truthful after Day 6's bounded
    external-lane reconciliation
  - `README.md` remains broadly truthful and can stay deferred unless the next
    property batch truly changes the user-visible assurance reading
- The useful Day 7 clarification is explicit now:
  - Sprint 84's next contradiction center is no longer "prove that any
    maintained external differential lane exists"
  - it is also not "land a second direct-family external comparison just
    because the first one worked"
  - the strongest remaining bounded seam is deterministic seeded-property depth
    on retained lifecycle flows
  - failure-path numerical proof remains real, but it is explicitly later than
    the property-expansion design lane
  - iterative/eigs external adoption remains later work
  - benchmark and example surfaces still do not become correctness owners

### Validation
- This was a docs-only rerank day, so no build/test rerun was required.
- The rerank was grounded in direct rereads of:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `docs/maintainer_guide.md`
  - `README.md`

### Day 7 Exit State
- Sprint 84 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one deterministic seeded-property design lane.
- Support drift is separated from the real next assurance move.

## Day 8 - Seeded Property Expansion Design

### Goal
Define the bounded deterministic property-expansion contract for the
highest-value retained lifecycle seams after the Day 6 external differential
landing and Day 7 rerank.

### Actions
- Re-read the Day 7 rerank and the current property-owner surface in
  `tests/test_fuzz.c`.
- Re-scanned the current deterministic property lanes across:
  - small random LU / Cholesky / QR / SVD properties
  - large-`n` Cholesky CSC public lifecycle same-pattern properties
  - large-`n` LDL^T CSC public lifecycle same-pattern properties
- Re-read the strongest lifecycle-preservation support proofs in:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- Separated:
  - required property-expansion center
  - support-only direct-family follow-through if the batch truly forces it
  - lower-value non-touch property ideas
- Fixed the exact Day 9 implementation contract in writing.
- Recorded the design in working notes and a Day 8 artifact.

### Findings
- Sprint 84 now has one explicit second implementation contract:
  - required Day 9 center:
    - `tests/test_fuzz.c`
  - strongest support-only follow-through if the property batch truly forces
    it:
    - `tests/test_integration.c`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
  - strongest support-only wording if the contract truly forces movement:
    - `docs/maintainer_guide.md`
    - `README.md`
  - lower-value non-touch surfaces:
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
    - `benchmarks/bench_chol_csc.c`
    - `benchmarks/bench_refactor_csc.c`
    - examples and package/install surfaces
- The highest-value deterministic property lane is now fixed:
  - repeated-run public lifecycle invariants on the retained large-`n` CSC
    direct-family flows
  - reorder / factor / solve agreement properties that stay within the
    existing public lifecycle surface
  - residual and invariance properties on touched retained public flows
- The strongest Day 8 clarification is explicit now:
  - Day 9 should deepen deterministic lifecycle/property coverage in the
    existing property owner rather than reopen the Day 6 external differential
    lane
  - Day 9 should not widen into cancellation/error-path proof because that is
    the later Day 10 / Day 11 seam
  - Day 9 should not widen into iterative/eigs property inflation just because
    `tests/test_fuzz.c` already contains small QR/SVD random properties
  - benchmark and example surfaces remain non-oracle surfaces
  - maintainer and README wording stay support-only unless the landed property
    batch truly changes the maintained assurance reading

### Validation
- Re-read the Day 7 rerank and the live `tests/test_fuzz.c` property owner.
- Rechecked the strongest lifecycle-preservation support proofs in
  `tests/test_integration.c`, `tests/test_chol_csc.c`, and `tests/test_ldlt.c`
  so the property batch remains bounded to the right owner.
- Rechecked maintainer and README proof-ownership wording to keep support
  movement explicitly optional.

### Day 8 Exit State
- Sprint 84 now has one exact second implementation contract.
- Day 9 can stay bounded to deterministic seeded-property expansion in
  `tests/test_fuzz.c`.
- Later failure-path numerical proof and later-family assurance adoption remain
  explicitly deferred.

## Day 9 - Seeded Property Expansion Batch

### Goal
Land the bounded deterministic seeded-property expansion batch on the retained
large-`n` direct-family lifecycle owner.

### Actions
- Added one shared retained residual helper in `tests/test_fuzz.c`:
  - `property_assert_rel_residual_small`
- Added one bounded large-`n` Cholesky CSC property:
  - `test_property_large_n_cholesky_csc_reorder_repeat_solve_agreement`
- Added one bounded large-`n` LDL^T CSC property:
  - `test_property_large_n_ldlt_csc_reorder_repeat_solve_agreement`
- Kept the batch strictly inside the Day 8 fence:
  - no movement in `tests/test_integration.c`
  - no movement in `tests/test_chol_csc.c`
  - no movement in `tests/test_ldlt.c`
  - no movement in `docs/maintainer_guide.md`
  - no movement in `README.md`
- Ran the full required validation gate for a proof-surface widening batch.
- Recorded the batch in working notes and a Day 9 artifact.

### Findings
- Sprint 84 Day 9 landed one bounded deterministic property-expansion batch:
  - required implementation center:
    - `tests/test_fuzz.c`
  - strongest support-only follow-through that was not needed:
    - `tests/test_integration.c`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `docs/maintainer_guide.md`
    - `README.md`
- The new Cholesky property now proves, on retained large-`n` CSC-backed SPD
  lifecycle flows:
  - `SPARSE_REORDER_NONE` and `SPARSE_REORDER_AMD` explicit public lifecycle
    paths agree on the solved vector
  - repeated solves on the same analyzed/factored state remain numerically
    invariant
  - same-pattern refactor followed by repeated solve preserves that agreement
  - retained relative residuals stay small on both reorder lanes
- The new LDL^T property now proves the same bounded invariants on retained
  large-`n` CSC-backed indefinite lifecycle flows:
  - reorder agreement across `NONE` vs `AMD`
  - repeated-solve invariance on the same factor state
  - same-pattern refactor agreement
  - retained relative residual smallness on both lanes
- The useful Day 9 clarification is explicit now:
  - Sprint 84 now has deeper deterministic lifecycle/property coverage on the
    retained direct-family large-`n` public lifecycle seam
  - the strongest next seam is no longer basic property depth in
    `tests/test_fuzz.c`
  - later failure-path numerical proof remains separate work
  - iterative/eigs external adoption remains later work

### Validation
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- Maintained reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
- Representative retained outputs:
  - `test_fuzz` = `28 / 28`, `20544` assertions
  - reviewed CMake `test_fuzz` passed in `29.71 sec`
  - reviewed CMake `Total Test time (real)` = `454.57 sec`
  - reviewed CMake `test_reorder_nd` remained the dominant runtime anchor at
    `325.96 sec`

### Day 9 Exit State
- Sprint 84 now has one landed bounded deterministic seeded-property expansion
  batch.
- The retained large-`n` direct-family lifecycle owner now proves reorder
  agreement, repeated-solve invariance, and residual smallness in addition to
  the earlier same-pattern public-vs-one-shot alignment.
- Later failure-path numerical proof remains the next assurance seam.

## Day 10 - Failure-Path Numerical Proof Design

### Goal
Fix the bounded cancellation, error-path, and stress-fixture proof contract
for the most fragile retained public lifecycle guarantees after the Day 9
property batch.

### Actions
- Re-read the Day 9 landed property batch and the remaining fragile lifecycle
  seams it did not try to cover.
- Re-scanned the strongest current failure-path proof-owner surface in:
  - `tests/test_integration.c`
- Rechecked the strongest support-only family-local proof surfaces in:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- Re-separated:
  - required failure-path proof center
  - support-only direct-family and later-family follow-through if truly forced
  - lower-value non-touch stress ideas
- Fixed the exact Day 11 implementation contract in writing.
- Recorded the design in working notes and a Day 10 artifact.

### Findings
- Sprint 84 now has one explicit third implementation contract:
  - required Day 11 center:
    - `tests/test_integration.c`
  - strongest support-only follow-through if the failure-path batch truly
    forces it:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `docs/maintainer_guide.md`
  - lower-value non-touch surfaces:
    - `tests/test_iterative.c`
    - `tests/test_eigs.c`
    - `tests/test_fuzz.c`
    - benchmark and example surfaces
    - package/install/export surfaces
- The highest-value remaining fragile lifecycle seam is now fixed to:
  - cancellation and callback short-circuit guarantees on retained public
    direct and solver workflows
  - error-path factor / solve / refactor preservation, especially when callers
    retry after failure
  - zeroed-state, mismatched-state, and old-factor-preservation guarantees on
    the shared public lifecycle path
- The strongest Day 10 clarification is explicit now:
  - Day 11 should not reopen Day 6 external differential work
  - Day 11 should not reopen Day 9 property depth in `tests/test_fuzz.c`
  - Day 11 should stay centered on the existing integration proof owner
    because that file already owns the authoritative public lifecycle failure
    semantics across direct, iterative, and eigensolver cancellation surfaces
  - family-local tests remain support-only unless the batch exposes one truly
    local contradiction
  - maintainer/support wording remains optional unless the landed batch changes
    the maintained assurance reading

### Validation
- This was a docs-only design day, so no build/test rerun was required.
- The design was grounded in direct rereads of:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - the Day 9 working-notes close record

### Day 10 Exit State
- Sprint 84 now has one explicit third implementation contract.
- Day 11 can stay bounded to the retained integration failure-path proof owner.
- Policy / CI / support-surface alignment remains explicitly deferred until
  after the landed failure-path batch.

## Day 11 - Failure-Path Numerical Proof Batch

### Goal
Land the bounded failure-path numerical proof batch on the shared public
lifecycle owner without reopening family-local or support-only surfaces.

### Actions
- Re-read the Day 10 failure-path design and re-scanned the existing public
  lifecycle proof seams in:
  - `tests/test_integration.c`
- Identified the strongest remaining uncovered lifecycle contradiction:
  - retrying a later good same-pattern refactor after a failed refactor while
    reusing the same public `analysis` / `factors` objects
- Added three bounded retry-after-failure proofs:
  - linked-list Cholesky public lifecycle retry after `SPARSE_ERR_NOT_SPD`
  - CSC Cholesky public lifecycle retry after `SPARSE_ERR_NOT_SPD`
  - AMD LDL^T public lifecycle retry after rejected nnz drift
- Kept the batch centered entirely in the shared integration proof owner:
  - no `src/` production code changes
  - no `tests/test_chol_csc.c` or `tests/test_ldlt.c` follow-through
  - no maintainer or README wording follow-through
- Ran the required validation gate for a substantial proof batch.
- Recorded the batch in working notes and a Day 11 artifact.

### Findings
- Sprint 84 Day 11 landed one bounded failure-path numerical proof batch:
  - required implementation center:
    - `tests/test_integration.c`
  - strongest support-only follow-through that was not needed:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `docs/maintainer_guide.md`
- The landed retry-after-failure proofs are explicit now:
  - `test_public_lifecycle_refactor_failure_allows_retry`
  - `test_public_lifecycle_cholesky_csc_refactor_failure_allows_retry`
  - `test_public_lifecycle_ldlt_refactor_failure_allows_retry_amd`
- The shared public lifecycle owner now proves one stronger retained contract:
  - a failed refactor preserves the previously valid factor state
  - callers can still use that preserved state to solve the original problem
  - callers can then retry with a later good same-pattern matrix on the same
    public `analysis` / `factors` objects and recover correct solves
- The exact failure modes now covered are:
  - Cholesky linked-list retry after a bounded `SPARSE_ERR_NOT_SPD` failure
  - Cholesky CSC retry after a bounded `SPARSE_ERR_NOT_SPD` failure
  - LDL^T retry after a bounded rejected-pattern / nnz-drift failure under AMD
- The strongest Day 11 clarification is explicit now:
  - the remaining fragile lifecycle contradiction was on the shared retry
    semantics, not in family-local direct proofs
  - the integration proof owner was sufficient to close it
  - no support-surface wording or policy movement was required just because the
    failure-path owner deepened

### Validation
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- Maintained reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
- Representative retained outputs:
  - `test_integration` = `56 / 56`
  - reviewed CMake `Total Test time (real)` = `512.76 sec`
  - reviewed CMake `test_reorder_nd` remained the dominant runtime anchor at
    `366.43 sec`

### Day 11 Exit State
- Sprint 84 now has one landed bounded failure-path numerical proof batch.
- The shared public lifecycle owner proves preserved-old-factor solve behavior
  and successful later retry after failed refactor on linked-list Cholesky,
  CSC Cholesky, and AMD LDL^T lanes.
- Policy / CI / support-surface alignment remains later work because the
  landed batch did not force wording or surface-owner movement.

## Day 12 - Policy / CI / Support-Surface Alignment

### Goal
Fix the final Sprint 84 proof-owner, CI-reading, and Day 13 validation-queue
map after the Day 6, Day 9, and Day 11 assurance landings, while keeping
support-only surfaces bounded to what the sprint actually changed.

### Actions
- Re-read the landed Sprint 84 assurance owners:
  - `tests/test_chol_csc.c`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
- Rechecked the strongest retained support and later-family proof owners:
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- Rechecked the authoritative wording owners:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `.github/workflows/windows-ci.yml`
- Rechecked the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked representative reviewed examples and benchmark/reporting owners:
  - `build/quality-review-cmake/example_analysis`
  - `build/quality-review-cmake/example_basic_solve`
  - `build/quality-review-cmake/bench_refactor_csc`
  - `build/quality-review-cmake/bench_svd`
  - `make bench-canonical-report`
- Fixed the exact Day 13 validation queue in writing.

### Findings
- No new support-only edit is needed before the full sweep:
  - `docs/maintainer_guide.md` already remains truthful about the bounded
    direct-family external differential lane, the seeded property lane, and
    the public lifecycle oracle lane
  - `README.md` already remains truthful about the same proof-owner split and
    the Windows `test_fuzz` exclusion caveat
  - `.github/workflows/windows-ci.yml` already remains truthful that
    `test_fuzz` stays outside the reviewed Windows subset
- The final Sprint 84 proof-owner map is now explicit:
  - `tests/test_chol_csc.c` owns the bounded direct-family maintained external
    differential lane on the SPD Cholesky CSC path
  - `tests/test_fuzz.c` owns the bounded seeded generative lifecycle/property
    follow-through on the large-`n` CSC-backed Cholesky and LDL^T lanes
  - `tests/test_integration.c` owns the public lifecycle oracle surface for
    cancellation, preservation, rejection, repeated-run, and retry-after-
    failure guarantees
  - `tests/test_ldlt.c` remains the family-local LDL^T direct proof owner, not
    a Sprint 84 external-differential center
  - `tests/test_iterative.c` and `tests/test_eigs.c` remain retained
    later-family proof owners, not Sprint 84 adopted external-differential
    centers
- The representative executable support map is explicit now:
  - reviewed CMake proof owners:
    - `test_chol_csc`
    - `test_ldlt`
    - `test_fuzz`
    - `test_integration`
    - `test_iterative`
    - `test_eigs`
  - representative examples:
    - `example_analysis`
    - `example_basic_solve`
  - benchmark/reporting owners:
    - `bench_refactor_csc`
    - `bench_svd`
    - `make bench-canonical-report`
- The CI/platform-confidence reading is explicit now:
  - Linux and macOS reviewed/local paths still exercise `test_fuzz`
  - Windows still excludes `test_fuzz` from the reviewed subset
  - Sprint 84 therefore widens local and Linux/macOS assurance depth without
    creating new reviewed-Windows evidence claims
- Install/export proof remains explicitly out of scope for Day 13 because
  Sprint 84 did not move package, install, export, or runtime-package
  mechanics.

### Validation
- Rechecked `ctest -N --test-dir build/quality-review-cmake` and confirmed the
  live reviewed parity anchor remains `53`.
- Rechecked the presence of the Day 13 focused reviewed binaries and
  representative examples/benchmarks.
- Rechecked the maintained canonical benchmark-report command surface with
  `make -n bench-canonical-report`.

### Day 12 Exit State
- No further support-only edit is required before the full Sprint 84 sweep.
- The final proof-owner map and CI truth map are explicit in writing.
- The Day 13 validation queue is fixed with no ambiguity.
