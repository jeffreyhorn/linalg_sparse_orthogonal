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
