# Sprint 99 Working Notes

## Day 1 - Sprint 99 Scope & Closeout Baseline

### Goal

Open Sprint 99 with a current final-integration baseline from the merged
Sprint 98 tree. Day 1 does not re-audit every contradiction class yet, run the
comparison sweep, or decide on fixes. It defines the closeout workstreams,
authoritative inputs, validation expectations, and landing order that Days
2-14 should follow.

### Actions

- Re-read the Sprint 99 section of
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read Sprint 99 Day 1 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Sprint 90 claim fence and target-state retrospective.
- Re-read the Sprints 91-98 retrospective close states and residual queues.
- Re-read the Sprint 98 closeout handoff artifact for the immediate Sprint 99
  queue.
- Recorded authoritative inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the closeout baseline and landing order in
  `artifacts/day1-closeout-baseline.md`.

### Closeout Workstreams

Sprint 99 is organized around seven evidence-backed closeout workstreams:

1. End-state re-audit.
2. Final competitive comparison sweep.
3. Final fix decision.
4. Final bounded fix batch if evidence requires it.
5. Residual queue finalization.
6. Full validation and reporting sweep.
7. Sprint 99 and Epic 9 closeout package.

### Findings

- Sprint 90 froze the durable Epic 9 claim fence: bounded state-of-the-art
  sparse linear algebra library, maintained correctness/package/runtime
  comparison lanes, and explicit non-claims for fake platform symmetry, broad
  shared-library maturity, broad complex/mixed precision, and benchmark
  supremacy.
- Sprints 91-98 landed the main Epic 9 workstreams: compressed-first product
  convergence, dense/backend maturity, runtime/threading reduction, capability
  surface modernization, public narrative cleanup, maintainability extraction,
  build/package convergence, and assurance/comparison expansion.
- Sprint 98 handed Sprint 99 a concrete final queue around external
  correctness breadth, runtime/fill artifact classification, coverage
  topology, and CI/support-surface calibration.
- The final-fix decision must be evidence-gated. Broad polish, speculative
  expansion, and future comparison architecture belong in the residual queue
  unless the Sprint 99 audit finds a live closeout contradiction.
- Linux remains the strongest reviewed source of truth. macOS and Windows are
  calibrated proof surfaces with explicit scope limits.

### Validation Expectations

- Day 1 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only baseline pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 1 Exit State

Sprint 99 now has a scoped closeout baseline, authoritative input list, seven
workstreams, validation expectations, and a landing order. Day 2 can re-audit
the original Epic 9 contradiction classes against the live tree without
reopening broad implementation scope by default.

## Day 2 - Epic 9 Contradiction Class Re-audit

### Goal

Reconstruct the original Epic 9 contradiction classes from Sprint 90 and
compare each one against the live Sprint 99 tree. Day 2 classifies each class
as resolved, partially resolved, still active, or deliberate non-claim before
the final comparison scope is frozen.

### Actions

- Re-read Sprint 99 Day 2 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read Sprint 90 Day 3, Day 4, Day 5, and Day 7 artifacts to reconstruct
  the original product, backend, capability, maintainability, workflow,
  comparison, and non-goal contradiction classes.
- Checked current public and maintainer claim surfaces:
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - public headers under `include/`
- Checked current implementation and proof-owner shape with line-count and
  claim-boundary scans.
- Checked current reviewed validation, install/export, benchmark/reporting,
  coverage, and CI proof surfaces.
- Recorded the Day 2 end-state re-audit in
  `artifacts/day2-end-state-contradiction-reaudit.md`.

### Findings

- The original compressed-first product-model contradiction is partially
  resolved. Public CSR/CSC compressed-first entry paths and documentation now
  exist, but the front door and matrix shell still deliberately retain the
  orthogonal linked-list identity.
- The backend maturity contradiction is partially resolved. Dense/backend
  seams and LDLT CSC support are stronger, but broad portable backend maturity
  remains a residual, not a final-fix blocker.
- The capability breadth contradiction is partially resolved. Scalar/index
  seams are clearer and solver/eigensolver breadth improved, but broad complex
  and mixed precision remain deliberate non-claims.
- Runtime and threading are partially resolved. Reviewed runtime-control and
  benchmark signals are stronger, but OpenMP remains localized and bounded.
- Maintainability and proof-owner concentration remain active residuals.
  Several largest owners are smaller than the Sprint 90 baseline, but major
  source and test files remain large enough that Sprint 99 should not call the
  class fully resolved.
- Chronology cleanup is partially resolved. Public surfaces are cleaner, but
  sprint-era comments and historical proof-owner names remain in lower-level
  tests and implementation comments.
- Build/package/workflow duplication is partially resolved. The source-list
  manifest/checker and static-first package proof materially reduce drift, but
  Make, CMake, and CI remain intentionally independent reviewed surfaces.
- Comparison depth is partially resolved. Cholesky CSC and LDLT CSC now have
  maintained external dense-reference lanes, and reorder/fill has a bounded
  artifact lane. Broader solver-family, eigensolver, QR, SVD, and runtime
  ecosystem comparison remain residual.

### Preliminary Final-Fix Queue

Day 2 did not find a clear implementation blocker that must be fixed before
Sprint 99 can proceed. The strongest candidates for Days 3-6 review are:

- stale or overstated closeout wording if Day 3-5 evidence finds it
- workflow/test-count or install/export proof drift if validation exposes it
- benchmark/reporting claim drift if final evidence commands disagree with
  current documentation
- small support-surface corrections needed to keep non-claims explicit

### Validation

- Day 2 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only re-audit pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 2 Exit State

Sprint 99 now has a live contradiction-class map. The Day 3 comparison scope
should focus on proving or falsifying the partially resolved classes, not on
assuming a final fix batch is necessary.

## Day 3 - Competitive Comparison Scope Freeze

### Goal

Freeze the final Sprint 99 comparison lanes before executing evidence
commands. Day 3 selects the correctness, runtime/fill, package/install,
usability/documentation, and workflow/CI surfaces that Days 4-5 should use to
support or limit the final Epic 9 closeout claims.

### Actions

- Re-read Sprint 99 Day 3 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Sprint 90 comparison-and-measurement contract.
- Re-read Sprint 94, Sprint 97, and Sprint 98 closeout artifacts for current
  capability, package, workflow, and assurance proof owners.
- Checked current Makefile and CMake registrations for selected commands.
- Checked README, INSTALL, benchmark docs, maintainer guide, and workflows for
  the current claim boundaries.
- Recorded the frozen Day 3 comparison scope in
  `artifacts/day3-final-comparison-scope.md`.

### Frozen Evidence Lanes

- Correctness evidence:
  - Cholesky CSC external dense reference on `nos4` and `bcsstk04`
  - LDLT CSC external dense reference on `kkt5` and `kkt10`
  - strongest reviewed local baseline as the aggregate correctness guard
- Runtime/fill evidence:
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`
  - output interpreted as local calibration and artifact generation only
- Package/install/export evidence:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - static-first package shape and CMake consumer export proof
- Usability/documentation evidence:
  - stale-claim scans over README, INSTALL, benchmark docs, maintainer guide,
    public headers, and workflow comments
  - no command output should be converted into broader claims than its proof
    owner supports
- Workflow/CI evidence:
  - local `make quality-review-full` remains the strongest reviewed baseline
  - CI scope is interpreted asymmetrically: Linux strongest, macOS narrower
    reviewed plus supplemental confidence, Windows reviewed CMake-first subset

### Claim Boundaries

- Allowed:
  - bounded external correctness evidence for Cholesky CSC and LDLT CSC lanes
  - static-first install/export and CMake consumer proof
  - bounded local runtime/fill calibration on selected workload slices
  - stronger Epic 9 coherence and maintainability than the Sprint 90 baseline
- Disallowed:
  - broad solver-family external correctness parity
  - portable timing superiority
  - universal reorder/fill superiority
  - broad shared-library or dynamic ABI maturity
  - symmetric Linux/macOS/Windows reviewed parity
  - broad complex or mixed-precision maturity
- Deferred:
  - iterative, eigensolver/LOBPCG, QR, SVD, and broader LDLT external
    comparison architecture
  - generated reorder/fill reporting if repeated captures prove it is worth
    maintaining
  - expanded coverage ownership or thresholds

### Validation

- Day 3 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only scope freeze.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 3 Exit State

Sprint 99 now has a frozen comparison scope. Days 4-5 can execute evidence
commands against named proof owners and classify results as closeout-ready,
residual, or final-fix candidates without widening the claim surface.

## Day 4 - Final Correctness & Runtime Evidence Sweep

### Goal

Execute the Day 3 correctness and runtime/fill evidence commands, capture
outputs with enough detail for closeout, and classify each result as
closeout-ready, residual, or final-fix candidate.

### Actions

- Re-read Sprint 99 Day 4 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Day 3 final comparison scope.
- Ran LDLT external dense-reference helper positive and negative fixtures.
- Ran focused Cholesky CSC and LDLT CSC test binaries.
- Ran the bounded reorder/fill calibration command.
- Ran the canonical benchmark report command and inspected the report
  manifest/index.
- Recorded Day 4 evidence in
  `artifacts/day4-correctness-runtime-evidence.md`.

### Results

- `python3 tests/ldlt_external_dense_reference.py kkt5` passed and returned
  `OK 5` with the reference solution `1..5`.
- `python3 tests/ldlt_external_dense_reference.py kkt10` passed and returned
  `OK 10` with the expected `1..10` solution to floating-point precision.
- `python3 tests/ldlt_external_dense_reference.py nope` failed closed with
  `ERROR unknown fixture nope` and exit `1`.
- `make build/test_chol_csc && ./build/test_chol_csc` passed:
  - 92 tests run
  - 0 failed
  - 0 skipped
  - external `nos4` row: `max|x-x_ref|=4.690e-13`,
    `rel_residual=3.907e-15`
  - external `bcsstk04` row: `max|x-x_ref|=3.224e-11`,
    `rel_residual=3.010e-16`
- `make build/test_ldlt_csc && ./build/test_ldlt_csc` passed:
  - 98 tests run
  - 0 failed
  - 0 skipped
  - external `kkt5` row: `max|x-x_ref|=0.000e+00`,
    `rel_residual=0.000e+00`
  - external `kkt10` row: `max|x-x_ref|=3.553e-15`,
    `rel_residual=2.292e-16`
- `make bench-reorder-sprint86` passed and emitted the bounded two-fixture
  reorder/fill slice for `bcsstk14` and `Pres_Poisson`.
- `make bench-canonical-report` passed and wrote:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`
  - `build/bench-reports/canonical/bench_chol_csc.csv`
  - `build/bench-reports/canonical/bench_iterative_reuse.csv`
  - `build/bench-reports/canonical/bench_eigs_reuse.csv`
  - `build/bench-reports/canonical/index.tsv`
  - `build/bench-reports/canonical/manifest.txt`

### Classification

- Closeout-ready:
  - Cholesky CSC external dense-reference lane
  - LDLT CSC external dense-reference lane
  - bounded reorder/fill artifact generation
  - canonical benchmark report generation
- Residual/non-claim:
  - broader solver-family external correctness comparison
  - broader LDLT Matrix Market or indefinite corpus comparison
  - portable timing thresholds or benchmark superiority claims
  - generated reorder/fill report target beyond the existing artifact command
- Final-fix candidates:
  - none from Day 4 evidence

### Validation

- Day 4 changed planning documentation only after running evidence commands.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only evidence capture.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 4 Exit State

The selected correctness and runtime/fill evidence lanes are closeout-ready.
Day 5 can proceed to package, usability, documentation, and workflow evidence
without carrying a Day 4 final-fix candidate.

## Day 5 - Package, Usability & Workflow Evidence Sweep

### Goal

Execute the package/install/export proof commands and inspect documentation,
public narrative, maintainer guidance, and workflow scope wording against the
Day 3 final comparison scope.

### Actions

- Re-read Sprint 99 Day 5 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Day 3 final comparison scope and Day 4 evidence results.
- Ran the Make install/export proof script.
- Ran the CMake install/export and consumer proof script.
- Ran stale-claim scans across public docs, maintainer docs, public headers,
  and workflows.
- Ran boundary scans for static-first, benchmark, platform, and Windows CTest
  expectation language.
- Inspected CMake test registration and Windows workflow exclusion logic to
  verify the expected Windows CTest count.
- Recorded Day 5 evidence in
  `artifacts/day5-package-usability-workflow-evidence.md`.

### Results

- `bash tests/test_install.sh` passed:
  - 14 passed
  - 0 failed
  - static library installed
  - no shared-library artifacts installed
  - all 19 headers installed
  - `pkg-config` consumer and maintained example ran
  - uninstall removed library, headers, and pkg-config file
- `bash tests/test_cmake_install.sh` passed:
  - 16 passed
  - 0 failed
  - 0 skipped
  - static library, headers, CMake config/version/targets, and `sparse.pc`
    installed
  - `examples/cmake_example` configured, built, and ran through
    `find_package(Sparse)`
  - exact-version package lookup passed and mismatched-version lookup was
    rejected
- stale-claim scan found only negative/boundary language, not positive
  unsupported claims.
- boundary scan found the expected static-first, benchmark, Linux/macOS/Windows,
  and Windows expected-count fences.
- CMake has 54 source-level `add_sparse_test(...)` registrations. Windows
  excludes exactly three staged tests:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
  This matches `.github/workflows/windows-ci.yml` expected count `51`.

### Classification

- Closeout-ready:
  - Make install/export proof
  - CMake install/export and consumer proof
  - static-first package story
  - docs/usability non-claim boundaries
  - workflow platform-scope wording
  - Windows expected CTest count
- Residual/non-claim:
  - shared-library package contract
  - dynamic ABI guarantee
  - Windows install-validation lane
  - Windows Makefile parity
  - symmetric platform parity
  - portable benchmark/timing claims
- Final-fix candidates:
  - none from Day 5 evidence

### Validation

- Day 5 changed planning documentation only after running evidence commands.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only evidence capture.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 5 Exit State

Package, usability, documentation, and workflow evidence are closeout-ready.
Day 6 can make the final fix/no-fix decision with no Day 4 or Day 5
final-fix candidates outstanding.

## Day 6 - Final Fix Decision

### Goal

Decide whether Sprint 99 needs a last bounded implementation or support fix
batch before closeout. The decision must be based only on Day 2 contradictions
and Day 4-5 evidence, not on broad improvement pressure.

### Actions

- Re-read Sprint 99 Day 6 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Reviewed Day 2 candidate final-fix queue.
- Reviewed Day 4 correctness/runtime evidence and final-fix candidate list.
- Reviewed Day 5 package/usability/workflow evidence and final-fix candidate
  list.
- Rejected broad residual work as post-Epic-9 carry-forward or deliberate
  non-claim.
- Recorded the Day 6 fix/no-fix decision in
  `artifacts/day6-final-fix-decision.md`.

### Decision

No final implementation/support fix batch is selected.

Reasons:

- Day 4 correctness/runtime evidence produced no final-fix candidates.
- Day 5 package/usability/workflow evidence produced no final-fix candidates.
- Day 2 unresolved items are residual or deliberate non-claims, not blockers
  to truthful Epic 9 closeout.
- Starting a final fix batch would widen scope without an evidence-backed
  closeout contradiction.

### Residual Queue Draft

Carry forward:

- broader LDLT CSC Matrix Market or indefinite corpus comparison
- iterative solver external comparison design
- eigensolver/LOBPCG external comparison design
- QR/SVD external comparison architecture
- generated reorder/fill reporting only if repeated captures justify it
- continued large-source and giant-test extraction
- remaining chronology cleanup in lower-level proof surfaces

Preserve as non-claims:

- broad complex or mixed-precision maturity
- shared-library-first or dynamic ABI maturity
- symmetric platform parity
- Windows Makefile parity or install-validation lane
- portable timing superiority or universal reorder/fill superiority
- every-solver-family external correctness comparison

### Validation

- Day 6 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only decision artifact.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 6 Exit State

Sprint 99 has a written no-fix decision. Days 7 and 8 should become no-op
implementation evidence days unless new information appears; they should not
introduce edits outside this decision without first reopening the Day 6
boundary.

## Day 7 - Final Bounded Fix Batch 1

### Goal

Execute the first final-fix-batch day under the Day 6 no-fix decision. Because
no final fix batch was selected, Day 7 records no-op implementation evidence,
confirms no new blocker has appeared, and preserves the claim and validation
boundaries.

### Actions

- Re-read Sprint 99 Day 7 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Day 6 final fix/no-fix decision and Day 7-8 boundary.
- Confirmed Day 4 and Day 5 produced no final-fix candidates.
- Confirmed no new user request, command failure, or evidence artifact
  reopened the Day 6 decision.
- Did not edit source, headers, scripts, build files, workflows, benchmarks,
  tests, or public docs.
- Recorded the no-op implementation evidence in
  `artifacts/day7-final-fix-batch1-noop.md`.

### No-op Implementation Evidence

No implementation/support batch was started.

The Day 7 risk review keeps these items classified as residual or non-claim:

- broader external correctness comparison
- generated reorder/fill report target
- large-source and giant-test extraction
- remaining lower-level chronology cleanup
- broad complex/mixed-precision maturity
- shared-library-first or dynamic ABI package maturity
- symmetric platform parity
- portable timing or universal reorder/fill superiority

### Validation

- Day 7 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only no-op evidence pass.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 7 Exit State

Day 7 preserved the Day 6 no-fix decision. Day 8 should close the no-op fix
batch formally, reconcile proof-owner readiness for broad validation, and hand
the residual items to Day 9 classification.

## Day 8 - Final Bounded Fix Batch 2

### Goal

Close the final-fix-batch window under the Day 6 no-fix decision. Because no
fix batch was selected or reopened, Day 8 reconciles proof-owner readiness,
confirms no adjacent owner needs edits, and prepares the residual queue for
Day 9.

### Actions

- Re-read Sprint 99 Day 8 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read Day 6 final fix/no-fix decision and Day 7 no-op evidence.
- Confirmed no accepted fix candidates remain unresolved.
- Reconciled adjacent closeout proof owners:
  - correctness owners from Day 4
  - benchmark/report owners from Day 4
  - package/install/export owners from Day 5
  - docs/usability/workflow owners from Day 5
- Confirmed no implementation, support, public-doc, or workflow edits are
  justified before broad validation.
- Recorded final fix-batch closeout in
  `artifacts/day8-final-fix-closeout-noop.md`.

### Proof-Owner Readiness

- Correctness proof owners are ready for broad validation:
  - Cholesky CSC external correctness
  - LDLT CSC external correctness
- Runtime/fill and reporting owners are ready for broad validation:
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`
- Package proof owners are ready for broad validation:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- Workflow and documentation proof owners are closeout-safe:
  - Linux/macOS/Windows scope remains explicit
  - static-first package story remains explicit
  - benchmark timing remains threshold-free and local
  - non-claims remain visible

### Validation

- Day 8 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only closeout pass.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 8 Exit State

The no-op final-fix period is closed. There are no accepted fixes to implement
before validation. Day 9 should finalize the residual queue and non-claim list
from the rejected candidates and evidence-backed deferrals.

## Day 9 - Residual Queue Classification

### Goal

Finalize the post-Epic-9 residual queue by consolidating carry-forward work
from Sprints 90-98 and the Sprint 99 evidence sweep. Day 9 separates real
future work from deliberate non-claims, already-resolved items, and unsupported
claims that would require removal.

### Actions

- Re-read Sprint 99 Day 9 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Reviewed residual sections from Sprints 90-98 retrospectives and closeout
  artifacts.
- Reviewed the Day 2 contradiction re-audit, Day 6 no-fix decision, and Day 8
  no-op fix closeout.
- Removed duplicate residual entries by grouping them under:
  - comparison depth
  - runtime/fill reporting
  - maintainability extraction
  - chronology cleanup
  - package/platform boundaries
  - coverage/workflow boundaries
- Classified each item as post-Epic-9 carry-forward, deliberate non-claim,
  already resolved, or unsupported claim to remove.
- Recorded the final queue in
  `artifacts/day9-final-residual-queue.md`.

### Final Classification

- Post-Epic-9 carry-forward:
  - broader external comparison architecture
  - generated reorder/fill reporting only if repeated captures justify it
  - continued large-source and giant-test extraction
  - lower-level chronology cleanup where it still improves current
    maintainability
  - optional build/test/benchmark/example registration convergence only where
    proof clarity is preserved
- Deliberate non-claims:
  - broad complex/mixed-precision maturity
  - shared-library-first package contract
  - dynamic ABI guarantee
  - symmetric platform parity
  - Windows Makefile parity or install-validation lane
  - portable timing or universal reorder/fill superiority
  - every-solver-family external correctness comparison
- Unsupported claims to remove:
  - none found by Days 4-5 scans
- Already resolved:
  - Sprint 90 planning package, target state, and comparison contract
  - Sprint 91 compressed-first constructor/public workflow proof
  - Sprint 92 bounded backend widening and observability
  - Sprint 93 bounded runtime/control cleanup
  - Sprint 94 bounded scalar/index capability widening
  - Sprint 95 public narrative/support cleanup
  - Sprint 96 selected source/proof-owner extraction
  - Sprint 97 source-list guard and static-first package proof
  - Sprint 98 LDLT CSC external correctness and bounded reorder/fill artifact
  - Sprint 99 Day 4-5 evidence sweeps and Day 6 no-fix decision

### Validation

- Day 9 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only residual classification.
- Follow-up hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_9/SPRINT_99`

### Day 9 Exit State

Sprint 99 has a final residual queue and non-claim list. No unresolved
closeout blocker is hidden in the queue. Day 10 can run the strongest reviewed
validation baseline from a no-fix, evidence-backed branch state.

## Day 10 - Full Reviewed Validation Sweep

### Goal

Run the strongest reviewed local validation baseline after all final-fix and
residual-queue decisions. Day 10 establishes the validated tree state that
later closeout writing must build from.

### Actions

- Re-read Sprint 99 Day 10 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Ran `make quality-review-full`.
- Waited for both reviewed lanes to complete:
  - Makefile `quality-review`
  - CMake `quality-review-cmake`
- Captured the validation result, skip/environment notes, and go/no-go
  decision in `artifacts/day10-reviewed-validation.md`.

### Validation Result

`make quality-review-full` passed.

- Makefile reviewed path:
  - `format-check` passed.
  - `lint` passed, including tooling-build, strict warning compilation,
    `clang-tidy`, and `cppcheck`.
  - `make test` passed with `All tests passed.`
  - `deadcode-check` passed its report-completeness gate.
- CMake reviewed parity path:
  - configure and clean rebuild passed.
  - `ctest -N` registered 54 tests.
  - Makefile/CMake test-count parity passed with 54 tests on each side.
  - full CTest passed: 100% tests passed, 0 tests failed out of 54.

### Implementation-Day Checks

- Day 10 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- A separate `make format && make lint && make test` chain is not required for
  the docs-only Day 10 changes.
- The stronger reviewed baseline, `make quality-review-full`, was run and
  passed.

### Day 10 Exit State

Sprint 99 now has a full reviewed validation record. Closeout package writing
can proceed from a branch state validated by both the Makefile reviewed lane
and the CMake reviewed parity lane.

## Day 11 - Install, Export, Example, And Reporting Validation

### Goal

Validate the install/export, example, consumer, benchmark, and reporting
surfaces that support the final Epic 9 product story. Day 11 re-runs the
selected Day 3 package and reporting commands after the Day 10 reviewed
baseline passed.

### Actions

- Re-read Sprint 99 Day 11 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Re-read the Day 3 final comparison scope and Day 5 package/usability
  evidence artifacts.
- Ran the Make install/export proof script.
- Ran the CMake install/export and consumer proof script.
- Built the full example set.
- Ran a representative example smoke set.
- Ran bounded reorder/fill calibration.
- Ran canonical benchmark report generation.
- Recorded command results, output locations, skipped lanes, and non-claims in
  `artifacts/day11-surface-validation.md`.

### Validation Result

- `bash tests/test_install.sh` passed:
  - 14 passed
  - 0 failed
  - static library and 19 headers installed
  - no shared-library artifacts installed
  - `pkg-config` consumer compiled, linked, and ran
  - maintained example source compiled and ran with the installed package
  - uninstall removed installed artifacts
- `bash tests/test_cmake_install.sh` passed:
  - 16 passed
  - 0 failed
  - 0 skipped
  - CMake configure, build, install, and export files passed
  - `examples/cmake_example` configured via `find_package(Sparse)`, built, and
    ran
  - exact version lookup worked and mismatched version lookup was rejected
- `make examples` passed and built 12 example binaries.
- Representative examples passed:
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- `make bench-reorder-sprint86` passed and emitted bounded `bcsstk14` and
  `Pres_Poisson` reorder/fill rows.
- `make bench-canonical-report` passed and wrote
  `build/bench-reports/canonical`.

### Output Locations

Canonical benchmark report files:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/index.tsv`
- `build/bench-reports/canonical/manifest.txt`

### Skipped Or Unvalidated Lanes

- Windows Makefile parity remains a non-claim.
- Windows install-validation remains a non-claim.
- shared-library-first packaging remains a non-claim.
- dynamic ABI guarantees remain a non-claim.
- package-manager integration remains residual.
- symmetric platform parity remains a non-claim.
- portable timing and universal reorder/fill superiority remain non-claims.

### Implementation-Day Checks

- Day 11 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- A separate `make format && make lint && make test` chain is not required for
  the docs-only Day 11 changes.
- The selected package, consumer, example, benchmark, and reporting validation
  commands all passed.

### Day 11 Exit State

Sprint 99 now has current surface-validation evidence for package,
install/export, CMake consumer, examples, bounded reorder/fill calibration, and
canonical benchmark reporting. Closeout package writing can proceed with the
static-first package claim, threshold-free benchmark/reporting claim, and
platform non-claims preserved.

## Day 12 - Final Closeout Evidence Package

### Goal

Consolidate the Sprint 99 Day 1-11 audit, comparison, fix-decision, residual,
validation, package, example, and reporting evidence into one final closeout
evidence package for Day 13 retrospective drafting.

### Actions

- Re-read Sprint 99 Day 12 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Reviewed the Day 1 authoritative inputs and closeout baseline.
- Reviewed the Day 2 contradiction re-audit.
- Reviewed the Day 3 final comparison scope and claim-language boundaries.
- Reviewed Day 4-5 correctness, benchmark, package, usability, and workflow
  evidence.
- Reviewed the Day 6-8 no-fix decision and no-op closeout artifacts.
- Reviewed the Day 9 final residual queue.
- Reviewed the Day 10 reviewed validation record.
- Reviewed the Day 11 surface-validation record.
- Wrote the final closeout evidence package in
  `artifacts/day12-closeout-evidence-package.md`.

### Evidence Package Contents

The Day 12 package includes:

- Day 1-11 evidence index
- resolved contradiction summary
- final competitive calibration summary
- package, consumer, example, and workflow summary
- final validation summary
- residual and non-claim summary
- supported and unsupported closeout-language lists
- implementation-day check decision

### Final Closeout Position

Supported:

- compressed-first entry paths materially improved while the linked-list shell
  remains the mutable compatibility owner
- bounded direct-family/backend maturity improved
- named Cholesky CSC and LDLT CSC external dense-reference lanes are maintained
- static-first install/export and CMake consumer proof are maintained and
  validated
- runtime/fill evidence is bounded, local, and calibration-oriented
- reviewed platform proof remains intentionally asymmetric

Not supported:

- whole-library compressed-first replacement
- broad complex or mixed-precision maturity
- broad backend-neutral acceleration maturity
- shared-library-first packaging or dynamic ABI guarantees
- symmetric Linux/macOS/Windows reviewed parity
- portable timing superiority or universal reorder/fill superiority
- every-solver-family external correctness comparison
- complete elimination of large-source or giant-test concentration

### Validation Position

- Day 10 `make quality-review-full` passed:
  - Makefile reviewed path passed
  - CMake reviewed parity passed
  - CMake and Makefile test counts matched at 54
  - full CTest passed 54/54
- Day 11 selected surface-validation commands passed:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - `make examples`
  - representative examples
  - `make bench-reorder-sprint86`
  - `make bench-canonical-report`

### Implementation-Day Checks

- Day 12 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- A separate `make format && make lint && make test` chain is not required for
  the docs-only Day 12 changes.

### Day 12 Exit State

Sprint 99 now has a final closeout evidence package that cites artifact-backed
evidence, keeps claim limits visible, and carries residual work forward without
weakening the validated baseline. Day 13 can draft the Sprint 99 and Epic 9
retrospectives from this package.

## Day 13 - Sprint 99 And Epic 9 Retrospective Drafts

### Goal

Draft the Sprint 99 retrospective, Epic 9 retrospective, lessons-learned
handoff notes, and Day 14 closeout edit list from the validated Day 12
evidence package.

### Actions

- Re-read Sprint 99 Day 13 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Reviewed the Day 12 closeout evidence package.
- Reviewed recent Epic 9 retrospective formats from Sprints 97 and 98.
- Drafted Sprint 99 retrospective content:
  - definition of done checklist
  - what went well
  - what did not go well
  - final metrics
  - residual deferred debt
  - key deliverables
- Drafted Epic 9 retrospective content:
  - closeout thesis
  - resolved or improved areas
  - unresolved areas
  - metrics
  - lessons
  - handoff guidance
- Drafted lessons-learned and validation notes.
- Drafted the Day 14 closeout edit list.
- Recorded all Day 13 material in
  `artifacts/day13-retrospective-drafts.md`.

### Draft Boundaries

- Day 13 produced drafts only.
- Final `RETROSPECTIVE.md`, Epic retrospective, and handoff documents remain
  Day 14 work.
- The drafts preserve Day 12 claim limits:
  - no full compressed-first replacement claim
  - no broad complex or mixed-precision maturity claim
  - no broad backend-neutral acceleration claim
  - no shared-library-first or dynamic ABI claim
  - no symmetric platform parity claim
  - no portable timing or universal reorder/fill superiority claim
  - no every-solver-family external correctness claim
  - no complete large-source or giant-test elimination claim

### Day 14 Edit List

Day 14 should:

- create final `docs/planning/EPIC_9/SPRINT_99/RETROSPECTIVE.md`
- create final Epic 9 retrospective or closeout document
- create or finalize post-Epic-9 handoff notes if needed
- link the final documents to Day 12 evidence and Day 9 residual queue
- cross-check final claims against supported and unsupported closeout language
- run documentation hygiene checks
- confirm no code/build/workflow/script/test surfaces changed before applying
  the docs-only validation rule

### Implementation-Day Checks

- Day 13 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- A separate `make format && make lint && make test` chain is not required for
  the docs-only Day 13 changes.

### Day 13 Exit State

Sprint 99 now has grounded retrospective drafts and a bounded Day 14 closeout
edit list. Day 14 can finalize the Sprint 99 retrospective, Epic 9
retrospective, and post-Epic-9 handoff without reopening sprint scope.

## Day 14 - Epic 9 Closeout And Handoff

### Goal

Finalize the Sprint 99 retrospective, Epic 9 retrospective, and post-Epic-9
handoff package from the validated Day 12 evidence package and Day 13 drafts.

### Actions

- Re-read Sprint 99 Day 14 in
  `docs/planning/EPIC_9/SPRINT_99/PLAN.md`.
- Reviewed `artifacts/day13-retrospective-drafts.md`.
- Created final Sprint 99 retrospective:
  - `docs/planning/EPIC_9/SPRINT_99/RETROSPECTIVE.md`
- Created final Epic 9 retrospective:
  - `docs/planning/EPIC_9/EPIC_9_RETROSPECTIVE.md`
- Created final post-Epic-9 handoff package:
  - `docs/planning/EPIC_9/POST_EPIC_9_HANDOFF.md`
- Created the Day 14 closeout artifact:
  - `artifacts/day14-closeout-and-handoff.md`
- Preserved Day 12 claim limits and Day 9 residual ownership.

### Final Documents

- Sprint 99 retrospective:
  - `docs/planning/EPIC_9/SPRINT_99/RETROSPECTIVE.md`
- Epic 9 retrospective:
  - `docs/planning/EPIC_9/EPIC_9_RETROSPECTIVE.md`
- Post-Epic-9 handoff:
  - `docs/planning/EPIC_9/POST_EPIC_9_HANDOFF.md`
- PR closeout summary:
  - `docs/planning/EPIC_9/SPRINT_99/artifacts/day14-closeout-and-handoff.md`

### Final Closeout Position

Epic 9 closes from a validated and documented baseline:

- Day 10 `make quality-review-full` passed.
- Makefile and CMake test counts matched at 54.
- full CTest passed 54/54.
- Day 11 package, consumer, example, benchmark, and reporting validation
  passed.
- Day 12 consolidated the evidence package.
- Day 14 finalized the Sprint, Epic, and handoff documents.

### Implementation-Day Checks

- Day 14 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- A separate `make format && make lint && make test` chain is not required for
  the docs-only Day 14 changes.
- Final documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 99 and Epic 9 closeout docs
- Required closeout files are present:
  - `docs/planning/EPIC_9/SPRINT_99/RETROSPECTIVE.md`
  - `docs/planning/EPIC_9/EPIC_9_RETROSPECTIVE.md`
  - `docs/planning/EPIC_9/POST_EPIC_9_HANDOFF.md`
  - `docs/planning/EPIC_9/SPRINT_99/artifacts/day14-closeout-and-handoff.md`

### Day 14 Exit State

Sprint 99 deliverables are complete and internally consistent. Epic 9 closes
from a documented, validated baseline, with explicit post-Epic-9 handoff items
for the next planning cycle.
