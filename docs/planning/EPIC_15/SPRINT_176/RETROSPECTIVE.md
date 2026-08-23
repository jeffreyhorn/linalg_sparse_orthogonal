# Sprint 176 Retrospective

**Sprint:** 176 - Allocation-Failure Evidence, Claim Recalibration & Epic
Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-176`)
**Status:** Complete

## Source Artifact Note

Sprint 176 was executed from the active Epic 15 project-plan section for
Sprint 176 and lives under `docs/planning/EPIC_15/SPRINT_176/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 176 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Inventoried allocation-heavy solver and shared subsystem surfaces.
- [x] Selected exactly one allocation-failure proof target: iterative
      repeated-run workspace handles.
- [x] Added private deterministic allocation-failure hooks in the internal
      allocation helper layer.
- [x] Added selected CG, GMRES, and MINRES repeated-run handle
      allocation-failure cleanup tests.
- [x] Fixed invalid GMRES prepare behavior so bad restart arguments do not
      publish private handle state.
- [x] Added a maintained focused gate,
      `make iterative-allocation-failure-gate`.
- [x] Added the `allocation_failure` CTest label to the existing
      `test_iterative` registration.
- [x] Documented selected iterative handle cleanup invariants in public and
      maintainer docs.
- [x] Recalibrated public claim wording so allocation-failure proof remains
      selected and local.
- [x] Ran integrated allocation, package, report, workflow, benchmark, and
      full C validation.
- [x] Created the final Epic 15 retrospective and residual queue.
- [x] Preserved broad allocation-failure, state-of-the-art, external parity,
      package-manager, shared-library, dynamic ABI, runtime-loader, broad
      platform, hosted generated API HTML, and release non-claims.

## What Went Well

1. **The sprint closed one allocation-failure gap completely.** The selected
   iterative repeated-run handle family now has deterministic fail-injection
   tests, cleanup invariants, documentation, and a maintained gate.

2. **The proof stayed narrow.** The implementation did not convert selected
   CG/GMRES/MINRES handle cleanup evidence into broad allocation-failure or
   state-of-the-art reliability claims.

3. **A real cleanup defect was found and fixed.** Invalid GMRES restart
   preparation no longer allocates or publishes private handle state before
   rejecting bad arguments.

4. **Validation matched the changed surface.** Source/header edits ran
   `make format && make lint && make test`; later documentation closeout work
   used focused guards and `git diff --check`.

5. **The Epic 15 closeout is evidence-bound.** The final Epic retrospective
   connects Sprints 167-176 to earned claims, retained non-claims, validation
   evidence, and residual queue entries.

6. **Generated and hosted evidence boundaries remained explicit.** The sprint
   preserved selected hosted report/comparison wording without implying broad
   generated report or hosted API publication.

## What Didn't Go Well

1. **The source prompt path was stale.** The request pointed to Epic 12 even
   though the active Sprint 176 plan was under Epic 15.

2. **Claim governance remains broad and distributed.** README, maintainer
   docs, report-index tests, package guards, workflow guards, and planning
   artifacts all carry pieces of the support-tier story.

3. **The allocation-failure proof remains intentionally narrow.** Direct
   solvers, eigensolvers, matrix construction, package/install flows,
   generated-report tooling, and many allocation-heavy paths remain unproved.

4. **Hosted evidence cannot be completed locally.** Day 12 could prove local
   workflow and guard behavior, but final hosted PR CI remains the activation
   evidence for hosted lanes.

5. **Repeated full gates are expensive.** The complete C gate is necessary
   after source/header changes, but it remains a significant closeout cost.

## Final Metrics

### Validation

| Metric | Sprint 176 close state |
| --- | --- |
| focused iterative build/run | passed: `make build/test_iterative && build/test_iterative` |
| focused allocation-failure gate | passed: `make iterative-allocation-failure-gate` |
| focused CMake allocation label proof | passed: `ctest -L allocation_failure` |
| package-manager deferral guard | passed |
| static package/shared ABI deferral guard | passed |
| report-index normalization test | passed |
| selected comparison workflow guard | passed |
| benchmark canonical freshness test | passed |
| full C quality gate | passed: `make format && make lint && make test` |
| final docs hygiene | passed: `git diff --check` |

### Changed Surface

| Metric | Sprint 176 close state |
| --- | ---: |
| C source files changed | 3 |
| internal header files changed | 1 |
| public header files changed | 1 |
| Make targets added | 1 |
| CMake test labels added | 1 |
| focused allocation-failure tests added | 5 |
| public/maintainer docs changed | 2 |
| sprint artifacts | 14 |
| sprint retrospective files | 1 |
| epic retrospective files | 1 |

### Claim Governance

| Metric | Sprint 176 close state |
| --- | ---: |
| selected allocation-failure claims added | 1 |
| broad allocation-failure claims added | 0 |
| state-of-the-art claims added | 0 |
| portable performance superiority claims added | 0 |
| external-library ecosystem parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI support claims added | 0 |
| runtime-loader support claims added | 0 |
| broad platform parity claims added | 0 |
| release evidence claims added | 0 |

## Closed Claim

Sprint 176 closes this Epic 15 allocation-failure evidence claim:

The iterative repeated-run handle APIs have deterministic allocation-failure
cleanup evidence for selected CG, GMRES, and MINRES prepare/growth paths.
The proof is maintained by `make iterative-allocation-failure-gate`, the
existing `test_iterative` executable, and the `allocation_failure` CTest
label.

This does not claim broad allocation-failure cleanup coverage across direct
solvers, eigensolvers, matrix construction, package/install flows,
generated-report tooling, one-shot iterative calls, or unrelated allocation
paths.

## Epic 15 Closeout

Sprint 176 also closes Epic 15 by creating
`docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`. The Epic retrospective
records Sprint 167-176 outcomes, major evidence/productization outcomes,
validation evidence, earned claims, retained non-claims, final residuals,
state-of-the-art assessment, key deliverables, and completion status.

## Follow-Up Risks

1. **Broader allocation-failure coverage remains residual.** Apply the Sprint
   176 pattern to one additional allocation-heavy subsystem at a time.

2. **Hosted generated API HTML remains local-only.** A future epic should
   either select a hosted publication path or keep local-only support guarded.

3. **Package-manager provider support remains deferred.** Select one provider
   before adding any package-manager availability claim.

4. **Shared-library and dynamic ABI support remain deferred.** Reopen only
   with ABI policy, symbol/export, loader, install, and downstream validation.

5. **Windows report freshness remains unsupported.** Add a Windows-safe
   report lane or retain a stronger deferral.

6. **Workflow target inventories remain repetitive.** Factor selected target
   inventories before broadening hosted comparison coverage.

## Sprint 177 Readiness

The branch is ready for final review after commit/PR closeout. The highest
value next work is selecting one residual and closing it end to end with the
same evidence-bound standard used in Sprint 176.
