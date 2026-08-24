# Sprint 177 Retrospective

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Duration:** 14 days (Days 1-14 landed on branch `sprint-177`)
**Status:** Complete

## Source Artifact Note

Sprint 177 was executed from the Epic 16 project-plan section for Sprint 177
and lives under `docs/planning/EPIC_15/SPRINT_177/` because that was the
requested output path. The sprint artifacts preserve this source/path mismatch
explicitly so future readers do not infer that the work came from an Epic 15
project-plan section.

## Definition Of Done Checklist

- [x] Created Sprint 177 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited and deduplicated Epic 13-15 residuals.
- [x] Classified residuals by user value, claim risk, implementation risk,
      testability, evidence need, sprint cost, and closure quality.
- [x] Inventoried repository surfaces that own public claims, evidence,
      generated reports, generated API docs, package/install behavior,
      workflows, tests, benchmarks, examples, and large review areas.
- [x] Defined and populated an evidence/status matrix for current support
      tiers, evidence locality, owner files, validation commands, artifacts,
      non-claims, and next actions.
- [x] Selected bounded Sprint 178-186 closure targets for Epic 16.
- [x] Created acceptance gates for all selected implementation and closeout
      targets.
- [x] Created a quality surface map that names validation requirements by
      changed file surface.
- [x] Froze public claim boundaries and protected non-claims.
- [x] Prepared actionable Sprint 178 and Sprint 179 handoffs.
- [x] Reconciled all Sprint 177 project-plan items against produced
      artifacts.
- [x] Ran documentation-level validation with `git diff --check`.

## What Went Well

1. **The sprint closed the planning gap before implementation began.** Sprint
   177 converted broad Epic 16 residuals into a concrete evidence matrix,
   selected-gap register, acceptance gates, and handoffs.

2. **Closure targets stayed bounded.** The selected Sprint 178-186 work favors
   complete closure of named gaps over partial progress on broad claims such
   as state-of-the-art parity, package-manager ecosystem support, shared
   library ABI, broad Windows parity, broad generated report parity, or broad
   allocation-failure safety.

3. **The evidence model is explicit.** The matrix separates hosted proof,
   local-only proof, source-controlled context, decision-only rows, deferrals,
   unsupported surfaces, and advisory documentation.

4. **Future validation is easier to select.** The quality surface map ties
   documentation, workflow, report tooling, package/install, public-header,
   C source, benchmark, and example changes to the checks expected before
   closeout.

5. **Prior review traps were carried forward.** Workflow guard placement,
   artifact upload fail-closed scope, duplicate manifest rows, missing-row
   failures, Windows CTest count drift, package wording, allocation-failure
   terminology, public error-contract ordering, generated docs staging, and
   source registration drift are now called out in gate artifacts.

6. **The first implementation handoffs are actionable.** Sprint 178 has an
   allocation-failure proof batch 2 handoff, and Sprint 179 has a generated
   API HTML status handoff with owner files, first actions, validation,
   pass criteria, stop criteria, and review traps.

## What Didn't Go Well

1. **The requested output path is confusing.** Sprint 177 is an Epic 16
   baseline sprint, but the requested path is under Epic 15. The sprint
   handled this by recording the mismatch in the plan, working notes, and
   artifacts.

2. **The claim surface remains distributed.** README, INSTALL, maintainer
   guide, API docs, benchmark docs, workflow comments, package guards,
   report-index tests, and planning artifacts all carry pieces of the support
   boundary.

3. **No implementation evidence was added.** That was appropriate for this
   baseline sprint, but all selected closure targets still require later
   implementation, validation, and claim recalibration.

4. **Several target decisions remain intentionally unresolved.** Generated API
   HTML status, package-manager provider support, Windows report freshness,
   shared-library ABI, and broad comparison/performance posture still need
   future sprint decisions or guarded deferrals.

5. **The validation burden remains high for C/header changes.** Later sprints
   must continue running `make format && make lint && make test` whenever
   implementation or public header files change.

## Final Metrics

### Validation

| Metric | Sprint 177 close state |
| --- | --- |
| documentation hygiene | passed: `git diff --check` |
| C source/header quality gate | not required: documentation-only sprint |
| hosted CI evidence | not applicable: planning-only branch |
| package/install validation | not required: no package files changed |
| report/tooling validation | not required: no report scripts changed |

### Planning Surface

| Metric | Sprint 177 close state |
| --- | ---: |
| sprint plan files | 1 |
| working notes files | 1 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |
| selected Sprint 178-186 targets | 9 |
| acceptance gates created | 9 |
| explicit non-goal categories preserved | 12 |

### Claim Governance

| Metric | Sprint 177 close state |
| --- | ---: |
| broad state-of-the-art claims added | 0 |
| broad external-library parity claims added | 0 |
| broad package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI support claims added | 0 |
| broad Windows parity claims added | 0 |
| broad generated-report parity claims added | 0 |
| broad allocation-failure claims added | 0 |

## Selected Epic 16 Closure Targets

Sprint 177 selected these bounded targets for Sprints 178-186:

| Sprint | Selected target |
| --- | --- |
| 178 | Allocation-failure proof batch 2 |
| 179 | Generated API HTML publication decision |
| 180 | Package-manager provider decision |
| 181 | Selected report target manifest |
| 182 | Windows report freshness decision |
| 183 | Additional bounded external comparison family |
| 184 | Public header coherence batch 3 |
| 185 | Large test/source review-surface reduction |
| 186 | Final validation, claim calibration, and closeout |

## Closed Planning Claim

Sprint 177 closes this planning claim:

The Epic 16 baseline now has a documented residual queue, evidence/status
matrix, selected closure target register, acceptance gates, validation surface
map, claim-boundary freeze, and first implementation handoffs.

This does not claim that any selected Epic 16 implementation target has been
completed. It only establishes the evidence contract for completing those
targets in later sprints.

## Follow-Up Risks

1. **Allocation-failure proof remains narrow.** Sprint 178 should select one
   additional subsystem and close it completely without implying broad
   allocation-failure safety.

2. **Generated API HTML status remains a product decision.** Sprint 179 should
   select hosted publication, retained artifact, committed output, or stronger
   local-only status before changing navigation or claims.

3. **Package-manager provider support remains unearned.** Sprint 180 should
   either prove one provider path or strengthen the deferral and public
   non-claim.

4. **Report target duplication remains a drift risk.** Sprint 181 should
   centralize selected oracle, comparison, performance, artifact, expected-row,
   and support-tier metadata.

5. **Windows report freshness remains unresolved.** Sprint 182 should promote
   exactly one Windows-safe path or close the lane as a guarded deferral.

6. **Large review surfaces still slow review.** Sprint 185 should reduce one
   selected test/source surface without changing behavior.

## Sprint 178 Readiness

Sprint 178 should begin from:

- `artifacts/day12-handoff-package.md`
- `artifacts/day8-gate-templates.md`
- `artifacts/day10-quality-surface-map.md`
- `artifacts/day11-claim-boundary-freeze.md`

The highest-value next action is to select one additional allocation-heavy
subsystem and prove deterministic allocation-failure cleanup, successful retry,
focused validation, and scoped public wording end to end.
