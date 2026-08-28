# Sprint 186 Retrospective

**Sprint:** 186 - Epic 16 Final Validation, Claim Calibration & Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-186`)
**Status:** Complete

## Source Artifact Note

Sprint 186 was executed from the Epic 16 project-plan section for Sprint 186
and lives under `docs/planning/EPIC_16/SPRINT_186/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 186 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reconciled Sprint 177-185 evidence and status records against final
      sprint artifacts, validation records, decisions, and residuals.
- [x] Classified prior Epic 16 items as Complete, Narrowed, Deferred,
      Residualized, or Superseded.
- [x] Calibrated README, INSTALL, maintainer guide, report docs, package docs,
      generated API docs, and QR-facing docs against earned evidence.
- [x] Updated the Epic 16 project plan with evidence-linked closeout status.
- [x] Ran focused integrated validation for docs, package/provider status,
      manifests, workflow guards, generated API checks, selected report
      freshness, and external comparison tests.
- [x] Ran C-adjacent guards and the full repository quality gate.
- [x] Created `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`.
- [x] Created `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md`.
- [x] Preserved explicit non-claims for broad state-of-the-art, external
      parity, portable performance, package-manager support, shared-library
      support, dynamic ABI, broad Windows parity, and hosted generated API
      HTML.

## What Went Well

1. **The evidence matrix made closeout mechanical.** Day 3 resolved Sprints
   177-185 into 48 Complete, 2 Narrowed, 2 Deferred, 2 Residualized, and 0
   Superseded rows, which gave later claim and project-plan updates a stable
   source of truth.

2. **Claim calibration stayed scoped.** Days 5-7 touched public and maintainer
   docs without adding unsupported package, API, Windows, comparison,
   performance, release, or state-of-the-art claims.

3. **Project-plan status became reviewable.** Day 8 added evidence-linked
   status rows without changing the original sprint scope or estimates.

4. **Validation was split usefully.** Day 10 ran focused docs/package/report
   checks first; Day 11 then ran C-adjacent guards and the full repository
   quality gate.

5. **Residuals are consumable.** Day 13 published a prioritized residual queue
   with owner surfaces, closure targets, expected evidence, validation
   commands, and deferral horizons.

## What Didn't Go Well

1. **The closeout evidence is spread across many surfaces.** README, INSTALL,
   maintainer docs, API docs, package docs, report schemas, manifests, guards,
   sprint artifacts, and retrospectives all carry parts of the final claim
   posture.

2. **Environment gaps remain.** Local `pwsh` is unavailable, so PowerShell
   validation remains residualized instead of proven locally.

3. **Homebrew proof success is still blocked outside code.** The proof script
   exists, but full proof success remains unavailable until standalone license
   metadata or an alternate formula license strategy is selected.

4. **The full quality gate remains long-running.** `make lint` and `make test`
   passed, but they remain the expensive part of final validation.

5. **Day 14 still needed wording cleanup.** The Epic retrospective had a
   Day-12-era completion note that needed final closeout wording after the
   residual queue was published.

## Final Metrics

### Validation

| Metric | Sprint 186 close state |
| --- | --- |
| focused integrated validation | passed on Day 10 |
| API docs validation | passed: `make api-docs-validate` and `make api-docs-freshness` |
| API docs coverage | 18 checked-in public headers, 18 generated reference pages, 18 generated source pages |
| QR header/docs guard | passed: `make qr-header-docs-guard` |
| package guards | passed: static package and package-manager deferral checks |
| corpus/workflow/report tests | passed: schema, selected target manifest, selected workflow, normalizer, and external comparison tests |
| selected oracle freshness | passed: 54 normalized rows |
| selected comparison freshness | passed: 39 normalized rows |
| matmul allocation gate | passed: 18 tests, 185 assertions, 0 failures |
| LDLT CSC helper guard | passed |
| source-list check | passed: 49 library sources |
| full repository quality gate | passed: `make format && make lint && make test` |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 186 close state |
| --- | ---: |
| Epic retrospective files added | 1 |
| Epic residual queue files added | 1 |
| Sprint planning directories added | 1 |
| Sprint daily artifacts | 14 |
| Sprint retrospective files | 1 |
| public/maintainer docs changed | 7 |
| project-plan files changed | 1 |
| C source files changed | 0 |
| public header files changed | 0 |
| generated API files staged | 0 |
| generated report files staged | 0 |

### Claim Governance

| Metric | Sprint 186 close state |
| --- | ---: |
| unqualified state-of-the-art claims added | 0 |
| broad external parity claims added | 0 |
| portable performance claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI claims added | 0 |
| broad Windows parity claims added | 0 |
| hosted generated API claims added | 0 |
| release readiness claims added | 0 |

## Closed Claim

Sprint 186 closes this Epic 16 final closeout claim:

Epic 16 evidence has been reconciled, public claims have been calibrated,
project-plan status has been updated, focused and full validation have passed,
the Epic 16 retrospective has been drafted, and the next-epic residual queue
has been published.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-final-closeout-intake.md](./artifacts/day1-final-closeout-intake.md);
- [day2-evidence-matrix-baseline.md](./artifacts/day2-evidence-matrix-baseline.md);
- [day3-reconciled-evidence-matrix.md](./artifacts/day3-reconciled-evidence-matrix.md);
- [day4-public-claim-inventory.md](./artifacts/day4-public-claim-inventory.md);
- [day5-user-facing-claim-calibration.md](./artifacts/day5-user-facing-claim-calibration.md);
- [day6-maintainer-report-claim-calibration.md](./artifacts/day6-maintainer-report-claim-calibration.md);
- [day7-api-header-claim-calibration.md](./artifacts/day7-api-header-claim-calibration.md);
- [day8-project-plan-status-update.md](./artifacts/day8-project-plan-status-update.md);
- [day9-integrated-validation-plan.md](./artifacts/day9-integrated-validation-plan.md);
- [day10-focused-integrated-validation.md](./artifacts/day10-focused-integrated-validation.md);
- [day11-full-repository-quality-gate.md](./artifacts/day11-full-repository-quality-gate.md);
- [day12-retrospective-draft.md](./artifacts/day12-retrospective-draft.md);
- [day13-residual-queue-handoff.md](./artifacts/day13-residual-queue-handoff.md);
- [day14-closeout-review-pr-handoff.md](./artifacts/day14-closeout-review-pr-handoff.md);
- [../EPIC_16_RETROSPECTIVE.md](../EPIC_16_RETROSPECTIVE.md);
- [../EPIC_16_RESIDUAL_QUEUE.md](../EPIC_16_RESIDUAL_QUEUE.md).

No new product capability, public API contract, solver behavior, generated API
publication path, package-manager support path, shared-library support,
dynamic ABI guarantee, Windows report freshness lane, broad external parity,
portable performance, release readiness, or state-of-the-art claim was added.

## Next-Epic Readiness

Future planning should start from
[../EPIC_16_RESIDUAL_QUEUE.md](../EPIC_16_RESIDUAL_QUEUE.md).

| Future need | Sprint 186 handoff |
| --- | --- |
| Homebrew proof completion | Resolve standalone license metadata or formula license strategy before claiming full proof success. |
| Windows validation | Provide `pwsh` locally or assign hosted PowerShell validation ownership before promoting report freshness. |
| Windows report freshness | Select one Windows-safe freshness lane with exact workflow, artifact, manifest, and expected-row evidence. |
| Generated API publication | Reopen only with an explicit product decision for hosted, retained, or committed generated output. |
| External comparison breadth | Add one bounded family at a time with fixture, metric, report, manifest, docs, and freshness evidence. |
| Review-surface reduction | Select exactly one next cluster and repeat the behavior-preserving extraction pattern. |
