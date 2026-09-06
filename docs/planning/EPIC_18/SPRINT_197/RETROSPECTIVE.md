# Sprint 197 Retrospective

**Sprint:** 197 - Epic 18 Final Validation, Claim Calibration & Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-197`)
**Status:** Complete for the requested final-validation branch path, with an
explicit numbering caveat

## Source Artifact Note

Sprint 197 was requested under `docs/planning/EPIC_18/SPRINT_197/`, but the
cited Epic 18 project-plan section is labeled Sprint 206. This sprint preserves
the requested `SPRINT_197` path while tracking final-validation item numbers
206.1 through 206.6 for traceability.

Because Sprints 198 through 205 have no branch-local implementation artifacts,
validation records, or PR evidence on this branch, this retrospective treats
their work as pending future execution. It does not promote future sprint
intent into completed support, package, Windows, benchmark, API publication,
release, ABI, or state-of-the-art claims.

## Definition Of Done Checklist

- [x] Created Sprint 197 plan, working notes, artifact directory, daily
      artifacts, final closeout package, and retrospective.
- [x] Reconciled current Epic 18 evidence sources and classified missing Sprint
      198-205 artifacts as future evidence rather than completed proof.
- [x] Audited public claim surfaces for package-manager, Windows, benchmark,
      generated API, release, ABI, reliability, comparison, and
      state-of-the-art boundaries.
- [x] Audited maintainer/API/corpus/schema surfaces for evidence ownership,
      validation routing, generated API policy, selected report semantics, and
      retained non-claims.
- [x] Recorded public and maintainer/API no-promotion decisions because the
      available evidence does not support stronger claims.
- [x] Added an interim Epic 18 project-plan status snapshot and item-level
      status ledger.
- [x] Ran focused validation for docs/API local-only policy, Windows PowerShell
      ownership, package-manager deferral, static package/ABI deferral, and
      source-list confidence.
- [x] Recorded the full-gate decision and confirmed no `.c` or `.h` files
      changed, so `make format && make lint && make test` was not required.
- [x] Created `EPIC_18_RETROSPECTIVE.md` with outcomes, validation evidence,
      changed surfaces, earned claims, non-claims, residuals, and
      state-of-the-art assessment.
- [x] Published `EPIC_18_RESIDUAL_QUEUE.md` with prioritized residuals, closure
      targets, owner surfaces, expected evidence, validation commands, and
      claim boundaries.
- [x] Completed Day 14 final closeout review and PR summary inputs.

## What Went Well

1. **The sprint kept evidence and intent separate.** Planned Sprint 198-205
   work was not represented as completed proof.

2. **The numbering mismatch stayed visible.** The plan, working notes,
   project-plan snapshot, retrospective, and closeout artifacts all preserve
   the requested `SPRINT_197` path while pointing at the Sprint 206
   final-validation scope.

3. **Claim calibration was conservative.** Public and maintainer/API surfaces
   were audited, but no support claims were widened without implementation or
   hosted evidence.

4. **Validation matched the changed surface.** The sprint ran docs/API,
   Windows ownership, package/static deferral, source-list, and patch-hygiene
   checks, and it documented why generated report, benchmark, helper, and full
   C gates were skipped.

5. **The residual queue is actionable.** Each residual has owner surfaces,
   expected evidence, validation commands, and claim boundaries instead of a
   loose backlog label.

## What Didn't Go Well

1. **The sprint executed final validation before implementation sprints.** That
   makes most Epic 18 work necessarily pending rather than complete.

2. **The project-plan numbering mismatch adds review cost.** Future closeout
   work should either use the project plan's final-validation sprint path or
   explicitly update the plan before execution.

3. **No new implementation proof was added.** The branch improves planning,
   evidence governance, residual handoff, and validation records only.

4. **High-value product gaps remain open.** Package-manager support, Windows
   freshness, additional reliability proof, review-surface reduction,
   benchmark platform freshness, Windows QR comparison, API publication,
   release readiness, ABI support, and state-of-the-art evidence still require
   future sprint work.

## Final Metrics

### Validation

| Metric | Sprint 197 close state |
| --- | --- |
| final `git diff --check` | passed during Day 14 validation |
| final `make docs-check` | passed during Day 14 validation |
| API docs freshness | passed during Day 10 focused validation |
| Windows PowerShell guard | passed during Day 10 focused validation, with local `pwsh` unavailable recorded as an environment residual |
| package-manager deferral guard | passed during Day 10 focused validation |
| static package deferral guard | passed during Day 10 focused validation |
| source-list check | passed during Day 10 focused validation with 49 library sources |
| final `.c`/`.h` diff check | passed with no output |
| final staged-file check | passed with no staged files |
| tracked generated artifact check | passed with no tracked generated API/build/cache artifacts |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 197 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic retrospective files added | 1 |
| Epic residual queue files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 0 |
| Maintainer/API/corpus/schema documentation files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| In progress with numbering caveat | 6 |
| Pending future execution | 48 |
| Partial final-validation evidence | 4 |
| Requested final-validation evidence recorded | 2 |
| Pending final-validation work | 0 |
| Complete implementation items | 0 |
| Narrowed implementation items | 0 |
| Deferred implementation items | 0 |
| Residual queue entries published | 10 |
| Superseded items | 0 |

## Closed Claim

Sprint 197 closes this bounded claim:

The requested Sprint 197 final-validation branch now has a complete
day-by-day plan, working notes, evidence reconciliation, claim audits,
no-promotion records, interim project-plan status snapshot, focused validation
matrix, focused validation log, full-gate decision log, Epic 18 retrospective
draft, prioritized residual queue, final closeout review package, and sprint
retrospective. This is documentation, planning, validation, and evidence
governance work.

This claim does not include new solver behavior, public API/ABI changes,
package-manager support, broad platform parity, portable performance proof,
release readiness, broad external-library parity, broad allocation-failure
coverage, hosted generated API publication, or unqualified state-of-the-art
status.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-closeout-intake.md](./artifacts/day1-closeout-intake.md);
- [day2-outcome-ledger.md](./artifacts/day2-outcome-ledger.md);
- [day3-evidence-conflicts.md](./artifacts/day3-evidence-conflicts.md);
- [day4-public-claim-audit.md](./artifacts/day4-public-claim-audit.md);
- [day5-maintainer-api-claim-audit.md](./artifacts/day5-maintainer-api-claim-audit.md);
- [day6-public-recalibration.md](./artifacts/day6-public-recalibration.md);
- [day7-maintainer-api-recalibration.md](./artifacts/day7-maintainer-api-recalibration.md);
- [day8-project-plan-status.md](./artifacts/day8-project-plan-status.md);
- [day9-integrated-validation-matrix.md](./artifacts/day9-integrated-validation-matrix.md);
- [day10-focused-validation-log.md](./artifacts/day10-focused-validation-log.md);
- [day11-full-quality-gate-log.md](./artifacts/day11-full-quality-gate-log.md);
- [day12-retrospective-draft.md](./artifacts/day12-retrospective-draft.md);
- [day13-residual-queue-and-claim-decision.md](./artifacts/day13-residual-queue-and-claim-decision.md);
- [day14-final-closeout-review.md](./artifacts/day14-final-closeout-review.md);
- [../EPIC_18_RETROSPECTIVE.md](../EPIC_18_RETROSPECTIVE.md);
- [../EPIC_18_RESIDUAL_QUEUE.md](../EPIC_18_RESIDUAL_QUEUE.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Homebrew/package-manager support remains blocked | Future product/legal metadata owner | Add approved license metadata, run selected Homebrew proof to success, update guards, run install/docs validation, and promote only the earned support tier. |
| Selected Windows Cholesky freshness remains unpromoted | Future hosted Windows evidence owner | Review hosted artifacts, update selected manifest metadata if supported, rerun normalizer/workflow/PowerShell guards, and calibrate docs. |
| Additional allocation-failure owners remain unproved | Future selected reliability owner | Select one owner, record invariants, add deterministic failure/retry tests, add focused gate, update docs, and run full C validation when implementation changes. |
| Additional review surfaces remain large | Future maintainability owner | Select one high-risk cluster, extract or refactor with behavior-preserving guards, and rerun focused/full validation. |
| Additional hosted benchmark freshness remains unowned | Future benchmark/platform owner | Select one platform/row, add methodology metadata, create hosted evidence, and retain non-portable performance wording. |
| Windows QR incompatible comparison remains local-only | Future Windows comparison owner | Add MSVC/CMake proof, inspect artifacts, update exact manifest metadata if supported, and retain broad QR/Windows non-claims. |
| Generated API publication policy remains undecided | Future product/docs infrastructure owner | Decide hosted/artifact/committed/local-only policy, implement matching guards, and run docs/API freshness/link validation. |
| Adoption/support simplification remains undone | Future documentation owner | Add quick reference, centralize support truth, normalize diagnostics wording, and run claim guards. |
| Release, shared-library, and dynamic ABI readiness remain long-horizon | Future product/platform owner | Define release criteria, ABI policy, shared-library metadata, loader behavior, package selectors, and validation matrix. |
| State-of-the-art positioning remains unearned | Future methodology/research owner | Define external baselines, matrix suites, metrics, thresholds, platforms, package provenance, reliability semantics, and reviewed hosted evidence. |

## Next-Epic Readiness

Sprint 197 leaves a prioritized handoff rather than a loose backlog.

| Future need | Sprint 197 handoff |
| --- | --- |
| Package-manager support | `EPIC_18_RESIDUAL_QUEUE.md` priority 1 with exact Homebrew proof and non-claim boundary. |
| Windows report freshness | Priorities 2 and 6 separate selected Cholesky promotion from QR incompatible comparison promotion. |
| Reliability proof | Priority 3 repeats the selected-owner allocation-failure proof pattern. |
| Reviewability | Priority 4 selects one bounded review-surface reduction instead of broad cleanup. |
| Benchmark evidence | Priority 5 requires hosted evidence and methodology metadata before any platform claim. |
| API publication | Priority 7 separates product policy from local-only generated API freshness. |
| Adoption coherence | Priority 8 focuses on quick-reference and diagnostics simplification without support drift. |
| Long-horizon claims | Priorities 9 and 10 keep release/ABI/state-of-the-art work explicit and unpromoted. |

## Final Assessment

Sprint 197 is complete as a requested final-validation closeout package for the
current branch state. It is not an implementation sprint and does not complete
the Epic 18 productization work planned for Sprints 198 through 205.

The branch is ready for review as a planning and evidence-governance PR.
