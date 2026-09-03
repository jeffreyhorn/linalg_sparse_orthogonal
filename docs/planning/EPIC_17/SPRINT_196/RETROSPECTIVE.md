# Sprint 196 Retrospective

**Sprint:** 196 - Epic 17 Final Validation, Claim Calibration & Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-196`)
**Status:** Complete; Epic 17 evidence reconciled, public and maintainer
claims calibrated, project-plan status updated, residual queue published,
Epic retrospective completed, and closeout validation packaged

## Source Artifact Note

Sprint 196 was executed from the Epic 17 project-plan section for Sprint 196
and lives under `docs/planning/EPIC_17/SPRINT_196/` with its plan, working
notes, daily artifacts, closeout package, and retrospective in one package.
The sprint also publishes the Epic-level closeout artifacts
`docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` and
`docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md`.

## Definition Of Done Checklist

- [x] Created Sprint 196 plan, working notes, artifact directory, daily
      artifacts, closeout package, and retrospective.
- [x] Reconciled Sprint 187 through Sprint 195 outcomes, validation records,
      decisions, residuals, and claim boundaries.
- [x] Consolidated residuals into a prioritized Epic 17 residual queue with
      closure targets, evidence requirements, validation commands, and
      long-horizon deferrals.
- [x] Audited README, INSTALL, maintainer, benchmark, API, corpus, planning,
      retrospective, and residual documentation for overbroad support,
      platform, package, performance, reliability, release, ABI, external
      parity, and state-of-the-art claims.
- [x] Recalibrated public and maintainer documentation so active support truth
      remains discoverable without widening evidence.
- [x] Added evidence-linked closeout status tables for Epic 17 project-plan
      items across Sprints 187 through 196.
- [x] Published `EPIC_17_RETROSPECTIVE.md` with outcomes, validation evidence,
      changed surfaces, earned claims, non-claims, residuals, lessons, and
      state-of-the-art assessment.
- [x] Completed focused integrated validation across package/install, Windows,
      selected comparison, selected performance, review-surface, reliability,
      docs/API, corpus, and report freshness evidence owners.
- [x] Completed the full-quality decision for the documentation/planning-only
      Sprint 196 changed surface.
- [x] Confirmed no `.c` or `.h` files changed, so `make test` was not required
      by the user quality-check rule.
- [x] Preserved explicit non-claims for package-manager support, broad Windows
      parity, broad external-library parity, portable performance, release
      readiness, shared-library/dynamic ABI support, hosted generated API
      publication, broad allocation-failure coverage, and unqualified
      state-of-the-art status.

## What Went Well

1. **The sprint converted a large epic into reviewable evidence.** Sprint 196
   pulled together nine prior sprint outcomes, decisions, validation records,
   and residuals into one project-plan status pass, one Epic retrospective,
   and one residual queue.

2. **Claim calibration stayed grounded in evidence.** Public and maintainer
   docs were updated from the evidence ledger and claim audit instead of
   widening support language around adjacent proof material.

3. **Residuals stayed visible.** The sprint did not hide package, Windows,
   performance, API publication, ABI, reliability, or state-of-the-art gaps
   behind broad "complete" wording. The residual queue gives future planning
   exact owner surfaces and closure evidence.

4. **Validation was matched to changed surfaces.** Day 11 ran focused gates
   across the Epic 17 evidence owners, while Day 12 ran `make format` and
   `make lint` and documented why `make test` was not required without C or
   header changes.

5. **The retrospective became an evidence index.** The Epic 17 retrospective
   now links sprint retrospectives, closeout artifacts, validation evidence,
   residuals, and non-claims in one place.

6. **Generated and environment residuals were not over-read.** Ignored Doxygen
   output, ignored build artifacts, local unavailable PowerShell, and hosted
   Windows promotion needs stayed separated from pass evidence.

## What Didn't Go Well

1. **The final claim surface is still broad.** README, INSTALL, maintainer,
   benchmark, API, corpus, project-plan, retrospective, and residual docs all
   carry parts of the support story, so future changes still need targeted
   claim checks.

2. **Some key residuals are product decisions.** Package-manager support,
   shared-library/dynamic ABI support, hosted generated API publication,
   release readiness, and broad Windows parity cannot be closed by local code
   edits alone.

3. **Hosted evidence remains selected-lane scoped.** Sprint 196 records the
   current state clearly, but it does not replace hosted review for selected
   Windows freshness promotion, Windows/macOS benchmark freshness, or hosted
   generated API publication.

4. **Performance evidence remains intentionally narrow.** The Epic 17
   performance lane is useful methodology and freshness evidence, but still
   does not prove portable timing thresholds or superiority.

5. **The closeout depended on many historical artifacts.** The reconciliation
   is useful, but future epics should keep the final status ledger current
   throughout the epic to reduce end-of-epic review load.

## Final Metrics

### Validation

| Metric | Sprint 196 close state |
| --- | --- |
| package-manager deferral guard | passed during Day 5 and Day 11 validation |
| static package deferral guard | passed during Day 5 and Day 11 validation |
| source install checks | passed during Day 11 validation |
| CMake install checks | passed during Day 11 validation |
| Windows PowerShell guard | passed during Day 5, Day 6, and Day 11 validation |
| selected report target manifest tests | passed during Day 5 and Day 11 validation |
| report normalizer tests | passed during Day 5, Day 6, and Day 11 validation |
| selected comparison freshness | passed during Day 11 validation with 46 rows |
| selected oracle freshness | passed during Day 11 validation with 54 rows |
| selected performance docs/freshness tests | passed during Day 11 validation |
| QR and LDLT review-surface guards | passed during Day 11 validation |
| symbolic allocation-failure registration and focused gate | passed during Day 11 validation |
| corpus schema validation | passed during Day 11 validation |
| API docs freshness | passed during Day 6 and Day 11 validation |
| final `make format` | passed with no `.c` or `.h` diff |
| final `make lint` | passed strict warning builds, clang-tidy, and cppcheck |
| final `make docs-check` | passed after Day 14 closeout documentation updates |
| final `git diff --check` | passed |
| final `.c`/`.h` diff check | passed with no output |
| final `make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 196 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint closeout artifacts added | 1 |
| Sprint retrospective files added | 1 |
| Epic retrospective files added | 1 |
| Epic residual queue files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Corpus documentation files changed | 1 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Complete | 50 |
| Complete with guarded residual | 2 |
| Complete with hosted evidence pending at closeout | 1 |
| Complete with residual narrowed | 2 |
| Narrowed | 2 |
| Deferred | 1 |
| Residualized | 2 |

The count covers all 60 Epic 17 project-plan item rows.

## Closed Claim

Sprint 196 closes this bounded closeout claim:

Epic 17 now has a reconciled evidence ledger, calibrated public and maintainer
claim surfaces, evidence-linked project-plan status tables, a completed Epic
retrospective, a prioritized residual queue, focused integrated validation, a
full-quality decision for the changed documentation/planning surface, and a
review-ready closeout package. The closeout is documentation and evidence
governance work; it does not add new solver behavior, public API/ABI,
package-manager support, broad platform parity, portable performance proof,
release readiness, broad external-library parity, broad allocation-failure
coverage, hosted generated API publication, or unqualified state-of-the-art
status.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-closeout-intake.md](./artifacts/day1-closeout-intake.md);
- [day2-outcome-ledger.md](./artifacts/day2-outcome-ledger.md);
- [day3-residual-triage.md](./artifacts/day3-residual-triage.md);
- [day4-claim-surface-audit.md](./artifacts/day4-claim-surface-audit.md);
- [day5-public-claim-recalibration.md](./artifacts/day5-public-claim-recalibration.md);
- [day6-maintainer-api-recalibration.md](./artifacts/day6-maintainer-api-recalibration.md);
- [day7-project-plan-status.md](./artifacts/day7-project-plan-status.md);
- [day8-retrospective-outline-and-metrics.md](./artifacts/day8-retrospective-outline-and-metrics.md);
- [day9-epic-retrospective-draft.md](./artifacts/day9-epic-retrospective-draft.md);
- [day10-prioritized-residual-queue.md](./artifacts/day10-prioritized-residual-queue.md);
- [day11-integrated-focused-validation.md](./artifacts/day11-integrated-focused-validation.md);
- [day12-full-quality-decision.md](./artifacts/day12-full-quality-decision.md);
- [day13-final-claim-retrospective-review.md](./artifacts/day13-final-claim-retrospective-review.md);
- [day14-closeout-review-package.md](./artifacts/day14-closeout-review-package.md);
- [../EPIC_17_RETROSPECTIVE.md](../EPIC_17_RETROSPECTIVE.md);
- [../EPIC_17_RESIDUAL_QUEUE.md](../EPIC_17_RESIDUAL_QUEUE.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Package-manager/Homebrew support remains blocked | Future product/legal metadata owner | Add approved root license metadata and exact Homebrew formula license value, run proof to exit `0`, rerun package/static/install guards, and calibrate docs. |
| Selected Cholesky Windows freshness promotion remains pending | Future hosted Windows evidence owner | Review hosted Windows artifacts, promote exact manifest metadata if supported, and rerun selected freshness, normalizer, manifest, and PowerShell guards. |
| Additional allocation-failure owners remain unproved | Future selected reliability owner | Select one owner, record invariants, add deterministic failure/retry tests, focused gate, docs, and full C validation if code changes. |
| Additional QR review-surface clusters remain large | Future review-surface owner | Select one cluster, extract helpers with behavior-preserving guard coverage, and rerun focused/full validation. |
| Windows/macOS selected benchmark freshness remains unowned | Future hosted platform performance owner | Add hosted platform selected freshness lane with exact artifact scope and methodology metadata without broadening performance claims. |
| Windows QR incompatible freshness remains local-only | Future Windows comparison owner | Add MSVC/CMake proof, inspect hosted artifacts, promote exact metadata only if evidence supports it, and retain broad QR parity non-claims. |
| Long-horizon product/platform/research gaps remain | Future epic/product owner | Shared-library/dynamic ABI support, broad Windows parity, optional package baselines, broad external-library parity, hosted timing thresholds, portable performance, release benchmarks, OS OOM/concurrency semantics, hosted generated API publication, and state-of-the-art positioning all need explicit future scope and evidence. |

## Next-Epic Readiness

Sprint 196 leaves Epic 17 closed with a prioritized handoff rather than a
loose backlog.

| Future need | Sprint 196 handoff |
| --- | --- |
| Next selected implementation closure | Start from [EPIC_17_RESIDUAL_QUEUE.md](../EPIC_17_RESIDUAL_QUEUE.md), especially priorities E17-RQ-001, E17-RQ-005, E17-RQ-022, E17-RQ-016, E17-RQ-013, and E17-RQ-006. |
| Claim review | Use `day4-claim-surface-audit.md`, `day13-final-claim-retrospective-review.md`, README, INSTALL, and maintainer guide as the current claim-boundary map. |
| Validation selection | Use `day11-integrated-focused-validation.md` and `day12-full-quality-decision.md` to choose focused gates for package, Windows, reports, performance, review-surface, reliability, docs/API, and C/header changes. |
| State-of-the-art positioning | Keep the Epic 17 retrospective assessment: the project has stronger selected evidence but no unqualified state-of-the-art sparse linear algebra claim. |

## Validation Retrospective

Sprint 196 did not change `.c` or `.h` files, so the user-requested full C
test gate was not required. The sprint still ran focused evidence-owner gates,
`make format`, `make lint`, and repeated `make docs-check` after closeout
documentation updates.

The final validation commands were:

```sh
git diff --check
git diff --name-only -- '*.c' '*.h'
make docs-check
```

Earlier Day 11 and Day 12 validation also passed the focused package/install,
Windows, selected report, selected performance, review-surface, reliability,
docs/API, format, and lint gates recorded in the sprint artifacts.
