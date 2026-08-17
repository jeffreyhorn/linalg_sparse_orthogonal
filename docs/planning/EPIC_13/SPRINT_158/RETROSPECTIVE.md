# Sprint 158 Retrospective

**Sprint:** 158 - Generated API HTML Publication Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-158`)
**Status:** Complete

## Source Artifact Note

Sprint 158 was planned and executed from the Epic 14 Sprint 158 package under
`docs/planning/EPIC_14/SPRINT_158/`. This retrospective is written to the
requested path, `docs/planning/EPIC_13/SPRINT_158/RETROSPECTIVE.md`, and links
back to the Epic 14 Sprint 158 source artifacts for evidence.

## Definition Of Done Checklist

- [x] Created Sprint 158 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Ran and recorded the current Doxygen API documentation baseline.
- [x] Captured Doxygen warnings, generated page inventory, generated output
      tracking state, local path scan, and generated `sparse_version.h`
      behavior.
- [x] Chose the local-only generated API HTML publication path and rejected
      committed or hosted Doxygen HTML for this sprint.
- [x] Added a recurring generated API page-coverage guard:
      `scripts/check_api_docs_coverage.py`, `make api-docs-coverage`, and
      `make docs-check`.
- [x] Fixed the selected Doxygen warning categories with comment-only public
      header updates.
- [x] Aligned `docs/api_reference.md`, `docs/maintainer_guide.md`, and README
      command guidance with the selected local-only publication path.
- [x] Preserved source-header-first API ownership and generated
      `sparse_version.h` install-artifact ownership.
- [x] Reconciled generated API documentation claims and residuals.
- [x] Published a Sprint 159 hosted-report handoff that keeps hosted report
      promotion separate from Doxygen API HTML publication.
- [x] Ran generated-doc validation, docs hygiene, claim scans, and the full
      public-header quality gate required by the Sprint 158 header edits.

## What Went Well

1. **The generated API docs decision closed cleanly.** Sprint 158 selected a
   local-only generated HTML path with a maintained guard instead of leaving
   `docs/api/html/` in a stale or ambiguous state.

2. **The guard is deterministic and reviewable.** `make docs-check` now
   regenerates Doxygen output and verifies reference/source pages for the
   checked-in public headers without committing generated HTML.

3. **Warning closure stayed narrow.** The Day 9 fixes removed all selected
   Doxygen warnings with comment-only public-header changes and no declaration,
   macro, struct layout, or behavior changes.

4. **Documentation policy now matches repository state.** API reference and
   maintainer docs both say generated HTML is local-only, ignored, and current
   only for a checkout where `make docs-check` has just passed.

5. **Unsupported claims stayed blocked.** The sprint did not convert ignored
   local generated HTML into hosted, source-controlled, release, package, ABI,
   platform, performance, parity, or state-of-the-art evidence.

6. **The next-sprint boundary is concrete.** Sprint 159 can focus on hosted
   generated report promotion without re-deciding Doxygen API HTML policy.

## What Didn't Go Well

1. **The retrospective path is inconsistent with the execution package.** The
   Sprint 158 plan and artifacts live under `EPIC_14/SPRINT_158`, while this
   requested retrospective path is under `EPIC_13/SPRINT_158`. The retrospective
   records that mismatch explicitly to avoid hiding the evidence path.

2. **Generated HTML remains unhosted.** This was the correct Sprint 158 product
   decision, but users still do not have hosted Doxygen API HTML.

3. **The generated version header remains outside Doxygen pages.** The sprint
   documented the policy, but there is still no generated Doxygen page for the
   installed `sparse_version.h` output under the current checked-in-header
   input set.

4. **The branch touched public headers.** Even though the changes were
   comment-only, the sprint correctly paid the full validation cost with
   `make format && make lint && make test`.

5. **The artifact trail is large.** Fourteen daily artifacts provide useful
   traceability, but the shortest reviewer path is the Day 14 closeout plus
   Day 12 validation and Day 13 reconciliation.

## Final Metrics

### Validation

| Metric | Sprint 158 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | yes, comment-only |
| full C quality gate required | yes |
| full C quality gate | passed: `make format && make lint && make test` |
| generated-doc guard | passed: `make docs-check` |
| checked-in public headers covered | 18 |
| generated reference pages covered | 18 |
| generated source pages covered | 18 |
| remaining Doxygen warnings | 0 in final `docs-check` runs |
| generated `sparse_version.h` policy | installed-header policy row; not an expected Doxygen page |
| final `git diff --check` | passed |
| trailing-whitespace scans | passed |
| claim scan | passed; matches were non-claims, historical option analysis, or unrelated bounded-evidence language |
| generated API HTML tracking | ignored: `!! docs/api/` |

### Artifact Package

| Metric | Sprint 158 close state |
| --- | ---: |
| daily artifacts under `EPIC_14/SPRINT_158/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| new scripts | 1 |
| Make targets added | 2 plus aggregate `docs-check` |
| source files changed | 0 |
| public headers changed | 3 |
| generated HTML files committed | 0 |

## Closed Claim

Sprint 158 closes this generated API HTML publication claim:

The project now has an explicit local-only generated API HTML policy, a
deterministic `make docs-check` freshness/page-coverage guard for the
configured checked-in public-header input set, zero selected Doxygen warnings,
aligned public and maintainer documentation, and validation evidence for the
touched public-header and documentation surfaces without committing or hosting
generated Doxygen HTML.

This claim is supported by:

- [PLAN.md](../../EPIC_14/SPRINT_158/PLAN.md);
- [WORKING_NOTES.md](../../EPIC_14/SPRINT_158/WORKING_NOTES.md);
- [day1-api-docs-intake.md](../../EPIC_14/SPRINT_158/artifacts/day1-api-docs-intake.md);
- [day2-doxygen-baseline.md](../../EPIC_14/SPRINT_158/artifacts/day2-doxygen-baseline.md);
- [day3-public-header-coverage-map.md](../../EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md);
- [day4-warning-triage-policy.md](../../EPIC_14/SPRINT_158/artifacts/day4-warning-triage-policy.md);
- [day5-publication-options.md](../../EPIC_14/SPRINT_158/artifacts/day5-publication-options.md);
- [day6-publication-decision.md](../../EPIC_14/SPRINT_158/artifacts/day6-publication-decision.md);
- [day7-page-coverage-check-design.md](../../EPIC_14/SPRINT_158/artifacts/day7-page-coverage-check-design.md);
- [day8-coverage-implementation.md](../../EPIC_14/SPRINT_158/artifacts/day8-coverage-implementation.md);
- [day9-warning-fix-batch.md](../../EPIC_14/SPRINT_158/artifacts/day9-warning-fix-batch.md);
- [day10-policy-alignment.md](../../EPIC_14/SPRINT_158/artifacts/day10-policy-alignment.md);
- [day11-publication-finalization.md](../../EPIC_14/SPRINT_158/artifacts/day11-publication-finalization.md);
- [day12-validation-evidence.md](../../EPIC_14/SPRINT_158/artifacts/day12-validation-evidence.md);
- [day13-claim-reconciliation.md](../../EPIC_14/SPRINT_158/artifacts/day13-claim-reconciliation.md);
- [day14-closeout-handoff.md](../../EPIC_14/SPRINT_158/artifacts/day14-closeout-handoff.md).

## Sprint 159 Readiness

Sprint 159 should begin from these settled Sprint 158 boundaries:

| Starting item | Required posture |
| --- | --- |
| Generated API HTML | Local-only, ignored, validated by `make docs-check`; not committed or hosted. |
| API declaration authority | `docs/api_reference.md` plus checked-in public headers. |
| Generated version header | Owned by install artifacts, `VERSION`, and install-validation tests; not an expected Doxygen page under current input. |
| Hosted report promotion | Keep separate from Doxygen API HTML publication. |
| Claim language | Preserve non-claims for package, ABI, platform, performance, external-library parity, and state-of-the-art coverage unless independently validated. |

Recommended Sprint 159 prerequisites:

1. Select claim-bearing generated oracle/comparison report families for hosted
   promotion.
2. Keep non-selected report families explicitly local-only or advisory.
3. Define artifact retention, branch freshness, and failure semantics before
   adding hosted jobs.
4. Add public and maintainer wording naming exactly which hosted rows are
   reviewed evidence.
5. Avoid implying hosted Doxygen API HTML publication unless a later sprint
   explicitly funds that lane.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 158 close:

- hosted Doxygen API HTML publication;
- committed `docs/api/html/`;
- generated Doxygen page for installed `sparse_version.h`;
- broad generated API reference completeness beyond configured checked-in
  public-header input pages;
- hosted generated report promotion for selected oracle/comparison rows;
- package, ABI, platform, performance, external-library parity, and
  state-of-the-art claims not proven by Sprint 158 evidence.

Still consciously constrained rather than silently solved:

- no generated API HTML as release evidence;
- no source-controlled generated HTML freshness claim;
- no hosted generated API documentation claim;
- no broad API completeness claim beyond `make docs-check` coverage;
- no dynamic ABI, shared-library, package-manager, broad platform, performance,
  parity, or state-of-the-art claim from generated API documentation.

## Key Deliverables

- [PLAN.md](../../EPIC_14/SPRINT_158/PLAN.md)
- [WORKING_NOTES.md](../../EPIC_14/SPRINT_158/WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-api-docs-intake.md](../../EPIC_14/SPRINT_158/artifacts/day1-api-docs-intake.md)
- [day2-doxygen-baseline.md](../../EPIC_14/SPRINT_158/artifacts/day2-doxygen-baseline.md)
- [day3-public-header-coverage-map.md](../../EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md)
- [day4-warning-triage-policy.md](../../EPIC_14/SPRINT_158/artifacts/day4-warning-triage-policy.md)
- [day5-publication-options.md](../../EPIC_14/SPRINT_158/artifacts/day5-publication-options.md)
- [day6-publication-decision.md](../../EPIC_14/SPRINT_158/artifacts/day6-publication-decision.md)
- [day7-page-coverage-check-design.md](../../EPIC_14/SPRINT_158/artifacts/day7-page-coverage-check-design.md)
- [day8-coverage-implementation.md](../../EPIC_14/SPRINT_158/artifacts/day8-coverage-implementation.md)
- [day9-warning-fix-batch.md](../../EPIC_14/SPRINT_158/artifacts/day9-warning-fix-batch.md)
- [day10-policy-alignment.md](../../EPIC_14/SPRINT_158/artifacts/day10-policy-alignment.md)
- [day11-publication-finalization.md](../../EPIC_14/SPRINT_158/artifacts/day11-publication-finalization.md)
- [day12-validation-evidence.md](../../EPIC_14/SPRINT_158/artifacts/day12-validation-evidence.md)
- [day13-claim-reconciliation.md](../../EPIC_14/SPRINT_158/artifacts/day13-claim-reconciliation.md)
- [day14-closeout-handoff.md](../../EPIC_14/SPRINT_158/artifacts/day14-closeout-handoff.md)
