# Sprint 196 Day 9 Artifact: Epic Retrospective Draft

**Date:** 2026-09-03
**Sprint item coverage:** 196.5, with support for 196.1, 196.2, 196.3, and
196.6
**Day 9 goal:** Write the first complete Epic 17 retrospective draft from
reconciled evidence and calibrated claims.

## Summary

Day 9 converts the Day 8 retrospective outline into a complete draft at
`docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md`. The draft now covers all
Epic 17 sprints, summarizes the major outcomes, records the current
project-plan status snapshot, describes validation evidence and changed
surfaces, states earned claims and retained non-claims, previews the residual
queue, and includes an initial state-of-the-art assessment.

The retrospective remains marked as draft complete rather than final because
Sprint 196 Days 10-14 still own final residual queue publication, integrated
validation, full validation decision, final key-deliverable links, and
project-plan closeout reconciliation.

## Files Updated

| File | Update |
| --- | --- |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Replaced the Day 8 scaffold with a complete Day 9 retrospective draft. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Added Day 9 evidence sources, open questions, and validation notes. |
| `docs/planning/EPIC_17/SPRINT_196/artifacts/day9-epic-retrospective-draft.md` | Added this Day 9 artifact. |

## Retrospective Sections Drafted

| Section | Day 9 status |
| --- | --- |
| Epic objective | Drafted from Epic 17 project plan and Sprint 196 outcome ledger. |
| Sprint outcomes | Drafted for Sprints 187-196. |
| Major outcomes | Drafted across evidence, package, Windows, comparison, performance, review-surface, adoption/API, reliability, and claim governance. |
| Project-plan status | Drafted from Day 7/Day 8 current status counts. |
| Validation evidence | Drafted from Sprint 188-195 retrospectives and Sprint 196 focused validations. |
| Changed surface | Drafted from sourced retrospective metrics, with final rollup deferred to Day 14. |
| Earned claims | Drafted with qualifiers from Day 2, Day 5, Day 6, and Day 7 evidence. |
| Non-claims | Drafted with package, Windows, performance, ABI, release, generated API, and reliability boundaries. |
| Residual queue | Preview drafted from Day 3 triage; final publication remains Day 10. |
| State-of-the-art assessment | Drafted as a bounded negative assessment with evidence requirements for any future claim. |
| What went well / could be better | Drafted from Sprint 187-195 retrospectives and Sprint 196 notes. |
| Open questions | Drafted and assigned to Days 10-14. |
| Key deliverables | Draft list added, with final link completion deferred to Day 13. |

## Claim Boundaries Preserved

- Package-manager/Homebrew support remains blocked by approved license
  metadata and exact Homebrew license identifier.
- Windows selected Cholesky remains guarded workflow evidence pending hosted
  evidence review and manifest metadata promotion.
- Selected comparison evidence remains local-only and fixture/target scoped.
- Selected performance evidence remains Linux selected, methodology-bound,
  threshold-free, and non-portable.
- Selected reliability proof remains limited to
  `sparse_symbolic_cholesky()` output allocation.
- Shared-library support, dynamic ABI compatibility, runtime-loader behavior,
  release readiness, broad Windows parity, broad package-manager
  distribution, hosted generated API publication, portable performance, and
  unqualified state-of-the-art sparse linear algebra status remain non-claims.

## Open Questions Before Final Closeout

| Question | Owner day |
| --- | --- |
| What exact final residual queue file and link structure should be published? | Day 10 |
| Which focused gates should form the Sprint 196 integrated validation bundle after the residual queue is added? | Day 11 |
| Is any full C quality gate required by Sprint 196 final edits, or are all remaining changes documentation-only? | Day 12 |
| What final project-plan status counts should replace the current draft counts once 196.4-196.6 are complete? | Day 14 |
| Should the final retrospective keep any draft caveat language after final validation passes? | Day 14 |

## Validation

- `git diff --check`
- `make docs-check`

Day 9 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.
