# Sprint 196 Day 8 Artifact: Retrospective Outline and Metrics

**Date:** 2026-09-03
**Sprint item coverage:** 196.5, with support for 196.1, 196.2, 196.3, and
196.6
**Day 8 goal:** Draft the Epic 17 retrospective structure, outcome metrics,
closed-claim sections, residual placeholders, and missing-evidence checklist
before writing final retrospective prose.

## Summary

Day 8 creates the initial
`docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` outline. The outline follows
the Epic 15 and Epic 16 retrospective style: objective, sprint outcomes, major
outcomes, project-plan status, validation evidence, changed surface, earned
claims, non-claims, residual queue, state-of-the-art assessment, retrospective
themes, and key deliverables.

The file is intentionally marked as a draft outline because Sprint 196 has not
yet completed integrated validation, final residual queue publication, final
state-of-the-art assessment, or Day 14 closeout reconciliation.

## Files Updated

| File | Update |
| --- | --- |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Added the draft Epic 17 retrospective outline with sourced initial metrics and TODO markers for Day 9-14 finalization. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Added Day 8 structure, sourced metrics, missing-evidence checklist, and validation results. |
| `docs/planning/EPIC_17/SPRINT_196/artifacts/day8-retrospective-outline-and-metrics.md` | Added this Day 8 planning artifact. |

## Retrospective Format Sources

| Source | Pattern reused |
| --- | --- |
| `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md` | Evidence-bound earned claims, non-claims, residual queue, state-of-the-art assessment, retrospective themes, and key deliverables. |
| `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md` | Project-plan status table, validation-evidence boundaries, major outcome table, and final closeout residual style. |
| Sprint 187-195 retrospectives | Sprint-specific outcome wording, validation anchors, changed-surface metrics, residuals, and non-claim boundaries. |
| Sprint 196 Day 2 and Day 3 artifacts | Consolidated outcome ledger and residual triage source of truth. |
| Sprint 196 Day 7 artifact | Current project-plan status snapshot and status vocabulary. |

## Initial Metrics

| Metric | Current sourced value | Source |
| --- | ---: | --- |
| Epic 17 sprints planned | 10 | `PROJECT_PLAN.md`; Sprint 187-196 directories. |
| Sprint retrospectives complete before Sprint 196 | 9 | Sprint 187-195 `RETROSPECTIVE.md` files. |
| Sprint 196 retrospective status | 0 final files | Sprint 196 Day 8 outline only; final retrospective pending. |
| Current Complete item rows | 46 | Sprint 196 Day 7 project-plan status pass, corrected on Day 8. |
| Current Complete through public and maintainer/API passes rows | 1 | Sprint 196 Day 7 project-plan status pass, corrected on Day 8. |
| Current Complete with guarded residual rows | 2 | Sprint 196 Day 7 project-plan status pass. |
| Current Complete with hosted evidence pending rows | 1 | Sprint 196 Day 7 project-plan status pass. |
| Current Complete with residual narrowed rows | 2 | Sprint 196 Day 7 project-plan status pass. |
| Current Narrowed rows | 2 | Sprint 196 Day 7 project-plan status pass, corrected on Day 8. |
| Current Deferred rows | 1 | Sprint 196 Day 7 project-plan status pass, corrected on Day 8. |
| Current Residualized rows | 2 | Sprint 196 Day 7 project-plan status pass, corrected on Day 8. |
| Current Pending rows | 2 | Sprint 196 Day 7 project-plan status pass. |
| Current In progress rows | 1 | Sprint 196 Day 7 project-plan status pass. |
| Library sources in source-list checks | 49 | Sprint 193, 194, and 195 retrospectives. |
| Checked-in public headers covered by generated API docs | 18 | Sprint 194 retrospective and Sprint 196 docs checks. |
| Generated API reference pages | 18 | Sprint 194 retrospective and Sprint 196 docs checks. |
| Generated API source pages | 18 | Sprint 194 retrospective and Sprint 196 docs checks. |
| Selected oracle freshness rows | 54 | Sprint 194 retrospective. |
| Selected comparison freshness rows | 46 | Sprint 191 and Sprint 194 retrospectives. |
| Hosted selected performance lanes added or hardened | 1 | Sprint 192 retrospective. |
| Selected review-surface clusters reduced | 1 | Sprint 193 retrospective. |
| Selected allocation-failure owners proved in Epic 17 | 1 | Sprint 195 retrospective. |

## Missing Evidence Checklist

| Evidence gap | Needed before final retrospective |
| --- | --- |
| Final Sprint 196 integrated validation | Run Day 11 integrated validation after retrospective and residual queue surfaces are present. |
| Final residual queue artifact | Publish Day 10 residual queue with priorities, closure targets, validation gates, and non-claim boundaries. |
| Final project-plan status counts | Update counts after 196.4, 196.5, and 196.6 are complete. |
| Final changed-surface rollup | Count Sprint 196 closeout surfaces after all remaining Day 9-14 edits are complete. |
| Final state-of-the-art assessment | Reconcile earned claims against final residual queue and validation evidence. |
| Final key deliverable links | Add Sprint 196 retrospective and residual queue links once those files exist. |

## Claim Boundaries For Final Text

- Package-manager/Homebrew support remains unclaimed until approved root
  license metadata and exact Homebrew license identifier exist.
- Windows selected Cholesky remains guarded workflow evidence until hosted
  evidence and selected metadata are promoted together.
- Selected comparisons remain fixture/target scoped.
- Selected performance remains methodology-bound, threshold-free, and
  non-portable.
- Selected reliability proof remains limited to
  `sparse_symbolic_cholesky()` output allocation.
- Shared-library support, dynamic ABI compatibility, release readiness,
  broad Windows parity, broad package-manager distribution, portable
  performance, and unqualified state-of-the-art sparse linear algebra status
  remain non-claims.

## Validation

- `git diff --check`
- `make docs-check`

Day 8 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.
