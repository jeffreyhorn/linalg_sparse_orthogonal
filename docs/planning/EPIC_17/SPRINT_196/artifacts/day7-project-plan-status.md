# Sprint 196 Day 7 Artifact: Project Plan Status Pass

**Date:** 2026-09-03
**Sprint item coverage:** 196.3, with support for 196.1, 196.2, and 196.6
**Day 7 goal:** Mark Epic 17 project-plan items complete, narrowed,
deferred, residualized, or still pending with evidence links and explicit
claim boundaries.

## Summary

Day 7 updates `docs/planning/EPIC_17/PROJECT_PLAN.md` with closeout-status
tables for Sprints 187 through 196. The pass uses the Day 2 outcome ledger and
Day 3 residual queue as the source of truth, then records item-level evidence
links directly beside the original Epic 17 work plan.

The important result is that selected Epic 17 closures are visible without
turning narrowed or residualized work into broader support claims. Sprint 196
itself is marked as current rather than final where later closeout days still
own integrated validation, the Epic retrospective, and final residual queue
publication.

## Files Updated

| File | Update |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Added closeout-status tables to every Epic 17 sprint section. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Added Day 7 status vocabulary, count snapshot, residual boundaries, and validation notes. |
| `docs/planning/EPIC_17/SPRINT_196/artifacts/day7-project-plan-status.md` | Added this project-plan status artifact. |

## Status By Sprint

| Sprint | Status outcome |
| --- | --- |
| 187 | All baseline, gap-ledger, acceptance-gate, quality-map, and handoff items marked Complete. |
| 188 | Package proof hardening landed, while license metadata and full Homebrew support remain Deferred or Residualized. |
| 189 | PowerShell validation ownership landed, with hosted CI evidence called out as pending at sprint closeout. |
| 190 | One selected Windows Cholesky workflow path landed, while manifest promotion and broad Windows report freshness remain narrowed/residualized. |
| 191 | One local-only selected QR incompatible least-squares comparison family marked Complete. |
| 192 | One selected Linux hosted performance evidence lane marked Complete, with threshold policy intentionally Narrowed. |
| 193 | One selected QR external-reference review surface marked Complete. |
| 194 | Adoption/API coherence simplification marked Complete. |
| 195 | One selected symbolic Cholesky allocation-failure owner proof marked Complete. |
| 196 | Evidence reconciliation and project-plan status marked Complete; integrated validation, retrospective, and final residual publication remain scheduled closeout work. |

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| Complete | Selected scope landed with evidence and no retained blocker for that selected scope. |
| Complete with guarded residual | Guarded proof/docs landed, but a named blocker prevents broader support or promotion. |
| Complete with hosted evidence pending at closeout | Source-controlled workflow ownership landed, with hosted PR-CI evidence still serving as the final promotion owner at that sprint closeout. |
| Complete with residual narrowed | A broader gap was reduced to one guarded path, but the broad claim remains unearned. |
| Narrowed | The final evidence intentionally covers less than the original planned wording could imply. |
| Deferred | Work was not implemented or promoted because a prerequisite decision or evidence source was missing. |
| Residualized | The item produced an explicit residual with closure evidence instead of final support/promotion. |
| Pending | Sprint 196 closeout work scheduled for a later day. |
| In progress | Partial Sprint 196 evidence exists, but final publication is incomplete. |

## Item Count Snapshot

| Status family | Count |
| --- | ---: |
| Complete | 46 |
| Complete through public and maintainer/API passes | 1 |
| Complete with guarded residual | 2 |
| Complete with hosted evidence pending at closeout | 1 |
| Complete with residual narrowed | 2 |
| Narrowed | 2 |
| Deferred | 1 |
| Residualized | 2 |
| Pending | 2 |
| In progress | 1 |

These counts include all 60 current Epic 17 project-plan rows and Sprint
196's current closeout state. They are not final Epic 17 counts until Days
8-14 complete the retrospective, integrated validation, final residual queue,
and Day 14 reconciliation.

## Claim Boundaries Preserved

- Package-manager/Homebrew support remains unclaimed until approved root
  license metadata and exact Homebrew license identifier exist.
- Windows selected Cholesky remains guarded workflow evidence until hosted
  evidence and selected manifest metadata are promoted together.
- Selected comparisons remain fixture/target scoped and do not imply broad
  external ecosystem parity.
- Selected performance remains methodology-bound, threshold-free, and
  non-portable.
- Selected reliability proof remains limited to
  `sparse_symbolic_cholesky()` output allocation.
- Epic 17 still does not claim release readiness, shared-library/dynamic ABI
  support, broad Windows parity, broad package-manager distribution, portable
  performance, or unqualified state-of-the-art sparse linear algebra status.

## 196.3 Acceptance Evidence

| Completion criterion | Evidence |
| --- | --- |
| Project-plan status is updated. | `docs/planning/EPIC_17/PROJECT_PLAN.md` now contains closeout-status tables for Sprints 187-196. |
| Evidence links are attached to completed and narrowed items. | Each status row references a Sprint closeout artifact, retrospective, or Sprint 196 evidence artifact. |
| Narrowed, deferred, residualized, and superseded work is not hidden. | Sprint 188, 190, 192, and 196 rows explicitly identify narrowed, deferred, residualized, pending, or in-progress work. |

## Validation

- `git diff --check`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_performance_docs.py`
- `make docs-check`

Day 7 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.
