# Sprint 186 Day 8: Project Plan Status Update

## Purpose

Update the Epic 16 project plan with evidence-linked closeout status so future
reviewers can see which items are complete, narrowed, deferred, residualized,
or still planned in the active Sprint 186 closeout sequence.

## Source Inputs

| Source | Day 8 use |
| --- | --- |
| `docs/planning/EPIC_16/PROJECT_PLAN.md` | Target project-plan status surface. |
| `artifacts/day3-reconciled-evidence-matrix.md` | Final status matrix for Sprints 177-185. |
| `artifacts/day4-public-claim-inventory.md` | Residual and claim-calibration context. |
| `artifacts/day5-user-facing-claim-calibration.md` | User-facing claim calibration evidence for package and install surfaces. |
| `artifacts/day6-maintainer-report-claim-calibration.md` | Maintainer/report claim calibration evidence. |
| `artifacts/day7-api-header-claim-calibration.md` | Generated API and QR header-coherence calibration evidence. |

## Project Plan Changes

| Section | Change |
| --- | --- |
| Sprint 177 | Added closeout status table with all six items marked Complete and linked to baseline, matrix, gate, quality, and handoff artifacts. |
| Sprint 178 | Added closeout status table with all six items marked Complete for the selected `sparse_matmul()` allocation-failure proof. |
| Sprint 179 | Added closeout status table with item 179.2 marked Narrowed for the local-only generated API decision and the remaining items marked Complete. |
| Sprint 180 | Added closeout status table with item 180.2 marked Narrowed for the local Homebrew proof path and item 180.6 marked Residualized for missing standalone license metadata. |
| Sprint 181 | Added closeout status table with all six items marked Complete for selected report target manifest ownership. |
| Sprint 182 | Added closeout status table with items 182.2 and 182.3 marked Deferred for Windows report freshness and item 182.6 marked Residualized for unavailable local PowerShell validation. |
| Sprint 183 | Added closeout status table with all six items marked Complete for the selected Cholesky comparison family. |
| Sprint 184 | Added closeout status table with all six items marked Complete for declaration-preserving QR header coherence. |
| Sprint 185 | Added closeout status table with all six items marked Complete, including PR #205 review-fix evidence. |
| Sprint 186 | Added current closeout status table with 186.1-186.3 marked Complete and 186.4-186.6 marked Planned for Days 9-13. |

## Status Vocabulary

Sprints 177 through 185 use the Day 3 final closeout vocabulary:

- Complete
- Narrowed
- Deferred
- Residualized
- Superseded

Sprint 186 is still active on Day 8, so item rows 186.4 through 186.6 use
Planned until integrated validation, retrospective, and handoff work is
completed later in the sprint.

## Status Summary

| Status | Count | Rows |
| --- | ---: | --- |
| Complete | 51 | All complete rows from Sprints 177-185 plus Sprint 186 items 186.1-186.3. |
| Narrowed | 2 | 179.2, 180.2. |
| Deferred | 2 | 182.2, 182.3. |
| Residualized | 2 | 180.6, 182.6. |
| Superseded | 0 | No project-plan item was replaced by a later incompatible path. |
| Planned | 3 | 186.4, 186.5, 186.6. |

## Non-Complete Item Rationale

| Item | Status | Rationale | Residual or closure target |
| --- | --- | --- | --- |
| 179.2 | Narrowed | Generated API HTML was deliberately closed as a strengthened local-only path, not a hosted/publication path. | `R186-HOSTED-API`. |
| 180.2 | Narrowed | Homebrew was selected as a local formula/tap proof path, not provider support. | Provider support remains a non-claim. |
| 180.6 | Residualized | Full Homebrew proof success is blocked while standalone license metadata is absent. | `R186-PKG-LICENSE`. |
| 182.2 | Deferred | Windows report freshness closed as a formal deferral rather than a promoted freshness lane. | `R186-WIN-REPORT-FRESHNESS`. |
| 182.3 | Deferred | Deferral artifact and guard behavior were implemented instead of hosted Windows report freshness. | `R186-WIN-REPORT-FRESHNESS`. |
| 182.6 | Residualized | Local PowerShell parse validation could not run because `pwsh` is unavailable. | `R186-WIN-PWSH`. |
| 186.4 | Planned | Integrated validation is scheduled for Days 9 through 11. | Day 9 validation plan and Day 10-11 validation results. |
| 186.5 | Planned | Epic retrospective is scheduled for Day 12. | `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`. |
| 186.6 | Planned | Next-epic residual handoff is scheduled for Day 13. | Prioritized residual queue with closure targets. |

## Residual Queue Carried Forward

| Residual ID | Source | Closure target |
| --- | --- | --- |
| R186-PKG-LICENSE | Sprint 180 | Add approved standalone license metadata or decide an alternate formula license strategy before claiming full Homebrew proof success. |
| R186-WIN-PWSH | Sprint 182 | Run PowerShell parse/workflow checks in an environment with `pwsh`, or document hosted-only validation ownership. |
| R186-WIN-REPORT-FRESHNESS | Sprint 182 | Promote one Windows-safe selected freshness lane or keep the formal deferral with updated blockers. |
| R186-HOSTED-API | Sprint 179 | Revisit hosted or retained generated API HTML publication only with explicit product value and guards. |
| R186-BROAD-COMPARISON | Sprint 183 | Add future comparison evidence one bounded family at a time. |
| R186-REVIEW-SURFACE-NEXT | Sprint 185 | Select exactly one future large review surface before further extraction. |

## Validation

Day 8 changed documentation files only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

Required validation:

```sh
git diff --check
```
