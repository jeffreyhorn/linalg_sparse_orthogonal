# Day 6: Maintainer Claim Recalibration Batch Two

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Update maintainer, benchmark, corpus, API, and planning-adjacent
documentation to align with the final claim boundary.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 6 applies the maintainer/API portion of the Day 3 claim audit. The change
is intentionally narrow: align generated API policy ownership and maintainer
evidence ownership with the latest merged Epic 18 sprint closures without
promoting any broad support claim.

Day 6 does not update the Epic-level project-plan status, retrospective, or
residual queue. Those current-status edits remain assigned to Day 7, Day 11,
and Day 12 respectively.

## Changed Maintainer/API Surfaces

| File | Change | Claim rationale |
| --- | --- | --- |
| `docs/maintainer_guide.md` | The generated API section now says Sprint 179 made the original local-only decision, Sprint 186 retained it, and Sprint 204 is the current policy owner. | Preserves historical context while making the current local-only policy owner unambiguous. |
| `docs/maintainer_guide.md` | The hosted/generated API residual wording now says any future hosted HTML, retained artifact, or committed generated output path must deliberately reopen the product decision and validate the selected publication policy before docs may claim it. | Prevents historical Sprint 186 residual wording from looking like the current owner after Sprint 204. |
| `docs/maintainer_guide.md` | The evidence ownership heading now covers Epic 17 and Epic 18, selected comparison freshness names Sprint 203 artifacts, and adoption/API coherence names Sprint 204 and Sprint 205 artifacts plus `make support-docs-guard`. | Keeps maintainer evidence routing aligned with merged Epic 18 closures. |
| `docs/api_reference.md` | The generated API section now mirrors the Sprint 179/Sprint 186/Sprint 204 ownership chain. | Keeps the user-facing API reference aligned with maintainer policy. |

## Reviewed But Not Edited

| Surface | Day 6 disposition |
| --- | --- |
| `benchmarks/README.md` | No edit needed. Day 3 found selected benchmark wording claim-safe, and Sprint 202 selected hosted performance freshness remains bounded. |
| `tests/corpus/README.md` | No edit needed. Windows QR and Cholesky selected freshness boundaries already match Sprint 199 and Sprint 203 outcomes. |
| `tests/corpus/manifests/selected_report_targets.tsv` | No edit needed. Selected target metadata remains the source of truth and should not change without new evidence. |
| `tests/corpus/schemas/report_index_fields.md` | No edit needed. Report-index field contract remains claim-safe. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Not edited on Day 6. Day 7 owns project-plan status implementation. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Not edited on Day 6. Day 11 owns the final retrospective update. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Not edited on Day 6. Day 12 owns residual queue recalibration. |

## Claim Boundary After Edits

| Area | Day 6 status |
| --- | --- |
| Generated API HTML | Local-only ignored output under Sprint 204 current policy. |
| Hosted API docs | Not claimed; future work must reopen the publication decision and validate the selected policy. |
| Retained generated-doc artifacts | Not claimed. |
| Committed generated HTML | Not claimed. |
| Package-manager distribution | Not claimed; Sprint 198 remains selected developer-mode local static source formula proof only. |
| Windows selected QR freshness | Not claimed; Sprint 203 remains local proof plus re-deferral. |
| Portable performance/release/state-of-the-art | Not claimed. |

## Validation And Hygiene

| Check | Day 6 result |
| --- | --- |
| `git diff --check` | Passed after Day 6 edits. |
| `make api-docs-freshness` | Passed after `docs/api_reference.md` and generated API policy wording changed. |
| `make support-docs-guard` | Passed after support/claim routing wording in the maintainer guide changed. |
| C/header quality gate | Not required for Day 6; no `.c` or `.h` edits. |
| Generated-output status | `make api-docs-freshness` may generate ignored local API output; no generated output should be staged. |
| Guard scripts/workflows/manifests | Not edited on Day 6. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Maintainer documentation agrees with public support boundaries. | Met. `docs/maintainer_guide.md` and `docs/api_reference.md` now both name Sprint 204 as the current local-only generated API policy owner. |
| Evidence docs distinguish current status from historical sprint records. | Met for generated API policy and maintainer evidence ownership. Sprint 179/Sprint 186 remain historical context; Sprint 204 is current policy. |
| Item 206.2 has maintainer-facing implementation progress with recorded evidence. | Met. This artifact records the changed surfaces and retained non-claims. |

## Day 6 Disposition

Day 6 is complete. Day 7 should implement the project-plan status update using
the Day 4 status design and the Day 5-Day 6 claim recalibration results.
