# Sprint 187 Day 1: Baseline Intake

## Purpose

Establish the Sprint 187 baseline, source artifact map, owner-surface map,
risks, open questions, and Day 2 gap-ledger schema.

## Project-Plan Boundaries

| Item | Boundary captured for Sprint 187 |
| --- | --- |
| 187.1 | Convert Codex review findings into a prioritized ledger with owner files and claim risks. |
| 187.2 | Reconcile Epic 16 residuals with the new review findings and deduplicate repeated gaps. |
| 187.3 | Select complete gaps for Epic 17 and preserve broad unsupported claims as non-goals. |
| 187.4 | Define acceptance gates for validation commands, hosted/local ownership, artifacts, and support-tier wording. |
| 187.5 | Map required quality checks by changed surface. |
| 187.6 | Produce handoff records for later package and Windows work. |

## Source Artifacts

| Source | Role in Sprint 187 |
| --- | --- |
| `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` | Primary review findings and state-of-the-art assessment. |
| `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` | Step-by-step closure strategy and candidate validation commands. |
| `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md` | Earned claims, retained non-claims, validation evidence, and final Epic 16 status. |
| `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` | Inherited residuals for package, Windows, hosted API, comparison, and review-surface work. |
| `README.md`, `INSTALL.md`, `docs/api_reference.md`, `docs/maintainer_guide.md`, `benchmarks/README.md` | Current public and maintainer-facing claim surfaces. |
| `Makefile`, `CMakeLists.txt`, `.github/workflows/*.yml`, package scripts, report scripts, manifests, tests, benchmarks, examples, `include/`, and `src/` | Owner surfaces for later acceptance gates and quality maps. |

## Initial Closure Families

| Family | Initial source | Sprint 187 planning responsibility |
| --- | --- | --- |
| Package proof | `R186-PKG-LICENSE`; Codex review package findings | Define exact Homebrew/license proof gates for Sprint 188. |
| PowerShell validation | `R186-WIN-PWSH` | Define validation ownership and hosted/local expectations for Sprint 189. |
| Windows report freshness | `R186-WIN-REPORT-FRESHNESS` | Define promotion or renewed-deferral gates for Sprint 190. |
| External comparison | `R186-BROAD-COMPARISON`; Codex review state-of-the-art gaps | Define one bounded comparison family gate for Sprint 191. |
| Performance evidence | Codex review performance gaps | Define methodology-bound hosted performance gates for Sprint 192. |
| Review-surface reduction | `R186-REVIEW-SURFACE-NEXT`; Codex review maintainability hotspots | Define candidate ranking and no-behavior-change gates for Sprint 193. |
| Adoption/API simplification | Codex review usability/documentation/coherence gaps | Define support matrix, diagnostics, and docs simplification gates for Sprint 194. |
| Reliability proof | Codex review coverage/reliability gaps | Define owner-selection and deterministic proof gates for Sprint 195. |

## Day 2 Ledger Schema

The Day 2 ledger should use this schema:

| Field | Required meaning |
| --- | --- |
| Gap ID | Stable identifier; preserve Epic 16 residual IDs where applicable. |
| Source | Review section, todo phase, prior residual, source/doc/CI evidence, or current non-claim. |
| Area | Efficiency, maintainability, usability, documentation, coherence, test coverage, packaging, platform, performance, comparison, reliability, or state-of-the-art readiness. |
| Finding | Short description of the gap. |
| Owner surfaces | Files, directories, workflows, scripts, tests, reports, or docs that own closure. |
| Current evidence | Existing tests, guards, docs, CI, artifacts, or explicit non-claims. |
| Claim risk | Unsupported claim that could be implied by careless wording or incomplete proof. |
| User value | Why closing the gap matters to users or maintainers. |
| Closure candidate | Candidate, not selected, or long-horizon. |
| Candidate sprint | Tentative Sprint 188-195 target if selected. |
| Required validation | Commands or hosted checks expected for closure. |
| Non-goals | Explicit breadth that must remain out of scope. |

## Risks Captured

- Duplicating Epic 16 residuals instead of preserving traceability.
- Selecting too many partial closures.
- Overclaiming package support from a local Homebrew proof.
- Treating Windows validation ownership as broad Windows parity.
- Treating selected comparison or performance evidence as state-of-the-art
  parity.
- Allowing maintainability extraction to change behavior without an explicit
  focused proof.

## Completion Notes

Day 1 completed the working-notes scaffold, source artifact inventory,
owner-surface map, risk register, open-question register, and Day 2 ledger
schema. The sprint is ready to convert the Codex review into structured
ledger rows on Day 2.

