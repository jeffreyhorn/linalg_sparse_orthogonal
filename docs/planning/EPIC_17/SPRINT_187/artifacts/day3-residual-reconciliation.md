# Sprint 187 Day 3: Epic 16 Residual Reconciliation

## Purpose

Reconcile the Epic 16 residual queue with the Day 2 Epic 17 review ledger,
deduplicate inherited residuals, preserve useful residual IDs for traceability,
and identify which residuals should remain long-horizon instead of becoming
Sprint 187-196 closure targets.

## Reconciliation Inputs

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` | Source of six inherited residual IDs, owner surfaces, closure targets, expected evidence, validation commands, and deferral horizons. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day2-review-intake-matrix.md` | Source of the initial 16-row Epic 17 review ledger. |
| `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` | Source of review findings and state-of-the-art readiness assessment. |
| `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` | Source of the step-by-step closure strategy. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Source of planned Sprint 188-196 closure slots. |

## Deduplicated Residual Mapping

| Epic 16 residual | Theme | Epic 17 mapping | Candidate sprint | Day 3 disposition | Rationale |
| --- | --- | --- | --- | --- | --- |
| `R186-PKG-LICENSE` | Homebrew local proof blocker | `E17-GAP-001 / R186-PKG-LICENSE` | Sprint 188 | Selected closure candidate | This is a complete, near-term package proof closure with clear owner files, proof script, package guards, and docs. |
| `R186-WIN-PWSH` | PowerShell validation environment | `E17-GAP-002 / R186-WIN-PWSH` | Sprint 189 | Selected closure candidate | PowerShell validation ownership is a prerequisite for any credible Windows report freshness promotion. |
| `R186-WIN-REPORT-FRESHNESS` | Windows selected report freshness | `E17-GAP-003 / R186-WIN-REPORT-FRESHNESS` | Sprint 190 | Selected closure candidate | This is a bounded platform/report decision that can either promote one selected lane or renew the formal deferral. |
| `R186-HOSTED-API` | Generated API publication | Folded into `E17-GAP-008` documentation/coherence and retained generated-API long-horizon notes | None by default | Long-horizon retained decision | Epic 16 intentionally selected local-only generated API HTML. Hosted publication is lower priority than package, Windows, comparison, performance, maintainability, adoption, and reliability closures unless later ranking explicitly reselects it. |
| `R186-BROAD-COMPARISON` | Future bounded comparison breadth | `E17-GAP-004 / R186-BROAD-COMPARISON` | Sprint 191 | Selected closure candidate | The broad parity concern remains a non-claim, but one bounded family is feasible and directly supports numerical credibility. |
| `R186-REVIEW-SURFACE-NEXT` | Future review-surface reduction | `E17-GAP-006 / R186-REVIEW-SURFACE-NEXT`; broad context in `E17-GAP-011` | Sprint 193 | Selected closure candidate | A single selected cluster can be fully closed with no-behavior-change invariants, helper/source ownership, focused tests, and full C validation. |

## Duplicate Handling

No Epic 16 residual was dropped. Each residual now has exactly one primary
Epic 17 disposition:

- selected closure candidate;
- long-horizon retained decision; or
- retained non-claim context.

The Day 2 review ledger remains broader than the Epic 16 residual queue. Some
Day 2 rows do not map to inherited residual IDs because they came from the new
Codex review:

| Day 2 gap | Relationship to Epic 16 residuals |
| --- | --- |
| `E17-GAP-005` performance evidence | New review-derived gap; adjacent to state-of-the-art readiness but not an Epic 16 residual. |
| `E17-GAP-007` adoption usability | New review-derived gap; informs Sprint 194. |
| `E17-GAP-008` documentation/coherence | New review-derived gap; absorbs hosted API publication as long-horizon documentation-product context. |
| `E17-GAP-009` reliability proof | New review-derived gap; extends prior selected allocation-failure proof pattern. |
| `E17-GAP-010` broad numerical robustness | Long-horizon review-derived state-of-the-art gap. |
| `E17-GAP-012` storage model efficiency | Long-horizon review-derived architecture gap. |
| `E17-GAP-013` broad coverage campaign | Long-horizon review-derived test-evidence gap, except for one selected reliability owner. |
| `E17-GAP-014` shared-library/dynamic ABI | Retained non-claim; not selected by default. |
| `E17-GAP-015` broad Windows parity | Retained non-claim; selected Windows work remains PowerShell/report-freshness only. |
| `E17-GAP-016` state-of-the-art positioning | Retained non-claim until final closeout calibration. |

## Reconciled Closure Candidate List

| Candidate | Source IDs | Planned sprint | Complete-closure target |
| --- | --- | --- | --- |
| Homebrew proof completion | `E17-GAP-001`, `R186-PKG-LICENSE` | Sprint 188 | Resolve license metadata or alternate formula strategy and prove the full local Homebrew workflow. |
| PowerShell validation ownership | `E17-GAP-002`, `R186-WIN-PWSH` | Sprint 189 | Add an owned PowerShell validation command and hosted/local ownership without claiming report freshness yet. |
| Windows report freshness decision | `E17-GAP-003`, `R186-WIN-REPORT-FRESHNESS` | Sprint 190 | Promote one selected Windows-safe report freshness lane or renew formal deferral with stronger guards. |
| Bounded external comparison family | `E17-GAP-004`, `R186-BROAD-COMPARISON` | Sprint 191 | Add one fixture-bound comparison family with metrics, tolerances, manifests, reports, docs, and freshness proof. |
| Methodology-bound performance lane | `E17-GAP-005` | Sprint 192 | Promote one selected performance lane with methodology metadata, hosted freshness, and claim-safe docs. |
| Selected review-surface reduction | `E17-GAP-006`, `R186-REVIEW-SURFACE-NEXT`, narrowed slice of `E17-GAP-011` | Sprint 193 | Reduce exactly one high-risk source/test cluster with behavior-preserving validation. |
| Adoption and API coherence | `E17-GAP-007`, `E17-GAP-008` | Sprint 194 | Simplify user-facing workflows and support truth while preserving necessary non-claims. |
| Selected reliability proof | `E17-GAP-009`, narrowed slice of `E17-GAP-013` | Sprint 195 | Add deterministic failure-path proof for one selected owner. |

## Long-Horizon Residual List

| Residual or gap | Long-horizon reason | Retained boundary |
| --- | --- | --- |
| `R186-HOSTED-API` | Hosted generated API publication is a product-documentation decision and was intentionally narrowed to local-only generated HTML in Epic 16. | Generated API HTML remains local-only unless a future sprint explicitly selects hosted/retained/committed output and adds publication/freshness guards. |
| `E17-GAP-010` | Broad numerical robustness and representative corpus proof cannot be fully closed across all solver families inside Epic 17. | Add future evidence one bounded family at a time. |
| `E17-GAP-011` broad form | Multi-module source/test refactoring would spread effort too thin and increase behavior risk. | Sprint 193 may close one selected cluster only. |
| `E17-GAP-012` | Replacing the orthogonal linked-list storage identity is a major architecture program. | Keep storage-model replacement out of Epic 17 unless narrowed to one backend proof. |
| `E17-GAP-013` broad form | Broad fuzz/property/differential coverage across all solvers is larger than one epic. | Sprint 195 may close one selected reliability owner only. |
| `E17-GAP-014` | Shared-library and dynamic ABI support need ABI policy, symbol visibility, loader behavior, and package proofs. | Retain static-first package contract. |
| `E17-GAP-015` broad form | Broad Windows parity exceeds selected PowerShell/report freshness work. | Retain CMake-first Windows support boundaries. |
| `E17-GAP-016` | State-of-the-art positioning needs broad external, performance, numerical, platform, packaging, and release evidence. | Final closeout should calibrate claims, not assert broad state-of-the-art status. |

## Day 4 Inputs

Day 4 should inventory owner files and current evidence for the reconciled
candidate list:

1. package proof owner files and validation scripts;
2. Windows CI, PowerShell, report, manifest, and support-tier owners;
3. comparison runner, fixture, manifest, and report owners;
4. benchmark/performance report owners;
5. maintainability candidate source/test files and guard owners;
6. adoption/API docs and examples;
7. reliability/failure-path proof owners.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
