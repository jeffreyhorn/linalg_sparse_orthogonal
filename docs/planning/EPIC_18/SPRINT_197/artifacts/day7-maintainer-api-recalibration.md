# Sprint 197 Day 7 Maintainer and API Documentation Recalibration

## Purpose

Day 7 applies final-validation item 206.2 to maintainer-facing, API,
generated-report, and corpus/schema documentation. It verifies that evidence
ownership and claim boundaries are clear enough for future maintainers without
promoting unsupported Epic 18 outcomes.

## Decision

No maintainer, API, corpus, schema, benchmark-adjacent, or generated-report
documentation edits were made on Day 7.

The Day 5 maintainer/API claim audit found that the owner docs already identify
the exact gates and artifacts for selected comparison freshness, Windows
PowerShell validation, generated API local freshness, benchmark freshness,
package support, reliability proof, and review-surface changes. The Day 2
ledger still classifies Sprints 198 through 205 as future evidence, so there is
no new support tier, freshness platform, package proof, API publication policy,
or reliability/review-surface closure to promote.

## Maintainer/API Surface Review

| Surface | Evidence owner already present | Day 7 action | Future edit trigger |
| --- | --- | --- | --- |
| `docs/maintainer_guide.md` selected comparison freshness guidance | `make report-index-comparison-freshness`, selected target manifest, hosted Linux/macOS lanes, bounded Windows Cholesky workflow path, target-specific Windows command, selected artifact name, and retained non-claims. | Retained as-is. | Sprint 199 or 203 promotes, re-defers, renames, or changes selected comparison metadata, workflow scope, artifact path, support tier, or claim contract. |
| `docs/maintainer_guide.md` Windows PowerShell validation ownership | `make windows-powershell-guard`, hosted `--require-pwsh` interpretation, selected snippet parsing scope, and separation from report freshness. | Retained as-is. | Workflow snippets, Windows support wording, selected artifact names, PowerShell parser behavior, or local/hosted PowerShell semantics change. |
| `docs/maintainer_guide.md` generated API guidance | `make api-docs-freshness`, `make docs-check`, Doxygen inputs, local-only staging enforcement, ignored generated HTML, and no hosted/API-publication claim. | Retained as-is. | Sprint 204 changes generated API publication, artifact retention, hosted docs, Doxygen scope, or source-controlled output policy. |
| `docs/maintainer_guide.md` benchmark freshness guidance | `make bench-canonical-report-freshness`, selected manifest row semantics, methodology fields, and threshold-free hosted interpretation. | Retained as-is. | Sprint 202 adds or changes a hosted platform/row or methodology metadata. |
| `docs/api_reference.md` | Checked-in public headers as API truth; generated HTML local-only; no package, ABI, Windows parity, portable performance, or state-of-the-art claim. | Retained as-is. | Generated API publication/local-only decision changes or public-header ownership changes. |
| `tests/corpus/README.md` | Report family and selected target manifest handoff, freshness commands, selected row interpretation, Windows Cholesky guarded path, and local-only/generated evidence caveats. | Retained as-is. | Selected report target rows, artifact paths, workflow platforms, or family semantics change. |
| `tests/corpus/schemas/report_index_fields.md` | Metadata authority for selection scope, support tier, freshness policy, claim scope, non-claims, owners, workflow artifacts, and platform promotion rules. | Retained as-is. | Manifest fields or validation semantics change. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Planning-only future sprint scope and final-validation expectations. | Retained for Day 8 status work. | Day 8 applies outcome/status annotations after the no-promotion basis is documented. |

## Evidence-Owner Routing Notes

| Change area | Maintainer/API routing to preserve |
| --- | --- |
| Package-manager and Homebrew support | `docs/maintainer_guide.md`, `INSTALL.md`, `packaging/homebrew/`, Homebrew proof command, package deferral guards, and install checks. |
| Windows selected Cholesky freshness | `docs/maintainer_guide.md`, `INSTALL.md`, `tests/corpus/README.md`, selected target manifest, Windows workflow, PowerShell guard, normalizer target filtering, and hosted Windows evidence. |
| Windows QR incompatible comparison | Selected comparison manifest, report generator, Windows-safe generation path, normalizer/workflow tests, and explicit re-deferral until MSVC/CMake evidence exists. |
| Selected benchmark freshness | `benchmarks/README.md`, selected manifest benchmark row, canonical benchmark bundle, benchmark freshness checker, hosted platform evidence, and methodology metadata. |
| Generated API publication | `docs/api_reference.md`, `docs/maintainer_guide.md`, Doxyfile, local generated HTML ignore policy, `make docs-check`, and `make api-docs-freshness`. |
| Allocation-failure reliability proof | Focused allocation-failure gate, owner-specific tests, registration guards, cleanup/retry invariants, and full C quality gate when implementation changes. |
| Review-surface reduction | Source-list checks, helper registration guards, focused owner tests, extraction invariants, and full C quality gate when source or headers change. |

## Retained Maintainer/API Non-Claims

- Maintainer guidance does not treat hosted PowerShell parsing as selected
  report freshness or artifact publication evidence.
- Windows selected Cholesky remains a guarded path until hosted evidence,
  manifest metadata, support tier, and claim contract are reviewed together.
- Windows QR incompatible comparison remains outside promoted selected Windows
  freshness until a future sprint adds MSVC/CMake generation evidence and
  reviewed metadata.
- Generated API HTML remains local-only generated output, not source-controlled,
  hosted, artifact-published, or release evidence.
- Benchmark freshness remains selected and methodology-bound; it does not
  promote raw timings, unselected rows, portable speedups, platform parity, or
  state-of-the-art performance.
- Package-manager support, shared-library support, dynamic ABI compatibility,
  runtime-loader behavior, and broad ecosystem parity remain unclaimed.
- Planning docs do not supersede evidence-owner docs; project-plan status must
  point at artifacts, checks, or explicit residual records.

## Day 7 Change Log

- Updated only Sprint 197 planning artifacts.
- Recorded a maintainer/API documentation no-promotion decision.
- Preserved existing maintainer, API, corpus, schema, benchmark, and project
  plan wording.
- Deferred project-plan outcome annotation to Day 8.

## Item 206.2 Evidence

Day 7 completes the maintainer/API portion of item 206.2 for this branch. Public
claim recalibration was recorded on Day 6; maintainer/API recalibration is now
recorded as evidence-backed no-op because current owner docs already match the
available evidence and future implementation sprint artifacts are absent.

## Validation Notes

- Reviewed the Day 5 maintainer/API claim audit before deciding not to edit
  maintainer/API docs.
- Re-scanned maintainer guide, API reference, corpus README, report-index schema
  docs, and Epic 18 project plan for selected comparison, PowerShell, generated
  API, Homebrew, support tier, freshness, Windows, publication, local-only, and
  state-of-the-art wording.
- No production code, public headers, maintainer/API docs, corpus/schema docs,
  generated reports, benchmark docs, or project-plan status rows were edited on
  Day 7.
