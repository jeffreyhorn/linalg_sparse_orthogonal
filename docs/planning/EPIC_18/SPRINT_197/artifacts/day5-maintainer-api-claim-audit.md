# Day 5 Maintainer, API, and Planning Claim Audit

## Purpose

Day 5 audits maintainer-facing proof interpretation, generated API policy,
corpus/report schema ownership, and planning status surfaces. It prepares
later claim recalibration and project-plan status edits without changing those
claim surfaces yet.

## Maintainer/API Audit

| Surface | Current owner role | Finding |
| --- | --- | --- |
| `docs/maintainer_guide.md` selected comparison freshness | Selected comparison generation, manifest authority, hosted lane interpretation, Windows Cholesky boundary. | Accurate and claim-safe; future Windows promotion must update metadata, support tier, artifact scope, and docs together. |
| `docs/maintainer_guide.md` Windows PowerShell validation | PowerShell snippet ownership and hosted `--require-pwsh` interpretation. | Correctly separates PowerShell validation from CMake, CTest, report generation, uploads, package proofs, and freshness. |
| `docs/maintainer_guide.md` generated API reference | Generated API local-only policy and freshness interpretation. | Accurate but tied to earlier decisions; Sprint 204 must update if publication policy changes. |
| `docs/api_reference.md` | Source-controlled API reference and generated HTML routing. | Correctly identifies public headers as source of truth and generated HTML as ignored local-only output. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index field and selected target policy owner. | Correctly prevents fake deferral rows and `workflow_platforms` widening without reviewed metadata. |
| `tests/corpus/README.md` | Corpus/report evidence interpretation. | Must be kept synchronized with manifest and schema when selected targets are promoted. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint status and planning source. | Do not mark final status until implementation evidence exists. |

## Planning Status Edit Inventory

| Area | Later status edit | Evidence prerequisite |
| --- | --- | --- |
| Sprint 197 | Complete/narrow/supersede baseline or closeout scaffold. | Gap ledger, residual selection, acceptance gates, claim map, validation. |
| Sprint 198 | Complete or residualize Homebrew proof. | License metadata, formula proof, package guards, install checks, docs. |
| Sprint 199 | Promote or re-defer selected Windows Cholesky freshness. | Hosted run, artifact review, manifest metadata, PowerShell/normalizer/freshness tests. |
| Sprint 200 | Complete one additional allocation-failure proof. | Owner invariant, deterministic tests, focused gate, registration guard. |
| Sprint 201 | Complete one selected review-surface reduction. | Ranking, extraction, behavior preservation, guard, focused tests. |
| Sprint 202 | Complete one additional hosted benchmark freshness lane. | Platform/row metadata, hosted bundle, freshness checks, benchmark docs. |
| Sprint 203 | Promote or re-defer Windows QR incompatible comparison. | MSVC/CMake proof, selected artifacts, manifest update, QR/freshness tests. |
| Sprint 204 | Record generated API publication decision. | Decision, publication or local-only guard, docs/API freshness, link checks if added. |
| Sprint 205 | Complete support/adoption consolidation. | Public docs, quick reference, diagnostics vocabulary, claim guards, validation. |
| Sprint 206 | Close Epic 18. | Final validation log, retrospective, residual queue, claim decision table. |

## Generated API Boundary

- Source-controlled API truth is `docs/api_reference.md` plus checked-in public
  headers under `include/`.
- `docs/api/html/` is generated locally, ignored, and not hosted,
  artifact-published, committed, or release evidence.
- `make docs-check` and `make api-docs-freshness` own local generated API
  freshness and staging policy.
- Generated installed header `sparse_version.h` remains install-policy
  evidence, not an expected Doxygen page.
- Do not imply dynamic ABI compatibility, shared-library support,
  package-manager distribution, broad Windows parity, external-library parity,
  or completeness beyond configured Doxygen inputs.

## Validation-Owner Update Plan

| Change area | Required validation owners |
| --- | --- |
| Package/Homebrew support | Homebrew proof, package/static guards, install tests, docs checks. |
| Windows selected freshness | PowerShell guard, workflow tests, normalizer tests, selected freshness command, hosted Windows evidence. |
| Benchmark platform freshness | Benchmark freshness tests, canonical report freshness, hosted platform evidence, docs checks. |
| Generated API publication | Docs/API freshness, local-only or publication guard, link checks if introduced. |
| Reliability proof | Focused allocation-failure gate, registration guard, full C gate when C/header files change. |
| Review-surface reduction | Source-list check, helper guard, focused tests, CMake parity if registration changes, full C gate when needed. |

## Acceptance Evidence

- Maintainer/API owner surfaces are classified.
- Generated API publication remains local-only pending Sprint 204 decision.
- Project-plan status edits are listed with evidence prerequisites.
- Validation owners are mapped to the claim areas they protect.

