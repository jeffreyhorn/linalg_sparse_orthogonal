# Day 3: Claim Surface Audit

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Audit public and maintainer documentation for claims that must be
updated, narrowed, or retained as non-claims.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 3 audits claim-bearing surfaces before claim recalibration edits. It uses
the Day 2 outcome ledger as the source of truth for selected Sprint 197-205
outcomes. No public docs, maintainer docs, source code, workflows, manifests,
schemas, or guards are changed on Day 3.

## Audited Surfaces

| Surface | Claim role | Day 3 finding |
| --- | --- | --- |
| `README.md` | Public entry point for API, benchmark, report, Windows, and generated API boundaries. | Mostly current. It keeps generated API local-only, selected benchmark threshold-free, and Windows selected freshness unpromoted. |
| `INSTALL.md` | Public support/readiness authority. | Current support matrix reflects Sprint 198-205 outcomes: local Homebrew proof material, guarded/re-deferred Windows Cholesky, deferred Windows QR, Linux/macOS selected performance freshness, selected allocation-failure owners, local generated API, package-manager non-claim, ABI deferral, and state-of-the-art non-claim. |
| `docs/api_reference.md` | Source-controlled API route and generated API local-only policy. | Current boundary is claim-safe: generated HTML is local-only ignored output, not hosted, retained, source-controlled, package, ABI, broad Windows, release, or state-of-the-art evidence. |
| `docs/cookbook.md` | Sprint 205 compact adoption quick-reference surface. | Claim-safe routing table; rows keep package, Windows, QR, SVD, generated API, and portable performance limits visible. |
| `docs/solver_selection.md` | Detailed solver-family and evidence-boundary guide. | Claim-safe and detailed; selected evidence is fixture-local and does not imply broad parity, package/ABI, performance, release, or state-of-the-art status. |
| `benchmarks/README.md` | Benchmark/report interpretation owner. | Claim-safe; benchmark rows are local/selected evidence and do not become portable performance proof. |
| `tests/corpus/README.md` | Corpus/report-index interpretation owner. | Claim-safe; selected target rows preserve non-claims for broad report freshness, Windows, package/ABI, performance, release, and state-of-the-art status. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected target metadata authority. | Current rows carry bounded claim scopes and non-claims. `SRT-BENCH-REFACTOR-CSC-NOS4` includes Linux/macOS hosted selected performance metadata and package-manager non-claims. QR incompatible remains Linux/macOS only. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index field contract and claim-boundary explanation. | Retains selected-report and non-claim vocabulary; no Day 3 edit required. |
| `docs/maintainer_guide.md` | Maintainer claim-boundary, guard, API, benchmark, corpus, and support interpretation. | Mostly current, including Sprint 204 local-only API policy and Sprint 205 support-truth routing. Historical Sprint 179/186 generated API references should remain historical or be clarified during Day 6/Day 13 if they conflict with Sprint 204 current policy wording. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Epic plan and accumulated current-status ledger. | Needs later update so explicit Sprint 206 artifacts become the current closeout path rather than relying only on historical `SPRINT_197` final-validation evidence. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Epic-level current retrospective. | Stale. It still marks Sprint 205 pending and retains older non-claim wording for work now closed in selected scope by Sprints 200-205. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Epic residual handoff. | Partly stale. Package and Windows Cholesky residuals should distinguish selected closure or re-deferral evidence from broader residual claims. |

## Claim Classification

| Claim area | Classification | Evidence source | Required Day 5-Day 12 handling |
| --- | --- | --- | --- |
| Local source build and first solve | Earned public claim | README, Makefile/test suite, examples, `INSTALL.md` support matrix | Retain; no stronger claim needed. |
| Unix Make static install and `pkg-config` consumer | Earned bounded public claim | `INSTALL.md`, `tests/test_install.sh`, `sparse.pc.in` | Retain static-first wording and Unix scope. |
| Installed CMake consumer | Earned bounded public claim | `INSTALL.md`, `tests/test_cmake_install.sh`, `cmake/SparseConfig.cmake.in` | Retain exact-version static package boundary. |
| Windows MSVC CMake install/downstream | Earned bounded validation claim | `INSTALL.md`, `.github/workflows/windows-ci.yml`, PowerShell validator | Retain CMake route only; no Makefile/pkg-config/broad Windows parity. |
| Developer-mode local Homebrew static source proof | Earned selected proof, not broad support | Sprint 198 artifacts and local proof records | Update residual/current-status docs so selected proof is closed while broad package-manager distribution remains unclaimed. |
| Package-manager distribution | Unsupported broad claim | `INSTALL.md` support matrix; package-manager deferral guard | Retain non-claim for Homebrew/core, bottles, Linuxbrew, tap, vcpkg, Conan, pkgsrc, and distro/system packages. |
| Windows selected Cholesky freshness | Closed re-deferral, not promoted claim | Sprint 199 retrospective/artifacts; `INSTALL.md`; corpus docs | Update residual/current-status docs to say guarded workflow evidence exists but selected freshness remains re-deferred. |
| Windows QR incompatible freshness | Closed re-deferral, not promoted claim | Sprint 203 retrospective/artifacts; selected target manifest; corpus docs | Retain no hosted Windows/MSVC QR promotion. |
| Selected allocation-failure owner proof | Earned selected proof | Sprint 200 artifacts; `INSTALL.md` allocation-failure row | Update Epic retrospective non-claims so selected `sparse_symbolic_lu()` proof is not described as absent. |
| Review-surface reduction | Earned selected maintainability claim | Sprint 201 artifacts and SVD helper guard evidence | Update Epic retrospective non-claims so selected SVD helper reduction is not described as absent. |
| Linux/macOS selected benchmark freshness | Earned bounded hosted evidence | Sprint 202 artifacts, selected target manifest, README/benchmark docs | Retain selected threshold-free wording; update aggregate non-claims away from "not done" phrasing. |
| Generated API publication decision | Earned local-only policy decision | Sprint 204 artifacts; `docs/api_reference.md`; API docs guards | Retain local-only policy. Do not promote hosted publication or retained artifacts. |
| Support matrix and quick reference | Earned documentation consolidation | Sprint 205 artifacts; `INSTALL.md`; `docs/cookbook.md`; maintainer guide | Update Epic retrospective status so Sprint 205 is closed. |
| Shared-library/dynamic ABI support | Unsupported broad claim | `INSTALL.md`, static package deferral guard | Retain deferred status. |
| Release readiness | Unsupported broad claim | Epic planning docs, install/API/benchmark non-claims | Retain non-claim. |
| State-of-the-art status | Unsupported broad claim | Epic planning docs, README/INSTALL/benchmark/corpus/maintainer non-claims | Retain non-claim. |

## Duplicate Caveat Review

| Topic | Current duplication state | Day 3 disposition |
| --- | --- | --- |
| Package-manager support | README, INSTALL, maintainer guide, API docs, solver docs, corpus/benchmark docs all retain non-claims. | Duplication is acceptable near high-risk entry points. Prefer links to `INSTALL.md#support-readiness-matrix` during Day 5-Day 6 if wording becomes noisy. |
| Windows selected freshness | README, INSTALL, corpus docs, maintainer guide, and residual docs repeat the unpromoted/re-deferred state. | Retain in high-risk report surfaces; update stale residual/current-status docs for selected re-deferral wording. |
| Generated API local-only policy | README, API reference, maintainer guide, and INSTALL repeat local-only generated HTML boundaries. | Retain; consider clarifying historical Sprint 179/186 references only if they conflict with Sprint 204 current-policy wording. |
| Portable performance and benchmark claims | README, benchmarks, solver selection, corpus docs, maintainer guide, and manifest non-claims repeat threshold-free/non-portable limits. | Retain; these are high-risk claims and current wording is consistent. |
| State-of-the-art non-claim | README, INSTALL, solver selection, corpus docs, benchmark docs, maintainer guide, and Epic docs repeat the non-claim. | Retain. Final retrospective should keep the broad state-of-the-art non-claim while acknowledging selected governance/evidence improvements. |

## Stale Or Weak Wording To Fix Later

| Surface | Issue | Planned owner |
| --- | --- | --- |
| `EPIC_18_RETROSPECTIVE.md` Sprint 205 outcome row | Still says Sprint 205 is pending future execution even though Sprint 205 is merged and closed. | Day 11 retrospective update. |
| `EPIC_18_RETROSPECTIVE.md` non-claims | Says additional allocation-failure proof, additional review-surface reduction, hosted selected benchmark freshness on one additional platform, generated API publication policy, and adoption/support simplification are not currently claimed. Those are now selected closures or selected policy decisions, though broader claims remain unearned. | Day 11 retrospective update after Day 5-Day 7 status work. |
| `EPIC_18_RESIDUAL_QUEUE.md` E18-RQ-001 | Treats package/Homebrew work as pending future execution. It should record Sprint 198 selected local proof closure while preserving broad package-manager residuals. | Day 12 residual queue update. |
| `EPIC_18_RESIDUAL_QUEUE.md` E18-RQ-002 | Treats selected Windows Cholesky promotion as pending future execution. It should record Sprint 199 selected sprint closure as a re-deferral with guarded workflow evidence. | Day 12 residual queue update. |
| `PROJECT_PLAN.md` Sprint 206 status row | Points only to `SPRINT_197` artifacts for final-validation evidence. | Day 4-Day 7 project-plan status update to include explicit `SPRINT_206` branch evidence. |
| `docs/maintainer_guide.md` generated API history | Mentions Sprint 179 and Sprint 186 historical generated API decisions. The current policy is Sprint 204 stronger local-only; historical references are acceptable only if not read as the current owner. | Day 6 maintainer-doc check; Day 13 consistency hardening. |

## Proposed Stronger Claims

No stronger broad public support claim is proposed by the Day 3 audit.

The only claim-strengthening candidates are status-accuracy updates:

| Candidate wording | Evidence source | Boundary |
| --- | --- | --- |
| Sprint 198 closed a selected developer-mode local Homebrew static source proof. | Sprint 198 retrospective and artifacts. | Does not claim Homebrew/core, bottles, Linuxbrew, public tap, or broad package-manager support. |
| Sprint 200 closed selected `sparse_symbolic_lu()` allocation-failure owner proof. | Sprint 200 retrospective and artifacts. | Does not claim broad allocation-failure reliability. |
| Sprint 201 closed selected SVD helper review-surface reduction. | Sprint 201 retrospective and artifacts. | Does not claim repository-wide review-surface cleanup. |
| Sprint 202 closed selected macOS hosted benchmark freshness. | Sprint 202 hosted evidence artifacts. | Does not claim portable performance or timing thresholds. |
| Sprint 204 closed stronger local-only generated API policy. | Sprint 204 artifacts and API docs guards. | Does not claim hosted generated API publication. |
| Sprint 205 closed selected support matrix/quick-reference consolidation. | Sprint 205 retrospective and artifacts. | Does not claim broader adoption program or support promotion. |

## Unsupported Claims Routed To Non-Claims Or Residuals

| Unsupported claim | Routing |
| --- | --- |
| Broad package-manager distribution or Homebrew/core readiness | Keep as non-claim in README/INSTALL/API/maintainer docs; residual queue should distinguish selected local proof from broad distribution. |
| Bottles, Linuxbrew, public tap maintenance, vcpkg, Conan, pkgsrc, distro packages | Keep as explicit package-manager non-claims. |
| Broad Windows parity or Windows selected QR/benchmark freshness | Keep as non-claims; Windows Cholesky remains guarded/re-deferred and QR incompatible remains outside Windows selected freshness. |
| Shared-library or dynamic ABI support | Keep deferred in INSTALL and maintainer guide. |
| Hosted generated API publication, retained generated-doc artifacts, committed generated HTML | Keep unclaimed under Sprint 204 local-only policy. |
| Portable performance, timing thresholds, platform parity, backend superiority | Keep benchmark/report non-claims. |
| Release readiness | Keep residual/unsupported until release, packaging, ABI, and validation policy exist. |
| State-of-the-art sparse linear algebra status | Keep broad non-claim; selected evidence does not meet external-baseline, platform, package, performance, ABI, or release requirements. |

## Validation And Hygiene

| Check | Day 3 result |
| --- | --- |
| `git diff --check` | Planned after Day 3 artifact creation. |
| C/header quality gate | Not required for Day 3; no `.c` or `.h` edits. |
| Generated-output status | No generated output intentionally created. |
| User-facing claim drift | No public docs edited on Day 3. |
| Guard scripts/workflows/manifests | Not edited on Day 3. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 206.2 has a complete claim audit before edits begin. | Met. Public, maintainer, corpus, benchmark, API, support, and planning surfaces are classified. |
| Every proposed stronger claim has a specific Epic 18 evidence source. | Met. Only selected status-accuracy updates are proposed, each with sprint evidence. |
| Unsupported claims are routed to non-claims or residuals. | Met. Broad package, Windows, ABI, hosted API, performance, release, and state-of-the-art claims remain non-claims or residuals. |

## Day 3 Disposition

Day 3 is complete. Day 4 should design the final project-plan status update so
the explicit `SPRINT_206` branch becomes the current closeout path while
preserving the historical Sprint 197 numbering caveat.
