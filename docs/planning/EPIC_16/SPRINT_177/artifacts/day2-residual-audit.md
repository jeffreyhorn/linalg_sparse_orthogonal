# Sprint 177 Day 2: Epic 13-15 Residual Queue Audit

## Purpose

Day 2 extracts unresolved residuals from the Epic 13, Epic 14, and Epic 15
retrospectives and detailed residual artifacts. It deduplicates repeated
themes by the claim or product surface they would enable, preserves predecessor
history, and records current blockers and evidence-owner surfaces for later
classification.

## Inputs Reviewed

| Source | Residual signal used |
| --- | --- |
| `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md` | Highest-priority next-epic candidates and long-horizon deferrals. |
| `docs/planning/EPIC_13/SPRINT_156/artifacts/day11-residual-queue-publication.md` | Owner roles, blockers, prerequisites, and promotion gates for Epic 13 residuals. |
| `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` | Epic 14 retained candidates and long-horizon deferrals. |
| `docs/planning/EPIC_14/SPRINT_166/artifacts/day13-final-residual-queue-and-closeout-prep.md` | Priority queue with blockers, prerequisites, and promotion gates. |
| `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md` | Freshest residual queue after PR #195 and Epic 15 closeout. |
| `docs/planning/EPIC_16/reviews/review-codex-2026-08-23.md` | Current Codex review findings and state-of-the-art assessment. |
| `docs/planning/EPIC_16/reviews/todo-codex-2026-08-23.md` | Step-by-step closure themes for Epic 16. |

## Consolidation Rules

- Merge residuals by the support claim or product surface they would enable,
  not by sprint number.
- Treat Epic 15 residuals as current unless a later PR explicitly closed them.
- Preserve older Epic 13/Epic 14 residual IDs as history when the same gap was
  narrowed but not fully closed.
- Separate complete-closure candidates from broad long-horizon product or
  research work.
- Do not select final Epic 16 targets on Day 2. Day 3 classification and the
  evidence/status matrix must happen first.

## Deduplicated Current Residual Queue

| ID | Current residual | Prior history | Current blocker | Evidence-owner surfaces | Candidate closure shape |
| --- | --- | --- | --- | --- | --- |
| S177-R01 | Broader allocation-failure coverage beyond iterative repeated-run handles | Newest in Epic 15; related to Epic 15 Sprint 176 proof boundary. | Current deterministic proof covers only CG, GMRES, and MINRES repeated-run handle prepare/growth cleanup. | `src/sparse_alloc_internal.*`, selected subsystem source, relevant tests, `Makefile`, `CMakeLists.txt`, `docs/maintainer_guide.md`, README quality wording. | Select one additional allocation-heavy subsystem and add deterministic fail-at-count proof, cleanup invariants, focused gate, and scoped docs. |
| S177-R02 | Generated API HTML hosted publication or stronger local-only status | E13-R15, Epic 14 P2, Epic 15 residual 2. | Generated API HTML remains local-only and cannot be cited as hosted/current user-facing publication. | `Doxyfile`, `docs/api_reference.md`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh`, README, maintainer guide, workflows if hosted. | Decide hosted/artifact/committed/local-only status and enforce the selected status with freshness and staging checks. |
| S177-R03 | Package-manager provider support or stronger provider deferral | E13-R03, Epic 14 P1, Epic 15 residual 3. | No provider recipe, provenance, install proof, cleanup proof, upgrade behavior, or registry readiness exists. | `INSTALL.md`, README, `docs/maintainer_guide.md`, `scripts/package_manager_deferral_check.sh`, package templates, provider prototype files if selected. | Select one provider and prove it, or strengthen formal deferral with exact blockers and updated guards. |
| S177-R04 | Shared-library and dynamic ABI product support | E13-R04/E13-R05, Epic 14 P1, Epic 15 residual 4. | Static-first-only is the current product decision; export/import, symbol visibility, SONAME/install-name/DLL, ABI policy, loader proof, and installed shared consumers are absent. | `CMakeLists.txt`, `Makefile`, `sparse.pc.in`, CMake package templates, install tests, static package deferral guard, public headers. | Reopen only with a funded ABI product sprint; otherwise retain guarded static-first non-claim. |
| S177-R05 | Windows generated report freshness | Epic 14 P2 macOS/Windows report freshness parity; Epic 15 residual 5. | Windows remains CMake-first and does not claim report generation/freshness; report commands may carry shell/path/dependency assumptions. | `.github/workflows/windows-ci.yml`, report scripts, Python tests, README, INSTALL, maintainer guide, selected report manifests once created. | Promote one Windows-safe selected report lane or add a stronger deferral artifact and guard. |
| S177-R06 | Selected oracle freshness beyond Linux | Epic 15 residual 6; related to Epic 14 hosted evidence promotion. | macOS selected comparison freshness is reviewed, but selected oracle freshness remains Linux-hosted only. | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py`, corpus/report docs. | Add macOS selected oracle lane only if runtime/dependency constraints are acceptable; otherwise retain scoped non-claim. |
| S177-R07 | Additional bounded external comparison family | E13-R10/E13-R12/E13-R13, Epic 14 P2, Epic 15 residual 7. | Existing comparison evidence remains fixture-local for selected QR, partial-SVD, and LU families; broad external parity remains unsupported. | `scripts/run_external_comparison.py`, comparison tests, selected report freshness, corpus docs, README, solver-selection docs. | Add one family with fixtures, metrics, tolerances, expected rows, report integration, freshness, and non-parity wording. |
| S177-R08 | Portable or broader performance publication | E13-R14, Epic 14 P1/P2 performance items, Epic 15 residual 8. | Only one selected Linux hosted performance freshness row exists; portable performance and superiority claims remain unsupported. | `benchmarks/`, `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`, `tests/test_bench_canonical_freshness.py`, CI workflows, benchmark docs. | Add one additional methodology-bound hosted performance row or retain current narrow evidence with explicit non-claims. |
| S177-R09 | Public-header coherence breadth | E13-R16, Epic 14 P1, Epic 15 residual 9. | Prior cleanup covered selected headers only; QR, SVD, LDLT, IC, ILU, reorder, and analysis surfaces remain uneven. | `include/`, `docs/api_reference.md`, examples, tutorial, cookbook, solver-selection docs, header guard scripts. | Select one high-risk header family and run declaration-preserving cleanup with docs and validation. |
| S177-R10 | Workflow and selected report target-list duplication | Epic 15 residual 10; reinforced by PR review history around workflow guards. | Selected target metadata is repeated in workflows, Python tests, scripts, docs, and generated report expectations. | `.github/workflows/*.yml`, `tests/test_selected_comparison_workflow.py`, `tests/test_normalize_report_index.py`, report scripts, maintainer guide. | Add a canonical selected-target manifest and make guards read it with duplicate/missing/mis-scoped checks. |
| S177-R11 | Broad generated report hosting/freshness | E13-R07/E13-R08, Epic 14 P3, Epic 15 long-horizon residual. | Selected rows have gates; unselected/advisory/deferred/generated families do not have broad hosted freshness. | report index normalizer, report-family manifests, workflows, maintainer guide, README. | Keep broad hosting deferred, or select one family at a time with exact runtime, artifact, and claim policy. |
| S177-R12 | Release packaging evidence and package-provider upgrade behavior | E13-R03 long-horizon; Epic 15 long-horizon residual. | No release workflow, provider upgrade proof, signed/provenance artifacts, binary package policy, or registry publishing evidence exists. | VERSION, package docs, install docs, provider scripts if any, release workflow if introduced. | Long-horizon unless a release/product sprint is explicitly selected. |
| S177-R13 | Large source/test review surface and maintainability drag | Epic 15 review finding; indirectly related to recurring header/proof-owner expansion. | Several proof-owner tests and solver files are very large; new coverage often expands already-large files. | large files under `tests/` and `src/`, Make/CMake registration, source-list checks, maintainer guide. | Extract one helper/proof-owner cluster with no behavior change and full validation. |
| S177-R14 | Claim governance remains distributed | Epic 15 "Could Be Better"; related to report target duplication. | README, INSTALL, maintainer guide, workflows, scripts, benchmark docs, package docs, report manifests, and planning artifacts all own support-tier fragments. | public docs, maintainer guide, report manifests, workflow comments, guard scripts. | Create an evidence/status matrix and use it as a current claim/status authority. |

## Older Residuals Closed Or Narrowed By Later Epics

| Older residual | Current status after Epic 15 |
| --- | --- |
| Hosted promotion for selected local-only oracle/comparison rows | Narrowed: Linux selected oracle/comparison and macOS selected comparison freshness exist; macOS oracle and broad report freshness remain residual. |
| One bounded QR comparison expansion | Narrowed: selected QR comparison families exist; broad QR comparison and additional bounded family work remain residual only if selected. |
| One bounded partial-SVD comparison publication | Narrowed: selected partial-SVD comparison exists; broader SVD/partial-SVD comparison remains bounded future work. |
| Windows package parity decision | Narrowed: Windows CMake install/downstream validation exists; Windows Makefile and `pkg-config` execution parity remain explicit non-claims. |
| Generated API HTML refresh/publication ambiguity | Narrowed: local-only decision exists; hosted publication remains residual. |
| Hosted performance publication decision | Narrowed: one selected Linux hosted performance freshness row exists; portable/broader performance remains residual. |
| Package-manager distribution readiness | Narrowed: formal deferral and guard exist; provider proof remains residual. |
| Shared-library ABI product design | Narrowed: static-first-only decision and guards exist; actual shared-library/dynamic ABI product remains residual. |

## Blocker And Evidence-Owner Notes

| Residual ID | Owner role | Current blocker | Minimum evidence needed before claim can widen |
| --- | --- | --- | --- |
| S177-R01 | Failure-path/test owner | Proof pattern is limited to one iterative family. | Deterministic failure tests for one new subsystem, cleanup invariant docs, focused gate, and retry proof. |
| S177-R02 | Documentation/API owner | No hosted generated API publication policy. | Product decision plus hosted artifact/freshness proof or stronger local-only guard. |
| S177-R03 | Package/distribution owner | No selected provider recipe or provider CI proof. | Provider decision, recipe/provenance, install/downstream/version/cleanup proof, support-tier docs. |
| S177-R04 | Package/ABI owner | Static-first-only support decision blocks shared-library claims. | Symbol/export policy, platform shared metadata, ABI policy, loader proof, installed shared consumers. |
| S177-R05 | Platform/report owner | Windows report runtime and tooling assumptions are unproven. | Windows-safe selected report command or explicit guarded deferral with exact blockers. |
| S177-R06 | Platform/corpus owner | macOS oracle runtime/dependency cost not selected. | macOS selected oracle lane with artifacts or retained non-claim. |
| S177-R07 | Comparison owner | Existing comparisons are selected and fixture-local. | One new comparison family with source-controlled references, metrics, tolerances, rows, freshness, docs. |
| S177-R08 | Benchmark owner | One narrow Linux row is not portable methodology. | Additional hosted row or explicit retention of narrow evidence. |
| S177-R09 | Header/API owner | Remaining public headers lack uniform cleanup. | Selected header baseline, declaration-preserving cleanup, docs/example updates, full gate if headers change. |
| S177-R10 | Report/workflow owner | Target metadata is duplicated across YAML/tests/docs. | Canonical selected-target manifest plus manifest-driven guards. |
| S177-R11 | Report governance owner | Broad generated families lack runtime/support policy. | Selected-family-by-family promotion or broad deferral matrix. |
| S177-R12 | Release/package owner | No release/provider upgrade provenance. | Release policy, artifact provenance, provider upgrade proof, registry readiness. |
| S177-R13 | Maintainability owner | Large files and proof-owner clusters increase review cost. | No-behavior-change extraction with registration checks and full validation. |
| S177-R14 | Product/docs owner | Support-tier truth is distributed. | Evidence/status matrix plus docs links and guard strategy. |

## Long-Horizon Non-Goals Unless Explicitly Selected

These gaps remain visible but should not be treated as Sprint 177 selections:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against SuiteSparse, PETSc,
  Trilinos, Eigen, SciPy, LAPACK, vendor libraries, or package-manager
  ecosystems;
- portable performance superiority or backend superiority;
- shared-library support across Linux, macOS, and Windows;
- dynamic ABI compatibility policy and compatibility testing;
- runtime-loader behavior guarantees;
- broad Windows platform/package parity;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad generated report hosting for every report family;
- release packaging evidence and package-provider upgrade behavior;
- broad allocation-failure proof across every allocation path.

## Day 2 Deliverables

- Deduplicated current residual queue.
- Older residual closure/narrowing table.
- Blocker and evidence-owner notes.
- Long-horizon non-goal list.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Prior-epic residuals are visible in one place | Complete | Deduplicated queue above consolidates Epic 13-15 residuals. |
| Duplicate residuals are collapsed without losing history | Complete | Queue includes predecessor history and closed/narrowed table. |
| Broad state-of-the-art, ABI, platform, and ecosystem parity gaps remain explicit non-goals unless selected later | Complete | Long-horizon non-goals are listed separately and not selected on Day 2. |

