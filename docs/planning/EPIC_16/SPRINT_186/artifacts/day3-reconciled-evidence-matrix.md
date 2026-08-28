# Sprint 186 Day 3: Reconciled Evidence Matrix

## Purpose

Resolve the Day 2 weak-evidence queue and classify every Epic 16 project-plan
item from Sprints 177 through 185 with a final closeout status. This artifact
is the evidence basis for Day 4 claim inventory and Day 8 project-plan status
updates.

## Reconciliation Checks Performed

| Check | Result | Closeout effect |
| --- | --- | --- |
| Sprint 177-185 plan, working-notes, retrospective, and artifact packages | Present for every sprint. Sprint 182 intentionally has one extra decision artifact. | Prior sprint evidence can be used for final closeout. |
| Standalone license metadata for Homebrew local proof | No `LICENSE`, `COPYING`, or `NOTICE` file is present at repo root. | Homebrew local proof remains residualized as a proof-path blocker, not a support claim. |
| Local PowerShell availability for Windows report checks | `pwsh` is not available in the local environment. | Windows report freshness remains deferred with environment caveat and guard evidence. |
| Selected report target manifest | `tests/corpus/manifests/selected_report_targets.tsv` has seven selected rows plus header, including Sprint 183 Cholesky. | Manifest authority and selected Cholesky comparison evidence remain current. |
| Guard registrations | Makefile exposes `matmul-allocation-failure-gate`, `ldlt-csc-helper-guard`, `qr-header-docs-guard`, `api-docs-validate`, and `api-docs-freshness`. | Final validation planning can reuse existing guards. |
| PR #205 merged state | `master` includes merge commit `df945760`, with post-review commit `a64c1bc0` for LDLT CSC kernel override restoration. | Sprint 185 evidence must include the review fix in final validation notes. |

## Final Status Matrix

| Item | Item name | Final status | Evidence | Final rationale |
| --- | --- | --- | --- | --- |
| 177.1 | Residual Queue Audit | Complete | `SPRINT_177/artifacts/day2-residual-audit.md`; `day3-residual-classification.md` | Residuals were extracted and classified for Epic 16 target selection. |
| 177.2 | Evidence Status Matrix | Complete | `SPRINT_177/artifacts/day5-matrix-schema.md`; `day6-populated-matrix.md` | Matrix schema and baseline were created; Sprint 186 reconciles them against final outcomes. |
| 177.3 | Closure Target Selection | Complete | `SPRINT_177/artifacts/day7-target-selection.md` | Sprints 178-186 were selected as bounded Epic 16 closure targets. |
| 177.4 | Acceptance Gate Templates | Complete | `SPRINT_177/artifacts/day8-gate-templates.md`; `day9-gate-completion.md` | Acceptance gates exist and later sprint retrospectives record their validation evidence. |
| 177.5 | Quality Surface Map | Complete | `SPRINT_177/artifacts/day10-quality-surface-map.md` | Quality checks by change surface exist and feed Sprint 186 Day 9 validation planning. |
| 177.6 | Sprint Setup and Handoff | Complete | `SPRINT_177/artifacts/day12-handoff-package.md`; `day13-reconciliation.md`; `day14-closeout.md` | Sprint 177 produced complete handoff records for subsequent implementation sprints. |
| 178.1 | Subsystem Selection Detail | Complete | `SPRINT_178/artifacts/day3-subsystem-selection.md` | The selected proof surface is `sparse_matmul()` workspace allocation. |
| 178.2 | Cleanup Invariant Record | Complete | `SPRINT_178/artifacts/day4-cleanup-invariant.md`; `day9-cleanup-error-contracts.md` | Cleanup, stale-output suppression, retry, and non-publication rules are documented. |
| 178.3 | Harness Extension | Complete | `SPRINT_178/artifacts/day5-harness-design.md`; `day6-harness-implementation.md` | The private allocation-failure harness was extended for the selected subsystem. |
| 178.4 | Regression Tests | Complete | `SPRINT_178/artifacts/day7-first-regression.md`; `day8-coverage-expansion.md`; retrospective validation | Focused regressions cover selected `sparse_matmul()` allocation failures, stale-output suppression, and retry behavior. |
| 178.5 | Focused Gate | Complete | `SPRINT_178/artifacts/day10-focused-gate.md`; Makefile target `matmul-allocation-failure-gate` | The focused gate and registration guard are present. |
| 178.6 | Claim Documentation and Validation | Complete | `SPRINT_178/artifacts/day11-scoped-claim-documentation.md`; `day12-integrated-validation.md`; retrospective | Documentation and validation close the selected `sparse_matmul()` claim while retaining broad allocation-failure non-claims. |
| 179.1 | Doxygen Surface Audit | Complete | `SPRINT_179/artifacts/day2-doxygen-surface-audit.md`; `day3-warning-and-coverage-audit.md` | Generated API inputs, warnings, ignored outputs, and navigation were audited. |
| 179.2 | Publication Decision | Narrowed | `SPRINT_179/artifacts/day5-publication-decision-matrix.md`; `day6-product-decision-record.md` | Generated API HTML closed as strengthened local-only status, not hosted publication. |
| 179.3 | Implementation | Complete | `SPRINT_179/artifacts/day7-implementation-design.md`; `day8-core-implementation.md`; `day9-enforcement-completion.md` | Local-only enforcement and generated API freshness behavior were implemented. |
| 179.4 | Freshness and Staging Guard | Complete | `SPRINT_179/artifacts/day10-freshness-and-staging-guard.md`; `make api-docs-freshness` | Freshness and generated-output staging guards exist. |
| 179.5 | Navigation Update | Complete | `SPRINT_179/artifacts/day11-navigation-and-claim-update.md`; retrospective | README, API reference, and maintainer navigation point to the supported local API path. |
| 179.6 | Verification | Complete | `SPRINT_179/artifacts/day12-focused-verification.md`; `day13-integrated-validation.md`; `day14-closeout-and-handoff.md` | Generated API docs checks and whitespace validation passed; no C gate was required for the docs/script-only sprint. |
| 180.1 | Provider Feasibility Audit | Complete | `SPRINT_180/artifacts/day2-package-surface-audit.md`; `day3-vcpkg-feasibility.md`; `day4-homebrew-feasibility.md`; `day5-conan-feasibility.md`; `day6-pkgsrc-feasibility.md`; `day7-provider-decision-matrix.md` | Provider candidates were compared and Homebrew was selected as the local proof path. |
| 180.2 | Product Decision Record | Narrowed | `SPRINT_180/artifacts/day8-product-decision-record.md` | The outcome is a local Homebrew formula/tap proof path, not Homebrew support. |
| 180.3 | Recipe or Deferral Artifact | Complete | `SPRINT_180/artifacts/day9-artifact-design.md`; `day10-artifact-implementation.md`; Homebrew template/notes | Source-controlled provider material exists, with standalone license metadata retained as residual proof blocker. |
| 180.4 | Proof Script | Complete | `SPRINT_180/artifacts/day11-proof-script-design.md`; `day12-proof-script-implementation.md`; `scripts/homebrew_local_formula_proof.sh` | The proof script exists and fails claim-safely while license metadata is absent. |
| 180.5 | Guard and Docs Update | Complete | `SPRINT_180/artifacts/day13-guard-and-docs-update.md`; package-manager guards | Guards and docs preserve non-support wording. |
| 180.6 | Validation | Residualized | `SPRINT_180/artifacts/day14-integrated-validation-and-closeout.md`; retrospective | Package and install checks passed, but full Homebrew proof success is residualized until standalone license metadata exists. |
| 181.1 | Target Inventory | Complete | `SPRINT_181/artifacts/day1-report-target-intake.md`; `day2-report-surface-inventory.md`; `day3-workflow-and-guard-duplication.md` | Selected oracle, comparison, benchmark, workflow, and report metadata surfaces were inventoried. |
| 181.2 | Manifest Schema | Complete | `SPRINT_181/artifacts/day4-manifest-schema-design.md`; `day5-manifest-prototype.md`; `day6-parser-and-schema-checks.md` | The selected target manifest schema and parser checks exist. |
| 181.3 | Guard Refactor | Complete | `SPRINT_181/artifacts/day7-normalizer-refactor-design.md`; `day8-report-guard-refactor-batch-1.md`; `day9-report-guard-refactor-batch-2.md` | Report normalizer and workflow guards now read or validate against the manifest authority. |
| 181.4 | Workflow Scope Checks | Complete | `SPRINT_181/artifacts/day10-workflow-scope-checks.md` | YAML scope checks are manifest-backed and bounded to selected workflow blocks. |
| 181.5 | Documentation Alignment | Complete | `SPRINT_181/artifacts/day11-documentation-alignment.md` | Maintainer and report docs describe the selected target manifest as authority. |
| 181.6 | Validation | Complete | `SPRINT_181/artifacts/day12-diagnostics-and-drift-tests.md`; `day13-integrated-validation.md`; `day14-closeout-and-handoff.md` | Manifest, normalizer, selected workflow, freshness, and Python checks passed. |
| 182.1 | Windows Report Audit | Complete | `SPRINT_182/artifacts/day1-windows-freshness-scope-intake.md`; `day2-windows-workflow-and-toolchain-audit.md`; `day3-report-command-compatibility-audit.md`; `day4-artifact-and-data-semantics-audit.md` | Windows report command, workflow, and artifact assumptions were audited. |
| 182.2 | Candidate Selection | Deferred | `SPRINT_182/artifacts/day5-candidate-decision-matrix.md`; `day6-decision-record-design.md`; `windows-report-freshness-deferral-decision.md` | The selected product decision is formal Windows report freshness deferral. |
| 182.3 | CI or Deferral Implementation | Deferred | `SPRINT_182/artifacts/day7-implementation-batch-1.md`; `day8-implementation-batch-2.md`; decision artifact | Deferral artifact and guard behavior were implemented instead of a hosted Windows freshness lane. |
| 182.4 | Manifest Integration | Complete | `SPRINT_182/artifacts/day9-manifest-and-support-tier-alignment.md`; selected manifest | Windows status is represented in manifest/support-tier documentation. |
| 182.5 | Documentation Alignment | Complete | `SPRINT_182/artifacts/day10-documentation-alignment.md`; retrospective | Docs retain the Windows CMake/MSVC support boundary and do not claim report freshness. |
| 182.6 | Validation | Residualized | `SPRINT_182/artifacts/day11-guard-and-failure-diagnostics-hardening.md`; `day12-validation-sweep.md`; `day13-decision-reconciliation.md`; retrospective | Local checks passed with expected diagnostics; local PowerShell parse validation remains environment-residual because `pwsh` is unavailable. |
| 183.1 | Family Selection | Complete | `SPRINT_183/artifacts/day1-comparison-family-intake.md`; `day2-existing-comparison-surface-audit.md`; `day3-candidate-family-inventory.md`; `day4-family-selection.md` | The selected added comparison family is `cholesky_spd_tridiag_5`. |
| 183.2 | Fixture and Metric Contract | Complete | `SPRINT_183/artifacts/day5-fixture-and-metric-contract.md`; `day6-helper-and-fixture-implementation.md` | Source-controlled fixture and metric contract are defined for the selected Cholesky target. |
| 183.3 | Harness Extension | Complete | `SPRINT_183/artifacts/day7-runner-extension-design.md`; `day8-runner-implementation.md` | External comparison runner support was extended for the selected Cholesky family. |
| 183.4 | Report Integration | Complete | `SPRINT_183/artifacts/day9-report-integration.md`; `day10-freshness-gate-and-workflow-guard.md`; selected manifest | Cholesky selected comparison report is generated, indexed, freshness-checked, and manifest-registered. |
| 183.5 | Documentation Alignment | Complete | `SPRINT_183/artifacts/day11-documentation-alignment.md`; `day13-claim-review-and-hardening.md` | Documentation preserves selected-fixture comparison wording and broad parity non-claims. |
| 183.6 | Validation | Complete | `SPRINT_183/artifacts/day12-integrated-validation.md`; `day14-closeout-and-handoff.md`; retrospective | Comparison, Python/report, focused C, full C, and whitespace checks passed. |
| 184.1 | Header Family Selection | Complete | `SPRINT_184/artifacts/day1-sprint-intake.md`; `day2-declaration-baseline.md`; `day3-family-selection-and-contract-map.md` | QR was selected and declaration baseline was captured. |
| 184.2 | Contract Cleanup | Complete | `SPRINT_184/artifacts/day4-core-contract-cleanup.md`; `day5-advanced-contract-cleanup.md` | QR lifecycle, ownership, error-code, tolerance, workspace, option/result, cancellation, rank, and nullspace wording was cleaned. |
| 184.3 | Declaration Organization | Complete | `SPRINT_184/artifacts/day6-organization-guardrail-design.md`; `day7-coherent-header-sections.md` | QR declarations were reorganized into coherent sections without declaration-set drift. |
| 184.4 | Example and Docs Alignment | Complete | `SPRINT_184/artifacts/day8-documentation-alignment-map.md`; `day9-example-contract-alignment.md`; `day10-reference-documentation-alignment.md` | QR-facing examples and docs were aligned with the cleaned header contract. |
| 184.5 | Mechanical Guard | Complete | `SPRINT_184/artifacts/day11-mechanical-guard-implementation.md`; `scripts/check_qr_header_docs_guard.sh` | A focused QR header/docs guard protects declaration and unsupported-claim boundaries. |
| 184.6 | Validation | Complete | `SPRINT_184/artifacts/day12-focused-validation-pass.md`; `day13-full-validation-and-final-cleanup.md`; retrospective | QR guard, API docs validation, examples, full C gate, declaration diff, and whitespace checks passed. |
| 185.1 | Cluster Selection | Complete | `SPRINT_185/artifacts/day1-review-surface-intake.md`; `day2-candidate-cluster-baseline.md`; `day3-selected-cluster-decision.md` | `tests/test_ldlt_csc.c` was selected as the single review-surface reduction target. |
| 185.2 | Extraction Design | Complete | `SPRINT_185/artifacts/day4-helper-boundary-design.md`; `day5-registration-guardrail-design.md` | Header-only helper boundaries and no-behavior-change rules were defined before extraction. |
| 185.3 | Mechanical Extraction | Complete | `SPRINT_185/artifacts/day6-initial-helper-extraction.md`; `day7-fixture-setup-extraction.md`; `day8-proof-owner-cleanup.md`; PR #205 commit `a64c1bc0` | Supernode, fixture/setup, dense-oracle, and native-wrapper helpers were extracted; post-review override restoration fix is merged. |
| 185.4 | Drift Guard Update | Complete | `SPRINT_185/artifacts/day9-drift-guard-update.md`; `scripts/check_ldlt_csc_helper_guard.sh` | The LDLT CSC helper guard protects helper-header presence, include ownership, and registration boundaries. |
| 185.5 | Maintenance Note | Complete | `SPRINT_185/artifacts/day10-maintenance-invariants.md`; `day11-contributor-guidance-alignment.md`; `docs/maintainer_guide.md` | Maintainer guidance explains helper ownership and validation expectations. |
| 185.6 | Validation | Complete | `SPRINT_185/artifacts/day12-focused-cluster-validation.md`; `day13-full-quality-gate.md`; retrospective; PR #205 fix validation | Focused LDLT CSC tests, helper guard, source-list check, full C gate, and review-fix validation passed. |

## Weak-Evidence Resolution

| ID | Resolution | Final classification impact |
| --- | --- | --- |
| D2-WE-001 | Sprint 177 selected targets all have traceable outcomes: Sprint 178 complete, Sprint 179 narrowed local-only, Sprint 180 local proof path with residual proof blocker, Sprint 181 complete, Sprint 182 deferred, Sprint 183 complete, Sprint 184 complete, Sprint 185 complete, Sprint 186 active. | Sprint 177 rows remain Complete; later rows carry narrowed, deferred, or residualized status where appropriate. |
| D2-WE-002 | Homebrew provider artifacts and proof script exist, but no standalone license metadata is present. | 180.6 is Residualized for full proof success; 180.2 remains Narrowed to local proof path. |
| D2-WE-003 | Windows report freshness has formal deferral evidence and local `pwsh` is unavailable. | 182.2 and 182.3 are Deferred; 182.6 is Residualized for environment-dependent parse validation. |
| D2-WE-004 | Cholesky comparison evidence is selected-fixture-only and manifest-registered. | Sprint 183 rows are Complete with a claim boundary against broad Cholesky/external parity. |
| D2-WE-005 | QR header cleanup has guard and declaration-diff evidence with zero signature drift. | Sprint 184 rows are Complete with declaration-preserving claim boundary. |
| D2-WE-006 | PR #205 merged the kernel override restoration fix after Sprint 185 retrospective creation. | Sprint 185 rows remain Complete, with Day 3 final evidence explicitly including commit `a64c1bc0`. |

## Final Status Summary

| Final status | Count | Rows |
| --- | ---: | --- |
| Complete | 48 | All rows except 179.2, 180.2, 180.6, 182.2, 182.3, and 182.6. |
| Narrowed | 2 | 179.2, 180.2 |
| Deferred | 2 | 182.2, 182.3 |
| Residualized | 2 | 180.6, 182.6 |
| Superseded | 0 | No project-plan item was replaced by a later incompatible path. |

## Residual Candidates For Day 13

| Residual ID | Source | Priority signal | Closure target | Expected validation |
| --- | --- | --- | --- | --- |
| R186-PKG-LICENSE | Sprint 180 | Blocks full Homebrew local proof success. | Add approved standalone license metadata or explicitly decide an alternate formula license strategy. | `bash scripts/homebrew_local_formula_proof.sh`; `bash scripts/package_manager_deferral_check.sh`; install checks. |
| R186-WIN-PWSH | Sprint 182 | Blocks local PowerShell parse validation and keeps Windows report freshness deferred. | Run PowerShell parse/workflow checks in an environment with `pwsh`, or document hosted-only validation ownership. | PowerShell parse check plus selected report/workflow guard tests. |
| R186-WIN-REPORT-FRESHNESS | Sprint 182 | Windows selected report freshness remains a formal product deferral. | Select and prove one Windows-safe freshness lane or retain deferral with updated blockers. | Selected manifest validation, workflow guard tests, and Windows hosted/local freshness evidence. |
| R186-HOSTED-API | Sprint 179 | Generated API HTML remains local-only. | Decide hosted publication or retained artifact path only if product value justifies it. | API docs generation, publication/freshness guard, staging guard, docs navigation checks. |
| R186-BROAD-COMPARISON | Sprint 183 | Comparison evidence remains selected-fixture-only. | Add one bounded family at a time with fixture, metric, report, manifest, and claim evidence. | External comparison runner tests, selected freshness checks, manifest validation. |
| R186-REVIEW-SURFACE-NEXT | Sprint 185 | Other large review surfaces remain outside Sprint 185. | Select exactly one future large review surface and repeat the behavior-preserving extraction pattern. | Focused cluster validation, registration guards, full C gate when C/H files change. |

## Claim Calibration Inputs

Days 4 through 7 should preserve these boundaries:

- `sparse_matmul()` allocation-failure cleanup is the selected proof, not broad
  allocation-failure coverage.
- Generated API HTML is supported as local-only regenerated output, not hosted
  or committed generated documentation.
- Homebrew is a selected local formula/tap proof path, not package-manager
  support.
- Windows report freshness is formally deferred; Windows support remains
  CMake/MSVC configure, build, `ctest`, and static-first install/downstream
  validation.
- External comparison evidence covers selected fixtures and selected report
  rows, not broad external-library parity.
- QR header coherence is declaration-preserving; it does not add public API
  signatures or broad QR behavior claims.
- LDLT CSC review-surface reduction is behavior-preserving helper extraction,
  not solver behavior, correctness, or performance improvement.

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required. Required validation:

```sh
git diff --check
```
