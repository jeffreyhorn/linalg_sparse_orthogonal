# Sprint 186 Day 2: Evidence Matrix Baseline

## Purpose

Build the initial cross-sprint evidence/status matrix for Epic 16 closeout.
This baseline maps every project-plan item from Sprints 177 through 185 to
source-controlled artifacts, validation evidence, and claim surfaces. Day 3
will reconcile weak evidence rows and finalize classifications.

## Matrix Status Rules

| Status | Day 2 meaning |
| --- | --- |
| Complete | The item has an apparent source-controlled outcome and validation or documentation evidence. |
| Complete with residual | The item delivered the scoped outcome, but its retrospective records follow-up work or retained non-claims. |
| Narrowed | The item delivered a deliberately smaller product decision or proof than a broad support claim would imply. |
| Deferred | The item intentionally closed as a deferral with blockers and guard coverage. |
| Needs Day 3 check | Evidence exists, but the row needs closer reconciliation before final closeout status. |

## Initial Evidence Matrix

| Item | Item name | Day 2 status | Primary evidence | Validation evidence | Claim surfaces affected | Day 3 follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| 177.1 | Residual Queue Audit | Complete | `SPRINT_177/artifacts/day2-residual-audit.md`; `day3-residual-classification.md` | `SPRINT_177/RETROSPECTIVE.md` validation: `git diff --check` | Epic 16 residual queue; planning artifacts | Confirm residuals were consumed or carried forward. |
| 177.2 | Evidence Status Matrix | Complete | `SPRINT_177/artifacts/day5-matrix-schema.md`; `day6-populated-matrix.md` | `git diff --check` | Evidence/status matrix | Reconcile the Day 6 matrix against Sprint 178-185 outcomes. |
| 177.3 | Closure Target Selection | Complete | `SPRINT_177/artifacts/day7-target-selection.md` | `git diff --check` | Sprint 178-186 target register | Confirm all selected targets now have outcomes. |
| 177.4 | Acceptance Gate Templates | Complete | `SPRINT_177/artifacts/day8-gate-templates.md`; `day9-gate-completion.md` | `git diff --check` | Acceptance gates and review traps | Confirm later validation records cite or satisfy the gates. |
| 177.5 | Quality Surface Map | Complete | `SPRINT_177/artifacts/day10-quality-surface-map.md` | `git diff --check` | Required quality checks by change type | Use as Day 9 validation-plan input. |
| 177.6 | Sprint Setup and Handoff | Complete | `SPRINT_177/artifacts/day12-handoff-package.md`; `day13-reconciliation.md`; `day14-closeout.md` | `git diff --check` | Sprint handoffs | Confirm all handoffs either closed or residualized. |
| 178.1 | Subsystem Selection Detail | Complete | `SPRINT_178/artifacts/day3-subsystem-selection.md` | Sprint 178 focused validation records | Allocation-failure proof scope | Confirm selected subsystem is `sparse_matmul()` workspace allocation. |
| 178.2 | Cleanup Invariant Record | Complete | `SPRINT_178/artifacts/day4-cleanup-invariant.md`; `day9-cleanup-error-contracts.md` | `make matmul-allocation-failure-gate` | Cleanup and stale-output non-claims | Confirm invariant wording survives final docs. |
| 178.3 | Harness Extension | Complete | `SPRINT_178/artifacts/day5-harness-design.md`; `day6-harness-implementation.md` | registration guard and CMake selector evidence in retrospective | Private allocation-failure hook use | Confirm no public allocation-failure API claim was introduced. |
| 178.4 | Regression Tests | Complete | `SPRINT_178/artifacts/day7-first-regression.md`; `day8-coverage-expansion.md` | `make matmul-allocation-failure-gate`; `make test` | Deterministic failure-path proof | Confirm final test names and coverage are still present. |
| 178.5 | Focused Gate | Complete | `SPRINT_178/artifacts/day10-focused-gate.md` | `python3 tests/test_matmul_allocation_failure_gate_registration.py`; CTest labels | Focused gate documentation | Confirm gate remains registered. |
| 178.6 | Claim Documentation and Validation | Complete with residual | `SPRINT_178/artifacts/day11-scoped-claim-documentation.md`; `day12-integrated-validation.md`; `day14-closeout.md` | `make format && make lint && make test`; docs hygiene | README and maintainer allocation-failure claims | Preserve narrow `sparse_matmul()` scope. |
| 179.1 | Doxygen Surface Audit | Complete | `SPRINT_179/artifacts/day2-doxygen-surface-audit.md`; `day3-warning-and-coverage-audit.md` | Doxygen coverage script for 18 public headers | Generated API docs | Confirm header input count/status is current. |
| 179.2 | Publication Decision | Narrowed | `SPRINT_179/artifacts/day5-publication-decision-matrix.md`; `day6-product-decision-record.md` | local-only guard evidence | API reference and generated API status | Preserve local-only generated HTML decision. |
| 179.3 | Implementation | Complete | `SPRINT_179/artifacts/day7-implementation-design.md`; `day8-core-implementation.md`; `day9-enforcement-completion.md` | `make api-docs-freshness`; local-only guard | `docs/api_reference.md`; `docs/api` non-staging | Confirm generated output remains untracked. |
| 179.4 | Freshness and Staging Guard | Complete | `SPRINT_179/artifacts/day10-freshness-and-staging-guard.md` | `make api-docs-freshness`; `bash -n scripts/check_api_docs_local_only.sh` | Generated API freshness | Include in Day 9 validation matrix. |
| 179.5 | Navigation Update | Complete | `SPRINT_179/artifacts/day11-navigation-and-claim-update.md` | docs whitespace and generated API checks | README, API reference, maintainer guide | Check for stale hosted/publication wording. |
| 179.6 | Verification | Complete | `SPRINT_179/artifacts/day12-focused-verification.md`; `day13-integrated-validation.md`; `day14-closeout-and-handoff.md` | `make api-docs-freshness`; `git diff --check` | Generated API validation evidence | Confirm no C gate required for final edits unless headers change. |
| 180.1 | Provider Feasibility Audit | Complete | `SPRINT_180/artifacts/day2-package-surface-audit.md`; `day3-vcpkg-feasibility.md`; `day4-homebrew-feasibility.md`; `day5-conan-feasibility.md`; `day6-pkgsrc-feasibility.md`; `day7-provider-decision-matrix.md` | package checks in retrospective | Package-manager provider decision | Confirm Homebrew remains selected proof path. |
| 180.2 | Product Decision Record | Narrowed | `SPRINT_180/artifacts/day8-product-decision-record.md` | package-manager guard evidence | Package docs and README support tiers | Preserve proof-path wording, not support wording. |
| 180.3 | Recipe or Deferral Artifact | Complete with residual | `SPRINT_180/artifacts/day9-artifact-design.md`; `day10-artifact-implementation.md`; Homebrew template/notes | Homebrew local proof exits claim-safe unavailable on missing license metadata | Homebrew formula/tap proof | Day 3 should classify license metadata blocker. |
| 180.4 | Proof Script | Complete with residual | `SPRINT_180/artifacts/day11-proof-script-design.md`; `day12-proof-script-implementation.md` | `bash -n scripts/homebrew_local_formula_proof.sh`; proof exits `2` claim-safe unavailable | Package proof script | Confirm unavailable proof remains a non-claim. |
| 180.5 | Guard and Docs Update | Complete | `SPRINT_180/artifacts/day13-guard-and-docs-update.md` | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh` | README, INSTALL, maintainer guide, package metadata | Verify final claim calibration does not promote Homebrew support. |
| 180.6 | Validation | Complete with residual | `SPRINT_180/artifacts/day14-integrated-validation-and-closeout.md`; `SPRINT_180/RETROSPECTIVE.md` | install checks, CMake install checks, package guards, Ruby syntax | Package/provider validation record | Carry standalone license metadata as residual. |
| 181.1 | Target Inventory | Complete | `SPRINT_181/artifacts/day1-report-target-intake.md`; `day2-report-surface-inventory.md`; `day3-workflow-and-guard-duplication.md` | report/index tests in retrospective | Selected report targets | Confirm target set remains six rows. |
| 181.2 | Manifest Schema | Complete | `SPRINT_181/artifacts/day4-manifest-schema-design.md`; `day5-manifest-prototype.md`; `day6-parser-and-schema-checks.md` | `python3 scripts/validate_corpus_schema.py`; manifest tests | `tests/corpus/manifests/selected_report_targets.tsv` | Verify schema fields still satisfy later Windows/comparison status. |
| 181.3 | Guard Refactor | Complete | `SPRINT_181/artifacts/day7-normalizer-refactor-design.md`; `day8-report-guard-refactor-batch-1.md`; `day9-report-guard-refactor-batch-2.md` | normalizer, selected workflow, and manifest tests | Workflow/report guard behavior | Confirm duplicated target lists stayed removed. |
| 181.4 | Workflow Scope Checks | Complete | `SPRINT_181/artifacts/day10-workflow-scope-checks.md` | `python3 tests/test_selected_comparison_workflow.py` | YAML guard scope | Include workflow guard in integrated validation. |
| 181.5 | Documentation Alignment | Complete | `SPRINT_181/artifacts/day11-documentation-alignment.md` | docs and report freshness checks | Maintainer/report docs | Check for stale manifest authority wording. |
| 181.6 | Validation | Complete | `SPRINT_181/artifacts/day12-diagnostics-and-drift-tests.md`; `day13-integrated-validation.md`; `day14-closeout-and-handoff.md` | selected oracle/comparison/benchmark freshness; Python compile checks | Selected report validation | Confirm benchmark freshness sequential caveat is preserved. |
| 182.1 | Windows Report Audit | Complete | `SPRINT_182/artifacts/day1-windows-freshness-scope-intake.md`; `day2-windows-workflow-and-toolchain-audit.md`; `day3-report-command-compatibility-audit.md`; `day4-artifact-and-data-semantics-audit.md` | Python/report diagnostics in retrospective | Windows report freshness | Confirm audit supports final deferral. |
| 182.2 | Candidate Selection | Deferred | `SPRINT_182/artifacts/day5-candidate-decision-matrix.md`; `day6-decision-record-design.md`; `windows-report-freshness-deferral-decision.md` | guard diagnostics and expected local stale warnings | Windows support tiers | Preserve formal deferral wording. |
| 182.3 | CI or Deferral Implementation | Deferred | `SPRINT_182/artifacts/day7-implementation-batch-1.md`; `day8-implementation-batch-2.md`; decision artifact | selected workflow and manifest diagnostics | Workflow and report guards | Confirm no hosted Windows freshness lane is claimed. |
| 182.4 | Manifest Integration | Complete | `SPRINT_182/artifacts/day9-manifest-and-support-tier-alignment.md` | corpus schema and manifest diagnostics | Selected target manifest, support-tier docs | Confirm manifest records Windows deferred status. |
| 182.5 | Documentation Alignment | Complete | `SPRINT_182/artifacts/day10-documentation-alignment.md` | docs/report diagnostics | README, INSTALL, maintainer guide, report-index docs | Check final docs retain Windows CMake-only support boundary. |
| 182.6 | Validation | Complete with residual | `SPRINT_182/artifacts/day11-guard-and-failure-diagnostics-hardening.md`; `day12-validation-sweep.md`; `day13-decision-reconciliation.md`; `day14-closeout-and-handoff.md` | Python checks passed; `pwsh` unavailable locally | Windows freshness validation | Day 3 should classify `pwsh` unavailability as environment caveat. |
| 183.1 | Family Selection | Complete | `SPRINT_183/artifacts/day1-comparison-family-intake.md`; `day2-existing-comparison-surface-audit.md`; `day3-candidate-family-inventory.md`; `day4-family-selection.md` | comparison runner tests | External comparison family | Confirm selected family is `cholesky_spd_tridiag_5`. |
| 183.2 | Fixture and Metric Contract | Complete | `SPRINT_183/artifacts/day5-fixture-and-metric-contract.md`; `day6-helper-and-fixture-implementation.md` | Cholesky helper test; focused Cholesky C test | Fixture/metric docs | Preserve narrow fixture claim. |
| 183.3 | Harness Extension | Complete | `SPRINT_183/artifacts/day7-runner-extension-design.md`; `day8-runner-implementation.md` | `python3 tests/test_run_external_comparison.py` | External comparison runner | Confirm no broad external-library parity claim. |
| 183.4 | Report Integration | Complete | `SPRINT_183/artifacts/day9-report-integration.md`; `day10-freshness-gate-and-workflow-guard.md` | selected comparison freshness; report-index tests | Selected comparison report | Include selected comparison freshness in Day 9 validation matrix. |
| 183.5 | Documentation Alignment | Complete | `SPRINT_183/artifacts/day11-documentation-alignment.md`; `day13-claim-review-and-hardening.md` | docs/report checks | README, solver-selection, maintainer guide, corpus/report docs | Check Cholesky claim remains selected-fixture only. |
| 183.6 | Validation | Complete | `SPRINT_183/artifacts/day12-integrated-validation.md`; `day14-closeout-and-handoff.md`; retrospective | `make format && make lint && make test`; Python/report checks | Comparison validation evidence | Confirm tracked C/header diffs remain none after formatting. |
| 184.1 | Header Family Selection | Complete | `SPRINT_184/artifacts/day1-sprint-intake.md`; `day2-declaration-baseline.md`; `day3-family-selection-and-contract-map.md` | declaration baseline/check evidence | QR public header docs | Confirm selected family is QR. |
| 184.2 | Contract Cleanup | Complete | `SPRINT_184/artifacts/day4-core-contract-cleanup.md`; `day5-advanced-contract-cleanup.md` | `make qr-header-docs-guard`; API docs validation | QR lifecycle/ownership/tolerance/workspace docs | Confirm no signature drift. |
| 184.3 | Declaration Organization | Complete | `SPRINT_184/artifacts/day6-organization-guardrail-design.md`; `day7-coherent-header-sections.md` | sorted comment-stripped declaration-set diff | QR header declaration organization | Confirm bounded declaration moves only. |
| 184.4 | Example and Docs Alignment | Complete | `SPRINT_184/artifacts/day8-documentation-alignment-map.md`; `day9-example-contract-alignment.md`; `day10-reference-documentation-alignment.md` | examples build and QR example smokes | examples, tutorial, solver-selection, API reference | Confirm examples still build if later docs touched. |
| 184.5 | Mechanical Guard | Complete | `SPRINT_184/artifacts/day11-mechanical-guard-implementation.md` | `make qr-header-docs-guard`; `make api-docs-validate` | QR unsupported-claim guard | Include guard in Day 9 validation matrix. |
| 184.6 | Validation | Complete | `SPRINT_184/artifacts/day12-focused-validation-pass.md`; `day13-full-validation-and-final-cleanup.md`; `day14-retrospective-ready-handoff.md` | `make format && make lint && make test`; examples and docs checks | QR header validation evidence | Confirm generated API HTML remains unstaged. |
| 185.1 | Cluster Selection | Complete | `SPRINT_185/artifacts/day1-review-surface-intake.md`; `day2-candidate-cluster-baseline.md`; `day3-selected-cluster-decision.md` | focused LDLT CSC validation later in sprint | Review-surface reduction | Confirm selected cluster is `tests/test_ldlt_csc.c`. |
| 185.2 | Extraction Design | Complete | `SPRINT_185/artifacts/day4-helper-boundary-design.md`; `day5-registration-guardrail-design.md` | focused build/test plan | Helper ownership and no-behavior-change contract | Confirm header-only strategy remained true. |
| 185.3 | Mechanical Extraction | Complete | `SPRINT_185/artifacts/day6-initial-helper-extraction.md`; `day7-fixture-setup-extraction.md`; `day8-proof-owner-cleanup.md` | `./build/test_ldlt_csc`; full C gate | LDLT CSC helper headers | Confirm process-global override review fix is included after PR #205. |
| 185.4 | Drift Guard Update | Complete | `SPRINT_185/artifacts/day9-drift-guard-update.md` | `make ldlt-csc-helper-guard`; `make source-list-check` | Helper-header registration boundaries | Include helper guard in Day 9 validation matrix. |
| 185.5 | Maintenance Note | Complete | `SPRINT_185/artifacts/day10-maintenance-invariants.md`; `day11-contributor-guidance-alignment.md`; `docs/maintainer_guide.md` | docs/guard checks | Maintainer guide helper ownership | Check final maintainer docs do not overstate behavior changes. |
| 185.6 | Validation | Complete | `SPRINT_185/artifacts/day12-focused-cluster-validation.md`; `day13-full-quality-gate.md`; `day14-review-ready-handoff.md`; retrospective | `make format && make lint && make test`; helper guard; source-list check | Review-surface validation evidence | Confirm no production/public API claims were added. |

## Initial Weak-Evidence And Residual List

| ID | Source row | Issue | Day 3 reconciliation target |
| --- | --- | --- | --- |
| D2-WE-001 | 177.1 through 177.6 | Sprint 177 created planning infrastructure; later sprint outcomes need to be reconciled back to the original residual and selected-target records. | Mark which selected targets closed, narrowed, deferred, or remain residual. |
| D2-WE-002 | 180.3, 180.4, 180.6 | Homebrew local proof exists but completes as claim-safe unavailable because standalone license metadata is missing. | Classify as narrowed provider proof path with license metadata residual. |
| D2-WE-003 | 182.2, 182.3, 182.6 | Windows report freshness closed as a formal deferral; local `pwsh` parse validation was unavailable. | Classify as deferred with environment caveat and guard evidence. |
| D2-WE-004 | 183.1 through 183.6 | The added comparison family is intentionally one selected Cholesky fixture, not broad Cholesky/external parity. | Preserve selected-fixture-only claim wording during calibration. |
| D2-WE-005 | 184.1 through 184.6 | QR header declarations were reorganized without declaration-set drift; final closeout should not imply API additions. | Confirm declaration-preserving claim and guard evidence. |
| D2-WE-006 | 185.3 and 185.6 | PR #205 review added a post-retrospective fix for kernel override restoration in an extracted helper. | Include the merged review fix in final evidence and validation notes. |

## Documentation Claim Surfaces Identified

| Surface | Evidence rows that affect it |
| --- | --- |
| `README.md` | 178.6, 179.5, 180.5, 182.5, 183.5, 184.4 |
| `INSTALL.md` | 180.5, 182.5 |
| `docs/maintainer_guide.md` | 178.6, 179.5, 180.5, 181.5, 182.5, 183.5, 184.4, 185.5 |
| `docs/api_reference.md` and generated API status docs | 179.2 through 179.6, 184.2 through 184.6 |
| report-index and selected-report docs | 181.1 through 181.6, 182.4, 183.4 |
| package-manager docs and metadata | 180.1 through 180.6 |
| public header and example docs | 184.1 through 184.6 |
| Epic 16 planning and retrospective docs | 177.1 through 186.6 |

## Day 3 Handoff

Day 3 should convert this baseline into the reconciled evidence matrix by:

1. checking each weak-evidence row against current source files and merged PR
   state;
2. marking final statuses with the Day 1 vocabulary;
3. identifying exact residual closure targets for package provider proof,
   Windows report freshness, broad comparison parity, declaration/API claims,
   and post-review helper validation;
4. preserving non-claims for broad package-manager support, Windows report
   freshness, hosted generated API HTML, shared-library ABI, portable
   performance, external-library parity, and state-of-the-art status.

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required. Required validation:

```sh
git diff --check
```
