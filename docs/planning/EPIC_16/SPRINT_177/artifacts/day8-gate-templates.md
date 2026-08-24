# Sprint 177 Day 8: Acceptance Gate Template Design

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_177/`
**Status:** Complete

## Purpose

Define reusable acceptance gate templates for the first five selected Epic 16
closure targets. These templates convert Day 7 target selection into concrete
pass/fail expectations for Sprints 178-182.

## Gate Template Fields

Every Epic 16 implementation sprint should fill these fields before code or
workflow changes are treated as complete:

| Field | Required meaning |
| --- | --- |
| Target | Selected Sprint 177 closure target. |
| Owner files | Files that must be changed, guarded, or inspected. |
| Required evidence | Concrete proof that the selected target is closed. |
| Validation commands | Commands that must pass before closeout. |
| Pass definition | What is enough to claim closure. |
| Fail definition | What blocks closure or forces residualization. |
| Claim boundary | Positive wording allowed after the gate passes. |
| Protected non-claims | Adjacent claims that remain unsupported. |
| Documentation updates | Public or maintainer docs that must be aligned. |
| Handoff artifact | Sprint artifact that records final evidence and residuals. |

## Gate 1: Allocation-Failure Proof Batch 2

**Target sprint:** 178
**Residual:** S177-R01
**Matrix row:** ESM-010

| Field | Acceptance requirement |
| --- | --- |
| Owner files | Selected subsystem implementation and tests; likely `src/sparse_alloc_internal.*`, subsystem source files, `tests/test_*.c`, `Makefile`, and `CMakeLists.txt` if a new target is added. |
| Required evidence | Deterministic injected allocation failure covers one additional subsystem, proves cleanup on failure, proves no stale public state publication, and proves successful retry after reset. |
| Validation commands | Focused subsystem gate; `make format`; `make lint`; `make test`; CMake/CTest validation if test registration changes. |
| Pass definition | The selected subsystem has a named fail-at-count or equivalent harness, at least one failure case per selected ownership path, cleanup assertions, recovery assertions, and a focused Make/CTest entry or label. |
| Fail definition | Failures are nondeterministic, cleanup cannot be asserted, public state can be partially published, retry behavior is unproven, or the docs imply broad allocation-failure coverage. |
| Claim boundary | One additional named subsystem has deterministic allocation-failure cleanup evidence. |
| Protected non-claims | No broad allocation-failure guarantee across all solvers, constructors, package/install flows, generated tooling, or unrelated allocation paths. |
| Documentation updates | README feature bullet, maintainer allocation-failure section, selected sprint artifacts, and any new validation target description. |
| Handoff artifact | Sprint 178 cleanup invariant and regression-gate artifact. |

## Gate 2: Generated API HTML Publication Or Local-Only Status

**Target sprint:** 179
**Residual:** S177-R02
**Matrix rows:** ESM-005, ESM-011

| Field | Acceptance requirement |
| --- | --- |
| Owner files | `Doxyfile`, `docs/api_reference.md`, public headers, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh`, `Makefile`, README, maintainer guide, and any workflow or publication metadata if selected. |
| Required evidence | A product decision chooses exactly one status: hosted publication, retained CI artifact, committed generated output, or strengthened local-only policy. The chosen status has matching freshness/staging guards. |
| Validation commands | `make docs-check`; `make api-docs-freshness`; `git diff --check`; workflow syntax/guard checks if hosted or artifact publication is added. |
| Pass definition | Docs navigation, Doxygen inputs, generated-output behavior, freshness checks, ignored/staged-file policy, and public support wording all match the selected status. |
| Fail definition | Generated HTML status remains ambiguous, stale output can be cited as current, generated files can be staged unexpectedly, or docs imply hosted/source-controlled publication without evidence. |
| Claim boundary | Generated API HTML has a single enforced product status and a maintained validation path. |
| Protected non-claims | No dynamic ABI, shared-library, package-manager, broad Windows, external-library parity, release, or completeness claim beyond configured Doxygen inputs. |
| Documentation updates | README command list, `docs/api_reference.md`, maintainer guide generated API section, tutorial/cookbook navigation if affected. |
| Handoff artifact | Sprint 179 publication decision and freshness/staging guard artifact. |

## Gate 3: Package-Manager Provider Proof Or Deferral

**Target sprint:** 180
**Residual:** S177-R03
**Matrix rows:** ESM-002, ESM-003, ESM-004

| Field | Acceptance requirement |
| --- | --- |
| Owner files | `scripts/package_manager_deferral_check.sh`, package metadata templates, `INSTALL.md`, README, maintainer guide, provider proof or deferral artifacts, and optional provider prototype files if selected. |
| Required evidence | The sprint either proves one static-first provider path with a local proof script or publishes a stronger formal deferral with exact blockers and fail-closed guards. |
| Validation commands | Provider proof or deferral script; `bash scripts/package_manager_deferral_check.sh`; install checks if package metadata changes; `git diff --check`; full quality gates if C/header files change. |
| Pass definition | Exactly one provider decision is recorded, package-manager wording and metadata are consistent with that decision, unsupported providers remain absent or guarded, and static-first/non-ABI boundaries remain intact. |
| Fail definition | The sprint introduces provider wording without proof, creates recipe files without a guard decision, weakens static-first metadata, or implies broad package-manager support. |
| Claim boundary | One package-manager provider path is proven, or provider support is more strongly and explicitly deferred. |
| Protected non-claims | No broad package-manager ecosystem support, binary package availability, upgrade behavior, registry readiness, shared-library support, or dynamic ABI compatibility. |
| Documentation updates | README, INSTALL support split, maintainer package section, package metadata comments, and Sprint 180 decision artifact. |
| Handoff artifact | Sprint 180 provider decision and proof/deferral validation artifact. |

## Gate 4: Selected Report Target Manifest

**Target sprint:** 181
**Residual:** S177-R10
**Matrix rows:** ESM-006, ESM-007, ESM-009, ESM-013

| Field | Acceptance requirement |
| --- | --- |
| Owner files | New selected-target manifest, `Makefile`, `scripts/normalize_report_index.py`, `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, benchmark report scripts, workflow guard tests, `.github/workflows/*.yml`, README, maintainer guide, and benchmark docs. |
| Required evidence | A source-controlled manifest owns selected oracle, comparison, performance, artifact, expected-row, support-tier, and workflow upload metadata with duplicate detection. |
| Validation commands | `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_normalize_report_index.py`; selected freshness Make targets; Python compile checks for changed scripts; `git diff --check`. |
| Pass definition | Workflows, guards, and docs read or validate against manifest-owned expectations, duplicates fail clearly, upload blocks remain fail-closed, and selected target changes require one manifest update. |
| Fail definition | Target lists remain hand-duplicated in several owners, missing/duplicate rows pass silently, workflow upload checks are only broad substring checks, or selected target support tiers can drift. |
| Claim boundary | Selected report target metadata has one canonical reviewed owner used by local and workflow guards. |
| Protected non-claims | No broad report-index freshness, unselected oracle/comparison/performance freshness, release proof, package/ABI support, platform parity, or state-of-the-art claim. |
| Documentation updates | Maintainer normalized report workflow, README report command notes, benchmark report-index handoff, and workflow comments. |
| Handoff artifact | Sprint 181 manifest schema, guard refactor, and validation artifact. |

## Gate 5: Windows Report Freshness Promotion Or Deferral

**Target sprint:** 182
**Residual:** S177-R05
**Matrix rows:** ESM-008, ESM-012, ESM-013

| Field | Acceptance requirement |
| --- | --- |
| Owner files | `.github/workflows/windows-ci.yml`, selected-target manifest if available, report scripts, workflow guard tests, README, INSTALL, maintainer guide, and Sprint 182 decision artifact. |
| Required evidence | The sprint either adds one Windows-safe reviewed generated report freshness path or closes Windows report freshness as an explicit guarded deferral with exact blockers. |
| Validation commands | Windows workflow guard or PowerShell/YAML syntax review; selected report freshness checks where locally feasible; `python3 tests/test_selected_comparison_workflow.py`; report-index tests if metadata changes; `git diff --check`. |
| Pass definition | If promoted, the Windows lane runs one bounded freshness target and uploads/scopes artifacts fail-closed. If deferred, public docs and guards reject accidental Windows report freshness claims. |
| Fail definition | Windows report freshness wording is promoted without a working lane, shell/path/runtime assumptions are unresolved, artifact upload checks are not scoped, or docs imply broad Windows parity. |
| Claim boundary | Windows report freshness is proven for exactly one selected family or explicitly deferred with guard coverage. |
| Protected non-claims | No Windows Makefile parity, Windows pkg-config execution parity, broad Windows parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or all-report parity. |
| Documentation updates | README CI bullet, INSTALL supported platforms, maintainer platform/report sections, workflow comments, and selected-target manifest docs if used. |
| Handoff artifact | Sprint 182 promotion or deferral decision and validation artifact. |

## Cross-Gate Review Checks

Every Day 8 gate must preserve these checks:

- A selected target must have a pass/fail command or a formal decision record.
- Adjacent broad claims must remain explicit non-claims.
- Hosted evidence must name exact workflow job scope.
- Local-only evidence must name a command that maintainers can run.
- Source-controlled metadata is ownership evidence, not proof that a generator
  just ran.
- Workflow artifact uploads must be fail-closed in the exact selected upload
  block.

## Day 9 Handoff

Day 9 should complete the remaining gate templates for:

- additional bounded external comparison family;
- public header coherence batch 3;
- large review-surface reduction;
- final validation, claim calibration, and closeout.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Each selected Day 8 target has a pass/fail definition | Complete | Gates 1-5 include pass and fail definitions. |
| Each gate names owner files, validation commands, and claim boundaries | Complete | Gate tables include owners, commands, claim boundary, and protected non-claims. |
| Gates prevent broad claims from adjacent evidence | Complete | Cross-gate review checks and protected non-claims reject package, ABI, platform, report, and state-of-the-art overclaims. |
