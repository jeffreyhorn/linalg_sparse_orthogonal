# Day 1: Sprint Intake And Closeout Scope

## Purpose

Establish Sprint 176 scope, source references, artifact layout, inherited Epic
15 evidence, and final closeout boundaries before selecting an
allocation-failure subsystem or editing any claim surface.

## Source Artifact Decision

The Day 1 source of truth is the active Epic 15 project plan:

```text
docs/planning/EPIC_15/PROJECT_PLAN.md
Sprint 176: Allocation-Failure Evidence, Claim Recalibration & Epic Closeout
```

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but that path is
not the active merged Sprint 176 planning source. Sprint 176 records this
mismatch and proceeds from Epic 15.

## Active Sprint 176 Scope

The active project-plan section defines these implementation items:

| Item | Scope | Day 1 interpretation |
| --- | --- | --- |
| 176.1 Subsystem Selection | Choose one allocation-heavy solver or shared subsystem for failure-path proof. | Day 1 does not select the subsystem; it establishes the intake categories and evidence boundaries needed for Days 2-3 selection. |
| 176.2 Failure Harness | Add deterministic allocation-failure or cleanup-path tests for the selected subsystem. | Harness work starts only after candidate inventory and selection. |
| 176.3 Cleanup Invariants | Document ownership and cleanup invariants for the selected subsystem. | Documentation must be tied to the selected proof and must not imply broad coverage. |
| 176.4 Claim Recalibration | Update README, docs indexes, non-claim tables, and evidence ledger to match completed work. | Final claim edits must map to named evidence from Sprints 167-176. |
| 176.5 Epic Retrospective | Create the Epic 15 retrospective with earned claims, non-claims, residuals, and validation evidence. | Retrospective drafting depends on allocation proof and final validation records. |
| 176.6 Final Validation | Run required quality gates and record final status. | Validation scope must follow touched files and include the full C gate if C/header files change. |

## Artifact Layout

Sprint 176 uses the same layout as the recent Epic 15 sprints:

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_15/SPRINT_176/PLAN.md` | Day-by-day plan and completion criteria. |
| `docs/planning/EPIC_15/SPRINT_176/WORKING_NOTES.md` | Running source notes, decisions, assumptions, stop conditions, and daily log. |
| `docs/planning/EPIC_15/SPRINT_176/artifacts/` | Daily evidence artifacts. |
| `docs/planning/EPIC_15/SPRINT_176/RETROSPECTIVE.md` | Sprint retrospective to be created after Day 14. |
| `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md` | Epic closeout retrospective to be created after final sprint evidence exists. |

## Sprint 167-175 Evidence Inventory

| Sprint | Completed evidence | Sprint 176 closeout use |
| --- | --- | --- |
| 167 | Epic 15 baseline, evidence ledger, residual audit, claim gates, and non-claim register. | Provides the original ledger and the `E15-016` allocation/failure-path deferral that Sprint 176 must close narrowly. |
| 168 | Hosted selected performance publication lane for `bench_refactor_csc` with threshold-free metadata and CI artifact path. | Include as earned selected hosted performance evidence, not portable superiority. |
| 169 | Performance methodology hardening for repeat count, warmup, variance, threshold, runner, and selected-lane sentinel policy. | Preserve methodology-bound interpretation in final claims. |
| 170 | Shared-library ABI product decision and static-first-only package posture guarded by `scripts/static_package_deferral_check.sh`. | Retain shared-library, dynamic ABI, and runtime-loader non-claims. |
| 171 | Formal package-manager deferral with executable provider non-claim guard. | Retain provider support non-claims unless future evidence exists. |
| 172 | LU public-header coherence batch and tutorial signature repair with focused LU docs guard. | Include as public-header/API coherence evidence, not behavioral or ABI expansion. |
| 173 | Generated API HTML local-only publication decision and local-only guard. | Retain no-hosted-generated-API-HTML claim boundary. |
| 174 | Additional bounded LU external comparison family and selected comparison freshness integration. | Include as fixture-local comparison evidence, not broad LU or external-library parity. |
| 175 | Linux/macOS selected comparison hosted freshness promotion and workflow guards. | Include as selected hosted report freshness evidence, not broad report freshness or Windows freshness. |

## Closeout Category Map

| Category | Current evidence state | Sprint 176 output expected |
| --- | --- | --- |
| Allocation failure | Deferred broadly; no deterministic selected-subsystem proof is recorded yet. | One selected subsystem has a deterministic allocation-failure or cleanup-path proof. |
| Cleanup invariants | Functional tests and header comments cover normal ownership expectations, but not all failure cleanup paths. | Selected-subsystem ownership and cleanup invariants are documented with validation references. |
| Claims | Public docs now contain many narrow earned claims and explicit non-claims. | README, maintainer guide, report/index docs, and planning records agree with the final evidence. |
| Evidence ledger | Sprint 167 ledger exists; implementation sprints added evidence across performance, ABI/package, API docs, comparison, and platform freshness. | Final ledger posture marks completed, narrowed, deferred, and retained non-claim rows. |
| Documentation | Docs are claim-sensitive but distributed across README, INSTALL, maintainer guide, benchmark docs, corpus docs, API docs, tutorial, and planning artifacts. | Final recalibration keeps evidence source and support tier visible to users and maintainers. |
| Validation | Prior sprints used focused validation and full C gates when C/header files changed. | Final validation record lists required gates, focused checks, skipped checks, and evidence activation boundaries. |
| Residual queue | Prior retrospectives retain broad package, ABI, provider, platform, external-parity, hosted docs, and state-of-the-art residuals. | Epic 15 closeout lists residuals explicitly and avoids hidden product decisions. |

## Retained Non-Claims At Intake

Sprint 176 begins with these retained non-claims:

- no broad allocation-failure cleanup guarantee across all solvers;
- no allocation-failure proof beyond the single selected subsystem;
- no unqualified state-of-the-art sparse linear algebra status;
- no broad external-library ecosystem parity;
- no portable performance superiority;
- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader behavior support;
- no package-manager provider availability;
- no Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- no broad platform parity;
- no Windows generated report freshness;
- no hosted publication for all generated reports;
- no hosted generated API HTML publication;
- no release evidence claim.

## Day 1 Stop Conditions

Future Sprint 176 work should stop and revise if it:

- selects multiple allocation-failure targets instead of one;
- uses one selected failure proof to imply broad OOM cleanup guarantees;
- changes `.c` or `.h` files without the full C quality gate;
- exposes test-only allocation controls as unsupported public API;
- updates public claim wording without a named evidence source;
- broadens package-manager, shared-library, dynamic ABI, runtime-loader,
  performance, platform, external-library, release, or state-of-the-art
  claims;
- treats local generated output as hosted evidence;
- stages generated output under ignored build or docs output directories.

## Day 1 Completion Record

- Sprint 176 scope is tied to the active Epic 15 project-plan section.
- The prompt path mismatch is recorded in the plan, working notes, and this
  artifact.
- The artifact directory exists at
  `docs/planning/EPIC_15/SPRINT_176/artifacts/`.
- Sprint 167-175 retrospectives are inventoried for closeout evidence.
- Closeout categories are identified before subsystem selection begins.
- Retained non-claims are visible before implementation work begins.
