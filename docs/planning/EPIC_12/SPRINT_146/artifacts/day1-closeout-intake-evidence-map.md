# Day 1 Closeout Intake Evidence Map

## Scope

Day 1 establishes the Sprint 146 closeout baseline for Epic 12. It does not reopen Sprint 137-145 implementation work unless later validation identifies a concrete failure. Its job is to define what evidence exists, which Sprint 146 day owns each project-plan item, and what wording must remain a non-claim unless final evidence supports promotion.

## Sprint 146 Inputs

- Project-plan section: `docs/planning/EPIC_12/PROJECT_PLAN.md`, Sprint 146, "Epic 12 Final Validation, Claim Recalibration & Closeout".
- Sprint 146 plan: `docs/planning/EPIC_12/SPRINT_146/PLAN.md`.
- Prior sprint evidence: Sprint 137-145 plans, retrospectives, working notes, artifacts, tests, docs, generated-output rules, and CI support-tier records.

## Evidence Families

| Family | Baseline Evidence | Closeout Treatment |
| --- | --- | --- |
| Corpus | Maintained fixture manifests, generator metadata, corpus schema checks, and oracle references from Sprint 138. | Use as the source of truth for numerical corpus coverage. Generated local outputs are reproducibility evidence, not checked-in freshness claims. |
| QR | Sprint 139 QR corpus lane, `qr_rank_deficient_6x4_nullspace_v1`, rank/nullity checks, nullspace residual proof, and solver-selection notes. | Claim bounded QR residual closure for the reviewed fixture only; keep broad QR parity as a residual/non-claim. |
| Partial-SVD | Sprint 140 clustered and repeated spectrum fixture, partial-SVD corpus test, projector and residual checks, convergence-status checks, and fail-closed behavior. | Claim bounded partial-SVD edge-case closure for reviewed fixture families only; keep broad SVD parity as a non-claim. |
| Report | Sprint 141 normalized report family index, row schema, freshness gates, and generated-vs-source-controlled report boundary. | Claim normalized index semantics and stale-row detection. Do not claim generated report freshness from source-controlled artifacts. |
| Runtime/Backend | Sprint 142 typed runtime controls, backend governance docs, sentinel scope, benchmark evidence boundaries, and maintainer notes. | Claim governed backend selection and sentinel boundaries. Do not claim portable performance or backend superiority. |
| Package | Sprint 143 static-first package decision, CMake export, pkg-config metadata, install validation, and ABI wording. | Claim static archive install/package metadata within documented lanes. Do not claim shared-library ABI or package-manager support. |
| Platform | Sprint 144 Linux source-of-truth lane, macOS reviewed static-first install/export proof, Windows reviewed CMake subset, and staged blockers. | Claim each platform only at its documented support tier. Keep Windows Make/pkg-config/install-validation parity out of public claims. |
| Adoption | Sprint 145 README, INSTALL, CMake example, cookbook, solver-selection guide, header cleanup, and claim map. | Claim high-level adoption surfaces only where linked to maintained examples and bounded support docs. |
| Validation | Sprint 137-145 quality gates, CI lane decisions, and Sprint 146 final validation package. | Final claim promotion requires quality evidence plus CI reconciliation or an explicit hosted-only limitation. |

## Sprint 146 Item Ownership

| Item | Project-Plan Estimate | Day-Level Owner |
| --- | ---: | --- |
| Item 1: Final Evidence Inventory | 20h | Days 1-3 |
| Item 2: Full Quality Baseline | 28h | Days 4-5 |
| Item 3: Cross-Platform/CI Reconciliation | 20h | Days 6-7 |
| Item 4: Claim and Non-Claim Audit | 24h | Days 8-9 |
| Item 5: Residual Queue Publication | 24h | Days 10-11 |
| Item 6: Epic 12 Retrospective | 26h | Days 12 and 14 |
| Item 7: Final Project Plan Reconciliation | 24h | Days 13-14 |

## Final Closeout Criteria

1. Every public claim maps to an artifact, test, report row, documented support tier, or reviewed CI lane.
2. Every unsupported claim is downgraded to a non-claim or residual with a promotion gate.
3. The final validation package records local gates, hosted CI reconciliation, platform lane status, and any hosted-only caveats.
4. Residuals include affected surface, current blocker, prerequisite evidence, and a closure gate.
5. Epic 12 retrospective and project-plan reconciliation agree on completed, deferred, and residual work.
6. Any state-of-the-art claim requires direct comparative evidence against appropriate sparse linear algebra baselines.

## Non-Claim Register

| Non-Claim | Reason It Remains Guarded |
| --- | --- |
| Unqualified state-of-the-art sparse linear algebra library | Epic 12 evidence is broad product hardening, not a direct comparative benchmark or feature-parity proof against mature libraries. |
| Broad QR/SVD/partial-SVD parity | Evidence covers reviewed fixtures and bounded residuals, not full solver-family parity. |
| Portable performance superiority | Runtime and benchmark work defines sentinels and governance, not cross-platform performance dominance. |
| Shared-library ABI support | Package decision remains static-first; shared-library ABI is not promoted. |
| Package-manager distribution support | Install metadata exists, but package-manager recipes and support ownership remain out of scope. |
| Windows full parity | Windows support is CMake-first with staged exclusions, not Makefile/pkg-config/full install-validation parity. |
| Generated report freshness from repository rows | Generated reports are reproducible outputs; source-controlled rows only describe metadata and expected freshness rules. |

## Validation Requirements

| Change Type | Gate |
| --- | --- |
| Markdown closeout artifacts | `git diff --check` and trailing-whitespace scan. |
| Source or header edits | `make format && make lint && make test`. |
| Report index edits | Report normalization and freshness checks. |
| Corpus metadata edits | Corpus schema validation and affected corpus tests. |
| Package/install edits | Make install validation, CMake install validation, and downstream consumer proof. |
| CI workflow edits | Syntax inspection and hosted CI reconciliation. |

## Stop Conditions

- A required validation gate fails and cannot be fixed without widening scope.
- Review, CI, or artifact evidence is unavailable but required for claim promotion.
- A public claim depends on generated artifacts that are not reproducible from source.
- Platform or package wording implies support beyond the documented lane.
- State-of-the-art wording appears without direct comparative proof.

## Day 2 Handoff

Day 2 should convert this intake map into the first detailed final evidence inventory, starting with corpus, QR, partial-SVD, report, and solver-selection evidence. It should preserve the Day 1 guardrails and identify any missing artifacts before the full quality baseline begins.
