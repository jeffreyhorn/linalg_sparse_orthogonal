# Sprint 170 Day 1: ABI Intake

## Purpose

Establish the Sprint 170 scope, prior evidence baseline, retained non-claims,
and stop conditions before auditing shared-library ABI readiness.

## Source Artifact Note

The sprint prompt references `docs/planning/EPIC_12/PROJECT_PLAN.md`, but the
active merged Sprint 170 project-plan section is in
`docs/planning/EPIC_15/PROJECT_PLAN.md` under "Sprint 170: Shared-Library ABI
Product Decision". Sprint 170 proceeds from the Epic 15 section and records
the mismatch for traceability.

## Sprint 170 Scope

Sprint 170 closes the shared-library ABI product question with an explicit
decision and enforceable package metadata behavior.

The Epic 15 project-plan items are:

| Item | Scope | Estimate |
| --- | --- | ---: |
| 170.1 ABI Surface Audit | Audit exported headers, structs, constants, versioning, and lifecycle semantics for ABI readiness. | 34 hours |
| 170.2 Build-System Feasibility | Review Make/CMake behavior for static-only, shared-library rejection, symbol visibility, and install metadata. | 28 hours |
| 170.3 Product Decision Record | Create a decision record choosing static-first-only continuation or a staged shared-library path. | 30 hours |
| 170.4 Guard Updates | Update build/package tests so unsupported shared-library or ABI claims cannot appear accidentally. | 32 hours |
| 170.5 Documentation Alignment | Update README, install docs, package docs, and non-claim tables to match the decision. | 24 hours |
| 170.6 Validation | Run install/package checks and docs sanity checks. | 20 hours |

## Starting Evidence

Sprint 170 starts from a static-first package baseline:

- the project installs a static archive package surface;
- Make install validation checks the static library, installed headers,
  `pkg-config` metadata, downstream compile/link behavior, and uninstall
  behavior;
- CMake install/export validation checks static target exports, config/version
  metadata, downstream consumers, and unsupported shared-artifact absence;
- static package deferral checks reject unsupported packaging or ABI wording;
- README, benchmark docs, and maintainer docs already preserve non-claims for
  shared-library support, dynamic ABI stability, runtime-loader behavior,
  package-manager distribution, and broad platform parity.

## Prior-Sprint Handoff

### Sprint 167

Sprint 167 established the Epic 15 evidence ledger and made unsupported
surfaces explicit. Sprint 170 uses that ledger to keep shared-library,
dynamic ABI, package-manager, runtime-loader, and platform-parity claims out
of public wording until they are backed by a product decision and validation.

### Sprint 169

Sprint 169 hardened selected performance methodology and explicitly handed off
this boundary: selected performance evidence is not package, shared-library,
dynamic ABI, runtime-loader, or package-manager evidence. Sprint 170 must keep
that separation intact while auditing package and ABI surfaces.

## Retained Non-Claims

Sprint 170 starts by retaining these non-claims:

- no shared-library build support;
- no dynamic ABI stability;
- no runtime-loader behavior proof;
- no stable exported-symbol contract;
- no symbol-versioning support;
- no package-manager distribution support;
- no installed shared-consumer support;
- no broad Windows/macOS package parity;
- no Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- no performance or backend evidence from package/install proof;
- no external-library parity from package/install proof.

## Stop Conditions

Stop and revise if Sprint 170 work:

- turns source-compatible headers into an ABI-stability claim;
- treats static archive install proof as shared-library support;
- describes CMake package discovery as package-manager distribution;
- describes `pkg-config` metadata as dynamic loader validation;
- treats Linux install proof as broad platform parity;
- cites Sprint 169 selected performance evidence as package or ABI proof;
- removes static-first guard wording without adding an equivalent enforced
  decision boundary;
- changes `.c` or `.h` files without the full C quality gate;
- leaves README, package docs, or maintainer docs inconsistent with the final
  product decision.

## Day 1 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Sprint 170 working-notes baseline | Complete | Added `WORKING_NOTES.md` with prior evidence, retained non-claims, stop conditions, and assumptions. |
| Artifact directory structure | Complete | Added `docs/planning/EPIC_15/SPRINT_170/artifacts/`. |
| Source artifact note | Complete | Recorded Epic 12 prompt path mismatch and active Epic 15 source. |
| Prior-sprint handoff summary | Complete | Captured Sprint 167 evidence-ledger and Sprint 169 performance-boundary handoffs. |
| ABI/product-decision stop conditions | Complete | Listed claim, guard, generated-output, and validation stop conditions. |
| Day 1 ABI-intake artifact | Complete | This file. |

## Validation

Day 1 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Sprint 170 scope is tied to the active Epic 15 project plan. | Complete | The source artifact note identifies the active Sprint 170 section. |
| Retained package/ABI non-claims are explicit. | Complete | Non-claims are listed in both working notes and this artifact. |
| No shared-library or ABI support claim is introduced by planning. | Complete | The artifact preserves static-first baseline wording and unsupported-claim stop conditions. |
