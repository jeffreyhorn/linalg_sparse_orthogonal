# Sprint 170 Day 14: Sprint Closeout And Sprint 171 Handoff

## Purpose

Day 14 closes Sprint 170 by reconciling the sprint deliverables against
project-plan items 170.1 through 170.6, confirming that the shared-library ABI
product decision is source-controlled and guarded, recording final validation,
checking staged-output hygiene, and preparing a handoff for Sprint 171
package-manager readiness or deferral work.

## Product Decision

Sprint 170 selected a static-first-only package and ABI posture.

Current releases may claim maintained static archive build/install behavior,
Unix Make install/`pkg-config` proof, Unix CMake install/export proof, Linux
and macOS static-first package CI lanes, Windows CMake install/downstream
validation for the maintained static-first package surface, generated package
version metadata, and exact CMake package-version compatibility.

Current releases must not claim shared-library packaging, dynamic ABI
compatibility, stable dynamic symbols, runtime-loader behavior, Linux SONAME
support, macOS install-name/RPATH support, Windows DLL/import-library support,
static/shared package selectors, package-manager distribution, Windows Makefile
install parity, Windows `pkg-config` command execution parity, broad platform
parity, or state-of-the-art package/ABI status.

Canonical decision record:

`docs/planning/EPIC_15/SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md`

## Project-Plan Reconciliation

| Item | Name | Status | Evidence |
| --- | --- | --- | --- |
| 170.1 | ABI Surface Audit | Complete | Days 1-4 audited prior ABI evidence, public headers/layouts, lifecycle ownership, and static archive symbol/visibility readiness. |
| 170.2 | Build-System Feasibility | Complete | Days 5-7 reviewed Make, CMake, package metadata, CI lanes, and claim surfaces. |
| 170.3 | Product Decision Record | Complete | Day 9 accepted static-first-only continuation and listed supported claims, unsupported claims, alternatives, and future shared-library gates. |
| 170.4 | Guard Updates | Complete | Days 10-12 designed and implemented static package deferral guard updates for the decision record, Makefile static archive contract, exact `sparse.pc.in` metadata, and public documentation decision citations. |
| 170.5 | Documentation Alignment | Complete | Day 12 aligned README, INSTALL, and maintainer guide wording with the selected decision and preserved non-claims. |
| 170.6 | Validation | Complete | Day 13 ran integrated install/package, guard, documentation, and diff validation; Day 14 reran final lightweight guard and claim checks. |

## Final Validation Record

Day 14 final lightweight validation:

```sh
bash -n scripts/static_package_deferral_check.sh
bash scripts/static_package_deferral_check.sh
rg -n "Shared-library packaging|BUILD_SHARED_LIBS|dynamic ABI|static-first|Sprint 170|pkg-config execution parity|package-manager|state-of-the-art" README.md INSTALL.md docs/maintainer_guide.md
git diff --check
```

Results:

- shell syntax check passed;
- static package deferral guard passed;
- targeted documentation claim scan showed expected static-first claims and
  explicit non-claims;
- diff hygiene passed.

Day 13 integrated install/package validation remains the final local installed
consumer proof for this sprint:

- `bash tests/test_install.sh` passed with 23 passes and 0 failures;
- `bash tests/test_cmake_install.sh` passed with 27 passes, 0 failures, and
  0 skips.

## Generated-Output Staging Check

`git status --short --branch` showed only source, documentation, and Sprint 170
planning changes. No generated build, report, cache, install-prefix, or
temporary validation artifacts were listed for staging.

## Sprint 171 Handoff

Sprint 171 can start from this package/ABI boundary:

- Treat the static archive package surface as the only maintained package
  product.
- Keep `BUILD_SHARED_LIBS=ON` rejected unless a future shared-library product
  plan selects all required ABI, visibility, loader, and consumer-proof gates.
- Treat package-manager distribution as unsupported until provider-specific
  install, dependency, metadata, license, version, and downstream-consumer
  proof exists.
- Do not infer Windows Makefile parity or Windows `pkg-config` execution
  parity from the Windows CMake install/downstream lane.
- Before any package-manager readiness work, decide whether the target is
  documentation-only deferral, Unix package-manager preparation, or a
  provider-specific proof lane.
- Reuse `scripts/static_package_deferral_check.sh`,
  `tests/test_install.sh`, and `tests/test_cmake_install.sh` as the minimum
  package-claim regression stack.

## Day 14 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Final Sprint 170 validation record | Complete | Final lightweight checks and Day 13 installed proofs are recorded. |
| Project-plan item reconciliation | Complete | Items 170.1 through 170.6 are reconciled to evidence artifacts. |
| Generated-output staging check | Complete | No generated outputs are listed for staging. |
| Sprint 171 handoff | Complete | Package-manager readiness or deferral boundaries are listed. |
| Day 14 sprint-closeout artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| The shared-library ABI product decision is source-controlled and enforceable. | Complete | The Day 9 decision record is source-controlled and guarded by `scripts/static_package_deferral_check.sh`. |
| Documentation and guards match the decision. | Complete | README, INSTALL, maintainer guide, and the deferral guard cite or enforce the static-first-only decision. |
| Sprint 171 can begin from a clear package/ABI boundary. | Complete | The handoff names the supported package surface and unsupported package-manager/shared-library claims. |
