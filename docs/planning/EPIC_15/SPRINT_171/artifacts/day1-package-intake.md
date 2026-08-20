# Sprint 171 Day 1: Sprint Intake And Package Boundary Baseline

## Purpose

Day 1 establishes the Sprint 171 baseline for package-manager readiness work.
The sprint starts after Sprint 170 selected a static-first-only package and
ABI posture. Sprint 171 must choose one package-manager readiness path or
formally document and enforce package-manager deferral without broadening
source-install, shared-library, dynamic ABI, runtime-loader, Windows parity,
or platform claims.

## Source Artifact Note

The request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active
merged Sprint 171 section lives in
`docs/planning/EPIC_15/PROJECT_PLAN.md` under "Sprint 171:
Package-Manager Readiness First Provider".

This sprint will use the Epic 15 project-plan section as the source of truth.

## Sprint Scope

Sprint 171 implements project-plan items 171.1 through 171.6:

| Item | Name | Day 1 Interpretation |
| --- | --- | --- |
| 171.1 | Provider Selection | Select one provider path such as vcpkg, Homebrew, or explicit deferral after candidate inventory. |
| 171.2 | Recipe or Deferral Artifact | Add the selected recipe/proof artifact or formal unsupported-provider decision record. |
| 171.3 | Local Proof Script | Add validation for install, compile, version query, and cleanup when provider support is selected, or deferral enforcement when not selected. |
| 171.4 | Package Claim Guard | Guard source install, CMake/`pkg-config` install, and package-manager support as distinct claims. |
| 171.5 | User Documentation | Add concise provider guidance or explicit package-manager non-claim wording. |
| 171.6 | Verification | Run install validation and provider proof or deferral checks. |

## Sprint 170 Handoff Summary

Sprint 170 closed the shared-library ABI decision with static-first-only
continuation:

- the maintained package product is the static archive package surface;
- Make install/`pkg-config` and CMake install/export are validated source
  install surfaces;
- Linux and macOS carry reviewed static-first package lanes;
- Windows carries reviewed CMake install/downstream validation for the
  maintained static-first package surface;
- Windows `sparse.pc` inspection remains metadata-only;
- shared-library builds, dynamic ABI compatibility, runtime-loader behavior,
  package-manager distribution, Windows Makefile parity, and Windows
  `pkg-config` execution parity remain non-claims.

Sprint 171 must not convert that source-install evidence into a
package-manager claim without provider-specific proof.

## Package-Manager Claim Boundary

Package-manager readiness is distinct from:

- building from source with Make;
- installing from source with Make;
- discovering an installed static package with `pkg-config`;
- installing/exporting from source with CMake;
- discovering an installed static package with `find_package(Sparse)`;
- Windows CMake install/downstream validation;
- static archive package metadata.

A package-manager support claim requires its own selected provider,
source-controlled recipe or decision artifact, validation path, cleanup
behavior, user documentation, and guard coverage.

## Retained Non-Claims

Sprint 171 starts with these unsupported claims:

| Area | Retained Non-Claim |
| --- | --- |
| Package managers | No vcpkg, Homebrew, Conan, distro package, pkgsrc, or other provider support claim. |
| Provider behavior | No provider-managed dependency resolution, version compatibility, binary package, license policy, checksum policy, or source archive policy claim. |
| Windows package parity | No Windows Makefile install parity and no Windows `pkg-config` execution parity. |
| Shared libraries | No shared-library build/install support and no static/shared selector support. |
| ABI | No dynamic ABI compatibility or stable dynamic symbol-list claim. |
| Runtime loaders | No Linux SONAME, macOS install-name/RPATH, Windows DLL/import-library, or loader behavior claim. |
| Platform scope | No broad platform parity beyond the reviewed lanes. |
| Product status | No state-of-the-art package, distribution, install, or ABI claim. |

## Stop Conditions

Stop and revise before proceeding if future Sprint 171 work:

1. claims provider support before provider selection and validation exist;
2. treats source install, CMake package discovery, or `pkg-config` metadata as
   package-manager distribution evidence;
3. weakens Sprint 170 static-first or shared-library ABI guards;
4. introduces shared-library, dynamic ABI, runtime-loader, or broad platform
   support claims without a new product decision and proof;
5. treats Windows CMake install/downstream validation as Windows Makefile or
   Windows `pkg-config` execution parity;
6. stages generated package archives, build outputs, install prefixes, or
   provider cache artifacts;
7. changes `.c` or `.h` files without running the full C quality gate.

## Day 1 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Sprint 171 working-notes baseline | Complete | `WORKING_NOTES.md` records scope, assumptions, non-claims, and stop conditions. |
| Artifact directory structure | Complete | `docs/planning/EPIC_15/SPRINT_171/artifacts/` exists with this Day 1 artifact. |
| Source artifact note | Complete | The Epic 12/Epic 15 path mismatch is recorded. |
| Sprint 170 handoff summary | Complete | Static-first package and package-manager non-claims are carried forward. |
| Package-manager stop conditions | Complete | Conditions are listed here and in working notes. |
| Day 1 package-intake artifact | Complete | This file. |

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
| Sprint 171 scope is tied to the active Epic 15 project plan. | Complete | The source artifact note names the active project-plan section. |
| Source install and package-manager support are clearly separated. | Complete | The package-manager claim boundary separates source install, CMake, `pkg-config`, and provider support. |
| No package-manager support claim is introduced by planning alone. | Complete | All provider paths remain candidates or non-claims until selected and validated. |
