# Day 1 Package ABI Intake

## Purpose

Day 1 establishes the Sprint 143 package/ABI scope before audit or
implementation work begins. It consumes the Sprint 142 handoff, records the
current package proof surface, maps project-plan items to day owners, and
freezes the initial claim boundaries for the shared-library ABI versus
static-first-only product decision.

## Source Inputs Reviewed

| Input | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 143 section | Authoritative item list, estimates, deliverables, and prerequisites. |
| `docs/planning/EPIC_12/SPRINT_143/PLAN.md` | Day-by-day execution plan and Day 1 completion criteria. |
| `docs/planning/EPIC_12/SPRINT_142/artifacts/day13-claim-closure-and-sprint143-handoff.md` | Concrete package/ABI handoff and stop conditions from runtime/backend governance. |
| `docs/planning/EPIC_12/SPRINT_142/RETROSPECTIVE.md` | Sprint 143 readiness and residual deferred package/platform debt. |
| Current package/build/docs surfaces | Live baseline for the Day 2-4 audits and Day 5 product decision. |

## Inherited Handoff Summary

| Handoff | Intake interpretation |
| --- | --- |
| Runtime/backend public-control boundary | Existing typed controls are caller-facing. Environment, build, and report controls must not become package-stable public ABI by accident. |
| Static-first install baseline | The repo currently maintains a static archive install/export surface through Make, CMake, and `pkg-config`. |
| Sentinel non-claim boundary | `S2` and `S3` timing rows are local advisory runtime/backend context, not package, ABI, platform, or portable performance proof. |
| Shared-library decision gate | Sprint 143 must choose one path: shared-library ABI support with proof, or stricter static-first-only support. |
| Platform tier dependency | Package/ABI work must not imply macOS or Windows reviewed parity; Sprint 144 owns platform promotion. |

## Current Package Surface Map

| Surface | Current owner | Current signal | Day 2-4 audit owner |
| --- | --- | --- | --- |
| Public headers | `include/*.h`, `include/sparse_version.h.in` | 18 source headers plus generated `sparse_version.h` are installed under `include/sparse`. | Day 2 |
| Version source | `VERSION`, Make/CMake generation, `sparse.pc.in`, `SparseConfigVersion.cmake` | Single repo version drives generated header, CMake project version, exact package version, and `pkg-config --modversion`. | Days 2-3 |
| Static library target | `Makefile`, `CMakeLists.txt` | Make builds `build/libsparse_lu_ortho.a`; CMake declares `add_library(sparse_lu_ortho STATIC ...)`. | Days 3-5 |
| Make install/uninstall | `Makefile` | Installs static archive, headers, generated version header, and `sparse.pc`; uninstall removes those artifacts. | Day 3 |
| CMake install/export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Installs archive, headers, config/version files, target export, and `sparse.pc`; exports `Sparse::sparse_lu_ortho`. | Day 3 |
| `pkg-config` metadata | `sparse.pc.in` | Emits include path and `-lsparse_lu_ortho -lm` plus build-option flags; no shared/static selector is present. | Day 3 |
| Static/shared guard | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh`, docs | `BUILD_SHARED_LIBS=ON` is rejected and guard script checks absence of shared ABI/export metadata. | Days 3-5 |
| Install proof | `tests/test_install.sh` | Verifies Make install, no shared artifacts, 19 headers, `pkg-config`, exact version, downstream compile/link/run, and uninstall cleanup. | Day 9 |
| CMake proof | `tests/test_cmake_install.sh` | Verifies CMake install/export, static imported target metadata, exact/mismatched version behavior, `pkg-config` version, and installed consumer. | Day 9 |
| CI package lanes | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Linux has reviewed static-first package contract; macOS/Windows have supplemental package confidence with narrower support-tier wording. | Day 4 and Day 10 |
| Package report rows | `tests/corpus/manifests/report_families.tsv`, normalizer/tests/docs | Source-controlled package proof-owner rows describe ownership; generated install-run proof still requires local command execution. | Day 10 |
| Public docs | `README.md`, `INSTALL.md` | Current wording describes maintained static package surface and shared-library deferral. | Day 11 |
| Maintainer docs | `docs/maintainer_guide.md` | Records static-first proof stack, package lanes, support-tier split, and non-claims. | Day 11 |

## Day-Level Item Ownership

| Project-plan item | Day owner(s) | Intake notes |
| --- | --- | --- |
| Item 1: ABI Feasibility Audit | Days 1-4 | Day 1 intake, Day 2 public header/symbol audit, Day 3 install/export metadata audit, Day 4 platform/loader risk audit. |
| Item 2: Product Decision | Day 5 | Use Day 2-4 evidence to select shared-library ABI support or static-first strengthening. |
| Item 3: Implementation Batch | Days 6-8 | Day 6 design, Day 7 first implementation batch, Day 8 completion/repair batch. |
| Item 4: Downstream Consumer Proof | Day 9 | Strengthen Make, `pkg-config`, CMake, version, unsupported-artifact, and loader proof if applicable. |
| Item 5: CI/Packaging Alignment | Day 10 | Align CI/support-tier wording and package report metadata. |
| Item 6: Documentation Alignment | Day 11 | Align README, INSTALL, package metadata comments, maintainer docs, and non-claims. |
| Item 7: Validation and Closeout | Days 12-14 | Focused validation, full quality gate/claim closure, final closeout, and Sprint 144 handoff. |

## Initial Claim Boundaries

| Claim area | Initial boundary |
| --- | --- |
| Static-first support | Current maintained package surface is static archive install/export through Make, CMake, and `pkg-config`. This remains subject to Day 2-5 audit and product decision. |
| Shared-library ABI | Not claimed. `BUILD_SHARED_LIBS=ON` currently fails at configure time. |
| Dynamic loader behavior | Not claimed. No loader, RPATH/install-name, import-library, or runtime relocation proof has been added. |
| ABI compatibility | Not claimed beyond the current source/header build contract. No symbol-versioning, visibility, or long-term layout policy exists yet. |
| Package-manager support | Not claimed. Local install proof is not distro, Homebrew, vcpkg, conan, MSI, or binary-distribution proof. |
| Platform parity | Not claimed. Linux is the strongest reviewed package lane; macOS/Windows package jobs remain supplemental as currently documented. |
| Runtime/backend sentinels | Not package proof. `S2`/`S3` remain local advisory rows and `S5` remains the local hard wall-check gate. |
| Portable performance | Not claimed by package or sentinel rows. |
| State-of-the-art status | Not affected by package/ABI productization. |

## Initial Stop Conditions

| Stop condition | Reason |
| --- | --- |
| Both shared-library and static-first paths remain active after Day 5. | Sprint 143 must choose one product path to finish completely. |
| Shared-library support would require unsupported platform or package-manager claims. | That would exceed Sprint 143 scope and conflict with Sprint 144 ownership. |
| Package metadata would imply ABI stability before symbol/visibility/versioning proof exists. | Avoid accidental dynamic ABI claims. |
| CI wording would promote macOS or Windows package confidence to reviewed parity. | Platform promotion belongs to Sprint 144. |
| Runtime/backend sentinel timing appears in package proof wording. | Sentinel rows are local runtime/backend context, not package or performance proof. |
| Required package or quality gates fail. | Stop for repair or user input before proceeding. |

## Day 2 Audit Inputs

Day 2 should begin with:

- installed public header list and generated `sparse_version.h` semantics;
- public structs, enums, typedefs, macros, and function declarations;
- ABI-sensitive type-width and struct-layout surfaces;
- symbol-listing commands for the current static archive;
- visibility/export macro absence from `include/`;
- shared-library proof requirements for symbols, visibility, and layout.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 143 project-plan item has a day-level owner. | Complete | Day-level item ownership table maps Items 1-7 to Days 1-14. |
| Sprint 142 runtime/backend sentinel evidence is kept separate from package proof. | Complete | Handoff and claim-boundary tables classify sentinel rows as non-package proof. |
| Shared-library ABI and static-first support boundaries are explicit before audit work begins. | Complete | Initial claim boundaries and stop conditions separate current static-first support from unclaimed shared ABI support. |
