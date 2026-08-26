# Sprint 180 Day 6: pkgsrc Feasibility Audit

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Evaluate pkgsrc as a static-first package-manager provider proof candidate and
complete the four-provider feasibility audit set before the Day 7 decision
matrix. Day 6 compares pkgsrc package requirements against the project package
surface, records prototype blockers, defines proof requirements, and keeps
pkgsrc support unclaimed.

## External pkgsrc References

Official pkgsrc and NetBSD pkgsrc documentation consulted on 2026-08-25:

| Topic | Source |
| --- | --- |
| pkgsrc overview | <https://www.netbsd.org/docs/pkgsrc/> |
| pkgsrc project overview | <https://www.pkgsrc.org/> |
| Package components | <https://www.netbsd.org/docs/pkgsrc/components.html> |
| Creating a new package | <https://www.netbsd.org/docs/pkgsrc/creating.html> |
| Build process | <https://www.netbsd.org/docs/pkgsrc/build.html> |
| Fixing packages | <https://www.netbsd.org/docs/pkgsrc/fixes.html> |
| Package options | <https://www.netbsd.org/docs/pkgsrc/options.html> |
| buildlink methodology | <https://www.netbsd.org/docs/pkgsrc/buildlink.html> |

Relevant current expectations from those sources:

- pkgsrc is a centralized package-management system for Unix-like systems,
  with source and binary package workflows;
- NetBSD and SmartOS are core pkgsrc environments, while other Unix-like
  systems generally require pkgsrc bootstrap before package building;
- a package normally has package metadata files such as `Makefile`, `distinfo`,
  description/packing-list material, optional patches, and supporting files;
- `Makefile` records package identity, category, upstream source, maintainer,
  homepage, comment, license, work directory, build system, and dependencies;
- `distinfo` records distribution-file and patch checksums and is regenerated
  by pkgsrc tooling;
- package creation commonly starts with `url2pkg`, then dependency, buildlink,
  PLIST, and `pkglint` work;
- package builds proceed through fetch, extract, patch, configure, build, and
  install phases, with provider-specific customization normally limited to
  phase variables or `do-*` hooks;
- dependency handling may use `DEPENDS`, `BUILD_DEPENDS`, `TOOL_DEPENDS`,
  `TEST_DEPENDS`, `USE_TOOLS`, and `buildlink3.mk`;
- package options use the pkgsrc options framework and must not introduce
  unproven feature or ABI claims.

## Local Baseline

| Check | Day 6 result |
| --- | --- |
| `bmake` on PATH | Not present in this environment. |
| `pkg_info` on PATH | Not present in this environment. |
| Existing pkgsrc artifacts | No active pkgsrc package skeleton, `PLIST`, `distinfo`, `DESCR`, `buildlink3.mk`, or `options.mk` files outside planning. |
| Version source | `VERSION` currently reports `2.2.0`. |
| Build system fit | Make and CMake both install a static archive, public headers, generated version header, and static package metadata; CMake also installs exact-version config files and `SparseTargets.cmake`. |
| License metadata | No standalone `LICENSE`, `COPYING`, or `NOTICE` file was found. README only states research and educational purpose. |
| Current guard posture | `scripts/package_manager_deferral_check.sh` intentionally fails if pkgsrc package files appear before the provider decision and guard update. |

## pkgsrc Fit Assessment

| Criterion | Assessment |
| --- | --- |
| Static-first fit | Conditional. pkgsrc can package static libraries, but the package must drive the existing static install surface and avoid implying shared-library, dynamic ABI, runtime-loader, or ABI-stability support. |
| CI feasibility | Low to medium. This environment lacks pkgsrc tools, so Day 6 cannot prove bootstrap, `bmake`, `make package`, `pkg_add`, `pkg_info`, PLIST, or cleanup behavior. Hosted CI would need deterministic bootstrap or a prepared pkgsrc environment. |
| Package complexity | High. A credible prototype needs a pkgsrc package skeleton, metadata, `distinfo`, `PLIST`, possible patches, dependency/buildlink policy, package linting, install/package proof, and bootstrap handling. |
| User value | Medium but narrower for the current project. pkgsrc can reach NetBSD, SmartOS, and cross-Unix users, but the repository currently has stronger direct package-surface evidence for CMake, macOS, and Windows-oriented consumers. |
| Proof completeness | Low today. Existing Make/CMake install tests are useful prerequisites, but no pkgsrc package build, install, version query, downstream compile/link/run, package database query, binary package, or cleanup proof exists. |
| Maintenance cost | High. Upstream archive/checksum updates, pkgsrc tree layout, category choice, PLIST drift, patch maintenance, bootstrap variation, buildlink policy, and platform-specific fixes would require ongoing ownership. |
| Claim risk | High unless bounded. A pkgsrc package can imply NetBSD/SmartOS support, broad Unix portability, binary-package availability, package database integration, upgrade behavior, and pkgsrc-current or pkgsrc-wip acceptance. |

## Package Metadata, Build Phases, Dependencies, And Platform Scope

| Area | pkgsrc implication |
| --- | --- |
| Package identity | A prototype must choose category, package name, version, maintainer, homepage, comment, license, and upstream source policy. |
| Source and checksums | A source archive URL and checksum policy are required before `distinfo` can be meaningful. Local-source shortcuts must be documented as prototype-only. |
| Build driver | The package could use the Make install path or CMake build path, but the selected path must match existing static-first proof and avoid provider-specific ABI expansion. |
| Build phases | Fetch, extract, patch, configure, build, install, package, and test behavior need evidence. Custom `do-*` hooks should be minimized. |
| PLIST | Installed headers, static archive, generated CMake files, and generated `sparse.pc` entries must be represented and checked for drift. |
| Patches | Patches are not expected for the current source layout, but pkgsrc may require packaging-specific fixes or install-path adjustments that would need checksum and maintenance policy. |
| Dependencies | Baseline package has no third-party pkgsrc dependencies. System math, optional OpenMP, mutex, or pthread behavior must not become implicit provider claims. |
| buildlink | A first local proof likely does not need public `buildlink3.mk`; if downstream pkgsrc consumers are promised, buildlink policy and ABI dependency bounds must be designed. |
| Options | OpenMP, mutex, shared-library, or build-system options should remain absent or fixed off unless explicitly selected and proven. |
| Platform scope | A local proof must name the host environment and pkgsrc tree/bootstrap source. It cannot imply NetBSD, SmartOS, macOS, Linux, or broad Unix support without per-platform evidence. |

## Prototype Shape

A local pkgsrc prototype would likely include either a package skeleton outside
the project tree for evaluation or source-controlled provider files only after
the Day 8 product decision and guard update:

| File or command | Role |
| --- | --- |
| pkgsrc package `Makefile` | Defines package identity, source, build system, license, install path, and dependencies. |
| `DESCR` | Provides pkgsrc package description without broad support claims. |
| `PLIST` | Records installed static archive, headers, CMake package files, and `sparse.pc`. |
| `distinfo` | Records source archive and patch checksums. |
| optional `patches/*` | Packaging-specific fixes if the existing install surface is insufficient. |
| optional `buildlink3.mk` | Downstream pkgsrc consumer integration if selected and proven. |
| proof script | Handles missing pkgsrc tools clearly, bootstraps or selects a pkgsrc tree, runs package lint/build/install/query/downstream checks, and cleans generated state. |

Day 6 does not add these files because Sprint 180 has not selected pkgsrc as
the product decision and the current deferral guard intentionally rejects
pkgsrc package artifacts.

## Required Proof Before pkgsrc Support

| Evidence | Requirement |
| --- | --- |
| Tool availability | Script handles missing `bmake`, `pkg_info`, and pkgsrc tree clearly without implying support. |
| Deterministic pkgsrc environment | Proof names the pkgsrc tree, bootstrap source, platform, compiler, and package database location used for the check. |
| Package skeleton | Source-controlled package files exist only after product decision and guard update. |
| Static-only package | Package build and installed files preserve the static archive contract and do not claim shared libraries or dynamic ABI support. |
| Source/checksum policy | `distinfo` is generated from a defined source archive or documented local-source prototype input. |
| PLIST proof | Installed file list is checked against headers, static archive, generated CMake files, and `sparse.pc`. |
| Package build/install | Proof runs package build/install or a documented local equivalent and queries installed package metadata. |
| Downstream consumer | A small consumer compiles, links, runs, and resolves version/package metadata from the installed pkgsrc package. |
| Cleanup | Package database entries, work directories, installed files, bootstrap state, and temporary outputs are removed or isolated. |
| Docs and guard | README, INSTALL, maintainer guide, and package-manager guard distinguish local skeleton proof from pkgsrc-current, pkgsrc-wip, binary package, or broad Unix support. |

## Blockers

| Blocker | Impact |
| --- | --- |
| No local `bmake` or `pkg_info` executable on PATH | Day 6 cannot run pkgsrc package build, install, query, lint, or cleanup checks locally. |
| No selected provider decision yet | Adding pkgsrc package files today would violate current deferral guard and Sprint 180 sequencing. |
| No pkgsrc package skeleton | No package metadata, PLIST, distinfo, patch, package build, package database, or downstream consumer proof exists. |
| No standalone license file | pkgsrc package metadata and any wider distribution claim cannot identify package license confidently. |
| No source archive/checksum policy | `distinfo` cannot be maintained honestly without an immutable source input and checksum workflow. |
| No buildlink/options policy | Downstream pkgsrc integration, ABI dependency bounds, optional features, and static-only behavior are not modeled. |
| No platform proof | NetBSD, SmartOS, macOS, Linux, and other Unix-like environments remain unproven. |
| No guard update | Current guard correctly rejects pkgsrc artifacts until the product decision changes. |

## Claim Risks

- A local pkgsrc skeleton can be misread as pkgsrc-current or pkgsrc-wip
  acceptance.
- A package build on one host can be misread as broad NetBSD, SmartOS, macOS,
  Linux, or Unix support.
- A generated binary package can be misread as hosted binary-package
  availability.
- PLIST and buildlink files can imply stable installed-file and ABI behavior
  beyond the current static-first package contract.
- Provider patches can hide source-install assumptions that are not validated
  by the existing Make and CMake package proofs.
- Package database integration can be misread as upgrade, dependency, or
  system-package compatibility.

## Day 6 Decision

pkgsrc remains technically eligible for the Sprint 180 decision matrix only as
a local package skeleton proof candidate. It should be treated as a
low-priority first-provider candidate because its immediate proof burden is
higher than vcpkg, Homebrew, and Conan in this repository: no local pkgsrc
tools are available, no package skeleton exists, package source/checksum and
license policy are unresolved, and meaningful proof requires bootstrap,
PLIST, package database, platform, and cleanup evidence.

pkgsrc should not be described as pkgsrc-current-ready, pkgsrc-wip-ready,
binary-package-backed, NetBSD-supported, SmartOS-supported, or broadly
Unix-package-manager-supported without additional provider-specific evidence.

## Day 6 Deliverables

- pkgsrc fit assessment
- pkgsrc package complexity notes
- pkgsrc local and CI proof requirements
- pkgsrc claim-risk notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day6-pkgsrc-feasibility.md`

## Validation

Day 6 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider package skeleton, or public user-facing docs were
modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| pkgsrc is evaluated against the shared provider criteria. | Complete | Fit assessment and package metadata/build phases/dependencies sections above. |
| The four-provider feasibility audit is complete. | Complete | Day 3 vcpkg, Day 4 Homebrew, Day 5 Conan, and Day 6 pkgsrc artifacts now exist. |
| pkgsrc remains eligible or is rejected with evidence. | Complete | Day 6 decision keeps pkgsrc eligible only as a local skeleton proof and marks it as a low-priority first-provider candidate. |
