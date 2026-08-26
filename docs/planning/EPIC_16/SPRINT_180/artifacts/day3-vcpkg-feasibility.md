# Sprint 180 Day 3: vcpkg Feasibility Audit

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Evaluate vcpkg as a static-first package-manager provider proof candidate
before the Day 7 decision matrix. Day 3 compares current vcpkg port
requirements against the project package surface, records prototype blockers,
defines proof requirements, and keeps provider support unclaimed.

## External vcpkg References

Official vcpkg documentation consulted on 2026-08-25:

| Topic | Source |
| --- | --- |
| Port structure | <https://learn.microsoft.com/en-us/vcpkg/concepts/ports> |
| Packaging tutorial | <https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging> |
| Overlay ports | <https://learn.microsoft.com/en-us/vcpkg/concepts/overlay-ports> |
| Triplet variables | <https://learn.microsoft.com/en-us/vcpkg/users/triplets> |
| `vcpkg.json` metadata | <https://learn.microsoft.com/en-us/vcpkg/reference/vcpkg-json> |
| Port usage documentation | <https://learn.microsoft.com/en-us/vcpkg/maintainers/handling-usage-files> |

Relevant current expectations from those sources:

- a vcpkg port is a versioned recipe with package metadata and build/install
  instructions;
- a port requires `portfile.cmake` and package metadata such as `vcpkg.json`;
- new ports should use `vcpkg.json` rather than deprecated `CONTROL` files;
- overlay ports are valid local filesystem ports and can be selected with
  `--overlay-ports`;
- `VCPKG_LIBRARY_LINKAGE` and `VCPKG_CRT_LINKAGE` are triplet-controlled
  linkage preferences;
- vcpkg package metadata includes fields for name, version, description,
  homepage, license, dependencies, features, and supports expressions;
- vcpkg can inspect or generate usage information after a port install.

## Local Baseline

| Check | Day 3 result |
| --- | --- |
| `vcpkg` on PATH | Not present in this environment. |
| Existing vcpkg artifacts | No `vcpkg.json`, `vcpkg-configuration.json`, `portfile.cmake`, or active `ports/` provider tree outside planning. |
| Version source | `VERSION` currently reports `2.2.0`. |
| Build system fit | CMake project has an explicit static library target, exact-version package metadata, install/export rules, generated `sparse.pc`, and `BUILD_SHARED_LIBS=ON` rejection. |
| License metadata | No standalone `LICENSE`, `COPYING`, or `NOTICE` file was found. README only states research and educational purpose. |
| Current guard posture | `scripts/package_manager_deferral_check.sh` intentionally fails if vcpkg recipe artifacts appear before the provider decision and guard update. |

## vcpkg Fit Assessment

| Criterion | Assessment |
| --- | --- |
| Static-first fit | Strong. The project already builds an explicit static CMake target and rejects shared-library configuration. A vcpkg port could call `vcpkg_check_linkage(ONLY_STATIC_LIBRARY)` and then use the existing CMake install/export surface. |
| CI feasibility | Medium. Hosted Linux and Windows runners can plausibly bootstrap vcpkg, but Day 3 cannot prove runtime cost or tool availability locally because `vcpkg` is not installed here. A CI lane would need bounded setup, caching policy, and claim-safe skip/fail behavior. |
| Recipe complexity | Medium. A local overlay prototype likely needs one overlay directory with `vcpkg.json`, `portfile.cmake`, possible usage handling, and proof scripting. Registry readiness would add source archive, checksum, version database, policy, and review overhead. |
| User value | High. vcpkg has strong C/C++ and Windows adoption value, and the existing CMake package surface gives users a natural `find_package(Sparse)` consumption path. |
| Proof completeness | Medium. Existing install tests prove the underlying CMake package shape, but no provider install, provider downstream consumer, provider version query, vcpkg usage output, or vcpkg cleanup path exists yet. |
| Maintenance cost | Medium to high. Local overlay maintenance is manageable; registry-ready maintenance is higher because source/checksum, license, version, dependency, supports, and update policy must remain current. |
| Claim risk | High unless bounded. Local overlay proof could be mistaken for official registry support, binary package availability, provider-managed upgrades, broad Windows support, shared-library support, or dynamic ABI compatibility. |

## Static-Only And Feature Assessment

| Area | vcpkg implication |
| --- | --- |
| Static library | Feasible through existing CMake static target plus a portfile static-linkage check. |
| Shared-library requests | Must fail or remain unsupported; the project already rejects `BUILD_SHARED_LIBS=ON`, and the vcpkg proof must not weaken that boundary. |
| OpenMP option | Current `SPARSE_OPENMP` is optional and off by default. A first vcpkg proof should keep it off unless dependency and platform behavior are explicitly selected. |
| Mutex option | Current `SPARSE_MUTEX` is optional and off by default. A first vcpkg proof should keep it off unless thread dependency/link flags are explicitly selected. |
| Dependencies | Baseline build depends on CMake, a C compiler, and the system math library; optional OpenMP and mutex behavior should not be enabled implicitly. |
| Versioning | Port metadata can use `VERSION=2.2.0`, but registry-ready versioning would need a defined source/archive reference and update policy. |
| License metadata | Blocked for provider readiness until the project has accurate provider-compatible license metadata and an installable copyright source. |

## Prototype Shape

A local overlay prototype would likely include:

| File or command | Role |
| --- | --- |
| `ports/sparse/vcpkg.json` or equivalent overlay path | Port metadata: name, version, description, homepage, license, dependencies, supports, and features. |
| `ports/sparse/portfile.cmake` | Acquire source, enforce static linkage, configure/build/install with CMake, fix up package config files, remove duplicate debug includes if needed, and install copyright metadata. |
| optional `usage` handling | Confirm generated or custom usage shows `find_package(Sparse)` and `Sparse::sparse_lu_ortho` without broad provider claims. |
| proof script | Bootstrap or locate vcpkg, install the overlay into an isolated root/triplet, build a downstream consumer, query version or usage metadata, and clean generated state. |

Day 3 does not add these files because Sprint 180 has not selected vcpkg as
the product decision and the current deferral guard intentionally rejects
provider recipe artifacts.

## Required Proof Before vcpkg Support

| Evidence | Requirement |
| --- | --- |
| Tool availability | Script handles missing `vcpkg` clearly without implying support. |
| Isolated install | vcpkg installs the overlay into an isolated package tree or root. |
| Static-only package | Installed files and vcpkg metadata contain no shared-library, DLL, SONAME, install-name, runtime-loader, or static/shared selector support claim. |
| Downstream consumer | A vcpkg-installed consumer configures, builds, links, runs, and reports expected solver/version behavior. |
| Version behavior | vcpkg metadata and installed package version match `VERSION`; exact-version or usage-query behavior is recorded where feasible. |
| Cleanup | Temporary overlay roots, build trees, downloads, packages, archives, and installed trees are cleaned or explicitly ignored. |
| Docs | README, INSTALL, and maintainer guide distinguish local overlay proof from registry readiness and broad package-manager support. |
| Guard update | `scripts/package_manager_deferral_check.sh` is replaced or narrowed so selected vcpkg prototype files are allowed only in the approved location while other providers remain guarded. |

## Blockers

| Blocker | Impact |
| --- | --- |
| No local `vcpkg` executable on PATH | Day 3 cannot run an install proof locally. |
| No selected provider decision yet | Adding vcpkg recipe files today would violate the current deferral guard and Sprint 180 sequencing. |
| No standalone license file | vcpkg provider metadata and copyright installation cannot be completed confidently. |
| No source archive/checksum policy | Registry-ready or reproducible archive-based proof is not defined. |
| No vcpkg-specific downstream proof | Existing CMake install tests are useful prerequisites but do not prove vcpkg behavior. |
| No provider guard update | Current guard correctly rejects vcpkg artifacts until the product decision changes. |

## Claim Risks

- A local overlay can be misread as official vcpkg registry support.
- A successful vcpkg install can be misread as binary package availability or
  provider-managed upgrade behavior.
- Windows vcpkg value can be misread as broad Windows package-manager support
  or Windows Makefile/`pkg-config` parity.
- vcpkg triplet linkage can be misread as project support for shared builds,
  dynamic ABI compatibility, or runtime-loader behavior.
- Usage output can promote `find_package` convenience without preserving
  static-first and local-overlay boundaries unless checked.

## Day 3 Decision

vcpkg remains eligible for the Sprint 180 decision matrix as a local overlay
proof candidate. It should not be described as registry-ready, accepted
upstream, binary-package-backed, or broadly supported.

If vcpkg is selected on Day 8, the first implementation path should be a
source-controlled local overlay prototype plus proof script, not public
registry support.

## Day 3 Deliverables

- vcpkg fit assessment
- vcpkg recipe complexity notes
- vcpkg local and CI proof requirements
- vcpkg claim-risk notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day3-vcpkg-feasibility.md`

## Validation

Day 3 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider recipe, or public user-facing docs were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| vcpkg is evaluated against the shared provider criteria. | Complete | Fit assessment and static-only/feature sections above. |
| Prototype blockers and required proof are concrete. | Complete | Prototype shape, required proof, and blockers sections above. |
| vcpkg remains eligible or is rejected with evidence. | Complete | Day 3 decision keeps vcpkg eligible for local overlay proof only. |
