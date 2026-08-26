# Sprint 180 Day 5: Conan Feasibility Audit

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Evaluate Conan as a static-first package-manager provider proof candidate
before the Day 7 decision matrix. Day 5 compares current Conan 2 recipe,
package ID, generator, and package-test expectations against the project
package surface, records prototype blockers, defines proof requirements, and
keeps Conan support unclaimed.

## External Conan References

Official Conan documentation consulted on 2026-08-25:

| Topic | Source |
| --- | --- |
| Creating a package | <https://docs.conan.io/2/tutorial/creating_packages/create_your_first_package.html> |
| Testing packages | <https://docs.conan.io/2/tutorial/creating_packages/test_conan_packages.html> |
| `conan create` | <https://docs.conan.io/2/reference/commands/create.html> |
| CMakeDeps | <https://docs.conan.io/2/reference/tools/cmake/cmakedeps.html> |
| `package_info()` | <https://docs.conan.io/2/reference/conanfile/methods/package_info.html> |
| Conanfile attributes | <https://docs.conan.io/2/reference/conanfile/attributes.html> |
| Settings and options | <https://docs.conan.io/2/reference/binary_model/settings_and_options.html> |

Relevant current expectations from those sources:

- a Conan package normally has a root `conanfile.py` recipe and may include a
  `test_package` project;
- `test_package` validates that consumers can link against and reuse the
  created package;
- `conan create` exports the recipe, computes the graph, builds/packages, runs
  `package_info()`, and then runs the `test_package` flow when present;
- CMake consumers commonly use generated config files through CMakeDeps and
  CMakeToolchain;
- if a package installs its own CMake config files, `package_info()` may be
  reduced for private use, but ConanCenter-style or multi-build-system use
  generally needs explicit package information;
- `package_type` is strongly recommended and includes `static-library`;
- settings and options influence Conan package IDs, so shared/static,
  OpenMP/mutex, compiler, build type, and profile policy must be explicit.

## Local Baseline

| Check | Day 5 result |
| --- | --- |
| `conan` on PATH | Not present in this environment. |
| Existing Conan artifacts | No `conanfile.py`, `conanfile.txt`, or active `test_package` provider tree outside planning. |
| Version source | `VERSION` currently reports `2.2.0`. |
| Build system fit | CMake installs an explicit static target, generated version header, exact-version package files, `SparseTargets.cmake`, and `sparse.pc`; Make also proves source install. |
| License metadata | No standalone `LICENSE`, `COPYING`, or `NOTICE` file was found. README only states research and educational purpose. |
| Current guard posture | `scripts/package_manager_deferral_check.sh` intentionally fails if `conanfile.py` or `conanfile.txt` appears before the provider decision and guard update. |

## Conan Fit Assessment

| Criterion | Assessment |
| --- | --- |
| Static-first fit | Good. A Conan recipe can declare `package_type = "static-library"` and drive the existing CMake install/export path. The recipe must not offer or imply shared-library support unless a separate product decision changes that boundary. |
| CI feasibility | Medium. Conan is not available locally, so Day 5 cannot prove runtime cost or profile behavior. Hosted Linux, macOS, or Windows jobs could install Conan, but the proof must bound profile setup, cache behavior, build time, and claim-safe missing-tool behavior. |
| Recipe complexity | Medium to high. Conan needs a root recipe, settings/options policy, generator/toolchain decisions, `package_info()` or installed-config policy, possible source/export policy, and a `test_package` consumer. |
| User value | Medium to high. Conan is useful for cross-platform C/C++ consumers, especially teams already using Conan profiles and binary caches. It is less directly aligned with the existing macOS/Windows public surface than Homebrew/vcpkg without recipe proof. |
| Proof completeness | Medium. Existing CMake package tests provide a strong prerequisite, but no Conan package creation, package ID, profile, generated config, test_package, version, cache, cleanup, or missing-tool proof exists yet. |
| Maintenance cost | High. Recipe revisions, profile compatibility, settings/options matrix, binary model, dependency metadata, source/license policy, and possible remote/binary-cache behavior require ongoing ownership. |
| Claim risk | High unless bounded. Conan can imply provider-managed binaries, profile compatibility, remote availability, dependency resolution, upgrade behavior, and cross-platform package support beyond the static source install contract. |

## Package ID, Options, Generators, Dependencies, And Profiles

| Area | Conan implication |
| --- | --- |
| Package type | A first proof should declare or otherwise enforce static-library behavior and avoid shared-library options. |
| Settings | Recipe settings likely include `os`, `arch`, `compiler`, and `build_type`; these affect package IDs and must be recorded for proof interpretation. |
| Options | A `shared` option should either be absent, fixed to `False`, or rejected. Optional OpenMP and mutex behavior should stay off unless selected with explicit package ID policy. |
| Generators | A recipe/test package likely needs CMakeToolchain and CMakeDeps, or a clearly documented choice to consume the installed `SparseConfig.cmake` files directly. |
| `package_info()` | If relying on installed CMake config files only, document the limitation. For broader Conan consumer support, set CMake target/file properties and library metadata explicitly. |
| Dependencies | Baseline package has no third-party Conan dependencies. System math library and optional OpenMP/thread behavior must not become implicit provider claims. |
| Profiles | Local proof must name the Conan profile, compiler, build type, and host/build context; missing or auto-generated profiles cannot be treated as broad support. |
| Versioning | Recipe version can match `VERSION=2.2.0`, but remote or binary support needs source/export and revision policy. |

## Prototype Shape

A local Conan prototype would likely include:

| File or command | Role |
| --- | --- |
| `conanfile.py` | Root recipe with name, version, license, package type, settings, options policy, CMake build/package logic, and package metadata. |
| `test_package/conanfile.py` | Conan consumer that requires the package and drives a small build. |
| `test_package/CMakeLists.txt` | Consumer CMake project using `find_package(Sparse)` or Conan-generated target metadata. |
| `test_package/src/example.c` | Downstream compile/link/run proof against the Conan package. |
| proof script | Handles missing `conan`, creates or selects an isolated profile/cache, runs `conan create`, captures package graph/version/package ID evidence, runs `test_package`, and cleans generated state. |

Day 5 does not add these files because Sprint 180 has not selected Conan as
the product decision and the current deferral guard intentionally rejects
Conan recipe artifacts.

## Required Proof Before Conan Support

| Evidence | Requirement |
| --- | --- |
| Tool availability | Script handles missing `conan` clearly without implying support. |
| Isolated cache/profile | Proof uses a bounded Conan home/cache/profile so generated packages do not leak into source control or user state unexpectedly. |
| Recipe source | Source-controlled recipe exists only after product decision and guard update. |
| Static-only package | Recipe and built package enforce static library behavior and reject shared-library, runtime-loader, dynamic ABI, or static/shared selector claims. |
| CMake consumer | `test_package` configures, builds, links, runs, and reports expected solver/version behavior. |
| Package ID evidence | Proof records settings/options/profile/build type and makes clear which package ID was tested. |
| Version behavior | Recipe version and installed package version match `VERSION`; exact-version or CMake package version behavior is recorded where feasible. |
| Cleanup | Conan cache, build folders, package outputs, test-package builds, and temporary profiles are cleaned or explicitly ignored. |
| Docs and guard | README, INSTALL, maintainer guide, and package-manager guard distinguish local recipe proof from ConanCenter, binary cache, remote, or broad provider support. |

## Blockers

| Blocker | Impact |
| --- | --- |
| No local `conan` executable on PATH | Day 5 cannot run `conan create` or package tests locally. |
| No selected provider decision yet | Adding Conan recipe files today would violate current deferral guard and Sprint 180 sequencing. |
| No Conan recipe or `test_package` | No package ID, profile, CMakeDeps, package_info, downstream consumer, or cleanup proof exists. |
| No standalone license file | Conan metadata and any remote readiness claim cannot identify package license confidently. |
| No package ID/options policy | Shared/static, OpenMP, mutex, compiler, build type, and profile behavior are not modeled. |
| No remote/binary policy | ConanCenter, private remote, binary cache, package revisions, and upgrade behavior are unearned. |
| No guard update | Current guard correctly rejects Conan artifacts until the product decision changes. |

## Claim Risks

- A local recipe can be misread as ConanCenter availability.
- A successful `conan create` can be misread as binary cache or remote
  package support.
- One profile can be misread as broad compiler/platform/profile support.
- Conan package IDs can make shared/static, OpenMP, mutex, and build-type
  behavior look supported without explicit policy.
- Generated CMake config behavior can conflict with the project's installed
  `SparseConfig.cmake` target naming if not tested carefully.
- Conan dependency resolution can be misread as provider-managed upgrade
  compatibility or dependency policy.

## Day 5 Decision

Conan remains eligible for the Sprint 180 decision matrix as a local recipe
proof candidate. It should not be described as ConanCenter-ready,
remote-hosted, binary-cache-backed, or broadly package-manager-supported.

Compared with vcpkg and Homebrew, Conan has higher proof complexity because the
sprint would need to model package ID, profiles, CMake generator behavior, and
`test_package` semantics before making even a local support claim.

## Day 5 Deliverables

- Conan fit assessment
- Conan recipe complexity notes
- Conan local and CI proof requirements
- Conan claim-risk notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day5-conan-feasibility.md`

## Validation

Day 5 changed planning artifacts only. No `.c`, `.h`, package metadata,
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
| Conan is evaluated against the shared provider criteria. | Complete | Fit assessment and package-ID/options/profile sections above. |
| CMake package integration requirements are explicit. | Complete | Generator, `package_info()`, CMake consumer, and prototype sections above. |
| Conan remains eligible or is rejected with evidence. | Complete | Day 5 decision keeps Conan eligible for local recipe proof only. |
