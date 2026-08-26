# Sprint 180 Day 2: Current Package Surface Audit

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Inventory the current package metadata, install/export proof, downstream
consumer proof, version behavior, cleanup behavior, package-manager
non-claims, and guard behavior before Sprint 180 evaluates provider
candidates. Day 2 records the maintained static-first package surface as the
baseline; it does not select a provider or add provider recipe files.

## Source Package Metadata

| Surface | Owner file | Day 2 finding |
| --- | --- | --- |
| Make install metadata | `Makefile` | `make install` installs the static archive, public headers, generated `sparse_version.h`, and a generated `sparse.pc`; `make uninstall` removes those installed files. |
| CMake install/export metadata | `CMakeLists.txt` | CMake installs the explicit static target archive, public headers, generated version header, `SparseTargets.cmake`, exact-version package files, and generated `sparse.pc`. |
| pkg-config template | `sparse.pc.in` | The template describes "Static archive package metadata for sparse linear algebra" with `Cflags` and `Libs` for the maintained archive link surface. |
| CMake package config template | `cmake/SparseConfig.cmake.in` | The template includes installed `SparseTargets.cmake` and checks required components without provider wording. |
| Version source | `VERSION`, `include/sparse_version.h.in`, generated install headers | Version metadata flows into generated headers, CMake package version files, and `sparse.pc`. |

## Install And Downstream Proof Owners

| Proof owner | Current proof |
| --- | --- |
| `tests/test_install.sh` | Runs Make install into a temporary prefix, checks installed static archive and header count, rejects shared-library artifacts, validates `sparse.pc` presence, validates `pkg-config --exists`, exact version, prefix/libdir/includedir, cflags, libs, static libs, no `Libs.private`, static archive description, no unsupported package/ABI wording, downstream compile/link/run for a small consumer and the maintained example, then runs Make uninstall and checks cleanup. |
| `tests/test_cmake_install.sh` | Runs CMake configure/build/install into a temporary prefix, checks installed static archive, headers, CMake package files, `SparseTargets.cmake`, `sparse.pc`, static imported target metadata, install-prefix include/archive metadata, absence of shared/loader/static-shared selector metadata, absence of source/build path leaks, `find_package(Sparse)` configure/build/run, exact-version configure/build/run, mismatched-version rejection, and `pkg-config --modversion`. |
| `scripts/static_package_deferral_check.sh` | Guards the Sprint 170 static-first package decision, rejects `BUILD_SHARED_LIBS=ON`, checks static target/install metadata, and protects shared-library/dynamic ABI deferrals. |
| `scripts/package_manager_deferral_check.sh` | Guards Sprint 171 package-manager deferral, provider recipe absence, provider-neutral package metadata, and public package-manager non-claims. |
| `scripts/normalize_report_index.py` | Emits source-controlled package proof-owner rows for the install scripts, metadata templates, and deferral guards. |

## Current Package-Manager Claim Surface

| File | Current wording role |
| --- | --- |
| README | Routes users to source install via Make or CMake, describes installed metadata as static-archive scoped, and states package-manager support is not currently provided. |
| INSTALL | Owns the support split: Unix Make install/`pkg-config`, CMake install/export, and package-manager deferral. It explicitly keeps vcpkg, Homebrew, Conan, pkgsrc, distro/system packages, provider registries, taps, recipes, and binary packages unsupported. |
| `docs/maintainer_guide.md` | Names package/install proof owners and says to run `scripts/package_manager_deferral_check.sh` when changing package-manager wording, provider recipe files, package metadata templates, or provider support claims. |
| `docs/api_reference.md` | Routes install support to `INSTALL.md` and keeps package-manager distribution outside API-reference scope. |
| `.github/workflows/ci.yml` | Linux package-contract lane is intended to run Make install/`pkg-config`, CMake install/export, and static-first deferral proof; comments keep package-manager support out of scope. |
| `.github/workflows/macos-ci.yml` | macOS install jobs run Make install/`pkg-config` and CMake install/export proof; comments keep package-manager support out of scope. |
| `.github/workflows/windows-ci.yml` | Windows install/downstream lane is CMake-first with metadata-only `sparse.pc` inspection; comments reject Windows package-manager support and Windows `pkg-config` execution parity. |

## Unsupported Provider Failure Behavior

| Unsupported path | Current fail-closed behavior |
| --- | --- |
| Provider recipe artifacts | `scripts/package_manager_deferral_check.sh` fails if unselected provider recipe artifacts such as `vcpkg.json`, `vcpkg-configuration.json`, `portfile.cmake`, `conanfile.py`, `conanfile.txt`, `Formula/`, `pkgsrc/`, Debian files, or spec files appear outside excluded trees. |
| Provider wording in package metadata | The package-manager guard fails if `sparse.pc.in` or `cmake/SparseConfig.cmake.in` gains vcpkg, Homebrew, Conan, pkgsrc, distro, registry-ready, binary-package, or package-manager support wording. |
| Public non-claim drift | The package-manager guard fails if README, INSTALL, or maintainer guide no longer keep package-manager support scoped as a non-claim. |
| Shared-library or ABI widening | `scripts/static_package_deferral_check.sh` fails if static-first metadata, shared-library deferral wording, or `BUILD_SHARED_LIBS=ON` rejection weakens. |

## Implementation Candidates For Later Days

| Candidate owner | Why it matters for provider feasibility |
| --- | --- |
| `tests/test_install.sh` | Provides the strongest local pattern for isolated prefix install, downstream compile/link/run, version query, and cleanup. |
| `tests/test_cmake_install.sh` | Provides the strongest local pattern for CMake install/export, `find_package`, exact-version behavior, and static imported-target metadata. |
| `sparse.pc.in` | Must stay provider-neutral unless the product decision and guard explicitly allow selected provider metadata. |
| `cmake/SparseConfig.cmake.in` | Must stay provider-neutral and static imported-target scoped unless a provider proof requires bounded changes. |
| `scripts/package_manager_deferral_check.sh` | Will need replacement or update if Sprint 180 selects a provider proof path; otherwise it should be strengthened for a renewed deferral. |
| `scripts/static_package_deferral_check.sh` | Should remain unchanged unless provider proof exposes a static-first package-contract gap. |
| README, INSTALL, maintainer guide | Must be updated after, not before, the product decision so source install, CMake/pkg-config install, and provider status remain distinct. |

## Day 2 Audit Notes

- The maintained proof surface already covers source install, downstream
  compile/link/run behavior, exact version behavior, and cleanup for Make and
  CMake install paths.
- The existing package-manager guard proves deferral, not provider readiness.
- No package metadata template currently contains provider support wording.
- Source-controlled package proof-owner rows are useful ownership metadata,
  but they are not a substitute for running install validation commands.
- The current workflow files contain package-proof comments and jobs that are
  relevant to provider feasibility. The Day 2 audit records them as current
  owner surfaces and does not change workflow YAML.

## Day 2 Deliverables

- package metadata inventory
- install/export and downstream proof inventory
- package-manager claim and non-claim inventory
- current guard and failure-behavior notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day2-package-surface-audit.md`

## Validation

Day 2 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, or public user-facing docs were modified, so full C quality
gates and install validation scripts are not required for this day.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every relevant package surface is accounted for before provider evaluation. | Complete | Source package metadata, proof owner, guard, workflow, and normalized package row sections above. |
| Current supported and unsupported package-manager wording is explicit. | Complete | Claim surface and unsupported-provider failure behavior sections above. |
| Implementation candidates are tied to existing proof owners. | Complete | Implementation candidate table maps later work to exact owner files. |
