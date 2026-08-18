# Sprint 165 Day 1 Sprint Intake And Package Surface Inventory

## Purpose

Day 1 establishes the Sprint 165 package-boundary hardening scope before any
package metadata, install script, workflow, or public documentation edits
occur. The goal is to inventory current static-first package surfaces, record
the source-plan path mismatch, and carry forward package, ABI, platform, and
performance non-claim handoffs from earlier sprints.

## Source Plan

The current authoritative Sprint 165 source is
`docs/planning/EPIC_14/PROJECT_PLAN.md`, section "Sprint 165: Static-First
Package Boundary Hardening". The prompt referenced the older Epic 12
project-plan path, which does not contain the current Sprint 165 section.

## Handoff Inputs Reviewed

| Input | Relevant Sprint 165 Rule |
| --- | --- |
| Sprint 162 Windows package decision | Windows package support remains CMake-first; Windows Makefile install/uninstall parity and Windows `pkg-config` execution parity remain unsupported. |
| Sprint 163 performance publication handoff | Benchmark, sentinel, hosted-report, and methodology rows cannot be reused as package, ABI, runtime-loader, shared-library, or package-manager proof. |
| Sprint 164 public-header/API handoff | Public-header cleanup and API docs must preserve static-first package and ABI non-claim boundaries. |
| `INSTALL.md` maintained install contract | Operational package commands, downstream consumer guidance, support split, and install validation belong here. |
| `docs/maintainer_guide.md` package/ABI contract | Maintainer proof interpretation and package/ABI policy live here. |
| `README.md` installation summary | README remains the short public front door and routes detailed package interpretation to `INSTALL.md`. |

## Current Package Surface Inventory

| Surface | Current Role | Day 1 Finding |
| --- | --- | --- |
| `CMakeLists.txt` | CMake build, static library target, install/export, generated version header, package config, and `BUILD_SHARED_LIBS=ON` rejection. | Primary CMake owner for static-first package behavior and shared-library deferral. |
| `cmake/SparseConfig.cmake.in` | Installed CMake package config template. | Minimal config imports `SparseTargets.cmake`; audit later for accidental shared/dynamic wording through generated outputs. |
| `sparse.pc.in` | Installed pkg-config metadata template. | Description is explicitly static archive scoped; current template emits `Cflags` and `Libs` from installed prefix variables. |
| `Makefile` install/uninstall targets | Unix-side install of static archive, public headers, generated version header, and `sparse.pc`; uninstall cleanup. | Primary Make install shape owner; current behavior is static archive plus metadata, not shared library. |
| `tests/test_install.sh` | Make install/pkg-config proof. | Checks static library, absence of shared artifacts, header count, `sparse.pc` fields, link flags, downstream consumers, exact version, and uninstall cleanup. |
| `tests/test_cmake_install.sh` | CMake install/export proof. | Checks CMake package files, imported target metadata, exact/mismatched version behavior, installed consumer build/run, and package metadata boundaries. |
| `scripts/static_package_deferral_check.sh` | Static-first deferral guard. | Expected owner for fail-closed `BUILD_SHARED_LIBS=ON` behavior and absence of unsupported shared ABI metadata/selectors. |
| `examples/cmake_example/` | Maintained downstream CMake consumer. | Useful proof fixture for installed `find_package(Sparse)` behavior. |
| `.github/workflows/ci.yml` | Linux reviewed source of truth and package lane. | Hosted Linux package-contract validation owner. |
| `.github/workflows/macos-ci.yml` | macOS install/pkg-config and CMake install/export package lanes. | Hosted macOS package validation owner; not broad macOS package-manager proof. |
| `.github/workflows/windows-ci.yml` | Windows CMake-first package validation. | Checks static `.lib`, CMake metadata, metadata-only `sparse.pc`, generated and maintained CMake consumers; not Windows Make/pkg-config parity. |
| `README.md` | Public install summary and support boundaries. | Current text routes package detail to `INSTALL.md` and names static-first, shared-library, ABI, and Windows parity boundaries. |
| `INSTALL.md` | Operational package guide. | Primary user-facing package contract and validation command owner. |
| `docs/maintainer_guide.md` | Maintainer package/ABI policy and proof interpretation. | Primary policy owner for source-controlled proof rows, local run evidence, and platform/package non-claims. |

## Initial Evidence Boundaries

Sprint 165 can support:

- stronger validation and documentation around the maintained static archive
  package surface;
- clearer `BUILD_SHARED_LIBS=ON` fail-closed behavior;
- stricter checks against accidental shared-library metadata;
- clearer separation between package version metadata and dynamic ABI support;
- refreshed installed downstream consumer proof for static archive behavior;
- better alignment between README, INSTALL, maintainer docs, CMake metadata,
  pkg-config metadata, examples, and CI package lanes.

Sprint 165 cannot by itself support:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- Linux SONAME, macOS install-name/RPATH, or Windows DLL/import-library
  policies;
- package-manager distribution;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad platform package parity;
- performance, backend superiority, or state-of-the-art claims.

## Initial Risk Register

| Risk | Control |
| --- | --- |
| Package docs imply ABI stability because exact-version metadata exists. | Keep exact-version wording tied to package metadata only; explicitly preserve dynamic ABI non-claim. |
| Shared-library requests become warning-only or silently build static output. | Keep `BUILD_SHARED_LIBS=ON` fail-closed and validate the rejection wording. |
| Installed CMake metadata implies shared imported targets or loader behavior. | Inspect generated/install package metadata for shared imported metadata, loader terms, and static/shared selectors. |
| `sparse.pc` grows unsupported private dependencies or package-manager language. | Keep `sparse.pc` static archive scoped and validate absent unsupported wording. |
| Windows metadata inspection is misread as `pkg-config` execution support. | Keep Windows proof explicitly CMake-first and metadata-only for `sparse.pc`. |
| Linux/macOS package CI is overstated as package-manager or broad platform proof. | Tie CI lanes to install/export proof only, not package-manager distribution or broad platform parity. |
| Performance/report evidence is cited as package or ABI evidence. | Keep Sprint 163 methodology artifacts outside the package proof chain. |

## Day 2 Handoff

Day 2 should perform the package metadata audit using this focused surface set:

- `CMakeLists.txt`
- `cmake/SparseConfig.cmake.in`
- `sparse.pc.in`
- `Makefile` install/uninstall section
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `scripts/static_package_deferral_check.sh`
- `examples/cmake_example/`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

The audit should classify each finding as one of:

- supported static archive contract;
- unsupported wording or metadata;
- stale historical wording;
- validation gap;
- deferred product decision.

## Validation Notes

Day 1 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 1.

## Completion Check

- Sprint 165 scope is tied to the Epic 14 project plan.
- Package metadata and validation owners are identified.
- Shared-library, ABI, runtime-loader, package-manager, and platform parity
  non-claims are recorded before package-boundary edits begin.
