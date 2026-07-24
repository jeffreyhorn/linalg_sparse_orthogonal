# Sprint 133 Day 1 - Package and ABI Intake

## Purpose

Day 1 establishes Sprint 133 scope, artifact structure, source-area intake,
day-level owners, validation lanes, and duplicate fences for package, ABI,
install/export, shared-library, package metadata, and downstream consumer
surfaces.

This is a documentation-only intake artifact. It does not change build-system
behavior, install scripts, public headers, generated package metadata,
downstream consumer tests, platform workflows, or public support claims.

## Project-Plan Item Map

| Item | Sprint 133 project-plan item | Day owner |
| --- | --- | --- |
| 1 | ABI/Product Decision Audit | Days 1-4 |
| 2 | Shared-Library Design Or Deferral | Days 5-6 |
| 3 | Build/Install Contract Batch | Days 7-8 |
| 4 | ABI/Symbol Proof | Days 9-10 |
| 5 | Downstream Consumer Proof | Days 4 and 11-12 |
| 6 | Validation | Days 10 and 13 |
| 7 | Closeout | Days 13-14 |

## Authoritative Inputs

| Input | Role |
| --- | --- |
| Sprint 133 project-plan section | Defines sprint goal, seven work items, deliverables, and 168-hour budget. |
| Sprint 133 `PLAN.md` | Defines 14-day execution path and validation expectations. |
| Sprint 118 residual owner map | Assigns package/ABI decision work to this path and lists required static-first deferral or shared-library proof gates. |
| Sprint 118 package/ABI decision template | Provides decision fields for static/shared/ABI/package-manager support, platform impact, validation, drift checks, and residual handoff. |
| Sprint 112 package artifacts | Prior static-first package surface audit, Make install proof, and CMake install/export proof baseline. |
| Sprint 115 package/platform artifacts | Prior package-manager support decision, Linux/macOS/Windows install-lane decisions, and platform install deferral context. |
| `README.md` | Front-door package summary and current shared-library deferral wording. |
| `INSTALL.md` | Operational static-first install/export and downstream consumer truth. |
| `docs/maintainer_guide.md` | Maintainer package/platform support truth and package validation ownership. |

## Current Package Baseline

| Surface | Current observed baseline | Sprint 133 interpretation |
| --- | --- | --- |
| Product contract | Static-first install/export surface. | Baseline to preserve unless Day 5 explicitly selects shared-library support. |
| Make install | Installs `libsparse_lu_ortho.a`, installed headers, generated `sparse_version.h`, and `sparse.pc`; uninstall removes those files. | Unix-side static archive and `pkg-config` proof owner. |
| CMake install/export | Installs static target export, `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, `sparse.pc`, and headers. | Installed CMake consumer proof owner. |
| `BUILD_SHARED_LIBS` | CMake warns that the maintained package surface remains static-first and continues with a static target. | Deferral/enforcement surface, not shared support. |
| Shared-library artifacts | Install tests check that `.so`, `.so.*`, `.dylib`, and `.dll` artifacts are absent. | Static-first proof; not dynamic-loader proof. |
| Version metadata | Repo `VERSION` feeds generated header, CMake project version, CMake config version, and `sparse.pc` version. | Version consistency proof surface, not ABI compatibility policy by itself. |
| Package managers | No real package-manager recipes or manager-specific install proof are present. | Deferred/non-claim until a future owner adds recipes and proof. |
| Platform install parity | Linux is strongest reviewed truth; macOS has supplemental static-first Make install/`pkg-config`; Windows carries CMake-first consumer story, not separate reviewed install validation. | Platform support wording must stay tiered unless workflows and proof change. |

## Source-Area Intake

| Source area | Current role | Sprint 133 interpretation |
| --- | --- | --- |
| `include/*.h` | Installed public headers. | Audit for source-compatibility and ABI-sensitive declarations before shared support decisions. |
| `include/sparse_version.h.in` and generated `sparse_version.h` | Version macro template and installed generated header. | Version metadata surface; ABI policy still requires explicit decision. |
| `CMakeLists.txt` | Static library target, `BUILD_SHARED_LIBS` handling, install/export rules, generated version header, CMake package files, and pkg-config generation. | Primary build/install contract owner. |
| `cmake/SparseConfig.cmake.in` | Installed CMake package config template. | CMake downstream consumer package entry point. |
| `sparse.pc.in` | pkg-config metadata template. | pkg-config downstream consumer metadata owner. |
| `Makefile` install/uninstall rules | Unix install and uninstall implementation. | Static archive, installed header, generated version, and `sparse.pc` install owner. |
| `tests/test_install.sh` | Make install/uninstall and `pkg-config` compile/link/run proof. | Local static-first Unix package validation lane. |
| `tests/test_cmake_install.sh` | CMake install/export, no-shared-artifact, CMake consumer, exact-version, mismatch-version, and pkg-config version proof. | Local installed CMake consumer validation lane. |
| `examples/cmake_example/` | Maintained installed CMake consumer example. | Downstream consumer proof fixture. |
| `README.md` | User-facing quick package summary. | Must not imply unsupported shared ABI or package-manager support. |
| `INSTALL.md` | Detailed install and validation documentation. | Current package support truth and validation command owner. |
| `docs/maintainer_guide.md` | Maintainer support, platform, package, and validation policy. | Support-tier and non-claim owner. |
| `.github/workflows/` | Reviewed/supplemental platform lanes. | Touch only if platform install support wording or workflow proof changes. |

## Initial Validation Lanes

| Lane | Command | When required |
| --- | --- | --- |
| Docs hygiene | `git diff --check` and Sprint 133 markdown whitespace scan | Every Sprint 133 documentation-only day. |
| Make install proof | `bash tests/test_install.sh` | Make install, uninstall, `sparse.pc`, or pkg-config consumer changes. |
| CMake install/export proof | `bash tests/test_cmake_install.sh` | CMake target, install/export, CMake package, version, or installed CMake consumer changes. |
| pkg-config focused proof | Staged `pkg-config --cflags --libs --modversion sparse` plus compile/link/run consumer | pkg-config metadata changes or consumer-proof strengthening. |
| Static-first deferral proof | No-shared-artifact inspection in install validation | Static-first enforcement or documentation changes. |
| Shared-library proof | Shared artifact inspection, loader/runtime proof, symbol/version proof, and downstream shared-link proof | Only if Day 5 selects shared-library support. |
| Full C quality | `make format && make lint && make test` | Any `.c` or `.h` file change. |
| Platform workflow proof | Workflow-equivalent local or CI evidence plus expected-count/support wording updates | Any platform install workflow or support-tier change. |

## Duplicate Fences and Non-Claims

Sprint 133 must preserve these boundaries until a later artifact explicitly
changes them:

- static archive install proof is not shared-library support;
- installed public headers are not dynamic ABI stability proof;
- version macros and package versions are not ABI compatibility policy;
- absence of shared artifacts is static-first enforcement evidence, not a
  runtime-loader validation lane;
- `pkg-config` and CMake package metadata are downstream consumer metadata, not
  package-manager recipes;
- local Unix install scripts are not macOS or Windows install parity;
- supplemental macOS Make install/`pkg-config` confidence is not full reviewed
  macOS install/export parity;
- Windows CMake-first consumer support is not Windows Makefile parity or a
  separate reviewed install-validation lane;
- package documentation can clarify current truth but must not broaden public
  support without implementation and validation evidence.

## Day 2 Handoff

Day 2 should audit the installed public header and symbol exposure surface. The
first pass should include:

- all headers under `include/` and generated `sparse_version.h`;
- public structs, enums, typedefs, macros, inline helpers, and declarations;
- declarations that expose storage layout or dependency types;
- version and feature macros visible to installed consumers;
- headers included by `tests/test_install.sh` and `examples/cmake_example/`;
- installed headers that are source-compatibility surfaces but not yet dynamic
  ABI contracts.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 133 project-plan item has a day-level owner. | Complete | Project-plan item map and working-notes day-level ownership table map Items 1-7 to Days 1-14. |
| Inherited static-first support truth is preserved before new decisions. | Complete | Current package baseline and duplicate fences preserve static-first install/export truth and defer shared ABI/package-manager/platform parity claims. |
| Package, ABI, install, and downstream consumer surfaces are visible before design or implementation begins. | Complete | Source-area intake and validation-lane tables identify public headers, build/install metadata, package docs, install tests, and downstream consumer proof owners. |
