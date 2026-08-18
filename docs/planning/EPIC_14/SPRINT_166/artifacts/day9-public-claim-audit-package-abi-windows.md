# Sprint 166 Day 9: Public Claim Audit Part 2

## Purpose

Day 9 audits package, Windows, shared-library, ABI, runtime-loader, and
static/shared selector wording. The goal is to keep the maintained
install/export story accurate: real static-first package support exists, while
shared-library support, dynamic ABI compatibility, runtime-loader behavior,
package-manager distribution, Windows Makefile parity, and Windows
`pkg-config` command execution parity remain explicit product residuals.

## Audited Surfaces

| Surface | Role |
| --- | --- |
| `README.md` | Public install/package and cross-platform contract wording. |
| `INSTALL.md` | Main install/export and package support contract. |
| `docs/api_reference.md` | Public API reference support/non-claim boundary. |
| `docs/tutorial.md` | User-facing package handoff links. |
| `docs/cookbook.md` | Workflow handoff and non-claim wording. |
| `docs/solver_selection.md` | Solver guidance and package/platform non-claims. |
| `docs/maintainer_guide.md` | Maintainer package/ABI contract and proof-owner surface. |
| `CMakeLists.txt` | Static target, install/export metadata, and shared-library rejection. |
| `sparse.pc.in` | Installed `pkg-config` metadata template. |
| `tests/test_install.sh` | Unix Make install/`pkg-config` proof. |
| `tests/test_cmake_install.sh` | Unix CMake install/export proof. |
| `scripts/static_package_deferral_check.sh` | Static-first deferral guard. |
| `.github/workflows/*.yml` | Hosted package/platform support-tier wording. |

## Scan Terms

The audit scanned for:

- package-manager and distribution wording;
- shared-library and shared library wording;
- `BUILD_SHARED_LIBS`;
- dynamic ABI compatibility and ABI stability;
- runtime-loader and runtime loader wording;
- Linux SONAME, macOS install-name/RPATH, Windows DLL/import-library wording;
- `pkg-config`, Windows `pkg-config`, and Windows Makefile parity wording;
- static/shared selectors;
- `vcpkg`, `conan`, Homebrew, `apt`, `dnf`, and `pacman`.

## Classification Summary

| Wording class | Classification | Result |
| --- | --- | --- |
| Static-first package support | Supported claim | README, INSTALL, CMake, `sparse.pc.in`, package scripts, and CI workflows consistently describe an installed static archive package surface. |
| Unix Make install/`pkg-config` support | Supported claim | Linux and macOS reviewed package lanes run Unix Make install/`pkg-config` proof; `tests/test_install.sh` validates installed files, metadata, downstream compile/link/run, and uninstall cleanup. |
| CMake install/export support | Supported claim | Linux, macOS, and Windows reviewed CMake install/downstream lanes validate the static imported target and installed downstream consumers. |
| Shared-library support | Explicit non-claim | `BUILD_SHARED_LIBS=ON` is rejected; docs and guards keep shared-library packaging deferred. |
| Dynamic ABI compatibility | Explicit non-claim | Current exact-version package metadata is not described as ABI compatibility; dynamic ABI policy remains deferred. |
| Runtime-loader behavior | Explicit non-claim | Loader metadata and runtime-loader validation remain absent and deferred. |
| Package-manager distribution | Explicit non-claim | Package-manager names appear only in dependency-install commands or unsupported-wording guards, not as distribution support claims. |
| Static/shared selectors | Explicit non-claim/guardrail | CMake package and `sparse.pc` metadata intentionally expose no static/shared selector. |
| Windows Makefile parity | Explicit non-claim | Windows remains reviewed CMake-first; no Windows Make install/uninstall parity is claimed. |
| Windows `pkg-config` execution parity | Explicit non-claim | Windows installs and inspects `sparse.pc` metadata but does not execute `pkg-config`. |

## Cleanup Applied

| File | Change | Reason |
| --- | --- | --- |
| `INSTALL.md` | Changed "Windows `pkg-config` parity" to "Windows `pkg-config` execution parity". | Aligns the install guide with README, maintainer guide, workflow, and static deferral guard wording. The supported Windows surface is metadata inspection through CMake install/downstream validation, not command execution parity. |

No broader package documentation changes were needed. Existing package wording
already distinguishes supported static-first install/export proof from
deferred package-manager, shared-library, ABI, runtime-loader, and Windows
parity work.

## Static-First Support Statement

The supported package contract is:

- static archive install/export;
- installed public headers;
- `pkg-config` metadata for Unix Make-style downstream consumers;
- CMake package metadata with a static imported target;
- exact-version package compatibility checks;
- downstream compile/link/run proof for maintained examples and generated
  consumers;
- Linux and macOS reviewed Unix package proof;
- Linux, macOS, and Windows reviewed CMake install/downstream proof.

The package contract does not include:

- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager distribution;
- static/shared package selectors;
- Linux SONAME policy;
- macOS install-name/RPATH policy;
- Windows DLL/import-library behavior;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity.

## Validation

| Check | Result | Notes |
| --- | --- | --- |
| Package/ABI/Windows claim scan | Pass | Hits were supported static-first claims, explicit non-claims, guardrails, dependency install commands, or historical sprint records. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Confirmed shared-library rejection, static install metadata, no shared export/ABI metadata, no package selector, deferred support wording, Windows non-claim wording, and no unselected Windows package execution. |
| Targeted stale/risky package wording scan | Pass | No current public/package/workflow hits for stale Windows `pkg-config` parity wording or unsupported supported-claim phrasing. |
| `git diff --check` | Pass | No whitespace errors reported after artifact creation. |

## Sprint 165 Residual Confirmation

The Sprint 165 residuals remain visible product decisions, not hidden defects:

- shared-library support requires export/import policy, symbol visibility
  policy, platform loader metadata, installed shared consumer proof, and
  runtime-loader validation;
- dynamic ABI compatibility requires a versioning and compatibility policy
  beyond exact-version package metadata;
- package-manager distribution requires separate packaging/channel ownership;
- Windows Makefile install/uninstall parity requires a reviewed Windows
  Makefile package path;
- Windows `pkg-config` command execution parity requires a selected Windows
  provider and downstream proof.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Package and ABI wording remains static-first and bounded. | Complete | Audit classification and deferral guard pass preserve the static archive package contract. |
| Windows support wording matches reviewed hosted evidence. | Complete | Windows is documented as CMake-first with `sparse.pc` metadata inspection only and no Makefile/`pkg-config` execution parity. |
| Residual package decisions are not hidden as implementation gaps. | Complete | Sprint 165 residuals are restated as explicit product decisions. |
