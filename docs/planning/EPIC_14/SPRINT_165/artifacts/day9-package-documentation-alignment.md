# Sprint 165 Day 9 Package Documentation Alignment

## Purpose

Day 9 aligns public and maintainer package documentation with the current
static-first proof scripts. The goal is to make supported install guidance easy
to find while keeping package-manager, shared-library, runtime-loader, Windows
Makefile, Windows `pkg-config`, and dynamic ABI support as explicit non-claims.

## Documentation Changes

| File | Change | Reason |
| --- | --- | --- |
| `README.md` | Added a short note that Unix install proof validates installed include and library paths by filesystem identity. | Keeps the project front door aligned with the Day 8 path-normalized proof behavior without adding implementation detail to first-use commands. |
| `INSTALL.md` | Added operational detail for filesystem-identity checks on `pkg-config` prefix/libdir/includedir/cflags/libs paths. Added that focused install scripts reject unsupported shared-library, loader, package-manager, or dynamic ABI wording in their installed package files. Repeated that Windows `sparse.pc` checks are metadata-only inspection, not `pkg-config` command execution. | Makes validation steps repeatable and prevents users from reading Windows metadata inspection as Windows pkg-config parity. |
| `docs/maintainer_guide.md` | Updated the `tests/test_install.sh` ownership row to mention filesystem-identity path validation and semantic output checks. Added an explicit maintainer note that Windows `sparse.pc` inspection must not be cited as Windows `pkg-config` command execution. | Keeps maintainer proof-owner guidance synchronized with Day 8 script behavior. |
| `CMakeLists.txt` | Clarified the `sparse.pc` generation comment to keep generated flags aligned with semantic install tests. | Keeps package metadata comments aligned with the proof scripts without changing install behavior. |

## Cross-Link Map

| Source | Package Boundary Result |
| --- | --- |
| `README.md` | Routes detailed install behavior to `INSTALL.md`, summarizes static-first install, and keeps shared-library, package-manager, dynamic-loader, Windows Makefile, and Windows `pkg-config` parity as non-claims. |
| `INSTALL.md` | Owns operational install commands, validation commands, supported/deferred platform proof boundaries, and local proof interpretation. |
| `docs/maintainer_guide.md` | Owns maintainer proof-owner interpretation, validation command ownership, and package/ABI non-claim policy. |
| `docs/api_reference.md` | Explicitly states that API reference content does not imply dynamic ABI compatibility, shared-library support, package-manager distribution, or broad platform parity. |
| `docs/tutorial.md` | No package-manager, shared-library, runtime-loader, Windows pkg-config, or dynamic ABI support claim found in the Day 9 scan. |
| `docs/cookbook.md` | No package-manager, shared-library, runtime-loader, Windows pkg-config, or dynamic ABI support claim found in the Day 9 scan. |
| `docs/solver_selection.md` | No package-manager, shared-library, runtime-loader, Windows pkg-config, or dynamic ABI support claim found in the Day 9 scan. |

## Supported Package Reading After Alignment

Users can rely on:

- static archive install through Make on Unix-like environments;
- static archive install/export through CMake;
- downstream Unix `pkg-config` compile/link/run proof;
- downstream CMake `find_package(Sparse)` configure/build/run proof;
- exact package version checks for package resolution;
- hosted Linux and macOS static-first package proof lanes;
- hosted Windows CMake-first install/downstream validation with metadata-only
  `sparse.pc` inspection.

Users should not infer:

- package-manager distribution;
- shared-library support;
- runtime-loader behavior;
- dynamic ABI compatibility;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad platform package parity.

## Validation

Static package boundary checker:

```text
bash scripts/static_package_deferral_check.sh
```

Result: passed.

Cross-link package/support scan:

```text
rg -n 'package-manager support|package-manager distribution|shared-library support|dynamic ABI|runtime-loader|Windows `pkg-config`|metadata-only|pkg-config execution parity|Makefile parity' \
  docs/tutorial.md docs/cookbook.md docs/solver_selection.md docs/api_reference.md \
  README.md INSTALL.md docs/maintainer_guide.md
```

Result: no overclaim found. Hits were expected non-claim language in
`README.md`, `INSTALL.md`, `docs/api_reference.md`, and
`docs/maintainer_guide.md`.

Whitespace check:

```text
git diff --check
```

Result: passed.

## Completion Check

- Users can find the supported static install path from README and INSTALL.
- Maintainer validation steps now match the current proof scripts.
- Cross-linked docs do not overstate package-manager, shared-library,
  runtime-loader, Windows pkg-config, Windows Makefile, or dynamic ABI support.
