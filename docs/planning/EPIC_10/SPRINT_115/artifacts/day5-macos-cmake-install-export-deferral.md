# Day 5 macOS CMake Install and Export Deferral

## Purpose

Day 5 applies the Day 4 macOS CMake install/export parity criteria and decides
whether to add a reviewed macOS install/export lane or publish an explicit
deferral.

## Decision

Defer reviewed macOS CMake install/export parity in Sprint 115.

No macOS workflow change is made. The current macOS support split remains:

- reviewed Apple Clang path for compile-quality, CMake parity, wall-check, and
  sanitizer coverage;
- supplemental Homebrew GCC direct build/test/wall-check coverage;
- supplemental Make install/`pkg-config` confidence path;
- local Unix-side `tests/test_cmake_install.sh` proof for CMake
  install/export and installed `find_package(Sparse)` consumers.

## Evidence Reviewed

| Evidence | Result |
|---|---|
| `.github/workflows/macos-ci.yml` | Already distinguishes reviewed Apple Clang checks from supplemental Homebrew GCC and supplemental Make install/`pkg-config` confidence. |
| `tests/test_cmake_install.sh` | Provides local CMake configure/build/install, installed package config files, exact-version behavior, and installed `find_package(Sparse)` consumer proof. |
| Sprint 112 Day 11 macOS follow-through | Recorded local CMake install/export proof but explicitly did not promote it to reviewed macOS install/export parity. |
| `README.md` | Describes macOS as reviewed Apple Clang plus supplemental GCC and static-first Make install/`pkg-config` verification. |
| `INSTALL.md` | States reviewed platform claims remain narrower than local install scripts and explicitly says macOS does not claim full reviewed install/export parity. |
| `docs/maintainer_guide.md` | Records macOS CMake install/export proof as local evidence, not reviewed macOS parity. |

## Criteria Application

| Criterion | Assessment |
|---|---|
| Material support-truth improvement | Limited. Current public wording already distinguishes supplemental Make install confidence from full reviewed CMake install/export parity. |
| Stable Apple Clang temp-prefix execution | Likely feasible, because the local script uses a temp prefix and CMake is already exercised on macOS. |
| Narrow static-first claim | Feasible, but still broadens reviewed macOS package ownership beyond the current support model. |
| Documentation readiness | Existing wording already accurately describes the deferred status. |
| CI ownership cost | A new reviewed install/export job would add runtime and future failure ownership for package metadata without changing a required public claim. |

## Deferral Contract

Until a future sprint explicitly promotes macOS CMake install/export parity:

- `tests/test_cmake_install.sh` remains local Unix-side proof.
- macOS CI does not claim full reviewed install/export parity.
- The supplemental macOS install job remains limited to Make
  install/`pkg-config` confidence.
- Public package wording must not imply that macOS reviewed CI validates
  installed CMake package consumers.
- Changes to CMake install/export behavior should still run
  `bash tests/test_cmake_install.sh` locally and record evidence.

## Missing Proof for Future Promotion

A future reviewed macOS CMake install/export claim should add:

1. A macOS CI step or job that runs `bash tests/test_cmake_install.sh` under
   Apple Clang on `macos-latest`.
2. Explicit workflow wording that the lane proves static-first CMake
   install/export only.
3. Documentation updates in README, INSTALL, and maintainer guide.
4. Runtime and dependency ownership for CMake/package metadata failures.
5. Confirmation that the lane does not imply shared-library, dylib, dynamic
   ABI, package-manager, or runtime-loader support.

## Support Wording Assessment

No wording changes are required on Day 5:

- `.github/workflows/macos-ci.yml` already says the install/`pkg-config` job
  is supplemental and not reviewed install/export parity.
- `README.md` already presents macOS package confidence as supplemental
  static-first Make install/`pkg-config` verification.
- `INSTALL.md` already says macOS does not claim full reviewed install/export
  parity.
- `docs/maintainer_guide.md` already records local CMake install/export proof
  as local evidence rather than reviewed macOS parity.

## Non-Claims

- No reviewed macOS CMake install/export parity claim is added.
- No macOS package-manager support claim is added.
- No macOS shared-library or dylib support claim is added.
- No dynamic ABI compatibility claim is added.
- No runtime-loader, install-name, rpath, or symbol visibility behavior claim
  is added.
- No workflow, script, build metadata, public API, or install-header behavior
  changed.

## Day 5 Validation

Day 5 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 5 because no `.c` or `.h` files
changed.
