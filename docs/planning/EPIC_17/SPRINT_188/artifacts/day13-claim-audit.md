# Sprint 188 Day 13: Package Claim Audit

## Purpose

Audit the Sprint 188 Homebrew proof work against the final validation evidence
and decide whether the sprint can promote a package-manager support claim or
must retain a bounded residual blocker.

## Final State Decision

Sprint 188 retains a guarded residual blocker.

The repository has source-controlled local Homebrew proof material, a hardened
proof script, a guarded temporary formula template, and passing package/install
validation around the static package surface. It does not have approved
standalone root license metadata, and no exact `SPARSE_HOMEBREW_LICENSE`
identifier is selected. The selected Homebrew proof therefore exits `2` before
archive, render, install, or `brew test` work.

This is blocker evidence, not Homebrew install support evidence.

## Evidence Reviewed

| Evidence | Result | Claim impact |
| --- | --- | --- |
| Root `LICENSE`, `COPYING`, or `NOTICE` metadata scan | Absent | Homebrew proof cannot be promoted. |
| Exact `SPARSE_HOMEBREW_LICENSE` value | Unselected | Formula metadata remains blocked until project-approved license metadata exists. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and support remains unclaimed. |
| Missing-license proof progress scan | Passed | Proof stops before temp archive, formula render, install, or `brew test` work. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and selected local Homebrew proof boundary remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package support and shared-library/dynamic ABI non-claims remain guarded. |
| `bash tests/test_install.sh` | Passed | Existing Make install and `pkg-config` static package surface remains valid. |
| `bash tests/test_cmake_install.sh` | Passed | Existing CMake install/export static package surface remains valid. |

## Touched Documentation Audit

| Surface | Audit result |
| --- | --- |
| `README.md` | States package-manager support is not currently provided and treats the Homebrew state as a provider-proof blocker, not availability. |
| `INSTALL.md` | Distinguishes source install/static package support from package-manager support and keeps the Homebrew blocker out of user-facing install paths. |
| `packaging/homebrew/README.md` | Documents local proof-only scope, exit-code interpretation, generated-output policy, and retained non-claims. |
| `docs/maintainer_guide.md` | Documents proof command ownership, exit-code meaning, validation commands, and support-promotion limits. |
| Sprint 188 artifacts and working notes | Record the missing-license residual and do not promote broader provider support. |

## Retained Non-Claims

Sprint 188 does not claim:

- Homebrew install availability;
- Homebrew/core readiness or submission status;
- bottle or hosted binary package support;
- Linuxbrew support;
- public tap maintenance;
- vcpkg, Conan, pkgsrc, distro/system package support;
- provider registry readiness;
- binary package install/update/uninstall support;
- shared-library package support;
- dynamic ABI compatibility;
- static/shared package selection knobs; or
- broad package-manager distribution.

## Revisit Criteria

The Homebrew local proof can be reconsidered only after all of these are true:

1. Approved standalone root license metadata exists as `LICENSE`, `COPYING`,
   or `NOTICE`.
2. `SPARSE_HOMEBREW_LICENSE` is set to the accurate matching Homebrew license
   identifier.
3. `scripts/homebrew_local_formula_proof.sh` exits `0`.
4. The proof reaches archive creation, formula render, install, installed
   metadata validation, `brew test`, uninstall, and cleanup.
5. `scripts/package_manager_deferral_check.sh` passes.
6. `scripts/static_package_deferral_check.sh` passes.
7. User-facing docs are updated only to the exact proof level achieved.

Any broader provider claim, including Homebrew/core, bottles, Linuxbrew,
public taps, or other package managers, requires a separate product decision
and separate evidence.

## Retrospective Inputs

- Sprint 188 completed proof hardening and documentation calibration without
  inventing license metadata.
- The main residual is narrow and explicit: approved standalone root license
  metadata is absent.
- Package-manager support wording is now guarded by scripts and documented
  for maintainers.
- The existing static install/package surface continues to pass Make and
  CMake install validation.
- Day 14 should close the sprint as a guarded residual rather than a promoted
  Homebrew support claim.
