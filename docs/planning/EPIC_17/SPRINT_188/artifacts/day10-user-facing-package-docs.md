# Sprint 188 Day 10: User-Facing Package Docs

## Purpose

Calibrate `README.md` and `INSTALL.md` to the exact Day 8 and Day 9 proof
state. The repository has local Homebrew proof material, but Homebrew install
support remains unclaimed because approved standalone root license metadata is
absent.

## Changed Surfaces

| Surface | Change |
| --- | --- |
| `README.md` | Reworded the installation package-manager paragraph to describe Sprint 188's current blocker state directly. |
| `INSTALL.md` | Reworded the support split so local Homebrew proof material is distinguished from user-facing Homebrew install support. |
| `INSTALL.md` | Added the current proof phase boundary: the proof exits before archive, render, install, or `brew test` work while approved standalone root license metadata is absent. |
| `INSTALL.md` | Added that no exact `SPARSE_HOMEBREW_LICENSE` value is selected until approved root metadata exists and placeholder values are blocker evidence. |
| `INSTALL.md` | Updated the package evidence note to match the Day 9 guard state. |

## Public Support Boundary

| Support surface | Day 10 wording |
| --- | --- |
| Source install | Supported through Make and CMake paths. |
| Installed static package surface | Supported through install validation and downstream consumer checks. |
| Local Homebrew proof material | Present as proof material under `packaging/homebrew/` and `scripts/homebrew_local_formula_proof.sh`. |
| Homebrew install support | Unclaimed while approved standalone root license metadata is absent. |
| Package-manager distribution | Unsupported. |
| Shared-library package support | Unsupported and explicitly deferred. |
| Dynamic ABI stability | Unsupported and explicitly deferred. |

## Retained Non-Claims

Day 10 retains these public non-claims:

- Homebrew/core readiness;
- bottles or hosted binary artifacts;
- Linuxbrew support;
- public tap maintenance;
- vcpkg, Conan, pkgsrc, distro/system package support;
- provider registry readiness;
- binary package install/update/uninstall support;
- shared-library package support;
- dynamic ABI compatibility;
- static/shared package selector support; and
- broad package-manager distribution.

## Guard Alignment

The Day 10 docs match the Day 9 guard state:

1. source/static install support remains first;
2. local Homebrew proof material may be described as proof material;
3. missing approved standalone root license metadata remains the active
   blocker;
4. Homebrew install support remains unclaimed unless the proof exits `0`; and
5. unsupported provider, bottle, tap, binary, shared-library, and dynamic ABI
   claims remain absent.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and stops before archive/render/install/test work because root license metadata is absent. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and the current Homebrew blocker wording remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain enforced. |

## Day 11 Handoff

Day 11 should update Homebrew-specific and maintainer documentation from the
same proof state:

1. keep local proof material separate from user-facing install support;
2. document approved license metadata requirements;
3. document placeholder license metadata as blocker evidence;
4. preserve cleanup/generated-output policy; and
5. keep package guards as the required validation before claim promotion.

## Validation Scope

Day 10 changed user-facing documentation and planning documentation but no
`.c` or `.h` files, so the full C quality gate is not required.
