# Sprint 186 Day 5: User-Facing Claim Calibration

## Purpose

Calibrate README, INSTALL, and Homebrew package-proof documentation against the
Day 3 reconciled evidence matrix and the Day 4 public claim inventory.

## Scope

Day 5 addresses these Day 4 calibration items:

| ID | Surface | Day 5 status |
| --- | --- | --- |
| D4-CAL-001 | `README.md` package/install sections | Calibrated. |
| D4-CAL-002 | `INSTALL.md` support split | Calibrated. |
| D4-CAL-003 | `packaging/homebrew/README.md` | Calibrated. |
| D4-CAL-010 | state-of-the-art/support-tier language across public docs | Reviewed for Day 5 surfaces; no state-of-the-art, package support, ABI, or platform parity claim was added. |

## Documentation Changes

| File | Change |
| --- | --- |
| `README.md` | Added Sprint 186 closeout wording that treats the missing standalone license metadata as a residual provider-proof blocker rather than Homebrew availability, Homebrew/core readiness, tap support, bottles, Linuxbrew support, or broad package-manager distribution. |
| `INSTALL.md` | Added a support-split bullet classifying the missing standalone license metadata as a residual proof blocker and not a user-facing Homebrew installation path. |
| `packaging/homebrew/README.md` | Added proof-only closeout wording requiring approved standalone license metadata and successful render/install/`brew test`/uninstall/cleanup before the template can be presented as an available Homebrew install method. |

## Earned Claims Preserved

| Claim family | Day 5 result |
| --- | --- |
| Static-first source install | Preserved. README and INSTALL still direct users to Make or CMake source install as the maintained path. |
| Unix `pkg-config` and CMake installed-consumer proof | Preserved. Day 5 did not change install commands or package metadata. |
| Windows CMake static package metadata boundary | Preserved. Day 5 did not promote Windows Makefile or `pkg-config` execution parity. |
| Homebrew local proof path | Preserved as proof material only. The selected template, notes, and script remain source-controlled proof artifacts. |

## Non-Claims Preserved

Day 5 preserves these non-claims:

- package-manager support is not currently provided;
- Homebrew support, Homebrew/core readiness, public tap support, bottles, and
  Linuxbrew support are not claimed;
- vcpkg, Conan, pkgsrc, distro/system package, provider registry, and binary
  package support are not claimed;
- full Homebrew proof success is not claimed while standalone license metadata
  is absent;
- shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, static/shared selectors, and broad package-manager distribution
  remain unsupported;
- Windows `pkg-config` execution parity and Windows Makefile parity remain
  unsupported;
- no portable performance, release-readiness, broad platform parity, or
  state-of-the-art claim was added.

## Residuals Carried Forward

| Residual | Day 5 handling |
| --- | --- |
| R186-PKG-LICENSE | Remains active. Add approved standalone license metadata or decide an alternate formula license strategy before claiming full Homebrew proof success. |
| R186-WIN-PWSH | Unchanged by Day 5. Day 6 owns Windows/report documentation calibration. |
| R186-WIN-REPORT-FRESHNESS | Unchanged by Day 5. Day 6 owns Windows/report documentation calibration. |

## Validation

Day 5 changed documentation files only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

Required focused validation:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```
