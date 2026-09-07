# Sprint 198 Day 11: Public Package Documentation

## Purpose

Update public package-manager documentation to match the current Sprint 198
evidence: the selected local Homebrew formula proof remains blocked by missing
approved license metadata and no support tier is promoted.

## Public Documentation Changes

| File | Change |
| --- | --- |
| `README.md` | Added Sprint 198 wording that the current local Homebrew proof exits claim-safely before archive, render, install, or `brew test` work because approved standalone root license metadata and the exact Homebrew formula license identifier are missing. |
| `INSTALL.md` | Updated the package-manager deferral bullet from Sprint 188 to Sprint 198 and added the missing exact `SPARSE_HOMEBREW_LICENSE` value to the blocker summary. |
| `packaging/homebrew/README.md` | Updated the proof-only status paragraph from Sprint 186 closeout to Sprint 198 and made the exact license identifier an explicit prerequisite before the template can be presented as an install method. |

## Support Tier

No stronger support tier is earned on Day 11. The public documentation may
state only that local Homebrew formula proof material exists and currently
stops at the metadata blocker.

Package-manager support, package-manager distribution, Homebrew/core,
bottles, Linuxbrew, public tap support, binary packages, vcpkg, Conan, pkgsrc,
distro/system packages, shared-library package support, dynamic ABI support,
runtime-loader behavior, and broad provider support remain unclaimed.

## Guard Alignment

The Day 11 wording keeps the package-manager guard vocabulary intact:

- `package-manager support`;
- `local Homebrew formula proof`;
- `package-manager distribution`;
- `not a user-facing Homebrew installation path`;
- `do not present this template as an available Homebrew install method`.

## Validation

| Command | Result |
| --- | --- |
| `bash scripts/package_manager_deferral_check.sh` | Passed after preserving the Homebrew README non-claim marker. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `make docs-check` | Passed. |
| `git diff --check` | Passed. |

Day 11 changes documentation and a shell guard only. No `.c` or `.h` files
were modified, so the full C quality gate is not required.
