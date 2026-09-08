# Sprint 198 Day 10: Package Guard Promotion

## Purpose

Promote guard coverage for the Sprint 198 package-manager evidence state so
support wording remains tied to approved metadata and proof success rather
than planned intent.

## Day 10 Disposition

The package-manager guard was updated to recognize the Sprint 198 metadata
blocker records. No Homebrew support wording is promoted because the Day 9
proof attempt still exits `2` before archive, checksum, render, install,
installed-surface validation, `brew test`, or uninstall work.

The static-package guard already preserves the maintained static archive
package boundary and shared-library/dynamic ABI non-claims, so Day 10 does not
change its behavior.

## Guard Change

`scripts/package_manager_deferral_check.sh` now verifies:

- the Sprint 198 license metadata decision artifact exists;
- the Sprint 198 end-to-end proof run artifact exists;
- the license decision records that no approved standalone root metadata or
  exact Homebrew identifier is present;
- guessed license terms remain rejected;
- Homebrew/package-manager support remains unclaimed;
- the Day 9 proof run records exit `2`;
- archive, checksum, render, install, static surface validation, `brew test`,
  and uninstall remain unearned;
- broader Homebrew non-claims remain explicit.

This extends the existing guard without weakening its prior checks for the
Sprint 171 deferral record, provider recipe absence, selected Homebrew proof
boundary, package metadata neutrality, and public non-claims.

## Static Package Guard Review

`scripts/static_package_deferral_check.sh` remains appropriate for Day 10. It
continues to guard:

- `BUILD_SHARED_LIBS=ON` rejection;
- explicit static target declaration;
- Makefile static archive install/uninstall behavior;
- CMake static archive install metadata;
- absence of shared export/ABI metadata;
- absence of package static/shared selectors;
- public shared-library and dynamic ABI non-claims;
- Windows package non-claim wording.

No Day 10 change is needed there because no static-package support tier changed.

## Focused Validation

| Command | Result |
| --- | --- |
| `bash scripts/package_manager_deferral_check.sh` | Passed after the Sprint 198 metadata blocker checks were added. |
| `bash scripts/static_package_deferral_check.sh` | Passed with no Day 10 changes required. |
| `bash scripts/homebrew_local_formula_proof.sh` | Exited `2` at the expected missing standalone root metadata gate. |
| `git diff --check` | Passed. |

## Claim Boundary

Day 10 promotes guard coverage, not package-manager support. Homebrew/core,
bottles, Linuxbrew, public taps, binary packages, provider registry readiness,
shared-library package support, dynamic ABI compatibility, runtime-loader
behavior, and broad package-manager support remain unclaimed.

## Day 11 Handoff

Day 11 should update public package documentation only to reflect the current
guarded blocker state unless approved license metadata and proof success
arrive before documentation promotion begins.

## Validation

Day 10 changes a shell guard and planning documentation only. No `.c` or `.h`
files were modified, so the full C quality gate is not required.
