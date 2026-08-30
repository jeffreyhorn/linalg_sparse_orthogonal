# Sprint 188 Day 9: Package Guard Alignment

## Purpose

Align package-manager and static-package guards with the current Sprint 188
proof state. The repository has local Homebrew proof material, but the full
proof remains blocked because no approved standalone root license metadata
exists.

## Current Proof State

| Field | State |
| --- | --- |
| Root `LICENSE`, `COPYING`, or `NOTICE` | Absent. |
| Exact `SPARSE_HOMEBREW_LICENSE` | Unselected. |
| Homebrew proof command | Expected exit `2`. |
| Archive/render/install/test work | Must not run while root license metadata is absent. |
| Public support wording | Must keep Homebrew support unclaimed. |
| Package report metadata | Not changed. No package report normalization update is required. |

## Guard Changes

| Guard | Change |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Detects whether root standalone license metadata exists before running the Homebrew proof script. |
| `scripts/package_manager_deferral_check.sh` | When root metadata is absent and the proof exits `2`, requires output to name the missing standalone license blocker. |
| `scripts/package_manager_deferral_check.sh` | When root metadata is absent, fails if proof output shows archive, render, install, or `brew test` work started. |
| `scripts/package_manager_deferral_check.sh` | Requires `INSTALL.md` to state the blocker is not a user-facing Homebrew installation path. |
| `scripts/package_manager_deferral_check.sh` | Requires `packaging/homebrew/README.md` to keep the template from being presented as an available Homebrew install method. |

## Guard Alignment Matrix

| Surface | Expected Day 9 guard behavior |
| --- | --- |
| Local Homebrew proof material | Allowed as source-controlled proof material. |
| Full Homebrew proof success | Blocked until approved root license metadata and accurate Homebrew license metadata exist. |
| Missing-license proof output | Must exit `2`, keep support unclaimed, and stop before archive/render/install/test work. |
| Public Homebrew wording | Must describe the local proof blocker, not an available install route. |
| Unselected providers | vcpkg, Conan, pkgsrc, distro packages, public taps, bottles, hosted binaries, and provider registry readiness remain rejected. |
| Static package policy | Static-first package metadata remains guarded separately by `scripts/static_package_deferral_check.sh`. |

## Package Report Metadata Decision

Day 9 did not change selected report target metadata or package report rows.
The package report normalization checks are not required for this day.

If later Sprint 188 work changes package report metadata, run:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and stops before archive/render/install/test work because root license metadata is absent. |
| `scripts/package_manager_deferral_check.sh` | Passed | Guard matches the current proof state and keeps public Homebrew support unclaimed. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain enforced. |

## Day 10 Handoff

Day 10 should calibrate README and INSTALL wording from the current Day 9
guard state:

1. source/static install support stays first;
2. local Homebrew proof material may be described only as proof material;
3. missing approved standalone root license metadata remains the active
   blocker;
4. Homebrew support must remain unclaimed unless the proof exits `0`; and
5. provider, bottle, tap, binary, shared-library, and dynamic ABI non-claims
   must remain explicit.

## Validation Scope

Day 9 changed shell scripts and planning documentation but no `.c` or `.h`
files, so the full C quality gate is not required.
