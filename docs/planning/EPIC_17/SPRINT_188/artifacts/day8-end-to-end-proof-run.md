# Sprint 188 Day 8: End-to-End Homebrew Proof Run

## Purpose

Run the full local Homebrew proof command against the current Sprint 188
license state, classify the result as pass/block/fail, and verify no
temporary proof output or temporary formula state is retained.

## Current License State

| Field | Day 8 state |
| --- | --- |
| Approved root `LICENSE`, `COPYING`, or `NOTICE` | Absent. |
| Selected `SPARSE_HOMEBREW_LICENSE` value | None selected. |
| Proof expectation | Exit `2` before render/archive/install/test work and keep Homebrew support unclaimed. |
| Support promotion eligibility | Not eligible. Package wording must not promote Homebrew install support. |

## Proof Runs

| Run | Command | Exit | Result |
| --- | --- | ---: | --- |
| 1 | `scripts/homebrew_local_formula_proof.sh` | 2 | Expected unavailable blocker: no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists for provider metadata. |
| 2 | `scripts/homebrew_local_formula_proof.sh` | 2 | Same expected unavailable blocker reproduced. |

Both runs stopped through the claim-safe unavailable path and printed that the
local Homebrew proof remains unclaimed.

## Pass/Block/Fail Classification

| Classification | Day 8 result |
| --- | --- |
| Pass | Not reached. A pass requires approved root license metadata, accurate `SPARSE_HOMEBREW_LICENSE`, render, archive, checksum, install, installed-surface validation, `brew test`, uninstall, and cleanup success. |
| Block | Active. The proof is blocked by missing approved standalone root license metadata. |
| Fail | Not observed. No render, archive, checksum, install, test, uninstall, or cleanup failure occurred because the proof stopped before those phases. |

## Cleanup Verification

| Check | Result | Interpretation |
| --- | --- | --- |
| Generated Homebrew outputs under `packaging/homebrew` | Clean | No rendered formula, archive, log, bottle, or local tap output is present. |
| Temporary Homebrew formula install | Clean | `sparse-lu-ortho-local` is not installed as a Homebrew formula. |
| Recent proof temporary roots | Clean | No recent `sparse-homebrew-proof.*` temporary root remains under the local temp directory. |

## Guard Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and the selected local Homebrew proof boundary remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

## Support Promotion Decision Input

Day 8 does not provide support promotion evidence. The current state remains:

- local Homebrew proof material exists;
- the proof stops claim-safely because approved standalone root license
  metadata is absent;
- no full formula render/install/`brew test` proof has been earned; and
- Homebrew install support remains unclaimed.

Day 9 should align package guards to this blocker state and ensure any allowed
wording describes only local proof material plus the missing-license blocker,
not user-facing Homebrew availability.

## Validation Scope

Day 8 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
