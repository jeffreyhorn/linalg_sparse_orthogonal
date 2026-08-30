# Sprint 188 Day 12: Integrated Package Validation

## Purpose

Run the selected Sprint 188 package validation gate across Homebrew proof,
package guards, install proofs, documentation hygiene, generated-output
cleanup, and C quality-gate applicability.

## Changed Surface Review

| Surface | Day 12 decision |
| --- | --- |
| Homebrew proof script | Validate with shell syntax, expected blocker proof, and package-manager guard. |
| Package-manager guard | Validate directly because Sprint 188 changed Homebrew proof and package wording. |
| Static-package guard | Validate directly because Sprint 188 touched static/package support wording and installed metadata expectations. |
| Install and downstream docs | Validate with Make install and CMake install proofs because the sprint touched install-facing package guidance. |
| Package report metadata | Not changed during Day 12; package report normalization and freshness checks are not required. |
| C source/header files | No `.c` or `.h` files changed; `make format && make lint && make test` is not required. |

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | The proof remains unavailable because no standalone root `LICENSE`, `COPYING`, or `NOTICE` metadata exists. |
| Missing-license proof progress scan | Passed | The unavailable proof stops before temp archive, formula render, install, or `brew test` work. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and the selected local Homebrew proof boundary remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package support and shared-library/dynamic ABI deferrals remain guarded. |
| `bash tests/test_install.sh` | Passed | Make install/uninstall, static archive install, installed headers, `pkg-config`, and downstream consumers remain valid. |
| `bash tests/test_cmake_install.sh` | Passed | CMake install/export, exact-version consumers, static imported target metadata, and package metadata remain valid. |
| `git diff --check` | Passed | Current diff has no whitespace errors. |
| Trailing-whitespace scan | Passed | Changed docs and scripts have no trailing whitespace. |
| Homebrew generated-output scan | Passed | No generated formula, archive, log, bottle, or local tap output exists under `packaging/homebrew`. |
| Sprint 188 markdown link check | Passed | Sprint-local markdown links resolve. |

## Proof State

The Homebrew proof still exits at the expected unavailable state:

```text
homebrew-local-formula-proof: UNAVAILABLE: formula rendering blocked: no standalone LICENSE, COPYING, or NOTICE file exists for provider metadata
homebrew-local-formula-proof: local Homebrew proof remains unclaimed
```

This is blocker evidence, not pass evidence. Sprint 188 may document the local
proof material and the guarded blocker state, but it must not promote Homebrew
install support while the proof exits `2`.

## Skipped Checks

| Check | Reason |
| --- | --- |
| `python3 scripts/normalize_report_index.py --family package --check` | Package report metadata and package report artifacts were not changed. |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Package report metadata and package report artifacts were not changed. |
| `make format && make lint && make test` | No `.c` or `.h` files were modified. |

## Day 13 Handoff

Day 13 can perform the final claim audit from a passing validation baseline.
The expected residual decision is to keep Homebrew install support unclaimed
unless approved standalone root license metadata is added and the local proof
exits `0` with package guards still passing.
