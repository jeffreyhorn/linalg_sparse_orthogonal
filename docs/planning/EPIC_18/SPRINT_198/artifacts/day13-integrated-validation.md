# Sprint 198 Day 13: Integrated Validation

## Purpose

Run the required Sprint 198 package, docs, install, and changed-surface
validation checks, and record residuals without converting unavailable
Homebrew proof evidence into support claims.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash scripts/homebrew_local_formula_proof.sh` | Exit `2` | Expected unavailable blocker: no standalone root `LICENSE`, `COPYING`, or `NOTICE` exists for provider metadata. The proof stops before archive, checksum, render, install, installed-surface validation, `brew test`, or uninstall work. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager guard verifies Sprint 171 deferral, Sprint 198 metadata-blocker records, provider recipe absence, selected Homebrew local proof boundary, package metadata neutrality, and public non-claims. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-package guard verifies static archive package contract and shared-library/dynamic ABI non-claims. |
| `bash tests/test_install.sh` | Passed | Make install/uninstall and pkg-config installed-consumer validation passed: 23 passed, 0 failed. |
| `bash tests/test_cmake_install.sh` | Passed | CMake configure/build/install, installed package metadata, exact-version `find_package`, and downstream consumer validation passed: 27 passed, 0 failed, 0 skipped. |
| `make docs-check` | Passed | Doxygen generated API docs and API docs coverage passed for checked-in public headers. |
| `git diff --check` | Passed | No whitespace errors in the current diff. |

## Changed-Surface Assessment

| Surface | Changed? | Validation |
| --- | --- | --- |
| Public package docs | Yes | Package-manager guard, static-package guard, and docs check passed. |
| Maintainer guide | Yes | Package-manager guard, static-package guard, and docs check passed. |
| Package-manager guard script | Yes | `bash scripts/package_manager_deferral_check.sh` passed after Sprint 198 blocker checks were added. |
| Homebrew formula template | No | Prior Day 6 syntax/placeholder validation remains current; package guard still validates selected proof boundary. |
| Install metadata | No | `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh` passed. |
| `.c` / `.h` files | No | Full C gate is not required by the sprint rule. |

## Residual

The selected Homebrew proof remains unavailable because the repository still
does not contain approved standalone root license metadata or an exact
matching Homebrew formula license identifier. This is a metadata approval
residual, not a local tool residual: local `brew`, `cmake`, `ruby`, `tar`,
`shasum`, and `cc` were available during earlier sprint checks.

## Generated Artifact Review

No generated Homebrew proof outputs were found under `packaging/homebrew`.
Validation did not add tracked generated API documentation changes to the
working tree.

## Claim Boundary

Day 13 validates the current blocker and package/install documentation
surfaces. It does not earn Homebrew support, Homebrew/core readiness, bottles,
Linuxbrew support, public tap support, binary package distribution,
shared-library package support, dynamic ABI support, runtime-loader behavior,
or broad package-manager support.

## Day 14 Handoff

Day 14 should finalize closeout and retrospective inputs with Sprint 198 item
status split between blocked metadata/proof work, partial guard/docs alignment
for the blocker state, and completed validation of the current guarded
surfaces.
