# Sprint 207 Day 12 Integrated Validation

## Purpose

Day 12 ran the integrated validation set for the Sprint 207 package-provider
continued-deferral path. The validation covers the selected local Homebrew
proof, package-provider guards, static package boundaries, docs/support guards,
new Python regressions, install/downstream package checks, cleanup hygiene, and
the C-gate applicability decision.

## Validation Results

| Command | Result | Notes |
| --- | --- | --- |
| `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` | Passed | Completed local static source formula proof, downstream `brew test`, uninstall, and cleanup. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-first package and shared-library/dynamic ABI deferral boundary stayed intact. |
| `make docs-check` | Passed | Doxygen generation and API docs coverage passed. |
| `make support-docs-guard` | Passed | Support/readiness quick-reference docs remained aligned. |
| `python3 tests/test_package_manager_deferral_guard.py` | Passed | Sprint 207 provider-overclaim regression fixtures passed. |
| `python3 -m py_compile tests/test_package_manager_deferral_guard.py` | Passed | Python regression syntax check passed. |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed | Shell syntax check passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Deferral records, provider artifact absence, forbidden claims, embedded local proof boundary, metadata neutrality, and public non-claims passed. |
| `bash tests/test_install.sh` | Passed | Make install/uninstall and downstream `pkg-config` static package consumers passed: 23 passed, 0 failed. |
| `bash tests/test_cmake_install.sh` | Passed | CMake install/export and downstream `find_package(Sparse)` consumers passed: 27 passed, 0 failed, 0 skipped. |
| `git diff --check` | Passed | No whitespace errors. |

## Cleanup Checks

| Check | Result |
| --- | --- |
| `brew list --formula | rg '^sparse-lu-ortho-local$' || true` | No installed proof formula remained. |
| `brew tap | rg '^sparse-lu-ortho/local-proof-' || true` | No temporary proof tap remained. |
| `find packaging/homebrew -maxdepth 3 ...` | No generated archive, log, formula, bottle, or `Formula/` output was present. |

## C Gate Applicability

No `.c` or `.h` files are changed in the Sprint 207 branch state after Day 12.
The Day 12 plan requires `make format && make lint && make test` only when C
source or header files are modified, so the full C gate was not required for
this integrated validation pass.

## Blockers And Follow-Up

No validation blockers were found. Day 13 should review the changed package
guard, regression fixtures, user docs, maintainer docs, and validation records
for overclaim risk before closeout.
