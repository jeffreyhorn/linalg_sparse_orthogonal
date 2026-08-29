# Sprint 188 Day 7: Downstream Consumer Test Proof

## Purpose

Harden and document the local Homebrew formula `test do` downstream consumer
proof. The proof must demonstrate only the installed static package surface and
must not imply public tap, bottle, binary package, shared-library, dynamic ABI,
or broad package-manager support.

## Changes Made

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Adds `verify_formula_test_contract` to preflight the formula template's downstream test contract before license-gated rendering/install work. |
| `scripts/homebrew_local_formula_proof.sh` | Requires the template to keep exact-version `find_package(Sparse ...)`, `Sparse::sparse_lu_ortho`, installed public headers, successful executable output assertions, installed package checks, and shared-artifact rejection. |
| `scripts/package_manager_deferral_check.sh` | Guards that the proof script keeps downstream CMake consumer, imported-target link, and shared-artifact rejection preflight checks. |
| `packaging/homebrew/README.md` | Documents the downstream `test do` contract and its non-claim boundary. |

## Formula Test Contract

The Homebrew template's `test do` block must continue to:

1. create a temporary downstream CMake project;
2. require exact-version `find_package(Sparse #{expected_version} EXACT REQUIRED)`;
3. link `Sparse::sparse_lu_ortho`;
4. compile against installed public sparse headers;
5. create and use a minimal `SparseMatrix`;
6. assert successful executable output;
7. verify the installed static archive;
8. verify installed CMake package metadata;
9. verify installed pkg-config metadata; and
10. reject shared-library artifacts after the downstream test.

## Pass/Fail Interpretation

| Result | Interpretation |
| --- | --- |
| `brew test` passes after successful install-surface validation | The local formula proof has demonstrated one installed downstream CMake consumer for the maintained static archive package surface. |
| `brew test` fails | Package support promotion is blocked until the failure is diagnosed and fixed. |
| Template contract preflight fails | Package support promotion is blocked before rendering/install because the downstream consumer proof no longer matches the required boundary. |
| Proof exits `2` before `brew test` | The local proof remains unavailable and unclaimed because an earlier blocker, currently missing root license metadata, is still active. |

## Diagnostics and Cleanup

The proof script keeps the existing `brew test` diagnostics behavior: failed
test output is printed from the test log before the proof exits with failure.
The script also keeps uninstall-on-exit active after installation begins, so
failures during installed-surface validation or `brew test` attempt to remove
the temporary formula.

Temporary proof roots are removed by default unless `--keep-temp` is selected
for diagnostics. Kept logs, rendered formulae, archives, taps, caches, build
trees, install prefixes, and bottle outputs remain uncommitted proof outputs.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Passed | The formula template parses as Ruby. |
| Template marker audit for exact `find_package`, imported target, installed header use, output assertion, package metadata checks, and shared-artifact rejection | Passed | The current `test do` block matches the downstream consumer contract. |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable before `brew test` because no standalone root license metadata exists. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and downstream consumer proof guards remain intact. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

## Day 8 Handoff

Day 8 should run the full local Homebrew proof path with the selected license
state:

1. if approved root metadata remains absent, confirm expected exit `2` before
   render/install/test and keep support unclaimed;
2. if approved metadata is added, set the accurate `SPARSE_HOMEBREW_LICENSE`
   value and run the full render, archive, install, installed-surface, `brew
   test`, uninstall, and cleanup sequence;
3. verify no generated proof outputs are staged; and
4. record the proof result as pass, blocker, or failure.

## Validation Scope

Day 7 changed shell scripts and documentation but no `.c` or `.h` files, so
the full C quality gate is not required.
