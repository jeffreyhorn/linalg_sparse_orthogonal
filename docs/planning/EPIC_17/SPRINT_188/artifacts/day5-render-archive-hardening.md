# Sprint 188 Day 5: Render and Archive Proof Hardening

## Purpose

Harden the first half of `scripts/homebrew_local_formula_proof.sh` so render
and archive failures stop early, produce clear diagnostics, and cannot be
mistaken for package support evidence.

## Changes Made

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Moves license metadata detection before temporary archive creation, so missing root metadata stops before archive/checksum work. |
| `scripts/homebrew_local_formula_proof.sh` | Tracks root `LICENSE`, `COPYING`, or `NOTICE` entries selected by metadata detection for later archive inclusion. |
| `scripts/homebrew_local_formula_proof.sh` | Adds `verify_source_archive` to ensure successful archives contain required source, package metadata, example, and standalone license entries. |
| `scripts/package_manager_deferral_check.sh` | Guards that the proof script retains required source archive entry verification. |
| `packaging/homebrew/README.md` | Documents that license metadata is validated before archive creation and must be included in any future successful proof archive. |

## Render and Archive Behavior

| Condition | Day 5 behavior | Claim effect |
| --- | --- | --- |
| Missing root `LICENSE`, `COPYING`, or `NOTICE` | Proof exits `2` before temporary archive creation. | Homebrew support remains unclaimed. |
| Missing `SPARSE_HOMEBREW_LICENSE` after approved metadata exists | Proof exits `2` before rendering the formula. | Homebrew support remains unclaimed. |
| Placeholder `SPARSE_HOMEBREW_LICENSE` after approved metadata exists | Proof exits `2` before rendering the formula. | Homebrew support remains unclaimed. |
| Archive cannot be created | Proof exits nonzero failure. | Sprint must stop and fix the proof before any support promotion. |
| Archive is missing required entries | Proof exits nonzero failure with the missing entry name. | Sprint must stop and fix the archive before any support promotion. |
| Rendered formula has unresolved placeholders | Proof exits nonzero failure through the existing render check. | Sprint must stop and fix the render path before any support promotion. |

## Required Archive Entries

Future successful source archives must include:

- `CMakeLists.txt`
- `Makefile`
- `VERSION`
- `sparse.pc.in`
- `cmake`
- `include`
- `src`
- `examples`
- every detected root `LICENSE`, `COPYING`, or `NOTICE` metadata file

The archive remains temporary proof output and must not be committed.

## Cleanup Policy

Day 5 preserves the existing cleanup policy:

- temporary roots are removed by default;
- `--keep-temp` may retain generated proof material for diagnostics;
- retained diagnostic output is not source-controlled evidence by itself;
- rendered formula files, archives, taps, logs, caches, build trees, install
  prefixes, and bottle outputs must stay out of version control.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof exits before archive creation because no root standalone license metadata exists; Homebrew support remains unclaimed. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims, selected Homebrew boundary, placeholder metadata rejection, and archive verification guard remain intact. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

## Day 6 Handoff

Day 6 can start from a hardened render/archive phase and focus on install
surface proof behavior:

1. verify installed static archive, headers, CMake package files, and
   `sparse.pc`;
2. keep shared-library artifacts rejected;
3. preserve failure cleanup after partial installs;
4. keep unsupported provider or ABI wording out of installed metadata; and
5. record retry behavior for failed proof attempts.

## Validation Scope

Day 5 changed shell scripts and documentation but no `.c` or `.h` files, so
the full C quality gate is not required.
