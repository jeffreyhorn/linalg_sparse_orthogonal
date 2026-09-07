# Sprint 198 Day 3: Root Metadata Implementation

## Purpose

Execute the Day 3 root metadata implementation step according to the Day 2
license decision, without inventing license terms or a Homebrew identifier.

## Day 3 Disposition

Root license metadata is not implemented on Day 3 because Day 2 found no
approved standalone root license metadata and no exact Homebrew formula
license identifier in source-controlled project evidence.

This is a blocked implementation outcome, not a proof success. The correct
claim-safe state remains:

- no root `LICENSE`, `COPYING`, or `NOTICE` is added without approval;
- no guessed `SPARSE_HOMEBREW_LICENSE` value is selected;
- Homebrew formula rendering remains guarded;
- the local formula proof exits `2` before archive/render/install/test work;
- Homebrew and broad package-manager support remain unclaimed.

## Metadata Implementation Check

| Check | Result | Interpretation |
| --- | --- | --- |
| Root `LICENSE`, `COPYING`, or `NOTICE` present | No | Active metadata blocker remains. |
| Approved Homebrew license identifier present | No | Formula metadata cannot be implemented accurately. |
| Formula template license placeholder present | Yes | Template remains guarded until approved render input exists. |
| Proof script missing-license stop condition present | Yes | Fail-safe behavior remains intact. |
| Proof script placeholder rejection present | Yes | Placeholder metadata cannot become proof metadata. |
| Package docs describe blocker status | Yes | Public support remains unclaimed. |

## Archive Inclusion Expectation

Once approved metadata exists, the selected root metadata file must be included
at the source archive root beside:

- `CMakeLists.txt`;
- `Makefile`;
- `VERSION`;
- `sparse.pc.in`;
- `cmake/`;
- `include/`;
- `src/`;
- `examples/`.

The current proof script already appends detected root license metadata entries
to the archive input list and verifies them with `verify_source_archive()`.
Day 3 does not alter that behavior because there is no approved metadata file
to include yet.

## Stale Wording Review

The current public and maintainer package wording remains claim-safe because it
states that:

- package-manager support is not currently provided;
- local Homebrew formula proof material exists as blocker/provenance evidence;
- no exact `SPARSE_HOMEBREW_LICENSE` value is selected until approved root
  license metadata exists;
- Homebrew/core, bottles, Linuxbrew, taps, other package managers, binary
  packages, shared-library package support, and dynamic ABI support remain
  unsupported.

One planning status row in `docs/planning/EPIC_18/PROJECT_PLAN.md` was stale
after Day 1 and Day 2 created Sprint 198 artifacts. Day 3 updates that row so
Sprint 198 is no longer described as having no artifact directory.

## Day 4 Handoff

Day 4 should keep formula metadata wiring blocked unless approved license
inputs are provided before implementation. If no approved inputs are available,
Day 4 should verify the formula template remains placeholder-guarded and
document that no render metadata promotion is possible.

## Validation

Day 3 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
