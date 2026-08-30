# Sprint 188 Day 2: License Strategy Decision

## Purpose

Decide the Sprint 188 standalone license metadata strategy before changing the
Homebrew formula proof, package guards, or public package wording.

## Inputs Reviewed

| Input | Finding |
| --- | --- |
| Root metadata search | No root `LICENSE`, `COPYING`, or `NOTICE` file exists. |
| Repository license text search | No authoritative project license text, SPDX identifier, or copyright/license block was found in source-controlled project metadata. |
| Homebrew formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` requires `__SPARSE_HOMEBREW_LICENSE__` but does not define the license value. |
| Homebrew proof script | `scripts/homebrew_local_formula_proof.sh` requires a standalone root license metadata file and non-empty `SPARSE_HOMEBREW_LICENSE`. |
| Package docs | `README.md`, `INSTALL.md`, `packaging/homebrew/README.md`, and `docs/maintainer_guide.md` currently state that local Homebrew proof remains blocked by missing standalone license metadata. |
| Package guards | Package-manager and static-package guards currently pass while the proof remains blocked and unclaimed. |

## Decision

Sprint 188 must not invent license terms or select a Homebrew license
identifier without project-owner approval. The selected Day 2 strategy is:

1. Treat the repository as having no approved standalone license metadata yet.
2. Keep the Homebrew proof unavailable and unclaimed until a project-approved
   root `LICENSE`, `COPYING`, or `NOTICE` file exists.
3. Do not set a default `SPARSE_HOMEBREW_LICENSE` value in the formula
   template, proof script, package docs, or maintainer docs.
4. Preserve the current formula proof behavior where missing license metadata
   exits `2` and states that local Homebrew proof remains unclaimed.
5. Use Day 3 to implement a source-controlled license-decision/blocker record
   and any claim-safe guard or documentation updates needed to keep the
   blocker explicit.

This is an alternate formula license strategy in the Sprint 188 sense: the
proof remains guarded until approved metadata exists, rather than using
inaccurate provider metadata to force a proof pass.

## Rejected Options

| Option | Reason rejected |
| --- | --- |
| Add a guessed `LICENSE` file | No authoritative license text or project-owner approval was found in the repository. |
| Use a guessed SPDX identifier such as `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, or `Apache-2.0` | No source-controlled evidence supports any of these identifiers. |
| Use `SPARSE_HOMEBREW_LICENSE=NOASSERTION` or a local placeholder | Homebrew formula metadata would be inaccurate as support evidence, and the proof would risk passing with non-actionable license data. |
| Remove the Homebrew license field from the formula | It would weaken provider metadata and avoid rather than resolve the selected blocker. |
| Claim Homebrew support from the existing template alone | The Sprint 187 gates require a successful local proof or a guarded blocker; template existence is not support evidence. |

## Selected Metadata Owner

| Field | Day 2 decision |
| --- | --- |
| Root metadata owner | No approved file selected yet. Future approval should add `LICENSE`, `COPYING`, or `NOTICE` at the repository root. |
| Preferred proof path after approval | Add approved standalone root metadata, set `SPARSE_HOMEBREW_LICENSE` to the matching accurate Homebrew license identifier, and rerun the full local proof. |
| Day 3 implementation path | Add a blocker/decision artifact and update working notes so the sprint cannot silently treat the missing metadata as proof success. |
| Archive inclusion rule | Once approved metadata exists, the proof archive must include the selected root metadata file. |

## Homebrew License Identifier

No exact Homebrew license identifier is selected on Day 2.

The identifier remains blocked until the project owner approves the repository
license. After approval, the identifier must satisfy all of these conditions:

1. It matches the approved root license metadata.
2. It is accepted by the rendered Homebrew formula.
3. It is passed through `SPARSE_HOMEBREW_LICENSE`.
4. It is documented as local-proof metadata, not package-manager distribution
   support.

## Formula Metadata Implication

The formula metadata source is intentionally unresolved. The Homebrew proof
must continue to stop before formula rendering/install success while both
conditions are not true:

- an approved root standalone license metadata file exists; and
- `SPARSE_HOMEBREW_LICENSE` is set to the matching accurate identifier.

This preserves the Sprint 187 package acceptance gate and prevents a false
local Homebrew proof pass.

## Documentation Implication

Documentation may say:

- local Homebrew proof material exists;
- the proof is blocked by missing approved standalone license metadata;
- the blocker is not a user-facing Homebrew install path; and
- package-manager, Homebrew/core, bottle, Linuxbrew, public tap, binary
  package, shared-library, and dynamic ABI support remain unsupported.

Documentation must not say:

- the package has an approved Homebrew license identifier;
- Homebrew install support is available;
- Homebrew/core, bottles, Linuxbrew, public taps, or other provider support is
  ready; or
- shared-library package support or dynamic ABI stability exists.

## Day 3 Handoff

Day 3 should implement the safe metadata path selected here:

1. Add or update a source-controlled decision/blocker artifact for the missing
   approved root license metadata.
2. Keep `scripts/homebrew_local_formula_proof.sh` fail-safe on missing
   metadata and missing `SPARSE_HOMEBREW_LICENSE`.
3. Add guard coverage only if a drift point is found where placeholder or
   inaccurate license metadata could pass.
4. Keep public docs calibrated to blocker status unless project-approved
   license metadata is provided before Day 3 implementation.
5. Re-run package guards after any docs or guard updates.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
