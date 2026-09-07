# Sprint 198 Day 2: License Metadata Decision

## Purpose

Record the Sprint 198 license metadata decision before changing root license
files, Homebrew formula metadata, package guards, or public package-manager
wording.

## Inputs Reviewed

| Input | Finding |
| --- | --- |
| Root metadata search | No root `LICENSE`, `COPYING`, or `NOTICE` file exists. |
| Public README license section | `README.md` says the project is for research and educational purposes, but does not provide standalone license terms or a Homebrew/SPDX-style license identifier. |
| Install guidance | `INSTALL.md` keeps package-manager support unclaimed and states no exact `SPARSE_HOMEBREW_LICENSE` value is selected until approved root license metadata exists. |
| Homebrew formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` contains `license "__SPARSE_HOMEBREW_LICENSE__"` and requires render-time substitution. |
| Homebrew proof script | `scripts/homebrew_local_formula_proof.sh` requires standalone root metadata and a non-placeholder `SPARSE_HOMEBREW_LICENSE` before archive/render/install/test work. |
| Package docs | `packaging/homebrew/README.md` requires a project-approved Homebrew license identifier matching a root `LICENSE`, `COPYING`, or `NOTICE`. |
| Maintainer guide | `docs/maintainer_guide.md` keeps package-manager support unclaimed until root metadata, exact identifier, proof exit `0`, guards, and docs land together. |
| Prior proof records | Sprint 188 and Epic 18 residual records identify missing standalone license metadata as the active blocker and reject guessed identifiers. |

## Decision

No approved standalone root license metadata or exact Homebrew formula license
identifier is selected on Day 2.

The Sprint 198 implementation path remains blocked until a project-owner or
legal/product decision provides both:

1. a standalone root `LICENSE`, `COPYING`, or `NOTICE` file; and
2. the exact matching Homebrew formula license identifier for
   `SPARSE_HOMEBREW_LICENSE`.

Sprint 198 must not invent license terms from the README sentence or select a
guessed identifier such as `MIT`, `BSD-2-Clause`, `BSD-3-Clause`,
`Apache-2.0`, or `NOASSERTION`. The current proof behavior remains correct:
missing approved metadata exits `2`, stops before archive/render/install/test
work, and keeps local Homebrew proof unclaimed.

## Selected Metadata Owner

| Field | Day 2 disposition |
| --- | --- |
| Root metadata file | Not selected. Future approval must add a root `LICENSE`, `COPYING`, or `NOTICE`. |
| Archive path | Not selected. Once approved, the selected root metadata file must be included at the archive root. |
| Homebrew identifier | Not selected. Future approval must provide an exact Homebrew-accepted identifier matching the root metadata. |
| Formula metadata source | Remains unresolved and guarded by `SPARSE_HOMEBREW_LICENSE`. |
| Support wording | Homebrew/package-manager support remains unclaimed. |

## Rejected Options

| Option | Reason rejected |
| --- | --- |
| Treat the README research/educational sentence as package license metadata | It is not a standalone license grant and does not define a Homebrew formula identifier. |
| Add a guessed root `LICENSE` file | The repository contains no authoritative source for the exact terms. |
| Set `SPARSE_HOMEBREW_LICENSE` to a common guessed identifier | A guessed identifier would create inaccurate provider metadata. |
| Use `NOASSERTION`, `UNKNOWN`, `TBD`, or placeholder text | The proof script and package docs already define placeholder values as blocker evidence, not proof metadata. |
| Remove the formula `license` field | That would avoid the selected provider metadata requirement instead of closing it. |
| Claim Homebrew support from the template and proof script alone | The local formula proof has not completed archive, render, install, `brew test`, uninstall, and cleanup. |

## Invalid Metadata Values

Guards and proof logic must continue to reject:

- missing root `LICENSE`, `COPYING`, or `NOTICE`;
- empty `SPARSE_HOMEBREW_LICENSE`;
- unresolved `__SPARSE_HOMEBREW_LICENSE__`;
- `NOASSERTION`;
- `UNKNOWN`;
- `TBD`;
- `TODO`;
- `FIXME`;
- `PLACEHOLDER`;
- any value that is documented as placeholder metadata rather than an approved
  Homebrew identifier.

## Formula and Guard Implication

No formula metadata implementation should proceed until the approved metadata
exists. The proof script should continue to fail early with exit `2` before it
creates a temporary archive, renders a formula, installs, runs `brew test`, or
claims success.

Package-manager and static-package guards should continue to pass only while
public wording keeps Homebrew support, broader package-manager support,
Homebrew/core readiness, bottles, Linuxbrew, public taps, binary packages,
shared-library package support, and dynamic ABI support out of scope.

## Day 3 Handoff

Day 3 should not add a guessed root license file. Instead, it should:

1. preserve the current fail-safe proof behavior;
2. add or tighten source-controlled blocker documentation if any stale wording
   implies an approved license decision already exists;
3. keep formula metadata unresolved until approved inputs are provided;
4. run package guards after any documentation or guard edits; and
5. carry the missing metadata decision blocker into the sprint evidence ledger.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required.
