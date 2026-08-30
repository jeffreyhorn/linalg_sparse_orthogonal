# Sprint 188 Day 3: Metadata Implementation

## Purpose

Implement the Day 2 safe metadata path without inventing project license
terms. Day 3 records the active blocker, keeps package proof unclaimed, and
adds a lightweight guard so placeholder Homebrew license metadata cannot become
future proof evidence.

## Day 2 Decision Applied

| Day 2 decision | Day 3 implementation |
| --- | --- |
| Do not invent root license terms. | No `LICENSE`, `COPYING`, or `NOTICE` file was added. |
| Do not choose a guessed `SPARSE_HOMEBREW_LICENSE` value. | No default Homebrew license identifier was added to the formula, proof script, docs, or environment. |
| Keep the proof unavailable and unclaimed until approved metadata exists. | The proof script still exits `2` when standalone root license metadata is absent. |
| Prevent inaccurate metadata from forcing a proof pass. | `scripts/homebrew_local_formula_proof.sh` now rejects placeholder license identifiers before rendering a formula. |
| Keep guard/doc alignment. | `scripts/package_manager_deferral_check.sh`, `packaging/homebrew/README.md`, and `docs/maintainer_guide.md` now include placeholder-license guard expectations. |

## Changed Surfaces

| Surface | File | Change |
| --- | --- | --- |
| Proof script | `scripts/homebrew_local_formula_proof.sh` | Rejects placeholder `SPARSE_HOMEBREW_LICENSE` values such as `NOASSERTION`, `UNKNOWN`, `TBD`, `TODO`, `FIXME`, `PLACEHOLDER`, and unresolved template placeholder text. |
| Package guard | `scripts/package_manager_deferral_check.sh` | Requires the proof script to retain the placeholder-license rejection message. |
| Package docs | `packaging/homebrew/README.md` | Documents that placeholder license values are blocker evidence, not proof metadata. |
| Maintainer docs | `docs/maintainer_guide.md` | Documents that future metadata must use an accurate matching Homebrew license identifier. |

## Metadata State

| Field | Day 3 state |
| --- | --- |
| Root standalone license metadata | Still absent. No approved license text exists in source control. |
| Homebrew license identifier | Still unselected. No exact identifier can be chosen without approved root metadata. |
| Formula rendering | Still blocked before install proof while root metadata is absent. |
| Archive inclusion | The proof script already includes root `LICENSE`, `COPYING`, or `NOTICE` files when they exist; no archive proof can pass until approved metadata is added. |
| Claim state | Homebrew support remains unclaimed. The repository has local proof material, not a user-facing Homebrew install route. |

## Placeholder License Rejection

The proof script now treats these values as unavailable blocker evidence:

- `NOASSERTION`
- `UNKNOWN`
- `TBD`
- `TODO`
- `FIXME`
- `PLACEHOLDER`
- unresolved `__SPARSE_HOMEBREW_LICENSE__` template text
- values containing `placeholder`, `Placeholder`, or `PLACEHOLDER`

The rejection exits through the existing unavailable path, so the result stays
claim-safe: package support remains unclaimed instead of failing later as an
ambiguous formula or install error.

## Proof-State Interpretation

| Proof result | Interpretation |
| --- | --- |
| Exit `0` | Local Homebrew source formula proof passed for the maintained static archive package surface. This is not expected until approved root license metadata exists. |
| Exit `2` on missing root metadata | Expected Sprint 188 blocker state. Homebrew support remains unclaimed. |
| Exit `2` on missing or placeholder `SPARSE_HOMEBREW_LICENSE` | Expected blocker state after root metadata exists but accurate formula metadata is not supplied. |
| Any other nonzero exit | Proof failure that must be fixed before any package wording promotion. |

## Validation Requirements

Because Day 3 changed scripts and documentation but no `.c` or `.h` files,
the required validation is:

```sh
scripts/homebrew_local_formula_proof.sh
scripts/package_manager_deferral_check.sh
scripts/static_package_deferral_check.sh
git diff --check
```

The Homebrew proof command is expected to exit `2` until approved root license
metadata exists. Package support must remain unclaimed in that state.

## Day 3 Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root license metadata exists; Homebrew support remains unclaimed. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and selected Homebrew boundary remain guarded, including the placeholder-license check. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `git diff --check` | Passed | No whitespace errors were introduced. |

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 3.

## Day 4 Handoff

Day 4 should audit `packaging/homebrew/sparse-lu-ortho.rb.in` against the
updated metadata policy:

1. required placeholders remain present;
2. formula license metadata is injected only from `SPARSE_HOMEBREW_LICENSE`;
3. local static source formula scope remains explicit;
4. `test do` still proves a downstream CMake consumer only; and
5. no provider, bottle, public tap, shared-library, or dynamic ABI support is
   implied.

## Validation Scope

Day 3 did not modify `.c` or `.h` files, so the full C quality gate is not
required.
