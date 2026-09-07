# Sprint 198 Day 6: Formula Render Validation

## Purpose

Validate the local Homebrew formula render contract without promoting guessed
license metadata or proceeding past the current root metadata blocker.

## Day 6 Disposition

Rendered-formula validation remains blocked before render execution because no
approved standalone root `LICENSE`, `COPYING`, or `NOTICE` exists. This is the
correct fail-fast behavior: a rendered formula would require an exact
Homebrew license identifier, and Day 2 found no approved source for that
identifier.

Day 6 therefore validates the render path by checking template syntax,
placeholder presence, render-time rejection behavior, tool availability, and
the current pre-render failure mode.

## Template Validation

| Check | Result | Evidence |
| --- | --- | --- |
| Ruby syntax | Pass | `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` reports `Syntax OK`. |
| Homepage placeholder | Present | `__SPARSE_HOMEBREW_HOMEPAGE__`. |
| Archive URL placeholder | Present | `__SPARSE_FORMULA_URL__`. |
| SHA-256 placeholder | Present | `__SPARSE_FORMULA_SHA256__`. |
| Version placeholder | Present | `__SPARSE_VERSION__`. |
| License placeholder | Present | `__SPARSE_HOMEBREW_LICENSE__`. |
| Static install expectation | Present | Template checks `libsparse_lu_ortho.a`. |
| Downstream CMake consumer | Present | Template uses exact-version `find_package(Sparse ...)` and links `Sparse::sparse_lu_ortho`. |

## Render Rejection Contract

The render helper in `scripts/homebrew_local_formula_proof.sh` rejects:

- missing environment replacements via `ENV.fetch`;
- empty replacements via `abort("empty replacement for ...")`;
- unresolved `__SPARSE_*__` placeholders after substitution.

The metadata detector rejects:

- missing root license metadata;
- empty `SPARSE_HOMEBREW_LICENSE`;
- placeholder-like license identifiers.

Because missing root metadata is checked before `SPARSE_HOMEBREW_LICENSE`
content, negative runs with `NOASSERTION` or `PLACEHOLDER` still stop at the
root metadata gate on this branch. Placeholder-license rejection remains a
guarded later-stage check once root metadata exists.

## Tool Availability

| Tool | Path |
| --- | --- |
| `brew` | `/usr/local/bin/brew` |
| `ruby` | `/usr/bin/ruby` |
| `cmake` | `/usr/local/bin/cmake` |
| `tar` | `/usr/bin/tar` |
| `shasum` | `/usr/bin/shasum` |

The current render blocker is not missing local tools.

## Command Results

| Command | Exit | Interpretation |
| --- | ---: | --- |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | 0 | Template syntax is valid. |
| `bash scripts/homebrew_local_formula_proof.sh` | 2 | Stops before archive/render/install/test because root metadata is absent. |
| `SPARSE_HOMEBREW_LICENSE=NOASSERTION bash scripts/homebrew_local_formula_proof.sh` | 2 | Still stops at missing root metadata before evaluating placeholder license value. |
| `SPARSE_HOMEBREW_LICENSE=PLACEHOLDER bash scripts/homebrew_local_formula_proof.sh` | 2 | Still stops at missing root metadata before evaluating placeholder license value. |

## Claim Boundary

Day 6 does not create placeholder-free rendered formula evidence. It confirms
that rendering is protected by metadata gates. Homebrew/core readiness,
bottles, Linuxbrew, public tap support, binary package distribution,
shared-library support, dynamic ABI support, and broad package-manager support
remain unclaimed.

## Day 7 Handoff

Day 7 should review installed-surface proof expectations while keeping install
execution blocked until render can proceed with approved root license metadata
and an exact Homebrew license identifier.

## Validation

Day 6 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
