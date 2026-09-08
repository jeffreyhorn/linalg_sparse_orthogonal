# Sprint 198 Day 4: Formula Metadata Wiring

## Purpose

Review and document the Homebrew formula metadata wiring path after Day 2 and
Day 3 established that approved license inputs are not available.

## Day 4 Disposition

Formula license metadata is not promoted on Day 4 because no approved
standalone root license metadata or exact Homebrew formula license identifier
exists. The correct implementation state remains guarded:

- `packaging/homebrew/sparse-lu-ortho.rb.in` keeps
  `license "__SPARSE_HOMEBREW_LICENSE__"`;
- `scripts/homebrew_local_formula_proof.sh` requires
  `__SPARSE_HOMEBREW_LICENSE__` to be present in the template;
- render-time replacement aborts on empty values;
- render-time replacement aborts on unresolved `__SPARSE_*__` placeholders;
- license detection exits `2` before archive/render/install/test work when no
  standalone root metadata exists;
- placeholder license values remain unavailable blocker evidence.

## Formula Metadata Contract

| Field | Template source | Day 4 status |
| --- | --- | --- |
| Homepage | `__SPARSE_HOMEBREW_HOMEPAGE__` | Guarded placeholder; substituted only during temporary render. |
| Source archive URL | `__SPARSE_FORMULA_URL__` | Guarded placeholder; generated from temporary local archive. |
| SHA-256 | `__SPARSE_FORMULA_SHA256__` | Guarded placeholder; generated from local archive checksum. |
| Version | `__SPARSE_VERSION__` | Guarded placeholder; sourced from `VERSION`. |
| License | `__SPARSE_HOMEBREW_LICENSE__` | Guarded placeholder; blocked until approved metadata and exact identifier exist. |
| Build dependency | `depends_on "cmake" => :build` | Static local source proof dependency only. |
| Installed artifact | `libsparse_lu_ortho.a` | Static archive proof only. |
| Downstream test | exact-version `find_package(Sparse ...)` and `Sparse::sparse_lu_ortho` | Installed CMake consumer proof path remains present. |

## Placeholder Rejection

The proof script rejects invalid formula metadata before it can be interpreted
as support evidence:

- missing root `LICENSE`, `COPYING`, or `NOTICE` exits `2`;
- empty `SPARSE_HOMEBREW_LICENSE` exits `2`;
- `NOASSERTION`, `UNKNOWN`, `TBD`, `TODO`, `FIXME`, `PLACEHOLDER`, unresolved
  template text, and case variants containing `placeholder` exit `2`;
- unresolved `__SPARSE_*__` tokens in the rendered formula abort rendering.

## Static Proof Boundary

The formula template remains scoped to a temporary local static source proof.
It installs through CMake, verifies `libsparse_lu_ortho.a`, rejects shared
artifacts, and uses `brew test` to build a downstream CMake consumer from the
installed package surface. This is not a Homebrew/core, bottle, Linuxbrew,
public tap, binary package, shared-library, dynamic ABI, or broad
package-manager support claim.

## Day 5 Handoff

Day 5 should review archive and checksum proof behavior. If approved license
metadata is still unavailable, Day 5 should keep archive creation blocked and
document the exact archive entries expected once approval exists.

## Validation

Day 4 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
