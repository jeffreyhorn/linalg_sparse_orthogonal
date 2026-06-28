# Sprint 94 Day 7 - Scalar-Family Widening Batch

## Scope
- Required center:
  - `include/sparse_types.h`
  - `src/sparse_matrix.c`
- Directly forced follow-through:
  - `src/sparse_matrix_internal.h`
  - `tests/test_sparse_matrix.c`
- Explicitly not widened in this batch:
  - broader solver-family implementation owners
  - wider index/ABI surfaces
  - broader maintainer or package wording

## Landed Changes
- Widened the touched matrix-shell internal storage seam:
  - `Node.value` now uses `sparse_scalar_t`
- Widened the touched matrix-shell build/import seam:
  - `make_node(...)` now takes `sparse_scalar_t`
  - `SparseBuildEntry.value` now uses `sparse_scalar_t`
  - the builder overwrite-selection path now keeps `sparse_scalar_t`
  - Matrix Market pattern defaults now enter through `sparse_scalar_t`
- Tightened the public scalar contract wording in `include/sparse_types.h` so
  the touched matrix-shell storage/build seam is described truthfully
- Added one focused proof extension:
  - `test_matrix_public_scalar_alias` now exercises `sparse_transpose(...)`
    so the widened builder path is covered through the public scalar alias
    surface

## Preserved Invariants
- The reviewed build remains real-only and still binds `sparse_scalar_t` to
  `double`
- Matrix-shell ordering, storage layout, and one-shot workflow semantics stay
  unchanged
- Wider index width interpretation remains unchanged
- No broad solver-family numeric widening was implied or claimed

## Validation
- `make format`
- `make lint`
- `make test`

## Result
- The first Sprint 94 scalar landing is now real on the touched matrix-shell
  owner rather than merely preparatory naming.
- The remaining capability work can now rerank from a validated
  post-landing baseline instead of from a still-`double` storage/build seam on
  the shared matrix-shell path.
