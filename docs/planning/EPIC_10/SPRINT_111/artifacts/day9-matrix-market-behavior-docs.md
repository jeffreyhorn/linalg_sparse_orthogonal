# Day 9 Matrix Market Behavior Documentation

## Purpose

Day 9 updates the public Matrix Market documentation so users can understand
ownership, duplicate handling, zero behavior, pattern defaults, symmetric
expansion, and error behavior without reading source, tests, or sprint
artifacts. The wording must stay inside the existing public API boundary:
`sparse_load_mm(...)` and `sparse_save_mm(...)` are public functions, but there
is no public Matrix I/O module and no public Matrix builder API.

## Touched Files

- `docs/matrix_market.md`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day9-matrix-market-behavior-docs.md`

## Source-of-Truth Inputs

- `include/sparse_matrix.h`
- `src/sparse_matrix_io.c`
- `src/sparse_matrix_build_internal.c`
- `tests/test_sparse_io.c`
- `tests/test_sparse_matrix.c`
- `examples/example_matrix_market.c`

## Documented Behavior

| Behavior | Documentation Result |
|---|---|
| Public surface | Matrix Market support is exposed through `sparse_load_mm(...)` and `sparse_save_mm(...)`, not a public module. |
| Loaded matrix ownership | Successful load returns a caller-owned `SparseMatrix *` freed with `sparse_free(...)`; failed load leaves `*mat_out` as `NULL`. |
| Save format | Save writes coordinate real general with full double precision through `%.15g`. |
| Pattern input | Pattern entries have no value field and load as `1.0`. |
| Integer input | Integer values are read and stored as `double`. |
| Symmetric input | Off-diagonal entries are mirrored; symmetric inputs must be square. |
| Duplicate coordinates | Last entry for a coordinate in file order wins. |
| Final zero | If the winning value for a coordinate is `0.0`, the coordinate is omitted from stored sparse entries. |
| I/O errors | `SPARSE_ERR_IO` captures system errno for `sparse_errno()`. |
| Parse errors | Unsupported/malformed Matrix Market inputs return `SPARSE_ERR_PARSE`. |
| Success errno state | Successful load/save resets `sparse_errno()` to `0`. |

## Public/Private Boundary

The public docs intentionally avoid these claims:

- no public Matrix I/O module;
- no public builder API;
- no private source-owner names;
- no guarantee that Matrix Market parsing is a benchmark or proof surface;
- no unsupported dense array, complex, skew-symmetric, or Hermitian support.

## Validation

Day 9 changed documentation only, so validation is:

- `git diff --check`
- trailing-whitespace scan over touched docs

## Completion Criteria Status

- Matrix Market behavior matches the Sprint 110 implementation and existing
  regression tests.
- Docs avoid public builder and public Matrix I/O module claims.
- Behavior details are discoverable from `docs/matrix_market.md`.
- Wording is consistent with public headers and `example_matrix_market`.
