# Day 6 CSR/CSC Construction Example

## Purpose

Day 6 implements the Day 5 decision to keep compressed-first construction in a
single small teaching example while covering both caller-owned CSR and
caller-owned CSC inputs. The example should reinforce `docs/solver_selection.md`
without introducing private headers, public builder claims, or solver behavior
changes.

## Touched Files

- `examples/example_compressed_input.c`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day6-csr-csc-construction-example.md`

## Implementation Summary

`examples/example_compressed_input.c` now demonstrates:

- building a public `SparseMatrix` shell from caller-owned CSR arrays through
  `sparse_from_csr(...)`;
- building a public `SparseMatrix` shell from caller-owned CSC arrays through
  `sparse_create_from_csc(...)`;
- mutating caller-owned CSR and CSC value arrays after construction to prove
  the returned matrices are independent copies;
- using the normal one-shot LU workflow after compressed construction;
- freeing returned matrices with `sparse_free(...)`.

The example stays scoped to public headers:

- `sparse_csr.h`
- `sparse_lu.h`
- `sparse_matrix.h`

No private matrix builder, Matrix I/O source owner, or internal storage header
is exposed.

## Public Workflow Boundary

| Workflow | Demonstrated API | Ownership Rule |
|---|---|---|
| CSR diagnostic import | `sparse_from_csr(...)` | Input arrays remain caller-owned; returned matrix is caller-owned. |
| CSC simple import | `sparse_create_from_csc(...)` | Input arrays remain caller-owned; returned matrix is caller-owned. |
| One-shot solve after import | `sparse_copy(...)`, `sparse_lu_factor(...)`, `sparse_lu_solve(...)` | The factorization mutates the working copy, not the imported original. |
| Cleanup | `sparse_free(...)` | Free returned matrices; do not free stack-owned compressed arrays. |

## Validation Plan

Required validation for this example-source day:

- `make examples`
- `./build/example_compressed_input`
- `git diff --check`
- trailing-whitespace scan over touched docs and example source

## Completion Criteria Status

- CSR construction remains covered by a copyable example.
- CSC construction now has a compact public example path.
- Memory ownership is clear for both compressed inputs.
- The example uses only public headers.
- The example matches the Day 4 solver-selection guide recommendation to use
  compressed-first constructors when input already exists as CSR or CSC.
