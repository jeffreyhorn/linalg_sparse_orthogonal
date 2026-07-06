# Day 13 Matrix Shell Public-Behavior Review

## Purpose

Day 13 reviews `src/sparse_matrix.c` before any future matrix-shell extraction.
The goal is to make public behavior, private-header dependencies, source-list
requirements, and validation guardrails explicit. No source movement is
performed on Day 13.

## Current Ownership Snapshot

| Owner | Lines | Current Role |
|---|---:|---|
| `src/sparse_matrix.c` | 1,359 | Public matrix lifecycle, mutation, access, arithmetic, matvec, Matrix Market I/O, permutation compatibility, factor-state reset behavior, memory/norm reporting, and print helpers. |
| `include/sparse_matrix.h` | 614 | Public API and installed matrix-shell contract. |
| `src/sparse_matrix_internal.h` | 251 | Private `SparseMatrix`, node pool, permutation arrays, factor-state hooks, tolerance helpers, and graph adjacency declarations. |
| `src/sparse_matrix_state_internal.h` | 58 | Private inline compatibility predicates for identity permutations and factored-state requirements. |

The file size is not the primary risk. The central issue is that
`src/sparse_matrix.c` concentrates observable compatibility behavior used by
nearly every solver family.

## Public Behavior Inventory

Future matrix-shell extraction must preserve the following public behavior:

- lifecycle:
  - `sparse_create`;
  - `sparse_free`;
  - `sparse_copy`;
  - `sparse_transpose`;
- mutation:
  - `sparse_insert`;
  - `sparse_remove`;
  - `sparse_set`;
  - `sparse_scale`;
  - `sparse_add_inplace`;
- access and shape:
  - `sparse_get`;
  - `sparse_get_phys`;
  - `sparse_rows`;
  - `sparse_cols`;
  - `sparse_nnz`;
  - `sparse_memory_usage`;
- matrix properties and cached state:
  - `sparse_is_symmetric`;
  - `sparse_norminf`;
- factor and permutation compatibility:
  - `sparse_mark_factored`;
  - `sparse_row_perm`;
  - `sparse_col_perm`;
  - `sparse_inv_row_perm`;
  - `sparse_inv_col_perm`;
  - `sparse_reset_perms`;
- arithmetic and operator paths:
  - `sparse_add`;
  - `sparse_matmul`;
  - `sparse_matvec`;
  - `sparse_matvec_block`;
- Matrix Market and diagnostics:
  - `sparse_save_mm`;
  - `sparse_load_mm`;
  - `sparse_print_dense`;
  - `sparse_print_entries`;
  - `sparse_print_info`.

The most sensitive behaviors are logical-to-physical permutation semantics,
factor-state clearing after mutation, silent-zero `sparse_get` compatibility,
cached norm invalidation, Matrix Market parsing/error behavior, and solver
entry compatibility after compressed construction.

## Private Dependency Map

`src/sparse_matrix_internal.h` and `src/sparse_matrix_state_internal.h` are
used beyond `src/sparse_matrix.c`. Current dependency owners include:

- direct and incomplete factorization:
  - `src/sparse_lu.c`;
  - `src/sparse_lu_csr.c`;
  - `src/sparse_cholesky.c`;
  - `src/sparse_chol_csc.c`;
  - `src/sparse_ldlt.c`;
  - `src/sparse_ilu.c`;
  - `src/sparse_ic.c`;
- iterative and spectral paths:
  - `src/sparse_iterative.c`;
  - `src/sparse_iterative_block.c`;
  - `src/sparse_iterative_minres.c`;
  - `src/sparse_svd.c`;
  - `src/sparse_svd_partial.c`;
  - `src/sparse_bidiag.c`;
- dense, compressed, graph, and reorder support:
  - `src/sparse_dense.c`;
  - `src/sparse_csr.c`;
  - `src/sparse_graph_core.c`;
  - `src/sparse_reorder.c`;
  - `src/sparse_reorder_amd_qg.c`;
  - `src/sparse_analysis.c`;
- private state and analysis owners:
  - `src/sparse_factor_state_internal.c`;
  - `src/sparse_analysis_internal.h`;
  - `src/sparse_bicgstab_internal.h`;
  - `src/sparse_colamd_internal.h`;
  - `src/sparse_chol_csc_internal.h`;
  - `src/sparse_ldlt_csc_internal.h`;
- tests that intentionally inspect private layout:
  - `tests/test_etree.c`;
  - `tests/test_lu_csr.c`;
  - `tests/test_ic.c`.

This dependency map means a future source split should avoid widening private
header coupling. If a helper moves, the helper should remain private and should
not create a new installed header.

## Candidate Split Areas

| Candidate Area | Public Behavior Risk | Future Preconditions | Day 13 Decision |
|---|---|---|---|
| Shell allocation and node-pool helpers | Lifecycle, memory accounting, OOM cleanup, row/column header initialization. | Focused lifecycle/OOM review and `test_sparse_matrix` coverage. | Defer. |
| Bulk entry builder | Duplicate handling, zero drop, sort order, bounds checks, CSR/CSC import, copy, transpose, Matrix Market load. | Compressed constructor and Matrix Market guardrails plus source-list parity. | Defer. |
| Matrix Market I/O | File format, 1-based coordinates, symmetric/pattern expansion, errno, parse failures, logical order. | `test_sparse_io`, known matrix fixtures, and save/load round trips. | Defer. |
| Arithmetic/matvec owner | Logical permutation semantics, factor-state invalidation, cache invalidation, OpenMP behavior. | `test_sparse_arith`, `test_matmul`, `test_sparse_matrix`, and solver smoke tests. | Defer. |
| Factor/permutation compatibility owner | `sparse_copy`, `sparse_mark_factored`, `sparse_reset_perms`, direct-solver compatibility, row/column permutation accessors. | Direct solver, reorder, Cholesky/LU, and compatibility tests. | Defer. |
| Print/debug helpers | Output formatting and logical ordering. | Print-specific smoke tests if behavior moves. | Lowest risk, but not valuable enough for Sprint 108. |

No candidate is approved for Sprint 108 because the current workstream is a
public-behavior review, not a code movement day.

## Build-System Requirements For Future Extraction

Any future matrix-shell source split must update all source-membership owners:

- `Makefile` `LIB_SRCS`;
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`;
- `build-metadata/library_sources.txt`.

The future extraction must then run:

```sh
make source-list-check
```

If the split changes public headers, install headers, CTest registration, or
reviewed CI test counts, that change needs a separate public-surface review
before merge.

## Validation Guardrails

Focused validation should match the moved behavior:

| Changed Area | Required Focused Tests |
|---|---|
| Lifecycle, mutation, access, copy, transpose, memory, norm, get semantics | `make build/test_sparse_matrix && ./build/test_sparse_matrix` |
| Matrix Market load/save or print order | `make build/test_sparse_io && ./build/test_sparse_io` |
| CSR/CSC constructor bridge into public shell | `make build/test_csr && ./build/test_csr` |
| Scale/add/add-in-place behavior or norm invalidation | `make build/test_sparse_arith && ./build/test_sparse_arith` |
| Matrix multiply behavior | `make build/test_matmul && ./build/test_matmul` |
| Permutation or factor-state compatibility | `make build/test_reorder build/test_sparse_lu build/test_cholesky && ./build/test_reorder && ./build/test_sparse_lu && ./build/test_cholesky` |
| Matvec semantics used by solver families | Add iterative/eigensolver/SVD smoke tests appropriate to the touched path. |

For any code, header, build, or source-list change, the broad required gate is:

```sh
make format && make lint && make test
git diff --check
```

## Extraction Prerequisites

Do not extract from `src/sparse_matrix.c` until a future PR has:

1. one named candidate area and no unrelated cleanup;
2. an explicit public behavior inventory for the moved functions;
3. a private-header dependency plan that does not widen installed API surface;
4. Make/CMake/manifest source-list parity in the same change;
5. focused tests that observe the exact moved behavior;
6. solver smoke tests if permutation, factor-state, matvec, or compressed
   constructor behavior changes;
7. a reviewed no-change statement for public API, install headers, CTest
   registration, and reviewed test counts unless those surfaces are
   intentionally changed.

## Day 13 Decision

Keep `src/sparse_matrix.c` intact for Sprint 108. Treat it as central public
matrix-shell behavior rather than simple maintainability debt. Future work
should start with one narrow candidate and the validation plan above, then
move code only after source-list parity and public-behavior guardrails are in
place.

## Completion Criteria Status

- Public behavior and compatibility constraints are explicit.
- Private-header and downstream dependency surfaces are mapped.
- Future shell extraction prerequisites are named.
- Validation guardrails are tied to candidate areas.
- No central matrix source move landed without a guardrail plan.
