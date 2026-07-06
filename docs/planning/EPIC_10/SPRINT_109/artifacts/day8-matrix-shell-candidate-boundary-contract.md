# Day 8 Matrix Shell Candidate Boundary Contract

## Purpose

Day 8 chooses one future `src/sparse_matrix.c` public-behavior owner candidate
and documents the private dependencies, source-list requirements, focused
tests, and solver-smoke gates needed before any future matrix-shell extraction.

Day 8 moves no code.

## Starting Evidence

Sprint 108 already established that `src/sparse_matrix.c` is central public
matrix-shell territory, not simple line-count cleanup. Current size snapshot:

| File | Lines | Role |
|---|---:|---|
| `src/sparse_matrix.c` | 1359 | Mutable public matrix shell and compatibility implementation. |
| `include/sparse_matrix.h` | 614 | Public API and installed matrix contract. |
| `tests/test_sparse_io.c` | 511 | Primary Matrix Market load/save behavior tests. |
| `tests/test_sparse_matrix.c` | 1296 | Primary lifecycle, mutation, access, copy, transpose, and print behavior tests. |

## Responsibility Cluster Map

| Cluster | Representative Functions | Movement Risk |
|---|---|---|
| Lifecycle and shell allocation | `sparse_create`, `sparse_free`, shell buffer helpers, node pool helpers | High: allocation, OOM cleanup, identity permutations, factor-state initialization. |
| Copy and transpose | `sparse_copy`, `sparse_transpose` | High: uses bulk-entry builder, preserves factor/permutation state for copy, physical transpose semantics. |
| Mutation and access | `sparse_insert`, `sparse_remove`, `sparse_set`, `sparse_get`, `sparse_get_phys` | High: factor-state reset, cached norm invalidation, logical/physical index contracts. |
| Shape and properties | `sparse_rows`, `sparse_cols`, `sparse_nnz`, `sparse_memory_usage`, `sparse_is_symmetric`, `sparse_norminf` | Medium/high: public silent-zero behavior, atomic cached norm, solver preconditions. |
| Factor/permutation compatibility | `sparse_mark_factored`, permutation accessors, `sparse_reset_perms` | High: direct-solver compatibility and matrix-shell solve behavior. |
| Arithmetic and operators | `sparse_scale`, `sparse_add`, `sparse_add_inplace`, `sparse_matmul`, `sparse_matvec`, `sparse_matvec_block` | High: solver smoke dependency, OpenMP behavior, logical permutation semantics. |
| Matrix Market I/O | `sparse_save_mm`, `sparse_load_mm` | Candidate: public behavior is narrow enough to contract for a later split. |
| Print/debug helpers | `sparse_print_dense`, `sparse_print_entries`, `sparse_print_info` | Lower risk, but less valuable and best left out of the first split. |

## Selected Future Owner Candidate

Selected candidate:

```text
src/sparse_matrix_io.c
```

Scope:

- `sparse_save_mm`;
- `sparse_load_mm`;
- Matrix Market-specific parsing and formatting helpers;
- Matrix Market-specific errno and parse behavior.

Explicit exclusions:

- `sparse_print_dense`;
- `sparse_print_entries`;
- `sparse_print_info`;
- lifecycle and shell allocation;
- mutation/access helpers;
- copy/transpose behavior;
- arithmetic and matvec behavior;
- factor-state and permutation compatibility helpers.

Rationale: Matrix Market load/save is the narrowest valuable public-behavior
owner because it has a clear API pair, focused tests, and direct external file
format semantics. It is still not low-risk enough to move in Sprint 109 because
`sparse_load_mm` currently depends on static bulk-entry construction shared by
copy and transpose behavior.

## Private Dependency Plan

Future movement must resolve these private dependencies before code moves:

| Dependency | Current Owner | Requirement |
|---|---|---|
| `SparseBuildEntry` | `src/sparse_matrix.c` static typedef | Either move to a private builder owner or keep I/O in `src/sparse_matrix.c`. Do not expose publicly. |
| `sparse_matrix_build_from_entries` | `src/sparse_matrix.c` static helper | Required by `sparse_copy`, `sparse_transpose`, and `sparse_load_mm`; future I/O split needs a private declaration or a separate builder source. |
| logical permutation arrays | `SparseMatrix` private struct | `sparse_save_mm` writes logical order through `row_perm` and `inv_col_perm`; tests must guard this. |
| errno capture | `sparse_errno_internal.h` | Preserve `sparse_errno()` capture/reset behavior exactly. |
| checked stream writes | `sparse_stream_printf_checked` helpers | May move with I/O if kept private to source files; avoid installed headers. |
| allocation overflow helpers | `sparse_alloc_internal.h` and checked size helpers | Preserve parse and allocation failure paths. |

Recommended future sequencing:

1. First decide whether a private `src/sparse_matrix_build_internal.c` owner is
   warranted for `SparseBuildEntry` and `sparse_matrix_build_from_entries`.
2. Only after that, move Matrix Market load/save into
   `src/sparse_matrix_io.c`.
3. Keep all declarations private or existing public declarations in
   `include/sparse_matrix.h`; do not add an installed helper header.

## Source-List Requirements

If a later sprint adds `src/sparse_matrix_io.c`, update all source-membership
owners in the same change:

- `Makefile` `LIB_SRCS`;
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`;
- `build-metadata/library_sources.txt`.

Then run:

```sh
make source-list-check
```

The new source should be ordered immediately after `src/sparse_matrix.c` unless
a private builder source is added first, in which case the builder should sit
between the shell owner and I/O owner.

## Focused Public Behavior Tests

Required focused tests for a future Matrix Market split:

```sh
make build/test_sparse_io build/test_sparse_matrix build/test_known_matrices build/test_integration build/test_csr
./build/test_sparse_io
./build/test_sparse_matrix
./build/test_known_matrices
./build/test_integration
./build/test_csr
```

Coverage expectations:

| Test | Required Coverage |
|---|---|
| `test_sparse_io` | round trips, rectangular matrices, precision, symmetric and pattern files, parse failures, errno capture/reset, invalid paths, empty/1x1/negative values, save after permutation. |
| `test_sparse_matrix` | duplicate Matrix Market load last-write-wins, print helper separation, matrix public behavior remains intact. |
| `test_known_matrices` | file-backed known fixtures still load and solve. |
| `test_integration` | file-backed integration workflows and public lifecycle refactor lanes remain intact. |
| `test_csr` | SuiteSparse fixtures loaded through Matrix Market still feed compressed constructor/solver smoke. |

## Solver-Smoke Gates

Because `sparse_load_mm` is used throughout the solver suite, a future I/O move
must include smoke validation beyond `test_sparse_io`:

- direct solver fixture load smoke:
  `test_sparse_lu`, `test_cholesky`, `test_ldlt`;
- iterative fixture load smoke:
  `test_iterative`, `test_bicgstab`, `test_minres`;
- spectral and SVD fixture load smoke:
  `test_eigs`, `test_svd`;
- graph/reorder fixture load smoke if SuiteSparse path handling changes:
  `test_graph`, `test_reorder_nd`.

For a code move, the branch must still run:

```sh
make format && make lint && make test
git diff --check
```

## No-Drift Requirements

Future Matrix Market movement must not change:

- `include/sparse_matrix.h` declarations or installed header membership unless
  reviewed as a public API change;
- Matrix Market accepted formats: coordinate real/integer/pattern and
  general/symmetric;
- 1-based input coordinate conversion;
- symmetric mirror expansion;
- pattern value default `1.0`;
- logical-order save behavior after permutations;
- duplicate entry last-write behavior through the bulk builder;
- `sparse_errno()` capture/reset behavior;
- CTest registration or reviewed test counts.

## Sprint 109 Decision

Sprint 109 selects `src/sparse_matrix_io.c` as the future candidate owner for
Matrix Market load/save behavior, but does not move matrix-shell code.

The immediate follow-up is Day 9 validation and no-move proof against the
selected contract.

## Completion Criteria Status

- One future matrix-shell owner is selected and bounded.
- Private-header and builder dependencies are documented.
- Source-list requirements are explicit.
- Focused public behavior tests and solver-smoke gates are explicit.
- No matrix-shell code moved without independent low-risk evidence.
