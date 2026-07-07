# Day 4 Matrix Market Boundary Plan

## Purpose

Day 4 prepares Matrix Market source movement after the Day 3 builder ownership
decision. The plan is intentionally builder-first: Matrix Market load/save can
move only after the shared bulk-entry builder has a private owner with
source-list parity and focused validation.

No Matrix Market code moves on Day 4. This artifact is the implementation
checklist for Day 5 and the validation checklist for Day 6.

## Source Inputs

- `docs/planning/EPIC_10/SPRINT_110/PLAN.md`, Day 4.
- `docs/planning/EPIC_10/SPRINT_110/artifacts/day3-matrix-builder-ownership-decision.md`.
- `src/sparse_matrix.c`.
- `src/sparse_matrix_internal.h`.
- `src/sparse_alloc_internal.h`.
- `src/sparse_errno_internal.h`.
- `include/sparse_matrix.h`.
- `Makefile`.
- `CMakeLists.txt`.
- `build-metadata/library_sources.txt`.
- `tests/test_sparse_matrix.c`.
- `tests/test_sparse_io.c`.
- `tests/test_csr.c`.
- `tests/test_integration.c`.
- `tests/test_suitesparse.c`.
- `tests/test_qr.c`.

## Movement Decision

Proceed with a planned split in this order:

1. Move the shared Matrix builder seam to a private builder source:
   - `src/sparse_matrix_build_internal.c`
2. Move Matrix Market load/save behavior to a private Matrix I/O source:
   - `src/sparse_matrix_io.c`

The builder source must land first. If the builder movement fails validation,
Matrix Market movement must be deferred rather than duplicated or exposed
publicly.

## Matrix Market Dependency Map

| Dependency | Current Location | Matrix I/O Requirement |
|---|---|---|
| `sparse_save_mm` | `src/sparse_matrix.c` | Move to `src/sparse_matrix_io.c` only after private traversal dependencies are available. |
| `sparse_load_mm` | `src/sparse_matrix.c` | Move to `src/sparse_matrix_io.c` only after the private builder owner exists. |
| `SparseBuildEntry` | `src/sparse_matrix.c` static typedef | Move to private builder source before `sparse_load_mm` moves. |
| `sparse_matrix_build_from_entries` | `src/sparse_matrix.c` static helper | Move to private builder source and expose only through internal declarations. |
| `sparse_stream_vprintf_checked` / `sparse_stream_printf_checked` | `src/sparse_matrix.c` static helpers | Either move with Matrix I/O or become private stream helpers; no public exposure. |
| `sparse_set_errno_` | `src/sparse_errno_internal.h` | Keep internal errno capture/reset behavior identical. |
| allocation helpers | `src/sparse_alloc_internal.h` | Preserve checked allocation, size conversion, and overflow behavior. |
| `SparseMatrix`, `Node`, row/column headers, logical permutations | `src/sparse_matrix_internal.h` | Needed by save traversal and builder construction; remains private. |
| public declarations | `include/sparse_matrix.h` | Stay unchanged; no new public API. |

## Source-List And Package Checklist

If Day 5 implements the split, update all reviewed source membership lists in
the same commit:

| File | Required Update |
|---|---|
| `Makefile` | Add `$(SRCDIR)/sparse_matrix_build_internal.c` and `$(SRCDIR)/sparse_matrix_io.c` after `$(SRCDIR)/sparse_matrix.c`. |
| `CMakeLists.txt` | Add `src/sparse_matrix_build_internal.c` and `src/sparse_matrix_io.c` after `src/sparse_matrix.c`. |
| `build-metadata/library_sources.txt` | Add both new sources after `src/sparse_matrix.c` in the same order. |

Required order:

1. `src/sparse_matrix.c`
2. `src/sparse_matrix_build_internal.c`
3. `src/sparse_matrix_io.c`
4. `src/sparse_factor_state_internal.c`

Run:

```sh
make source-list-check
```

## Focused Matrix Validation Plan

Build focused binaries directly through the existing pattern rule and run them:

```sh
make build/test_sparse_matrix build/test_sparse_io build/test_csr
build/test_sparse_matrix
build/test_sparse_io
build/test_csr
```

Coverage expected:

| Surface | Expected Coverage |
|---|---|
| `test_sparse_matrix` | copy, transpose, duplicate-last-write-wins, silent-zero behavior, matrix shell basics. |
| `test_sparse_io` | Matrix Market roundtrip, rectangular, precision, nnz, 1x1, empty, negative values, symmetric, pattern, bad input, errno capture/reset, permutation roundtrip. |
| `test_csr` | Adjacent CSR/CSC constructor behavior remains independent and unchanged. |

## Solver-Smoke Validation Plan

At least one loaded-matrix solver-smoke lane must run after the split. Prefer
two lanes because Matrix Market loaded data feeds both broad integration and a
solver-family proof surface:

```sh
make build/test_integration build/test_suitesparse build/test_qr
build/test_integration
build/test_suitesparse
build/test_qr
```

Coverage expected:

| Surface | Expected Coverage |
|---|---|
| `test_integration` | load-factor-solve-save workflow and all-reference-matrix integration. |
| `test_suitesparse` | loaded SuiteSparse matrices through direct solver paths. |
| `test_qr` | loaded matrices through QR solve, reconstruction, sparse/dense QR, and refinement lanes. |

## Full Quality Gate For Implementation

Because Day 5 is expected to touch `.c`, build-system, and source-list files,
the final implementation gate is:

```sh
make format && make lint && make test
```

Additional required checks:

```sh
git diff --check
rg -n "[ \t]+$" <touched docs and source files>
make source-list-check
```

## Day 5 Implementation Checklist

1. Create `src/sparse_matrix_build_internal.c`.
2. Move `SparseBuildEntry`, the comparator, and
   `sparse_matrix_build_from_entries` into the private builder source.
3. Add the narrowest internal declaration needed by `src/sparse_matrix.c` and
   future `src/sparse_matrix_io.c`.
4. Update `sparse_copy`, `sparse_transpose`, and `sparse_load_mm` references to
   use the private builder declaration.
5. Create `src/sparse_matrix_io.c`.
6. Move `sparse_save_mm` and `sparse_load_mm` into Matrix I/O source.
7. Decide whether checked stream-print helpers move with Matrix I/O or become a
   private shared helper; keep them internal either way.
8. Update Makefile, CMake, and `build-metadata/library_sources.txt` in the
   required order.
9. Run source-list and focused validation.
10. Run the full quality gate before committing any C/build changes.

## No-Drift Requirements

Day 5/6 must verify:

- no public function declarations change;
- no installed header changes occur;
- no helper target is added;
- no reviewed CTest registration count changes intentionally or accidentally;
- duplicate-last-write-wins behavior remains unchanged;
- final-zero-drop behavior remains unchanged;
- Matrix Market errno capture and clear-on-success behavior remains unchanged;
- copy permutation/factor-state cloning remains outside the builder helper;
- CSR/CSC constructor behavior remains unchanged.

## Fallback Plan

If the private builder source cannot compile or validate without broad matrix
shell exposure:

1. stop Matrix Market movement;
2. revert only the Day 5 implementation attempt before committing;
3. publish a no-split deferral artifact explaining the blocker;
4. keep all Matrix Market behavior in `src/sparse_matrix.c`.

## Completion Criteria Status

- Matrix Market movement is safely planned with builder-first sequencing.
- Every required build/source-list touchpoint is listed.
- Validation covers file I/O behavior and solver use after load.
- No Matrix Market code has moved on Day 4.
- Day 5 has an explicit implementation checklist and fallback path.
