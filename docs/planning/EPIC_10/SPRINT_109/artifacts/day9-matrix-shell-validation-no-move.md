# Day 9 Matrix Shell Validation & No-Move Decision

## Purpose

Day 9 validates the Day 8 matrix-shell candidate boundary against focused
public behavior and solver workflow expectations, then decides whether Sprint
109 should move matrix-shell code.

Day 9 moves no code.

## Focused Matrix Validation

Commands:

```sh
make build/test_sparse_io build/test_sparse_matrix build/test_known_matrices build/test_integration build/test_csr
./build/test_sparse_io
./build/test_sparse_matrix
./build/test_known_matrices
./build/test_integration
./build/test_csr
```

Results:

| Test | Result | Count |
|---|---|---:|
| `test_sparse_io` | Passed | 26 |
| `test_sparse_matrix` | Passed | 63 |
| `test_known_matrices` | Passed | 15 |
| `test_integration` | Passed | 58 |
| `test_csr` | Passed | 19 |

These tests cover Matrix Market round trips, parse failures, errno behavior,
permuted logical save behavior, duplicate entry handling, public lifecycle
lanes, compressed constructor entry paths, known fixtures, and file-backed
integration workflows.

## Representative Solver-Smoke Validation

Commands:

```sh
./build/test_sparse_lu
./build/test_cholesky
./build/test_ldlt
./build/test_iterative
./build/test_eigs
./build/test_svd
```

Results:

| Test | Result | Count |
|---|---|---:|
| `test_sparse_lu` | Passed | 40 |
| `test_cholesky` | Passed | 21 |
| `test_ldlt` | Passed | 89 |
| `test_iterative` | Passed | 80 |
| `test_eigs` | Passed | 31 |
| `test_svd` | Passed | 98 |

These smoke lanes prove that the current Matrix Market loader remains part of
solver-facing behavior, not just a standalone file parser.

## Solver Dependency Map

`sparse_load_mm` is used by many solver families and corpus tests, including:

| Family | Example Tests |
|---|---|
| Direct solvers | `test_cholesky`, `test_ldlt`, `test_direct_csc_dispatch`, `test_chol_csc_supernodal` |
| Iterative solvers | `test_iterative`, `test_bicgstab`, `test_minres` |
| Spectral and SVD | `test_eigs`, `test_svd`, partial-SVD helper tests |
| Graph and reordering | `test_graph`, `test_reorder_nd` |
| Integration and known fixtures | `test_integration`, `test_known_matrices`, `test_suitesparse` |

This confirms that a future Matrix Market split needs solver-smoke validation
in addition to focused I/O tests.

## Public API and Header Drift Check

Day 9 made no changes to:

- `include/sparse_matrix.h`;
- `src/sparse_matrix.c`;
- `src/sparse_matrix_internal.h`;
- `src/sparse_matrix_state_internal.h`;
- install-header membership;
- CTest registration;
- matrix test targets.

Current branch source-list changes are limited to the earlier Day 4
eigensolver extraction, not matrix-shell movement.

## Source-List and Private-Header Requirements

The Day 8 requirements remain valid for a future split:

- add `src/sparse_matrix_io.c` only with matching `Makefile`,
  `CMakeLists.txt`, and `build-metadata/library_sources.txt` updates;
- run `make source-list-check`;
- keep public declarations in `include/sparse_matrix.h` unchanged unless
  explicitly reviewed as public API work;
- keep Matrix Market helpers private;
- resolve `SparseBuildEntry` and `sparse_matrix_build_from_entries` before
  moving `sparse_load_mm`.

The builder dependency is the main blocker. `sparse_matrix_build_from_entries`
currently supports `sparse_copy`, `sparse_transpose`, and `sparse_load_mm`.
Moving only the loader would either expose that static helper prematurely or
force unrelated copy/transpose movement, both of which violate the Day 8 scope.

## Sprint 109 No-Move Decision

Do not move matrix-shell code in Sprint 109.

`src/sparse_matrix_io.c` remains the selected future owner candidate for Matrix
Market load/save behavior, but movement is deferred until a future sprint can
first resolve private bulk-builder ownership and add direct proof for the moved
owner.

## Downstream Contract

A future implementation sprint can consume this contract as follows:

1. Decide whether `SparseBuildEntry` and `sparse_matrix_build_from_entries`
   become a private builder owner.
2. If yes, move builder code with focused copy/transpose/load tests.
3. Then move Matrix Market load/save into `src/sparse_matrix_io.c`.
4. Preserve public API/install headers and reviewed CTest surfaces unless
   intentionally changed.
5. Run focused matrix tests, solver-smoke gates, source-list parity, and the
   full quality gate for code movement.

## Completion Criteria Status

- Matrix-shell contract is evidence-backed by focused tests.
- Public behavior remains the proof owner.
- No matrix-shell movement occurred without explicit low-risk evidence.
- Downstream planning can consume the selected `src/sparse_matrix_io.c`
  contract and the builder prerequisite.
