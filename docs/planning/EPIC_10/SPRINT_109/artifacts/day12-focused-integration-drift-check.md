# Day 12 Focused Integration & Drift Check

## Purpose

Day 12 validates every Sprint 109 touched code, test, and build-system surface
before final closeout. It focuses on the Day 4 eigensolver source split and the
Day 11 QR giant-test cleanup, then verifies public API, install-header,
source-list, helper-target, and CTest registration drift.

## Touched Surfaces

Current touched implementation and build surfaces:

| Surface | Change Type |
|---|---|
| `src/sparse_eigs.c` | Removed the dense Jacobi implementation from the public eigensolver owner. |
| `src/sparse_eigs_dense_internal.c` | Added the private dense Jacobi implementation owner. |
| `Makefile` | Added the private eigensolver source to `LIB_SRCS`. |
| `CMakeLists.txt` | Added the private eigensolver source to the static library source list. |
| `build-metadata/library_sources.txt` | Added the private eigensolver source to the reviewed manifest. |
| `tests/test_qr.c` | Added one local exact-RHS setup helper and replaced seven repeated setup blocks. |

No matrix-shell source, public header, private header, install rule, or CTest
target source changed.

## Focused Make Validation

Command:

```sh
make source-list-check && \
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg \
  build/test_sprint29_integration build/test_qr && \
./build/test_eigs && \
./build/test_eigs_thick_restart && \
./build/test_eigs_lobpcg && \
./build/test_sprint29_integration && \
./build/test_qr
```

Results:

| Check | Result | Count |
|---|---|---:|
| `make source-list-check` | Passed | 46 library sources |
| `test_eigs` | Passed | 31 tests |
| `test_eigs_thick_restart` | Passed | 21 tests |
| `test_eigs_lobpcg` | Passed | 27 tests |
| `test_sprint29_integration` | Passed | 3 tests |
| `test_qr` | Passed | 73 tests |

The focused Make lanes cover the moved dense Jacobi caller family,
thick-restart and LOBPCG Rayleigh-Ritz users, cross-feature eigensolver
refinement/progress behavior, and the Day 11 QR proof-owner cleanup.

## Focused CMake and CTest Validation

Configured a fresh ignored CMake inspection tree:

```sh
cmake -S . -B build/day12-cmake-ctest \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Built the touched CMake test targets:

```sh
cmake --build build/day12-cmake-ctest \
  --target test_eigs test_eigs_thick_restart test_eigs_lobpcg \
  test_sprint29_integration test_qr -j2
```

Ran focused CTest:

```sh
ctest --test-dir build/day12-cmake-ctest \
  -R "^(test_eigs|test_eigs_thick_restart|test_eigs_lobpcg|test_sprint29_integration|test_qr)$" \
  --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 5
```

Focused CTest registration:

| CTest # | Test | Result |
|---:|---|---|
| 20 | `test_qr` | Passed |
| 46 | `test_sprint29_integration` | Passed |
| 47 | `test_eigs` | Passed |
| 48 | `test_eigs_thick_restart` | Passed |
| 49 | `test_eigs_lobpcg` | Passed |

## CMake Registration and Helper-Target Drift

After building the full CMake default target in `build/day12-cmake-ctest`,
registration inspection reported:

```text
Total Tests: 54
```

The focused test registrations remain:

```text
Test #20: test_qr
Test #46: test_sprint29_integration
Test #47: test_eigs
Test #48: test_eigs_thick_restart
Test #49: test_eigs_lobpcg
```

No CMake test target was added, removed, renamed, or split. The Day 11 QR
helper remains a local static helper inside `tests/test_qr.c`; it does not add
a compiled helper library or shared helper header.

## Source-List Parity Evidence

`make source-list-check` passed with 46 library sources.

The only build-system diff is the same private source addition in all three
source-list owners:

```text
src/sparse_eigs_workspace_internal.c
src/sparse_eigs_dense_internal.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_thick_restart.c
src/sparse_eigs.c
```

Fresh CMake `compile_commands.json` includes:

```text
src/sparse_eigs_dense_internal.c
```

as a `sparse_lu_ortho` library compile unit.

## Public API and Install-Header Drift

Header drift command:

```sh
git diff --name-only -- include src/*.h tests/*.h
```

Result:

```text
<no output>
```

Install-surface evidence:

- `include/` has no diff.
- `src/*.h` has no diff.
- `tests/*.h` has no diff.
- `CMakeLists.txt` install rules have no diff.
- `Makefile` install header loop has no diff.

The new dense Jacobi owner is a private `.c` file and does not change any
installed header or public declaration.

## Matrix-Shell Drift

Matrix-shell drift command:

```sh
git diff --name-only -- src/sparse_matrix.c include/sparse_matrix.h \
  tests/test_sparse_io.c tests/test_sparse_matrix.c
```

Result:

```text
<no output>
```

This confirms Day 8 and Day 9 matrix-shell work remains documentation and
validation only; no matrix public behavior changed on this branch.

## Metrics

| Metric | Value |
|---|---:|
| `src/sparse_eigs.c` lines | 1412 |
| `src/sparse_eigs_dense_internal.c` lines | 129 |
| `tests/test_qr.c` lines | 3194 |
| `s21_dense_sym_jacobi` implementation owners | 1 |
| `make_qr_exact_rhs` call sites | 7 |
| library sources after split | 46 |
| CTest registrations | 54 |
| public/header diffs | 0 |
| new helper targets | 0 |

## Completion Criteria Status

- Every touched code, test, and build-system surface has a focused validation
  result.
- Make and CMake focused lanes pass.
- Source-list parity passes.
- Public API and install-header no-drift evidence is explicit.
- CTest registration remains at 54 tests.
- No helper-target or matrix-shell drift was introduced.
- No unresolved Day 12 validation gap remains before final closeout.
