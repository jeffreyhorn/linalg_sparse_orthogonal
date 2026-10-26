# Day 1: Public ABI Width Inventory

**Date:** 2026-10-24
**Scope:** Sprint 219 item 219.1, first 10 of 20 planned hours
**Source revision:** `145b5a5a21dbb5191d5d1e1e2550b80c98819861`
**Host:** macOS x86_64, Apple Clang 17.0.0, static package 2.2.0

This is a pre-change inventory, not a claim that 64-bit installed consumers
work. The probes below inspect width identity only; no mismatched consumer
passes an index-bearing value, array, or struct to the archive. Day 2 owns
the enforcement design and negative-test contract.

## Type And Symbol Chain

`include/sparse_types.h:18-67` includes generated `sparse_version.h`, defaults
`SPARSE_IDX_BITS` to 32 when absent, maps 32 to `int32_t` and 64 to `int64_t`,
defines `IDX_MAX` and the `SPARSE_PRIDX`/`SPARSE_SCNIDX` format fragments,
and rejects other values. Its C11 static assertion checks only that the
*consumer's own* `idx_t` matches its own macro. The header tells callers to
rebuild with the library's width but does not encode the library's selection.
`include/sparse_version.h.in` contains version fields only; both build systems
generate and install it without a width value.

`src/sparse_types.c:10` implements `sparse_idx_bits()` as the library object's
compile-time `SPARSE_IDX_BITS`. This is a useful explicit comparison point,
but it is not called automatically before other API calls. The existing
`tests/test_sparse_matrix.c:442-454` checks `sizeof(idx_t)`, the macro, and
the library query in a same-build test. The archive exports ordinary C names
such as `sparse_create` and `sparse_idx_bits`; names do not identify index
width. The code also branches on `SPARSE_IDX_BITS` in
`src/sparse_ldlt_dense.c`, `src/sparse_dense.c`, and `src/sparse_vector.c`, so
the width selection is more than a header typedef.

## Public Boundary Map

Every public header with a direct `idx_t` occurrence is included below. The
last column names a candidate regression owner, not a passing test claim.

| Boundary / headers | Index-bearing exchange | Owner and candidate test |
| --- | --- | --- |
| Core types and progress: `sparse_types.h` | `idx_t`, `IDX_MAX`, format macros; `sparse_progress_t.step/total` cross the `sparse_progress_cb_t` callback boundary; `sparse_idx_bits()` reports the archive width. | `tests/test_sparse_matrix.c` width check; add installed width-query and callback-payload cases. |
| Linked matrix and IO: `sparse_matrix.h` | `SparseMatrix` is opaque, but create/insert/get/set/remove take `idx_t`; rows/cols/nnz return it; four permutation accessors return library-owned `const idx_t *`. Matrix Market load/save exchanges index values through the opaque matrix. | `tests/test_sparse_matrix.c`, `tests/test_sparse_io.c`; installed create/insert and permutation-accessor consumer. |
| Reorder and vector helpers: `sparse_reorder.h`, `sparse_vector.h` | Caller-owned permutation arrays, `sparse_bandwidth` return, and vector length arguments depend on width. | `tests/test_reorder.c`, `tests/test_sparse_vector.c`; installed permutation and vector-length smoke. |
| Compressed formats: `sparse_csr.h`, `sparse_lu_csr.h` | `SparseCsr`, `SparseCsc`, and `LuCsr` expose index counts and `idx_t *` offset/index arrays; `DenseBlock` exposes index bounds; conversion, pivot arrays, block counts, and dense-LU dimensions/pivots cross the boundary. | `tests/test_csr.c`, `tests/test_lu_csr.c`; installed CSR/CSC round trip and pivot-array case. Do not pass a mismatched struct. |
| Symbolic/numeric analysis: `sparse_analysis.h` | Reorder options contain `idx_t`; `sparse_symbolic_pub_t` exposes `col_ptr/row_idx/n/nnz`; `sparse_analysis_t` and `sparse_factors_t` contain index fields, permutations, and nested symbolic structures. | `tests/test_etree.c`, `tests/test_integration.c`; analysis/CSC consumer with exact struct layout. |
| Direct factors: `sparse_lu.h`, `sparse_ldlt.h`, `sparse_ilu.h`, `sparse_ic.h` | LU block RHS count; public LDLT/ILU structs carry index dimension/permutation fields, LDLT inertia outputs are `idx_t *`, ILUT options use `idx_t`, and IC/ILU preconditioner functions accept `idx_t n`. | `tests/test_sparse_lu.c`, `tests/test_ldlt.c`, `tests/test_ilu.c`, `tests/test_ic.c`; selected installed factor/solve and callback checks. |
| QR, SVD, bidiagonal, dense: `sparse_qr.h`, `sparse_svd.h`, `sparse_bidiag.h`, `sparse_dense.h` | Public QR/SVD/bidiagonal/dense structs contain index dimensions/ranks and QR permutation pointers; rank/nullity outputs, low-rank `k`, dense dimensions, and tridiagonal iteration counts cross the boundary. | `tests/test_qr.c`, `tests/test_svd.c`, `tests/test_bidiag.c`, `tests/test_dense.c`; selected rank and conversion cases. |
| Iterative and eigen: `sparse_iterative.h`, `sparse_eigs.h` | Iteration/options/result structs contain `idx_t`; `sparse_precond_fn` and `sparse_matvec_fn` receive `idx_t n`; progress callbacks receive structs containing it; repeated-run handles are caller-owned wrappers around opaque state; eigensolver options/results and `n/k` inputs depend on width. | `tests/test_iterative.c`, `tests/test_eigs.c`; installed callback-signature and selected solve case. |

`sparse_cholesky.h` has no direct `idx_t` token but its public operations
consume `SparseMatrix *` and are therefore still part of the package-width
usage boundary. `sparse_iter_handle_t` and `sparse_eigs_handle_t` expose a
`void *internal_state` wrapper rather than an index array; that does not make
their dimension-taking entry points width-independent. A pointer to opaque
`SparseMatrix` avoids caller-visible layout coupling, not the scalar index
argument/return coupling. Public layouts above require matching compilation
even when test matrices happen to use small indices.

## Current Build And Package Propagation

| Route | Build selection and installed files | Current consumer behavior / owner |
| --- | --- | --- |
| Make | `Makefile:5-10,58-61` has fixed C11 flags and no width definition. `make install` at `Makefile:1076-1097` copies the archive, source headers, generated version header, and `sparse.pc` to `PREFIX`; it does not record archive width. | `sparse.pc.in` emits only `-I${includedir}` in `Cflags`. `tests/test_install.sh` checks a basic pkg-config consumer and static artifact presence; extend it for both widths and mismatch rejection. |
| CMake | `CMakeLists.txt:45-56,110-165` builds the static archive without a width selector. Its PUBLIC definitions cover optional OpenMP/mutex, not `SPARSE_IDX_BITS`. `CMakeLists.txt:409-481` installs headers, archive, `SparseTargets.cmake`, config/version files, and `sparse.pc`. | The imported `Sparse::sparse_lu_ortho` target exports include/link requirements but no index-width compile definition. `tests/test_cmake_install.sh` and `examples/cmake_example` prove `find_package`; extend with two clean prefixes and a conflicting-width negative consumer. |
| Validator and hosted gates | `scripts/check_static_install_contract.py` validates the static file set, exact pkg-config fields, CMake target identity/location, and package version. `tests/test_static_install_contract.py` mutates those properties. | The validator's exact `Cflags: -I${includedir}` expectation would reject a future width flag until updated; it currently accepts both width-agnostic installed prefixes. Linux/macOS run Make and CMake install proofs; Windows reviews the CMake install route only (`.github/workflows/{ci,macos-ci,windows-ci}.yml`). |

The generated `SparseTargets.cmake` inspected in the fresh CMake prefix has
`INTERFACE_INCLUDE_DIRECTORIES` and `INTERFACE_LINK_LIBRARIES "m"`, with no
`INTERFACE_COMPILE_DEFINITIONS` for width. The installed `sparse.pc` from
either route has no width-bearing cflag. Thus a consumer can define
`SPARSE_IDX_BITS=64` independently while using a default 32-bit archive.

## Reproduced Baseline

All commands below were run from the repository root on the source revision
above. Build and install prefixes are under ignored `build/` directories.

```sh
make -j4 BUILDDIR=build/s219-day1-default build/s219-day1-default/libsparse_lu_ortho.a
make BUILDDIR=build/s219-day1-default build/s219-day1-default/test_sparse_matrix
build/s219-day1-default/test_sparse_matrix
make BUILDDIR=build/s219-day1-default PREFIX="$PWD/build/s219-day1-make-prefix" install
PKG_CONFIG_PATH="$PWD/build/s219-day1-make-prefix/lib/pkgconfig" pkg-config --cflags sparse
python3 scripts/check_static_install_contract.py build/s219-day1-make-prefix --mode make --platform unix --version 2.2.0
cmake -S . -B build/s219-day1-cmake -DCMAKE_INSTALL_PREFIX="$PWD/build/s219-day1-cmake-prefix"
cmake --build build/s219-day1-cmake --target sparse_lu_ortho --parallel 4
cmake --install build/s219-day1-cmake
python3 scripts/check_static_install_contract.py build/s219-day1-cmake-prefix --mode cmake --platform unix --version 2.2.0
```

The default matrix test passed 63/63 cases, including
`test_idx_width_contract`. Make and CMake package validators both returned
`static-install-contract: PASS`. The pkg-config cflags were only the install
include path. The Make archive and its installed copy had the same SHA-256
`bbd63a58dc9d4db4303dfda85c81727ab5d6a5ece496d50a98a3d6f04d210bdf`;
the separately built CMake archive had SHA-256
`8faae4c85098b3d6a394efbb0acd6d63608500c74a92dad63c80c72ef193ebff`.
These hashes identify this local build, not a portable binary contract.

For a safe mismatch probe, compile only a query of the consumer typedef and
the archive's width function. This command is intentionally **not** a solver
or struct-layout test:

```sh
printf '%s\n' '#include <sparse/sparse_types.h>' '#include <stdio.h>' \
  'int main(void) { printf("consumer=%zu library=%zu\n", sizeof(idx_t) * 8, sparse_idx_bits()); return 0; }' |
  cc -std=c11 -DSPARSE_IDX_BITS=64 \
    -I"$PWD/build/s219-day1-make-prefix/include" -x c - -x none \
    -L"$PWD/build/s219-day1-make-prefix/lib" -lsparse_lu_ortho -lm \
    -o build/s219-day1-default/installed_mismatch_probe
build/s219-day1-default/installed_mismatch_probe
```

Compilation/linking and execution returned 0, printing
`consumer=64 library=32`. The default-width control used the same source
without the override:

```sh
printf '%s\n' '#include "sparse_types.h"' '#include <stdio.h>' \
  'int main(void) { printf("consumer=%zu library=%zu\n", sizeof(idx_t) * 8, sparse_idx_bits()); return 0; }' |
  cc -std=c11 -Ibuild/s219-day1-default/include -Iinclude -x c - -x none \
    build/s219-day1-default/libsparse_lu_ortho.a -lm \
    -o build/s219-day1-default/width_default_probe
build/s219-day1-default/width_default_probe
```

It returned 0 with `consumer=32 library=32`. The actual installed mismatch
probe used pkg-config's emitted include and link flags; the expanded paths
above make its inputs explicit.
No index argument or layout-bearing object was passed across the mismatch.

The maintained CMake example also configured and **linked** against the
fresh 32-bit CMake prefix with a forced consumer flag:

```sh
cmake -S examples/cmake_example -B build/s219-day1-cmake-consumer-mismatch \
  -DCMAKE_PREFIX_PATH="$PWD/build/s219-day1-cmake-prefix" \
  -DCMAKE_C_FLAGS=-DSPARSE_IDX_BITS=64
cmake --build build/s219-day1-cmake-consumer-mismatch --parallel 2
```

Both commands returned 0. The resulting solver program was **not run**:
running it would cross the known mismatched index ABI and add no useful
evidence. This is a compile/link acceptance observation, not evidence of
correctness or safety.

## Risks And Next Gate

| Risk | Evidence | Day 2-10 test/design owner |
| --- | --- | --- |
| Silent mixed-width scalar calls | 64/32 query probe links and exits 0; ordinary symbols are width-neutral. | Define a check before unsafe use and explicit mismatch-negative consumers. |
| Exposed structs and arrays can be misread or overwritten | CSR/CSC, analysis, factor, QR/SVD, iterative/eigs layouts and callbacks contain `idx_t`. | Choose a package/compile guard and test a representative struct, pointer, and callback only in same-width builds. |
| Width is lost in generated metadata | Version header, CMake target, and pkg-config cflags omit width; validators still pass. | Make/CMake selection and export, generated identity, validator mutations. |
| Stale or cross-prefix artifacts can hide mistakes | Build systems currently allow ambient width defines; a stale archive can retain a different width. | Separate build/prefix trees and archive-identity checks. |
| 64-bit capability is unproven | The current fresh build and tests are default 32-bit; only a consumer typedef was forced to 64. | A real 64-bit library build plus selected solver/IO/conversion and hosted lane. |

The current evidence supports a default-width static build and a reproducible
enforcement gap. It does **not** support 64-bit installed-consumer correctness,
mixed-width use, shared libraries, or cross-version ABI stability.
