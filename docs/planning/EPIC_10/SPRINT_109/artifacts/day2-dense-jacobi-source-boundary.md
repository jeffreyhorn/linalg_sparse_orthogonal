# Day 2 Dense Jacobi Source Boundary

## Purpose

Day 2 revalidates `s21_dense_sym_jacobi` as the only Sprint 109 candidate for
an eigensolver source move. The goal is to decide whether the candidate is
narrow enough to prepare for extraction while preserving public eigensolver
behavior, build-system/source-list parity, and reviewed test surfaces.

This artifact does not move code. It records the live dependency map, proposed
private source owner, build-system update checklist, focused validation plan,
and go/no-go criteria required before Day 4 can extract anything.

## Live Symbol Location

| Symbol | Current Owner | Declaration | Direct Callers |
|---|---|---|---|
| `s21_dense_sym_jacobi` | `src/sparse_eigs.c` | `src/sparse_eigs_internal.h` | `src/sparse_eigs_thick_restart.c`, `src/sparse_eigs_lobpcg.c` |

The function currently lives near the end of `src/sparse_eigs.c`, after the
public/workspace implementation wrapper and before the LOBPCG commentary block.
It is already declared in the private eigensolver header, so extraction does
not require a public declaration.

## Function Contract

`s21_dense_sym_jacobi` currently:

- accepts a dense symmetric `K x K` matrix in column-major `A_scratch`;
- destroys `A_scratch` while diagonalizing it;
- writes ascending eigenvalues to `theta_out`;
- writes matching orthonormal eigenvectors as columns of `Q_out`;
- returns `SPARSE_ERR_NULL` for null buffers;
- returns `SPARSE_ERR_BADARG` for `K < 1`;
- handles `K == 1` without sweeps;
- uses only local dense arithmetic, `idx_t`, `sparse_err_t`, `SPARSE_OK`,
  `SPARSE_ERR_NULL`, `SPARSE_ERR_BADARG`, `sqrt`, and `fabs`.

The helper does not use sparse matrix storage, public eigensolver options,
workspace storage, shift-invert state, backend dispatch, or public result
structures.

## Direct Caller Map

| Caller | File | Role | Validation Implication |
|---|---|---|---|
| `s21_dense_sym_jacobi(T_arrow, K, theta_arrow, Y_arrow)` | `src/sparse_eigs_thick_restart.c` | Dense eigensolve over the thick-restart arrowhead Ritz problem. | Must run `test_eigs_thick_restart`; also keep `test_sprint29_integration` because cross-feature eigensolver/refinement coverage depends on maintained spectral behavior. |
| `s21_dense_sym_jacobi(G, K_eff, theta_full, Y)` | `src/sparse_eigs_lobpcg.c` | Dense Rayleigh-Ritz solve inside LOBPCG. | Must run `test_eigs_lobpcg`; also run `test_eigs` to protect public/default eigensolver behavior. |

No direct call remains in `src/sparse_eigs.c`; after extraction,
`src/sparse_eigs.c` can stop owning the implementation while the private
declaration remains shared.

## Dependency Map

| Dependency | Source | Extraction Handling |
|---|---|---|
| `idx_t` | `include/sparse_types.h` via private/internal include chain | Include `sparse_eigs_internal.h` from the new private source, matching current internal source style. |
| `sparse_err_t`, `SPARSE_OK`, `SPARSE_ERR_NULL`, `SPARSE_ERR_BADARG` | public error/type definitions included by internal header path | Keep existing private declaration; no public header change. |
| `sqrt`, `fabs` | `<math.h>` | New private source must include `<math.h>` or inherit it through an explicit local include. Prefer explicit include. |
| column-major dense buffer convention | function contract only | Preserve comments and tests; do not rename or alter signature during extraction. |
| ascending sort and Q-column permutation | function body only | Move exactly with function body; no helper split inside the move. |

No dependency on:

- `SparseMatrix`;
- `sparse_eigs_opts_t`;
- public eigensolver result structs;
- eigensolver handle/workspace storage;
- LDLT shift-invert setup;
- backend dispatch/default selection.

## Proposed Private Source Owner

Recommended new file:

```text
src/sparse_eigs_dense_internal.c
```

Recommended declaration location:

```text
src/sparse_eigs_internal.h
```

Rationale:

- the symbol is already a private eigensolver helper shared by two backend
  owners;
- adding a narrower private header would create more churn than value for a
  single helper;
- the file name states the dense spectral helper ownership without implying a
  public API or generic dense-matrix package;
- keeping the same declaration avoids public header and install-header drift.

## Build-System and Source-List Update Checklist

If Day 4 extracts the helper, update all source membership owners in the same
change:

| Surface | Required Update |
|---|---|
| `Makefile` | Add `$(SRCDIR)/sparse_eigs_dense_internal.c` near existing eigensolver internal sources in `LIB_SRCS`. |
| `CMakeLists.txt` | Add `src/sparse_eigs_dense_internal.c` near existing eigensolver sources in `add_library(sparse_lu_ortho STATIC ...)`. |
| `build-metadata/library_sources.txt` | Add `src/sparse_eigs_dense_internal.c` in the same relative eigensolver source order. |
| public headers | No change. |
| install headers | No change. |
| helper targets | No change. |
| CTest registration | No change. |

Source-list parity gate:

```sh
make source-list-check
```

## Focused Validation Plan

Before broad quality gates, run the focused eigensolver lanes that exercise the
two direct callers and public spectral workflow:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_sprint29_integration
```

If source membership changes, also run:

```sh
make source-list-check
```

Because extracting the helper would touch `.c` and build/source-list files, the
required broad branch gate remains:

```sh
make format && make lint && make test
git diff --check
```

If reviewed CMake or CI test-count surfaces change unexpectedly, stop and
investigate before committing.

## Go Criteria

Day 4 may extract `s21_dense_sym_jacobi` only if all of these stay true:

- move exactly one function body and its local explanatory comment;
- keep the existing signature and private declaration;
- add only the private source file and required source-list/build membership;
- require no public header or install-header update;
- require no helper target or CTest registration change;
- preserve all direct callers unchanged except for linking to the moved symbol;
- focused eigensolver validation and source-list parity are available.

## No-Go Criteria

Do not extract on Day 4 if any of these occur:

- extraction requires moving grow-m, refinement, dispatch/defaults,
  handle/workspace glue, shift-invert, or shared Lanczos kernels;
- extraction requires a public API or install-header change;
- Makefile, CMake, and `build-metadata/library_sources.txt` cannot be updated
  together;
- focused validation cannot include thick-restart and LOBPCG;
- CTest registration or reviewed test counts drift unexpectedly;
- the change bundles dense Jacobi movement with unrelated eigensolver cleanup.

## Day 2 Decision

`s21_dense_sym_jacobi` remains the only approved Sprint 109 source-move
candidate. It is narrow enough to prepare for extraction because it has two
direct runtime callers, no direct public API ownership, and no sparse matrix or
workspace dependency. It is not behavior-free: it is shared by thick-restart and
LOBPCG Rayleigh-Ritz paths, so source-list parity and focused cross-backend
validation are mandatory before any code movement.

Day 2 approves planning for a Day 4 extraction attempt, not the extraction
itself. Day 3 must prepare source-list parity and validation harness details
before code moves.

## Completion Criteria Status

- The dense Jacobi location, declaration, direct callers, and dependencies are
  mapped from the live tree.
- The proposed private source owner and declaration strategy are defined.
- Build-system, source-list, public-header, install-header, helper-target, and
  CTest implications are explicit before edits begin.
- Focused validation and go/no-go criteria are recorded.
- No code moved on Day 2.
