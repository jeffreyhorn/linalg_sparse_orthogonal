# Day 12 Eigensolver Feasibility Closeout

## Purpose

Day 12 converts the Day 11 eigensolver source feasibility review into an
actionable future handoff. The closeout deliberately does not move
`src/sparse_eigs.c`; it records the exact requirements a future extraction PR
must satisfy before changing source ownership.

## Sprint 108 Decision

Sprint 108 lands no eigensolver source split.

The only Day 12 preparatory change is this documentation artifact and the
matching working-notes update. This preserves the current reviewed spectral
behavior while leaving a concrete checklist for a later narrow extraction.

No Day 12 change is made to:

- `.c` or `.h` files;
- `Makefile`;
- `CMakeLists.txt`;
- `build-metadata/library_sources.txt`;
- public API declarations;
- installed headers;
- helper targets;
- CTest registration;
- reviewed Linux, macOS, or Windows test counts.

## Future Extraction Checklist

A future eigensolver extraction PR should proceed in this order:

1. Select exactly one source-owner candidate.
2. Prefer the private dense helper seam:
   `s21_dense_sym_jacobi` into a private source such as
   `src/sparse_eigs_dense_internal.c`.
3. Keep the existing private declaration in `src/sparse_eigs_internal.h`
   unless a separate boundary review proves a narrower private header is
   worthwhile.
4. Update all source-membership owners in the same commit:
   - `Makefile` `LIB_SRCS`;
   - `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`;
   - `build-metadata/library_sources.txt`.
5. Run the source-list parity gate before focused tests:

```sh
make source-list-check
```

6. Run focused eigensolver validation:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_sprint29_integration
```

7. Run the required broad quality gate for code or build changes:

```sh
make format && make lint && make test
git diff --check
```

8. If the source split changes CMake registration, test registration, or
   reviewed CI surfaces, run the reviewed CMake and test-count checks before
   merge.

## Explicit Non-Claims

This closeout does not claim that `src/sparse_eigs.c` is ready for a broad
split. It only identifies the lowest-risk future seam.

This closeout does not claim that dense Jacobi movement is behavior-free. The
helper is shared by thick-restart and LOBPCG Rayleigh-Ritz paths, so movement
must be validated across those backends.

This closeout does not claim that grow-m, shift-invert, refinement, dispatch,
handle glue, or shared Lanczos kernels are extraction-ready. Those regions
remain behavior-sensitive.

## Residual Eigensolver Source Queue

| Priority | Work Item | Required Evidence Before Movement |
|---:|---|---|
| 1 | Private dense Jacobi helper owner | Make/CMake/manifest parity, `make source-list-check`, focused `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and `test_sprint29_integration` validation. |
| 2 | Grow-m refinement audit | Residual and oracle coverage for closed-form, shift-invert, refined eigenpairs, singular-shift retry, and `NEAREST_SIGMA` behavior. |
| 3 | Dispatch/defaults boundary audit | Backend-selection and public result semantics evidence that covers defaults, option normalization, convergence counts, and error paths. |
| 4 | Handle/workspace glue audit | Evidence that public handle preparation, reusable workspace behavior, and workspace-backed solve paths remain unchanged. |
| 5 | Shared Lanczos kernel audit | Cross-backend numerical evidence for grow-m, thick-restart, LOBPCG, and focused internal tests. |

## Future No-Go Conditions

Do not split eigensolver sources if any of the following are true:

- the source-list parity gate is not part of the change;
- the movement requires a public header or install-header change without a
  separate API review;
- the change alters CTest registration or reviewed test counts accidentally;
- focused validation omits thick-restart or LOBPCG when moving dense Jacobi;
- focused validation omits Sprint 29 integration when touching refinement,
  shift-invert, or public residual semantics;
- the change bundles dense helper movement with dispatch, grow-m, or
  refinement movement.

## Day 12 Validation Scope

Day 12 changes planning artifacts only. Required validation is documentation
hygiene:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_108
```

No `make format && make lint && make test` gate is required for Day 12 itself
because Day 12 does not modify code, headers, build files, or source-list
membership.

## Completion Criteria Status

- The eigensolver handoff is actionable without implying an unearned split.
- Source-list and build-system dependencies are explicit.
- Focused and broad future validation commands are recorded.
- Sprint 108 preserves current eigensolver public behavior and reviewed build
  surfaces.
