# Day 11 Eigensolver Source Feasibility Boundary

## Purpose

Day 11 maps the future `src/sparse_eigs.c` extraction boundary without landing
a risky source split. The goal is to preserve Sprint 103 spectral comparison
evidence, public eigensolver behavior, source-list parity, and reviewed build
surfaces while identifying the lowest-risk future seam.

## Source Ownership Snapshot

| Owner | Lines | Current Role |
|---|---:|---|
| `src/sparse_eigs.c` | 1,538 | Public front door, grow-m Lanczos, shared Lanczos helpers, dense Jacobi, shift-invert, refinement, dispatch, and handle glue. |
| `src/sparse_eigs_internal.h` | 631 | Private eigensolver declarations shared by focused tests and backend owners. |
| `src/sparse_eigs_workspace_internal.c` | 267 | Existing reusable eigensolver workspace storage owner. |
| `src/sparse_eigs_workspace_internal.h` | 82 | Private workspace owner/view contract. |
| `src/sparse_eigs_thick_restart.c` | 915 | Existing thick-restart backend owner. |
| `src/sparse_eigs_lobpcg.c` | 401 | Existing LOBPCG backend owner. |
| `include/sparse_eigs.h` | 651 | Public API and install-header contract. |

The largest backend bodies are already split. Remaining `src/sparse_eigs.c`
work is mostly shared orchestration, public behavior, or cross-backend kernel
logic.

## Build-System and Source-List Surface

Any future eigensolver source split must update these in the same reviewed
order:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`
- `build-metadata/library_sources.txt`

The parity checker is `scripts/check_library_sources.py`, exposed through:

```sh
make source-list-check
```

A future split must also preserve install-header boundaries. A new private
helper source should not create a public header or install-header change unless
a separate public API review approves it.

## Candidate Boundary Assessment

| Candidate | Representative Symbols | Consumers | Feasibility | Decision |
|---|---|---|---|---|
| Dense symmetric Jacobi helper | `s21_dense_sym_jacobi` | thick-restart arrowhead path, LOBPCG Rayleigh-Ritz path, focused internal tests | Medium; small, well-named, and isolated, but shared across backends and source-list membership. | Best future seam, but do not move on Day 11. |
| Grow-m Lanczos outer loop | `s46_run_growm_backend`, `s20_ritz_pairs`, `s20_select_indices`, `s20_lift_ritz_vectors` | default public eigensolver, shift-invert, grow-m tests, thick-restart shared helpers | High risk; public result semantics and residual reporting live here. | Defer. |
| Shift-invert and refinement | `s20_op_shift_invert`, `s29_refine_pair`, `s29_refine_eigenpairs`, `s29_maybe_refine` | `NEAREST_SIGMA`, LDLT setup, Sprint 29 integration, residual/refinement tests | High risk; crosses direct-solver setup and public residual behavior. | Defer. |
| Public dispatch/defaults | `s46_default_public_opts`, `s46_select_backend`, `s46_sparse_eigs_sym_impl`, `sparse_eigs_sym*` | public API, backend selection, handle reuse, reviewed CTest surface | High risk; this is the public behavior boundary. | Defer. |
| Handle/workspace glue | `sparse_eigs_handle_prepare`, `s49_eigs_handle_prepare_backend`, `sparse_eigs_sym_with_workspace_internal` | public handle API, workspace storage, dispatch path | Medium-high; storage is already split, glue is behavior-sensitive. | Defer. |
| Shared MGS/Lanczos kernels | `s21_mgs_reorth`, `lanczos_iterate`, `lanczos_iterate_op` | grow-m, thick-restart, tests, shared spectral kernels | High risk; numerical drift here affects every backend. | Defer. |

## Recommended Future Seam

The only plausible first split is a private dense spectral helper owner for
`s21_dense_sym_jacobi`, for example:

```text
src/sparse_eigs_dense_internal.c
```

Potential scope:

- move `s21_dense_sym_jacobi` only;
- keep declaration in `src/sparse_eigs_internal.h` unless a future boundary
  proves a narrower private header is worthwhile;
- update Makefile, CMake, and `build-metadata/library_sources.txt`;
- run source-list parity before focused tests;
- preserve all public headers and CTest registration.

This is not approved for Day 11. It is a future handoff candidate because even
this seam affects thick-restart and LOBPCG.

## Grow-M Refinement Boundary

Grow-m refinement is not a Day 11 split candidate.

Reasons:

- `NEAREST_SIGMA` setup owns an LDLT factorization of `A - sigma I`.
- refinement uses Rayleigh-quotient iteration, residual recomputation, and
  singular-shift retry behavior.
- `result->residual_norm`, `result->n_converged`, and partial convergence
  semantics are public behavior.
- Sprint 29 integration and Sprint 103 comparison artifacts rely on the current
  residual/refinement contract.

Future work should audit and document this region before any code movement,
then validate shift-invert, refinement, public dispatch, and Sprint 29
integration together.

## Cross-Backend Validation Plan

If a future split touches dense Jacobi, shared kernels, dispatch, refinement,
or source membership, run at least:

```sh
make source-list-check
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_sprint29_integration
make format && make lint && make test
git diff --check
```

For any CMake or reviewed test-count implication, also run the reviewed CMake
surface checks used by CI before closeout.

## Non-Changes for Day 11

Day 11 intentionally changes no:

- `.c` or `.h` files;
- Makefile or CMake source membership;
- `build-metadata/library_sources.txt`;
- public API or install-header surface;
- helper target;
- CTest registration or reviewed test count.

## Day 11 Decision

Do not split `src/sparse_eigs.c` on Day 11. Carry forward a future dense Jacobi
helper split candidate behind explicit source-list parity and cross-backend
validation. Carry grow-m refinement, dispatch, handle glue, and shared kernel
movement as deferred higher-risk work.

## Completion Criteria Status

- Eigensolver source ownership is inventoried.
- Dense Jacobi and grow-m refinement boundaries are evaluated.
- Build-system and source-list follow-through is mapped.
- Cross-backend validation is specified for any future split.
- No risky source extraction landed prematurely.
