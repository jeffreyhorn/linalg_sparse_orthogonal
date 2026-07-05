# Day 11 Eigensolver Source Boundary

## Purpose

Day 11 freezes the `src/sparse_eigs.c` source boundary before any Sprint 107
eigensolver extraction. The goal is to reduce maintainability risk without
breaking the Sprint 103 spectral comparison surfaces, reviewed dispatch
behavior, or build-system parity.

## File Snapshot

- `src/sparse_eigs.c`: 1,538 lines.
- Existing split owners:
  - `src/sparse_eigs_workspace_internal.c`
  - `src/sparse_eigs_thick_restart.c`
  - `src/sparse_eigs_lobpcg.c`
- Internal headers:
  - `src/sparse_eigs_internal.h`
  - `src/sparse_eigs_workspace_internal.h`
- Primary test owners:
  - `tests/test_eigs.c`
  - `tests/test_eigs_thick_restart.c`
  - `tests/test_eigs_lobpcg.c`
  - `tests/test_sprint29_integration.c`
  - backend dispatch and source-list parity tests that exercise reviewed
    eigensolver registration.

## Sprint 103 Comparison Surfaces To Protect

Sprint 103 added evidence around external comparison and claim discipline for
the spectral stack. Any eigensolver source movement must preserve:

- LOBPCG Laplacian residual and orthogonality evidence.
- LOBPCG `bcsstk04` preconditioner comparison behavior.
- Thick-restart exact diagonal residual and orthogonality evidence.
- Thick-restart bounded peak-basis behavior.
- Grow-m eigensolver residual narrative, closed-form spectra, shift-invert,
  and refinement behavior.
- The current claim boundary: iteration counts are diagnostic, and the project
  still does not claim broad ARPACK or SciPy eigensolver parity.

## Source Ownership Map

| Group | Representative Symbols | Consumers | Day 11 Disposition |
|---|---|---|---|
| Shared Lanczos and selection kernels | `s21_mgs_reorth`, `lanczos_iterate`, `lanczos_iterate_op`, `s20_lanczos_starting_vector`, `s20_spectrum_scale`, `s20_select_indices`, `s20_lift_ritz_vectors` | grow-m, thick restart, LOBPCG, focused tests | Do not split in Sprint 107; these are cross-backend comparison-critical kernels. |
| Shift-invert and refinement path | `s20_op_shift_invert`, `s29_refine_anchor`, `s29_refine_pair`, `s29_refine_eigenpairs`, `s29_maybe_refine` | grow-m public path, Sprint 29 integration coverage, residual comparison tests | Do not split in Sprint 107; moving these would expose static refinement details or create a new private source owner without reducing review risk enough. |
| Workspace and handle glue | `s49_eigs_handle_workspace`, `s49_eigs_handle_ensure`, `s49_eigs_handle_prepare_backend`, `sparse_eigs_handle_init`, `sparse_eigs_handle_free`, `sparse_eigs_handle_prepare` | public handle API, workspace internal implementation, dispatch path | Do not split in Sprint 107; workspace storage is already separated, and handle glue is tied to public API behavior. |
| Public dispatch and defaults | `s46_default_public_opts`, `s46_validate_public_entry`, `s46_select_backend`, `s46_run_growm_backend`, `s46_run_backend`, `s46_sparse_eigs_sym_impl`, `sparse_eigs_sym*` | public API, backend selection tests, reviewed CTest surface | Do not split in Sprint 107; this is the public behavior boundary. |
| Dense symmetric Jacobi helper | `s21_dense_sym_jacobi` | thick restart, LOBPCG, direct focused tests | Possible future split candidate only after a dedicated source-list and CMake parity day. Do not move on Day 12. |

## Split Candidate Decision

No Sprint 107 Day 12 source split is selected.

The least risky future candidate is a small dense-eigensolver helper owner for
`s21_dense_sym_jacobi`, but it is still shared by thick restart and LOBPCG.
Moving it now would require new source membership, internal declaration review,
and cross-backend validation for limited local simplification.

## No-Split Rationale

- The largest backend bodies already live outside `src/sparse_eigs.c` in
  thick-restart and LOBPCG source files.
- The remaining plausible split groups cross comparison-critical semantics,
  public dispatch behavior, or workspace handle behavior.
- A new eigensolver helper source file would require Make, CMake, source-list,
  and reviewed parity updates before it has clear maintainability payoff.
- Sprint 103 artifacts emphasize comparison evidence and claim hygiene, so
  Sprint 107 should not move spectral logic without a stronger dedicated
  validation lane.
- `src/sparse_eigs.c` is large, but its current size is less risky than a
  rushed split that obscures grow-m, shift-invert, refinement, or dispatch
  behavior.

## Build-System Follow-Through For Any Future Split

If a future sprint selects an eigensolver split, it must update and validate:

- Makefile library/object source membership.
- CMake target source membership.
- Any source-list parity checker or reviewed CMake source-count expectation.
- Internal declarations in `src/sparse_eigs_internal.h` or a narrower private
  header.
- Public install-header exclusion: no new public header should be installed
  unless a separate public API review approves it.
- CTest registration expectations if test binaries are added, removed, or
  renamed.

## Focused Validation Plan For A Future Split

Any future eigensolver source split should run at least:

```sh
make build/test_eigs && ./build/test_eigs
make build/test_eigs_thick_restart && ./build/test_eigs_thick_restart
make build/test_eigs_lobpcg && ./build/test_eigs_lobpcg
make build/test_sprint29_integration && ./build/test_sprint29_integration
make format && make lint && make test
```

If source-list, CMake, or reviewed CTest registration changes, the relevant
source-list and CMake parity checks must run before closeout.

## Day 12 Direction

Day 12 should write an explicit no-split deferral artifact and perform no
source extraction. The residual eigensolver queue should carry forward:

- possible dense Jacobi helper split behind dedicated build-system parity work;
- possible grow-m dispatch/refinement boundary audit after a comparison-focused
  validation lane exists;
- no public eigensolver API or install-header movement from Sprint 107.
