# Sprint 110 Day 7 Eigensolver Behavior Owner Selection

## Purpose

Day 7 selects one bounded eigensolver behavior owner for Day 8 validation while
avoiding duplicate dense Jacobi work and preventing behavior-heavy source
movement from being treated as simple maintainability cleanup.

## Prior Evidence Reviewed

- Sprint 109 `day6-growm-refinement-shared-kernel-audit.md`.
- Sprint 109 `day7-dispatch-handle-shift-invert-audit.md`.
- Sprint 109 retrospective eigensolver closeout notes.
- Sprint 110 working notes through the Day 6 Matrix I/O closure.
- Current source and test ownership in:
  - `src/sparse_eigs.c`;
  - `src/sparse_eigs_workspace_internal.c`;
  - `src/sparse_eigs_internal.h`;
  - `include/sparse_eigs.h`;
  - `tests/test_eigs.c`;
  - `tests/test_eigs_thick_restart.c`;
  - `tests/test_eigs_lobpcg.c`;
  - `tests/test_sprint29_integration.c`.

## Completed Work Excluded

Dense Jacobi is excluded from Sprint 110 Day 7 work because Sprint 109 already:

- selected `s21_dense_sym_jacobi` as the only approved low-risk eigensolver
  source split;
- moved it into `src/sparse_eigs_dense_internal.c`;
- registered the source in Makefile, CMake, and
  `build-metadata/library_sources.txt`;
- validated the focused eigensolver Make and CMake lanes.

No Sprint 110 Day 7 item should repeat, rename, or rejustify that extraction.

## Candidate Owner Inventory

| Candidate | Current Owner | Risk | Day 7 Decision |
|---|---|---|---|
| Defaults and option validation | `s46_default_public_opts`, `s46_validate_public_entry`, `sparse_eigs_handle_prepare` | Public behavior: `opts == NULL`, error codes, backend enums, `refine` prerequisites. | Defer source movement. Validate only through selected handle/workspace owner. |
| Backend dispatch | `s46_select_backend`, `s46_run_backend` | Public behavior: AUTO priority, `backend_used`, backend result and error propagation. | Defer. Needs direct dispatch-policy proof before movement. |
| Public handle/workspace bridge | `sparse_eigs_handle_prepare`, `sparse_eigs_sym_with_handle`, `s49_eigs_handle_prepare_backend` | Public behavior: reusable handle lifetime, explicit prepare, on-demand growth, all backend workspace views. | Select for Day 8 validation target. |
| Grow-m sizing and retry behavior | `s46_run_growm_backend`, `s49_eigs_growm_capacity` | Public behavior: retry growth, progress callbacks, partial results, peak basis, residuals. | Defer movement. |
| Refinement defaults and budgets | `s29_maybe_refine`, `s29_refine_eigenpairs`, `s29_refine_pair` | Public behavior: mutates returned eigenpairs, preserves backend return codes, handles cancellation boundaries. | Defer movement. |
| Shift-invert setup | `s46_sparse_eigs_sym_impl`, `s20_op_shift_invert` | Public behavior: `NEAREST_SIGMA`, singular shifts, LDLT path reporting, inverse Ritz conversion. | Defer movement. |
| Shared Lanczos kernels | `s21_mgs_reorth`, `s20_lanczos_starting_vector`, `s20_spectrum_scale`, `s20_select_indices`, `s20_lift_ritz_vectors` | Cross-backend behavior: ordering, residual scale, reorthogonalization, vector lifting. | Defer movement. |

## Selected Owner

Selected Day 8 target: **public handle/workspace bridge validation**.

This owner includes:

- `sparse_eigs_handle_init`;
- `sparse_eigs_handle_free`;
- `sparse_eigs_handle_prepare`;
- `sparse_eigs_sym_with_handle`;
- `s49_eigs_handle_ensure`;
- `s49_eigs_handle_prepare_backend`;
- grow-m, thick-restart, and LOBPCG workspace prepare calls through
  `src/sparse_eigs_workspace_internal.c`.

## Selection Rationale

The public handle/workspace bridge is the narrowest useful remaining
eigensolver behavior owner because it has:

- a clear public surface in `include/sparse_eigs.h`;
- direct tests for explicit prepare, reuse, validation, on-demand growth, and
  backend-specific workspace preparation;
- a bounded no-drift rule: no public header, install-header, source-list,
  helper-target, or CTest registration changes are needed for validation;
- lower behavior risk than dispatch/default movement, shift-invert movement,
  grow-m movement, refinement movement, or shared Lanczos-kernel movement.

The selection is **not** approval for source movement. It is approval for Day 8
to validate the owner and publish a no-move contract unless a smaller internal
cleanup is proven unnecessary.

## Day 8 Validation Checklist

Focused eigensolver validation should include:

- `test_eigs`;
- `test_eigs_thick_restart`;
- `test_eigs_lobpcg`;
- `test_sprint29_integration`.

Direct behavior expectations:

- zeroed handle initialization remains valid;
- explicit `sparse_eigs_handle_prepare` before solve remains valid;
- solve without prior prepare still allocates on demand;
- repeated solve reuses handle-owned workspace;
- later larger calls grow workspace safely;
- grow-m, thick-restart, and LOBPCG workspace prepare paths remain covered;
- invalid handle-prepare inputs preserve documented error behavior;
- handle cleanup remains idempotent and zeroes the public handle.

No-drift checks:

- no `include/sparse_eigs.h` change unless reviewed as a public behavior
  change;
- no install/export rule change;
- no source-list change unless code actually moves;
- no helper target addition;
- no reviewed CTest registration change;
- if code moves, run `make format && make lint && make test`.

## Explicit Deferrals

Day 7 defers the following from source movement:

- defaults and option validation;
- backend dispatch;
- grow-m sizing and retry behavior;
- refinement defaults and budgets;
- shift-invert setup;
- shared Lanczos kernels.

These candidates remain behavior-sensitive and require direct owner-specific
tests before any future source split.

## Completion Status

- Exactly one eigensolver behavior owner was selected for Day 8 validation.
- Dense Jacobi work was excluded as completed Sprint 109 work.
- Behavior-preservation tests were identified before any movement.
- Public header drift is fenced off.
- No eigensolver code moved on Day 7.
