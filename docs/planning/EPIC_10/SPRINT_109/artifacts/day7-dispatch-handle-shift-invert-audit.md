# Day 7 Dispatch, Handle & Shift-Invert Audit

## Purpose

Day 7 finishes the Sprint 109 eigensolver source-boundary audit around public
workflow glue: default options, option validation, backend dispatch, reusable
handles, workspace preparation, shift-invert setup, and direct-solver
interaction.

Day 7 moves no code.

## Audited Surfaces

| Surface | Primary Owner | Public Contract Touched |
|---|---|---|
| Default options | `s46_default_public_opts` | `opts == NULL` defaults in `include/sparse_eigs.h`. |
| Option validation | `s46_validate_public_entry`, `sparse_eigs_handle_prepare` | documented `SPARSE_ERR_*` returns for bad args, shape, symmetry, vector buffers, refine prerequisites, and backend enums. |
| Iteration budget | `s49_eigs_effective_max_iters` plus matching logic in `s46_sparse_eigs_sym_impl` | `max_iterations == 0` selects library default; positive values are honored or rejected if too small. |
| AUTO dispatch | `s46_select_backend`, `s46_run_backend` | `backend_used` reporting and AUTO priority among LOBPCG, thick-restart, and grow-m. |
| Public handle lifetime | `s49_eigs_handle_ensure`, `sparse_eigs_handle_init`, `sparse_eigs_handle_free`, `sparse_eigs_sym_with_handle` | reusable opaque `sparse_eigs_handle_t` lifetime and caller-owned result buffers. |
| Workspace preparation | `s49_eigs_handle_prepare_backend`, `s49_eigs_growm_capacity`, `s49_eigs_thick_restart_capacity` | prepare-time capacity reuse and on-demand growth across backends. |
| Shift-invert setup | `s46_sparse_eigs_sym_impl`, `s20_op_shift_invert` | `NEAREST_SIGMA`, singular-shift returns, `used_csc_path_ldlt`, and inverse Ritz value post-processing. |
| Refinement hook position | `s29_maybe_refine` call at the end of `s46_sparse_eigs_sym_impl` | refinement runs only after eligible backend returns and after shift-invert cleanup. |

## Dispatch and Defaults Boundary

Dispatch is not a low-risk utility split. It jointly owns:

- public defaults for `opts == NULL`;
- result initialization fields including `backend_used`;
- AUTO priority:
  1. explicit LOBPCG;
  2. explicit thick-restart;
  3. AUTO plus preconditioner, large `n`, and adequate block size routes to
     LOBPCG;
  4. AUTO large `n` routes to thick-restart;
  5. otherwise grow-m;
- option validation parity between one-shot calls and handle preparation;
- max-iteration defaulting and explicit-cap rejection;
- backend-specific result and error propagation;
- refinement hook eligibility after backend completion.

Classification: **no-go for Sprint 109 movement**.

Future movement would require a named public-dispatch/private owner, direct
tests for default option materialization and option rejection parity, focused
AUTO routing tests, and review of `include/sparse_eigs.h` wording.

## Handle and Workspace Boundary

The public handle path is behavior-sensitive because it is the only reusable
workspace surface exposed to callers. It owns:

- allocation and zero-initialization of `sparse_eigs_workspace_t`;
- lifetime cleanup through `sparse_eigs_handle_free`;
- backend-specific prepare sizing before execution;
- on-demand growth when later calls require larger capacity;
- consistency between `sparse_eigs_handle_prepare` and
  `sparse_eigs_sym_with_handle`;
- the internal bridge from opaque public handle state to private workspace
  views.

Classification: **no-go for Sprint 109 movement**.

Future movement would need tests that separately prove:

- zeroed handle use;
- explicit prepare before solve;
- solve without prior prepare;
- repeated solve reuse;
- growth after underprepared capacity;
- all three backend workspace views;
- cleanup idempotence.

## Shift-Invert Boundary

Shift-invert is not a standalone helper yet. In `s46_sparse_eigs_sym_impl`, it
is interleaved with public validation, backend dispatch, result reporting, and
cleanup:

1. copy `A`;
2. subtract `sigma` from the diagonal;
3. factor `(A - sigma I)` with LDLT AUTO backend selection;
4. record `result->used_csc_path_ldlt`;
5. switch the Lanczos operator to `s20_op_shift_invert`;
6. pass the factorization to whichever eigensolver backend dispatch selects;
7. post-process inverse Ritz values as `lambda = sigma + 1 / theta`;
8. free LDLT and shifted matrix state before optional refinement returns.

Classification: **no-go for Sprint 109 movement**.

Future movement would require an explicit private shift-invert owner and tests
for diagonal shifts, indefinite KKT shifts, exact singular shifts, eigenvector
correctness, wide-spectrum interior targets, CSC/linked-list LDLT path
reporting, and cross-backend NEAREST_SIGMA parity.

## Current Test Ownership

| Test File | Contract Coverage |
|---|---|
| `tests/test_eigs.c` | public validation, handle grow-m reuse/growth, shift-invert diagonal/indefinite/singular/eigenvector/wide-spectrum cases, LDLT path reporting, refinement behavior. |
| `tests/test_eigs_thick_restart.c` | thick-restart explicit backend, AUTO above/below threshold, grow-m parity, memory bounds, NEAREST_SIGMA parity. |
| `tests/test_eigs_lobpcg.c` | LOBPCG explicit backend, block-size validation, preconditioner validation, AUTO routing to grow-m/thick-restart/LOBPCG, NEAREST_SIGMA parity. |
| `tests/test_sprint29_integration.c` | refinement plus progress callback, and cancellation before refinement. |

This coverage is strong enough to block casual movement. It does not remove the
need for direct tests if the public workflow glue is split into new private
owners later.

## Public Contract Implications

Any future source split for Day 7 surfaces must review
`include/sparse_eigs.h` because these internals implement documented behavior:

- `opts == NULL` defaults;
- AUTO backend selection;
- `backend_used`;
- `used_csc_path_ldlt`;
- `NEAREST_SIGMA` singular-shift behavior;
- `max_iterations` default and positive cap semantics;
- handle lifetime and reuse;
- progress callback phase behavior;
- `refine = 1` requiring `compute_vectors = 1`;
- partial-convergence return and result-field semantics.

If a future split changes any of those contracts, it is not a private cleanup;
it is a public behavior change.

## Future Split Prerequisites

Before moving dispatch, handle/workspace, or shift-invert code:

1. Name the private owner by behavior, such as dispatch policy,
   reusable workspace bridge, or shift-invert operator setup.
2. Add or identify direct tests for the moved owner, not just end-to-end
   backend tests.
3. Preserve Makefile, CMake, and `build-metadata/library_sources.txt` source
   parity.
4. Preserve public header and install-header surfaces unless explicitly
   reviewed as a public behavior change.
5. Run focused eigensolver tests:
   `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
   `test_sprint29_integration`.
6. Run the full quality gate if code moves:
   `make format && make lint && make test`.
7. Inspect CTest registration and reviewed test counts if CMake surfaces move.

## Sprint 109 Decision

No dispatch, handle/workspace, or shift-invert code moves in Sprint 109.

The eigensolver source-boundary workstream is closed for Sprint 109 with only
the Day 4 dense Jacobi extraction implemented. Broader eigensolver movement
requires future direct proof ownership and public-contract review.

## Completion Criteria Status

- Dispatch/default behavior map recorded.
- Handle/workspace ownership notes recorded.
- Shift-invert source-boundary notes recorded.
- Future extraction no-go list and prerequisites recorded.
- No future extraction depends on an undocumented public behavior assumption.
- No code movement occurred outside the approved dense Jacobi boundary.
