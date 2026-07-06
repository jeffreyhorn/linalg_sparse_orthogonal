# Day 6 Grow-M, Refinement & Shared-Kernel Audit

## Purpose

Day 6 audits the eigensolver code that remains in `src/sparse_eigs.c` after the
Day 4 dense Jacobi extraction. The goal is to decide, with evidence, whether
grow-m, refinement, or shared Lanczos kernels should move during Sprint 109.

Day 6 moves no code.

## Audited Surfaces

| Surface | Primary Owner | Evidence |
|---|---|---|
| MGS reorthogonalization | `src/sparse_eigs.c` `s21_mgs_reorth` | Shared by standard Lanczos and thick-restart iteration. |
| Standard Lanczos recurrence | `src/sparse_eigs.c` `lanczos_iterate`, `lanczos_iterate_op` | Owns recurrence validation, invariant-subspace detection, final beta residual convention, and reorth gate. |
| Shared spectral helpers | `src/sparse_eigs.c` `s20_lanczos_starting_vector`, `s20_spectrum_scale`, `s20_select_indices`, `s20_lift_ritz_vectors` | Used across grow-m, thick-restart, shift-invert, LOBPCG parity tests, and internal test surfaces. |
| Refinement post-pass | `src/sparse_eigs.c` `s29_refine_pair`, `s29_refine_eigenpairs`, `s29_maybe_refine` | Mutates public result vectors/values in place and preserves backend return-code semantics. |
| Grow-m backend | `src/sparse_eigs.c` `s46_run_growm_backend` | Owns m sizing, retry growth, convergence gate, result fields, partial emission, and workspace view assumptions. |
| Workspace views | `src/sparse_eigs_workspace_internal.c` grow-m/thick-restart/LOBPCG prepare helpers | Source split must preserve shared workspace capacity and zeroing contracts. |
| Thick-restart consumer | `src/sparse_eigs_thick_restart.c` `s21_thick_restart_outer_loop` | Reuses starting-vector, selection, spectrum-scale, vector-lift, and MGS conventions. |
| LOBPCG consumer | `src/sparse_eigs_lobpcg.c` `s21_lobpcg_solve` and `s21_lobpcg_rr_step` | Reuses dense Jacobi, selection, residual-scale, and backend result conventions. |

## Grow-M Boundary

`s46_run_growm_backend` is not a low-risk helper split. It owns user-visible and
test-visible behavior:

- `m_cap`, `m_init`, and `m_grow` sizing policy;
- deterministic retry behavior through one starting vector;
- `progress_cb` phase timing and cancellation cleanup;
- cumulative `result->iterations` accounting across retries;
- `result->peak_basis_size`;
- Wu/Simon residual gate and `result->residual_norm`;
- partial result emission under `SPARSE_ERR_NOT_CONVERGED`;
- shift-invert eigenvalue conversion for `NEAREST_SIGMA`;
- grow-m workspace layout through `sparse_eigs_workspace_prepare_growm`.

Classification: **no-go for Sprint 109 movement**.

Future movement would need a dedicated grow-m owner, focused tests for
convergence, partial results, cancellation, progress callbacks, residual norms,
shift-invert output, workspace reuse, and source-list parity across Makefile,
CMake, and `build-metadata/library_sources.txt`.

## Refinement Boundary

The refinement helpers are behavior-sensitive because they run after backend
execution and directly mutate the public result:

- `s29_refine_pair` computes residuals, builds shifted copies of `A`, factors
  `(A - lambda I)`, perturbs singular shifts, solves inverse-iteration steps,
  normalizes vectors, and recomputes Rayleigh quotients.
- `s29_refine_eigenpairs` walks `result->n_converged`, mutates
  `result->eigenvalues` and `result->eigenvectors`, and recomputes
  `result->residual_norm`.
- `s29_maybe_refine` preserves backend return-code semantics by refining only
  `SPARSE_OK` and `SPARSE_ERR_NOT_CONVERGED` outputs and leaving cancellation
  or hard backend failures untouched.

Classification: **no-go for Sprint 109 movement**.

Future movement would need an explicit private refinement owner and focused
tests for default-off behavior, tightened residuals, LOBPCG refinement,
`refine_max_iters`, cancellation short-circuiting, singular-shift perturbation,
and post-refinement residual reporting.

## Shared-Kernel Boundary

The following helpers are small, but not behavior-free:

| Helper | Risk |
|---|---|
| `s21_mgs_reorth` | Encodes MGS stability, OpenMP thresholding, and shared alignment between standard and thick-restart Lanczos. |
| `s20_lanczos_starting_vector` | Encodes deterministic non-axis-aligned starting-vector behavior used by grow-m and thick-restart. |
| `s20_spectrum_scale` | Anchors relative residual scaling in grow-m, thick-restart, and LOBPCG-style comparisons. |
| `s20_select_indices` | Encodes output ordering for `LARGEST`, `SMALLEST`, and `NEAREST_SIGMA`, including shift-invert largest-magnitude selection. |
| `s20_lift_ritz_vectors` | Encodes full-space Ritz vector lifting and shift-invert eigenvector convention. |

Classification: **needs more proof before movement**.

These can become later split candidates only if the private owner is named by
responsibility, not by convenience, and the move adds direct unit coverage for
helper semantics plus cross-backend validation.

## Current Test Ownership

Focused tests already cover the behavior these helpers protect:

| Test File | Relevant Coverage |
|---|---|
| `tests/test_eigs.c` | grow-m handle reuse, shift-invert, near-singular behavior, zero/one matrix cases, refinement default-off/tightening/LOBPCG/budget behavior. |
| `tests/test_eigs_thick_restart.c` | thick-restart parity with grow-m, memory bounds, AUTO routing, NEAREST_SIGMA parity, locked progress, and single-phase grow-m equivalence. |
| `tests/test_eigs_lobpcg.c` | LOBPCG residual behavior, cross-backend parity, AUTO dispatch, explicit overrides, block-size behavior, and NEAREST_SIGMA parity. |
| `tests/test_sprint29_integration.c` | refinement with progress callback and cancellation before refinement. |

This coverage is sufficient to block unsafe movement; it is not sufficient to
justify another source split during Sprint 109 without adding direct moved
helper tests.

## Future Split Prerequisites

Before moving any Day 6 audited code, require all of the following:

1. A named private source owner with a narrower responsibility than
   `src/sparse_eigs.c`.
2. Makefile, CMake, and `build-metadata/library_sources.txt` source-list parity.
3. No public header, install-header, helper-target, or CTest registration drift.
4. Direct tests for the moved helper's ordering, residual scaling,
   reorthogonalization, or refinement contract.
5. Focused backend validation:
   `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and
   `test_sprint29_integration`.
6. Full quality gate if code moves:
   `make format && make lint && make test`.

## Sprint 109 Decision

No additional eigensolver code moves on Day 6.

The Day 4 dense Jacobi extraction remains the only approved Sprint 109
eigensolver implementation split so far. Grow-m and refinement are blocked as
Sprint 109 movement candidates, and shared kernels remain future candidates
only after stronger direct helper tests and cross-backend evidence.

Day 7 should continue with dispatch/default behavior, public handle/workspace
glue, and shift-invert ownership.

## Completion Criteria Status

- Grow-m/refinement dependency map recorded.
- Shared-kernel boundary classification recorded.
- No-go and future-proof prerequisites recorded.
- No code movement occurred outside the approved dense Jacobi boundary.
