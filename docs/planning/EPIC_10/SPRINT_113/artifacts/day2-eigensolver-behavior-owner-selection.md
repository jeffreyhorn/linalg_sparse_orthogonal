# Day 2 Eigensolver Behavior Owner Selection

## Purpose

Day 2 selects one behavior-sensitive eigensolver owner for Sprint 113 proof.
The selection must avoid duplicating Sprint 110's completed public
handle/workspace no-move contract and must not approve source movement before
direct owner-specific proof exists.

## Source and Test Surfaces Reviewed

| Surface | Role in Day 2 selection |
|---|---|
| `src/sparse_eigs.c` | Central owner for defaults, validation, dispatch, grow-m retry, refinement, shift-invert, and shared Lanczos helpers. |
| `src/sparse_eigs_workspace_internal.c` | Workspace prepare owner for grow-m, thick-restart, and LOBPCG views. |
| `src/sparse_eigs_internal.h` | Internal declarations for shared eigensolver helpers and backend workspace contracts. |
| `include/sparse_eigs.h` | Public option/result/handle contract; must not drift during behavior-owner selection. |
| `tests/test_eigs.c` | Primary grow-m, shift-invert, refinement, bad-arg, and handle/workspace coverage. |
| `tests/test_eigs_thick_restart.c` | Thick-restart AUTO dispatch and cross-backend behavior coverage. |
| `tests/test_eigs_lobpcg.c` | LOBPCG AUTO/explicit dispatch, nearest-sigma, and backend parity coverage. |
| `tests/test_sprint29_integration.c` | Cross-feature refinement/progress/cancellation integration coverage. |

## Candidate Comparison

| Candidate | Existing coverage | Movement risk | Testability | Day 2 decision |
|---|---|---|---|---|
| Defaults and option validation | `test_bad_args`, public handle validation tests, entry validation paths. | Public API and error-code behavior; high risk if moved without broad validation. | Direct but already partly covered. | Defer; useful but not the best non-duplicate owner. |
| Backend dispatch | AUTO small/mid/large/preconditioned/no-precondition/explicit tests in LOBPCG and thick-restart suites. | Public `backend_used` and backend result propagation; high coupling to thresholds. | Strong existing direct tests. | Defer; already has stronger focused coverage than other candidates. |
| Grow-m sizing and retry behavior | Partial coverage through grow-m tests, handle growth tests, peak basis checks, and progress/cancel integration. | Behavior-sensitive but mostly private; possible narrow proof around capacity/retry invariants. | Directly observable through `peak_basis_size`, `iterations`, progress callbacks, partial results, and returned eigenpairs. | **Select for Sprint 113 proof.** |
| Refinement defaults and budgets | Dedicated refinement tests in `tests/test_eigs.c` plus integration cancellation tests. | Coupled to returned vector mutation and backend status folding. | Strong existing direct tests. | Defer; already well pinned. |
| Shift-invert setup | Dedicated diagonal, indefinite, singular, eigenvector, wide-spectrum, CSC/linked-list threshold, thick-restart, and LOBPCG nearest-sigma coverage. | Coupled to LDLT, singular shifts, and inverse Ritz conversion. | Strong existing direct tests. | Defer; proof surface already broad. |
| Shared Lanczos kernels | Cross-backend parity exists, but helpers affect many backend behaviors. | High risk; kernel movement can perturb ordering, residual scale, reorthogonalization, and vector lifting. | Testable only through several backend suites. | Defer until a narrower kernel-specific invariant is proven. |
| Public handle/workspace source movement | Sprint 110 Day 8 validated and published no-move contract. | Public lifetime, prepare, reuse, growth, and cleanup behavior. | Already completed validation; moving would duplicate or supersede Sprint 110. | Exclude from Sprint 113 Day 2 selection. |

## Selected Owner

Selected owner: **grow-m sizing and retry behavior**.

Primary implementation area:

- `s49_eigs_effective_max_iters`;
- `s49_eigs_growm_capacity`;
- `s46_run_growm_backend`;
- grow-m workspace preparation via `sparse_eigs_workspace_prepare_growm`;
- progress callback emission at grow-m retry boundaries;
- partial-result publication when `m_cap` is exhausted.

Primary test area:

- `tests/test_eigs.c`, with adjacent coverage in
  `tests/test_sprint29_integration.c` when progress/cancel interactions are
  involved.

## Selection Rationale

Grow-m sizing and retry behavior is the narrowest useful remaining
eigensolver owner because:

- it is not the completed Sprint 110 handle/workspace no-move contract;
- it is less publicly exposed than defaults/validation or public handle
  semantics;
- it has observable behavior without changing public API:
  - `peak_basis_size`;
  - `iterations`;
  - `n_converged`;
  - `residual_norm`;
  - returned eigenpairs;
  - progress callback calls;
  - `SPARSE_ERR_NOT_CONVERGED` partial-result behavior;
- it is central enough to make source-boundary decisions meaningful;
- it still needs direct owner-specific proof before any source movement.

## Required Behavior Invariants

Day 3 should turn these invariants into focused proof:

1. Default `max_iterations == 0` derives a bounded effective max-iteration
   budget without exceeding natural Krylov limits.
2. Explicit `max_iterations` below the minimum required budget is rejected
   with `SPARSE_ERR_BADARG`.
3. Grow-m capacity is clamped by `n`, `max_iterations`, and the minimum valid
   Lanczos basis size.
4. `peak_basis_size` reports the allocated grow-m upper bound, not merely the
   final converged `m`.
5. Retry growth accumulates `iterations` across runs.
6. Progress callbacks are emitted at grow-m retry boundaries and cancellation
   exits cleanly.
7. Exhausting `m_cap` returns partial results and `SPARSE_ERR_NOT_CONVERGED`
   when convergence is not achieved.
8. Shift-invert and other non-grow-m owner behavior must not be broadened by
   this proof batch.

## Focused Test Plan

Day 3 should design tests around existing public behavior rather than adding a
public test seam:

- a small diagonal or tridiagonal fixture where default grow-m capacity can be
  inferred from `peak_basis_size`;
- an explicit too-small `max_iterations` case that verifies bad-argument
  rejection;
- a deliberately low-budget grow-m run that returns partial results with
  `SPARSE_ERR_NOT_CONVERGED`;
- a progress-callback case that confirms callback invocation and cancellation
  at a retry boundary;
- a focused run that confirms returned eigenpairs remain correct when the
  grow-m path converges.

Expected focused validation after tests land:

```sh
make build/test_eigs build/test_sprint29_integration
build/test_eigs
build/test_sprint29_integration
```

Because Day 4 is expected to modify `.c` test files, Day 13 should include the
full code quality chain unless later scope changes keep the branch
documentation-only:

```sh
make format && make lint && make test
```

## No-Claim List for Unselected Owners

Sprint 113 Day 2 does not approve source movement for:

- defaults and option validation;
- backend dispatch;
- refinement defaults and budgets;
- shift-invert setup;
- shared Lanczos kernels;
- public handle/workspace source movement.

It also does not claim:

- public eigensolver API changes;
- install-header changes;
- package/platform support expansion;
- shared-library or ABI support;
- reviewed CTest registration changes;
- broad eigensolver source decomposition.

## Completion Criteria

- Exactly one eigensolver owner is selected: grow-m sizing and retry behavior.
- The selected owner has direct observable behavior to test.
- Source movement remains blocked until Day 4 proof lands and Day 5 makes an
  evidence-backed movement/no-move decision.
