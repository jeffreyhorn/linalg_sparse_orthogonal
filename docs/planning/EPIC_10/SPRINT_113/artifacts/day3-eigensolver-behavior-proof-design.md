# Day 3 Eigensolver Behavior Proof Design

## Purpose

Day 3 turns the selected Sprint 113 eigensolver owner, **grow-m sizing and
retry behavior**, into concrete tests for Day 4.  The design uses existing
public results and callbacks only; it does not add a public test seam and does
not approve source movement.

## Exact Owner Surface

| Surface | Behavior to prove |
|---|---|
| `s49_eigs_effective_max_iters` | Default `max_iterations == 0` maps to the documented bounded budget; explicit budgets below the minimum are rejected. |
| `s49_eigs_growm_capacity` | Grow-m capacity is clamped by `n`, explicit/default `max_iterations`, and the minimum valid basis size. |
| `s46_run_growm_backend` | `peak_basis_size`, retry growth, progress callbacks, cancellation, accumulated `iterations`, and partial results are observable through the public result. |
| `sparse_eigs_workspace_prepare_growm` | The grow-m workspace is prepared at `m_cap`, so `peak_basis_size` reflects the allocated upper bound rather than the final converged `m`. |
| `sparse_eigs_sym` and `sparse_eigs_sym_with_handle` | Public entry points must preserve the same grow-m behavior without API drift. |

## Existing Coverage Baseline

Existing tests already cover adjacent behavior but not this owner as a focused
unit:

- `test_public_handle_growm_prepare_reuse_and_growth` proves public handle
  reuse and on-demand growth, but intentionally does not pin grow-m sizing
  formulas.
- `test_cross_eigs_refine_progress_cb` proves progress callbacks coexist with
  refinement, but does not assert retry-boundary step values.
- `test_cross_eigs_refine_cancel_short_circuits` proves cancellation exits
  cleanly at the first progress callback, but not retry behavior.
- SuiteSparse and stability tests exercise the grow-m path incidentally but
  are too broad for source-boundary proof.

Day 4 should add focused tests beside the existing grow-m and progress tests
instead of broadening public API or source metadata.

## Focused Test Cases

### Test 1: Default Grow-M Capacity Pins `peak_basis_size`

Target file: `tests/test_eigs.c`.

Fixture:

- diagonal matrix with `n = 12`;
- request `k = 2`;
- options:
  - `which = SPARSE_EIGS_LARGEST`;
  - `tol = 1e-12`;
  - `backend = SPARSE_EIGS_BACKEND_LANCZOS`;
  - `max_iterations = 0`.

Expected public observations:

- return `SPARSE_OK`;
- `backend_used == SPARSE_EIGS_BACKEND_LANCZOS`;
- `n_converged == 2`;
- eigenvalues are `12.0` and `11.0` within `1e-9`;
- `peak_basis_size == 12`.

Why this proves the default path:

- default effective max iterations are `max(10 * k + 20, 100) = 100`;
- grow-m capacity is `min(default_max_iterations, n) = 12`;
- the initial basis would converge with at most `m = 12`, but the assertion
  proves the result reports the allocated grow-m upper bound.

### Test 2: Explicit Capacity Pins `peak_basis_size`

Target file: `tests/test_eigs.c`.

Fixture:

- diagonal matrix with `n = 64`;
- request `k = 2`;
- options:
  - `which = SPARSE_EIGS_LARGEST`;
  - `tol = 1e-12`;
  - `backend = SPARSE_EIGS_BACKEND_LANCZOS`;
  - `max_iterations = 24`.

Expected public observations:

- return `SPARSE_OK`;
- `backend_used == SPARSE_EIGS_BACKEND_LANCZOS`;
- `n_converged == 2`;
- eigenvalues are `64.0` and `63.0` within `1e-9`;
- `peak_basis_size == 24`.

Why this proves the explicit path:

- the explicit budget is above the minimum `2 * k + 10 = 14`;
- grow-m capacity is `min(max_iterations, n) = 24`;
- `peak_basis_size` must report `m_cap`, not the final converged subspace.

### Test 3: Too-Small Explicit Budget Rejects

Target file: `tests/test_eigs.c`.

Fixture:

- diagonal matrix with `n = 16`;
- request `k = 3`;
- options:
  - `which = SPARSE_EIGS_LARGEST`;
  - `backend = SPARSE_EIGS_BACKEND_LANCZOS`;
  - `max_iterations = 15`.

Expected public observations:

- return `SPARSE_ERR_BADARG`;
- result fields remain safe to inspect but no convergence assertions are made.

Why this proves the validation path:

- minimum required budget is `min(2 * k + 10, n) = 16`;
- `15` is below that minimum and must reject before grow-m workspace use.

### Test 4: Retry Progress Steps Accumulate Iterations

Target file: `tests/test_eigs.c`.

Fixture:

- SPD tridiagonal matrix with `n = 64`, diagonal `4`, off-diagonal `-1`;
- request `k = 2`;
- options:
  - `which = SPARSE_EIGS_LARGEST`;
  - `tol = 1e-30`;
  - `backend = SPARSE_EIGS_BACKEND_LANCZOS`;
  - `max_iterations = 64`;
  - `progress_cb = growm_progress_record_cb`.

Expected public observations:

- return `SPARSE_ERR_NOT_CONVERGED` or `SPARSE_OK`; if it unexpectedly
  converges, the progress invariants still hold;
- at least two progress callbacks fire on retry boundaries;
- first callback has `phase == "lanczos"` and `step == 0`;
- subsequent callback steps are monotonically increasing;
- final `iterations` is at least the last callback step;
- `peak_basis_size == 64`.

Why this proves retry behavior:

- for `k = 2`, grow-m starts at `m_init = 36` and grows by
  `m_grow = 22`, so the retry sequence is expected to include `36`,
  `58`, and `64` unless convergence occurs first;
- using an unrealistically tight tolerance forces retry-boundary progress
  without depending on internal helper visibility.

Stability rule:

- Do not assert an exact callback count or exact final iteration count unless
  Day 4 confirms it is stable across platforms.  Assert monotonic public
  behavior instead.

### Test 5: Retry-Boundary Cancellation Exits Cleanly

Target file: `tests/test_eigs.c` or keep in `tests/test_sprint29_integration.c`
if the existing progress-helper pattern is reused.

Fixture:

- same SPD tridiagonal fixture as Test 4;
- request `k = 2`;
- options:
  - `which = SPARSE_EIGS_LARGEST`;
  - `tol = 1e-30`;
  - `backend = SPARSE_EIGS_BACKEND_LANCZOS`;
  - `max_iterations = 64`;
  - progress callback returns non-zero once `step > 0`.

Expected public observations:

- return `SPARSE_ERR_CANCELLED`;
- callback count is at least two;
- last callback step is greater than zero;
- no eigenvalue or residual assertions are made after cancellation.

Why this proves cancellation:

- existing coverage cancels at the first callback; this test proves the
  grow-m retry-boundary cancellation path after one completed Lanczos pass.

## Fixtures, Tolerances, and Seeds

No random seed is required.  The grow-m path uses the deterministic internal
starting vector from `s20_lanczos_starting_vector`.

Use two deterministic fixtures:

- diagonal matrices built with existing `build_diag` helpers for sizing and
  validation-path tests;
- an SPD tridiagonal matrix, either via an existing helper or a local helper
  kept near the tests, for retry/progress behavior.

Suggested tolerances:

- diagonal eigenvalue checks: `1e-9`;
- converged diagonal residual checks: not required for capacity tests unless
  vectors are requested;
- progress retry forcing: `tol = 1e-30`, with status allowed to be
  `SPARSE_OK` or `SPARSE_ERR_NOT_CONVERGED` where the proof is about progress
  and iteration accounting.

## Proof-Value Visibility Rules

Day 4 should keep these values visible at each call site:

- matrix size `n`;
- requested `k`;
- explicit `max_iterations`;
- derived expected `peak_basis_size`;
- expected largest eigenvalues for diagonal fixtures;
- progress callback step history;
- status expectations and allowed fallback status when convergence may vary.

Allowed helper extraction:

- a tiny progress recorder struct and callback local to `tests/test_eigs.c`;
- a tiny SPD tridiagonal builder only if an existing helper is not already
  convenient in `tests/test_eigs.c`.

Avoid helper extraction that hides:

- expected `m_cap` values;
- expected eigenvalues;
- callback step assertions;
- rejection thresholds.

## Day 4 Implementation Checklist

1. Add a grow-m progress recorder and callback near the existing eigensolver
   tests.
2. Add Test 1 for default `peak_basis_size` on `n = 12`, `k = 2`.
3. Add Test 2 for explicit `peak_basis_size` on `n = 64`, `k = 2`,
   `max_iterations = 24`.
4. Add Test 3 for `SPARSE_ERR_BADARG` with `n = 16`, `k = 3`,
   `max_iterations = 15`.
5. Add Test 4 for retry-boundary progress and monotonic accumulated steps.
6. Add Test 5 for cancellation after a nonzero retry-boundary step.
7. Register the tests in `tests/test_eigs.c` unless Day 4 finds a stronger
   reason to keep cancellation in `tests/test_sprint29_integration.c`.
8. Keep the public API, install headers, helper targets, and reviewed CTest
   registration unchanged.

## Validation Commands

Focused Day 4 validation:

```sh
make build/test_eigs
build/test_eigs
```

If cancellation coverage is implemented in `tests/test_sprint29_integration.c`,
also run:

```sh
make build/test_sprint29_integration
build/test_sprint29_integration
```

Because Day 4 is expected to modify `.c` tests, final validation for the code
change must include:

```sh
make format && make lint && make test
```

Documentation-only Day 3 validation remains:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_113
```

## Completion Criteria

- The Day 4 tests can be implemented without changing public API.
- The proof design observes grow-m behavior through public results and
  callbacks.
- Expected capacities, budgets, fixtures, tolerances, and allowed statuses are
  explicit.
- Source movement remains blocked until Day 4 proof is implemented and Day 5
  makes an evidence-backed movement or no-move decision.
