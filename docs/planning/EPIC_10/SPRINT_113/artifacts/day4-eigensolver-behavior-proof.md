# Day 4 Eigensolver Behavior Proof

## Purpose

Day 4 implements the focused grow-m sizing and retry proof designed on Day 3.
The proof validates behavior through existing public result fields and progress
callbacks only; no public API, install-header, helper-target, CTest, Make, or
CMake registration changed.

## Code Changes

Target file:

- `tests/test_eigs.c`

Added local test support:

- `build_shifted_tridiag`, a deterministic SPD tridiagonal fixture with
  diagonal `4` and off-diagonal `-1`;
- `growm_progress_record_t`, a small public-callback recorder for retry
  boundary steps;
- `growm_progress_record_cb`, a callback that records `phase` and `step` and
  can cancel after a configured step.

Added focused tests:

| Test | Proof |
|---|---|
| `test_growm_default_capacity_pins_peak_basis_size` | Default `max_iterations == 0` on `n = 12`, `k = 2` reports `peak_basis_size == 12` and returns the largest two diagonal eigenvalues. |
| `test_growm_explicit_capacity_pins_peak_basis_size` | Explicit `max_iterations = 24` on `n = 64`, `k = 1` reports `peak_basis_size == 24` and returns the largest diagonal eigenvalue. |
| `test_growm_too_small_explicit_iteration_budget_rejected` | Explicit `max_iterations = 15` with `n = 16`, `k = 3` rejects with `SPARSE_ERR_BADARG`, matching `min(2 * k + 10, n) = 16`. |
| `test_growm_retry_progress_steps_accumulate_iterations` | Tight-tolerance grow-m retry emits monotonic `"lanczos"` progress steps and keeps `result.iterations >= last_recorded_step`. |
| `test_growm_retry_boundary_cancellation_exits_cleanly` | Cancellation after a nonzero retry-boundary step returns `SPARSE_ERR_CANCELLED` without relying on partial eigenpair state. |

## Implementation Adjustment

The Day 3 explicit-capacity design proposed `n = 64`, `k = 2`,
`max_iterations = 24` with a repeated tail spectrum.  The first focused run
showed that the repeated tail can allow a duplicate top Ritz value for the
second requested pair.  Day 4 narrowed that test to `k = 1`, preserving the
explicit-capacity proof while avoiding a clustered-eigenvalue ordering claim
that is unrelated to grow-m sizing.

## Drift Assessment

No drift was introduced:

- no public eigensolver API changes;
- no install-header changes;
- no helper-target changes;
- no Make/CMake/source-list changes;
- no reviewed CTest registration changes;
- no eigensolver source movement.

## Validation Evidence

Focused validation:

```sh
make build/test_eigs && build/test_eigs
```

Result:

- `test_eigs` passed;
- `36` tests run;
- `0` failed;
- `345` assertions;
- new grow-m tests all passed.

Required full quality chain for `.c` test changes:

```sh
make format && make lint && make test
```

Result:

- `make format` completed;
- `make lint` completed, including strict warnings, `clang-tidy`, and
  `cppcheck`;
- `make test` completed;
- all tests passed.

## Source-Movement Implication

The selected owner now has direct behavior proof for:

- default grow-m capacity;
- explicit grow-m capacity;
- too-small explicit budget rejection;
- retry-boundary progress step publication;
- accumulated iteration accounting relative to retry progress;
- retry-boundary cancellation cleanup.

Day 5 can use this evidence to decide whether a narrow private source movement
is justified.  Until that Day 5 decision, source movement remains blocked.

## Completion Criteria

- Focused grow-m behavior tests are implemented.
- Focused eigensolver validation passes.
- Full quality chain passes for the `.c` test change.
- Public API, install-header, helper-target, and reviewed CTest surfaces remain
  unchanged.
