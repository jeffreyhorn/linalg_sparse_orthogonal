# Day 6 Eigensolver No-Move Contract

## Purpose

Day 6 executes the Day 5 decision for the selected grow-m sizing and retry
owner.  The chosen path is a no-move contract: preserve current source
ownership, keep the Day 4 behavior proof as the guardrail, and explicitly
defer broader eigensolver source movement until the shared Lanczos helper
boundary has direct proof.

## Executed Path

Executed path: **no move**.

No private eigensolver source file was added.  No Make, CMake, source-list, or
reviewed CTest metadata was changed.

## Current Owner Metrics

| Surface | Current metric |
|---|---:|
| `src/sparse_eigs.c` | 1412 lines |
| `src/sparse_eigs_workspace_internal.c` | 267 lines |
| `src/sparse_eigs_workspace_internal.h` | 82 lines |
| `tests/test_eigs.c` | 1758 lines |

## Current Grow-M Owner Locations

| Owner Piece | Location |
|---|---|
| Effective max-iteration budget | `src/sparse_eigs.c:769` |
| Grow-m capacity formula | `src/sparse_eigs.c:793` |
| Public handle grow-m preparation branch | `src/sparse_eigs.c:838` |
| Public handle validation path calling budget helper | `src/sparse_eigs.c:890` |
| Grow-m backend executor | `src/sparse_eigs.c:965` |
| Grow-m workspace preparation call inside executor | `src/sparse_eigs.c:1029` |
| Backend dispatch into grow-m executor | `src/sparse_eigs.c:1223` |
| Grow-m workspace implementation | `src/sparse_eigs_workspace_internal.c:84` |
| Grow-m workspace declaration | `src/sparse_eigs_workspace_internal.h:72` |

## Behavior Guards

The Day 4 tests protecting this owner are:

- `test_growm_default_capacity_pins_peak_basis_size`;
- `test_growm_explicit_capacity_pins_peak_basis_size`;
- `test_growm_too_small_explicit_iteration_budget_rejected`;
- `test_growm_retry_progress_steps_accumulate_iterations`;
- `test_growm_retry_boundary_cancellation_exits_cleanly`.

These tests guard:

- default `max_iterations == 0` capacity behavior;
- explicit capacity behavior;
- too-small explicit budget rejection;
- retry-boundary progress publication;
- accumulated iterations relative to retry progress;
- retry-boundary cancellation cleanup.

## No-Move Contract

Until a later proof explicitly covers the shared Lanczos helper boundary:

- keep `s46_run_growm_backend` in `src/sparse_eigs.c`;
- keep `s49_eigs_effective_max_iters` and `s49_eigs_growm_capacity` adjacent to
  public entry validation and backend dispatch;
- keep `sparse_eigs_workspace_prepare_growm` in
  `src/sparse_eigs_workspace_internal.c`;
- do not expose private Lanczos helpers only to enable source movement;
- do not move public handle/workspace grow-m preparation, because Sprint 110
  already established a public handle/workspace no-move contract;
- require the Day 4 grow-m tests to pass before any future grow-m movement;
- require Make/CMake/source-list parity checks for any future new private
  eigensolver source file.

## Deferred Movement Queue

Future movement remains deferred for:

1. `lanczos_iterate_op` helper boundary proof.
2. Ritz value selection proof for repeated and clustered spectra.
3. Ritz vector lifting proof with and without `compute_vectors`.
4. Partial-result publication after `m_cap` exhaustion.
5. Shift-invert eigenvalue conversion through the grow-m path.
6. Shared helper visibility rules that avoid public API and install-header
   drift.
7. New private source-file Make/CMake/source-list parity, if a later sprint
   moves the grow-m executor.

## Drift Assessment

Day 6 introduced no drift:

- no public API changes;
- no install-header changes;
- no helper-target changes;
- no Make/CMake/source-list changes;
- no reviewed CTest registration changes;
- no source movement.

## Validation Evidence

Focused validation:

```sh
make build/test_eigs && build/test_eigs
```

Result:

- passed;
- `test_eigs`: `36` tests, `0` failed, `345` assertions.

Documentation and diff hygiene:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_113 tests/test_eigs.c
```

Result:

- passed;
- no trailing whitespace found.

The full quality chain was already run after the Day 4 `.c` test change:

```sh
make format && make lint && make test
```

Day 6 made documentation-only changes after that run, so the full chain was not
rerun.

## Completion Criteria

- Day 5 no-move decision is executed.
- Current source ownership metrics are recorded.
- Focused grow-m validation passes.
- No public API, install-header, source-list, build metadata, or reviewed CTest
  drift occurs.
- Unproven eigensolver movement remains explicitly deferred.
