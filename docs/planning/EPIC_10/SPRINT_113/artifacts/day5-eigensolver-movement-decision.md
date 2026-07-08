# Day 5 Eigensolver Movement Decision

## Purpose

Day 5 decides whether the proven grow-m sizing and retry owner can move safely
after the Day 4 proof.  The decision must be evidence-backed, must avoid a
partial source split that hides coupling, and must leave Day 6 with an
executable movement or no-move path.

## Decision

Decision: **no source movement in Sprint 113 Day 6**.

Day 6 should publish a no-move contract for the grow-m owner and record the
future proof required before any private source extraction.

## Evidence Reviewed

Day 4 added direct proof for:

- default grow-m capacity and `peak_basis_size`;
- explicit grow-m capacity and `peak_basis_size`;
- too-small explicit `max_iterations` rejection;
- retry-boundary progress callback publication;
- accumulated iteration accounting relative to retry progress;
- retry-boundary cancellation cleanup.

Validation passed:

```sh
make build/test_eigs && build/test_eigs
make format && make lint && make test
```

No public API, install-header, helper-target, Make/CMake, source-list, or
reviewed CTest drift was introduced.

## Coupling Assessment

The selected grow-m owner spans these implementation surfaces:

| Surface | Movement readiness |
|---|---|
| `s49_eigs_effective_max_iters` | Small helper, but moving it alone would not move the selected behavior owner. |
| `s49_eigs_growm_capacity` | Small helper, but moving it alone would only relocate formula logic and leave retry behavior behind. |
| `s49_eigs_handle_prepare_backend` grow-m branch | Depends on backend selection and public handle workspace preparation.  Moving it would mix handle/workspace behavior already covered by Sprint 110's no-move contract. |
| `s46_run_growm_backend` | Core owner, but it depends on shared static Lanczos helpers and public result/status folding inside `src/sparse_eigs.c`. |
| `sparse_eigs_workspace_prepare_growm` | Already isolated in `src/sparse_eigs_workspace_internal.c`; no movement needed. |

`s46_run_growm_backend` currently depends on nearby static helpers and types
including:

- `lanczos_op_fn`;
- `lanczos_iterate_op`;
- `s20_lanczos_starting_vector`;
- `s20_ritz_pairs`;
- `s20_select_indices`;
- `s20_spectrum_scale`;
- `s20_lift_ritz_vectors`;
- `s29_eigs_now_s`.

Moving only the two sizing helpers would create a cosmetic split while leaving
the actual retry/progress/partial-result owner in the original large source.
Moving `s46_run_growm_backend` now would require exposing or relocating a
larger Lanczos helper cluster that Day 4 did not prove as a separate owner.

## Rejected Movement Options

| Option | Reason rejected |
|---|---|
| Move only `s49_eigs_effective_max_iters` and `s49_eigs_growm_capacity` | Too narrow; it would not materially reduce the grow-m owner boundary and would add source-list churn without moving the behavior proved on Day 4. |
| Move `s46_run_growm_backend` into a new grow-m source file immediately | Too broad; it requires exposing shared static Lanczos helpers or moving them together without direct helper-level proof. |
| Move grow-m workspace preparation | Already isolated in `src/sparse_eigs_workspace_internal.c`; no additional movement is needed. |
| Move public handle grow-m preparation | Duplicates or weakens Sprint 110's public handle/workspace no-move contract. |

## No-Move Contract

Until a later sprint proves the shared Lanczos helper boundary directly:

- keep `s46_run_growm_backend` in `src/sparse_eigs.c`;
- keep `s49_eigs_effective_max_iters` and `s49_eigs_growm_capacity` adjacent to
  public entry validation and grow-m dispatch;
- keep `sparse_eigs_workspace_prepare_growm` in
  `src/sparse_eigs_workspace_internal.c`;
- do not expose `lanczos_iterate_op`, Ritz selection helpers, spectrum scaling,
  vector lifting, or progress timing solely to enable a source split;
- preserve the Day 4 tests as the behavior guard for any future movement.

## Future Proof Requirements

Before moving grow-m execution into a separate private source file, a future
work item should add or identify direct proof for:

1. `lanczos_iterate_op` retry reproducibility under deterministic `v0`.
2. Ritz value ordering and selection for repeated or clustered spectra.
3. Ritz vector lifting with and without `compute_vectors`.
4. Partial-result publication after `m_cap` exhaustion.
5. Shift-invert eigenvalue conversion within the grow-m path.
6. Shared helper visibility rules that avoid expanding public API or install
   headers.
7. Make/CMake/source-list parity for any new private source file.

## Day 6 Execution Plan

Day 6 should execute the no-move path:

1. Create a no-move contract artifact.
2. Record current owner metrics:
   - `src/sparse_eigs.c` line count;
   - grow-m helper locations;
   - focused test names protecting the owner.
3. Confirm no Make/CMake/source-list/CTest registration drift.
4. Run focused validation:

```sh
make build/test_eigs && build/test_eigs
```

5. Run documentation checks for the Day 6 artifact:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_113
```

The full quality chain was already run after the Day 4 `.c` test change.  If
Day 6 changes code, Day 6 must rerun:

```sh
make format && make lint && make test
```

## Explicit Non-Claims

This decision does not claim:

- grow-m source decomposition is complete;
- shared Lanczos helpers are ready to move;
- backend dispatch is ready to move;
- shift-invert setup is ready to move;
- refinement defaults and budgets are ready to move;
- public handle/workspace movement is allowed;
- public API, install-header, package, or reviewed CTest changes are needed.

## Completion Criteria

- The movement decision is explicit: no move in Day 6.
- The reason is behavior-owner coupling, not lack of test coverage.
- Day 6 has an executable no-move contract path.
- Future movement requirements are recorded without duplicating completed
  Sprint 110 public handle/workspace work.
