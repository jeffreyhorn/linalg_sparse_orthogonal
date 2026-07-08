# Day 2 Lanczos Iterate Behavior Proof Design

## Purpose

Day 2 designs the direct proof for `lanczos_iterate_op` behavior across the
three Sprint 114 dispatch surfaces: the grow-m Lanczos path, the thick-restart
path, and the LOBPCG-adjacent path that compares LOBPCG output against the
same Lanczos operator behavior. The design is intentionally test-only: Day 3
should add focused tests without public API changes, source movement,
source-list edits, helper-target edits, or reviewed CTest membership changes.

## Dispatch-Path Inventory

| Path | Current owner | Observable behavior to prove | Current adjacent coverage |
|---|---|---|---|
| Basic grow-m Lanczos | `src/sparse_eigs.c` `s46_run_growm_backend` calls `lanczos_iterate_op` with `s20_op_matvec` or shift-invert operator. | Deterministic recurrence, `n_converged`, selected eigenvalues, backend identity, iteration/basis accounting, and public residual. | `tests/test_eigs.c` diagonal, tridiagonal, grow-m retry, progress, shift-invert, and handle tests. |
| Thick-restart Lanczos | `src/sparse_eigs.c` dispatches to `s21_thick_restart_outer_loop`; empty-state iteration delegates to the same Lanczos recurrence. | Empty-state parity with `lanczos_iterate`, full backend parity with grow-m on a deterministic fixture, and bounded output consistency. | `tests/test_eigs_thick_restart.c` `test_thick_restart_iterate_empty_state_matches_lanczos`, `test_thick_restart_matches_grow_m`, and `test_thick_restart_single_phase_matches_grow_m`. |
| LOBPCG-adjacent | `src/sparse_eigs.c` dispatches to `s21_lobpcg_solve`; comparisons use grow-m Lanczos as the adjacent oracle. | LOBPCG and grow-m Lanczos agree on eigenvalues for the same public fixture while keeping LOBPCG backend identity and residual checks visible. | `tests/test_eigs_lobpcg.c` `test_lobpcg_vs_lanczos_laplacian`, diagonal LOBPCG tests, and LOBPCG residual/orthogonality helpers. |

## Existing Coverage to Preserve

| Existing test | Keep / extend | Reason |
|---|---|---|
| `test_growm_explicit_capacity_pins_peak_basis_size` | Keep; add a nearby direct behavior test if implementation needs grow-m accounting proof. | Already proves explicit `max_iterations` maps to `peak_basis_size` and stays on `SPARSE_EIGS_BACKEND_LANCZOS`. |
| `test_growm_retry_progress_steps_accumulate_iterations` | Keep; do not overload with recurrence proof. | Already proves grow-m retry progress and accumulated iteration visibility. |
| `test_thick_restart_iterate_empty_state_matches_lanczos` | Extend or add a sibling test. | This is the narrowest direct proof that thick-restart empty-state iteration matches the base Lanczos recurrence. |
| `test_thick_restart_single_phase_matches_grow_m` | Keep as end-to-end parity. | It proves small complete-basis parity through public backend dispatch. |
| `test_lobpcg_vs_lanczos_laplacian` | Extend or add a sibling test. | It is the right LOBPCG-adjacent surface because LOBPCG itself does not call `lanczos_iterate_op`. |

## Fixture Choices

| Fixture | File | Purpose | Parameters |
|---|---|---|---|
| Diagonal SPD `diag(1..6)` | `tests/test_eigs_thick_restart.c` | Direct recurrence parity for empty-state thick restart vs `lanczos_iterate`. | `n = 6`, `m = 6`, fixed `v0`, `reorthogonalize = 1`, tolerances `1e-14` for `V`, `alpha`, and `beta`. |
| Shifted tridiagonal SPD | `tests/test_eigs.c` | Basic grow-m public proof that `lanczos_iterate_op` behavior is visible through backend result fields and retry accounting. | `n = 64`, `k = 2`, `which = LARGEST`, `tol = 1e-30`, `backend = LANCZOS`, `max_iterations = 64`. |
| Laplacian tridiagonal | `tests/test_eigs_lobpcg.c` | LOBPCG-adjacent parity against grow-m Lanczos. | `n = 30`, `k = 4`, both `LARGEST` and `SMALLEST`, `tol = 1e-10`, `max_iterations = 200`, eigenvalue tolerance `1e-7`. |

The Day 3 implementation should reuse existing builders where present. If a
new direct helper is needed, keep it local to the existing test translation
unit and keep matrix values, tolerances, expected values, and iteration budgets
visible at the test call site.

## Focused Test Checklist

1. Basic grow-m Lanczos path:
   - Add a focused public test near the existing grow-m retry tests in
     `tests/test_eigs.c`.
   - Force `SPARSE_EIGS_BACKEND_LANCZOS`.
   - Assert `backend_used == SPARSE_EIGS_BACKEND_LANCZOS`.
   - Assert `n_converged` or bounded partial-result fields according to the
     selected fixture.
   - Assert visible `peak_basis_size`, `iterations`, and residual behavior
     rather than reaching into private workspace internals.
2. Thick-restart path:
   - Prefer extending `test_thick_restart_iterate_empty_state_matches_lanczos`
     only if the existing assertions are missing a public behavior signal.
   - Otherwise add a sibling test that compares `m_actual`, `V`, `alpha`, and
     `beta` between `lanczos_iterate` and `lanczos_thick_restart_iterate` on a
     different deterministic fixture.
   - Keep `reorthogonalize = 1` and use `1e-14` equality-style tolerances for
     direct recurrence arrays.
3. LOBPCG-adjacent path:
   - Keep the proof public and adjacent: LOBPCG does not directly call
     `lanczos_iterate_op`.
   - Add or extend a parity test in `tests/test_eigs_lobpcg.c` that compares
     LOBPCG against grow-m Lanczos on the same deterministic matrix.
   - Assert both backend identities, convergence counts, eigenvalue agreement,
     and Ritz residuals where eigenvectors are requested.

## Boundary and Failure Cases

| Case | Target | Expected proof |
|---|---|---|
| Invalid direct recurrence input | Existing internal Lanczos tests only if already present nearby. | Do not add new public API just to expose private bad-argument behavior. |
| Small complete-basis fixture | Thick-restart empty-state parity. | `m_actual == m`, `V`, `alpha`, and `beta` match `lanczos_iterate`. |
| Retry-bound grow-m fixture | Basic grow-m path. | Public `iterations` accumulates, `peak_basis_size` stays capped, result fields remain coherent. |
| LOBPCG comparison fixture | LOBPCG-adjacent path. | LOBPCG and grow-m Lanczos agree on public eigenvalues without claiming shared implementation ownership. |

## Proof Visibility Rules

- Keep all matrices, dimensions, `k`, tolerances, iteration budgets, expected
  eigenvalues, backend selections, and residual limits visible inside the
  tests.
- Do not extract a cross-file helper for this proof batch.
- Do not add a new public symbol, install header, source file, source-list
  entry, Make target, CMake target, or CTest registration.
- If a local helper is needed, keep it in the same test file and name it after
  the proof owner rather than a broad reusable abstraction.
- Treat LOBPCG coverage as adjacent parity only. Do not claim that LOBPCG uses
  `lanczos_iterate_op` internally.

## Day 3 Implementation Targets

| Target file | Proposed test action | Validation |
|---|---|---|
| `tests/test_eigs.c` | Add one focused grow-m public behavior proof near the existing grow-m retry tests. | `make test TEST=test_eigs` if supported, otherwise `make test` after implementation. |
| `tests/test_eigs_thick_restart.c` | Extend the empty-state parity proof or add a sibling deterministic recurrence parity proof. | Run the thick-restart test binary through the repo's focused path if available. |
| `tests/test_eigs_lobpcg.c` | Add or extend one LOBPCG-adjacent parity proof with backend identities and residual checks. | Run the LOBPCG test binary through the repo's focused path if available. |

Because Day 3 will modify `.c` tests, it must finish with
`make format && make lint && make test` before proceeding.

## Validation Commands for Day 2

Day 2 changes documentation only. The required checks are:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114
```

No C quality gate is required for Day 2 because no `.c` or `.h` file changes
are made.

## Completion Criteria

- Basic grow-m, thick-restart, and LOBPCG-adjacent paths have concrete Day 3
  test targets.
- Fixture dimensions, tolerances, iteration budgets, and expected observables
  are explicit.
- The design avoids public API, source-list, helper-target, and reviewed CTest
  drift.
- LOBPCG is documented as adjacent parity rather than direct
  `lanczos_iterate_op` ownership.
