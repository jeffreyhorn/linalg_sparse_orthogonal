# Sprint 55 Day 6 - eigensolver decomposition batch 2 design

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Freeze the second eigensolver extraction boundary using the actual post-Day-5
source shape, so the next batch improves the residual `src/sparse_eigs.c`
ownership map instead of drifting into generic cleanup.

## Post-Day-5 seam map

After the LOBPCG extraction, the strongest remaining owned block inside
`src/sparse_eigs.c` is the Sprint 21 thick-restart cluster:

- restart-state lifecycle:
  - `lanczos_restart_state_free(...)`
- restart-state / arrowhead helpers:
  - `s21_arrowhead_to_tridiag(...)`
  - `s21_pick_locked(...)`
  - `s21_recompute_residual(...)`
  - `s21_build_dense_arrowhead(...)`
- thick-restart execution:
  - `lanczos_thick_restart_iterate(...)`
  - `s21_thick_restart_outer_loop(...)`

This is now the clearest second-batch ownership seam.

## Selected second-batch landing

Sprint 55 Batch 2 should extract the thick-restart cluster into:

- `src/sparse_eigs_thick_restart.c`

## File-boundary ownership map

### Keep in `src/sparse_eigs.c`

- public one-shot and handle entry points
- shared validation and result setup
- backend AUTO/explicit selection
- generic Lanczos helpers
- grow-m Lanczos path
- shift-invert and shared operator composition
- top-level dispatch/orchestration call sites

### Move to `src/sparse_eigs_thick_restart.c`

- `lanczos_restart_state_free(...)`
- `s21_arrowhead_to_tridiag(...)`
- `s21_pick_locked(...)`
- `s21_recompute_residual(...)`
- `s21_build_dense_arrowhead(...)`
- `lanczos_thick_restart_iterate(...)`
- `s21_thick_restart_outer_loop(...)`

## Internal declaration strategy

Batch 2 should keep using:

- `src/sparse_eigs_internal.h`

and should not mix in:

- a new `src/sparse_eigs_thick_restart_internal.h`
- broad private-header narrowing
- generic helper relocation unrelated to the thick-restart move

Reason:

- the ownership win comes from moving the backend-owned thick-restart
  implementation out of the shared front-door file
- a second taxonomy redesign in the same batch would weaken that ownership
  proof

## Shared-helper non-goals

The following should stay shared in Batch 2:

- `s21_dense_sym_jacobi(...)`
- `s20_select_indices(...)`
- generic Lanczos and reorthogonalization helpers
- shared workspace preparation and public-handle orchestration

Reason:

- they are cross-backend helpers
- moving them would turn Batch 2 into a generic helper shuffle instead of a
  backend-owned extraction

## Comment policy for Batch 2

Preserve:

- durable algorithm meaning
- restart-state invariants
- arrowhead / Ritz / convergence semantics

Reduce where touched:

- Sprint 21 chronology
- “Day X did Y” landing-history prose
- historical planning-order comments that no longer explain present code truth

Do not attempt:

- repo-wide eigensolver comment normalization
- cleanup of untouched grow-m or public-entry commentary

## Expected Day 7 touched files

Primary expected touched set:

- `src/sparse_eigs.c`
- `src/sparse_eigs_thick_restart.c` (new)
- `src/sparse_eigs_internal.h`
- `CMakeLists.txt`
- `Makefile`

Secondary touch only if needed:

- `tests/test_eigs_thick_restart.c`

Avoid by default:

- `include/sparse_eigs.h`
- `src/sparse_eigs_workspace_internal.h`
- `tests/test_eigs.c`
- `tests/test_eigs_lobpcg.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/example_eigs.c`

## Validation checklist

Before calling Batch 2 complete:

1. The thick-restart implementation cluster lives in
   `src/sparse_eigs_thick_restart.c`.
2. `src/sparse_eigs.c` still reads primarily as public/shared orchestration plus
   the grow-m Lanczos path.
3. Shared cross-backend helpers remain in shared ownership.
4. No public header/API changes are introduced.
5. Touched comments reflect durable code truth rather than Sprint 21 history.
6. Validation passes:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
7. High-signal follow-ons remain green:
   - `./build/test_eigs`
   - `./build/test_eigs_thick_restart`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`
   - `./build/bench_eigs_reuse`

## Conclusion

Day 6 fixes the second eigensolver extraction boundary explicitly:

- move the thick-restart restart-state / arrowhead cluster into
  `src/sparse_eigs_thick_restart.c`
- keep the shared public-entry and non-thick-restart orchestration in
  `src/sparse_eigs.c`
- reuse the existing internal header for Phase 1
- validate primarily through `tests/test_eigs_thick_restart.c` plus the broader
  eigensolver parity surfaces

That gives Sprint 55 a concrete, landed-state-driven Day 7 implementation plan.
