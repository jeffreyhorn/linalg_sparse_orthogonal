# Sprint 55 Day 4 - eigensolver decomposition batch 1 design

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Freeze the first eigensolver extraction boundary before editing permanent
implementation files, using the Day 3 LOBPCG-first ranking as the design
anchor.

## Selected first extraction seam

Sprint 55 Batch 1 should extract the LOBPCG backend into its own source file:

- new file:
  - `src/sparse_eigs_lobpcg.c`

The first moved function set should be:

- `s21_lobpcg_orthonormalize_block(...)`
- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`
- `s21_lobpcg_init_X(...)`

These functions already form a contiguous backend-owned block inside
`src/sparse_eigs.c` and have the strongest dedicated proof surface in
`tests/test_eigs_lobpcg.c`.

## File-boundary ownership map

### Keep in `src/sparse_eigs.c`

- public one-shot and handle entry points
- backend AUTO/explicit selection
- shared validation and result setup
- generic Lanczos helpers
- grow-m Lanczos path
- thick-restart path and restart-state machinery
- top-level backend dispatch/orchestration

### Move to `src/sparse_eigs_lobpcg.c`

- LOBPCG-specific block orthonormalization
- LOBPCG-specific Rayleigh-Ritz step
- LOBPCG outer-loop solver body
- LOBPCG deterministic initialization helper

## Internal declaration strategy

Sprint 55 Phase 1 should keep LOBPCG declarations in the existing:

- `src/sparse_eigs_internal.h`

and continue using the existing typed workspace views in:

- `src/sparse_eigs_workspace_internal.h`

Reason:

- the first batch already changes one major ownership axis:
  - source-file extraction
- adding a new private-header taxonomy in the same batch would combine:
  - source extraction
  - internal header redesign
- the current internal headers are already sufficient to support the move

Deferred by design:

- creation of a dedicated `src/sparse_eigs_lobpcg_internal.h`
- broader narrowing of `src/sparse_eigs_internal.h`

## Invariants the first batch must preserve

### Public contract invariants

- `sparse_eigs_sym(...)` behavior unchanged
- `sparse_eigs_sym_with_handle(...)` behavior unchanged
- `sparse_eigs_handle_prepare(...)` behavior unchanged
- one-shot vs handle relationship unchanged
- public options/result structs unchanged

### Backend-selection and reporting invariants

- explicit `SPARSE_EIGS_BACKEND_LOBPCG` routing unchanged
- AUTO routing unchanged
- `result->backend_used` unchanged
- progress/cancel behavior unchanged

### Workspace/reuse invariants

- `sparse_eigs_workspace_prepare_lobpcg(...)` contract unchanged
- zero-init/local-workspace fallback behavior unchanged
- handle/workspace reuse still preserves allocation/setup only
- no stale Ritz/search state reuse promise introduced

### Proof-surface invariants

- `tests/test_eigs_lobpcg.c` remains the primary direct proof surface
- public-handle LOBPCG proof in `tests/test_eigs.c` remains unchanged in
  meaning
- `benchmarks/bench_eigs_reuse.c` keeps explicit LOBPCG parity behavior
- `examples/example_eigs.c` keeps the same explicit LOBPCG workflow and output
  expectations

## Minimal comment policy for the first batch

Preserve:

- durable algorithm meaning
- invariants
- convergence semantics
- workspace ownership semantics

Reduce where touched:

- sprint chronology
- implementation-history narrative
- comments that explain landing order instead of present code truth

Do not try in Batch 1:

- repo-wide eigensolver comment normalization
- broad cleanup of untouched thick-restart or generic Lanczos sections

## Expected Day 5 touched files

Primary expected touched set:

- `src/sparse_eigs.c`
- `src/sparse_eigs_lobpcg.c` (new)
- `src/sparse_eigs_internal.h`

Secondary touch only if truly needed:

- `tests/test_eigs.c`

Avoid by default:

- `include/sparse_eigs.h`
- `src/sparse_eigs_workspace_internal.h`
- `tests/test_eigs_lobpcg.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/example_eigs.c`

## Landing checklist

Before calling Batch 1 complete:

1. LOBPCG helper/solver bodies live in `src/sparse_eigs_lobpcg.c`.
2. `src/sparse_eigs.c` still owns public orchestration and non-LOBPCG
   backends.
3. No public header/API changes are introduced.
4. Touched comments reflect durable code truth rather than sprint history.
5. Validation passes:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
6. High-signal follow-ons remain green:
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/example_eigs`
   - `./build/bench_eigs_reuse`

## Conclusion

Day 4 fixes the first eigensolver extraction boundary explicitly:

- move the LOBPCG backend into `src/sparse_eigs_lobpcg.c`
- keep the public-entry and non-LOBPCG orchestration in `src/sparse_eigs.c`
- reuse the existing internal headers for Phase 1
- preserve the full repeated-run/public/backend-reporting contract exactly

That gives Sprint 55 a concrete, bounded, maintainability-first Day 5 landing
plan.
