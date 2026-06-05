# Sprint 55 Day 3 - `sparse_eigs.c` seam audit

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Reduce `src/sparse_eigs.c` to concrete extraction seams before any code
movement begins, then rank the bounded eigensolver module targets by real
ownership value.

## Live hotspot state

The main eigensolver hotspot sizes at audit time are:

- `src/sparse_eigs.c` = `3233`
- `src/sparse_eigs_internal.h` = `620`
- `tests/test_eigs.c` = `1522`
- `tests/test_eigs_lobpcg.c` = `1196`
- `benchmarks/bench_eigs_reuse.c` = `253`
- `examples/example_eigs.c` = `285`

Interpretation:

- `src/sparse_eigs.c` is still the clearest large-source maintainability target
- the strongest proof surfaces are already large enough that extraction work
  must preserve explicit parity rather than rely on compile-only confidence

## Ownership bands inside `src/sparse_eigs.c`

The live function map reduces to three major ownership bands:

### 1. Generic Lanczos and public-entry orchestration

Representative functions:

- `lanczos_iterate(...)`
- `lanczos_iterate_op(...)`
- `s20_ritz_pairs(...)`
- `s46_validate_public_entry(...)`
- `s46_run_growm_backend(...)`
- `s46_run_backend(...)`
- `s46_sparse_eigs_sym_impl(...)`
- `sparse_eigs_sym(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_sym_with_workspace_internal(...)`

This band owns:

- public validation and result setup
- backend selection and top-level dispatch
- one-shot vs handle entry routing
- shift-invert and shared outer orchestration

### 2. Thick-restart Lanczos and restart-state machinery

Representative functions:

- `lanczos_restart_state_free(...)`
- `s21_arrowhead_to_tridiag(...)`
- `lanczos_restart_pick_locked(...)`
- `lanczos_restart_state_assemble(...)`
- `lanczos_thick_restart_iterate(...)`
- `s21_dense_sym_jacobi(...)`
- `s21_build_dense_arrowhead(...)`
- `s21_recompute_residual(...)`
- `s21_thick_restart_outer_loop(...)`

This band owns:

- restart-state storage and lifecycle
- arrowhead construction and spectrum helpers
- thick-restart outer-loop execution
- thick-restart-specific residual and restart logic

### 3. LOBPCG backend

Representative functions:

- `s21_lobpcg_orthonormalize_block(...)`
- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`

This band owns:

- LOBPCG-specific orthonormalization
- block Rayleigh-Ritz step
- LOBPCG outer-loop execution

## Ranked seam candidates

### Rank 1: LOBPCG backend extraction

Why it ranks first:

- already a contiguous backend-owned region
- already grouped in `src/sparse_eigs_internal.h`
- directly covered by `tests/test_eigs_lobpcg.c`
- lower risk of public-contract drift than moving the outer orchestration layer

Expected maintainability win:

- turns one backend-owned block into its own module
- leaves `src/sparse_eigs.c` more focused on shared orchestration and generic
  Lanczos-family logic
- materially shrinks the main hotspot file

### Rank 2: Thick-restart restart-state / arrowhead cluster extraction

Why it ranks second:

- also contiguous and large
- has a real backend/family-local identity
- already has explicit internal types and helper declarations

Why it does not rank first:

- it is more entangled with generic Lanczos assumptions and the public
  dispatch/orchestration layer than the LOBPCG block is
- it is therefore a better second batch once the first extraction has reduced
  the main file's surface area

### Rank 3: Residual orchestration cleanup in the remaining `src/sparse_eigs.c`

Why it ranks third:

- the remaining public-entry / handle layer should stay in the main file in
  Phase 1
- smaller generic helper reshuffles only make sense after the clearly owned
  backend blocks move out

## Rejected extraction orders

### Reject: move the public-entry / handle layer first

Reason:

- it is the highest cross-cutting glue layer
- moving it first would create more churn than ownership clarity

### Reject: split only by helper count or comment density

Reason:

- that would reduce line count without necessarily improving who owns what
- Sprint 55 needs maintainability gains, not cosmetic file movement

## Internal-header and comment implications

`src/sparse_eigs_internal.h` is already broad enough to support a module split:

- generic Lanczos declarations
- thick-restart declarations and types
- LOBPCG declarations
- internal repeated-run workspace entry points

That means Sprint 55 does not need a new abstraction vocabulary before
extraction begins.

It also confirms a second Day 3 conclusion:

- touched eigensolver implementation files should lose stale sprint-history
  narrative while preserving durable algorithm commentary

## Proposed first extraction boundary

Sprint 55's first eigensolver extraction should be:

- move the LOBPCG backend-owned helpers and solver body out of
  `src/sparse_eigs.c`
- keep the public API surface, backend selection, and top-level dispatch in the
  residual `src/sparse_eigs.c`
- keep `tests/test_eigs_lobpcg.c` as the primary proof surface for the first
  batch

## Conclusion

Day 3 reduces the eigensolver decomposition problem to a concrete ranked map:

1. LOBPCG backend extraction first
2. thick-restart restart-state / arrowhead cluster second
3. residual orchestration cleanup after the backend-owned slices move

That gives Sprint 55 a real maintainability-first first target instead of a
generic “split `src/sparse_eigs.c` somehow” instruction.
