# Sprint 46 Day 6 Artifact: Lanczos Migration Batch 1

## Purpose

Convert the main thick-restart Lanczos outer-loop heap bundle to the shared
reusable internal workspace/state seam while preserving the already-migrated
grow-m path, the family-specific restart-state owner, and the existing one-shot
public behavior.

## Main Day 6 Conclusion

Sprint 46 now has both primary Lanczos families on the shared reusable
workspace/state seam:

- grow-m Lanczos from Day 5
- thick-restart Lanczos from Day 6

The Day 6 batch stayed intentionally bounded:

- migrated thick-restart outer-loop basis/scratch ownership
- preserved `lanczos_restart_state_t` as the family-specific state owner
- did not widen into LOBPCG
- did not add public APIs
- did not start benchmark or documentation closeout work yet

## Landed Migration

### Thick-restart outer loop now uses the shared owner/view seam

`s21_thick_restart_outer_loop(...)` now:

- initializes a private `sparse_eigs_workspace_t`
- prepares a typed `sparse_eigs_thick_restart_workspace_view_t`
- binds the former manual heap bundle through shared typed slices:
  - `V`
  - `alpha`
  - `beta`
  - `v0`
  - `residual_vec`
  - `T_arrow`
  - `theta_arrow`
  - `Y_arrow`
  - `sel_idx`
  - `V_locked_tmp`
  - `theta_locked_tmp`
  - `beta_coupling_tmp`
- frees the shared owner on all exits instead of manually freeing the former
  per-call bundle

### Family-specific restart state stayed separate

Day 6 intentionally did **not** fold the following into the shared owner:

- `lanczos_restart_state_t`
- `lanczos_restart_state_assemble(...)`
- `lanczos_restart_state_free(...)`
- lock/restart orchestration
- residual recomputation flow

Interpretation:

- the shared seam owns buffers and typed views
- the restart-state object still owns the thick-restart-specific control/state
  contract

## Batch Scope

### What Day 6 completed

- the primary thick-restart outer-loop allocation bundle migration
- the main two-family Lanczos workspace pairing:
  - grow-m
  - thick-restart

### Explicit non-goals for Day 6

This batch did **not** yet migrate:

- LOBPCG call sites
- public wrappers
- repeated-run benchmark code
- maintainer memory-contract closeout

Interpretation:

- the right Day 6 proof was to finish the main Lanczos-family workspace landing
  before broadening into LOBPCG or closeout work

## Validation

Because `*.c` changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

The stronger reviewed baseline for this shared-layer/multi-family Lanczos batch
also passed:

```bash
make quality-review-full
```

Targeted touched-surface follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

## Sprint 46 Position After Day 6

The next migration order is now clearer:

1. both primary Lanczos families already prove the shared owner in live paths
2. LOBPCG can adopt the already-landed owner/view model next
3. wrapper, benchmark, and memory-contract closeout can follow after the main
   eigensolver families are on the shared seam

## Bottom Line

Day 6 delivered:

- a workspace-backed thick-restart Lanczos outer loop
- preserved family-specific restart-state ownership
- a completed primary Lanczos-family shared-workspace landing
- a fully green reviewed validation baseline for the touched eigensolver paths

That is the right bounded Day 6 migration batch for Sprint 46.
