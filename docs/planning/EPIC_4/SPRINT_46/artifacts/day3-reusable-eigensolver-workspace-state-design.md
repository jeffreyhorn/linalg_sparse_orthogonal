# Sprint 46 Day 3 Artifact: Reusable Eigensolver Workspace/State Design

## Purpose

Define the bounded internal reusable eigensolver workspace/state model that
Sprint 46 will implement before any code movement begins.

## Design Summary

Sprint 46 should use:

- one shared internal eigensolver storage owner
- typed solver-family views layered over that owner
- separate family-specific state objects where algorithm control/state must
  remain distinct from raw buffer ownership

This mirrors the internal-first pattern already proven in Sprint 45, but keeps
eigensolver-family state more explicit because thick-restart and LOBPCG carry
more algorithm-specific control than the migrated iterative paths did.

## Proposed Object Families

### 1. Shared eigensolver storage owner

The shared owner should manage:

- contiguous double-buffer ownership
- contiguous `idx_t` scratch ownership
- optional flag/int scratch ownership when the family needs it
- checked reserve/grow behavior
- tracked capacity for:
  - `n`
  - Lanczos basis/restart dimensions
  - block-size dimensions

Its job is capacity, ownership, and slicing. It does not own eigensolver
control flow.

### 2. Grow-m Lanczos view/state

This family should expose a typed view over the owner for:

- basis `V`
- starting vector `v0`
- `alpha`
- `beta`
- `theta_long`
- `subdiag`
- `Y_long`
- `sel_idx`

Solver-local state that remains outside the shared owner:

- active `m`
- `m_cap`
- grow/retry policy
- convergence bookkeeping

### 3. Thick-restart Lanczos view/state

This family should expose a typed view over the owner for:

- restart-phase basis `V`
- `alpha`
- `beta`
- `v0`
- `residual_vec`
- `T_arrow`
- `theta_arrow`
- `Y_arrow`
- `sel_idx`
- temporary locked-state bundles

Family-specific state that remains distinct:

- `lanczos_restart_state_t`
- `m_restart`
- `k_locked`
- restart orchestration / lock-selection logic

### 4. LOBPCG view/state

This family should expose a typed view over the owner for:

- `Q`
- `AQ`
- `G`
- `Y`
- `theta_full`
- `sel_idx`
- `X_new`
- outer-loop `X`
- `R`
- `W`
- `AX`
- `theta`
- optional `P` / direction storage
- convergence flags

Family-specific state that remains distinct:

- active `block_size`
- effective subspace size
- soft-lock policy state
- RR-step sequencing
- preconditioner-composed update flow

## Ownership and Lifecycle Rules

### Create

- zero-initialized owner/state is valid
- first prepare call performs allocation
- no separate public construction API is introduced

### Prepare

- typed prepare helpers receive dimensions and mode flags
- prepare helpers ensure capacity and populate typed slices
- prepare helpers may grow internal storage if current capacity is insufficient

### Reset Between Repeated Runs

- preserve capacity
- clear or reinitialize only the slices whose next run requires clean state
- do not preserve old Krylov/Ritz/search-direction mathematical state as a
  feature

This is reuse-by-capacity, not reuse-by-continuation.

### Resize

- resizing remains internal-only
- a larger repeated-run request may grow capacity
- no public "resize workspace" API is part of Sprint 46

### Destroy

- one internal free path returns the owner/state to zero form
- restart-state teardown must remain compatible with
  `lanczos_restart_state_free(...)`

## Optional and External-Dependency Rules

The reusable workspace/state model owns buffers only.

It does not own:

- shift-invert LDLT factors
- preconditioner callbacks
- preconditioner contexts
- operator callback contexts
- caller-owned result buffers

For optional buffers:

- LOBPCG direction/search buffers should be explicitly optional in the typed
  view/state contract
- thick-restart-specific extra buffers appear only when that family is
  prepared
- grow-m remains the simplest always-present bundle

## Internal-Only vs Wrapper-Facing Boundary

### Internal-only in Sprint 46

- shared eigensolver storage owner
- typed prepare helpers
- family-specific reusable view/state structs
- capacity helpers
- reset helpers

### Wrapper-facing but still internal

- one-shot `sparse_eigs_sym(...)` routing into reusable-core internals
- compatibility-preserving output/result handling

### Explicit non-goals for Sprint 46

- public workspace/state structs
- public init/free/reset APIs
- public repeated-run eigensolver handles
- public documentation that teaches a new explicit workspace API

## Expected Interaction Rules

### Stable-dimension repeated runs

This is the main Sprint 46 target:

- same `n`
- same or smaller `k`
- same or smaller restart/block settings

These workloads should reuse capacity without reallocating.

### Dimension growth

When `(n, k, restart, block_size)` exceed current capacity:

- internal prepare helpers may grow the workspace/state owner
- one-shot public behavior stays unchanged

### Shift-invert and preconditioned runs

- shift-invert and preconditioners compose with the reusable workspace/state
  model through callbacks and external factor/preconditioner contexts
- the workspace/state layer does not subsume those external dependencies

## First Implementation Order

The bounded landing order is:

1. shared eigensolver storage owner
2. grow-m Lanczos migration
3. thick-restart Lanczos migration
4. LOBPCG migration

This keeps the first code proof on the most regular repeated-run eigensolver
seams before absorbing LOBPCG's richer optional/preconditioned behavior.
