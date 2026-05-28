# Sprint 46 Day 2 Artifact: Eigensolver Workspace Seam Inventory

## Purpose

Refresh the live workspace/allocation seam map inside `src/sparse_eigs.c`
before Sprint 46 chooses the reusable-workspace/state migration order.

## Main Seam Classes

The current eigensolver subsystem reduces cleanly to four workspace seam
classes.

### 1. Shared spectral helper/support paths

This is the shared support layer that multiple eigensolver families depend on:

- `lanczos_iterate_op(...)`
- Ritz extraction / selection helpers
- dense arrowhead helpers
- dense Jacobi helpers
- restart-state assembly / cleanup support
- residual recomputation support

This layer is important, but it is not identical to the solver-family reusable
workspace layer.  Some of it is algorithm support rather than repeated-run
state ownership.

### 2. Grow-m Lanczos path

The grow-m path inside `sparse_eigs_sym(...)` still owns one-shot buffers for:

- full basis `V`
- starting vector `v0`
- `alpha`
- `beta`
- `theta_long`
- `subdiag`
- `Y_long`
- `sel_idx`

This is the cleanest first repeated-run workspace target because its main
allocation model is a straightforward full-basis plus spectral-scratch bundle.

### 3. Thick-restart Lanczos path

The thick-restart family already has partial state ownership via:

- `lanczos_restart_state_t`
- `lanczos_restart_state_free(...)`

But the main repeated-run buffers are still one-shot inside
`s21_thick_restart_outer_loop(...)`:

- restart-phase basis `V`
- `alpha`
- `beta`
- `v0`
- `residual_vec`
- dense arrowhead `T_arrow`
- `theta_arrow`
- `Y_arrow`
- `sel_idx`
- temporary locked-state buffers

This makes thick-restart the strongest second Lanczos-family adoption target
after the shared owner lands.

### 4. LOBPCG path

LOBPCG carries the most specialized repeated-run state:

- `Q`
- `AQ`
- `G`
- `Y`
- `theta_full`
- `sel_idx`
- `X_new`
- optional `P_new`
- outer-loop `X`
- `R`
- `W`
- `AX`
- `theta`
- `converged`

This path has the richest optional/preconditioner-dependent behavior and should
follow the Lanczos-family adoption once the shared owner is already proven.

## Shared Extraction Targets

The strongest common extraction targets are:

- basis / vector bundles:
  - `n * m`
  - `n * k`
  - `n * block_size`
- tridiagonal / Ritz / restart scratch:
  - `alpha`
  - `beta`
  - `theta_*`
  - `subdiag`
  - `sel_idx`
- dense projected-subproblem intermediates:
  - `K * K`
  - `cap * cap`
- packed temporary bundles that can back typed solver views

These are the shapes the shared reusable-workspace/state layer should own.

## Solver-Local State That Should Stay Local

The shared owner should not absorb solver-local control/math state such as:

- grow-m Lanczos outer grow/retry policy
- Wu/Simon convergence decisions
- thick-restart lock selection and restart orchestration
- LOBPCG soft-lock policy
- LOBPCG RR-step sequencing
- preconditioner invocation logic

Sprint 46 should reuse buffer ownership without flattening the eigensolver
algorithm boundaries.

## Optional / Mode-Dependent Asymmetry

The main optional-buffer asymmetry is concentrated in LOBPCG:

- optional conjugate-direction block `P_new`
- preconditioner-dependent residual/update flow through `W`
- block-size-dependent dense intermediates

Grow-m Lanczos and thick-restart Lanczos are more regular and are therefore
better first adopters of the new shared owner.

## One-Shot Wrapper vs Reusable-Core Split

The public compatibility-facing one-shot entry remains:

- `sparse_eigs_sym(...)`

The reusable-core candidates sit below it:

- grow-m Lanczos internals
- thick-restart outer loop and restart-state support
- LOBPCG outer-loop and RR-step work buffers

This matches Sprint 45's preserved pattern: keep the public API one-shot and
route internal repeated-run work through reusable workspace/state owners.

## First Migration Order

The live file supports this bounded landing order:

1. shared eigensolver buffer/state owner
2. grow-m Lanczos migration
3. thick-restart Lanczos migration
4. LOBPCG migration

That order keeps the first shared landing pointed at the most regular repeated-
run seams before absorbing the more specialized optional/preconditioned LOBPCG
surface.
