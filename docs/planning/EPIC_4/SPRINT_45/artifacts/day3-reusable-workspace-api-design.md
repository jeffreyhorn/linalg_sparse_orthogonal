# Sprint 45 Day 3 Artifact: Reusable Workspace API Design

## Purpose

Define the internal reusable workspace model for iterative solvers before
Sprint 45 begins implementation work.

## Main Day 3 Conclusion

Sprint 45 should not introduce one giant generic iterative context object.
The best bounded design is:

- one shared internal storage owner for contiguous reusable memory
- typed solver views layered over that owner
- one-shot public APIs preserved as wrappers that temporarily own or borrow the
  reusable internals

This stays consistent with:

- Sprint 41 shared allocation-helper rules
- Sprint 42 internal-first compatibility rules
- the live Day 2 seam map
- the existing BiCGSTAB internal workspace precedent

## Proposed Internal Object Model

### 1. Shared storage owner

The shared owner should be an internal-only object responsible for:

- contiguous `double` storage
- contiguous auxiliary integer/bool storage when needed
- capacity metadata for the largest prepared shape currently supported
- centralized create / destroy / reserve logic

The shared owner is not supposed to encode solver math state.

Its job is:

- size safely
- allocate once when possible
- reuse on repeated solves
- expose typed slices through prepare functions

### 2. Typed solver views

Solver code should consume typed views rather than manual offsets.

First-phase views:

#### CG view

- `r`
- `z`
- `p`
- `Ap`

#### GMRES view

- `v`
- `h`
- `cs`
- `sn`
- `g`
- `y`
- `w`

#### Block-CG view

- `R`
- `Z`
- `P`
- `AP`
- `bnorms`
- `rz`
- `conv`
- `rnorms`

#### MINRES view

- `v`
- `v_old`
- `w`
- `d0`
- `d1`
- `d2`
- optional `z`
- optional `z_tmp`

Interpretation:

- the shared layer owns storage and slicing
- solver code keeps typed names and direct pointer access

## Ownership and Lifecycle Rules

### Create / destroy

- create once at the owner boundary
- destroy once at the same boundary
- zeroed / empty owner remains safe to destroy

### Prepare

Each solver family should have an internal prepare helper that:

- validates the requested shape
- checks whether current capacity is sufficient
- grows storage if necessary
- returns a typed solver view

Prepare inputs should include the minimum shape facts that affect layout:

- `n`
- `restart`
- `nrhs`
- whether optional preconditioner vectors are required

### Reset between solves

Reset is per solve, not per allocation:

- working slices are reinitialized before each solve
- old solver math state is not preserved
- residual-history output remains caller-owned
- stagnation-tracker state resets every solve, even if its storage is reused

### Reuse rule

Reuse is allowed when:

- requested shape fits within current capacity
- the requested solver family's typed view can be prepared from the owner

Reuse does **not** imply:

- reusing old numerical Krylov state
- resuming a previous interrupted iteration
- preserving previous residual-history side effects

## Resize / Reject Contract

Sprint 45 should keep resize semantics internal and conservative:

- internal prepare helpers may grow capacity when a larger request arrives
- one-shot wrappers simply create/use/destroy their temporary owner
- there is no Sprint 45 public mismatch or "workspace too small" API
- overflow/allocation failure still return `SPARSE_ERR_ALLOC`

Interpretation:

- callers should not need new public lifetime management in Sprint 45
- repeated-solve efficiency is an internal capability in this sprint

## What Stays Internal-Only in Sprint 45

Internal-only:

- storage-owner type
- typed workspace-view types
- prepare/reset/resize helpers
- helper routines for contiguous slice derivation

Wrapper-facing but still not public API:

- convenience internal entry points that accept a prepared typed view
- one-shot public wrappers that allocate and free temporary workspace owners

Not in Sprint 45:

- public explicit workspace APIs
- public resize/mismatch semantics
- broad documentation/tutorial rewrite around new public workspace objects

## Preconditioner and Matrix-Free Interaction Rules

### Matrix-backed vs matrix-free

The same typed workspace view should serve:

- matrix-backed CG and matrix-free CG
- matrix-backed GMRES and matrix-free GMRES

The operator source changes, but the workspace shape does not.

### Optional preconditioner buffers

Optional preconditioner-dependent vectors should be handled in prepare-time
layout decisions, not with a separate workspace model.

Examples:

- MINRES optional `z` / `z_tmp`
- future extensions for other solver families if needed

### Callback and operator ownership

The shared storage owner should remain unaware of:

- matrix objects
- matvec callback pointers
- preconditioner callback pointers
- progress/logging callbacks

Those remain solver-entry concerns, not storage-owner concerns.

## Relationship to BiCGSTAB

BiCGSTAB already demonstrates that this codebase accepts:

- solver-owned internal workspace structs
- contiguous vector storage
- explicit alloc/free helpers

Sprint 45 should treat BiCGSTAB as:

- design evidence
- style precedent
- a likely future harmonization target

But not as:

- the first migration driver
- a required Day 5/Day 6 shared-owner adoption target

That keeps the first implementation batch focused on the larger one-shot
allocation seams in CG / GMRES / block CG / MINRES.

## Implementation Handoff

The next implementation steps should now be:

1. land the shared internal storage-owner and typed-view seam
2. adopt it first in scalar CG and matrix-free CG
3. extend it to GMRES
4. extend it to block CG
5. keep MINRES as the first bounded extension if the shared layer stays clean

## Bottom Line

Sprint 45 now has a concrete internal reusable workspace design:

- one shared storage owner
- typed solver views
- explicit create/prepare/reset/destroy rules
- internal-only resize semantics
- wrapper preservation
- BiCGSTAB treated as precedent rather than the first migration target

That is the right design boundary before implementation begins.
