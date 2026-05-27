# Sprint 45 Day 2 Artifact: Iterative Workspace Seam Inventory

## Purpose

Refresh the live iterative solver allocation and reuse map before Sprint 45
chooses the reusable-workspace landing order.

## Main Day 2 Conclusion

`src/sparse_iterative.c` no longer reads like one generic allocation hotspot.
It reduces cleanly to a small number of reusable-workspace seam classes:

- shared support state
- scalar CG family
- GMRES family
- block / multi-RHS family
- MINRES family
- existing separate-workspace BiCGSTAB precedent

This means Sprint 45 should design the workspace model around the actual packed
buffer families and wrapper boundaries already present in the file, rather than
around an abstract "all iterative solvers share one object" assumption.

## Live Seam Classification

### 1. Shared support seam

Current shared support state:

- `stag_tracker_t`
- `reshist_t`
- verbose/progress helpers

These are already reused across solver families and should remain compatible
with the future workspace model, but they are not themselves the main repeated
heap-allocation target.

### 2. Scalar CG family

Current scalar CG seam:

- `sparse_solve_cg(...)`
- `sparse_solve_cg_mf(...)`

Current repeated-allocation pattern:

- one packed allocation for:
  - `r`
  - `z`
  - `p`
  - `Ap`
- separate stagnation tracker allocation when enabled

Interpretation:

- CG is the cleanest first reusable-workspace target
- matrix-backed and matrix-free CG already share nearly the same workspace
  shape

### 3. GMRES family

Current GMRES seam:

- `sparse_solve_gmres(...)`
- `sparse_solve_gmres_mf(...)`

Current repeated-allocation pattern:

- two temporary `n`-vector early-exit checks
- one large packed bundle for:
  - Arnoldi basis `v`
  - Hessenberg `h`
  - Givens arrays `cs` / `sn`
  - Hessenberg RHS / triangular solve scratch `g` / `y`
  - temporary work vector `w`
- separate stagnation tracker allocation when enabled

Interpretation:

- GMRES is the most important large-workspace repeated-solve target
- the reusable model must support both scalar vectors and matrix-shaped
  Hessenberg/Arnoldi storage

### 4. Block / multi-RHS family

Current block seam:

- `sparse_cg_solve_block(...)`
- `sparse_gmres_solve_block(...)`
- `sparse_minres_solve_block(...)`
- `sparse_bicgstab_solve_block(...)`

But the internal behavior differs:

- block CG is a real direct workspace owner:
  - `R`
  - `Z`
  - `P`
  - `AP`
  - `bnorms`
  - `rz`
  - `conv`
  - `rnorms`
- block GMRES / MINRES / BiCGSTAB are currently per-column wrapper loops over
  the scalar solvers

Interpretation:

- block CG is the real block-workspace migration target
- the other block functions are primarily compatibility/composition surfaces in
  Sprint 45

### 5. MINRES family

Current MINRES seam:

- `sparse_solve_minres(...)`

Current repeated-allocation pattern:

- one packed bundle of:
  - `v`
  - `v_old`
  - `w`
  - `d0`
  - `d1`
  - `d2`
  - optional `z`
  - optional `z_tmp`
- separate stagnation tracker allocation when enabled

Interpretation:

- MINRES fits the same broad "packed vector bundle" model as CG
- its solver-state math remains more specialized, so the shared seam should
  stop at buffer ownership/slicing rather than trying to unify all recurrence
  state

### 6. BiCGSTAB precedent seam

Current BiCGSTAB seam:

- `sparse_solve_bicgstab(...)`
- `sparse_solve_bicgstab_mf(...)`
- `bicgstab_workspace_t`
- `bicgstab_workspace_alloc(...)`
- `bicgstab_workspace_free(...)`

Interpretation:

- BiCGSTAB is already the subsystem's one explicit internal workspace owner
- it is a design precedent, not the main first-phase migration target

## Shared vs Solver-Local Split

### Strongest shared extraction targets

The clearest shared work-buffer patterns are:

- packed graph-sized vector bundles
- matrix-free and matrix-backed variants sharing the same work layout
- optional preconditioner-dependent extra vectors
- block `n * nrhs` bundles
- checked contiguous bundle sizing and slice derivation

### State that should remain solver-local

Keep solver-local:

- recurrence scalars and stopping-state math
- GMRES restart/Arnoldi control decisions
- MINRES Lanczos / QR rotation state
- BiCGSTAB half-step stabilization and restart logic

Interpretation:

- the workspace model should share storage ownership and reset rules
- it should not attempt to turn every solver into the same control-state model

## Wrapper vs Reusable-Core Split

The strongest wrapper boundaries already visible are:

- matrix-backed GMRES over matrix-free GMRES
- per-column block GMRES over scalar GMRES
- per-column block MINRES over scalar MINRES
- per-column block BiCGSTAB over scalar BiCGSTAB

This means Sprint 45 can preserve the one-shot public API shape by keeping
wrapper layering explicit while moving the real repeated-allocation seams into
reusable internals.

## First Migration Order

### Extract-now / first-phase adoption targets

1. shared iterative buffer layer
2. scalar CG and matrix-free CG
3. GMRES and matrix-free GMRES
4. block CG

### Second-wave / bounded extension targets

- MINRES

### Use-as-precedent / defer-first targets

- BiCGSTAB

### Wrapper/composition surfaces

- block GMRES
- block MINRES
- block BiCGSTAB

## Bottom Line

The strongest Sprint 45 design target is now explicit:

- one reusable packed-buffer model for the main iterative one-shot allocation
  seams
- solver-local math state preserved locally
- wrapper layering preserved
- BiCGSTAB treated as precedent rather than as the first migration driver

That is the right Day 2 handoff into the reusable workspace API design work.
