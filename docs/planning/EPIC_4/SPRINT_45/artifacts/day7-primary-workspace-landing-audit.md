# Sprint 45 Day 7 Artifact: Primary Workspace Landing Audit

## Purpose

Audit the post-Day-6 iterative state so Sprint 45's remaining queue is driven by
the live code after the CG/GMRES workspace landings rather than by the broader
pre-implementation plan labels.

## Main Day 7 Conclusion

After Day 6, Sprint 45 no longer has a generic "iterative workspace migration"
queue.

It now has three distinct buckets:

- one real remaining multi-RHS workspace target:
  - block CG
- wrapper/composition paths that already mostly ride the migrated scalar seam:
  - block GMRES
  - block MINRES
  - block BiCGSTAB
- solver-local or specialized paths that should not be forced into the same
  Day 8 batch:
  - MINRES scalar workspace
  - BiCGSTAB's existing internal workspace precedent
  - support state such as stagnation/history tracking

That is the important Day 7 narrowing.

## Post-Day-6 State by Solver Family

### 1. CG family

Now on the shared workspace seam:

- `sparse_solve_cg(...)`
- `sparse_solve_cg_mf(...)`

Still outside the shared seam:

- `sparse_cg_solve_block(...)`

Interpretation:

- the CG family is no longer split between matrix-backed and matrix-free
  allocation models
- the only meaningful remaining CG migration target is the true multi-RHS block
  path

### 2. GMRES family

Now on the shared workspace seam:

- `sparse_solve_gmres_mf(...)`
- `sparse_solve_gmres(...)` via delegation

Residual block path:

- `sparse_gmres_solve_block(...)`

But the block GMRES path is currently a column loop over the scalar GMRES
entry point, not an independent packed block workspace implementation.

Interpretation:

- block GMRES is primarily a wrapper/composition surface
- it is not the main Day 8 workspace target

### 3. MINRES family

Still one-shot/local:

- `sparse_solve_minres(...)`
- `sparse_minres_solve_block(...)`

However the shape matters:

- scalar MINRES still owns a solver-specific 6/8-vector packed layout
- the block MINRES entry point is a column wrapper over the scalar solver

Interpretation:

- MINRES remains a legitimate later workspace candidate
- but it is not the strongest Sprint 45 Day 8 target compared with block CG

### 4. BiCGSTAB family

BiCGSTAB still sits in its own separate bucket:

- scalar and matrix-free BiCGSTAB already use `bicgstab_workspace_t`
- block BiCGSTAB is a wrapper-style per-column loop

Interpretation:

- Sprint 45 should treat BiCGSTAB as a precedent seam and compatibility check,
  not as the next main migration target

## Remaining Allocation Churn

### Strongest remaining direct target

`sparse_cg_solve_block(...)` still owns the clearest live repeated-allocation
bundle:

- `R`
- `Z`
- `P`
- `AP`
- `bnorms`
- `rz`
- `conv`
- `rnorms`

This aligns almost exactly with the already-landed
`sparse_block_cg_workspace_view_t`.

Interpretation:

- Day 8 should target block CG first
- the Day 5 typed view was already designed for this exact migration

### Lower-priority or later targets

- scalar MINRES one-shot packed workspace
- MINRES block wrapper path
- any further unification with BiCGSTAB workspace ownership

Interpretation:

- these remain valid later Sprint 45 or Sprint 46 concerns
- they are not needed to make Day 8 productive

## Wrapper vs Real Workspace Targets

### Real workspace migration target

- `sparse_cg_solve_block(...)`

### Mostly wrapper/composition surfaces

- `sparse_gmres_solve_block(...)`
- `sparse_minres_solve_block(...)`
- `sparse_bicgstab_solve_block(...)`

### Existing separate-workspace precedent

- `sparse_solve_bicgstab(...)`
- `sparse_solve_bicgstab_mf(...)`

Interpretation:

- the Day 8 batch should reduce real allocation churn, not just move wrapper
  code around
- block CG is the honest next internal workspace landing

## Internal Helper / Header Cleanup Notes

The current private helper surface already contains:

- `sparse_iter_workspace_prepare_block_cg(...)`
- `sparse_iter_workspace_prepare_minres(...)`

But live adoption is asymmetric:

- block-CG prepare helper is directly aligned with the next real migration
- MINRES prepare helper exists ahead of adoption and should remain internal-only
  until a later batch actually uses it

Interpretation:

- no new helper-layer redesign is needed before Day 8
- the main cleanup rule is simply to avoid widening the helper surface further
  until block CG lands

## Confirmed Day 8 Target Set

Day 8 should be bounded to:

1. `sparse_cg_solve_block(...)`
2. adoption of `sparse_block_cg_workspace_view_t`
3. preservation of current one-shot public behavior
4. no block GMRES / MINRES / BiCGSTAB churn unless a follow-on is obviously
   trivial

## Wrapper / Benchmark Follow-On Notes

After block CG lands, the remaining Sprint 45 queue should sequence as:

1. wrapper/composition review
2. repeated-solve benchmark evidence
3. optional later MINRES workspace extension only if it stays small

That keeps the sprint focused on measurable repeated-allocation reduction before
it shifts into comparison/benchmark proof.

## Bottom Line

Day 7 confirms:

- primary CG/GMRES workspace migration is complete
- block CG is the only strong remaining direct workspace target
- block GMRES / MINRES / BiCGSTAB are mostly wrapper/defer surfaces
- no new helper redesign is needed before Day 8

That is the right narrowed handoff for the next Sprint 45 batch.
