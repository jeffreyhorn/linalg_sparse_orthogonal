# Sprint 45 Day 12 Artifact: Workspace Contract and Residual Audit

## Purpose

Document the internal iterative workspace contract now established by Sprint 45,
classify the residual repeated-allocation seams still visible in the iterative
surface, and fix the exact Day 13 validation sweep shape before sprint closeout.

## Main Day 12 Conclusion

Sprint 45 now has a clear internal reusable-workspace contract, and the
remaining iterative queue is no longer a generic “allocation hotspot” concern.
It reduces to:

- migrated direct reusable-workspace paths already landed
- wrapper/composition surfaces that are no longer primary workspace targets
- specialized later solver-local seams such as scalar MINRES and the separate
  BiCGSTAB workspace precedent
- explicit non-goals that belong to later Epic 4 work

That is the right Day 12 state for a validation-and-closeout finish.

## Internal Workspace Contract

### 1. The contract is now centered on a private shared owner plus typed views

The live private workspace contract is:

- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_iterative_internal.h`

The shared owner now holds:

- contiguous `double` / `int` storage
- checked capacity metadata
- dimension/restart/nrhs capacity tracking

Typed reusable views are prepared from that owner for:

- CG
- GMRES
- block CG
- MINRES

Interpretation:

- the workspace layer is now a real internal contract surface
- later Epic 4 work can extend it without inventing a new ownership model

### 2. One-shot public APIs remain compatibility wrappers

The touched scalar one-shot entries now follow the same pattern:

- initialize a local internal workspace
- delegate to the reusable internal solver seam
- free the workspace before returning

This is now true for:

- `sparse_solve_cg(...)`
- `sparse_solve_gmres(...)`
- `sparse_solve_gmres_mf(...)`

Interpretation:

- Sprint 45 preserved the public one-shot API model
- repeated-solve reuse exists internally without forcing a public explicit
  workspace API in the same sprint

### 3. The shared workspace layer owns storage/layout, not solver behavior policy

The shared workspace seam owns:

- storage ownership
- checked reserve/grow behavior
- typed view derivation
- cheap reuse across stable dimensions

It does **not** own:

- recurrence scalars
- residual-history policy
- stagnation tracking
- callback/progress semantics
- preconditioner behavior
- block-wrapper orchestration

Interpretation:

- maintainers should treat this as a storage/layout layer
- solver policy should remain local to the solver implementations

## Migrated Direct Workspace-Reuse Set

The direct reusable-workspace adoption set after Day 11 is:

- scalar CG
- matrix-free CG
- scalar GMRES
- matrix-free GMRES
- block CG

Interpretation:

- Sprint 45 materially covered the main repeated-allocation targets it chose
- there is no hidden remaining CG/GMRES migration queue inside this sprint

## Residual Iterative Classification

### Residual Class A: wrapper/composition surfaces

- block GMRES
- block MINRES
- block BiCGSTAB

These now behave primarily as per-column compatibility wrappers over scalar
solves.

Interpretation:

- they are no longer the strongest next workspace-reuse targets
- they should be evaluated mainly as composition surfaces, not as separate
  storage-owner problems

### Residual Class B: specialized later solver-local workspace seams

- scalar MINRES
  - still owns a local packed `work` allocation inside
    `sparse_solve_minres(...)`
- scalar BiCGSTAB
- matrix-free BiCGSTAB
  - already use the separate `bicgstab_workspace_t` precedent in
    `src/sparse_bicgstab_internal.h`

Interpretation:

- scalar MINRES is the clearest still-local repeated-allocation seam left in
  `src/sparse_iterative.c`
- BiCGSTAB is not an “unmigrated gap”; it already has a valid separate internal
  workspace pattern and belongs to a later unification/evolution discussion

### Residual Class C: later Epic 4 non-goals

- eigensolver repeated-run workspace reuse
- public explicit iterative workspace APIs
- broader README/tutorial refresh for repeated-solve guidance
- broader benchmark CLI modernization

Interpretation:

- Sprint 45 intentionally does not solve these
- they should be handed forward explicitly rather than treated as hidden sprint
  incompleteness

## Day 13 Validation Plan

The authoritative Day 13 sweep should be:

```bash
make format
make lint
make test
make quality-review-full
```

The targeted follow-ons justified by the touched Sprint 45 surface are:

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`
- `./build/bench_iterative_reuse`
- `./build/example_matrix_free`

Interpretation:

- the validation floor is explicit
- the follow-ons are bounded to the sprint’s real touched surfaces

## Bottom Line

Day 12 makes the Sprint 45 end-state much clearer:

- the internal reusable-workspace contract is now explicit
- the direct migration set is complete enough to stop broad workspace churn
- the remaining iterative seams are classified rather than implicit
- the sprint’s non-goals are explicit
- Day 13 now has a concrete validation plan

That is the right documentation and residual-audit handoff for Sprint 45.
