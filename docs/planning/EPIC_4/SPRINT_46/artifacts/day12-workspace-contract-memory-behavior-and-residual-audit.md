# Sprint 46 Day 12 Artifact: Workspace Contract, Memory Behavior, and Residual Audit

## Purpose

Document the internal eigensolver workspace/state contract now established by
Sprint 46, classify the residual repeated-allocation seams still visible in the
eigensolver surface, and fix the exact Day 13 validation sweep shape before
sprint closeout.

## Main Day 12 Conclusion

Sprint 46 now has a clear internal reusable eigensolver workspace contract, and
the remaining eigensolver queue is no longer a generic “allocation hotspot”
concern. It reduces to:

- migrated direct reusable-workspace paths already landed
- family-local helper/state seams that are intentionally still local
- benchmark/example/public-buffer surfaces that are not workspace-owner gaps
- explicit non-goals that belong to later Epic 4 work

That is the right Day 12 state for a validation-and-closeout finish.

## Internal Workspace/State Contract

### 1. The contract is now centered on a private shared owner plus typed views

The live private workspace contract is:

- `src/sparse_eigs_workspace_internal.h`
- `src/sparse_eigs_workspace_internal.c`
- `src/sparse_eigs_internal.h`

The shared owner now holds:

- contiguous `double` / `idx_t` / `int` storage
- checked capacity metadata
- dimension/family capacity tracking:
  - `n_capacity`
  - `lanczos_capacity`
  - `restart_capacity`
  - `block_capacity`

Typed reusable views are prepared from that owner for:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

Interpretation:

- the workspace layer is now a real internal contract surface
- later Epic 4 work can extend it without inventing a new ownership model

### 2. One-shot public APIs remain compatibility wrappers

The touched public eigensolver entry now follows the same compatibility pattern
as Sprint 45’s iterative work:

- `sparse_eigs_sym(...)`
  - remains the one-shot public entry
  - delegates to the shared implementation with local one-shot workspace
    ownership
- `sparse_eigs_sym_with_workspace_internal(...)`
  - is the bounded internal repeated-run seam
  - reuses a caller-owned `sparse_eigs_workspace_t`

Interpretation:

- Sprint 46 preserved the public one-shot API model
- repeated-run reuse exists internally without forcing a public explicit
  workspace API in the same sprint

### 3. The shared workspace layer owns storage/layout, not eigensolver policy

The shared workspace seam owns:

- storage ownership
- checked reserve/grow behavior
- typed view derivation
- cheap reuse across stable dimensions
- zero/reset behavior for reused buffers

It does **not** own:

- convergence policy
- Ritz extraction policy
- restart/locking policy
- refinement policy
- shift-invert factor ownership
- public result-buffer ownership
- benchmark/reporting policy

Interpretation:

- maintainers should treat this as a storage/layout layer
- eigensolver policy should remain local to the solver implementations

## Migrated Direct Workspace-Reuse Set

The direct reusable-workspace adoption set after Day 11 is:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

Interpretation:

- Sprint 46 materially covered the main repeated-allocation eigensolver families
  it chose
- there is no hidden remaining grow-m/thick-restart/LOBPCG migration queue
  inside this sprint

## Residual Eigensolver Classification

### Residual Class A: family-local helper scratch and restart state

The following still own local allocations inside `src/sparse_eigs.c`:

- refinement scratch in `s29_refine_eigenpairs(...)`
- dense Jacobi helper scratch
- arrowhead-to-tridiagonal helper scratch
- `lanczos_restart_state_t` owned locked/restart buffers

Interpretation:

- these are real allocations, but they are not the main repeated-run bundle the
  shared owner was created to absorb
- they belong to later specialization/unification work, not Sprint 46
  incompleteness

### Residual Class B: public-buffer and benchmark/example surfaces

- caller-owned `sparse_eigs_t` output buffers
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/example_eigs.c`

Interpretation:

- these are composition/evidence surfaces, not workspace-owner gaps
- Sprint 46 intentionally does not turn them into a public explicit workspace
  API or a broad benchmark framework

### Residual Class C: later Epic 4 non-goals

- public explicit eigensolver workspace APIs
- broad benchmark CLI redesign
- broad public docs/tutorial refresh for repeated-run guidance
- corpus-wide repeated-run benchmark expansion

Interpretation:

- Sprint 46 intentionally does not solve these
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

The targeted follow-ons justified by the touched Sprint 46 surface are:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Interpretation:

- the validation floor is explicit
- the follow-ons are bounded to the sprint’s real touched surfaces

## Sprint 46 Non-Goals

Sprint 46 intentionally does **not** solve:

- public explicit eigensolver workspace APIs
- broad benchmark CLI redesign
- broad README/tutorial repeated-run guidance
- broad public repeated-run benchmark coverage beyond the landed narrow A/B
  proof

Interpretation:

- the sprint stays internal-first and compatibility-preserving
- later Epic 4 work can widen these surfaces deliberately instead of inheriting
  accidental churn

## Bottom Line

Day 12 makes the Sprint 46 end-state much clearer:

- the internal reusable-workspace/state contract is now explicit
- the direct migration set is complete enough to stop broad eigensolver
  workspace churn
- the remaining eigensolver seams are classified rather than implicit
- the sprint’s non-goals are explicit
- Day 13 now has a concrete validation plan

That is the right documentation and residual-audit handoff for Sprint 46.
