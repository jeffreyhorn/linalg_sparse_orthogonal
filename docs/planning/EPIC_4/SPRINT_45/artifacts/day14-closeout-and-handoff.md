# Sprint 45 Day 14: Closeout and Handoff

## Summary

Sprint 45 closes with a validated internal iterative workspace reuse package
plus bounded repeated-solve benchmark evidence.

The sprint now hands off:

- a shared internal iterative workspace owner
- typed reusable workspace views
- migrated direct reusable-workspace paths for the main scalar and block-CG
  targets
- compatibility-preserving one-shot wrapper structure
- wrapper/composition cleanup for the remaining block convenience surfaces
- direct repeated-solve benchmark evidence for scalar CG and GMRES
- a measured Day 13 validated baseline

This is a real internal repeated-run efficiency handoff, not just a cluster of
allocation substitutions.

## What Sprint 45 Accomplished

### 1. Landed a shared internal iterative workspace layer

The shared owner now lives in:

- `src/sparse_iterative_workspace_internal.h`
- `src/sparse_iterative_workspace_internal.c`

That layer now owns:

- contiguous reusable storage
- checked capacity / reserve behavior
- typed view preparation

Typed view seams now exist for:

- CG
- GMRES
- block CG
- MINRES

This gives later iterative work a real shared storage/layout seam rather than
forcing every solver family to manage one-shot packed allocations independently.

### 2. Migrated the main direct reusable-workspace solver paths

Sprint 45 landed direct reusable-workspace adoption for:

- scalar CG
- matrix-free CG
- scalar GMRES
- matrix-free GMRES
- block CG

These are the main repeated-allocation targets Sprint 45 set out to address.

### 3. Preserved one-shot public API compatibility

Sprint 45 did **not** introduce a new public explicit workspace API.

Instead, the touched scalar public entries now clearly behave as
compatibility-oriented one-shot wrappers:

- initialize local internal workspace
- delegate to reusable internal solver seam
- free local workspace on return

This preserved the public API model while still creating a real internal
repeated-solve path for benchmarking and later extension.

### 4. Simplified the remaining block compatibility wrappers

Sprint 45 also normalized the wrapper/composition surfaces for:

- block GMRES
- block MINRES
- block BiCGSTAB

These remain per-column compatibility layers over scalar solves rather than
independent solver implementations.

That is an important maintainability handoff even though those surfaces are not
the primary reusable-workspace owners.

### 5. Added direct repeated-solve benchmark evidence

Sprint 45 added:

- `benchmarks/bench_iterative_reuse.c`

That benchmark now provides a bounded A/B comparison between:

- repeated one-shot public scalar solves
- repeated reusable-workspace-backed internal scalar solves

for:

- scalar CG
- scalar GMRES

The benchmark outcome is now clear and stable enough to document honestly:

- convergence behavior matched exactly on the benchmarked cases
- the runtime effect is measurable but modest
- the timing direction can vary across local reruns

This is the right claim-safe Sprint 45 benchmark outcome.

## Final Validated Baseline

Sprint 45 closes from the Day 13 measured baseline:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Direct touched-surface reruns also passed:

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`
- `./build/bench_iterative_reuse`
- `./build/example_matrix_free`

## Residual Queue for Later Epic 4 Work

Sprint 45 intentionally does **not** finish all remaining iterative
repeated-run efficiency work.

The main later inherited queues are now:

- scalar MINRES workspace migration / unification with the shared owner
- later unification or evolution of the separate BiCGSTAB workspace precedent
- eigensolver repeated-run workspace reuse
- any future public explicit iterative workspace API only when later work takes
  that outward-facing scope on directly

The main later outward-facing non-goals left deliberately untouched are:

- broader benchmark CLI modernization
- README/tutorial/public repeated-solve guidance refresh
- broader public explicit workspace-handle design

These are deliberate Sprint 45 boundaries, not regressions or newly discovered
cleanup debt.

## `PROJECT_PLAN.md` Check

Sprint 45 did not surface any new deferred work beyond the later iterative and
eigensolver repeated-run queue already implied by the Epic 4 roadmap.

No `PROJECT_PLAN.md` update was needed at closeout.

## Bottom Line

Sprint 45 leaves behind the first real reusable-workspace package for
iterative repeated solves:

- shared internal workspace owner
- migrated primary scalar and block-CG workspace paths
- compatibility-preserving one-shot wrappers
- normalized block wrapper/composition surfaces
- direct repeated-solve benchmark evidence
- validated local reviewed baseline preserved

That is the correct Sprint 45 handoff for later iterative and eigensolver
repeated-run efficiency work.
