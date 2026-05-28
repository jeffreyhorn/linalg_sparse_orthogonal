# Sprint 46 Day 14: Closeout and Handoff

## Summary

Sprint 46 closes with a validated internal eigensolver workspace reuse package
plus bounded repeated-run benchmark evidence.

The sprint now hands off:

- a shared internal eigensolver workspace/state owner
- typed reusable workspace views for:
  - grow-m Lanczos
  - thick-restart Lanczos
  - LOBPCG
- migrated direct reusable-workspace paths across those three main eigensolver
  families
- a compatibility-preserving one-shot wrapper structure
- direct repeated-run benchmark evidence
- a measured Day 13 validated baseline

This is a real internal repeated-run efficiency handoff, not just a cluster of
allocation substitutions.

## What Sprint 46 Accomplished

### 1. Landed a shared internal eigensolver workspace layer

The shared owner now lives in:

- `src/sparse_eigs_workspace_internal.h`
- `src/sparse_eigs_workspace_internal.c`

That layer now owns:

- contiguous reusable storage
- checked capacity / reserve behavior
- typed view preparation

Typed reusable views now exist for:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

This gives later eigensolver work a real shared storage/layout seam rather than
forcing every family to manage one-shot packed allocations independently.

### 2. Migrated the main direct reusable-workspace eigensolver paths

Sprint 46 landed direct reusable-workspace adoption for:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

These are the main repeated-allocation targets Sprint 46 set out to address.

### 3. Preserved one-shot public API compatibility

Sprint 46 did **not** introduce a new public explicit workspace API.

Instead, the public eigensolver entry remains compatibility-oriented:

- `sparse_eigs_sym(...)`
  - validates input
  - chooses backend
  - delegates into the reusable internal workspace seam

This preserved the public API model while still creating a real internal
repeated-run path for benchmarking and later extension.

### 4. Added direct repeated-run benchmark evidence

Sprint 46 added:

- `benchmarks/bench_eigs_reuse.c`

That benchmark now provides a bounded A/B comparison between:

- repeated one-shot public eigensolver calls
- repeated reusable-workspace-backed internal eigensolver calls

for:

- grow-m Lanczos
- thick-restart Lanczos

The benchmark outcome is now clear and stable enough to document honestly:

- convergence behavior matched exactly on the benchmarked cases
- the runtime effect is measurable but modest
- the timing direction can vary across local reruns

This is the right claim-safe Sprint 46 benchmark outcome.

## Final Validated Baseline

Sprint 46 closes from the Day 13 measured baseline:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Direct touched-surface reruns also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

## Residual Queue for Later Epic 4 Work

Sprint 46 intentionally does **not** finish all remaining eigensolver
repeated-run efficiency work.

The main later inherited queues are now:

- family-local helper/state cleanup that remains intentionally local:
  - refinement scratch
  - dense Jacobi scratch
  - arrowhead/tridiagonal helper scratch
  - `lanczos_restart_state_t` internal restart state
- any future public explicit eigensolver workspace API only when later work
  takes that outward-facing scope on directly
- broader benchmark/doc surfaces only when later work takes that outward-facing
  scope on directly

The main later outward-facing non-goals left deliberately untouched are:

- broader benchmark CLI modernization
- README/tutorial/public repeated-run guidance refresh
- broader public explicit workspace-handle design

These are deliberate Sprint 46 boundaries, not regressions or newly discovered
cleanup debt.

## `PROJECT_PLAN.md` Check

Sprint 46 did not surface any new deferred work beyond the later eigensolver
workspace, public API, and benchmark/doc modernization queue already implied by
the Epic 4 roadmap.

No `PROJECT_PLAN.md` update was needed at closeout.

## Bottom Line

Sprint 46 leaves behind the first real reusable-workspace package for
eigensolver repeated runs:

- shared internal workspace/state owner
- migrated primary eigensolver workspace paths
- compatibility-preserving one-shot wrapper structure
- direct repeated-run benchmark evidence
- validated local reviewed baseline preserved

That is the correct Sprint 46 handoff for later advanced repeated-run
efficiency and any future outward-facing eigensolver workspace work.
