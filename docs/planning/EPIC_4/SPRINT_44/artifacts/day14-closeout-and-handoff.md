# Sprint 44 Day 14: Closeout and Handoff

## Summary

Sprint 44 closes with a validated Phase-2 decomposition of the residual graph
/ ND subsystem plus the first bounded large-test maintainability landing.

The sprint now hands off:

- an extracted FM refinement seam
- an extracted separator-policy / separator-lifting seam
- cleaned residual graph orchestration
- focused graph seam tests for the Phase-2 boundaries
- a first helper-consolidation proof point in `tests/test_qr.c`
- a measured Day 13 validation baseline

This is a real structural and maintainability handoff, not just a collection
of local cleanups.

## What Sprint 44 Accomplished

### 1. Extracted FM refinement

The FM refinement subsystem now lives in:

- `src/sparse_graph_refine.c`

That move pulled out:

- FM-local runtime state
- FM parser/helpers
- thick-restart perturbation support
- cut-weight evaluation
- bucket operations
- `graph_refine_fm(...)`

This removed the FM seam from exclusive dependence on the residual graph
orchestration file.

### 2. Extracted separator lifting and separator-policy selection

The separator-policy seam now lives in:

- `src/sparse_graph_separator.c`

That extraction moved:

- separator strategy / weight enums
- separator env-var parsers
- per-vertex separator scoring helpers
- `graph_edge_separator_to_vertex_separator(...)`

This gives later graph work a real separator-focused subsystem file rather than
forcing follow-on changes back through `src/sparse_graph.c`.

### 3. Preserved a coherent residual orchestration layer

The remaining `src/sparse_graph.c` is now much more clearly the residual layer
for:

- `graph_uncoarsen(...)`
- top-level partition orchestration
- retry/fallback glue
- orchestration-scoped runtime parsing

That is a better later-Epic-4 handoff point than the post-Sprint-43 residual
state.

### 4. Added focused Phase-2 seam protection

Sprint 44 added direct seam tests for:

- separator-policy behavior beyond the older smaller-side default
- post-split orchestration behavior under a stable non-default config

The explicit additions were:

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
- `test_partition_fifo_balanced_boundary_smoke`

These now protect the new extracted Phase-2 graph boundaries from silent
ownership drift in later work.

### 5. Landed the first bounded large-test maintainability batch

Sprint 44 also landed a small but real maintainability seam in:

- `tests/test_qr.c`

The new local helpers are:

- `assert_qr_reconstruction_below(...)`
- `assert_qr_true_residual_below(...)`

This reduced repetition in the largest maintained test binaries without
introducing a cross-file helper layer or changing the one-binary-per-test
model.

## Final Validated Baseline

Sprint 44 closes from the Day 13 measured baseline:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`

Direct touched-surface reruns also passed:

- `./build/test_graph`
- `./build/test_graph_fm_buckets`
- `./build/test_reorder_nd`
- `./build/test_reorder_amd_qg`
- `./build/test_qr`

## Residual Queue for Later Epic 4 Work

Sprint 44 intentionally does **not** finish all remaining graph cleanup or
large-test maintainability work.

The remaining later-phase graph queue is now:

- deeper residual orchestration simplification in `src/sparse_graph.c`
- any future retry/fallback glue sub-splitting only if later evidence justifies
  it
- any broader readability or policy cleanup no longer tied to a real ownership
  seam

The remaining large-test maintainability queue is:

- later helper/fixture consolidation candidates:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_svd.c`
- any future file splitting or broader domain-specific fixture extraction only
  when a later sprint takes that scope on directly

These are deliberate Sprint 44 deferrals, not regressions or newly discovered
cleanup debt.

## `PROJECT_PLAN.md` Check

Sprint 44 did not surface any new deferred work beyond the later graph and
large-test maintainability queue already present in Epic 4 planning.

No `PROJECT_PLAN.md` update was needed at closeout.

## Bottom Line

Sprint 44 leaves behind the Phase-2 completion of the graph decomposition plus
the first proof point for large-test maintainability work:

- FM refinement extracted
- separator-policy seam extracted
- residual graph orchestration narrowed and cleaned
- focused graph seam tests added
- QR test-helper consolidation landed
- validated local reviewed baseline preserved

That is the correct Sprint 44 handoff for later Epic 4 graph cleanup and
maintainability work.
