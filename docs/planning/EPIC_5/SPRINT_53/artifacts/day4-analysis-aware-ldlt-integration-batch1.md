# Sprint 53 Day 4: Analysis-Aware LDL^T Integration Batch I

## Purpose

Day 4 reduces the strongest Day 3 CSC ownership seam by unifying the
analysis-aware LDL^T CSC completion half that had still been duplicated between
the shared repeated-run path and the one-shot CSC dispatch path.

This is a bounded internal follow-through batch:

- no public direct-solver redesign
- no raw CSC/native storage exposure
- no broad contract rewrite

## Main Day 4 Result

Day 4 extracted and adopted a shared internal helper:

- `ldlt_csc_factor_with_resolved_analysis(...)`

That helper now owns the common CSC completion work once the LDL^T scalar
pre-pass and final analysis state have already been resolved:

- call `ldlt_csc_from_sparse_with_analysis(...)`
- seed the CSC factor's `pivot_size` from the resolved scalar pre-pass
- attempt `ldlt_csc_eliminate_supernodal(...)`
- fall back to the resolved scalar factor when the supernodal result is not
  retained
- write back through the public `sparse_ldlt_t` surface

The helper is now used from both:

- the one-shot CSC dispatch path in `src/sparse_ldlt.c`
- the shared repeated-run path in `src/sparse_analysis.c`

## Code Changes

### 1. Shared internal CSC completion helper

`src/sparse_ldlt_csc_internal.h` now declares:

- `ldlt_csc_factor_with_resolved_analysis(...)`

`src/sparse_ldlt.c` now implements it as the shared CSC completion helper for
the resolved-analysis case.

This helper intentionally assumes the caller has already handled the
indefinite-specific front half:

- scalar BK pre-pass
- final symmetric permutation resolution
- pre-permuted matrix creation when required
- direct caller-analysis reuse vs derived-analysis fallback

That keeps the helper narrow and honest about the real indefinite boundary.

### 2. One-shot LDL^T CSC path now uses the shared helper

`ldlt_factor_csc_path(...)` previously owned its own late-stage CSC completion
sequence after resolving the scalar pre-pass and building the pre-permuted
analysis state.

Day 4 replaced that duplicate tail with a call to:

- `ldlt_csc_factor_with_resolved_analysis(...)`

That keeps the one-shot path's dispatch behavior unchanged while reducing the
amount of duplicated CSC reasoning.

### 3. Shared repeated-run LDL^T CSC path now uses the same helper

`factor_ldlt_with_analysis_csc(...)` in `src/sparse_analysis.c` now also
delegates the resolved-analysis completion half through the same helper.

That means the shared repeated-run path and the one-shot CSC path still differ
only where they genuinely need to differ:

- whether the caller analysis can be reused directly
- whether a pre-permuted matrix plus derived analysis must be built

They no longer duplicate the later CSC completion tail once that analysis state
has already been settled.

## Preserved Contract

Day 4 intentionally preserved the bounded Sprint 52 semantics:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- reuse preserves symbolic/permutation setup, not stale numeric factor state
- the shared LDL^T path still reuses caller analysis directly only when the
  scalar BK pre-pass stays compatible with that analysis
- otherwise it still rebuilds analysis only on the pre-permuted matrix

This batch reduces internal duplication. It does not pretend the indefinite
path is as simple as the SPD Cholesky path.

## Regression Proof Added

Day 4 added focused integration proof in `tests/test_integration.c`:

- `test_ldlt_factor_opts_matches_explicit_analysis_path_indefinite_kkt`

The test uses an above-threshold indefinite KKT matrix so the CSC path is part
of the live behavior, then checks:

1. one-shot `sparse_ldlt_factor_opts(...)`
2. explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)`
3. solve parity against the same exact right-hand side

This is more direct proof than the pre-Day-4 state that the explicit shared
indefinite path and the one-shot CSC dispatch path remain behaviorally aligned
on an intended workload class.

## Validation

Because `*.c` / `*.h` changed, Day 4 ran the full required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Day 4 also ran the stronger reviewed baseline:

- `make quality-review-full`

That passed too.

Maintained truthfulness anchors after the batch:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 131.51 sec`

## What Day 4 Solved

- reduced duplicated LDL^T CSC completion logic between shared and one-shot
  paths
- made the shared-vs-one-shot orchestration split easier to reason about
- strengthened explicit indefinite public-path regression proof

## What Day 4 Did Not Solve

- the scalar BK pre-pass is still required
- derived-analysis fallback still exists when that pre-pass changes the final
  symmetric permutation
- there is still no LDL^T-specific factor-many benchmark equivalent to the
  Cholesky repeated-run proof

## Operational Result

Sprint 53 now has a cleaner indefinite CSC implementation base for the next
batch:

- the path exists
- the strongest completion-logic duplication seam is smaller
- the public explicit-analysis indefinite proof is stronger
- the strongest remaining follow-through work is now more clearly about deeper
  repeated-run behavior and proof, not duplicated CSC completion plumbing
