# Sprint 74 Day 11: Docs / Packaging / Test Alignment Batch

## Objective

Align the minimum maintained public and policy surfaces with the landed Sprint
74 capability contract, without widening into install/package churn or broader
capability claims.

## Touched Surfaces

- `README.md`
- `docs/maintainer_guide.md`

## What Landed

### 1. `README.md` now states the current shipped width/scalar contract directly

The Known Limitations section now says plainly that:

- reviewed builds still default to the 32-bit `idx_t` lane
- wider indices are the bounded compile-time `SPARSE_IDX_BITS=64` seam
- downstream callers must rebuild against that same width contract
- current shipped scalar support remains real-only
- `sparse_scalar_t` is the public dense-scalar owner on the touched iterative
  and eigs seam, but only as bounded preparation for later widening

### 2. `docs/maintainer_guide.md` now owns the narrower Sprint 74 interpretation

The maintainer guide now states directly that:

- Sprint 74 moved bounded width/scalar seams only
- reviewed builds still default to the 32-bit `idx_t` lane
- `SPARSE_IDX_BITS` is the bounded compile-time width contract
- `sparse_scalar_t` is the touched public dense-scalar owner while shipped
  scalar support remains real-only `double`
- later scalar breadth and later algorithm-family widening remain deferred

It also names the focused proof owners directly:

- `tests/test_sparse_matrix.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

## What Did Not Need Follow-Through

The Day 10 support-only map held:

- `INSTALL.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

No install/export, reviewed-platform, header-truthfulness, or example-adoption
claim had drifted after the Day 6 and Day 9 landings.

## Preserved Truthfulness

This batch preserves the exact Sprint 74 fence:

- no broadened reviewed-platform or install/export claim
- no fake 64-bit-complete story
- no fake complex-readiness or broad generic-scalar story
- proof ownership remains with the focused test surfaces, not with docs or
  examples

## Sanity Checks

This was a docs-only batch. I used the targeted sanity set instead of code-day
validation:

- diff review
- terminology/alignment checks
- touched-surface `wc -l`
- branch-state verification

Touched-surface raw `wc -l` counts:

- `README.md` = `1044`
- `docs/maintainer_guide.md` = `670`

## Bottom Line

Sprint 74 Day 11 stayed fully inside the Day 10 fence:

- one caller-facing follow-through surface
- one maintainer/policy follow-through surface
- no broader docs, packaging, header, example, or proof-owner churn
