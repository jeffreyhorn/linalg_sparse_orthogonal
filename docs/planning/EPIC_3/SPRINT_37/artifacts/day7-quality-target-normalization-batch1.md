# Sprint 37 Day 7 Quality-Target Normalization Batch I

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Implement the first Makefile normalization batch from the Day 4 design by
making the maintained quality-target ownership model more explicit in comments,
section layout, and operator guidance, while reducing the visibility gap around
the Sprint 36 sanitizer/build-tree caveat without changing target behavior.

## Executive Summary

Day 7 did not add new quality targets.

It made the existing target surface easier to reason about by:

- signaling the Day 4 ownership categories directly in `Makefile`
- adding explicit `make clean` return guidance to the reviewed wrappers
- mirroring the same operator rule in `README.md`

This was the right first normalization batch because the repo’s problem was
ambiguity and flatness, not missing functionality.

## What Changed

### 1. Category signaling in `Makefile`

The quality surface is now commented and grouped around the three Day 4
classes:

- maintained operator entry points
- helper / prerequisite plumbing
- tree-mutating instrumentation or alternate-build modes

This makes it clearer that:

- reviewed wrappers and direct gates are the stable maintainer entry points
- helper targets support that contract rather than compete with it
- sanitizer / OMP / coverage flows are intentionally different operational
  modes

### 2. Reviewed wrapper guidance now exposes the canonical reset

The reviewed wrappers now explicitly tell operators:

- if returning from `sanitize`
- `asan`
- `sanitize-all`
- `tsan`
- `omp`
- or `coverage*`

reset first with:

```bash
make clean
```

This is additive guidance only. The wrappers still run the same lower-level
commands they did before.

### 3. README now matches the same operator rule

The operator command map in `README.md` now includes a short “tree-mutating
local modes” section listing:

- `make sanitize`
- `make asan`
- `make sanitize-all`
- `make tsan`
- `make omp`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

and the same explicit return rule:

```bash
make clean
```

This keeps the wrapper banners and the written operator docs aligned.

## Why This Was The Right Scope

Day 4 explicitly rejected:

- renaming established targets
- inventing a second reset alias
- reopening the Sprint 34 or Sprint 36 contract semantics

Day 7 followed that rule.

It improves:

- clarity
- ownership signaling
- operator safety

without changing:

- `lint`
- `test`
- `check`
- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`
- `deadcode-report`
- `deadcode-check`

## Validation

This batch changed `Makefile` and `README.md` only.

Because no `*.c` or `*.h` files changed, the full `make format && make lint &&
make test` gate was not required for the review-comment rule used elsewhere in
the sprint.

The touched paths were validated directly with dry-run checks:

- `make -n quality-review-compile`
- `make -n quality-review`
- `make -n quality-review-cmake-compile`
- `make -n sanitize`
- `make -n coverage-gcovr`

These checks confirmed:

- wrapper order is unchanged
- lower-level invoked commands are unchanged
- tree-mutating targets still own `clean` on entry
- the new reset guidance is visible in the reviewed wrapper path

## Residual Queue After Day 7

Still deferred:

- any deeper physical reordering of target blocks beyond the first ownership
  pass
- any further cleanup inside the coverage / instrumentation area beyond
  category signaling
- any later build-tree isolation work for alternate modes or dead-code

Those remain later work because Day 7’s job was to improve clarity first, not
to expand or rename the quality surface.

## Day 7 Conclusion

Day 7 converted the Day 4 normalization design into a real first Makefile
batch without changing the maintained quality contract.

The repo now signals more clearly which quality targets are:

- stable top-level operator entry points
- helper plumbing
- or intentionally tree-mutating alternate-build modes

and the Sprint 36 sanitizer/build-tree caveat is now surfaced directly in the
reviewed wrapper/operator flow through the existing canonical reset:

- `make clean`
