# Sprint 67 Day 9: Shared ND Policy Convergence Batch

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Land the bounded shared ND policy convergence batch by moving the duplicated ND
compatibility/default-policy baseline to one internal owner while preserving
the shipped typed-analysis override contract.

## Landed Scope

Touched code surfaces:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`

Untouched by design:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `include/sparse_analysis.h`
- CSC backend files
- iterative/eigensolver files
- broader graph subsystem files

This stayed inside the exact Day 8 file fence.

## What Changed

### Shared internal owner

`src/sparse_reorder_nd.c` now exposes:

- `sparse_reorder_nd_default_policy()`

through `src/sparse_reorder_nd_internal.h` as the shared internal owner for:

- ND compatibility env-var parsing
- ND default-policy normalization

That keeps the direct `sparse_reorder_nd(...)` lane as the natural owner of the
legacy compatibility baseline.

### Analysis consumer convergence

`src/sparse_analysis.c` now starts `resolve_analysis_nd_policy(...)` from:

- `sparse_reorder_nd_default_policy()`

instead of duplicating its own ND compatibility parsers and hard-coded default
initialization for:

- root-bisect mode
- coarsening mode
- coarsest-bisection mode
- root-bisect max-n
- coarsen floor ratio
- coarsening CV fallthrough
- separator-lift strategy
- separator-lift weight

Typed analysis-option handling stays in `src/sparse_analysis.c`, so the public
analysis lifecycle still owns typed-field resolution while the shared helper now
owns the ND compatibility/default baseline.

## Preserved Contract

The landed batch preserves the shipped behavior fence:

- zero-init-safe `sparse_analysis_reorder_opts_t` still starts from the same
  effective ND compatibility/default baseline
- typed analysis values still override compatibility env vars exactly as
  shipped
- direct `sparse_reorder_nd(...)` still honors the compatibility path when no
  typed analysis layer is involved

One compatibility parser intentionally stayed local in `src/sparse_analysis.c`:

- `supernodal_postorder`

That field was not part of the Day 8 ND-policy convergence target, so keeping
it local avoids widening the batch into a broader analysis compatibility sweep.

## Maintainability Result

The strongest remaining ownership contradiction from Day 8 is now closed:

- one internal owner for ND compatibility/default-policy normalization
- one analysis consumer that layers typed values on top
- no second copy of the same ND parser/default logic inside
  `src/sparse_analysis.c`

So this lands a real maintainability reduction without widening into CSC,
iterative, eigensolver, or public-API redesign work.

## Validation Plan

Because this batch changed `*.c` / `*.h`, the required validation set is:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The existing proof homes remain sufficient unless validation reveals a concrete
behavior gap:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

## Exit State

Sprint 67 Day 9 closes with one bounded second-lane maintainability landing:

1. shared ND compatibility/default baseline owner:
   - `sparse_reorder_nd_default_policy()`
2. landed consumer convergence:
   - `src/sparse_analysis.c` now consumes that baseline
3. preserved behavior contract:
   - typed analysis values still win over compatibility env vars
4. non-widening fence held:
   - no CSC widening
   - no public API redesign
   - no extra proof-surface expansion
