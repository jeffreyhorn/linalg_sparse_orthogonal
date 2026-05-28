# Sprint 45 Day 5 Artifact: Shared Iterative Buffer Layer Batch 1

## Purpose

Land the first real Sprint 45 code batch by introducing the private reusable
iterative workspace owner and proving it in one bounded live solver path before
widening the migration to GMRES, block solvers, or benchmarks.

## Main Day 5 Conclusion

Sprint 45 now has a real shared iterative buffer layer, not just a design.

That layer is intentionally narrow in this first batch:

- private internal owner:
  - `src/sparse_iterative_workspace_internal.h`
  - `src/sparse_iterative_workspace_internal.c`
- maintained build wiring:
  - `Makefile`
  - `CMakeLists.txt`
- first live adoption:
  - scalar `sparse_solve_cg(...)`

The batch stayed within the Day 4 boundary:

- shared layer owns contiguous storage, checked reserve logic, and typed view
  preparation
- solver-local math, callbacks, stagnation policy, and recurrence logic stayed
  in `src/sparse_iterative.c`
- matrix-free CG, GMRES, block paths, and benchmark work remain later Sprint 45
  batches

## Landed Internal Workspace Layer

### New shared owner

`sparse_iter_workspace_t` now owns reusable backing storage for:

- double work buffers
- integer side buffers
- cached shape/capacity metadata for:
  - `n`
  - restart count
  - `nrhs`

This makes the common seam capacity-centric rather than solver-control-centric.

### New typed prepare helpers

The first shared helper surface now includes typed prepare helpers for:

- CG
- GMRES
- block CG
- MINRES

Interpretation:

- Sprint 45 does not need to widen public APIs to start getting reuse benefits
- the internal layer can prepare stable typed slices now and expand adoption
  incrementally

## First Live Adoption

### Scalar CG migrated

`sparse_solve_cg(...)` now:

- initializes a private `sparse_iter_workspace_t`
- prepares a typed `sparse_cg_workspace_view_t`
- binds:
  - `r`
  - `z`
  - `p`
  - `Ap`
- frees the shared owner on all exit paths instead of managing a raw local
  `work` bundle

### Explicit non-goals for Day 5

This batch did **not** yet migrate:

- `sparse_solve_cg_mf(...)`
- GMRES
- block CG
- MINRES call sites
- benchmark/example repeated-solve surfaces

Interpretation:

- the right Day 5 proof was the smallest high-value live path
- the batch proved the shared owner without broadening the migration surface too
  early

## Validation

Because `*.c` and `*.h` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Targeted touched-surface follow-ons also passed:

- `./build/test_iterative`
- `./build/test_stagnation`

One small implementation issue surfaced during the first lint pass:

- stale cleanup references in `src/sparse_iterative.c` from the scalar-CG
  migration boundary

That was fixed immediately, and the authoritative rerun from the top passed
fully.

## Sprint 45 Position After Day 5

The next migration order is now clearer:

1. matrix-free CG can adopt the same owner/view model
2. GMRES can reuse the already-landed typed prepare seam
3. block CG can adopt the shared double/int owner path
4. later benchmark work can then compare repeated one-shot solves against the
   new reusable internal path

## Bottom Line

Day 5 delivered:

- the first real shared iterative workspace owner
- typed internal solver views
- maintained build wiring
- a successful scalar-CG proof integration
- a fully green validation baseline for the touched iterative surface

That is the right bounded first code landing for Sprint 45.
