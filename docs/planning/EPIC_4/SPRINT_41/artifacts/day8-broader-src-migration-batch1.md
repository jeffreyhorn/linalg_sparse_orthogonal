# Sprint 41 Day 8 Artifact: Broader `src/` Migration Batch 1

## Purpose

Record the first broader `src/` migration batch after the completed first-wave
hotspot set by moving the two lowest-risk Day 7 targets onto the shared
allocation/overflow helper layer:

- `src/sparse_ic.c`
- `src/sparse_analysis.c`

## Batch Scope

Day 7's broader audit classified the remaining queue as:

- easy direct substitutions:
  - `src/sparse_ic.c`
  - `src/sparse_analysis.c`
- moderate helper-adapter cases:
  - `src/sparse_iterative.c`
  - `src/sparse_qr.c`
- specialized keep/defer:
  - `src/sparse_graph.c`

Day 8 intentionally stayed within the easy direct-substitution pair.

## `src/sparse_ic.c` Migration Result

### Shared helper adoption

Day 8 added the private helper include:

- `#include "sparse_alloc_internal.h"`

and replaced the direct workspace allocation seam for:

- `val`
- `pattern`
- `in_pat`

with:

- `sparse_calloc_array(...)`
- `sparse_malloc_array(...)`

### What changed semantically

Nothing algorithmic changed. The batch preserved:

- IC(0) symbolic/factor logic
- local cleanup ordering
- SPD/error behavior
- solve behavior

### Why this matters

This proves that the helper layer is not limited to the Day 4-6 hotspot set.
It now also owns the generic safety seam in a separate preconditioner/factor
module without forcing a redesign.

## `src/sparse_analysis.c` Migration Result

### Shared helper adoption

Day 8 added:

- `#include "sparse_alloc_internal.h"`

and migrated manual allocations in:

- `apply_supernodal_postorder(...)`
- permutation storage
- `etree` / `postorder` arrays
- `cc`
- `b_perm`
- `x_tmp`

to:

- `sparse_malloc_array(...)`

### What stayed local

The migration did **not** touch:

- reorder dispatch semantics
- symbolic-analysis meaning
- factor-type dispatch
- permutation interpretation
- factor-solve behavior

### Why this matters

This is the first broader proof in a bridge-style module that participates in
later Epic 4 lifecycle work. The batch improves generic safety consistency
without pulling Sprint 41 into public or architectural churn.

## Helper-Layer Pressure Result

Day 8 did **not** require a helper-layer extension.

The existing Day 4 API was sufficient for the broader pair:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

That is an important Sprint 41 result:

- the current helper layer is broad enough for the next mainline migrations
- the remaining pressure is migration-shape complexity, not missing primitives

## Residual Queue After Day 8

### Next mainline target

- `src/sparse_iterative.c`

### Optional bounded follow-on

- `src/sparse_qr.c`

### Explicit defer

- `src/sparse_graph.c`

Day 8 did not change the Day 7 queue order. It confirmed it.

## Validation Result

Because `*.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Highest-Value Conclusion

Sprint 41 now has a validated broader `src/` migration batch outside the
original hotspot list. `src/sparse_ic.c` and `src/sparse_analysis.c` both fit
the shared helper layer cleanly, and they did so without forcing any helper
API extension or broader lifecycle refactor. That leaves Day 9 free to tackle
the harder `src/sparse_iterative.c` seam from a stronger and cleaner baseline.
