# Sprint 41 Day 9 Artifact: Broader `src/` Migration Batch 2

## Purpose

Record the main broader `src/` consolidation closeout for Sprint 41 by landing
the next mainline target after the Day 8 pair:

- `src/sparse_iterative.c`

and by making the remaining local/deferred boundaries explicit.

## Batch Scope

Entering Day 9, the residual broader queue from Day 8 was:

- next mainline target:
  - `src/sparse_iterative.c`
- optional bounded follow-on:
  - `src/sparse_qr.c`
- explicit specialized defer:
  - `src/sparse_graph.c`

Day 9 intentionally completed the mainline target without widening the batch
to include `qr` or `graph`.

## `src/sparse_iterative.c` Migration Result

### Shared helper adoption

Day 9 added:

- `#include "sparse_alloc_internal.h"`

and moved the file's remaining generic safety seam onto the shared helper
layer across all its main workspace families:

- stagnation tracker allocation
- CG packed workspaces
- matrix-free CG packed workspaces
- GMRES fast-path scratch
- GMRES initial-residual scratch
- GMRES Hessenberg/Arnoldi packed workspace
- block-CG packed workspaces and per-column arrays
- MINRES packed workspace

The adopted helpers were:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`

### What changed semantically

Nothing algorithmic changed. Day 9 did **not** rewrite:

- Krylov method structure
- progress/cancel behavior
- residual-history handling
- block-solver semantics
- matrix-free dispatch meaning

The change stayed at the generic overflow/allocation boundary.

### Why this matters

This is the strongest Sprint 41 proof that the Day 4 helper layer is broad
enough for the current intended source-tree coverage. `src/sparse_iterative.c`
was the main moderate adapter-heavy case from Day 7, and it still fit the
helper layer without forcing an API redesign.

## Helper-Layer Sufficiency Result

Day 9 confirms that the current helper layer is sufficient for the main Sprint
41 intended coverage set.

Still sufficient:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

The remaining Sprint 41 pressure is now mostly about whether a file is worth a
bounded migration batch, not whether the helper layer lacks core primitives.

## Final Sprint 41 Broader `src/` Keep/Defer List

### Completed mainline broader targets

- `src/sparse_ic.c`
- `src/sparse_analysis.c`
- `src/sparse_iterative.c`

### Still local / deferred

#### `src/sparse_qr.c`

Status:

- still a real helper-alignment candidate
- intentionally left out of Day 9 to keep the mainline broader pass bounded

Why it remains deferred:

- its remaining raw allocations are broader and more mixed than the Day 9
  iterative seam
- including it would have turned Day 9 into a second wide multi-module batch

#### `src/sparse_graph.c`

Status:

- remains the strongest specialized keep/defer case

Why it remains deferred:

- allocation and scratch lifetimes are tightly bound to graph-structure and
  multilevel partitioner ownership
- helper substitution alone is not the whole maintainability problem there

## Validation Result

Because `src/sparse_iterative.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Highest-Value Conclusion

Sprint 41's main broader `src/` consolidation pass is now complete. The shared
helper layer has moved beyond the first-wave hotspot files and now covers the
main broader queue through `sparse_ic.c`, `sparse_analysis.c`, and
`sparse_iterative.c`. The remaining local surfaces are now explicit and
intentional: `sparse_qr.c` as a bounded later candidate, and `sparse_graph.c`
as a specialized later concern.
