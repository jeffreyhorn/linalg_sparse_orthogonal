# Sprint 41 Day 5 Artifact: Core Module Migration Batch 1

## Purpose

Record the first real hotspot migration batch for Sprint 41: removal of the
duplicated local multiply-overflow helper from the SVD/eigensolver pair and
migration of their shared arithmetic safety sites onto the Day 4 internal
utility layer.

## Batch Scope

Day 5 intentionally targeted the highest-signal pair identified in the Day 2
inventory:

- `src/sparse_svd.c`
- `src/sparse_eigs.c`

This batch stayed narrow on purpose:

- migrate duplicated overflow-multiplication helpers
- migrate repeated count-to-bytes guard sites
- preserve larger file-specific workspace ownership and cleanup choreography

It did **not** attempt to genericize:

- sibling-buffer allocation ordering
- algorithm-specific cleanup labels / goto structure
- SVD- or eigs-specific workspace composition rules

## Implemented Changes

### `src/sparse_svd.c`

Day 5 changed the SVD hotspot module by:

- adding the private helper include:
  - `#include "sparse_alloc_internal.h"`
- removing the file-local `size_mul_overflow(...)`
- switching repeated shared arithmetic checks to:
  - `sparse_size_mul_overflow(...)`

The migrated SVD call-site families include:

- extracted U/V workspace sizing
- bidiagonal diagonal/superdiagonal byte sizing
- full/economy output-size guards
- partial-SVD Lanczos workspace sizes
- singular-value/output buffer byte derivation
- pseudoinverse and low-rank dense-buffer size guards

### `src/sparse_eigs.c`

Day 5 changed the eigensolver hotspot module by:

- adding the private helper include:
  - `#include "sparse_alloc_internal.h"`
- removing the file-local `size_mul_overflow(...)`
- switching repeated shared arithmetic checks to:
  - `sparse_size_mul_overflow(...)`
- updating the retained explanatory workspace comment to name the shared helper
  rather than the removed local helper

The migrated eigs call-site families include:

- Lanczos work-vector byte sizing
- thick-restart `V`, `Y`, and dense `K×K` scratch sizing
- locked-vector expansion sizing
- thick-restart outer-loop workspace sizes
- LOBPCG dense block/workspace byte derivation

## What Stayed Local

Day 5 deliberately did **not** flatten the broader allocation/cleanup logic in
either file.

### SVD-specific logic that remains local

- bidiagonal / Lanczos workspace family composition
- output-shape-dependent allocation sequencing
- low-rank reconstruction ownership and cleanup flow

### Eigs-specific logic that remains local

- Lanczos / thick-restart / LOBPCG workspace family composition
- restart-state ownership transitions
- algorithm-specific cleanup/error propagation flow

This is the right boundary for Sprint 41:

- generic arithmetic safety moves to the shared helper layer
- algorithm-specific workspace choreography remains explicit and local

## Why This Batch Matters

Day 4 proved the helper layer in lower-risk integrations. Day 5 proves it in
the first real hotspot pair:

- both files were large measured duplication targets
- both carried the same local helper pattern
- both use many repeated workspace-size guards

This means Sprint 41 has now crossed from “helper layer exists” to “helper
layer is actively replacing high-value duplicated safety logic in major core
modules.”

## Validation Result

Because `*.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Highest-Value Conclusion

After Day 5, the first major solver hotspot pair no longer owns its own local
generic multiply-overflow helper. The shared helper layer now carries that
arithmetic-safety role, while the files retain only the specialized workspace
composition and cleanup logic that actually belongs to them.
