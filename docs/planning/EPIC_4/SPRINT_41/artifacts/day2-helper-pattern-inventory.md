# Sprint 41 Day 2 Artifact: Helper-Pattern Inventory

## Purpose

Measure the first Sprint 41 hotspot modules so the shared utility design starts
from actual duplicated allocation/overflow logic rather than a generic
"cleanup" assumption.

## Audited Core Modules

The Day 2 helper-pattern inventory covers the four modules named in the Sprint
41 plan and remediation handoff:

- `src/sparse_dense.c` (`503` lines)
- `src/sparse_svd.c` (`1726` lines)
- `src/sparse_eigs.c` (`3143` lines)
- `src/sparse_etree.c` (`761` lines)

## Measured Helper Buckets

### 1. Size multiplication overflow checks

Strong direct duplication exists in:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`

Each file carries a local `size_mul_overflow(size_t, size_t, size_t *)` helper
with the same effective contract:

- return `0` on success
- return nonzero on overflow
- use `a != 0 && b > SIZE_MAX / a`
- write the multiplied result through the output pointer on success

This is the clearest shared-helper candidate in the Day 2 inventory.

### 2. `idx_t` / `size_t` representability checks

The strongest representability logic lives in `src/sparse_etree.c`:

- cumulative totals are built in `size_t`
- public/internal symbolic structures still need `idx_t` storage
- representability is checked explicitly with cast-back validation

Examples:

- `(size_t)sym->col_ptr[j + 1] != total_nnz`
- `(size_t)sym_U->col_ptr[j + 1] != u_total`

This pattern is structurally different from the dense/SVD/eigs workspace
sizing path, but it still belongs in the Sprint 41 utility-design scope.

### 3. Count-to-bytes conversions

All four modules perform repeated "count -> bytes" derivation:

- dense:
  - `rows * cols`
  - `m * n`
  - `m * sizeof(double)`
- svd:
  - `mt*k -> mt_k_bytes`
  - `nt*k -> nt_k_bytes`
  - `m*k -> mk_bytes`
  - `n*k -> nk_bytes`
  - `k*sizeof(double) -> sigma_bytes`
- eigs:
  - `n*m_cap -> v_bytes`
  - `m_cap*m_cap -> K2_bytes / cc_bytes`
  - `n*k -> vk_bytes`
  - `n*block_size -> nb_bytes`
- etree:
  - `n * sizeof(idx_t)`
  - `(n + 1) * sizeof(idx_t)`
  - `nnz * sizeof(idx_t)`

The raw arithmetic differs, but the utility-design pressure is consistent:

- compute counts safely
- convert to bytes safely
- preserve current `SPARSE_ERR_ALLOC` or `NULL`-return behavior

### 4. Common allocation/free/reset helper shapes

The audited modules repeatedly show:

- allocate multiple sibling buffers after a shared validation block
- free the whole sibling set on any failure
- zero/init arrays immediately after validated allocation

This is strongest in:

- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- selected symbolic build paths in `src/sparse_etree.c`

This does not automatically imply one generic "allocate-everything" helper, but
it does mean Day 3 should decide whether narrow convenience helpers are worth
introducing for common size-derivation and zero-init paths.

## Shared vs Specialized Classification

### Direct shared-helper candidates

- `size_mul_overflow(size_t, size_t, size_t *)`
- one-dimensional `count * sizeof(T)` / count-to-bytes derivation helpers
- possibly a tiny helper for safe `(n + 1)` style count construction before
  byte conversion

### Near-shared patterns that likely need small adapters

- multi-buffer workspace sizing in SVD/eigs
- symbolic-structure array sizing in etree
- quotient-style overflow checks in dense that should collapse to the same
  shared utility semantics

### Truly specialized logic that should remain local

- etree symbolic-prefix-sum accumulation semantics
- zero-safe "allocate 1 when nnz == 0" row-index handling
- file-specific cleanup sequencing tied to factor/result object ownership

## Highest-Value Day 2 Conclusion

Sprint 41 does not need one giant safety-helper framework. It needs:

1. a small shared arithmetic/bytes utility core
2. a clear boundary around representability and accumulation helpers
3. a disciplined keep-local rule for symbolic and lifecycle-specific logic

That is the right input shape for Day 3's shared utility API design.
