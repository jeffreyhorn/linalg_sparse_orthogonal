# Sprint 41 Day 7 Artifact: Broader `src/` Migration Audit

## Purpose

Turn the post-Day-6 residual helper-consolidation queue into a concrete
broader `src/` migration map by:

- identifying the next high-value source modules
- separating easy direct substitutions from moderate adapter-heavy cases
- naming the files that should remain deferred because their local allocation
  logic is too specialized for Sprint 41's shared-helper boundary

## Starting Point After Day 6

Day 6 completed the planned first-wave hotspot set:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

So Day 7 is no longer about the initial hotspot cluster. It is about the
broader `src/` tree that still carries:

- direct `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))`
  families
- manual `SIZE_MAX / sizeof(...)` checks
- manual `SIZE_MAX - ...` accumulation guards
- modules that already use parts of the shared helper layer but still own
  larger local allocation seams

## Broader Queue Classification

### 1. Easy direct substitutions

These are the best next modules because their local safety seam is narrow and
their allocation sites are still mostly simple n-based arrays or obvious
count-derived buffers.

#### `src/sparse_ic.c`

Current signs:

- paired `SIZE_MAX / sizeof(...)` guards
- `calloc((size_t)n, sizeof(double))`
- `malloc((size_t)n * sizeof(idx_t))`
- `calloc((size_t)n, sizeof(char))`

Why it is a good next migration:

- the workspace ownership is compact
- the main factorization routine is structurally straightforward
- the cleanup logic is local but not allocator-specialized

Expected Day 8 shape:

- adopt `sparse_malloc_array(...)` / `sparse_calloc_array(...)`
- replace manual overflow checks with shared count/bytes helpers
- keep factorization semantics and cleanup choreography unchanged

#### `src/sparse_analysis.c`

Current signs:

- repeated `SIZE_MAX / sizeof(idx_t)` checks
- repeated `malloc((size_t)n * sizeof(idx_t))`
- compact permutation/etree/postorder work arrays

Why it is a good next migration:

- the array counts are still direct and easy to reason about
- this is a broader proof than the first-wave hotspot list without being
  algorithmically overgrown
- it helps normalize a bridge module that appears in later Epic 4 lifecycle
  work

Expected Day 8 shape:

- replace repeated manual n-based checks with shared helper calls
- keep symbolic-analysis ownership and reorder dispatch local

### 2. Moderate helper-adapter cases

These are still valuable Sprint 41 targets, but they are no longer simple
mechanical substitutions. The allocation seam is intertwined with larger
workspace packing or dispatch behavior.

#### `src/sparse_iterative.c`

Current signs:

- manual `SIZE_MAX / sizeof(...)` checks
- manual `SIZE_MAX - total` accumulation logic
- packed workspaces for Krylov methods
- side allocations tied to stagnation/history/progress state

Why it is moderate instead of easy:

- allocation logic is part of workspace layout, not just safety checking
- several solver families share overlapping but not identical buffer plans
- cleanup and cancellation behavior must stay exact

Recommended handling:

- take it after the easier `ic` / `analysis` pair
- keep the batch focused on safety/helper adoption, not workspace redesign

#### `src/sparse_qr.c`

Current signs:

- partial shared-helper adoption already exists
- remaining raw allocations are mixed into QR-specific work arrays and
  factor/result preparation

Why it is moderate instead of easy:

- it is not a fully untouched seam anymore
- the remaining drift is real, but the file is large enough that careless
  changes could blur the Sprint 41 boundary

Recommended handling:

- treat it as a follow-on if the Day 9 batch remains bounded after
  `sparse_iterative.c`

#### Other likely moderate follow-ons

The broader grep suggests additional later-sprint candidates in the same
class:

- `src/sparse_lu.c`
- `src/sparse_lu_csr.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_colamd.c`

These should not be mixed into Day 8 unless the live pressure from the
easier pair shows a missing helper-layer primitive.

### 3. Specialized keep/defer cases

These files still contain allocation/overflow logic, but Sprint 41 should not
pull them into routine helper substitution because their local allocation seam
is too tied to algorithm structure.

#### `src/sparse_graph.c`

Why it stands apart:

- largest file in `src/`
- multi-algorithm ownership:
  - graph construction
  - subgraph extraction
  - coarsening hierarchies
  - partition/refinement state
  - separator work arrays
- many local scratch lifetimes with graph-structure meaning

Recommended handling:

- defer from the routine Day 8/9 migration path
- revisit under a later decomposition or maintainability-focused pass

#### Other likely keep/defer surfaces

Selected reorder/symbolic-heavy files may belong here as well when their
allocation logic is bound to traversal or structure-building semantics rather
than reusable count/bytes arithmetic. The broader audit does not force those
files into Sprint 41 just because helper duplication exists.

## Prioritized Migration Order

### Recommended Day 8 batch

Primary targets:

- `src/sparse_ic.c`
- `src/sparse_analysis.c`

Reason:

- both are high-value and low-risk
- both directly exercise the Day 4 helper layer on broader `src/` code
- neither requires a redesign of workspace ownership

### Recommended Day 9 batch

Primary target:

- `src/sparse_iterative.c`

Optional secondary target if the batch remains bounded:

- `src/sparse_qr.c`

Reason:

- these are the strongest remaining broader modules after the easy pair
- both provide real consolidation value, but only after the low-risk batch is
  out of the way

### Explicitly deferred from the routine Sprint 41 migration path

- `src/sparse_graph.c`

Reason:

- it is the clearest case where shared-helper substitution alone is not the
  whole maintainability problem

## Helper-Layer Pressure Assessment

Day 7 does **not** show a clear need to redesign the Day 4 helper API before
continuing. The current layer still appears sufficient for the next broader
batch:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

The most likely future pressure is not “missing arithmetic helpers”; it is the
need to preserve file-local packed-workspace meaning while replacing manual
checks. That is a migration-shape issue, not necessarily an API-gap issue.

## Validation Result

This was a docs-only audit day. No `*.c` or `*.h` files changed, so the full
`make format` / `make lint` / `make test` gate was not required.

The audit was grounded in:

- the live broader `src/` duplication map
- source-size concentration
- representative reads of:
  - `src/sparse_ic.c`
  - `src/sparse_analysis.c`
  - `src/sparse_iterative.c`
  - `src/sparse_graph.c`

## Highest-Value Conclusion

Sprint 41 now has a usable broader migration map rather than a vague residual
queue. The next safe batch is the `sparse_ic.c` / `sparse_analysis.c` pair,
the next moderate batch is led by `sparse_iterative.c`, and `sparse_graph.c`
is explicitly deferred as a specialized later concern rather than being
dragged into Sprint 41's routine helper-consolidation path.
