# Sprint 41 Day 3 Artifact: Shared Utility API Design

## Purpose

Define the shared internal allocation/overflow helper layer concretely before
Day 4 implementation starts, including:

- file layout
- helper tiers
- macro/inline/function split
- preserved error semantics
- explicit non-goals

## Design Goals

The Sprint 41 shared utility layer should:

1. remove repeated low-level arithmetic helpers
2. normalize byte-count derivation
3. preserve current `SPARSE_ERR_ALLOC` / `NULL` behavior
4. stay internal-first
5. avoid collapsing legitimate symbolic or lifecycle-specific differences into
   a fake one-size-fits-all API

## Proposed File Layout

### Private header

- `src/sparse_alloc_internal.h`

Responsibilities:

- small pure arithmetic/bounds helpers
- count-to-bytes helpers
- `size_t` <-> `idx_t` representability helpers
- private declarations for any non-inline allocation helpers

### Private source

- `src/sparse_alloc_internal.c`

Responsibilities:

- non-inline allocation convenience wrappers if the shared API needs them
- one implementation point for shared alloc/error-translation behavior
- future internal extension point for helper instrumentation without touching
  every call site again

### Build integration

Day 4 implementation should add the source file to:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` library source list

The header will be picked up naturally by the existing `src/*.h` formatting and
linting surfaces.

## Proposed Helper Tiers

### Tier 1: header-inline arithmetic helpers

These are the strongest direct-consolidation targets and should live in
`sparse_alloc_internal.h` as `static inline` helpers:

```c
static inline int sparse_size_mul_overflow(size_t a, size_t b, size_t *out);
static inline int sparse_size_add_overflow(size_t a, size_t b, size_t *out);
static inline int sparse_count_bytes_overflow(size_t count, size_t elem_size,
                                              size_t *bytes);
static inline int sparse_idx_count_bytes_overflow(idx_t count,
                                                  size_t elem_size,
                                                  size_t *bytes);
static inline int sparse_size_to_idx_checked(size_t value, idx_t *out);
```

Rationale:

- these are tiny, pure, and heavily reused
- they match existing repo practice for small internal helpers
- inlining avoids function-call noise in hot or repeated validation paths
- they give later broader `src/` migration a stable arithmetic vocabulary

### Tier 2: source-backed allocation helpers

These should exist only where they add clear value beyond the arithmetic tier:

```c
sparse_err_t sparse_malloc_array(size_t count, size_t elem_size, void **out);
sparse_err_t sparse_calloc_array(size_t count, size_t elem_size, void **out);
```

Expected contract:

- validate `count * elem_size`
- perform allocation
- set `*out = NULL` on failure
- return:
  - `SPARSE_OK` on success
  - `SPARSE_ERR_ALLOC` on overflow or allocation failure

Rationale:

- useful for first-wave hotspot modules with repeated:
  - count validation
  - allocation
  - `SPARSE_ERR_ALLOC` translation
- still generic enough to reuse later in:
  - `sparse_qr.c`
  - broader workspace-heavy `src/` code

## Macro vs Inline vs Function Decision

### Do not use macros as the primary interface

Rejected for the main API because:

- macros obscure evaluation and debugging
- the repeated helper shapes already have natural typed signatures
- `static inline` gives the same ergonomics without preprocessor opacity

### Use `static inline` for pure arithmetic helpers

Use header-inline form for:

- multiplication overflow checks
- addition overflow checks
- byte-count derivation
- representability checks

These operations are:

- tiny
- side-effect-free except for output pointers
- likely to appear in many call sites over Sprint 41 and later sprints

### Use normal functions for shared allocation wrappers

Use source-backed functions only for helpers that:

- call `malloc` / `calloc`
- translate failure to `SPARSE_ERR_ALLOC`
- benefit from one implementation point

This keeps the boundary clear:

- arithmetic lives in the header
- heap behavior lives in the source

## Preserved Error Semantics

The shared layer must preserve the current behavioral contract:

- arithmetic helpers:
  - do not allocate
  - do not mutate global state
  - report failure only through return value
- allocation wrappers:
  - map overflow and allocation failure to `SPARSE_ERR_ALLOC`
  - do not widen public API surface
  - do not change caller-visible error classes

Important: the utility layer should not invent a universal zero-size policy.

Caller-owned zero-size behavior must remain explicit where it matters:

- empty dense matrices may keep `data = NULL`
- empty symbolic column-pointer paths may still allocate `1` element
- zero-safe row-index sentinel logic stays local where required

## Explicit Non-Goals

The Sprint 41 utility layer should not:

- become a public header
- absorb symbolic prefix-sum semantics from `sparse_etree.c`
- hide all multi-buffer cleanup choreography behind one giant helper
- rewrite lifecycle/state validation logic
- change cross-platform, reviewed-quality, or dead-code contracts

## First-Wave and Broader Reuse Targets

### First-wave Sprint 41 targets

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

### Broader `src/` queue already visible

- `src/sparse_qr.c`

This means the Day 4 implementation should be:

- small enough for the first-wave hotspot modules
- general enough that Day 7-9 broader migration does not require redesign

## Highest-Value Day 3 Conclusion

Sprint 41 should implement a small private safety-helper layer with:

- inline arithmetic/bounds helpers in `src/sparse_alloc_internal.h`
- optional source-backed shared allocation helpers in
  `src/sparse_alloc_internal.c`
- an explicit keep-local rule for symbolic accumulation, zero-size policy, and
  lifecycle-specific cleanup logic

That is the narrowest design that still solves the real duplication Day 2
measured.
