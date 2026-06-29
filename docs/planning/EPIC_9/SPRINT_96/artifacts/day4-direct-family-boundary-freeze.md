# Sprint 96 Day 4: Direct-Family Cleanup Boundary Freeze

## Purpose

Day 4 freezes the first direct-family implementation cleanup batch before
source edits begin. The selected batch is the LDLT dense block factor and
runtime-selected backend extraction from `src/sparse_ldlt_csc.c`.

Day 4 is a planning and boundary-freeze day. No `.c` or `.h` files are changed
by this artifact.

## Frozen Direct Target

Primary source owner:

- `src/sparse_ldlt_csc.c`

New source owner to create on Day 5:

- `src/sparse_ldlt_dense.c`

Contract owner to preserve:

- `src/sparse_ldlt_csc_internal.h`

Build owners to update:

- `Makefile`
- `CMakeLists.txt`

Proof owners:

- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`
- `tests/test_direct_csc_dispatch.c`
- `tests/test_direct_csc_regression.c`

## Exact Move Boundary

Move the current dense/backend implementation block from
`src/sparse_ldlt_csc.c:38-616` into `src/sparse_ldlt_dense.c`.

The moved block includes:

- dense/backend-specific includes:
  - `<limits.h>`
  - `<stdatomic.h>`
  - `<stdint.h>`
  - `<dlfcn.h>` under `#ifndef _WIN32`
- dense symmetric swap helper:
  - `ldlt_dense_sym_swap(...)`
- BLAS/LAPACK index guard:
  - `s64_ldlt_idx_to_blas_int_checked(...)`
- Accelerate/BLAS 2x2-pivot acceptance helper:
  - `s64_ldlt_accel_accepts_noperm_2x2_pivot(...)`
- non-Windows external backend state and helpers:
  - `s64_accel_dsytrf_fn`
  - `s64_ldlt_ext_provider_t`
  - `s64_ldlt_ext_handle`
  - `s64_ldlt_ext_dsytrf`
  - `s64_ldlt_ext_provider`
  - `s64_ldlt_ext_probe_state`
  - `s64_ldlt_store_symbol(...)`
  - `s64_ldlt_ext_probe_dense_factor(...)`
  - `s64_external_ldlt_dense_factor(...)`
- dense backend environment parsing:
  - `s64_ldlt_dense_backend_t`
  - `s64_read_ldlt_dense_backend_env(...)`
- exported internal dense/backend functions:
  - `ldlt_dense_factor_backend_name(...)`
  - `ldlt_dense_factor(...)`
  - `ldlt_dense_factor_selected(...)`

The first function remaining in `src/sparse_ldlt_csc.c` after Day 5 should be
`ldlt_csc_free(...)`, unless Day 5 finds a compile-driven need for a small
local helper to remain above it.

## Headers After The Move

`src/sparse_ldlt_dense.c` should include:

- `sparse_alloc_internal.h`
- `sparse_ldlt_csc_internal.h`
- `<limits.h>`
- `<math.h>`
- `<stdatomic.h>`
- `<stdint.h>`
- `<stdlib.h>`
- `<string.h>`
- `<dlfcn.h>` under `#ifndef _WIN32`

`src/sparse_ldlt_csc.c` should keep:

- `sparse_alloc_internal.h`
- `sparse_ldlt.h`
- `sparse_ldlt_csc_internal.h`
- `<math.h>`
- `<stdlib.h>`
- `<string.h>`

`src/sparse_ldlt_csc.c` should drop these includes if the compile confirms
they are no longer used there:

- `<limits.h>`
- `<stdatomic.h>`
- `<stdint.h>`
- `<dlfcn.h>`

Rationale: after the dense/backend block moves, `src/sparse_ldlt_csc.c` still
uses `SIZE_MAX`, `fabs`, `sqrt`, `malloc`, `calloc`, `free`, and `memcpy`, but
the `INT_MAX`, atomics, and dynamic-loader references are isolated to the new
dense/backend owner.

## Internal Contract

Keep these declarations in `src/sparse_ldlt_csc_internal.h` unchanged:

- `ldlt_dense_factor(...)`
- `ldlt_dense_factor_selected(...)`
- `ldlt_dense_factor_backend_name(...)`

Day 5 may update nearby comments only if they describe implementation
ownership. It should not rename the functions or add public declarations.

## Build Registration Plan

Add the new source immediately next to the existing LDLT CSC sources.

In `Makefile`, insert:

```make
           $(SRCDIR)/sparse_ldlt_dense.c \
```

near:

```make
           $(SRCDIR)/sparse_ldlt_csc.c \
           $(SRCDIR)/sparse_ldlt_csc_supernodal.c \
```

Recommended order:

```make
           $(SRCDIR)/sparse_ldlt_dense.c \
           $(SRCDIR)/sparse_ldlt_csc.c \
           $(SRCDIR)/sparse_ldlt_csc_supernodal.c \
```

In `CMakeLists.txt`, insert:

```cmake
    src/sparse_ldlt_dense.c
```

near:

```cmake
    src/sparse_ldlt_csc.c
    src/sparse_ldlt_csc_supernodal.c
```

Recommended order:

```cmake
    src/sparse_ldlt_dense.c
    src/sparse_ldlt_csc.c
    src/sparse_ldlt_csc_supernodal.c
```

## Day 5 Implementation Sequence

1. Create `src/sparse_ldlt_dense.c` with a file header that owns the dense LDLT
   factor and optional runtime-selected backend.
2. Move `src/sparse_ldlt_csc.c:38-616` into the new file.
3. Remove the moved block from `src/sparse_ldlt_csc.c`.
4. Update `src/sparse_ldlt_csc.c` file-level ownership comments so it no
   longer claims to own the dense primitive directly.
5. Keep `src/sparse_ldlt_csc_internal.h` declarations stable.
6. Register `src/sparse_ldlt_dense.c` in both Makefile and CMake.
7. Run focused compile/test checks as useful during development.
8. Run the required full quality chain before considering the code day done.

## Explicit Non-Goals

Day 5 should not include:

- public API changes
- public header edits under `include/`
- changes to `src/sparse_ldlt_csc_supernodal.c` unless compile feedback
  requires an include/comment update
- changes to sparse CSC allocation, conversion, writeback, validation, solve,
  native elimination, or supernodal orchestration behavior
- benchmark driver changes
- generated documentation edits
- proof-owner test splits
- LDLT linked-list implementation changes in `src/sparse_ldlt.c`

## Targeted Proof Plan

Development-time focused checks, if a quick local signal is useful:

- `make build/test_chol_csc`
- `make build/test_ldlt_csc`
- `make build/test_direct_csc_dispatch`
- `make build/test_direct_csc_regression`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_direct_csc_dispatch`
- `./build/test_direct_csc_regression`

Required completion check after any Day 5 or Day 6 `.c` or `.h` change:

```sh
make format && make lint && make test
```

## Stale-Reference Scans

After Day 5 implementation, run:

```sh
rg -n "ldlt_dense_factor|ldlt_dense_factor_selected|ldlt_dense_factor_backend_name" src include tests
rg -n "sparse_ldlt_dense|sparse_ldlt_csc" Makefile CMakeLists.txt src
rg -n "Sprint 19 Day 11|dense LDL\\^T primitive|runtime-selected backend" src/sparse_ldlt_csc.c src/sparse_ldlt_dense.c src/sparse_ldlt_csc_internal.h
```

The scans should show:

- dense/backend implementation in `src/sparse_ldlt_dense.c`
- dense function declarations still in `src/sparse_ldlt_csc_internal.h`
- supernodal call sites still using `ldlt_dense_factor_selected(...)`
- build systems registering `src/sparse_ldlt_dense.c`
- no stale claim in `src/sparse_ldlt_csc.c` that it owns the dense primitive

## Risk Notes

Platform guard risk:

- keep all `<dlfcn.h>` usage inside `#ifndef _WIN32`
- keep external backend probing behind `#ifndef _WIN32`
- keep Accelerate-only backend spelling behind `#ifdef __APPLE__`

Link/build risk:

- adding a new `.c` file requires both Makefile and CMake updates
- the internal header already provides prototypes, so no new public header
  contract should be needed

Behavior risk:

- this batch should be behavior-preserving source ownership movement
- no dense numerical recurrence changes should be mixed into the extraction
- no environment-variable semantics should change for
  `SPARSE_LDLT_DENSE_BACKEND`

Proof risk:

- direct proof density is high, especially in `tests/test_chol_csc.c`
- `tests/test_chol_csc.c` owns direct dense LDLT primitive checks at its
  `ldlt_dense_factor` cluster
- `tests/test_ldlt_csc.c` owns LDLT CSC supernodal behavior that depends on
  `ldlt_dense_factor_selected(...)`

## Day 4 Result

The first direct-family cleanup batch is frozen. Day 5 should perform one
bounded extraction: create `src/sparse_ldlt_dense.c`, move the dense LDLT
factor/backend implementation out of `src/sparse_ldlt_csc.c`, register the new
source in both build systems, preserve internal function signatures, and run
the required code-day validation chain.
