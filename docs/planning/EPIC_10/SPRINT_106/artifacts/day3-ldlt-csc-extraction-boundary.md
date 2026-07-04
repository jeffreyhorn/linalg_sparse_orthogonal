# Sprint 106 Day 3 LDLT/CSC Extraction Boundary

## Purpose

Day 3 freezes the first LDLT CSC extraction seam before implementation. The
goal is a small, reviewable source split that reduces the largest
implementation owner without changing LDLT CSC semantics, public APIs, test
registration, or solver claims.

## Inspected Surfaces

| surface | files | reason inspected |
|---|---|---|
| LDLT CSC implementation | `src/sparse_ldlt_csc.c` | largest source owner and Sprint 106 first implementation target |
| private LDLT CSC contract | `src/sparse_ldlt_csc_internal.h` | declares `LdltCsc`, lifecycle, row-adj, conversion, writeback, native, and supernodal contracts |
| supernodal LDLT CSC implementation | `src/sparse_ldlt_csc_supernodal.c` | paired LDLT CSC source already extracted; reference for private-owner style |
| dense LDLT primitive | `src/sparse_ldlt_dense.c` | shares private LDLT CSC header but should not be pulled into row-adj ownership |
| direct CSC tests | `tests/test_ldlt_csc.c`, `tests/test_direct_csc_regression.c` | direct coverage for row-adj, conversion, elimination, and regression behavior |
| build membership | `build-metadata/library_sources.txt`, `Makefile`, `CMakeLists.txt` | all must include any new library source in synchronized order |
| source-list checker | `scripts/check_library_sources.py` | must remain passing after the split |

## Candidate Seam Comparison

| candidate seam | value | risk | Day 3 decision |
|---|---|---|---|
| row-adjacency lifecycle and population | cohesive helper cluster; already has tests; directly reduces native/supernodal mixed responsibility | low-medium; must preserve allocation/free and swap invariants | select for Day 4 |
| conversion and analysis-aware construction | large comment-heavy section; central to symbolic-fill behavior | high; touches indefinite pre-pass assumptions and sym_L completeness | defer |
| writeback/public LDLT payload materialization | contained helper cluster; clear output responsibility | medium; touches public `sparse_ldlt_t` ownership and drop policy | defer until after row-adj split |
| wrapper rebuild/publish helpers | localized compatibility wrapper path | medium; overlaps linked-list LDLT behavior and fallback evidence | defer |

The selected Day 4 seam is row-adjacency lifecycle and population.

## Selected Extraction Boundary

### New Private Source Owner

Planned new file:

```text
src/sparse_ldlt_csc_rowadj.c
```

Role:

- own row-adjacency allocation helpers if introduced;
- own per-row adjacency free support if introduced;
- own `ldlt_csc_row_adj_append(...)`;
- own row-adjacency slot swapping for symmetric pivot swaps;
- own `ldlt_csc_populate_row_adj(...)`;
- keep row-adjacency comments near the row-adj implementation rather than
  buried inside native elimination.

The new file remains private implementation. It does not create a public
header, public API, exported install surface, or new test binary.

### Existing Source Owner After Split

`src/sparse_ldlt_csc.c` keeps:

- top-level LDLT CSC lifecycle entry points;
- sparse-to-CSC conversion;
- analysis-aware conversion;
- CSC-to-public writeback;
- validation;
- wrapper path;
- kernel selection;
- workspace lifecycle;
- native Bunch-Kaufman pivoting and cmod logic;
- supernodal orchestration call sites.

It should call row-adj helpers through declarations in
`src/sparse_ldlt_csc_internal.h`.

## Functions and Helper Responsibilities

### Move or Introduce in `src/sparse_ldlt_csc_rowadj.c`

| helper | current status | target responsibility |
|---|---|---|
| `ldlt_csc_row_adj_append` | non-static in `src/sparse_ldlt_csc.c`; declared in internal header | move as-is; preserve error codes and geometric growth |
| `ldlt_csc_populate_row_adj` | static in `src/sparse_ldlt_csc.c` | make internal non-static or declaration-visible; populate `F->row_adj[i]` after column writeback |
| row-adj slot swap helper | inline block inside `ldlt_csc_symmetric_swap` | introduce helper such as `ldlt_csc_row_adj_swap_slots(F, i, j)` to keep swap invariants together |
| row-adj allocation helper | duplicated allocation pattern in `ldlt_csc_alloc`, `ldlt_csc_from_sparse`, and `ldlt_csc_from_sparse_with_analysis` | optional Day 4 follow-through only if it reduces duplication without expanding risk |
| row-adj free helper | inline loop in `ldlt_csc_free` | optional Day 4 follow-through only if allocation helper is introduced |

### Keep in `src/sparse_ldlt_csc.c`

| helper | reason to keep |
|---|---|
| `ldlt_csc_alloc` | public internal lifecycle entry point also allocates D, D_offdiag, pivot_size, perm, and embedded Cholesky CSC |
| `ldlt_csc_free` | lifecycle function owns all factor arrays, not just row-adj |
| `ldlt_csc_cmod_unified` | numeric cmod logic reads row-adj but owns dense accumulator updates and 2x2 D_offdiag semantics |
| `ldlt_csc_symmetric_swap` | swaps matrix storage, D arrays, pivot metadata, perm, and row-adj slots; only the row-adj slot operation should be delegated |
| conversion helpers | higher semantic risk; defer |
| writeback helpers | public output materialization; defer |

## Include and Visibility Boundary

The new source should include:

```c
#include "sparse_alloc_internal.h"
#include "sparse_ldlt_csc_internal.h"

#include <stdint.h>
#include <stdlib.h>
```

`stdint.h` is needed if the moved helpers retain `SIZE_MAX` overflow guards.
`stdlib.h` is needed for `realloc` and any optional allocation/free helpers.

`src/sparse_ldlt_csc_internal.h` should continue to declare
`ldlt_csc_row_adj_append(...)` and should add declarations only for helpers
used across source files, likely:

```c
sparse_err_t ldlt_csc_populate_row_adj(LdltCsc *F, idx_t col);
void ldlt_csc_row_adj_swap_slots(LdltCsc *F, idx_t i, idx_t j);
```

Do not expose these through any public header under `include/`.

## Build-System Follow-Through

Day 4 source extraction must update these synchronized library source lists:

| file | required update |
|---|---|
| `build-metadata/library_sources.txt` | add `src/sparse_ldlt_csc_rowadj.c` adjacent to `src/sparse_ldlt_csc.c` |
| `Makefile` | add `$(SRCDIR)/sparse_ldlt_csc_rowadj.c` adjacent to the LDLT CSC entries in `LIB_SRCS` |
| `CMakeLists.txt` | add `src/sparse_ldlt_csc_rowadj.c` adjacent to the LDLT CSC entries |
| `scripts/check_library_sources.py` | no expected edit unless checker assumptions fail |

Recommended ordering:

```text
src/sparse_ldlt_dense.c
src/sparse_ldlt_csc.c
src/sparse_ldlt_csc_rowadj.c
src/sparse_ldlt_csc_supernodal.c
```

The exact order should match across all three membership surfaces.

## Test and Regression Surface

Focused tests that directly cover the selected seam:

| test owner | coverage relevance |
|---|---|
| `tests/test_ldlt_csc.c` | allocation, row-adj empty state, append ordering, geometric growth, argument checks, row-adj reference matching, native/wrapper parity, analysis-aware CSC behavior |
| `tests/test_direct_csc_regression.c` | row-adj reference matching on `bcsstk04` and direct CSC regression behavior |
| `tests/test_ldlt.c` | public LDLT writeback and dispatch behavior through linked-list/CSC paths |
| `tests/test_ldlt_backend_dispatch.c` | backend dispatch confidence if LDLT CSC linkage changes |

No test registration change is planned. If Day 4 adds only a library source and
moves functions, reviewed CTest count should remain unchanged.

## Day 4 Implementation Sequence

1. Add `src/sparse_ldlt_csc_rowadj.c`.
2. Move `ldlt_csc_row_adj_append(...)` from `src/sparse_ldlt_csc.c` to the new
   source.
3. Move or expose `ldlt_csc_populate_row_adj(...)` in the new source and keep
   all existing call sites.
4. Add `ldlt_csc_row_adj_swap_slots(...)` and replace only the row-adj slot
   swap block inside `ldlt_csc_symmetric_swap`.
5. Consider row-adj allocation/free helper extraction only if it remains
   small and mechanical; otherwise defer to avoid widening Day 4.
6. Update `src/sparse_ldlt_csc_internal.h`.
7. Update `build-metadata/library_sources.txt`, `Makefile`, and
   `CMakeLists.txt`.
8. Run source-list and focused LDLT CSC validation before the full gate.

## Focused Validation Commands

Minimum source extraction validation:

```sh
python3 scripts/check_library_sources.py
make build/test_ldlt_csc build/test_direct_csc_regression build/test_ldlt build/test_ldlt_backend_dispatch
./build/test_ldlt_csc
./build/test_direct_csc_regression
./build/test_ldlt
./build/test_ldlt_backend_dispatch
make format && make lint && make test
git diff --check
rg -n "[ \t]+$" src/sparse_ldlt_csc.c src/sparse_ldlt_csc_internal.h \
  src/sparse_ldlt_csc_rowadj.c build-metadata/library_sources.txt \
  Makefile CMakeLists.txt docs/planning/EPIC_10/SPRINT_106
```

If Day 4 changes only C implementation and build membership, no CTest
registration count should change. If any test membership changes unexpectedly,
stop and reconcile POSIX/Windows reviewed counts before proceeding.

## Explicit Non-Changes

Day 4 should not:

- change the `LdltCsc` struct layout;
- change public headers under `include/`;
- change `ldlt_csc_from_sparse(...)` or
  `ldlt_csc_from_sparse_with_analysis(...)` semantics;
- change `ldlt_csc_writeback_to_ldlt(...)` semantics;
- change native versus wrapper dispatch policy;
- change supernodal detection or dense LDLT behavior;
- add new tests or alter CTest registration unless a compile failure proves it
  necessary.

## Boundary Decision

Sprint 106 Day 4 should extract row-adjacency ownership into
`src/sparse_ldlt_csc_rowadj.c`. This is the smallest cohesive LDLT CSC seam
with strong local tests, clear private visibility, limited API risk, and known
Make/CMake/source-list follow-through.
