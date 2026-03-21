# Sprint 1 Daily Log

## Day 1 — Project Scaffolding & Consolidation

### Completed
- Created directory structure: `include/`, `src/`, `tests/`, `tests/data/`, `benchmarks/`, `archive/`, `scripts/`, `docs/`
- Moved all 7 original `.c` files and the compiled binary to `archive/`
- Added `archive/README.md` documenting the file evolution and known issues
- Initialized git repository
- Created `.gitignore`
- Created `CMakeLists.txt` with C11, strict warnings, optional sanitizers, library target, test/bench scaffolding
- Created `Makefile` as a simpler build alternative
- Created public headers:
  - `include/sparse_types.h` — `idx_t`, `sparse_err_t` enum, `sparse_strerror()`
  - `include/sparse_matrix.h` — full public API for sparse matrix data structure
  - `include/sparse_lu.h` — LU factorization and solve API
- Created private internal header:
  - `src/sparse_matrix_internal.h` — Node, NodeSlab, NodePool (with free-list), SparseMatrix struct
- Created implementation files:
  - `src/sparse_types.c` — error code to string
  - `src/sparse_matrix.c` — full implementation: pool with free-list, create/free/copy, insert/remove/get/set, matvec, Matrix Market I/O, display, permutation access
  - `src/sparse_lu.c` — LU factor (with snapshot fix for bug 3.1, forward-sub fix for bug 3.3), solve, individual phases, iterative refinement

### Notes
- Went beyond the original Day 1 plan to also complete Days 2-5 implementation work, since the API design and implementation flowed naturally from the consolidation
- All three critical bugs from the review are fixed in the new code:
  - Bug 3.1: `sparse_lu_factor` snapshots elimination row indices before modifying the matrix
  - Bug 3.3: `sparse_forward_sub` traverses the entire row without early break
  - Bug 3.6: `sparse_matvec` is provided so callers can compute proper residuals on the original matrix
- Pool allocator now includes a free-list for node reuse
- API uses `const` correctness and `sparse_err_t` throughout
- Rectangular matrices are supported in the data structure (`sparse_create(rows, cols)`)

## Day 2 — Header Files & API Design (Review & Polish)

### Completed
- Reviewed all 3 headers and 3 implementation files against the Day 2 plan
- Added `sparse_pivot_t` enum (`SPARSE_PIVOT_COMPLETE`, `SPARSE_PIVOT_PARTIAL`) to `sparse_types.h`
- Updated `sparse_lu_factor` signature to accept a pivoting strategy parameter
- Implemented partial pivoting in `sparse_lu.c` (searches only the pivot column)
- Added `SPARSE_ERR_SHAPE` error code for non-square matrix passed to LU
- Moved tuning constants to public header as overridable defines:
  - `SPARSE_NODES_PER_SLAB` (default 4096) — pool slab size
  - `SPARSE_DROP_TOL` (default 1e-14) — fill-in drop tolerance
- Internal header now derives `NODES_PER_SLAB` / `DROP_TOL` from public defines
- Added symmetric and pattern-only Matrix Market format support in `sparse_load_mm`
- Created `tests/smoke_test.c` — end-to-end verification:
  - Builds, links, and runs against the library
  - Tests complete pivoting, partial pivoting, copy, solve, residual, iterative refinement, and MM round-trip
  - All pass with zero residual
- Updated Makefile with generic test build rule and `make smoke` target

### Notes
- Most Day 2 work (headers, API design) was already done in Day 1
- Day 2 focused on reviewing for gaps and adding missing features from the plan
- Both pivoting strategies produce the correct solution for the 3x3 test matrix
- Days 3-6 from the original plan are also substantially complete
