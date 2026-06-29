# Sprint 96 Day 3: Source Extraction Design

## Purpose

Day 3 defines bounded source cleanup boundaries before any code movement. The
design uses the Day 2 rerank and a source-level inspection of the selected
hotspots to choose one direct-family cleanup target and one solver/algorithm
cleanup target.

This is a design gate only. No `.c` or `.h` files move on Day 3.

## Selected Direct-Family Target

Primary target: `src/sparse_ldlt_csc.c`

Selected cleanup boundary: extract the dense LDLT block factor and
runtime-selected backend ownership from `src/sparse_ldlt_csc.c` into a focused
source owner.

Recommended new source owner:

- `src/sparse_ldlt_dense.c`

Existing internal contract to keep stable:

- `src/sparse_ldlt_csc_internal.h`

The existing internal header already declares the stable callable surface:

- `ldlt_dense_factor(...)`
- `ldlt_dense_factor_selected(...)`
- `ldlt_dense_factor_backend_name(...)`

The cleanup should preserve those names and signatures so callers, tests, and
the paired supernodal implementation do not need a public contract change.

### Direct Responsibilities To Move

Move the dense/backend cluster currently at the top of
`src/sparse_ldlt_csc.c`:

- dense symmetric swap helper
- BLAS/LAPACK integer guard
- external backend provider enum and probe state
- `dlsym` function-pointer storage helper
- runtime external dense factor probe
- external dense factor wrapper
- dense backend environment parsing
- backend-name reporting
- builtin `ldlt_dense_factor(...)`
- `ldlt_dense_factor_selected(...)`

Keep these responsibilities in `src/sparse_ldlt_csc.c`:

- CSC allocation and free
- row-adjacency support
- CSC conversion and writeback
- CSC validation
- linked-list compatibility wrapper
- native sparse Bunch-Kaufman workspace and elimination
- CSC solve path
- top-level supernodal orchestration

### Direct Touched-File Plan

Expected implementation files for Days 4-6:

| File | Expected role |
|---|---|
| `src/sparse_ldlt_csc.c` | remove dense/backend cluster and keep CSC owner focused |
| `src/sparse_ldlt_dense.c` | new dense LDLT/backend source owner |
| `src/sparse_ldlt_csc_internal.h` | keep existing declarations; update comments only if needed |
| `Makefile` | add new source to `LIB_SRCS` |
| `CMakeLists.txt` | add new source to library sources |

Files to avoid unless Day 4 finds a concrete compile need:

- `include/*.h`
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_ldlt.c`
- `src/sparse_chol_csc.c`
- benchmark drivers
- generated documentation

### Direct Extraction Rationale

This boundary reduces review cost because the dense block factor and optional
runtime backend are conceptually separate from sparse CSC ownership. The
selected block is also bounded by existing internal function declarations, so
the cleanup can avoid a public API change.

This boundary is better than moving the native sparse elimination helpers first
because the native kernel is tightly coupled to `LdltCsc`, row adjacency, and
workspace state. It is also better than moving conversion/writeback first
because those helpers sit close to public `sparse_ldlt_t` payload publication
and can be revisited after the dense backend owner is separated.

### Direct Validation Owners

Any Day 4-6 code change in this lane must run:

- `make format`
- `make lint`
- `make test`

Targeted proof owners to inspect or rerun during development:

- `tests/test_ldlt_csc.c`
- `tests/test_chol_csc.c`
- `tests/test_direct_csc_dispatch.c`
- `tests/test_direct_csc_regression.c`

Targeted code-reference scans after implementation:

- `rg -n "ldlt_dense_factor|ldlt_dense_factor_selected|ldlt_dense_factor_backend_name" src include tests`
- `rg -n "sparse_ldlt_dense|sparse_ldlt_csc" Makefile CMakeLists.txt`

## Selected Solver/Algorithm Target

Primary target: `src/sparse_iterative.c`

Selected cleanup boundary: separate block-solver wrapper ownership from the
main scalar iterative solver owner, with Day 7 responsible for freezing whether
the first implementation batch moves the shared block helper plus wrappers or
only simplifies the local block wrapper cluster.

Recommended new source owner if Day 7 chooses a split:

- `src/sparse_iterative_block.c`

Existing internal contract to keep stable:

- `src/sparse_iterative_internal.h`
- `include/sparse_iterative.h`

The public API declarations for block solvers should remain unchanged:

- `sparse_cg_solve_block(...)`
- `sparse_gmres_solve_block(...)`
- `sparse_minres_solve_block(...)`
- `sparse_bicgstab_solve_block(...)`

### Solver Responsibilities To Move Or Isolate

Candidate block-solver cluster in `src/sparse_iterative.c`:

- `iter_block_column_solver_fn`
- `solve_block_independent_columns(...)`
- `solve_block_gmres_column(...)`
- `sparse_gmres_solve_block(...)`
- `solve_block_minres_column(...)`
- `sparse_minres_solve_block(...)`
- `solve_block_bicgstab_column(...)`
- `sparse_bicgstab_solve_block(...)`

Candidate responsibilities to keep in `src/sparse_iterative.c` for this sprint:

- public handle initialization/free/prepare helpers
- CG scalar and matrix-free paths
- GMRES scalar and matrix-free paths
- BiCGStab scalar and matrix-free paths
- shared stagnation tracker
- progress timing helper

Responsibilities already outside the main file and not first-choice Day 7-9
work:

- MINRES scalar implementation in `src/sparse_iterative_minres.c`
- reusable workspace implementation in
  `src/sparse_iterative_workspace_internal.c`
- BiCGStab workspace structs in `src/sparse_bicgstab_internal.h`

### Solver Touched-File Plan

Possible implementation files for Days 7-9:

| File | Expected role |
|---|---|
| `src/sparse_iterative.c` | remove or simplify block wrapper cluster |
| `src/sparse_iterative_block.c` | optional new block wrapper source owner |
| `src/sparse_iterative_internal.h` | expose only private declarations needed by moved block wrappers |
| `Makefile` | add new source if a split is chosen |
| `CMakeLists.txt` | add new source if a split is chosen |

Files to avoid unless Day 7 finds a concrete need:

- `include/sparse_iterative.h`
- `src/sparse_iterative_minres.c`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_bicgstab_internal.h`
- solver benchmarks
- generated documentation

### Solver Extraction Rationale

The block-wrapper boundary reduces review cost because the current file mixes
scalar solver kernels, matrix-free adapters, handle reuse, convergence policy,
and multi-RHS wrapper aggregation. The block wrappers are mostly orchestration:
they dispatch one column at a time to existing scalar solvers and aggregate
result metadata.

This boundary is better than moving CG or GMRES internals first because those
kernels carry more numerical recurrence state. It is also better than moving
stagnation tracking first because that helper is shared across scalar solvers
and would create private-header churn without removing much visible review
mass.

### Solver Validation Owners

Any Day 7-9 code change in this lane must run:

- `make format`
- `make lint`
- `make test`

Targeted proof owners to inspect or rerun during development:

- `tests/test_iterative.c`
- `tests/test_block_solvers.c`
- `tests/test_minres.c`
- `tests/test_bicgstab.c`

Targeted code-reference scans after implementation:

- `rg -n "sparse_.*solve_block|solve_block_|iter_block_column_solver" src include tests`
- `rg -n "sparse_iterative_block|sparse_iterative.c" Makefile CMakeLists.txt`

## Adjacent Hotspots And Deferral Reasons

| Candidate | Day 3 decision |
|---|---|
| `src/sparse_lu_csr.c` | defer; large direct owner but less aligned with selected LDLT CSC cleanup |
| `src/sparse_ldlt.c` | defer; public LDLT owner and adjacent proof surface would widen direct scope |
| `src/sparse_chol_csc.c` | defer; proof pressure is already concentrated in `tests/test_chol_csc.c` |
| `src/sparse_qr.c` | keep as solver alternate; choose only if Day 7 rejects iterative block cleanup |
| `src/sparse_eigs.c` | defer; handle/restart cleanup is valuable but not the default Sprint 96 lane |
| `src/sparse_matrix.c` | defer; shared matrix shell has high cross-subsystem blast radius |
| `src/sparse_svd.c` | defer; lower priority than iterative/QR/eigs for this sprint |
| internal graph/direct headers | touch only when selected source movement requires it |

## Day 4 Boundary Questions

Day 4 should answer these before editing code:

1. Should the new direct source be named exactly `src/sparse_ldlt_dense.c`, or
   should it carry a CSC-specific name such as `src/sparse_ldlt_csc_dense.c`?
2. Should the implementation move the whole dense/backend block in one batch,
   or split backend probe movement from builtin dense-factor movement?
3. Are there any existing build-order or platform-guard assumptions around
   `<dlfcn.h>`, `<stdatomic.h>`, or backend environment parsing that need a
   smaller first batch?
4. Which focused direct tests are practical to run during development before
   the required full quality chain?

## Day 7 Boundary Questions

Day 7 should answer these before editing solver code:

1. Is `src/sparse_iterative_block.c` the right source name, or should the block
   wrappers remain local and only be reorganized?
2. Which private helper declarations would need to move into
   `src/sparse_iterative_internal.h` if block wrappers move?
3. Should BiCGStab block wrapper movement wait until after GMRES/MINRES block
   wrappers prove the extraction pattern?
4. Which focused solver tests are practical to run during development before
   the required full quality chain?

## Validation Plan

Day 3 itself changed planning artifacts only:

- run `git diff --check`
- run a trailing-whitespace scan over `docs/planning/EPIC_9/SPRINT_96`

Implementation days that modify `.c` or `.h` files must run the full required
quality chain before proceeding:

- `make format && make lint && make test`

If Days 4-9 add source files, also scan for stale build references and confirm
the new files are registered in both Makefile and CMake.

## Day 3 Result

Sprint 96 now has documented ownership boundaries before code movement:

- direct-family cleanup should first separate LDLT dense/backend ownership from
  `src/sparse_ldlt_csc.c`
- solver-family cleanup should default to iterative block-wrapper ownership in
  `src/sparse_iterative.c`
- broad public API, benchmark, generated documentation, and multi-family source
  movement remain out of scope
