# Sprint 96 Day 7: Solver/Algorithm Cleanup Boundary Freeze

## Purpose

Day 7 freezes the second implementation cleanup batch before solver source
edits begin. The selected batch is the iterative block-solver ownership split
from `src/sparse_iterative.c`.

Day 7 is a planning and boundary-freeze day. No `.c` or `.h` files are changed
by this artifact.

## Frozen Solver Target

Primary source owner:

- `src/sparse_iterative.c`

New source owner to create on Day 8:

- `src/sparse_iterative_block.c`

Internal contract owner to use only if needed:

- `src/sparse_iterative_internal.h`

Build owners to update:

- `Makefile`
- `CMakeLists.txt`

Proof owners:

- `tests/test_block_solvers.c`
- `tests/test_minres.c`
- `tests/test_bicgstab.c`
- `tests/test_iterative.c`
- `tests/test_sprint10_integration.c`

## Exact Move Boundary

Move the current iterative block-solver cluster from `src/sparse_iterative.c`
into `src/sparse_iterative_block.c`.

The moved cluster should include:

- block CG public implementation:
  - `sparse_cg_solve_block(...)`
- shared independent-column block dispatch:
  - `iter_block_column_solver_fn`
  - `solve_block_independent_columns(...)`
- block GMRES adapter and public implementation:
  - `solve_block_gmres_column(...)`
  - `sparse_gmres_solve_block(...)`
- block MINRES adapter and public implementation:
  - `solve_block_minres_column(...)`
  - `sparse_minres_solve_block(...)`
- block BiCGStab adapter and public implementation:
  - `solve_block_bicgstab_column(...)`
  - `sparse_bicgstab_solve_block(...)`

The public declarations in `include/sparse_iterative.h` must remain unchanged.

## Helper Dependency Map

The new block owner needs existing public and internal solver helpers.

Expected includes for `src/sparse_iterative_block.c`:

- `sparse_iterative.h`
- `sparse_iterative_internal.h`
- `sparse_matrix_internal.h`
- `sparse_vector.h`
- `<math.h>`
- `<stdint.h>`
- `<stdlib.h>`
- `<string.h>`

The block CG implementation currently depends on:

- `sparse_matvec_block(...)`
- `sparse_iter_workspace_prepare_block_cg(...)`
- `sparse_iter_workspace_free(...)`
- `sparse_idx_to_size_checked(...)`
- `sparse_size_mul_overflow(...)`
- vector helpers such as `vec_norm2(...)`, `vec_zero(...)`, `vec_copy(...)`,
  `vec_dot(...)`, and `vec_axpy(...)`
- result reset and convergence marking semantics
- the same default CG option values used by scalar CG

The block GMRES, MINRES, and BiCGStab wrappers dispatch one right-hand side at
a time through existing scalar public solver functions. They should keep that
behavior unless a compile or test failure proves a narrower adjustment is
required.

## Internal Contract Plan

Prefer the smallest private contract that lets the block owner compile without
duplicating solver behavior.

Allowed Day 8 internal-header edits:

- add private declarations for result reset and converged-state helpers if
  moving block CG requires sharing them across files
- add comments that identify `src/sparse_iterative_block.c` as an internal
  consumer of iterative workspace helpers

Avoid broad internal API changes:

- do not expose scalar GMRES workspace internals to the block owner
- do not move stagnation tracker internals
- do not change public option structs or public result structs
- do not change public solver signatures

If Day 8 keeps local block defaults, those defaults must match the current
scalar CG default values:

- `max_iter = 1000`
- `tol = 1e-10`
- `atol = 0.0`
- `restart = 0`

## Files Expected To Change On Days 8-9

Expected implementation files:

- `src/sparse_iterative.c`
- `src/sparse_iterative_block.c`
- `src/sparse_iterative_internal.h`, only if helper declarations are required

Expected build files:

- `Makefile`
- `CMakeLists.txt`

Expected Sprint 96 planning files:

- `docs/planning/EPIC_9/SPRINT_96/WORKING_NOTES.md`
- Day 8 and Day 9 artifacts under
  `docs/planning/EPIC_9/SPRINT_96/artifacts/`

## Explicit Non-Goals

Days 8-9 should not include:

- public API changes
- public header edits under `include/`
- scalar CG algorithm changes
- scalar GMRES or matrix-free GMRES changes
- scalar MINRES changes in `src/sparse_iterative_minres.c`
- scalar or matrix-free BiCGStab algorithm changes
- iterative workspace implementation changes in
  `src/sparse_iterative_workspace_internal.c`
- BiCGStab private helper header changes in `src/sparse_bicgstab_internal.h`
- QR, eigensolver, SVD, LU, LDLT, Cholesky, matrix-shell, or benchmark-driver
  cleanup
- generated documentation edits
- proof-owner test rewrites

## Targeted Proof Plan

Development-time focused checks, if a quick local signal is useful:

- `make build/test_block_solvers`
- `make build/test_minres`
- `make build/test_bicgstab`
- `make build/test_iterative`
- `make build/test_sprint10_integration`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_iterative`
- `./build/test_sprint10_integration`

Required completion check after any Day 8 or Day 9 `.c` or `.h` change:

```sh
make format && make lint && make test
```

## Benchmark Sanity List

No benchmark behavior change is planned. The full build/lint/test chain should
still compile benchmark-related owners that depend on iterative solver headers
or objects. If Day 8 encounters a benchmark compile failure, inspect:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_convergence.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_main.c`

Benchmark source edits are out of scope unless required by a compile failure
from the block-solver source split.

## Stale-Reference Scans

After Day 8 implementation, run:

```sh
rg -n "sparse_.*solve_block|solve_block_|iter_block_column_solver" src include tests
rg -n "sparse_iterative_block|sparse_iterative.c" Makefile CMakeLists.txt src
rg -n "block-solver|block solver|right-hand side" src/sparse_iterative.c src/sparse_iterative_block.c src/sparse_iterative_internal.h
```

The scans should show:

- public block solver declarations still in `include/sparse_iterative.h`
- block solver implementations in `src/sparse_iterative_block.c`
- scalar solver implementations remaining in `src/sparse_iterative.c` and
  existing specialized owners
- build systems registering `src/sparse_iterative_block.c`
- no stale claim in `src/sparse_iterative.c` that it owns all block solver
  implementations

## Risk Notes

Result semantics risk:

- Block CG currently writes aggregate result fields directly. If result reset
  helpers move from file-local to internal scope, keep the field initialization
  and converged-state semantics identical.

Default-options risk:

- Block CG has its own algorithm and must preserve the current default option
  values when `opts == NULL`.

Adapter behavior risk:

- GMRES, MINRES, and BiCGStab block wrappers currently solve each column by
  dispatching to scalar public functions. Preserve that adapter behavior.

Scope risk:

- This cleanup is a source ownership split, not a numerical-method rewrite.
  Any solver convergence behavior change should be treated as a regression
  unless a proof-owner test demonstrates an existing defect.

## Day 7 Exit Decision

Day 8 should create `src/sparse_iterative_block.c` and move only the iterative
block-solver ownership cluster listed above. All unrelated solver families and
scalar solver algorithms remain out of scope.
