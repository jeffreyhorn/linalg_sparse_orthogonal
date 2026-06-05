# Sprint 56 Day 8 - Cholesky CSC decomposition batch

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Land the first bounded Cholesky CSC source split from the Day 7 design:

- extract the Cholesky-owned supernodal backend into a new owned source file
- keep the retained lifecycle/scalar/control path in `src/sparse_chol_csc.c`
- preserve the existing private-header contract
- preserve full proof and build parity

## Landed source split

New owned source file:

- `src/sparse_chol_csc_supernodal.c`

Moved function set:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

Retained in `src/sparse_chol_csc.c`:

- lifecycle / conversion ownership
- validation
- scalar workspace and native elimination/solve core
- wrapper / dispatch glue
- `chol_csc_writeback_to_sparse(...)`
- shared dense LDLT helpers:
  - `ldlt_dense_sym_swap(...)`
  - `ldlt_dense_factor(...)`

## Private contract and bounded retained-file tightening

The batch kept the existing private contract in:

- `src/sparse_chol_csc_internal.h`

Only bounded private-header change:

- top-level usage wording now names both
  `src/sparse_chol_csc.c` and `src/sparse_chol_csc_supernodal.c`
- the Cholesky dense/supernodal status comments now match the live ownership
  and behavior contract

The retained main file also needed one bounded analyzer-facing cleanup pass:

- `bsearch_row(...)`
- `chol_csc_scatter(...)`
- `chol_csc_gather(...)`

Those helpers now use explicit bounded slice/count logic instead of relying on
less obvious pointer/index coupling, but their behavioral contract did not
change.

Build-surface updates:

- `Makefile`
- `CMakeLists.txt`

No public header/API changes were introduced.

## Measured ownership reduction

Post-split line counts:

- `src/sparse_chol_csc.c` = `1625`
- `src/sparse_chol_csc_supernodal.c` = `544`
- `src/sparse_chol_csc_internal.h` = `979`

Compared with the Sprint 56 Day 1 baseline:

- `src/sparse_chol_csc.c`: `2194 -> 1625`

Interpretation:

- the retained main file dropped by `569` lines
- the new file is a real owned Cholesky backend slice rather than a tiny spill
  file

## Preserved behavior

The batch preserved:

- scalar versus supernodal Cholesky parity
- supernode-detection semantics
- `min_size` threshold behavior
- CSC writeback-to-sparse semantics
- one-shot and shared analysis-aware CSC routing
- public direct-solver lifecycle behavior

No test, benchmark, or example source change was needed to keep proof parity.

## Validation

Required gate:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 279.57 sec`

Focused follow-ons:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_integration` -> `37 / 37`
- `./build/example_analysis` -> residual `4.44e-16`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.53x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

## Conclusion

Sprint 56 Batch 2 is now a landed and validated decomposition result:

- the Cholesky-owned supernodal CSC backend has its own owned source file
- the retained main CSC file is materially smaller and more focused
- the private-header and build-surface changes stayed bounded
- the required local and reviewed validation baseline remained exact

This gives Sprint 56 a real second Phase 2 maintainability landing rather than
only a Cholesky decomposition design.
