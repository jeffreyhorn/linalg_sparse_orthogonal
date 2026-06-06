# Sprint 56 Day 5 - LDLT CSC decomposition batch 1

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Land the first bounded LDLT CSC source split from the Day 4 design:

- extract the supernodal helper cluster into a new owned source file
- keep the retained CSC lifecycle/native/orchestration path in
  `src/sparse_ldlt_csc.c`
- preserve the existing private-header contract
- preserve full proof and build parity

## Landed source split

New owned source file:

- `src/sparse_ldlt_csc_supernodal.c`

Moved function set:

- `ldlt_csc_bsearch_row_map(...)`
- `ldlt_csc_supernode_extract(...)`
- `ldlt_csc_supernode_writeback(...)`
- `ldlt_csc_supernode_eliminate_diag(...)`
- `ldlt_csc_supernode_eliminate_panel(...)`

Retained in `src/sparse_ldlt_csc.c`:

- lifecycle / conversion ownership
- wrapper compatibility path
- scalar/native Bunch-Kaufman kernel
- top-level `ldlt_csc_eliminate_supernodal(...)`
- solve path

## Private contract and build surfaces

The batch kept the existing private contract in:

- `src/sparse_ldlt_csc_internal.h`

Only bounded private-header change:

- top-level usage wording now names both
  `src/sparse_ldlt_csc.c` and `src/sparse_ldlt_csc_supernodal.c`

Build-surface updates:

- `Makefile`
- `CMakeLists.txt`

No public header/API changes were introduced.

## Measured ownership reduction

Post-split line counts:

- `src/sparse_ldlt_csc.c` = `2289`
- `src/sparse_ldlt_csc_supernodal.c` = `392`
- `src/sparse_ldlt_csc_internal.h` = `878`

Compared with the Sprint 56 Day 1 baseline:

- `src/sparse_ldlt_csc.c`: `2723 -> 2289`

Interpretation:

- the retained main file dropped by `434` lines
- the new file is a real owned backend slice, not a tiny spillover fragment

## Preserved behavior

The batch preserved:

- native versus wrapper routing
- permutation/pivot semantics
- `D` / `D_offdiag` / `pivot_size` handoff semantics
- direct CSC repeated-run behavior
- public direct-solver lifecycle behavior

No test or benchmark source change was needed to keep proof parity.

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
- `Total Test time (real) = 296.94 sec`

Focused follow-ons:

- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_ldlt` -> `84 / 84`
- `./build/test_integration` -> `37 / 37`
- `./build/example_analysis` -> residual `4.44e-16`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.52x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

## Conclusion

Sprint 56 Batch 1 is now a landed and validated decomposition result:

- the supernodal LDLT CSC helper cluster has its own owned source file
- the retained main CSC file is smaller and more focused
- the private-header and build-surface changes stayed bounded
- the full local and reviewed validation baseline remained exact

This gives Sprint 56 a real first Phase 2 maintainability landing rather than
only a decomposition design.
