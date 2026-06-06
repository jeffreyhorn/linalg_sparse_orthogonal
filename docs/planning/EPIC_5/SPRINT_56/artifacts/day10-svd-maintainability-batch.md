# Sprint 56 Day 10 - SVD maintainability batch

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Land the bounded SVD maintainability split selected on Day 9:

- extract the partial-SVD Lanczos backend into a new owned source file
- keep the retained full-SVD/public orchestration path in `src/sparse_svd.c`
- preserve the existing private-header contract
- remove stale sprint-history narrative from the touched SVD implementation
  area while preserving useful algorithm commentary
- preserve full proof and build parity

## Landed source split

New owned source file:

- `src/sparse_svd_partial.c`

Moved function set:

- `sparse_svd_partial(...)`

Retained in `src/sparse_svd.c`:

- low-rank sparse reconstruction toggle plus outer-product path
- reflector extraction and bidiagonal QR machinery
- full-SVD orchestration and full-mode basis padding
- application wrappers:
  - `sparse_svd_rank(...)`
  - `sparse_pinv(...)`
  - `sparse_svd_lowrank(...)`
  - `sparse_svd_lowrank_sparse(...)`
  - `sparse_cond(...)`

## Private contract, build surfaces, and touched comment cleanup

The batch kept the existing private contract in:

- `src/sparse_svd_internal.h`

Only bounded private-header change:

- the top-level usage wording now names the retained full-SVD path, the
  partial-SVD backend, and the selected proof/benchmark surfaces that still
  rely on the shared helper declarations

Build-surface updates:

- `Makefile`
- `CMakeLists.txt`

Touched permanent-code cleanup:

- removed stale sprint-history narrative from the retained `src/sparse_svd.c`
  blocks while keeping the durable algorithm/rationale commentary
- rechecked the touched SVD implementation surfaces with:
  - `rg -n "Sprint|Day [0-9]+" src/sparse_svd.c src/sparse_svd_partial.c src/sparse_svd_internal.h`
  - result: no remaining sprint-history matches

No public header/API changes were introduced.

## Measured ownership reduction

Post-split line counts:

- `src/sparse_svd.c` = `1323`
- `src/sparse_svd_partial.c` = `402`
- `src/sparse_svd_internal.h` = `22`

Compared with the Sprint 56 Day 1 baseline:

- `src/sparse_svd.c`: `1728 -> 1323`

Interpretation:

- the retained main file dropped by `405` lines
- the new file is a real owned partial-SVD backend slice rather than a tiny
  spill file

## Preserved behavior

The batch preserved:

- full-SVD/public entry behavior
- partial-SVD sigma-only and vector-recovery behavior
- bidiagonal QR helper ownership and semantics
- low-rank, pseudoinverse, rank, and condition-number wrapper behavior
- benchmark and example meaning

No public SVD contract change was introduced by the split.

## Validation

Required gate on the final Day 10 source state:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 281.71 sec`

Focused follow-ons:

- `./build/test_svd` -> `97 / 97`
- `./build/test_sprint8_integration` -> `7 / 7`
- `./build/bench_svd`
  - `nos4`: partial/full `2.3x`
  - `west0067`: partial/full `1.3x`
  - `bcsstk04`: partial/full `1.3x`
  - `steam1`: partial/full `11.8x`
  - `orsirr_1`: partial/full `260.9x`
- `./build/example_svd_lowrank`
  - condition number = `1.41e+03`
  - sparse low-rank `k=2`: `22 -> 6` nnz (`3.7x` compression)

## Conclusion

Sprint 56 Day 10 is now a landed bounded SVD maintainability result:

- the partial-SVD Lanczos backend has its own owned source file
- the retained main SVD file is materially smaller and more focused
- the private-header and build-surface changes stayed bounded
- the touched permanent-code commentary now reads as durable algorithm and
  ownership guidance rather than sprint history
- the required local and reviewed validation baseline remained exact

This gives Sprint 56 a real SVD maintainability landing rather than leaving
`src/sparse_svd.c` as a residual large-file cleanup item.
