# Sprint 57 Day 8 - solver-family test refactor batch 1

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Land the first bounded solver-family giant-test refactor by extracting the
partial-SVD proof family out of `tests/test_svd.c` into the Day 7-selected
build-neutral local helper seam.

## Files landed

- `tests/test_svd.c`
- `tests/test_svd_partial_helpers.h`

## Ownership change

### New owned seam

`tests/test_svd_partial_helpers.h` now owns the partial-SVD-family proof layer:

- `test_partial_svd_*`
- `test_partial_svd_vectors_*`

This includes the backend-oriented partial-SVD proof and the partial-SVD vector
proof without widening into unrelated full-SVD, low-rank, or condition-number
coverage.

### Retained in `tests/test_svd.c`

- Golub-Kahan extraction / validation groups
- bidiagonal and full-SVD groups
- low-rank / pseudoinverse / condition-number groups
- Sprint 29 outer-product / full-mode follow-through
- `main()` and existing `RUN_TEST(...)` order

## Preserved fence

The landing stayed inside the Day 7 boundary:

- no new test target
- no `Makefile` changes
- no `CMakeLists.txt` changes
- same `test_svd` binary shape
- same `main()` ownership in `tests/test_svd.c`
- same `RUN_TEST(...)` ordering
- same fixture coverage and proof intent

This was an ownership/readability change, not an SVD behavior change.

## Measured reduction

Line counts after landing:

- `tests/test_svd.c` = `2766`
- `tests/test_svd_partial_helpers.h` = `915`

Against the Day 7 baseline:

- `tests/test_svd.c`: `3746 -> 2766`

That is a real giant-test reduction while keeping the proof runner intact.

## Validation

### Required gate

- `make format`
- `make lint`
- `make test`

All passed.

### Reviewed baseline

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 225.66 sec`

### Focused touched-surface follow-ons

- `./build/test_svd` -> `97 / 97`
- `./build/test_sprint8_integration` -> `7 / 7`
- `./build/bench_svd`
- `./build/example_svd_lowrank`

Representative retained outputs:

- `bench_svd`
  - `nos4` partial/full = `2.1x`
  - `west0067` partial/full = `2.2x`
  - `bcsstk04` partial/full = `1.5x`
  - `steam1` partial/full = `11.3x`
  - `orsirr_1` partial/full = `236.2x`
- `example_svd_lowrank`
  - sparse low-rank `k=2`: `22 -> 6` nnz
  - compression = `3.7x`

## Conclusion

Sprint 57 Day 8 delivered the first solver-family giant-test refactor as a
clean owned seam:

- the partial-SVD proof family is now explicitly separated
- `tests/test_svd.c` is materially smaller
- the build/test surface stayed stable
- the reviewed baseline and focused SVD follow-ons stayed green

This gives Sprint 57 a real solver-family maintainability landing, not just a
design placeholder.
