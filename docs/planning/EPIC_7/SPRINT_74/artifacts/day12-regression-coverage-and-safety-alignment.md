# Sprint 74 Day 12: Regression Coverage & Safety Alignment

## Objective

Confirm that the landed Sprint 74 capability seams already have the right
focused proof owners, add only the minimum regression follow-through if a real
gap remains, and fix the exact Day 13 validation queue.

## Inputs Re-read

- `tests/test_sparse_matrix.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `README.md`
- `docs/maintainer_guide.md`

## Result

### 1. No new regression code was needed

The touched Sprint 74 seams already sit in the right focused proof owners:

- `tests/test_sparse_matrix.c` owns the width-contract lane
- `tests/test_iterative.c` owns the iterative public scalar seam
- `tests/test_eigs.c` owns the eigensolver public scalar seam

Those proof owners already cover the sustained contract points:

- `SPARSE_IDX_BITS`
- `IDX_MAX`
- `sparse_idx_bits()`
- `sparse_scalar_t`
- `sparse_scalar_bits()`

Adding broader or duplicate regression on Day 12 would reduce ownership
clarity rather than improve safety.

### 2. The maintained wording was already aligned after Day 11

No new docs or header wording was required:

- `README.md` already carries the bounded caller-facing capability summary
- `docs/maintainer_guide.md` already names the focused proof owners directly
- touched public headers already express the width/scalar seams truthfully

### 3. The real Day 12 output is the explicit Day 13 validation queue

The Day 13 validation set is now fixed to:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/quality-review-cmake/test_sparse_matrix`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_refactor_csc`
- `./build/quality-review-cmake/bench_chol_csc`
- `./build/quality-review-cmake/bench_iterative_reuse`
- `./build/quality-review-cmake/bench_eigs_reuse`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

## Build / Reference Ownership

The sustained Sprint 74 ownership split is now explicit:

- tests own the width/scalar contract truth
- examples remain adoption/context surfaces
- benchmarks remain capability/reporting context
- install scripts remain install/package proof surfaces

## Bottom Line

Sprint 74 Day 12 confirmed that the landed capability boundary already has the
right proof owners. No extra regression code was justified, and the exact Day
13 validation queue is now fixed from the post-Day-11 state.
