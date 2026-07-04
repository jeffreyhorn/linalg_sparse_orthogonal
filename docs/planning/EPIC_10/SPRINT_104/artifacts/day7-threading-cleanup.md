# Sprint 104 Day 7 Threading Cleanup

## Purpose

Day 7 implements the highest-value low-risk cleanup from the Day 6 audit:
make the OpenMP/runtime-control model explicit without changing behavior.

## Implemented Scope

| surface | change | behavior impact |
|---|---|---|
| `src/sparse_matrix.c` | documented that SpMV and block SpMV leave thread count to the OpenMP runtime | none |
| `src/sparse_eigs.c` | documented that MGS reorth does not call `omp_set_num_threads` or translate `SPARSE_*` settings into team size | none |
| `docs/algorithm.md` | added user-facing OpenMP runtime-control and nested-parallelism guidance | none |
| `docs/maintainer_guide.md` | added maintained OpenMP/runtime-control interpretation and validation expectations | none |
| `WORKING_NOTES.md` | recorded Day 7 cleanup and validation plan | none |

No public APIs, ABI fields, option structs, enum values, OpenMP schedules,
thresholds, or environment-variable semantics changed.

## Runtime-Control Model After Cleanup

The maintained model is now explicit:

- serial builds remain the default product path;
- `SPARSE_OPENMP` is compile-time opt-in;
- OpenMP team size and affinity remain owned by the OpenMP runtime;
- callers use normal OpenMP controls such as `OMP_NUM_THREADS`;
- the library does not expose `sparse_set_num_threads`;
- `SPARSE_MUTEX` remains separate from OpenMP runtime control;
- graph/reorder thread-local overrides remain internal scope mechanisms.

## Focused Runtime Validation

Because the source cleanup is comment-only and does not alter schedules or
thresholds, focused validation should prove the affected owners still compile
and run:

1. Build and run `test_omp` for the SpMV surface.
2. Build and run `test_eigs` for the MGS reorth owner.
3. Run the full required C quality gate because `.c` files changed.

## Non-Claims

This cleanup does not claim:

- OpenMP is enabled by default;
- OpenMP is available on every reviewed platform;
- nested parallelism is optimized;
- runtime thread count is configurable through a library API;
- graph/FM compatibility env vars are public threading controls;
- direct solver kernels are parallelized.

## Day 8 Handoff

Day 8 can design performance sentinels against a clearer runtime model. Any
sentinel that compares serial and OpenMP behavior should record build mode,
OpenMP runtime thread count, matrix class, and whether the path reaches OpenMP
directly through SpMV or indirectly through eigensolver/SVD composition.

## Completion Check

| criterion | status |
|---|---|
| highest-value Day 6 cleanup selected | complete |
| serial-build behavior preserved | complete |
| public option semantics preserved | complete |
| thread-count ownership documented | complete |
| global vs thread-local controls clarified | complete |
| validation plan recorded | complete |
