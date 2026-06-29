# Sprint 95 Day 11: Proof-Owner Naming Cleanup

## Purpose

Day 11 lands the bounded proof-owner rename batch selected on Day 10. The batch
moves the direct CSC dispatch tests from sprint-numbered names to
product-oriented owner names while preserving coverage and build behavior.

## Rename Batch

| Old owner | New owner | Notes |
|---|---|---|
| `tests/test_sprint18_integration.c` | `tests/test_direct_csc_dispatch.c` | Owns cross-threshold Cholesky CSC dispatch and forced backend parity. |
| `tests/test_sprint19_integration.c` | `tests/test_direct_csc_regression.c` | Owns the retained direct-family CSC regression bundle. |
| `tests/test_sprint20_integration.c` | `tests/test_ldlt_backend_dispatch.c` | Owns the LDL^T backend selector and AUTO/forced dispatch proof surface. |

## Reference Updates

| Surface | Update |
|---|---|
| `Makefile` | Replaced the three renamed test sources in `TEST_SRCS`. |
| `CMakeLists.txt` | Replaced the three CTest target names. |
| Renamed test files | Updated file headers and `TEST_SUITE_BEGIN(...)` labels. |
| `benchmarks/bench_ldlt_csc.c` | Updated the KKT fixture comment to cite `test_ldlt_backend_dispatch.c`. |
| `src/sparse_ldlt_csc.c` | Updated the retained NOTE reference to `test_direct_csc_regression.c`. |
| `src/sparse_ldlt_csc_internal.h` | Updated the supernodal fill-row note to `test_direct_csc_regression.c`. |
| `docs/maintainer_guide.md` | Added the three product-oriented proof owners to the direct-family proof map. |

## Deferred Work Preserved

- `test_sprint4_integration` remains unchanged because it is thread-gated and
  mentioned in the Windows staged-exclusion workflow.
- Older mixed owners remain unchanged until a split-first design exists:
  `test_sprint5_integration`, `test_sprint6_integration`,
  `test_sprint8_integration`, `test_sprint10_integration`,
  `test_sprint11_integration`, `test_sprint12_integration`,
  `test_sprint13_integration`, and `test_sprint29_integration`.
- Historical planning artifacts and captured logs keep the old sprint names.

## Validation Plan

Day 11 renamed `.c` files and changed build hooks, so the required quality chain
is:

```bash
make format
make lint
make test
```

Additional reference checks:

```bash
rg -n "test_sprint18|test_sprint19|test_sprint20" . --glob '!docs/planning/**' --glob '!build/**'
rg -n "test_direct_csc_dispatch|test_direct_csc_regression|test_ldlt_backend_dispatch" Makefile CMakeLists.txt benchmarks src tests docs/maintainer_guide.md
```

## Validation Result

- `make format && make lint && make test` passed.
- The renamed suites compiled and ran as:
  - `test_direct_csc_dispatch`
  - `test_direct_csc_regression`
  - `test_ldlt_backend_dispatch`
- Active stale-reference scan passed: no `test_sprint18`, `test_sprint19`, or
  `test_sprint20` references remain outside `docs/planning/**` and `build/**`.
- New-owner reference scan passed across Make, CMake, source, benchmark, test,
  and maintainer-guide surfaces.
- `git diff --check` passed.
- Trailing-whitespace scan passed for touched Day 11 files and Sprint 95
  planning artifacts.

## Day 11 Result

The selected direct CSC proof owners are now discoverable by product capability
rather than sprint chronology. Build and maintainer references point at the new
owners, while deferred historical names remain intentionally unchanged.
