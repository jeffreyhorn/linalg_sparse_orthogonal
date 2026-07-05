# Sprint 106 Day 10 - Direct and Graph Fixture Extraction

## Goal

Implement the first behavior-preserving giant-test helper extraction batch from
the Day 9 boundary. The extraction must reduce local helper ownership in large
tests without weakening assertions, changing test names, or changing reviewed
CTest registration.

## Helper Owners Added

### `tests/test_graph_fixtures.h`

New graph/reorder fixture helper owner for shared synthetic graph and partition
mechanics:

- `tf_make_grid_2d(...)`
- `tf_make_path_1d(...)`
- `tf_make_mesh_3d(...)`
- `tf_make_two_cliques_with_bridge(...)`
- `tf_check_partition_invariant(...)`
- `tf_count_partition_sides(...)`
- `tf_count_bipartition_sides(...)`
- `tf_compute_cut(...)`
- `tf_compute_side_weights(...)`

Affected tests:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Notes:

- Graph-local weighted edge mutation remains in `tests/test_graph.c`.
- The asymmetric boundary graph fixture remains in `tests/test_graph.c` because
  it is not shared by the reorder owner.
- The nested-dissection-specific cached SuiteSparse fixture copy helpers remain
  in `tests/test_reorder_nd.c` because they are stateful and family-scoped.

### `tests/test_direct_solver_helpers.h`

New direct-solver assertion helper owner for LU CSR proof mechanics:

- `tf_assert_sparse_matrices_equal(...)`
- `tf_verify_lu_csr_factorization(...)`
- `tf_sparse_residual_norminf(...)`

Affected test:

- `tests/test_lu_csr.c`

Notes:

- `tests/test_ldlt_csc.c` and `tests/test_direct_csc_regression.c` were not
  edited in this batch because the selected first direct-solver seam was LU CSR
  assertion/residual ownership.
- Focused LDLT CSC and direct CSC regression tests were still run to prove the
  adjacent direct-solver surface stayed clean.

## Test Registration

No test targets, `RUN_TEST(...)` names, test ordering, or CTest registration
changed. The helper extraction is header-only, so no Makefile or CMake target
source list updates were required.

Reviewed CMake registration remains:

- CMake tests: 54
- Makefile tests: 54

## Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 lines | 2,758 lines | -167 |
| `tests/test_reorder_nd.c` | 2,340 lines | 2,304 lines | -36 |
| `tests/test_lu_csr.c` | 1,899 lines | 1,806 lines | -93 |
| `tests/test_graph_fixtures.h` | 0 lines | 195 lines | +195 |
| `tests/test_direct_solver_helpers.h` | 0 lines | 93 lines | +93 |

## Validation

- Source-list validation passed:
  - `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (45 library sources)`
- Focused validation passed:
  - `make build/test_graph build/test_reorder_nd build/test_lu_csr build/test_ldlt_csc build/test_direct_csc_regression`
  - `./build/test_graph`: 61 tests, 0 failed, 0 skipped, 1,762 assertions
  - `./build/test_reorder_nd`: 35 tests, 0 failed, 1 skipped, 105 assertions
  - `./build/test_lu_csr`: 53 tests, 0 failed, 0 skipped, 1,062,184 assertions
  - `./build/test_ldlt_csc`: 100 tests, 0 failed, 0 skipped, 2,335 assertions
  - `./build/test_direct_csc_regression`: 8 tests, 0 failed, 0 skipped, 42 assertions
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed

## Deferred Work

- Integration lifecycle helper extraction remains deferred to Day 11.
- QR oracle/residual helper extraction remains deferred until after the direct
  and graph helper pattern has one complete validation cycle.
- Broader direct CSC fixture movement remains deferred; Day 10 intentionally
  started with assertion and residual helpers to avoid obscuring direct CSC test
  intent.

## Exit Criteria

Day 10 satisfies the fixture extraction criteria: giant tests are smaller,
helper ownership is clearer, assertions and test registration are unchanged,
focused direct/graph/reorder validation passes, the required full C quality gate
passes, and reviewed CMake test-count parity remains intact.
