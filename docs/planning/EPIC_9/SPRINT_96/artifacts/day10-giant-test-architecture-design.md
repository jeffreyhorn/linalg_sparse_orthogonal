# Sprint 96 Day 10: Giant-Test Architecture Design

## Purpose

Day 10 defines one bounded giant-test reduction before any proof-owner test
files move. The selected cleanup is a split of the largest retained proof owner,
`tests/test_chol_csc.c`, into one core Cholesky CSC proof owner and one
supernodal/writeback proof owner.

Day 10 is a design day. No `.c` or `.h` files are changed by this artifact.

## Reviewed Proof Owners

Largest retained proof-owner tests after the Day 5-9 source cleanups:

| File | Lines | Day 10 decision |
| --- | ---: | --- |
| `tests/test_chol_csc.c` | 5029 | Select for Sprint 96 test split |
| `tests/test_ldlt_csc.c` | 3680 | Keep as residual candidate |
| `tests/test_integration.c` | 3421 | Defer; broader workflow owner |
| `tests/test_qr.c` | 3234 | Defer; not in selected Sprint 96 lane |
| `tests/test_ldlt.c` | 2977 | Defer; lower current pressure |
| `tests/test_etree.c` | 2962 | Defer; cohesive proof owner |
| `tests/test_graph.c` | 2925 | Defer; cohesive proof owner |
| `tests/test_iterative.c` | 2841 | Defer; solver source cleanup already landed |
| `tests/test_svd.c` | 2766 | Defer; not in selected Sprint 96 lane |
| `tests/test_reorder_nd.c` | 2340 | Defer; lower current pressure |

`tests/test_chol_csc.c` is the strongest target because it is the largest test
owner and already contains explicit runner groups for core CSC behavior,
supernodal plumbing, writeback, and dispatch. That existing grouping lets the
cleanup reduce review cost without weakening proof ownership.

## Selected Cleanup Batch

Create a new test executable:

- `tests/test_chol_csc_supernodal.c`

Keep the existing executable:

- `tests/test_chol_csc.c`

Use the existing family-local helper header as the shared helper owner:

- `tests/test_chol_csc_supernodal_helpers.h`

The helper header should remain narrow and CSC-family-specific. Do not widen
`tests/test_solver_helpers.h` or `tests/test_framework.h` for this split.

## Split Boundary

Move these proof groups from `tests/test_chol_csc.c` to
`tests/test_chol_csc_supernodal.c`:

- supernode detection:
  - `run_supernode_detection_tests(...)`
  - tests currently beginning at `test_detect_supernodes_null_args(...)`
- supernodal postorder:
  - `run_supernodal_postorder_tests(...)`
- dense supernodal primitives and backend contracts:
  - `run_supernodal_dense_tests(...)`
  - dense Cholesky primitive checks
  - dense solve checks
  - dense backend environment-contract checks
  - dense LDLT factor cross-checks currently housed in this Cholesky CSC proof
    owner
- supernode extract/writeback plumbing:
  - `run_supernode_extract_writeback_tests(...)`
- diagonal block factor tests:
  - `run_supernode_diag_factor_tests(...)`
- panel and full batched path integration tests:
  - `run_supernode_panel_tests(...)`
- parametrised scalar/batched cross-checks:
  - `run_supernodal_parametrised_tests(...)`
- CSC linked-list writeback roundtrips and rejection tests:
  - `run_writeback_tests(...)`

Keep these proof groups in `tests/test_chol_csc.c`:

- allocation and growth tests
- conversion roundtrip tests
- permutation-cache tests
- symbolic validation and edge-case tests
- workspace/elimination scaffold tests
- scalar kernel tests
- solve/residual/shim tests
- transparent dispatch tests:
  - `run_dispatch_tests(...)`
  - external dense-reference tests

Dispatch stays in the core file for the first split because it crosses public
`sparse_cholesky_factor_opts(...)` behavior, environment/backend routing, and
external dense-reference platform guards. That can become a separate cleanup
only if the first split proves the remaining core file is still too dense.

## Expected Suite Shape

`tests/test_chol_csc.c` should keep:

```c
TEST_SUITE_BEGIN("chol_csc");
```

`tests/test_chol_csc_supernodal.c` should use:

```c
TEST_SUITE_BEGIN("chol_csc_supernodal");
```

The new suite should include only the moved runner groups. It should not change
assertions, matrices, tolerances, skip behavior, or helper semantics.

## Build And Platform Consequences

Makefile registration:

- add `$(TESTDIR)/test_chol_csc_supernodal.c \` immediately after
  `$(TESTDIR)/test_chol_csc.c \` in `TEST_SRCS`

CMake registration:

- add `add_sparse_test(test_chol_csc_supernodal)` immediately after
  `add_sparse_test(test_chol_csc)`

Platform handling:

- no new platform gate is planned
- preserve any existing `_WIN32` guards inside moved tests
- leave external dense-reference dispatch tests in `tests/test_chol_csc.c` for
  the first split to avoid broadening platform churn

Suite labels:

- keep the existing `chol_csc` label for the core owner
- introduce `chol_csc_supernodal` for the moved internal supernodal/writeback
  owner

## Validation Contract

During implementation, use focused checks for a fast signal:

```sh
make build/test_chol_csc
make build/test_chol_csc_supernodal
./build/test_chol_csc
./build/test_chol_csc_supernodal
```

Because Day 11 is expected to modify `.c` and build files, completion requires:

```sh
make format && make lint && make test
```

After implementation, run stale-reference scans:

```sh
rg -n "run_supernode|run_supernodal|run_writeback|chol_dense|ldlt_dense_factor" tests/test_chol_csc.c tests/test_chol_csc_supernodal.c tests/test_chol_csc_supernodal_helpers.h
rg -n "run_dispatch|external_dense|sparse_cholesky_factor_opts" tests/test_chol_csc.c tests/test_chol_csc_supernodal.c
rg -n "test_chol_csc_supernodal|test_chol_csc.c" Makefile CMakeLists.txt tests
```

Expected scan results:

- supernodal and writeback runner groups live in
  `tests/test_chol_csc_supernodal.c`
- dispatch and external dense-reference tests remain in `tests/test_chol_csc.c`
- both build systems register the new test executable
- helper usage remains family-local

## Explicit Non-Goals

The test split should not include:

- production source changes
- public header changes
- assertion rewrites
- tolerance changes
- fixture rewrites unrelated to the move
- broad shared test-helper expansion
- changes to `tests/test_ldlt_csc.c`
- changes to integration, QR, eigensolver, SVD, reorder, or graph proof owners
- generated documentation updates

## Residual Queue

Residual giant-test candidates after this batch:

- `tests/test_ldlt_csc.c`, especially if future direct-family cleanup lands in
  LDLT CSC supernodal or dense backend code
- `tests/test_integration.c`, if workflow-level integration ownership remains
  a review bottleneck
- the transparent dispatch group in `tests/test_chol_csc.c`, only after the
  first Cholesky CSC split is validated

## Day 10 Exit Decision

Day 11 should implement the bounded `tests/test_chol_csc_supernodal.c` split
described above. The first split should reduce the largest proof-owner file
without changing behavior, assertions, public API, or production source.
