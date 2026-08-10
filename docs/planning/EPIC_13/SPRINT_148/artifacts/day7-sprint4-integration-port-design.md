# Sprint 148 Day 7 Sprint 4 Integration Port Design

## Purpose

Day 7 turns the Day 4 `test_sprint4_integration` decision into an
implementation-ready plan. Day 4 selected a split-proof strategy as the safe
default, with full threaded promotion allowed only if the Day 5-6 portable
thread helper landed cleanly.

Day 6 did land the helper cleanly for `test_threads`, so Day 8 should first
attempt a direct helper-backed port of the existing `test_sprint4_integration`
source. The fallback remains a non-threaded split proof if MSVC or hosted
Windows behavior rejects the threaded SuiteSparse lane.

## Current Source Ownership

`tests/test_sprint4_integration.c` owns five integration checks:

| Test Function | Behavior | Thread Dependency | Windows Promotion Decision |
| --- | --- | --- | --- |
| `test_cholesky_csr_roundtrip_solve` | Factor SPD matrix, export Cholesky factor to CSR, import, mark factored, solve, and verify residual. | None | Promote. |
| `test_spmm_cholesky_reconstruct_nos4` | Load `nos4.mtx`, factor Cholesky, form `L * L^T`, and compare matvec reconstruction. | None | Promote. |
| `test_condest_via_lu_on_spd` | Load `nos4.mtx`, factor Cholesky and LU, estimate condition, solve, and check solution error. | None | Promote. |
| `test_concurrent_cholesky_suitesparse` | Four workers independently load `nos4.mtx`, factor Cholesky, solve scaled RHS values, and require residual `maxerr < 1e-8`. | Pthread mechanics only. | Promote through helper if feasible; otherwise retain staged. |
| `test_csr_cholesky_triangular` | Load `nos4.mtx`, factor Cholesky, export CSR, and verify lower-triangular structure. | None | Promote. |

The only source-level Windows blocker is direct pthread mechanics in the
concurrent SuiteSparse lane. `DATA_DIR`, SuiteSparse fixture loading, Cholesky,
LU, CSR, matmul, and residual behavior are already part of the broader CMake
test surface.

## Selected Primary Design

Use the Day 6 test-only helper to port the existing source directly.

Day 8 should make these mechanical changes:

| Current Pattern | Replacement |
| --- | --- |
| `#include <pthread.h>` | `#include "test_thread_helpers.h"` |
| `pthread_t threads[4]` | `test_thread_t threads[4]` |
| `pthread_create(&threads[t], NULL, thread_cholesky_suitesparse, &args[t])` | `test_thread_create(&threads[t], thread_cholesky_suitesparse, &args[t])` |
| `pthread_join(threads[t], NULL)` | `test_thread_join(&threads[t])` plus `ASSERT_EQ(rc, 0)` |

The worker function signature should stay unchanged:

```c
static void *thread_cholesky_suitesparse(void *arg);
```

This keeps the source aligned with `tests/test_threads.c` and avoids duplicate
Windows-native worker ownership.

## CMake Registration Plan

After the source compiles through the helper, Day 8 should promote
`test_sprint4_integration` in CMake:

```cmake
find_package(Threads)
add_sparse_test(test_threads)
add_sparse_test(test_sprint4_integration)
if(Threads_FOUND AND NOT WIN32)
    target_link_libraries(test_threads PRIVATE Threads::Threads)
    target_link_libraries(test_sprint4_integration PRIVATE Threads::Threads)
endif()
```

Rationale:

- Windows uses the helper's Win32 path and does not require `Threads::Threads`.
- POSIX keeps pthread linkage for both helper-backed tests.
- No new CTest name is needed if the direct port succeeds.
- The fallback split can still add a new CTest name if direct promotion fails.

## Expected Count Impact

| Surface | Count Impact |
| --- | --- |
| Local POSIX CTest | No expected count change; `test_sprint4_integration` is already registered locally. |
| Windows CMake after Day 6 thread promotion | Planned `57 -> 58` once `test_sprint4_integration` is registered and listed by hosted Windows `ctest -N`. |
| Full Sprint 148 plan after fuzz promotion | Still tentatively `56 -> 59` if `test_threads`, `test_sprint4_integration`, and `test_fuzz` all promote. |

Do not update `.github/workflows/windows-ci.yml` expected count during Day 8
unless the workflow policy has been deliberately moved earlier than Day 11.
The Day 11 CI promotion artifact remains the preferred owner for count and
staged-wording updates across all promoted tests.

## Fallback Split Design

If the direct helper-backed port fails on MSVC or hosted Windows, Day 8 should
fall back to a non-threaded split proof:

| Fallback Owner | Contents | Registration |
| --- | --- | --- |
| Existing `test_sprint4_integration` | Keep all five current tests, including threaded SuiteSparse lane. | POSIX-only, linked with `Threads::Threads`. |
| New `test_sprint4_integration_portable` | Four non-threaded tests: CSR roundtrip solve, `nos4` reconstruction, condest on SPD, CSR triangular proof. | Windows and POSIX CMake, no thread linkage. |

The fallback must explicitly retain the concurrent SuiteSparse lane as staged
for Windows and must not claim threaded Sprint 4 parity.

## Behavior-Preservation Checklist

Day 8 must preserve:

- `DATA_DIR` and `SS_DIR` fixture behavior;
- `nos4.mtx` reconstruction tolerance `maxdiff < 1e-8`;
- condition estimate positive-value check and Cholesky solve solution error
  `maxerr < 1e-10`;
- concurrent SuiteSparse worker count of four;
- per-thread scaled RHS behavior;
- concurrent SuiteSparse residual threshold `maxerr < 1e-8`;
- CSR lower-triangular check for every exported Cholesky factor entry;
- existing POSIX Makefile behavior and pthread-backed execution.

## Validation Checklist

Day 8 should run and record:

| Check | Purpose |
| --- | --- |
| `cmake -S . -B build` | Confirm CMake accepts the promoted registration. |
| `cmake --build build --target test_sprint4_integration` | Confirm focused integration build. |
| `ctest --test-dir build -R '^test_sprint4_integration$' --output-on-failure` | Confirm focused execution on the local platform. |
| `ctest --test-dir build -N` | Confirm local registration and record expected Windows delta separately. |
| `make format && make lint && make test` | Required because Day 8 will modify `.c` and CMake files. |

Hosted Windows MSVC CMake remains required before reviewed Windows wording can
claim the promoted integration surface.

## Support-Claim Boundaries

Allowed after hosted Windows proof:

- reviewed Windows CMake subset includes Sprint 4 Cholesky/CSR/condest
  integration coverage;
- if the direct helper-backed port passes, reviewed Windows CMake subset also
  includes the `test_sprint4_integration` concurrent independent SuiteSparse
  Cholesky worker lane.

Still not allowed:

- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broad Windows threading parity beyond the promoted tests;
- shared-library ABI, dynamic loader, or package-manager support.

## Rollback Rules

Roll back direct promotion and use the fallback split when:

- MSVC cannot compile the helper-backed source without changing the worker
  behavior;
- Windows execution fails because the concurrent SuiteSparse lane has a
  platform-specific runtime issue;
- POSIX execution changes test count, fixture loading, residual thresholds, or
  the concurrent worker count;
- CMake registration changes produce unexplained `ctest -N` count drift;
- workflow or documentation wording implies a broader Windows support claim
  than the promoted test supports.

## Day 8 Handoff

Implement the direct helper-backed port first. If focused build or execution
fails for platform-thread reasons, switch to the non-threaded split proof and
record the threaded SuiteSparse lane as a retained Windows-staged residual.
