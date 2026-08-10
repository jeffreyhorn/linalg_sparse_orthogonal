# Sprint 148 Day 5 Thread Test Port Design

## Purpose

Day 5 turns the Day 4 `test_threads` disposition into an implementation-ready
design. The selected path is a small test-only thread helper that preserves the
current POSIX pthread proof and makes the same `test_threads` source compile and
run in the reviewed Windows MSVC CMake lane.

Day 6 owns the code changes. This artifact fixes the intended helper boundary,
assertion policy, CMake registration path, validation checklist, and rollback
rules before implementation starts.

## Current Test Shape

`tests/test_threads.c` currently uses the same worker shape throughout:

```c
static void *worker_name(void *arg);
```

Each test allocates fixed-size worker handle arrays, creates every worker,
joins every worker, then checks per-worker success fields and numeric residuals.

| Behavior | Current Thread Count | Current Iterations | Preservation Requirement |
| --- | --- | --- | --- |
| Independent LU factor/solve workers | 4 | 1 solve per worker | Preserve separate matrices per worker and `max_error < 1e-10`. |
| Shared LU solve workers | 4 | 100 | Preserve shared factored matrix reads and residual `max_error < 1e-8`. |
| Shared Cholesky solve workers | 4 | 100 | Preserve shared factored matrix reads and residual `max_error < 1e-8`. |
| LU solve stress | 8 | 1000 | Preserve worker count and iteration count. |
| Cholesky solve stress | 8 | 1000 | Preserve worker count and iteration count. |
| Independent stress | 8 | 1 solve per worker | Preserve independent allocation/factor/solve behavior. |
| Concurrent `sparse_norminf` | 4 | 1000 | Preserve cached-norm concurrency coverage. |
| Concurrent norminf plus solve | 4 | 500 | Preserve one norm worker plus three solve workers. |
| Optional concurrent insert | 4 | row partitions | Preserve existing `SPARSE_MUTEX` opt-in behavior and default skip. |

## Helper API Boundary

Create a test-only helper header, tentatively `tests/test_thread_helpers.h`.
The helper should remain private to tests and should not expose any public
library API.

```c
typedef void *(*test_thread_fn)(void *);

typedef struct {
#ifdef _WIN32
    HANDLE handle;
    test_thread_fn fn;
    void *arg;
    void *result;
#else
    pthread_t handle;
#endif
} test_thread_t;

static int test_thread_create(test_thread_t *thread, test_thread_fn fn, void *arg);
static int test_thread_join(test_thread_t *thread);
```

### POSIX Mapping

On non-Windows platforms:

- include `<pthread.h>` inside the helper, not in `tests/test_threads.c`;
- call `pthread_create(&thread->handle, NULL, fn, arg)`;
- call `pthread_join(thread->handle, NULL)`;
- return the pthread return code from create/join;
- keep CMake `Threads::Threads` linkage for POSIX builds.

### Windows Mapping

On Windows:

- include `<windows.h>` inside the helper;
- use `CreateThread` with a tiny adapter function because Win32 workers return
  `DWORD`, while the existing test workers return `void *`;
- store `fn`, `arg`, and returned `result` in `test_thread_t`;
- return `0` from `test_thread_create` when `CreateThread` succeeds and `1`
  otherwise;
- call `WaitForSingleObject(thread->handle, INFINITE)`;
- call `CloseHandle(thread->handle)` after a successful or failed wait;
- return `0` only when wait and close succeed;
- do not add timeouts in the first port because the existing pthread tests wait
  indefinitely, and adding a timeout would change failure semantics.

The Windows adapter should look conceptually like:

```c
static DWORD WINAPI test_thread_entry(LPVOID arg) {
    test_thread_t *thread = (test_thread_t *)arg;
    thread->result = thread->fn(thread->arg);
    return 0;
}
```

## `test_threads.c` Refactor Rules

Day 6 should make only mechanical source changes in `tests/test_threads.c`:

| Current Pattern | Replacement |
| --- | --- |
| `#include <pthread.h>` | `#include "test_thread_helpers.h"` |
| `pthread_t threads[N]` | `test_thread_t threads[N]` |
| `pthread_create(&threads[t], NULL, worker, &args[t])` | `test_thread_create(&threads[t], worker, &args[t])` |
| `pthread_join(threads[t], NULL)` | `test_thread_join(&threads[t])` |

The worker functions, matrix construction, residual thresholds, stress counts,
iteration counts, and `SPARSE_MUTEX` conditional should remain unchanged.

## Assertion And Failure Diagnostics

The first implementation should preserve the existing assertion style:

- `ASSERT_EQ(rc, 0)` after every create call;
- `ASSERT_EQ(rc, 0)` after every join call after the join helper is introduced;
- existing per-worker `printf` diagnostics should remain unchanged;
- optional `SPARSE_MUTEX` insert coverage should keep the current default skip
  when `SPARSE_MUTEX` is not defined.

Adding explicit join return assertions is acceptable because it strengthens
diagnostics without changing the test behavior. If this produces too much churn
for Day 6, keep create assertions and use unchecked joins only as a temporary
step; the implementation artifact must record that residual.

## CMake Registration Plan

Day 6 should update the `test_threads` registration only after the helper and
local focused build succeed.

Proposed CMake shape:

```cmake
find_package(Threads)

add_sparse_test(test_threads)
if(Threads_FOUND AND NOT WIN32)
    target_link_libraries(test_threads PRIVATE Threads::Threads)
endif()

if(Threads_FOUND AND NOT WIN32)
    add_sparse_test(test_sprint4_integration)
    target_link_libraries(test_sprint4_integration PRIVATE Threads::Threads)
endif()
```

Rationale:

- Windows does not need `Threads::Threads` for the Win32 helper path.
- POSIX retains `Threads::Threads` linkage for pthread-backed helper code.
- `test_sprint4_integration` stays under its existing POSIX-only gate until
  Days 7-8 split or port it.
- The Windows workflow expected count should not change until `ctest -N`
  confirms the registered count.

## Expected Count Implication

If only `test_threads` is promoted on Day 6, the planned Windows reviewed CTest
count changes from `56` to `57`.

Do not update `.github/workflows/windows-ci.yml` unless the implementation
evidence includes:

- `test_threads` listed in CTest enumeration;
- total test count observed as `57`;
- local or hosted execution evidence for `test_threads`;
- workflow staged-exclusion text updated to remove `test_threads` from the
  staged blocker list while keeping `test_sprint4_integration` and `test_fuzz`
  staged.

## Validation Checklist

Day 6 should record the feasible local results in its implementation artifact:

| Check | Purpose |
| --- | --- |
| `cmake -S . -B build` | Confirm CMake config accepts promoted `test_threads`. |
| `cmake --build build --target test_threads` | Confirm local focused build. |
| `ctest --test-dir build -R '^test_threads$' --output-on-failure` | Confirm focused execution on the local platform. |
| `ctest --test-dir build -N` | Confirm local registration and count delta. |
| `make format && make lint && make test` | Required full C gate because Day 6 will modify `.c`/`.h` files. |

If the local platform cannot prove Windows behavior, Day 6 should still record
the local POSIX result and mark hosted Windows CMake proof as required before
reviewed Windows claims are updated.

## Residuals And Non-Claims

This design does not claim:

- public threading API support;
- Windows Makefile support;
- Windows `pkg-config` support;
- Windows install-validation parity;
- sanitizer parity on Windows;
- shared-library ABI, dynamic loader, or package-manager support;
- full `test_sprint4_integration` Windows promotion.

The only intended Day 6 support movement is that `test_threads` can become part
of the reviewed Windows CMake CTest subset after the implementation and hosted
proof pass.

## Rollback Rules

Roll back the Day 6 promotion if:

- the helper requires changing worker signatures throughout the test;
- POSIX `test_threads` behavior weakens or stress counts decrease;
- `SPARSE_MUTEX` opt-in behavior is removed or silently skipped differently;
- Windows registration changes without a matching expected-count update and
  enumeration evidence;
- local focused checks pass but full C gate fails and cannot be resolved within
  Day 6;
- docs or workflow output imply broader Windows parity than this single test
  promotion supports.

## Day 6 Handoff

Implement `tests/test_thread_helpers.h`, refactor `tests/test_threads.c` to use
the helper, promote only `test_threads` in CMake, and run the focused checks
before deciding whether to update Windows workflow count and staged wording.
