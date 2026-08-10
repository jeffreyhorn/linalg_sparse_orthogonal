# Sprint 148 Day 6 Thread Test Port Implementation

## Purpose

Day 6 implemented the selected `test_threads` portability design from Day 5.
The change removes raw pthread usage from `tests/test_threads.c`, isolates
platform thread APIs in a test-only helper, and changes CMake registration so
`test_threads` can be registered on Windows while `test_sprint4_integration`
remains POSIX-only for the Day 7-8 split decision.

## Implementation Summary

| Area | Change | Result |
| --- | --- | --- |
| Test helper | Added `tests/test_thread_helpers.h`. | Provides `test_thread_t`, `test_thread_create`, and `test_thread_join`. |
| POSIX mapping | Helper includes `<pthread.h>` and forwards create/join to pthread APIs. | Existing Linux/macOS behavior stays pthread-backed. |
| Windows mapping | Helper includes `<windows.h>` and wraps `CreateThread`, `WaitForSingleObject`, and `CloseHandle`. | `test_threads` no longer requires `<pthread.h>` on MSVC. |
| `test_threads` source | Replaced `pthread_t`, `pthread_create`, and `pthread_join` usage with helper calls. | Worker logic, thresholds, stress counts, and iteration counts are unchanged. |
| Join diagnostics | Added `ASSERT_EQ(rc, 0)` for helper joins. | Join failures now report through the test framework. |
| CMake | Registered `test_threads` outside the POSIX-only gate and kept `Threads::Threads` linkage for non-Windows builds. | `test_sprint4_integration` stays under `Threads_FOUND AND NOT WIN32`. |

## Behavior Preservation

The following `test_threads` behaviors remain unchanged:

- four independent LU factor/solve workers with `max_error < 1e-10`;
- four shared LU solve workers with 100 iterations and residual
  `max_error < 1e-8`;
- four shared Cholesky solve workers with 100 iterations and residual
  `max_error < 1e-8`;
- eight-thread LU and Cholesky stress paths with `STRESS_ITERS=1000`;
- eight-thread independent LU stress path;
- concurrent `sparse_norminf` cache exercise with four workers and 1000
  iterations;
- concurrent norminf-plus-solve proof with one norm worker and three solve
  workers;
- optional `SPARSE_MUTEX` concurrent insert proof and default skip behavior.

## CMake Registration Result

The CMake thread section now has this ownership:

| Test | Registration | Linkage |
| --- | --- | --- |
| `test_threads` | Always registered by CMake. | Links `Threads::Threads` only when `Threads_FOUND AND NOT WIN32`. |
| `test_sprint4_integration` | Still registered only when `Threads_FOUND AND NOT WIN32`. | Links `Threads::Threads` on POSIX only. |

This intentionally promotes only the `test_threads` source. The Sprint 4
integration split remains Day 7-8 work.

## Count And Claim Status

| Surface | Status |
| --- | --- |
| Local POSIX CTest enumeration | `59` total tests after implementation. This is unchanged from the POSIX surface because `test_threads` was already registered locally. |
| Planned Windows CTest enumeration | `56 -> 57` once hosted MSVC CMake enumeration sees `test_threads`. |
| `.github/workflows/windows-ci.yml` count | Not changed on Day 6; Day 11 owns the expected-count and workflow wording update after the sprint has the implementation batch and evidence map. |
| Reviewed Windows claim | Pending hosted Windows proof. Do not claim reviewed Windows thread coverage until Windows CI passes with the promoted test. |

## Focused Validation

| Check | Result |
| --- | --- |
| `cmake -S . -B build` | Passed. |
| `cmake --build build --target test_threads` | Passed. |
| `ctest --test-dir build -R '^test_threads$' --output-on-failure` | Passed: `1/1 Test #17: test_threads` passed in `1.10 sec`. |
| `ctest --test-dir build -N` | Passed locally; total tests reported as `59`. |

## Required Full Gate

Because Day 6 modified `.c` and `.h` files, the required full C gate was run:

```text
make format && make lint && make test
```

Result: passed. The final Makefile summary reported `All tests passed.`

## Residuals

- Hosted Windows MSVC proof is still required before support-tier wording can
  claim reviewed Windows `test_threads` coverage.
- `.github/workflows/windows-ci.yml` still has the old expected count and staged
  wording until the Day 11 CI promotion step.
- `test_sprint4_integration` still includes pthread APIs directly and remains
  POSIX-only pending Days 7-8.
- `test_fuzz` still depends on POSIX temp-file APIs and remains Day 9-10 work.

## Day 7 Handoff

Use the portable helper as a possible input for the Sprint 4 threaded lane, but
do not couple Day 7's design to full threaded Sprint 4 promotion. The Day 4
selected path remains to split non-threaded Sprint 4 integration coverage first
and retain or conditionally port the pthread-backed SuiteSparse concurrency
lane separately.
