# Sprint 148 Day 8: Sprint 4 Integration Port Implementation

## Purpose

Day 8 implements the selected helper-backed port for
`test_sprint4_integration`, promoting the existing Sprint 4 integration
coverage toward Windows CMake registration without weakening the POSIX test
surface.

## Implementation Summary

| Area | Change | Rationale |
| --- | --- | --- |
| Thread abstraction | Reused `tests/test_thread_helpers.h` in `tests/test_sprint4_integration.c`. | Keeps the worker function signature unchanged while hiding pthread versus Win32 lifecycle details. |
| Direct pthread usage | Removed the direct `<pthread.h>` include and replaced raw `pthread_t`, `pthread_create`, and `pthread_join` calls. | Eliminates the source-level Windows blocker identified during the Day 2 audit. |
| Join diagnostics | Added return-code assertions around `test_thread_join`. | Preserves failure visibility if a worker cannot be joined cleanly. |
| CMake registration | Registered `test_sprint4_integration` unconditionally through `add_sparse_test`. | Makes the target eligible for the reviewed Windows CMake CTest surface. |
| POSIX linkage | Kept `Threads::Threads` linkage only for non-Windows builds when `Threads_FOUND`. | Preserves pthread linkage on POSIX without imposing pthread requirements on MSVC. |

## Behavior Preservation

- The four non-threaded Sprint 4 integration checks remain in the same source
  file and keep their existing solver inputs, tolerances, and assertions.
- The concurrent SuiteSparse Cholesky lane remains active, still launches four
  workers, still uses scaled right-hand sides, and still requires
  `maxerr < 1e-8`.
- Worker function logic, result arrays, status aggregation, and diagnostic
  output were not changed.
- No split fallback was needed because the Day 6 helper supports the existing
  worker shape directly.

## CMake Registration Result

The CMake test-registration policy now treats `test_threads` and
`test_sprint4_integration` the same way:

- Both tests are registered by CMake on all platforms.
- POSIX builds link both tests to `Threads::Threads` when CMake finds the
  thread package.
- Windows builds avoid pthread linkage and rely on the helper's Win32 backend.

## Count And Claim Status

| Surface | Status |
| --- | --- |
| Local POSIX CTest count | `ctest --test-dir build -N` reports `Total Tests: 59`; this is unchanged locally because Sprint 4 integration was already registered on POSIX. |
| Planned Windows delta after Day 6 | `57 -> 58` once hosted Windows enumeration proves `test_sprint4_integration` is registered. |
| Aggregate planned Windows delta | `56 -> 58` for the promoted `test_threads` and `test_sprint4_integration` pair. |
| Workflow expected count | Not updated on Day 8; Day 11 owns CI count and staged-wording promotion after all selected ports are implemented. |
| Reviewed Windows claim | Still pending hosted Windows MSVC proof. |

## Focused Validation

| Command | Result |
| --- | --- |
| `cmake -S . -B build` | Passed. |
| `cmake --build build --target test_sprint4_integration` | Passed. |
| `ctest --test-dir build -R '^test_sprint4_integration$' --output-on-failure` | Passed: 1 test, `test_sprint4_integration`, 1.81 s. |
| `ctest --test-dir build -N` | Passed enumeration; local POSIX total remained 59. |

## Full Gate

Because Day 8 modified C source and CMake registration, the full C gate was
required.

| Command | Result |
| --- | --- |
| `make format && make lint && make test` | Passed. |

Notable full-suite confirmation:

- `test_threads` passed with 8 tests, 0 failures, 0 skips, and 132 assertions.
- `test_sprint4_integration` passed with 5 tests, 0 failures, 0 skips, and 48
  assertions.
- The complete `make test` run ended with `All tests passed.`

## Residuals

- Hosted Windows MSVC proof is still required before declaring the promoted
  Sprint 4 lane reviewed.
- `.github/workflows/windows-ci.yml` still carries the old expected count and
  staged wording until the Day 11 CI-promotion pass.
- `test_fuzz` remains staged for Windows and is the Day 9/Day 10 portability
  target.

## Day 9 Handoff

Design the portable fuzz temp-file strategy, preserving malformed-input parser
coverage, deterministic property lanes, large CSC lifecycle checks, and current
skip semantics before editing `test_fuzz`.
