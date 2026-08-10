# Sprint 148 Day 10: Fuzz And Property Port Implementation

## Purpose

Day 10 implements the Day 9 fuzz/property portability design. The result
removes the source-level Windows blocker from `test_fuzz` and promotes the
existing CTest target into the cross-platform CMake registration surface while
preserving POSIX behavior and deterministic parser/property coverage.

## Implementation Summary

| Area | Change | Rationale |
| --- | --- | --- |
| Platform includes | Guarded `<unistd.h>` behind non-Windows and added `_WIN32` `<windows.h>` support. | Removes the unconditional POSIX include that blocked MSVC compilation. |
| POSIX temp path | Kept `mkstemps(..., 4)` and `close(fd)` on non-Windows builds. | Preserves current Linux/macOS temp-file uniqueness and `.mtx` suffix behavior. |
| Windows temp path | Added `GetTempPathA`, `GetTempFileNameA`, and `MoveFileA` path setup. | Creates a unique temp file, moves it to a `.mtx`-suffixed path, and avoids POSIX APIs. |
| Cleanup | Switched cleanup to `remove(fuzz_tmp_path)`. | Works for both POSIX and Windows while preserving best-effort cleanup semantics. |
| Failure diagnostics | Added explicit Windows path-creation diagnostics and retained empty-path skip behavior. | Keeps parser fuzz skipped only when temp-file setup fails. |
| CMake registration | Removed the `NOT WIN32 AND NOT MSVC` gate around `test_fuzz`. | Makes the existing `test_fuzz` target eligible for reviewed Windows CMake CTest proof. |

## Behavior Preservation

- All 18 file-backed Matrix Market parser fuzz cases remain in
  `tests/test_fuzz.c`.
- The null-argument and nonexistent-file checks remain outside the temp-file
  skip block.
- LU, Cholesky, QR, SVD, large CSC lifecycle, and reorder/repeat property
  checks keep their existing deterministic seeds, sizes, and thresholds.
- The suite still uses one temp path for the file-backed parser cases and
  cleans that path at suite exit.
- POSIX builds still use the previous `mkstemps` path with `.mtx` suffix.
- Windows builds now avoid `<unistd.h>`, `mkstemps`, `close`, and `unlink`.

## CMake Registration Result

`test_fuzz` is now registered directly:

```cmake
add_sparse_test(test_sprint8_integration)
add_sparse_test(test_fuzz)
add_sparse_test(test_lu_csr)
```

The fallback split was not needed because the full helper path built and ran
locally.

## Count And Claim Status

| Surface | Status |
| --- | --- |
| Local POSIX CTest count | `ctest --test-dir build -N` reports `Total Tests: 59`; this is unchanged locally because `test_fuzz` was already registered on POSIX. |
| Planned Windows delta after Day 8 | `58 -> 59` once hosted Windows enumeration proves `test_fuzz` is registered. |
| Aggregate planned Windows delta | `56 -> 59` for the promoted `test_threads`, `test_sprint4_integration`, and `test_fuzz` set. |
| Workflow expected count | Not updated on Day 10; Day 11 owns CI count and staged-wording promotion for the batch. |
| Reviewed Windows claim | Still pending hosted Windows MSVC proof. |

## Focused Validation

| Command | Result |
| --- | --- |
| `cmake -S . -B build` | Passed. |
| `cmake --build build --target test_fuzz` | Passed. |
| `ctest --test-dir build -R '^test_fuzz$' --output-on-failure` | Passed: 1 test, `test_fuzz`, 36.28 s after the `.mtx` suffix fix. |
| `ctest --test-dir build -N` | Passed enumeration; local POSIX total remained 59. |

## Full Gate

Because Day 10 modified C source and CMake registration, the full C gate was
required and run after the final `.mtx` suffix implementation.

| Command | Result |
| --- | --- |
| `make format && make lint && make test` | Passed. |

Notable full-suite confirmation:

- `cppcheck` processed `tests/test_fuzz.c` with `_WIN32` defined and did not
  fail.
- `test_fuzz` passed with 28 tests, 0 failures, 0 skips, and 20,544
  assertions.
- The complete `make test` run ended with `All tests passed.`

## Residuals

- Hosted Windows MSVC CMake proof is still required before reviewed Windows
  wording can claim the promoted fuzz/property lane.
- `.github/workflows/windows-ci.yml` still carries the old expected count and
  staged wording until the Day 11 CI-promotion pass.
- Windows Makefile, Windows `pkg-config`, Windows reviewed install-validation,
  sanitizer parity, and unbounded fuzzing remain explicit non-claims.

## Day 11 Handoff

Batch the CMake and Windows CI promotion updates: update the expected Windows
CTest count from `56` to `59`, remove the three staged-test blocker statements
that no longer apply to the promoted CMake targets, and keep deferred Windows
Makefile/install/pkg-config wording explicit.
