# Sprint 148 Day 11: CMake And Windows CI Promotion Batch

## Purpose

Align the Windows CMake CI policy with the promoted Sprint 148 test targets
from Days 6, 8, and 10, while keeping non-promoted Windows support claims
explicitly out of scope.

## Reconciled Promotion Outcomes

| Day | Target | Outcome | Windows CTest Delta |
| --- | --- | --- | --- |
| Day 6 | `test_threads` | Refactored to `tests/test_thread_helpers.h`; CMake now registers the target on Windows and on POSIX builds with `Threads_FOUND`, linking `Threads::Threads` only for POSIX. | `56` to `57` |
| Day 8 | `test_sprint4_integration` | Refactored concurrent checks to the portable helper; CMake now registers the target on Windows and on POSIX builds with `Threads_FOUND`. | `57` to `58` |
| Day 10 | `test_fuzz` | Replaced POSIX-only temp-file mechanics with a portable `.mtx` temp path; CMake now registers the target on all platforms. | `58` to `59` |

## CMake State

No Day 11 `CMakeLists.txt` edit was required. The target-registration state
was already aligned by the Day 6, Day 8, and Day 10 implementation work:

- `test_threads` is registered outside the old Windows gate.
- `test_sprint4_integration` is registered outside the old Windows gate.
- `test_fuzz` is registered outside the old Windows/MSVC gate.
- `Threads::Threads` linkage remains conditional for the portable thread
  helper users.

## Windows CI Updates

Updated `.github/workflows/windows-ci.yml` to:

- raise `EXPECTED_WINDOWS_CTEST_COUNT` from `56` to `59`;
- replace old staged-exclusion comments for `test_threads`,
  `test_sprint4_integration`, and `test_fuzz`;
- state that the reviewed Windows CMake subset now includes the three promoted
  portable tests;
- keep hosted Windows `ctest -N` plus full `ctest` as the proof gate;
- preserve the non-claims for Windows Makefile parity, separate reviewed
  install validation, pkg-config parity, shared-library support, and dynamic
  ABI support.

## Count Rationale

The Sprint 148 Windows count moves from `56` to `59` because the branch removes
the three named Windows CMake exclusions:

1. `test_threads`
2. `test_sprint4_integration`
3. `test_fuzz`

Local POSIX CTest enumeration remains `Total Tests: 59`. Hosted Windows CI is
still required to prove that the MSVC lane enumerates and executes the same
reviewed CMake surface.

## Report And Support Metadata

No public report/support metadata was changed on Day 11. The older user-facing
Windows wording in `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md`
is intentionally left for Day 12 so the workflow policy promotion and public
documentation alignment remain separately reviewable.

Day 12 must update those documents to remove stale statements that
`test_threads`, `test_sprint4_integration`, and `test_fuzz` are outside the
reviewed Windows CMake subset.

## Validation

- `cmake -S . -B build`
- `ctest --test-dir build -N`
- `git diff --check`

Expected local CTest enumeration after the promotion batch: `Total Tests: 59`.

No `.c` or `.h` files were edited on Day 11, so the full C gate was not rerun
for this workflow-only change. The Day 10 full gate remains the latest source
validation for the promoted test implementations.

## Residuals

- Hosted Windows proof remains pending on the PR.
- Day 12 owns the public/support documentation refresh.
- Windows Makefile parity, separate reviewed install validation, pkg-config
  parity, shared-library support, and dynamic ABI support remain non-claims.
