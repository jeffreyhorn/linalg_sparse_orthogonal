# Sprint 148 Day 2 Staged Test Source Audit

## Purpose

Day 2 audits the source-level blockers in the three Windows-staged tests before
any portability design or CMake expected-count change. The audit separates
platform mechanics from test behavior so later days can choose direct ports,
Windows-native equivalents, split proof owners, retained staged status, or
explicit rejection without weakening the existing POSIX/Linux/macOS proof.

## Source Files Audited

| Test | Source File | Current Windows Status | Primary Blocker |
| --- | --- | --- | --- |
| `test_threads` | `tests/test_threads.c` | Staged out by CMake | Direct pthread API usage. |
| `test_sprint4_integration` | `tests/test_sprint4_integration.c` | Staged out by CMake | Direct pthread API usage in one integration lane. |
| `test_fuzz` | `tests/test_fuzz.c` | Staged out by CMake | POSIX temp-file APIs and `<unistd.h>`. |

## Per-Test Blocker Inventory

### `test_threads`

| Source Behavior | Exact Source Evidence | Blocker Class | Behavior To Preserve |
| --- | --- | --- | --- |
| Includes pthread directly. | `#include <pthread.h>` near file header. | pthread API | Any promoted Windows lane must compile without POSIX pthread headers. |
| Independent LU workers. | `thread_independent_lu` plus `pthread_create`/`pthread_join` in `test_independent_lu_threads`. | pthread API | Four independent matrix create/factor/solve workers; per-thread success and `max_error < 1e-10`. |
| Concurrent shared LU solves. | `thread_concurrent_solve` plus `test_concurrent_solve_shared`. | pthread API | Four threads solve against one factored LU matrix for 100 iterations and verify residual `max_error < 1e-8`. |
| Concurrent shared Cholesky solves. | `thread_concurrent_cholesky_solve` plus `test_concurrent_cholesky_solve`. | pthread API | Four threads solve against one factored Cholesky matrix for 100 iterations and verify residual `max_error < 1e-8`. |
| Stress shared LU and Cholesky solves. | `STRESS_THREADS=8`, `STRESS_ITERS=1000`, `test_lu_solve_stress`, `test_cholesky_solve_stress`. | pthread API and runtime budget | Eight worker threads, 1000 iterations, all worker success flags true. |
| Independent stress. | `test_independent_stress`. | pthread API | Eight independent LU worker paths complete successfully. |
| Concurrent `sparse_norminf`. | `thread_concurrent_norminf`, `test_concurrent_norminf`. | pthread API and atomic/cache behavior | Multiple concurrent norm calculations agree with expected norm and exercise cached norm safety. |
| Concurrent norminf plus solves. | `test_concurrent_norminf_and_solve`. | pthread API and shared-factor behavior | One norm thread and three solve threads share a factored matrix without corrupting success results. |
| Optional concurrent insert under `SPARSE_MUTEX`. | `#ifdef SPARSE_MUTEX` block with `thread_concurrent_insert` and `test_concurrent_insert`. | pthread API plus optional mutex build mode | Preserve opt-in shared insertion proof only when `SPARSE_MUTEX` is enabled; default lane prints skip. |

Key implementation constraint: this file is a broad thread-safety regression
suite, not a tiny compile-only smoke test. A Windows promotion must either
preserve the high-value lifecycle coverage or explicitly split smaller proof
owners without implying full pthread parity.

### `test_sprint4_integration`

| Source Behavior | Exact Source Evidence | Blocker Class | Behavior To Preserve |
| --- | --- | --- | --- |
| Includes pthread directly. | `#include <pthread.h>` near file header. | pthread API | Any promoted Windows lane must compile without POSIX pthread headers. |
| Most integration tests are not thread-specific. | `test_cholesky_csr_roundtrip_solve`, `test_spmm_cholesky_reconstruct_nos4`, `test_condest_via_lu_on_spd`, `test_csr_cholesky_triangular`. | CMake gate currently hides broader file | Cholesky/CSR roundtrip, SuiteSparse `nos4` reconstruction, LU condest on SPD input, and CSR triangular checks should remain covered on POSIX and may be candidates for split registration. |
| Concurrent Cholesky SuiteSparse lane. | `thread_cholesky_suitesparse` plus `test_concurrent_cholesky_suitesparse`. | pthread API | Four threads independently load `nos4.mtx`, factor with Cholesky, solve scaled RHS values, and verify residual `maxerr < 1e-8`. |
| SuiteSparse fixture dependency. | `SS_DIR DATA_DIR "/suitesparse"` and `sparse_load_mm(... "nos4.mtx")`. | data fixture/runtime dependency | Keep `DATA_DIR` behavior intact and do not turn a Windows thread port into an unrelated data-path change. |

Key implementation constraint: the current file-level CMake exclusion hides
four non-pthread integration tests from Windows because one lane uses pthread.
Day 4 should consider whether to split the threaded proof from the non-threaded
Sprint 4 integration coverage.

### `test_fuzz`

| Source Behavior | Exact Source Evidence | Blocker Class | Behavior To Preserve |
| --- | --- | --- | --- |
| Includes POSIX unistd. | `#include <unistd.h>`. | POSIX API | Windows-compatible lane must avoid direct `<unistd.h>` dependency under MSVC. |
| Creates one unique `.mtx` temp path. | `fuzz_tmp_path[256]`, `TMPDIR` fallback to `/tmp`, `mkstemps(fuzz_tmp_path, 4)`. | POSIX temp-file API | Unique writable `.mtx` path with deterministic suffix behavior for Matrix Market parser tests. |
| Closes and unlinks temp file. | `close(fd)` after `mkstemps`, `unlink(fuzz_tmp_path)` in cleanup. | POSIX file API | File descriptor/resource cleanup and final artifact cleanup. |
| Reuses temp path for parser fuzz cases. | `try_load_mm` writes text content to `fuzz_tmp_path`; binary and symmetric tests write directly. | temp-file lifecycle | Preserve parser fuzz behavior for malformed headers, dimensions, indices, NaN/Inf, binary garbage, UTF-8 comments, whitespace, comments, duplicate entries, and symmetric mirroring. |
| Skips file-backed fuzz cases if temp creation fails. | `if (fuzz_tmp_path[0]) { RUN_TEST(...) } else { printf("SKIP...") }`. | skip policy | Preserve explicit skip when a writable temp file cannot be created. |
| Non-file fuzz cases are independent. | `test_fuzz_null_args`, `test_fuzz_nonexistent_file`. | none | These can run without temp-file helper support. |
| Property tests use deterministic seeds. | seed loops and static seed arrays for LU, Cholesky, QR, SVD, Cholesky lifecycle, LDLT lifecycle, reorder/repeat checks. | deterministic property policy | Preserve seeded deterministic behavior and pass-count thresholds. |
| Large property lanes depend on CSC thresholds. | Uses `SPARSE_CSC_THRESHOLD + 12` and KKT/SPD builders. | runtime/fixture behavior | Do not reduce property coverage as a side effect of temp-file portability. |

Key implementation constraint: the actual Windows blocker is the temporary file
helper, not the entire property suite. Day 4 should consider separating parser
temp-file mechanics from property tests if that yields a safer Windows proof
owner.

## Pthread And POSIX API Usage Table

| API/Header | Files | Usage | Portability Risk |
| --- | --- | --- | --- |
| `<pthread.h>` | `tests/test_threads.c`, `tests/test_sprint4_integration.c` | Declares `pthread_t`, `pthread_create`, and `pthread_join`. | MSVC cannot compile this header/API directly. |
| `pthread_t` | `tests/test_threads.c`, `tests/test_sprint4_integration.c` | Fixed-size arrays of worker handles. | Requires abstraction or platform split. |
| `pthread_create` | `tests/test_threads.c`, `tests/test_sprint4_integration.c` | Starts worker functions returning `void *`. | Windows equivalent requires matching function signature or wrapper. |
| `pthread_join` | `tests/test_threads.c`, `tests/test_sprint4_integration.c` | Waits for every worker before assertions. | Windows equivalent must preserve join-before-assert behavior. |
| `<unistd.h>` | `tests/test_fuzz.c` | Provides `close` and `unlink` declarations. | MSVC lacks POSIX unistd. |
| `mkstemps` | `tests/test_fuzz.c` | Creates a unique writable `.mtx` temp path. | No direct MSVC equivalent; suffix-preserving helper needed or file-backed lane split. |
| `close` | `tests/test_fuzz.c` | Closes descriptor from `mkstemps`. | Windows uses different low-level close APIs unless using C runtime wrappers. |
| `unlink` | `tests/test_fuzz.c` | Removes the temp file during cleanup. | Windows deletion can use `remove` or platform-specific APIs, but lifecycle semantics must be checked. |

## Behavior-Preservation Map

| Test Surface | Must Preserve | Must Not Accidentally Claim |
| --- | --- | --- |
| Thread lifecycle | Independent worker allocation/factor/solve, shared LU/Cholesky solve safety, stress iterations, concurrent cached norm behavior, optional `SPARSE_MUTEX` insert proof. | Full pthread parity, broad Windows thread-safety proof beyond promoted tests, or Windows install/package support. |
| Sprint 4 integration | Cholesky CSR roundtrip, SuiteSparse `nos4` reconstruction, LU condest on SPD input, concurrent Cholesky SuiteSparse behavior, CSR triangular proof. | Broad Windows integration parity or broad SuiteSparse coverage. |
| Fuzz/property | Matrix Market malformed-input fuzz cases, explicit temp-creation skip, non-file argument checks, deterministic seeded LU/Cholesky/QR/SVD/property lanes, large CSC lifecycle/reorder properties. | Broad fuzzing completeness, Windows parser superiority, portable performance, or generated report proof. |

## Shared Helper Opportunities

| Candidate | Applies To | Opportunity | Risk |
| --- | --- | --- | --- |
| Tiny test-thread wrapper | `test_threads`, `test_sprint4_integration` | Hide POSIX/Windows thread creation and join behind a small test-only helper. | Wrapper can obscure thread-count, return-code, and join semantics if too generic. |
| Split thread proof files | `test_sprint4_integration` | Keep non-threaded integration tests registered on Windows while retaining POSIX concurrent SuiteSparse proof separately. | File split can create CMake/source-list drift and duplicate setup code. |
| Portable temp-file helper | `test_fuzz` | Replace `mkstemps`/`close`/`unlink` with a test-local helper that supports MSVC. | Must preserve unique `.mtx` path, cleanup, and skip-on-failure semantics. |
| File-backed versus property split | `test_fuzz` | Register non-file property tests on Windows even if parser temp-file helper remains staged. | Split must avoid implying full fuzz parser coverage on Windows. |

## Linux/macOS Preservation Warnings

- Do not remove the existing POSIX pthread proof while adding a Windows path.
- Do not lower stress counts or iteration budgets unless the reduction is
  explicitly tied to a separate Windows-specific lane and POSIX coverage
  remains intact.
- Do not hide `SPARSE_MUTEX` concurrent insertion coverage behind a Windows
  wrapper that changes its opt-in semantics.
- Do not replace `DATA_DIR` or SuiteSparse fixture behavior while solving
  pthread registration.
- Do not reduce deterministic property seed coverage in `test_fuzz` as part of
  the temp-file portability work.
- Do not update `EXPECTED_WINDOWS_CTEST_COUNT` until Day 3 registration audit
  and Day 4 disposition decisions explain the intended promoted surface.

## Day 3 Handoff

Day 3 should audit CMake and workflow registration against this source audit.
Specific questions:

1. Should non-threaded `test_sprint4_integration` behavior be split so Windows
   can run it independently from the pthread lane?
2. Should `test_fuzz` be split between file-backed parser fuzz cases and
   temp-file-free property cases?
3. Should `test_threads` use a shared test-thread wrapper or remain split by
   platform-specific proof owner?
4. How should CTest expected-count policy represent promoted subsets versus
   retained staged tests?

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every staged test blocker is tied to exact source behavior. | Complete | Per-test blocker tables identify pthread and POSIX API usage and the behavior each source path owns. |
| Behavior to preserve is separated from platform mechanics. | Complete | Behavior-preservation map distinguishes lifecycle/property coverage from pthread and temp-file APIs. |
| No implementation direction is chosen before the blocker audit is complete. | Complete | Helper opportunities and handoff questions are listed as candidates only; Day 4 owns the disposition decision. |
