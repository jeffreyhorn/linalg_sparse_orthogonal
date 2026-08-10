# Sprint 148 Day 9: Fuzz And Property Port Design

## Purpose

Day 9 turns the Day 4 `test_fuzz` decision into an implementation-ready
portable temp-file design. The goal is to promote the existing malformed
Matrix Market parser fuzz checks and deterministic property checks into the
reviewed Windows CMake surface without changing parser behavior or leaving
temporary-file residue behind.

## Current Source Ownership

`tests/test_fuzz.c` owns two behavior groups:

| Group | Tests | Platform Blocker | Promotion Target |
| --- | --- | --- | --- |
| File-backed Matrix Market fuzz | Empty file, malformed headers, missing dimensions, bad dimensions, truncated entries, bad indices, NaN/Inf values, huge dimensions, binary garbage, UTF-8 comments, whitespace, comments, duplicate entries, and symmetric flag handling. | The suite-level temp-file helper uses `<unistd.h>`, `mkstemps`, `close`, and `unlink`. | Promote through a portable temp-file helper. |
| Platform-neutral properties and argument checks | Null args, nonexistent path, LU/Cholesky/QR/SVD properties, large CSC Cholesky/LDLT lifecycle, and reorder/repeat properties. | None in test mechanics. | Keep in the same promoted `test_fuzz` target. |

The Day 2 audit showed that the blocker is the test harness temp-file
mechanics, not the fuzz cases or property logic.

## Selected Primary Design

Implement a small test-only temp-file helper inside `tests/test_fuzz.c` unless
the implementation grows enough to justify extraction. The helper should
preserve the existing suite shape:

```c
static char fuzz_tmp_path[256];
static void fuzz_init_tmp(void);
static void fuzz_cleanup_tmp(void);
static sparse_err_t try_load_mm(const char *content);
```

The helper should change only platform mechanics:

| Platform | Creation Strategy | Cleanup Strategy | Notes |
| --- | --- | --- | --- |
| POSIX | Keep `mkstemps(..., 4)` with a `.mtx` suffix, then `close(fd)`. | Keep `unlink(fuzz_tmp_path)`. | Preserves current Linux/macOS behavior and concurrent-safe uniqueness. |
| Windows/MSVC | Use the process temp directory plus a unique path generated with CRT-safe APIs such as `tmpnam_s` followed by an explicit `.mtx` suffix, or `_mktemp_s` on a bounded template under `%TEMP%`. | Use `remove(fuzz_tmp_path)` from `<stdio.h>`. | Avoids `<unistd.h>`, `mkstemps`, `close`, and `unlink`; keeps the file path readable by `sparse_load_mm`. |

The implementation should include `<unistd.h>` only on non-Windows builds and
include any Windows/CRT-specific headers behind `_WIN32`.

## Helper Semantics

The helper must provide:

- one writable path for the whole process, matching the current suite-level
  temp-file lifecycle;
- a `.mtx` suffix so Matrix Market diagnostics and future suffix-sensitive
  code remain representative;
- deterministic behavior in test inputs and seeds, with uniqueness limited to
  avoiding concurrent job collisions;
- bounded path writes that cannot overflow `fuzz_tmp_path`;
- explicit failure diagnostics when temp-file creation fails;
- cleanup on normal suite exit;
- no stale file requirement when parser fuzz tests are skipped because the
  helper could not create a file.

## Cleanup And Residue Policy

Day 10 should keep cleanup conservative:

1. Initialize `fuzz_tmp_path` to an empty string before attempting creation.
2. Only run file-backed fuzz cases when `fuzz_tmp_path[0] != '\0'`.
3. After each `try_load_mm`, overwrite the same path rather than creating
   additional temp files.
4. Always call `fuzz_cleanup_tmp()` before suite exit.
5. Treat missing files during cleanup as non-fatal.
6. Avoid directory creation and avoid persistent fixture generation.

This preserves the current skip model while preventing nondeterministic residue
on Windows runners.

## Determinism And Timeout Policy

The port must not change:

- fixed property-test seed sequences;
- property pass thresholds;
- large CSC sizes tied to `SPARSE_CSC_THRESHOLD`;
- parser fuzz input strings;
- test registration names;
- local POSIX execution count or skip accounting.

The Windows lane should rely on the existing bounded property loops rather than
adding unbounded fuzzing or random iteration counts.

## CMake Registration Plan

After Day 10 removes the source-level Windows blocker, CMake should promote the
existing `test_fuzz` target:

```cmake
add_sparse_test(test_sprint8_integration)
add_sparse_test(test_fuzz)
add_sparse_test(test_lu_csr)
```

No new CTest name is needed when the full helper path succeeds.

Do not update `.github/workflows/windows-ci.yml` on Day 10 unless the sprint
intentionally moves the Day 11 CI promotion batch earlier. The preferred Day 11
owner should update `EXPECTED_WINDOWS_CTEST_COUNT` and staged wording after all
three promoted tests are in the tree.

## Expected Count Impact

| Surface | Count Impact |
| --- | --- |
| Local POSIX CTest | No expected count change; `test_fuzz` is already registered locally. |
| Windows CMake after Day 8 | Planned `58 -> 59` once `test_fuzz` is registered and listed by hosted Windows `ctest -N`. |
| Full Sprint 148 promotion plan | `56 -> 59` if `test_threads`, `test_sprint4_integration`, and `test_fuzz` all promote. |

The count remains a planning number until confirmed by `ctest --test-dir build
-C Release -N` on hosted Windows.

## Fallback Split Plan

If the full file-backed helper path fails on MSVC or hosted Windows, split the
surface instead of weakening parser coverage:

| Fallback Owner | Contents | Registration |
| --- | --- | --- |
| Existing `test_fuzz` | Keep the current POSIX file-backed parser fuzz cases and all property checks. | POSIX-only until the file-backed helper is fixed. |
| New `test_fuzz_portable` | Null-arg/nonexistent-path checks plus LU/Cholesky/QR/SVD and large CSC property checks. | Windows and POSIX CMake, no temp-file dependency. |

The fallback must explicitly record the file-backed malformed-input parser fuzz
cases as retained Windows-staged residuals. It should not claim full
file-backed fuzz parity.

## Validation Checklist

Day 10 should run and record:

| Check | Purpose |
| --- | --- |
| `cmake -S . -B build` | Confirm CMake accepts the promoted or split registration. |
| `cmake --build build --target test_fuzz` | Confirm focused fuzz build. |
| `ctest --test-dir build -R '^test_fuzz$' --output-on-failure` | Confirm focused execution on the local platform. |
| `ctest --test-dir build -N` | Confirm local registration and record expected Windows delta separately. |
| `make format && make lint && make test` | Required because Day 10 will modify `.c` and CMake files. |

## Support-Claim Boundaries

Allowed after hosted Windows proof:

- reviewed Windows CMake subset includes deterministic `test_fuzz` parser and
  property coverage;
- malformed Matrix Market file-backed parser cases are covered through the
  portable temp-file helper if the primary path lands;
- deterministic solver property coverage is included in reviewed Windows CTest.

Still not allowed:

- unbounded fuzzing or randomized fuzz campaign coverage;
- sanitizer parity;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- support for temp-file semantics outside this test helper.

## Rollback Rules

Roll back full promotion or use the fallback split when:

- MSVC cannot compile the helper without platform-specific source leakage into
  the parser/property logic;
- Windows execution leaves persistent temp-file residue or cannot clean up the
  helper-created path;
- POSIX parser fuzz inputs, property seeds, pass thresholds, or skip behavior
  change unexpectedly;
- CMake registration changes produce unexplained `ctest -N` count drift;
- workflow or documentation wording implies broader fuzzing, sanitizer, or
  Windows platform parity than the promoted test supports.

## Day 10 Handoff

Implement the in-file portable temp-file helper first and promote the existing
`test_fuzz` CMake target if focused local validation passes. Use the split
fallback only if the primary helper path cannot preserve parser fuzz behavior
and cleanup semantics.
