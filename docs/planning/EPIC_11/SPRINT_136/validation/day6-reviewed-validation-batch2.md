# Sprint 136 Day 6 - Reviewed Validation Batch 2 Summary

## Scope

Day 6 ran the second reviewed validation batch from the Day 4 command plan:
C/header gate decision, local CMake configure/build, local CTest
registration/execution, and CMake install/export package proof.

The current branch state remains documentation-only. The CMake build trees
created for validation were removed after command capture.

## Command Results

| Command | Status | Interpretation |
| --- | --- | --- |
| `git diff --name-only -- '*.c' '*.h' && git ls-files --others --exclude-standard -- '*.c' '*.h'` | Passed; no output | No tracked or untracked C/header changes; full C quality gate is not required. |
| `cmake -S . -B build-sprint136-cmake` | Passed | Local CMake configure succeeds with AppleClang. |
| `cmake --build build-sprint136-cmake` | Passed | Local CMake build succeeds for library, tests, benchmarks, and examples. |
| `ctest --test-dir build-sprint136-cmake -N` | Passed | Local CTest registration contains 57 tests. |
| `ctest --test-dir build-sprint136-cmake --output-on-failure` | Passed | 57/57 tests passed, 0 failed, total real time 740.85 seconds. |
| `bash tests/test_cmake_install.sh` | Passed | CMake install/export package proof passed 21 checks, 0 failures, 0 skips. |

## CTest Registration Reconciliation

Local CTest registration reported 57 tests:

- this matches the non-Windows local count recorded in Sprint 134 platform-tier
  closeout context;
- Windows reviewed CTest count remains 54 after staged pthread/POSIX-backed
  exclusions;
- Day 6 local CTest evidence is local-platform evidence and does not promote
  Windows staged tests or macOS/Windows install/downstream confidence to
  reviewed parity.

## Full CTest Execution

Full CTest passed:

```text
100% tests passed, 0 tests failed out of 57
Total Test time (real) = 740.85 sec
```

The slowest observed test was `test_reorder_nd`, which passed in 383.68
seconds. This is useful local confidence but not a portable runtime claim.

## CMake Install/Export Proof

`bash tests/test_cmake_install.sh` passed:

- CMake configure/build/install;
- installed static library and 19 headers;
- installed `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
  `SparseTargets.cmake`, and `sparse.pc`;
- static imported target metadata;
- installed-prefix include/archive checks;
- no source-tree or build-tree paths in installed package metadata;
- downstream `examples/cmake_example` configure/build/run with
  `find_package(Sparse)`;
- exact installed version accepted;
- mismatched version rejected;
- `pkg-config` version `2.2.0`;
- 21 checks passed, 0 failed, 0 skipped.

This proof supports the local static CMake installed-consumer story. It does
not create shared-library, dynamic ABI, runtime-loader, package-manager, or
cross-platform install parity claims.

## Day 6 Result

Reviewed validation batch 2 passed.

No failures or stop conditions were encountered.
