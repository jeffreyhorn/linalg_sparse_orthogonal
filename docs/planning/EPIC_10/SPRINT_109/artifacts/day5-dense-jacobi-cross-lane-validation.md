# Day 5 Dense Jacobi Cross-Lane Validation

## Purpose

Day 5 proves the Day 4 dense Jacobi extraction across the focused eigensolver
lanes and build-system registration surfaces. The goal is to close the dense
Jacobi workstream before Sprint 109 moves on to broader behavior-sensitive
eigensolver audits.

Day 5 does not move more code.

## Focused Make Validation

Command:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration && \
./build/test_eigs && \
./build/test_eigs_thick_restart && \
./build/test_eigs_lobpcg && \
./build/test_sprint29_integration
```

Results:

| Test | Result | Count |
|---|---|---:|
| `test_eigs` | Passed | 31 |
| `test_eigs_thick_restart` | Passed | 21 |
| `test_eigs_lobpcg` | Passed | 27 |
| `test_sprint29_integration` | Passed | 3 |

The focused Make lanes cover the public eigensolver workflow, thick-restart
Rayleigh-Ritz caller, LOBPCG Rayleigh-Ritz caller, and cross-feature
eigensolver/refinement/progress workflow.

## Source-List Parity Evidence

Command:

```sh
make source-list-check
```

Result:

```text
source-list-check: PASS (46 library sources)
```

The eigensolver source order is identical across `Makefile`,
`CMakeLists.txt`, and `build-metadata/library_sources.txt`:

```text
src/sparse_eigs_workspace_internal.c
src/sparse_eigs_dense_internal.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_thick_restart.c
src/sparse_eigs.c
```

## CMake Compile-Command Evidence

Configured a local ignored CMake inspection tree:

```sh
cmake -S . -B build/day5-cmake-ctest \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Generated `build/day5-cmake-ctest/compile_commands.json` includes
`src/sparse_eigs_dense_internal.c` in the library compile command set between
`src/sparse_eigs_workspace_internal.c` and the backend owners
`src/sparse_eigs_lobpcg.c` and `src/sparse_eigs_thick_restart.c`.

## CMake/CTest Focused Validation

Commands:

```sh
cmake --build build/day5-cmake-ctest \
  --target test_eigs test_eigs_thick_restart test_eigs_lobpcg test_sprint29_integration -j2

ctest --test-dir build/day5-cmake-ctest \
  -R "^(test_eigs|test_eigs_thick_restart|test_eigs_lobpcg|test_sprint29_integration)$" \
  --output-on-failure
```

Results:

| CTest | Result |
|---|---|
| `test_sprint29_integration` | Passed |
| `test_eigs` | Passed |
| `test_eigs_thick_restart` | Passed |
| `test_eigs_lobpcg` | Passed |

CTest summary:

```text
100% tests passed, 0 tests failed out of 4
```

## CTest Registration Evidence

Local CMake registration inspection reports:

```text
Total Tests: 54
```

The focused eigensolver lanes remain registered in the same reviewed cluster:

```text
Test #46: test_sprint29_integration
Test #47: test_eigs
Test #48: test_eigs_thick_restart
Test #49: test_eigs_lobpcg
```

No test target was added, removed, renamed, or split for the dense Jacobi
extraction.

## Residual Risk

Dense Jacobi extraction risk is closed for Sprint 109 because both direct
runtime callers and both build surfaces pass focused validation.

Remaining eigensolver source-boundary risk is outside the moved helper:

- grow-m, refinement, and shared-kernel behavior remain Day 6 audit scope;
- dispatch/defaults, handle/workspace ownership, and shift-invert behavior
  remain Day 7 audit scope.

Those areas should not move in Sprint 109 without separate evidence and
validation.

## Completion Criteria Status

- Focused Make eigensolver validation passed.
- Focused CMake/CTest eigensolver validation passed.
- Source-list parity passed with 46 library sources.
- CTest registration reports 54 tests with the focused eigensolver lanes
  registered as tests 46-49.
- No reviewed test-count, target, public-header, install-header, helper-target,
  or source-list drift was introduced by Day 5.
