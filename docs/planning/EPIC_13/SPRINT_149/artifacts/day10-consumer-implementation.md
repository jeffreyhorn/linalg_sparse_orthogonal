# Sprint 149 Day 10: Downstream Consumer Implementation

## Purpose

Implement maintained Windows downstream consumer checks for normal and
versioned installed CMake package usage, while preserving the no-Windows-
`pkg-config` boundary.

## Files Changed

| File | Change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added generated non-versioned basic CMake consumer configure/build/run proof. |
| `docs/planning/EPIC_13/SPRINT_149/WORKING_NOTES.md` | Recorded Day 10 implementation and Day 11 handoff. |
| `docs/planning/EPIC_13/SPRINT_149/artifacts/day10-consumer-implementation.md` | Published this implementation artifact. |

No `.c` or `.h` repository files were changed. The new C source is generated
inside the hosted workflow under `$env:RUNNER_TEMP`.

## Added Basic Consumer Proof

Day 10 adds a generated downstream CMake project with:

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_basic_consumer C)
find_package(Sparse REQUIRED)
add_executable(basic_consumer main.c)
target_link_libraries(basic_consumer PRIVATE Sparse::sparse_lu_ortho)
```

The generated `main.c`:

- includes `sparse/sparse_types.h` and `sparse/sparse_matrix.h`;
- prints `sparse version:`;
- creates a 3x3 sparse matrix;
- inserts one entry;
- prints `nnz: 1`;
- frees the matrix;
- prints `OK`.

## Workflow Variables

| Variable | Purpose |
| --- | --- |
| `$basicConsumerSrc` | Temporary source directory under `$env:RUNNER_TEMP/sparse-basic-consumer-src`. |
| `$basicConsumerBuild` | Separate build directory `build-basic-consumer`. |
| `$basicConsumerExe` | Built executable path `build-basic-consumer/Release/basic_consumer.exe`. |
| `$basicConsumerText` | Multiline-normalized run output. |

## Consumer Checks

| Check | Implementation |
| --- | --- |
| Configure | `cmake -S $basicConsumerSrc -B $basicConsumerBuild -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$prefix"` |
| Build | `cmake --build $basicConsumerBuild --config Release` |
| Run | execute `basic_consumer.exe` and fail on nonzero `$LASTEXITCODE` |
| Output | require `sparse version:`, `nnz: 1`, and `OK` |

The basic generated consumer is separate from:

- the maintained `examples/cmake_example` consumer;
- the exact-version generated consumer;
- the mismatch-version fail-closed configure proof.

## Preserved Proof

Day 10 preserves the existing Windows installed CMake package proof:

- package metadata checks happen before consumer checks;
- maintained example still configures/builds/runs and checks
  `Sparse library version`, `Solution:`, and `OK`;
- exact-version generated consumer still configures/builds/runs through
  `find_package(Sparse $version EXACT REQUIRED)`;
- mismatch-version generated project still must fail configure;
- `$global:LASTEXITCODE = 0` is still reset after the expected mismatch
  configure failure.

## Boundary Preserved

The new generated consumer uses CMake `find_package(Sparse REQUIRED)`.
It does not invoke:

- Windows `pkg-config`;
- Makefile install/uninstall;
- package-manager resolution;
- shared-library loading.

The reviewed Windows package claim remains CMake install/downstream scoped.

## Hosted Evidence Requirement

The new consumer is reviewed only after the hosted Windows job passes:

`Windows reviewed CMake install/downstream validation path`

If hosted Windows fails in the generated basic consumer block, treat the
failure as a consumer-proof issue, not as a package metadata issue, because the
workflow now checks metadata before generated consumers run.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Downstream proof exercises installed package files rather than in-tree targets. | Complete | The generated project uses `find_package(Sparse REQUIRED)` with `CMAKE_PREFIX_PATH="$prefix"`. |
| Exact-version consumer builds and runs through the installed package. | Complete | Existing exact-version generated consumer remains preserved. |
| Mismatch-version consumer fails configure as expected. | Complete | Existing mismatch-version fail-closed proof remains preserved with `$LASTEXITCODE` reset. |
