# Sprint 149 Day 9: Downstream Consumer Proof Design

## Purpose

Design maintained Windows downstream consumer checks for normal installed
CMake usage, exact-version package behavior, and mismatch-version fail-closed
behavior.

The design keeps consumer configure/build/run proof separate from package-file
and metadata inspection. Package metadata failures should happen before
consumer failures; consumer failures should identify whether configure, build,
run, output, exact-version, or mismatch-version behavior broke.

## Sources Reviewed

| Source | Consumer-Proof Role |
| --- | --- |
| `examples/cmake_example/CMakeLists.txt` | Maintained normal installed consumer that uses `find_package(Sparse REQUIRED)` and links `Sparse::sparse_lu_ortho`. |
| `examples/cmake_example/main.c` | Maintained runtime proof source that prints version text, solves a small system, prints `Solution:`, and ends with `OK`. |
| `.github/workflows/windows-ci.yml` | Hosted Windows normal, exact-version, and mismatch-version consumer proof owner. |
| `tests/test_cmake_install.sh` | Unix local comparison for installed CMake consumer and version behavior. |

## Current Consumer Proof Inventory

| Consumer | Source Layout | Configure | Build | Run | Expected Output |
| --- | --- | --- | --- | --- | --- |
| Maintained example | `examples/cmake_example` in repo | `cmake -S examples/cmake_example -B $exampleBuild -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$prefix"` | `cmake --build $exampleBuild --config Release` | `$exampleBuild/Release/example.exe` | `Sparse library version`, `Solution:`, `OK` |
| Exact-version generated consumer | `$env:RUNNER_TEMP/sparse-version-exact-src` | generated `find_package(Sparse $version EXACT REQUIRED)` project with `CMAKE_PREFIX_PATH="$prefix"` | `cmake --build $versionExactBuild --config Release` | `$versionExactBuild/Release/version_exact.exe` | `Sparse library version`, `Solution:`, `OK` |
| Mismatch-version generated consumer | `$env:RUNNER_TEMP/sparse-version-mismatch-src` | generated `find_package(Sparse $mismatchVersion REQUIRED)` project with `CMAKE_PREFIX_PATH="$prefix"` | not expected | not expected | configure must fail |

## Normal Consumer Proof Design

The maintained example remains the normal downstream consumer proof because it
uses the public installed CMake package exactly like downstream callers:

```cmake
find_package(Sparse REQUIRED)
add_executable(example main.c)
target_link_libraries(example PRIVATE Sparse::sparse_lu_ortho)
```

The Windows workflow should keep these checks:

1. configure with `-DCMAKE_PREFIX_PATH="$prefix"`;
2. build with `--config Release`;
3. run `Release/example.exe`;
4. fail if `$LASTEXITCODE` is nonzero;
5. join PowerShell array output into `$outputText`;
6. require `Sparse library version`, `Solution:`, and `OK`.

## Exact-Version Consumer Proof Design

The exact-version consumer should remain generated outside the repository so
it proves an installed downstream project can request the package version
published by the install tree:

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_version_exact C)
find_package(Sparse $version EXACT REQUIRED)
add_executable(version_exact "<repo>/examples/cmake_example/main.c")
target_link_libraries(version_exact PRIVATE Sparse::sparse_lu_ortho)
```

Required behavior:

- configure succeeds with `CMAKE_PREFIX_PATH="$prefix"`;
- build succeeds in Release;
- `version_exact.exe` exits with code 0;
- output contains `Sparse library version`, `Solution:`, and `OK`.

## Mismatch-Version Fail-Closed Design

The mismatch-version consumer should request a lower same-major version:

- if `$minor > 0`, use `$major.($minor - 1).0`;
- else if `$patch > 0`, use `$major.$minor.($patch - 1)`;
- else fail the workflow because no lower same-major mismatch can be
  constructed from the current version.

The generated project:

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_version_mismatch C)
find_package(Sparse $mismatchVersion REQUIRED)
```

Required behavior:

- configure must fail;
- if configure succeeds, the workflow throws
  `Mismatched package version $mismatchVersion unexpectedly configured.`;
- reset `$global:LASTEXITCODE = 0` after the expected failure so the step
  reports success when fail-closed behavior is correct.

## Output Matching Rules

PowerShell command output can be a scalar string or an array of lines. Consumer
run checks should always normalize to text:

```powershell
$output = & $exampleExe 2>&1
$outputText = $output -join "`n"
```

Then check the required output tokens with `-notmatch` conditions. This avoids
the prior failure mode where multiline output contained `OK` but a scalar-only
match interpreted the output incorrectly.

## Source And Build Directory Layout

| Variable | Location | Purpose |
| --- | --- | --- |
| `$exampleBuild` | repository-relative `build-installed-example` | Build directory for maintained example. |
| `$versionExactSrc` | `$env:RUNNER_TEMP/sparse-version-exact-src` | Generated exact-version downstream source tree. |
| `$versionExactBuild` | repository-relative `build-version-exact` | Build directory for exact-version consumer. |
| `$versionMismatchSrc` | `$env:RUNNER_TEMP/sparse-version-mismatch-src` | Generated mismatch-version downstream source tree. |
| `$versionMismatchBuild` | repository-relative `build-version-mismatch` | Build directory for mismatch-version configure proof. |

Generated source trees should stay under `$env:RUNNER_TEMP` so they do not
dirty the repository. Build directories can remain repository-relative because
CI workspaces are disposable and the workflow already uses that pattern.

## Optional Day 10 Enhancement

Day 10 may add a generated non-versioned basic CMake consumer to mirror the
Unix generated `pkg-config` consumer without invoking Windows `pkg-config`.

The generated consumer should:

- live under `$env:RUNNER_TEMP/sparse-basic-consumer-src`;
- call `find_package(Sparse REQUIRED)`;
- include `sparse/sparse_types.h` and `sparse/sparse_matrix.h`;
- allocate a tiny sparse matrix;
- print version text, `nnz: 1`, and `OK`;
- configure/build/run through `CMAKE_PREFIX_PATH="$prefix"`;
- stay separate from exact-version proof.

This enhancement is optional because the current exact-version generated
consumer already exercises an installed generated project. It is useful if Day
10 wants a closer analog to the Unix generated basic consumer while preserving
Windows `pkg-config` execution as a non-claim.

## Failure Semantics

| Failure | Interpretation |
| --- | --- |
| Normal example configure fails | Installed `find_package(Sparse REQUIRED)` consumer path is broken. |
| Normal example build fails | Installed target or include/link metadata is broken after configure. |
| Normal example run fails | Installed library consumer runtime path is broken. |
| Normal example output lacks required tokens | Consumer ran but output contract changed or multiline matching broke. |
| Exact-version configure fails | Installed `SparseConfigVersion.cmake` exact-version behavior is broken. |
| Exact-version build/run/output fails | Exact-version downstream consumer path is broken. |
| Mismatch-version configure succeeds | Version fail-closed behavior is broken. |
| Mismatch-version cannot be constructed | Current versioning does not support this test's lower same-major strategy; stop and decide a new mismatch rule. |

## Day 10 Handoff

Day 10 should implement the optional generated basic CMake consumer if it can
be added without disrupting the current maintained example and version checks.
It should preserve:

- normal maintained example configure/build/run;
- exact-version configure/build/run;
- mismatch-version configure failure;
- `$LASTEXITCODE` handling after expected mismatch failure;
- output normalization through `-join "`n"`;
- no Windows `pkg-config` execution.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Consumer proof covers configure, build, run, and expected output. | Complete | Normal and exact-version designs require configure/build/run/output checks. |
| Version mismatch semantics are explicit and fail closed. | Complete | Mismatch-version design defines lower same-major construction and requires configure failure. |
| Multiline output and `$LASTEXITCODE` handling are specified. | Complete | Output matching and mismatch sections define `-join "`n"` and `$global:LASTEXITCODE = 0`. |
