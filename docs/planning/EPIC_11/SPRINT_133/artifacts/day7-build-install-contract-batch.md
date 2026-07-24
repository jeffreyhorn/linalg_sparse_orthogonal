# Sprint 133 Day 7 - Build and Install Contract Batch

## Purpose

Day 7 implements the first selected-contract build/install batch from the Day
5 product decision and Day 6 static contract design. The implementation keeps
the maintained package surface static-first and changes only the CMake response
to an explicit shared-library request.

## Implemented Change

| File | Change |
| --- | --- |
| `CMakeLists.txt` | Replaced warning-only `BUILD_SHARED_LIBS=ON` handling with a configure-time `FATAL_ERROR` that states shared-library packaging and dynamic ABI support are deferred. |

The static `add_library(sparse_lu_ortho STATIC ...)` target, install/export
rules, generated version metadata, `sparse.pc` generation, public headers,
install tests, package docs, and workflows were left unchanged.

## Selected Contract Behavior

| Behavior | Day 7 result |
| --- | --- |
| Default CMake configure/build/install | Preserved. |
| Static archive install | Preserved. |
| CMake package export | Preserved. |
| Installed CMake consumer | Preserved. |
| Exact package version behavior | Preserved. |
| No shared artifact install proof | Preserved. |
| `BUILD_SHARED_LIBS=ON` | Now fails configure with explicit static-first/shared-deferral wording. |
| Shared-library support | Still deferred and unclaimed. |
| Dynamic ABI compatibility | Still deferred and unclaimed. |

## Focused Failure Probe

Command shape:

```sh
cmake -S . -B "$tmp_dir/build-shared-request" -DBUILD_SHARED_LIBS=ON
```

Result: expected configure failure.

Required wording was observed:

- `BUILD_SHARED_LIBS=ON was requested`
- `static archive package surface`
- `Shared-library packaging`
- `dynamic ABI support are deferred`

The first harness attempt confirmed the fatal error but checked for one
line-wrapped phrase as a contiguous string. The rerun used line-wrapping-safe
checks and passed.

## Static Install/Export Validation

Command:

```sh
bash tests/test_cmake_install.sh
```

Result:

| Check group | Result |
| --- | --- |
| CMake configure | Pass |
| CMake build | Pass |
| CMake install | Pass |
| Static library installed | Pass |
| No shared-library artifacts installed | Pass |
| Headers installed | Pass, 19 files |
| `SparseConfig.cmake` installed | Pass |
| `SparseConfigVersion.cmake` installed | Pass |
| `SparseTargets.cmake` installed | Pass |
| `sparse.pc` installed | Pass |
| `cmake_example` configure/build/run | Pass |
| Exact-version package check | Pass |
| Mismatched-version rejection | Pass |
| `pkg-config` version | Pass, `2.2.0` |

Summary: 16 passed, 0 failed, 0 skipped.

## Static Compatibility Notes

- Existing static consumers are not intentionally changed.
- Existing CMake installed consumers continue to link
  `Sparse::sparse_lu_ortho`.
- Existing no-shared-artifact install checks remain the static-first proof.
- No `SPARSE_API`, export map, soname/install-name, ABI epoch, or
  static/shared package selector was added.
- Package-manager support remains absent and unclaimed.

## Changed-Surface Validation Matrix

| Touched surface | Validation run | Result |
| --- | --- | --- |
| `CMakeLists.txt` shared-request branch | Focused `BUILD_SHARED_LIBS=ON` configure failure probe | Pass |
| `CMakeLists.txt` default static install/export path | `bash tests/test_cmake_install.sh` | Pass |
| Sprint 133 documentation artifacts | `git diff --check` and Sprint 133 whitespace scan | Pending final Day 7 hygiene check |

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 7.

## Implementation Residual Queue

| Residual | Owner |
| --- | --- |
| Exact CMake installed header-count assertion | Day 11 or Day 13 |
| Installed CMake target path-origin checks | Day 11 |
| Exact `pkg-config` include/lib path checks | Day 12 |
| Static-first documentation alignment after implementation | Day 8 |
| ABI/symbol/static-deferral proof design | Day 9 |
| Integrated package validation | Day 13 |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected package contract is represented in build/install behavior. | Complete | `BUILD_SHARED_LIBS=ON` now fails configure with explicit static-first/shared-deferral wording. |
| Existing static consumers are not broken without an explicit decision. | Complete | `bash tests/test_cmake_install.sh` passed with 16 checks and 19 installed headers. |
| Remaining implementation work is narrow and documented. | Complete | Residual queue assigns CMake, pkg-config, docs, deferral proof, and integrated validation follow-up days. |
