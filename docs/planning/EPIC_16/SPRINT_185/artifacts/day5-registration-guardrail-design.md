# Sprint 185 Day 5: Registration Guardrail Design

## Purpose

Plan source-list, test-registration, and validation guardrails for the selected
`tests/test_ldlt_csc.c` helper extraction before code movement begins.

## Planned Day 6 Extraction

| Field | Decision |
| --- | --- |
| New file | `tests/test_ldlt_csc_supernode_helpers.h` |
| Extraction type | Header-only family-local test helper |
| Including file | `tests/test_ldlt_csc.c` |
| Existing proof-owner binary | `test_ldlt_csc` |
| New proof-owner binary | None planned |
| Production source changes | None planned |

## Registration Decision

No build-registration metadata should change for the first extraction pass.

| Surface | Decision | Reason |
| --- | --- | --- |
| `Makefile` `TEST_SRCS` | No change | No new test `.c` file or binary is created. |
| `CMakeLists.txt` `add_sparse_test(...)` | No change | `test_ldlt_csc` remains the registered proof owner. |
| `Makefile` `LIB_SRCS` | No change | No library `.c` source is extracted. |
| `CMakeLists.txt` `add_library(...)` | No change | No library source is added. |
| `build-metadata/library_sources.txt` | No change | The library source manifest tracks production `.c` files, not test helper headers. |
| `scripts/check_library_sources.py` | No change | Existing guard remains sufficient because library-source membership/order is unchanged. |

## Existing Guard Coverage

| Guard | Coverage for Day 6 |
| --- | --- |
| `make format` / `make format-check` | Covers `tests/*.h` through `ALL_TEST_SRC = $(wildcard $(TESTDIR)/*.c) $(wildcard $(TESTDIR)/*.h)`. |
| `make lint` | Runs strict source lint for library sources and `cppcheck` over `src` and `tests`; the focused test compile covers the included helper header. |
| `make build/test_ldlt_csc && ./build/test_ldlt_csc` | Proves the existing proof-owner binary still compiles and runs after the header is included. |
| `make test` | Runs the full Makefile test suite after the helper extraction. |
| `make quality-review-cmake-compile` | Not required for the planned header-only extraction, because no new CMake test binary is added. Use it only if Day 6 unexpectedly changes CMake registration. |
| `make source-list-check` | Not required for the planned header-only extraction, because no library source is added. Use it only if Day 6 unexpectedly changes library source registration. |

## Build-Dependency Caveat

The Makefile's generic test rule builds `build/<test>` from `tests/<test>.c`
and the library. It does not list included test helper headers as explicit
prerequisites. That means a helper-header-only edit can leave an existing
`build/test_ldlt_csc` binary looking up to date unless the focused validation
forces a rebuild.

Use this Day 6 focused command:

```sh
rm -f build/test_ldlt_csc
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Use a clean build state before the full C gate:

```sh
make clean
make format && make lint && make test
```

This keeps the required `make format && make lint && make test` gate intact
while ensuring the header-including test binary is actually rebuilt.

## Expected Build Artifacts

| Artifact | Expected source | Staging rule |
| --- | --- | --- |
| `build/test_ldlt_csc` | Focused test rebuild | Generated; do not stage. |
| `build/*.o` | Makefile compile products | Generated; do not stage. |
| `build/libsparse_lu_ortho.a` | Static library build | Generated; do not stage. |
| `build/include/sparse_version.h` | Generated version header | Generated; do not stage. |
| `build/quality-review-cmake/` | Only if CMake parity is unexpectedly needed | Generated; do not stage. |

## Rollback Criteria

- If moving helpers requires a new registered test binary, stop and update the
  Day 5 registration design before code movement continues.
- If moving helpers requires a production source split, stop and update
  Makefile, CMake, and `build-metadata/library_sources.txt` together only
  after recording a revised design.
- If the forced focused rebuild fails, stop and inspect the include/dependency
  boundary before broadening the extraction.
- If the helper header creates circular includes or macro-ordering sensitivity,
  revert the helper-boundary design in planning before continuing.
- Keep generated build, CMake, report, and coverage outputs unstaged.

## Day 6 Handoff

Day 6 can proceed with a single header-only extraction into
`tests/test_ldlt_csc_supernode_helpers.h`. The implementation should avoid
Makefile/CMake/source-list edits unless the extraction design changes and is
recorded first.

## Validation

Day 5 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.
