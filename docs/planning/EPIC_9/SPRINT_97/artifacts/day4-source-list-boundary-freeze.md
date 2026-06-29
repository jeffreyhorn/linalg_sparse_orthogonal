# Sprint 97 Day 4: Source-List Reduction Boundary Freeze

## Purpose

Day 4 freezes the first build-topology reduction batch before implementation.
The selected batch is a library-source manifest and parity checker for the
duplicated Make/CMake library source list.

Day 4 is a planning and boundary-freeze day. No build-system, script, `.c`, or
`.h` files are changed by this artifact.

## Frozen Target

Primary duplicated surfaces:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`

New tracked metadata owner to create on Day 5:

- `build-metadata/library_sources.txt`

New checker to create on Day 5:

- `scripts/check_library_sources.py`

Build owner to update on Day 5:

- `Makefile`

CMake owner to inspect but not edit in the first batch:

- `CMakeLists.txt`

Explicitly deferred owners:

- `.github/workflows/*.yml`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `README.md`
- `INSTALL.md`
- `CMakeLists.txt` configure-time hook

## Exact Manifest Boundary

Create `build-metadata/library_sources.txt` as a plain text manifest of library
source files.

Manifest rules:

- one source path per non-empty line
- paths are repo-relative
- paths use forward slashes
- comments begin with `#`
- entries remain in the current reviewed library order
- only library implementation sources under `src/*.c` are in scope
- no test, benchmark, example, public-header, internal-header, generated, or
  workflow files are in scope

Initial manifest content should match the current 42 library sources from
`Makefile` and `CMakeLists.txt`, in the same order as `Makefile` `LIB_SRCS`.

The manifest should not become a generated file in Day 5. It is the durable
source-membership reference checked against the existing build files.

## Exact Checker Boundary

Create `scripts/check_library_sources.py`.

Checker contract:

1. Read `build-metadata/library_sources.txt`.
2. Parse `Makefile` `LIB_SRCS`.
3. Parse `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`.
4. Normalize paths to repo-relative `src/*.c`.
5. Compare manifest, Make, and CMake source membership and order.
6. Print actionable differences for:
   - missing manifest entries
   - extra manifest entries
   - Make-only entries
   - CMake-only entries
   - order mismatches
7. Exit `0` only when all three lists match.

Implementation constraints:

- use Python 3 standard library only
- avoid shelling out to Make or CMake
- keep parser scope deliberately narrow to the current source-list syntax
- fail clearly if an expected block cannot be found
- do not rewrite files

The checker may be strict about formatting. If the source-list syntax changes,
the checker should fail and force a conscious update rather than silently
accepting a new topology.

## Makefile Hook Boundary

Add a Make target:

```make
.PHONY: source-list-check
source-list-check:
	@python3 scripts/check_library_sources.py
```

Add the target to the reviewed compile-quality path:

```make
quality-review-compile: ... source-list-check ...
```

Recommended placement:

1. `format-check`
2. `source-list-check`
3. `lint`

Rationale:

- source-list drift should fail before expensive lint/static-analysis work
- the reviewed Makefile path is the strongest local place to enforce the
  cross-build source-list contract
- CMake membership remains directly checked by the script, so a separate CMake
  configure-time hook is not needed in the first batch

Do not alter `LIB_SRCS` order or contents on Day 5 except as needed to match
the existing manifest. The goal is parity enforcement, not source-list
reordering.

## Explicit Non-Goals

Day 5 should not include:

- generating Makefile fragments
- generating CMake fragments
- replacing `LIB_SRCS` with an include
- replacing `add_library(...)` source entries with a generated file
- changing test registration
- changing benchmark registration
- changing example registration
- changing install/export behavior
- changing package claims
- changing Windows expected CTest count
- changing platform workflow messages
- adding a shared-library lane
- editing `.c` or `.h` files

## Retained Proof Surfaces

These surfaces must remain explicit after the Day 5 batch:

- Make `LIB_SRCS`
- CMake `add_library(sparse_lu_ortho STATIC ...)`
- `make quality-review-cmake-compile` CTest-count parity assertion
- Windows `EXPECTED_WINDOWS_CTEST_COUNT`
- Windows staged exclusions:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- macOS reviewed Apple Clang path
- macOS supplemental Homebrew GCC path
- macOS supplemental Make install/pkg-config proof
- Linux reviewed Makefile compile-quality path
- Linux reviewed CMake parity path
- Linux dead-code report/check path
- Make install/pkg-config proof
- CMake install/find_package proof
- static-first package wording in CMake, README, and INSTALL

## Day 5 Implementation Sequence

1. Create `build-metadata/library_sources.txt`.
2. Populate it from the current `Makefile` `LIB_SRCS` order.
3. Create `scripts/check_library_sources.py`.
4. Run the checker directly and fix parser or manifest issues until it passes.
5. Add `source-list-check` to `Makefile`.
6. Hook `source-list-check` into `quality-review-compile` between
   `format-check` and `lint`.
7. Run the validation commands below.
8. Record the implementation in the Day 5 artifact and working notes.

## Validation Plan

Required Day 5 validation if only manifest, script, and Makefile are changed:

```sh
python3 scripts/check_library_sources.py
make source-list-check
make quality-review-compile
git diff --check
```

Additional validation if CMake is changed unexpectedly:

```sh
make quality-review-cmake-compile
```

Additional validation if any `.c` or `.h` file is changed:

```sh
make format && make lint && make test
```

Day 5 should not need `.c` or `.h` edits. If code or header edits appear
necessary, stop and reassess the boundary before continuing.

## Rollback Plan

If the checker proves too brittle during Day 5:

1. Remove `scripts/check_library_sources.py`.
2. Remove `build-metadata/library_sources.txt`.
3. Remove `source-list-check` and the `quality-review-compile` hook from
   `Makefile`.
4. Preserve this Day 4 artifact and record the failure as a Day 5 residual.
5. Fall back to a documentation-only source-list ownership note for Day 5 and
   defer implementation to Day 6 or Sprint 98.

If the checker passes directly but the Make hook causes unacceptable local
quality cost:

1. Keep the manifest and checker.
2. Remove only the `quality-review-compile` hook.
3. Keep `make source-list-check` as a manual target.
4. Queue hook placement for Day 6 after validation-cost review.

## Day 4 Result

The first source-list reduction batch is ready to implement. The batch is
bounded to a library-source manifest, a narrow parity checker, and a Make
reviewed-path hook. It deliberately does not change source membership,
package behavior, test registration, platform workflow assertions, or public
claims.
