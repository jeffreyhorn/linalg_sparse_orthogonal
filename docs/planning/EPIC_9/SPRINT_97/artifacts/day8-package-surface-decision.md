# Sprint 97 Day 8: Package-Surface Decision

## Purpose

Day 8 lands the package decision framed by the Day 7 audit. The decision is to
preserve the maintained static-first package contract and explicitly defer
shared-library packaging until it has real build, metadata, consumer, and
platform proof.

## Decision

Sprint 97 preserves the static-first package contract.

Maintained package surface:

- Unix-like Make install of `libsparse_lu_ortho.a`
- CMake install/export of `Sparse::sparse_lu_ortho` as a static target
- public headers under the installed `sparse/` include directory
- generated `sparse_version.h`
- `pkg-config` metadata in `sparse.pc`
- exact-version CMake package metadata
- local install/export validation through `tests/test_install.sh` and
  `tests/test_cmake_install.sh`

Deferred package surface:

- shared-library build output
- shared-library install/export metadata
- dynamic ABI/version compatibility claims
- platform-specific runtime-loader handling
- Windows DLL/import-library packaging
- reviewed shared-library consumer tests
- full reviewed macOS or Windows install/export parity

## Updated Surfaces

### README

`README.md` now makes the front-door install summary explicit: downstream
consumer guidance is for the maintained static archive surface, and
shared-library packaging is deferred to a future proof-bearing change.

### INSTALL

`INSTALL.md` now records the deferral criteria for shared-library packaging:
build rules, package metadata, installed-consumer proof, and platform-specific
runtime-loader coverage.

The stale `quality-review-compile` summary was also refreshed from:

```text
format-check + lint
```

to:

```text
format-check + source-list-check + lint
```

That aligns the installation guide with the Day 5 source-list guard.

### CMake

`CMakeLists.txt` already enforced the static-first package surface. Day 8 only
removed sprint-history wording from the explanatory comment while preserving
the existing behavior:

- `BUILD_SHARED_LIBS=ON` remains accepted as a configure input
- CMake still prints the static-first status message
- `sparse_lu_ortho` remains declared as `STATIC`

No build rule, package export, or install destination was changed.

## Proof Ownership

| Proof surface | Ownership after Day 8 |
| --- | --- |
| Make static install and `pkg-config` consumer | `tests/test_install.sh` |
| CMake static install/export and `find_package` consumer | `tests/test_cmake_install.sh` |
| CMake static-first configure behavior | `CMakeLists.txt` status message and `STATIC` target declaration |
| Linux reviewed confidence | existing strongest CI proof lanes |
| macOS install confidence | supplemental static-first Make install/`pkg-config` workflow |
| Windows consumer confidence | reviewed CMake-first subset only |

## Focused Validation

Day 8 validation proved that the documented static-first decision still
matches live package behavior:

```sh
cmake -S . -B build/sprint97-day8-shared-probe \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/sparse-sprint97-day8
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/check_library_sources.py
git diff --check
rg -n "[ \t]+$" README.md INSTALL.md CMakeLists.txt docs/planning/EPIC_9/SPRINT_97
```

Observed results:

- CMake configure with `BUILD_SHARED_LIBS=ON` passed and printed the
  static-first status message.
- `tests/test_install.sh` passed: 13 passed, 0 failed.
- `tests/test_cmake_install.sh` passed: 15 passed, 0 failed, 0 skipped.
- `scripts/check_library_sources.py` passed:
  `source-list-check: PASS (42 library sources)`.
- `git diff --check` passed.
- trailing-whitespace scan passed with no matches.

## Day 8 Result

The repo now has one explicit package contract for Sprint 97: static-first is
the maintained package surface, and shared-library packaging is deferred until
future work adds the proof required to make it real.
