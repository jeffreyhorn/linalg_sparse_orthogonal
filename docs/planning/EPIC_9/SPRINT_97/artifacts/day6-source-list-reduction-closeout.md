# Sprint 97 Day 6: Source-List Reduction Closeout

## Purpose

Day 6 closes the source-list reduction started on Day 5. The closeout checks
whether any adjacent build, CMake, CI, or documentation surfaces contradict the
new validation ownership, and records the residual queue before Sprint 97
moves into package-surface decisions.

## Completed Reduction

The selected duplicated surface is the library implementation source list.
Sprint 97 now has three explicit reviewed surfaces:

- `build-metadata/library_sources.txt`
- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`

The Day 5 checker makes those surfaces cheaper to maintain by turning silent
drift into an explicit reviewed failure:

```sh
python3 scripts/check_library_sources.py
make source-list-check
```

The source list is still directly reviewable in Make and CMake. The reduction
is a parity guard, not hidden generation.

## Validation Ownership

Current ownership after Day 6:

| Surface | Owner | Enforcement |
| --- | --- | --- |
| Manifest membership | `build-metadata/library_sources.txt` | `source-list-check` |
| Make library list | `Makefile` `LIB_SRCS` | `source-list-check` and existing Make builds |
| CMake library list | `CMakeLists.txt` `add_library(...)` | `source-list-check` and existing CMake builds |
| Linux reviewed compile quality | `.github/workflows/ci.yml` | `make quality-review-compile` |
| macOS reviewed compile quality | `.github/workflows/macos-ci.yml` | `make quality-review-compile` |
| Windows CMake consumer subset | `.github/workflows/windows-ci.yml` | existing CMake configure/build/CTest count proof |

`source-list-check` remains wired into `quality-review-compile`. That keeps the
new guard on the reviewed Linux and macOS compile-quality path through existing
workflow commands without adding a new CMake configure-time Python dependency.

## Reconciled Surfaces

Reviewed for contradiction:

- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/check_library_sources.py`
- `build-metadata/library_sources.txt`
- Sprint 97 planning artifacts

No adjacent workflow label or package claim needed a Day 6 edit. The existing
workflow names still describe the same reviewed proof lanes, and the source
checker reaches CI through `make quality-review-compile`.

## Deferred Decisions

Day 6 deliberately defers:

- CMake configure-time execution of `scripts/check_library_sources.py`
- adding `source-list-check` to `quality-review-cmake-compile`
- replacing Make or CMake source lists with generated content
- centralizing test registration
- centralizing benchmark or example registration
- changing package/install/export claims
- changing Windows expected CTest count assertions

These remain residual build-topology items because each would expand proof
ownership beyond the selected Day 4/5 source-list batch.

## Validation

Day 6 re-ran the targeted proof and hygiene checks:

```sh
python3 scripts/check_library_sources.py
make source-list-check
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_97 build-metadata scripts/check_library_sources.py Makefile
```

Observed results:

- direct checker: `source-list-check: PASS (42 library sources)`
- Make target: `source-list-check: PASS (42 library sources)`
- `git diff --check`: passed
- trailing-whitespace scan: passed with no matches

No `.c` or `.h` files were modified during Day 6. The full
`make format && make lint && make test` chain was not required for this
documentation/reconciliation closeout.

## Day 6 Result

The selected source-list duplication is now lower-cost to maintain because
manifest, Make, and CMake drift is mechanically detected. Validation ownership
is explicit, CMake and CI surfaces do not contradict the new guard, and the
remaining topology work is documented as residual instead of hidden.
