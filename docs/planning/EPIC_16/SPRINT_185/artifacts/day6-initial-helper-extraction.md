# Sprint 185 Day 6: Initial Helper Extraction

## Purpose

Record the first no-behavior-change mechanical helper extraction for the
selected Sprint 185 review-surface cluster.

## Selected Cluster

| Field | Value |
| --- | --- |
| Selected cluster | `tests/test_ldlt_csc.c` |
| Day 3 baseline size | 3915 lines |
| Existing proof-owner binary | `test_ldlt_csc` |
| Extraction type | Family-local test helper header |

## Files Changed

| Path | Change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Included the new helper header and removed moved helper definitions. |
| `tests/test_ldlt_csc_supernode_helpers.h` | Added family-local LDLT CSC supernode helper definitions. |
| `docs/planning/EPIC_16/SPRINT_185/WORKING_NOTES.md` | Recorded Day 6 extraction, validation, and handoff notes. |

No production source, public header, internal API, Makefile, CMake, or
library-source manifest files changed.

## Moved Helpers

| Helper | Role |
| --- | --- |
| `build_dense_ldlt_with_pivots` | Dense lower-triangular `LdltCsc` fixture builder for supernode detection tests. |
| `cm_idx` | Column-major indexing helper for dense panel buffers. |
| `snapshot_supernode_state` | Pre/post writeback state snapshot helper for supernode round-trip checks. |
| `ldlt_csc_factor_state_matches` | Exact factor-state comparison for scalar/supernodal cross-checks. |
| `build_dense_spd` | Dense SPD fixture builder for supernodal scalar/batched parity checks. |

## Preserved Ownership

- `test_ldlt_csc` remains the only proof-owner binary for the selected cluster.
- No `RUN_TEST(...)` entries, test names, test bodies, fixture values,
  numerical tolerances, or `main` ordering changed.
- `_POSIX_C_SOURCE` and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` placement and
  behavior remain owned by `tests/test_ldlt_csc.c`.
- The helper header is included in the local helper block; `clang-format`
  normalizes that block alphabetically, and the focused/full gates confirm the
  helper remains dependency-clean in that formatted order.
- No public or internal solver API changed.

## Registration Result

| Surface | Day 6 result |
| --- | --- |
| `Makefile` `TEST_SRCS` | No change. |
| `CMakeLists.txt` `add_sparse_test(...)` | No change. |
| `Makefile` `LIB_SRCS` | No change. |
| `CMakeLists.txt` library source list | No change. |
| `build-metadata/library_sources.txt` | No change. |

The extraction stayed header-only and continued to compile through the
existing `test_ldlt_csc` binary.

## Review-Surface Result

| Path | Lines after Day 6 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3793 | Reduced from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | New family-local helper header. |

## Validation

Because Day 6 modified `.c` and `.h` files, the full C quality gate was
required and completed.

```sh
make format
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make lint
make test
```

Focused validation:

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed after forcing the stale binary out of `build/`. |
| `./build/test_ldlt_csc` | Passed: 100 tests, 0 failures, 0 skips, 3556 assertions. |

Full gate:

| Command | Result |
| --- | --- |
| `make format` | Passed. |
| `make lint` | Passed. |
| `make test` | Passed. |

## Day 7 Handoff

- Consider `tests/test_ldlt_csc_fixtures.h` as the next helper boundary.
- Prefer KKT, scaled-KKT, or two-pass fixture movement only if macro and
  external-reference sensitivity remains contained.
- Continue preserving `test_ldlt_csc` as the proof-owner binary unless a later
  artifact justifies a new registered proof owner.
- Keep registration unchanged unless a new test binary or production source
  file becomes necessary.
