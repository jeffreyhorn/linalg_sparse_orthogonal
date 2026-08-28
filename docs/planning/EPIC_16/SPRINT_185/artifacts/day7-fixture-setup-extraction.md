# Sprint 185 Day 7: Fixture And Setup Extraction

## Purpose

Record the second no-behavior-change mechanical extraction for the selected
Sprint 185 review-surface cluster.

## Selected Cluster

| Field | Value |
| --- | --- |
| Selected cluster | `tests/test_ldlt_csc.c` |
| Day 3 baseline size | 3915 lines |
| Day 6 size | 3793 lines |
| Existing proof-owner binary | `test_ldlt_csc` |
| Extraction type | Family-local test fixture header |

## Files Changed

| Path | Change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Included the new fixture header and removed moved KKT/setup helper definitions. |
| `tests/test_ldlt_csc_fixtures.h` | Added family-local KKT fixtures and analysis-backed two-pass setup helper. |
| `docs/planning/EPIC_16/SPRINT_185/WORKING_NOTES.md` | Recorded Day 7 extraction, validation, and handoff notes. |

No production source, public header, internal API, Makefile, CMake, or
library-source manifest files changed.

## Moved Helpers

| Helper | Role |
| --- | --- |
| `build_kkt_5x5` | Small KKT fixture used by with-analysis and external dense-reference tests. |
| `build_kkt_10x10` | Larger KKT fixture used by with-analysis, residual, external-reference, and min-size rejection tests. |
| `build_kkt_scaled_10x10` | Scaled KKT fixture used by the Sprint 102 external dense-reference lane. |
| `s20_two_pass_indefinite_factor` | Shared scalar-prepass plus analysis-backed supernodal factor setup helper. |

## Preserved Ownership

- `test_ldlt_csc` remains the only proof-owner binary for the selected cluster.
- No `RUN_TEST(...)` entries, test names, test bodies, fixture values,
  numerical tolerances, or `main` ordering changed.
- `_POSIX_C_SOURCE` and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` placement and
  behavior remain owned by `tests/test_ldlt_csc.c`.
- External dense-reference state allocation, process invocation, Windows skip
  handling, and assertion logic remain local in `tests/test_ldlt_csc.c`.
- No public or internal solver API changed.

## Registration Result

| Surface | Day 7 result |
| --- | --- |
| `Makefile` `TEST_SRCS` | No change. |
| `CMakeLists.txt` `add_sparse_test(...)` | No change. |
| `Makefile` `LIB_SRCS` | No change. |
| `CMakeLists.txt` library source list | No change. |
| `build-metadata/library_sources.txt` | No change. |

The extraction stayed header-only and continued to compile through the
existing `test_ldlt_csc` binary.

## Review-Surface Result

| Path | Lines after Day 7 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3639 | Reduced from 3793 after Day 6 and from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_fixtures.h` | 145 | New family-local KKT/two-pass fixture header. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Existing Day 6 family-local supernode helper header. |

## Validation

Because Day 7 modified `.c` and `.h` files, the full C quality gate was
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

## Day 8 Handoff

- Keep external dense-reference state and process helpers local unless a later
  cleanup proves that extraction is lower risk than locality.
- Prefer include/declaration cleanup and stale-comment review before moving
  another broad helper block.
- Continue preserving `test_ldlt_csc` as the proof-owner binary unless a later
  artifact justifies a new registered proof owner.
