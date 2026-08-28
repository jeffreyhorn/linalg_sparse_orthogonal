# Sprint 185 Day 8: Proof-Owner Cleanup

## Purpose

Record the final mechanical helper extraction and call-site cleanup for the
selected `tests/test_ldlt_csc.c` cluster.

## Selected Cluster

| Field | Value |
| --- | --- |
| Selected cluster | `tests/test_ldlt_csc.c` |
| Day 3 baseline size | 3915 lines |
| Day 7 size | 3639 lines |
| Proof owner | `test_ldlt_csc` |
| Extraction type | Family-local oracle helper header |

## Files Changed

| File | Day 8 change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Included the new oracle helper header and removed the moved oracle/native-wrapper helper definitions. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Added dense-oracle and native-wrapper comparison helper definitions. |
| `docs/planning/EPIC_16/SPRINT_185/WORKING_NOTES.md` | Recorded Day 8 extraction, validation, and Day 9 handoff. |

No production source, public header, internal API, Makefile, CMake, or library
manifest changed.

## Moved Helpers

| Helper | Role |
| --- | --- |
| `ldlt_lower_to_dense` | Lower-triangle dense copy oracle. |
| `dense_sym_swap` | Dense symmetric permutation oracle. |
| `dense_lower_equal` | Lower-triangle dense comparison. |
| `build_ldlt_from_triples` | Sparse-to-LDLT fixture builder for symmetric-swap tests. |
| `ldlt_column_nonzeros_match` | Zero-tolerant column comparison. |
| `ldlt_factorizations_match` | Native-wrapper factor comparison. |
| `check_native_matches_wrapper` | Wrapper/native factor parity assertion helper. |

## Preserved Ownership

- `test_ldlt_csc` remains the only proof-owner binary.
- Test bodies, `main`, `RUN_TEST(...)` ordering, test names, numerical
  tolerances, and fixture values are unchanged.
- Process-global native/wrapper override calls remain inside the moved helper
  and still reset to the default path on the successful path as before.
- No production API, internal solver API, source-list metadata, or build
  registration changed.
- External dense-reference helpers remain local in `tests/test_ldlt_csc.c`.

## Registration Result

| Surface | Day 8 result |
| --- | --- |
| `Makefile` `TEST_SRCS` | No change. |
| `CMakeLists.txt` `add_sparse_test(...)` | No change. |
| `Makefile` `LIB_SRCS` | No change. |
| `CMakeLists.txt` library source list | No change. |
| `build-metadata/library_sources.txt` | No change. |
| `make source-list-check` | PASS, 49 library sources. |

## Review-Surface Result

| Path | Lines after Day 8 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3469 | Reduced from 3639 after Day 7 and from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_fixtures.h` | 145 | Existing Day 7 family-local fixture header. |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 | New family-local oracle/native-wrapper helper header. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Existing Day 6 family-local supernode helper header. |

## Validation

Validation completed after the source/header edit:

```sh
make format
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make source-list-check
make lint
make test
```

Focused `test_ldlt_csc` validation passed with 100 tests, 0 failures, 0
skips, and 3556 assertions. `make source-list-check` passed with 49 library
sources. The full C gate passed through `make format`, `make lint`, and
`make test`.

## Deferred Candidates

- External dense-reference state and process helpers remain local.
- The random-symmetric builder remains local because it is shared across
  supernodal, native, and solve surfaces.
- Solve residual helpers remain local because they depend on later
  `rel_residual` proof-owner positioning.

## Day 9 Handoff

- Decide whether a selected-cluster guard is needed for the three helper
  headers.
- No library source guard changes are needed.
- If adding a guard, keep it focused on `test_ldlt_csc.c` including the three
  family-local headers.
