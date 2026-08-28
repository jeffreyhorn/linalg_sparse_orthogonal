# Sprint 185 Day 9: Drift Guard Update

## Purpose

Record the selected-cluster registration guard added after the Sprint 185
LDLT CSC helper extractions.

## Guard Added

| Field | Value |
| --- | --- |
| Script | `scripts/check_ldlt_csc_helper_guard.sh` |
| Make target | `make ldlt-csc-helper-guard` |
| Selected proof owner | `tests/test_ldlt_csc.c` / `test_ldlt_csc` |
| Helper scope | Family-local headers extracted during Days 6-8 |

## Guard Coverage

The guard checks that:

- `tests/test_ldlt_csc.c` exists.
- `Makefile` still registers `$(TESTDIR)/test_ldlt_csc.c` in `TEST_SRCS`.
- `CMakeLists.txt` still registers `add_sparse_test(test_ldlt_csc)`.
- The three extracted helper headers exist:
  - `tests/test_ldlt_csc_fixtures.h`;
  - `tests/test_ldlt_csc_oracle_helpers.h`;
  - `tests/test_ldlt_csc_supernode_helpers.h`.
- Each helper header keeps its include guard.
- `tests/test_ldlt_csc.c` includes each helper header exactly once.
- The helper headers are not named directly in Makefile or CMake
  registration.
- The helper headers are not listed in `build-metadata/library_sources.txt`.
- No helper stem is registered as a separate CMake test without a new
  proof-owner decision.

## Registration Result

| Surface | Day 9 result |
| --- | --- |
| `Makefile` | Added `ldlt-csc-helper-guard` target. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added selected-cluster guard script. |
| `CMakeLists.txt` | No change. |
| `build-metadata/library_sources.txt` | No change. |
| Library source list | Unchanged, 49 sources. |
| Test binary ownership | Existing `test_ldlt_csc` remains the only proof owner. |

## Limitations

- The guard verifies helper presence, include ownership, and registration
  drift; it does not prove solver behavior.
- The guard intentionally does not require the helper headers in
  `build-metadata/library_sources.txt` because that manifest tracks library
  `.c` sources.
- The guard intentionally does not create Make/CMake test-count parity
  evidence because no new test binary was added.
- Behavior preservation remains covered by focused `test_ldlt_csc` execution
  and the full C gate from Days 6-8 and later validation days.

## Validation

Validation completed:

```sh
bash -n scripts/check_ldlt_csc_helper_guard.sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Results:

- `bash -n scripts/check_ldlt_csc_helper_guard.sh`: passed.
- `make ldlt-csc-helper-guard`: passed.
- `make source-list-check`: PASS, 49 library sources.
- `git diff --check`: passed.

Day 9 changed a shell script, Makefile target, and planning artifacts. It did
not add new `.c` or `.h` edits, so the Day 9 focused guard/source-list
validation is the relevant gate for this day.

## Day 10 Handoff

- Draft the selected-cluster maintenance note.
- Document where future LDLT CSC helper additions belong.
- Reference `make ldlt-csc-helper-guard` as the guard for the extracted
  helper-header layout.
