# Day 4 LDLT CSC Proof Helper Extraction

## Purpose

Day 4 performs the first Sprint 107 code cleanup in `tests/test_ldlt_csc.c`.
The edit extracts one narrow row-adjacency assertion helper while preserving the
existing LDLT CSC proof behavior, reviewed test registration, and build-system
surface.

## Implemented Change

Added a local static helper:

```c
static void assert_row_adj_matches_l_pattern(const LdltCsc *F, idx_t row);
```

The helper lives immediately before `test_row_adj_matches_reference` and
checks one row at a time:

- scans `F->L` columns `c < row`;
- counts prior columns with a stored `L[row, c]`;
- verifies `F->row_adj_count[row]` matches that count;
- verifies each stored `F->row_adj[row][e]` is in range;
- verifies each stored prior column actually contains `row`;
- preserves the duplicate-detection property through count equality plus
  per-entry membership.

The test body now keeps the high-level proof loop visible:

```c
for (idx_t r = 0; r < n; r++)
    assert_row_adj_matches_l_pattern(F, r);
```

## Scope Control

No changes were made to:

- `RUN_TEST` registration;
- CMake or Makefile test lists;
- public or internal headers;
- compiled helper targets;
- LDLT CSC implementation sources.

This keeps Day 4 inside the selected Day 3 seam and avoids changing the
reviewed consumer or CTest surface.

## Size Impact

`tests/test_ldlt_csc.c` is now 3,885 lines after formatting. The selected
helper makes the row-adjacency proof block reusable and named with a net
one-line increase, which is acceptable because the proof intent at the call
site is clearer and the extraction stays local.

## Validation

Focused affected test:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Result: passed. The focused suite ran 100 tests with 0 failures.

Required C quality gate:

```sh
make format && make lint && make test
```

Result: passed. Formatting, lint, and the full Makefile test suite completed
successfully.

Additional hygiene checks:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_ldlt_csc.c
```

Result: pending at artifact creation time; these are expected to pass before
Day 4 closeout.

Final result: passed. `git diff --check` returned cleanly, and the
trailing-whitespace scan found no matches.

## Completion Criteria Mapping

- Local static helper extracted: complete.
- Row-adjacency proof intent remains visible in the test body: complete.
- No reviewed test-count or build-surface drift: complete.
- Focused affected suite passes: complete.
- Full required C quality gate passes: complete.
