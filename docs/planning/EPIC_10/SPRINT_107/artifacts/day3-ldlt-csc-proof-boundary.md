# Day 3 LDLT CSC Proof Boundary

## Purpose

Day 3 freezes the narrow `tests/test_ldlt_csc.c` proof-helper seam for the
first Sprint 107 code cleanup. The selected Day 4 edit must reduce repeated
row-adjacency proof logic without changing LDLT CSC behavior, public API
surface, reviewed test registration, or compiled helper targets.

## Inputs

- Sprint 107 Day 2 ranking:
  `docs/planning/EPIC_10/SPRINT_107/artifacts/day2-residual-boundary-rerank.md`
- residual owner: `tests/test_ldlt_csc.c`
- Sprint 106 completed source extraction:
  `src/sparse_ldlt_csc_rowadj.c`
- Sprint 106 constraint: only one narrow row-adjacency assertion or
  residual/oracle helper should be extracted from `tests/test_ldlt_csc.c`

## Inspected Proof Areas

### Row-Adjacency Assertions

Relevant tests and helpers:

- `test_row_adj_empty_round_trip`
- `test_row_adj_append_preserves_order`
- `test_row_adj_geometric_growth`
- `test_row_adj_append_arg_checks`
- `test_row_adj_swap_slots_moves_whole_row_state`
- `test_row_adj_matches_reference`

The first five tests are compact unit-style checks around allocation,
append/growth, argument validation, and swap-slot behavior. Their assertions
are intentionally local and should remain inline.

`test_row_adj_matches_reference` is the best Day 4 candidate. It builds a
reference view by scanning `F->L` and then checks every `F->row_adj[r]` entry
against that reference across 20 random indefinite cases. The row-local
expected-count and membership assertion block is repeated proof logic inside a
larger case loop and can become a local assertion helper while preserving the
test's high-level intent.

### Residual and Oracle Helpers

Relevant helpers:

- `rel_residual`
- `s20_solve_residual`
- `read_ldlt_external_dense_reference_solution`
- `assert_ldlt_external_dense_reference`

These helpers already exist and support Sprint 98, Sprint 102, and Sprint 20
external/dense-reference claims. Moving or reshaping them in Day 4 would be
larger than the Sprint 107 instruction because those helpers span solve,
external reference, and comparison semantics rather than the row-adjacency
assertion boundary.

### Dense Oracle and Symmetric Swap Proofs

Dense oracle checks around `ldlt_csc_symmetric_swap` and native-vs-wrapper
comparisons are important but out of scope for the first Sprint 107 extraction.
They are more coupled to factorization state shape and solve behavior than the
row-adjacency reference assertion block.

## Selected Day 4 Candidate

Extract a local assertion helper for row-adjacency membership:

```c
static void assert_row_adj_matches_l_pattern(const LdltCsc *F, idx_t row);
```

The helper should:

- scan `F->L` columns `c < row`;
- compute the expected number of prior columns with stored `L[row, c]`;
- assert `F->row_adj_count[row] == expected_count`;
- assert each stored `F->row_adj[row][e]` is in range `0 <= c < row`;
- assert each stored prior column actually contains `row`;
- keep failure messages close to the row/column being checked if practical;
- remain `static` and test-local in `tests/test_ldlt_csc.c`.

The edited test should keep this high-level loop visible:

```c
for (idx_t r = 0; r < n; r++)
    assert_row_adj_matches_l_pattern(F, r);
```

That preserves the proof intent: after native elimination, every row's
row-adjacency index must exactly match the lower-triangular `L` pattern.

## Non-Selected Candidates

| candidate | reason deferred |
|---|---|
| extract a residual/oracle helper | Existing residual/oracle helpers already exist and carry external comparison semantics. Moving them would be broader than "one narrow" Day 4 cleanup. |
| extract row-adj append/growth assertion helpers | The small unit tests are already readable and local; helper extraction would hide simple setup more than it helps. |
| extract dense oracle comparison helpers | Dense oracle and symmetric-swap checks are coupled to broader factor-state behavior and should not be the first bounded Sprint 107 edit. |
| create a shared test helper header | Sprint 107 explicitly avoids new compiled helper targets, and a shared header is unnecessary for one local assertion helper. |

## Placement and Naming

- File: `tests/test_ldlt_csc.c`
- Placement: near the row-adjacency test section, before
  `test_row_adj_matches_reference`
- Name: `assert_row_adj_matches_l_pattern`
- Linkage: `static`
- Scope: local to `tests/test_ldlt_csc.c`

No public header, internal library header, Makefile, CMake, or source-list
change is needed.

## Failure-Message Behavior

Day 4 should preserve or improve failure locality:

- count mismatch should identify the row;
- out-of-range row-adj entries should identify row and entry index;
- missing `L[row, c]` membership should identify row and candidate prior
  column.

If adding custom `TF_FAIL_` messages makes the helper noisier than the current
assertions, keep the existing assertion macros but leave the row/column values
as nearby local variables.

## Validation Plan

Because Day 4 will edit a `.c` test file, required validation is:

1. Focused affected test:

   ```sh
   make build/test_ldlt_csc && ./build/test_ldlt_csc
   ```

2. Full required C quality gate:

   ```sh
   make format && make lint && make test
   ```

3. Hygiene:

   ```sh
   git diff --check
   ```

No CTest registration count check is expected because Day 4 should not add,
remove, or rename `RUN_TEST` entries.

## Completion Check

- The Day 4 seam is narrow and local to `test_row_adj_matches_reference`.
- Direct CSC proof intent remains visible at the call site.
- No public API, install-header, build-system, or compiled helper target change
  is implied.
- Validation commands are known before the test edit starts.
