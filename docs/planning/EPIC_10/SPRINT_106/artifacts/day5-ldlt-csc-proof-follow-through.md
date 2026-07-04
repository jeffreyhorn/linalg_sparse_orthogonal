# Sprint 106 Day 5 - LDLT/CSC Test and Oracle Follow-Through

## Purpose

Day 5 reviewed the Day 4 LDLT CSC row-adjacency extraction from the proof and
documentation side. The goal was to close any helper-boundary coverage gap,
keep failure localization tight, and update ownership notes that still implied
row-adjacency support lived inside the larger LDLT CSC implementation owner.

## Reviewed Surfaces

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_rowadj.c`
- `src/sparse_ldlt_csc_internal.h`
- `tests/test_ldlt_csc.c`
- `tests/test_direct_csc_regression.c`
- `docs/maintainer_guide.md`
- `docs/algorithm.md`

## Proof Follow-Through

The extraction introduced three private row-adjacency helper responsibilities:

| helper responsibility | direct proof after Day 5 | indirect proof retained |
|---|---|---|
| append growth and argument checks | `tests/test_ldlt_csc.c` append/growth tests | native elimination setup |
| per-column row-adjacency population | `test_row_adj_matches_reference` | SuiteSparse row-adj regression path |
| slot swap during symmetric pivoting | `test_row_adj_swap_slots_moves_whole_row_state` | symmetric-swap and native-elimination paths |

Day 5 added direct coverage for the only gap: slot swapping. The new test
builds two row-adjacency rows with different lengths, capacities, and backing
pointers, swaps the slots, and verifies that pointer, count, capacity, and row
contents move as one ownership unit.

## Helper Movement Decision

No test helper was moved or renamed on Day 5. The existing local structure in
`tests/test_ldlt_csc.c` already keeps row-adjacency allocation, append, swap,
and native-elimination proofs close to the implementation they exercise. A
fixture extraction would add indirection without improving current failure
localization.

## Documentation and Ownership Updates

- `src/sparse_ldlt_csc.c` now states that row-adjacency support is owned by
  `src/sparse_ldlt_csc_rowadj.c`.
- `src/sparse_ldlt_csc_internal.h` now lists
  `src/sparse_ldlt_csc_rowadj.c` as a private-contract consumer.
- `docs/maintainer_guide.md` now names the row-adjacency helper owner and its
  direct/indirect proof surfaces.
- `docs/algorithm.md` now identifies the row-adjacency implementation owner
  and distinguishes it from the numeric cmod/pivot logic in
  `src/sparse_ldlt_csc.c`.

## Validation

Required checks passed:

- `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (43 library sources)`
- `make build/test_ldlt_csc && ./build/test_ldlt_csc`
  - `Tests run: 100`
  - `Tests failed: 0`
  - `Tests skipped: 0`
  - `Assertions: 2335`
- `make format && make lint && make test`
  - final output: `All tests passed.`
- `git diff --check`
- trailing-whitespace scan across touched source, documentation, and Sprint
  106 planning files
- final `python3 scripts/check_library_sources.py`

## Exit State

The Day 4 extraction now has a matching Day 5 proof update. Row-adjacency
append, population, and swap responsibilities have explicit test ownership,
and source/documentation comments point maintainers to the new private helper
owner.
