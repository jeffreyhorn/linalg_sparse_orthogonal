# Sprint 96 Day 12: Internal Comment & Rationale Cleanup

## Purpose

Day 12 removes stale sprint/day chronology from files touched in Days 5-11
where that chronology no longer helps reviewers understand current ownership,
invariants, or compatibility constraints.

## Files Reviewed

Implementation owners:

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_dense.c`
- `src/sparse_ldlt_csc_internal.h`
- `src/sparse_iterative.c`
- `src/sparse_iterative_block.c`
- `src/sparse_iterative_internal.h`

Proof owners:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal.c`
- `tests/test_chol_csc_supernodal_helpers.h`

Build files:

- `Makefile`
- `CMakeLists.txt`

## Cleanup Summary

Cholesky CSC tests:

- Rewrote the `tests/test_chol_csc.c` file header so it describes current
  core proof ownership instead of the original development sequence.
- Replaced main-runner comments such as "Day 1" and "Day 6" with semantic
  proof-group names.
- Rewrote stale drop-tolerance comments to describe the current immutable
  `col_ptr` and zero-padding contract.
- Renamed dispatch-local helpers from day-number names to intent-based names:
  `dispatch_build_spd`, `dispatch_spd_fixture_residual`, and
  `test_dispatch_bcsstk14_residual`.

Cholesky CSC supernodal tests:

- Rewrote section headers so they describe current proof groups:
  supernode detection, supernodal-etree corpus safety, dense primitives,
  diagonal-block factor, panel integration, parametrised cross-checks, and
  writeback.
- Replaced development-story comments around supernodal-etree postorder with
  durable numeric and corpus-safety contracts.
- Rewrote the Kuu regression comment around the maintained
  write-and-zero-pad fast path.
- Renamed family-local helper functions from day-number names to
  intent-based names:
  `chol_csc_value_at`, `chol_csc_values_match`,
  `count_grouped_supernode_columns`, `assert_supernodal_matches_scalar`,
  `factored_sparse_matches`, and `writeback_roundtrip_check`.
- Replaced `day10:` diagnostic prefixes with `writeback:`.

LDLT CSC implementation:

- Rewrote dense LDLT comments in `src/sparse_ldlt_dense.c` to describe the
  current dense Bunch-Kaufman primitive and supernodal caller contract.
- Rewrote selected `src/sparse_ldlt_csc.c` comments around atomic 2x2
  supernode boundaries, row-adjacency initialization, analysis-aware
  conversion, native Bunch-Kaufman stepping, solve ownership, and batched
  supernodal writeback.
- Rewrote selected `src/sparse_ldlt_csc_internal.h` comments around
  2x2-aware supernode detection, analysis-aware conversion, wrapper/native
  kernel selection, workspace ownership, symmetric swap ownership, and
  batched LDLT supernodal entry.

Solver cleanup files:

- Re-scanned `src/sparse_iterative.c`, `src/sparse_iterative_block.c`, and
  `src/sparse_iterative_internal.h`.
- No additional edits were needed: the Day 9 comments already describe the
  current block-solver split and private helper ownership.

## Intentionally Retained Rationale

Some historical context remains where it explains current compatibility
contracts:

- LDLT wrapper/native compatibility history remains where it explains why the
  wrapper path is still compiled and why tests can override kernel selection.
- Direct CSC regression suite names still contain older sprint numbers because
  those are existing test names outside the Sprint 96 cleanup scope.
- Build-system comments unrelated to files touched by the Sprint 96 cleanup
  lanes were left alone.

## Validation

Focused checks passed:

```sh
make build/test_chol_csc build/test_chol_csc_supernodal build/test_ldlt_csc build/test_direct_csc_dispatch build/test_direct_csc_regression
./build/test_chol_csc
./build/test_chol_csc_supernodal
./build/test_ldlt_csc
./build/test_direct_csc_dispatch
./build/test_direct_csc_regression
```

Focused suite results:

- `test_chol_csc`: 92 tests passed
- `test_chol_csc_supernodal`: 60 tests passed
- `test_ldlt_csc`: 96 tests passed
- `test_direct_csc_dispatch`: 10 tests passed
- `test_direct_csc_regression`: 8 tests passed

Required full code-day quality chain passed:

```sh
make format && make lint && make test
```

## Day 12 Exit State

Touched source and proof-owner files now emphasize current ownership,
invariants, and compatibility behavior. The remaining historical comments are
intentional rationale rather than stale implementation sequence notes.
