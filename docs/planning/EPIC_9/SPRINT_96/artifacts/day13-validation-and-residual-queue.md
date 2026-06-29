# Sprint 96 Day 13: Validation and Residual Queue

## Purpose

Day 13 validates the Sprint 96 source and proof-owner cleanup as a whole,
re-checks registrations and renamed owners, and freezes the remaining
maintainability queue for closeout.

## Full Validation

Required code-day quality chain passed:

```sh
make format && make lint && make test
```

Validation covered:

- `clang-format` over source, test, benchmark, example, and public-header
  inputs.
- strict warning compilation with `-Werror`.
- benchmark and example binary builds without execution.
- `clang-tidy` over the full source set.
- `cppcheck` over `src` and `tests`.
- the full test suite, ending with `All tests passed.`

## Registration Checks

New or moved Sprint 96 owners are registered in both build systems:

```text
Makefile:73:           $(SRCDIR)/sparse_iterative_block.c \
Makefile:90:           $(SRCDIR)/sparse_ldlt_dense.c \
Makefile:149:            $(TESTDIR)/test_chol_csc_supernodal.c \
CMakeLists.txt:98:    src/sparse_iterative_block.c
CMakeLists.txt:115:    src/sparse_ldlt_dense.c
CMakeLists.txt:237:add_sparse_test(test_chol_csc_supernodal)
```

The stale Cholesky helper and dispatch names removed on Day 12 are absent from
the split proof-owner files:

```sh
rg -n "day7_chol_csc_get|day8_chol_csc_match|day8_count_supernodes|day9_assert_batched_matches_scalar|day10_factored_matches|day10_roundtrip_check|day11_build_spd|day12_spd_dispatch_and_residual|test_dispatch_day12_bcsstk14_residual" \
  tests/test_chol_csc.c tests/test_chol_csc_supernodal.c tests/test_chol_csc_supernodal_helpers.h
```

The scan returned no matches.

## Current Hotspot Snapshot

Implementation owners after the Sprint 96 splits:

| File | Current lines | Day 13 disposition |
|---|---:|---|
| `src/sparse_ldlt_csc.c` | 2174 | completed fix-now direct cleanup |
| `src/sparse_ldlt_dense.c` | 590 | new dense LDLT primitive owner |
| `src/sparse_iterative.c` | 1495 | completed fix-now solver cleanup |
| `src/sparse_iterative_block.c` | 375 | new block iterative owner |

Proof owners after the Sprint 96 split:

| File | Current lines | Day 13 disposition |
|---|---:|---|
| `tests/test_chol_csc.c` | 2611 | completed core/dispatch proof owner |
| `tests/test_chol_csc_supernodal.c` | 2464 | new supernodal/writeback proof owner |
| `tests/test_chol_csc_supernodal_helpers.h` | 244 | shared Cholesky CSC proof helpers |

Remaining large proof owners from the Day 2 queue:

| File | Current lines | Day 13 disposition |
|---|---:|---|
| `tests/test_ldlt_csc.c` | 3680 | deferred adjacent direct CSC proof owner |
| `tests/test_integration.c` | 3421 | deferred broad lifecycle/progress owner |
| `tests/test_qr.c` | 3234 | deferred QR proof owner |
| `tests/test_iterative.c` | 2841 | deferred iterative proof owner |

## Completed Fix-Now Queue

- Direct-family source cleanup: `src/sparse_ldlt_dense.c` now owns dense
  LDLT primitive behavior that was previously embedded in
  `src/sparse_ldlt_csc.c`.
- Solver-family source cleanup: `src/sparse_iterative_block.c` now owns block
  CG/GMRES wrappers while `src/sparse_iterative.c` retains scalar, handle,
  and matrix-free solver ownership.
- Giant-test cleanup: `tests/test_chol_csc_supernodal.c` now owns Cholesky CSC
  supernodal/writeback proofs while `tests/test_chol_csc.c` retains core CSC
  and dispatch proofs.
- Internal rationale cleanup: touched source and proof-owner comments now
  emphasize durable ownership and invariants instead of implementation
  chronology.

## Residual Maintainability Queue

Deferred source owners:

- `src/sparse_qr.c`: still a large solver/algorithm owner paired with a giant
  proof file; keep for a QR-focused cleanup sprint.
- `src/sparse_eigs.c`: still a large solver owner with restart and handle
  coupling; keep for a solver lifecycle cleanup sprint.
- `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`, and
  `src/sparse_chol_csc.c`: remain direct-family cleanup candidates, but broad
  cross-owner movement would exceed Sprint 96's bounded scope.
- `src/sparse_matrix.c` and `src/sparse_svd.c`: remain residual because they
  are either highly shared or lower priority than the selected Sprint 96
  direct/iterative lanes.

Deferred proof owners:

- `tests/test_ldlt_csc.c`: adjacent to the LDLT CSC work, but still best left
  as a dedicated direct-proof cleanup because Sprint 96 already touched the
  largest direct source owner.
- `tests/test_integration.c`: broad public lifecycle and progress/cancel
  coverage should not be split opportunistically.
- `tests/test_qr.c`: should move with a QR source cleanup, not as a standalone
  Sprint 96 tail task.
- `tests/test_iterative.c`: remains large, but the Day 7-9 source extraction
  did not require assertion or registration movement in this proof owner.
- `tests/test_svd.c`, `tests/test_ldlt.c`, `tests/test_etree.c`,
  `tests/test_graph.c`, and `tests/test_reorder_nd.c`: remain broad residual
  proof-owner candidates.

Intentionally retained non-goals:

- public-header redesign
- benchmark command and harness renames
- generated API documentation edits
- broad `docs/algorithm.md` chronology modernization
- simultaneous splits of multiple giant proof owners
- historical sprint names in unrelated legacy tests and build comments

## Closeout Preparation Notes

- Day 14 can summarize Sprint 96 as a bounded maintainability sprint with
  three completed fix-now lanes: LDLT CSC dense extraction, iterative block
  extraction, and Cholesky CSC proof-owner split.
- Closeout should cite the full Day 13 quality chain as the final branch-wide
  validation before retrospective work.
- Residual work is separated into dedicated future cleanup lanes rather than
  being mixed into Sprint 96 closeout.
