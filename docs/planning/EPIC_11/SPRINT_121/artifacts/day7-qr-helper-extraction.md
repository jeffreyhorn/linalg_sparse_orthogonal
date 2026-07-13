# Sprint 121 Day 7: QR Helper Extraction

## Purpose

Extract the first bounded QR and least-squares proof-helper batch without
changing reviewed test ownership, CTest registration, source-list membership,
scenario tolerances, expected failures, or non-claim boundaries.

## Touched Surfaces

| Surface | Change |
|---|---|
| `tests/test_qr_helpers.h` | Added header-only QR fixture, generated-RHS, reconstruction, and residual helpers. |
| `tests/test_qr.c` | Included the helper header and replaced local QR fixture/reconstruction/residual helpers with `tf_qr_*` calls. |
| `tests/test_qr_solve.c` | Included the helper header and replaced duplicate QR solve helpers with shared `tf_qr_*` calls. |
| `docs/planning/EPIC_11/SPRINT_121/WORKING_NOTES.md` | Records Day 7 implementation evidence and residual helper queue. |
| `docs/planning/EPIC_11/SPRINT_121/artifacts/day7-qr-helper-extraction.md` | Adds this implementation artifact. |

No Makefile, CMake, workflow, package, benchmark, public API, production
source, or CTest membership surfaces were changed.

## Extracted Helper Boundaries

| Helper | New owner | Preserved behavior boundary |
|---|---|---|
| `tf_qr_idx_count_bytes` | `tests/test_qr_helpers.h` | Keeps checked allocation-size arithmetic for generated RHS and residual buffers. |
| `tf_qr_make_exact_rhs` | `tests/test_qr_helpers.h` | Keeps deterministic `x_exact[i] = i + 1` generated-RHS construction and `sparse_matvec` proof path. |
| `tf_qr_insert_or_free` | `tests/test_qr_helpers.h` | Keeps insertion assertion and cleanup behavior for deterministic QR fixtures. |
| `tf_qr_make_small_banded_4x3` | `tests/test_qr_helpers.h` | Keeps the small banded fixture used by QR round-trip and sparse-mode tests. |
| `tf_qr_make_duplicate_column_4x3` | `tests/test_qr_helpers.h` | Keeps duplicate-column rank-deficient fixture construction shared by QR and QR solve tests. |
| `tf_qr_make_near_duplicate_4x3` | `tests/test_qr_helpers.h` | Keeps near-duplicate fixture construction for near-rank-deficient QR evidence. |
| `tf_qr_make_tall_diagonal_dominant` | `tests/test_qr_helpers.h` | Keeps tall diagonal-dominant fixture construction for economy, sparse-mode, and refinement tests. |
| `tf_qr_reconstruction_max_error` | `tests/test_qr_helpers.h` | Keeps `A*P = Q*R` max-entry reconstruction measurement while callers keep tolerance interpretation. |
| `tf_qr_relative_residual_l2` | `tests/test_qr_helpers.h` | Keeps `||b - A*x||_2 / ||b||_2` measurement while callers keep residual targets and scenario labels. |

## Preserved Local Owners

The extraction deliberately left these boundaries in the scenario tests:

- `assert_qr_reconstruction_below` in `tests/test_qr.c`, because the caller
  owns reconstruction labels and tolerances.
- `assert_qr_solve_reconstruction_below` in `tests/test_qr_solve.c`, because
  SuiteSparse reconstruction tolerances remain solve-scenario specific.
- `assert_qr_solve_true_residual_below` in `tests/test_qr_solve.c`, because
  reported residual interpretation and true residual thresholds remain
  scenario-local.
- QR rank, nullspace, economy-mode, sparse-mode, least-squares, and
  QR-vs-LU assertions, because helpers should not hide proof interpretation.
- Minimum-norm helpers in `tests/test_colamd.c`, because ownership remains a
  Day 9 decision or explicit deferral.

## Source-List And CTest Impact

- Added a header-only test helper included by existing QR test executables.
- No new test executable was registered.
- No CTest count change is expected.
- No Makefile or CMake source membership change is required.

## Focused Behavior Evidence

The focused QR executable validates the extracted QR fixture,
reconstruction, residual, rank-deficient, economy, sparse-mode, and
refinement helper paths:

```text
make build/test_qr build/test_qr_solve && ./build/test_qr && ./build/test_qr_solve
test_qr:       63 tests, 0 failures, 0 skips, 576 assertions
test_qr_solve: 10 tests, 0 failures, 0 skips, 972 assertions
ALL TESTS PASSED
```

## Required Quality Evidence

Because Day 7 changed `.c` and `.h` files, the required quality chain is:

```text
make format && make lint && make test
All tests passed.
```

Additional cleanliness checks passed:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_11/SPRINT_121 tests/test_qr.c tests/test_qr_solve.c tests/test_qr_helpers.h || true
```

## Residual Helper Queue

| Candidate | Owner | Reason deferred |
|---|---|---|
| Minimum-norm norm/residual helpers | Day 9 or closeout | Current owner remains `tests/test_colamd.c`; moving it now could obscure historical COLAMD/reordering proof ownership. |
| Rank-deficient fixture expansion | Day 8 | Day 7 only moved existing duplicate/near-duplicate builders; new taxonomy-backed fixtures belong to Day 8. |
| Least-squares fixture expansion | Day 9 | Day 7 preserved current residual helpers; new compatible/incompatible LS evidence belongs to Day 9. |
| Assertion wrappers | Do not extract in Sprint 121 | Assertions encode tolerance, residual, rank, and non-claim semantics that must remain visible in scenario tests. |

## Non-Claims Preserved

Day 7 does not claim LAPACK, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, or
state-of-the-art parity. It only preserves existing in-repository QR and QR
solve behavior while moving reusable fixture and measurement code behind named
test-helper boundaries.

## Completion Criteria Status

| Criterion | Status |
|---|---|
| Item 4 QR helper extraction has an implemented first batch. | Complete. |
| Focused QR tests pass. | Complete. |
| Rank, residual, and least-squares interpretations remain visible. | Complete. |
