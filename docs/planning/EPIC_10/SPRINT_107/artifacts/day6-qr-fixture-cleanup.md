# Day 6 QR Fixture Cleanup

## Purpose

Day 6 implements the QR fixture-builder cleanup selected by the Day 5 boundary
while preserving QR proof clarity at the edited call sites.

## Implemented Builders

Added three local static builders in `tests/test_qr.c` near the existing helper
declarations:

```c
static SparseMatrix *make_qr_small_banded_4x3(int include_tail);
static SparseMatrix *make_qr_duplicate_column_4x3(double duplicate_scale);
static SparseMatrix *make_qr_near_duplicate_4x3(double perturbation);
```

The builders return `SparseMatrix *` or `NULL`, matching the existing test
pattern:

```c
ASSERT_NOT_NULL(A);
if (!A)
    return;
```

No shared test helper header, compiled helper target, public header, Makefile,
CMake, or test-registration change was introduced.

## Updated Call Sites

Updated the safe small-fixture call sites:

| helper | updated tests |
|---|---|
| `make_qr_small_banded_4x3(0)` | `test_q_roundtrip` |
| `make_qr_small_banded_4x3(1)` | `test_q_apply_multiple`, `test_sparse_mode_basic` |
| `make_qr_duplicate_column_4x3(1.0)` | `test_qr_solve_rank_deficient`, `test_known_nullspace` |
| `make_qr_duplicate_column_4x3(2.0)` | `test_qr_rank_deficient` |
| `make_qr_near_duplicate_4x3(1e-12)` | `test_qr_nearly_singular` |

`test_qr_refine_ill_conditioned` was intentionally left inline after checking
the live fixture: its near-duplicate matrix uses a different base progression
than `test_qr_nearly_singular`. Keeping it inline preserves the original
refinement-specific fixture semantics while still documenting it as future QR
cleanup debt.

## Proof Preservation

The cleanup preserved these proof statements at test sites:

- expected rank values and rank inequalities;
- reconstruction labels and tolerances;
- sparse-mode-vs-dense-mode solution and residual comparisons;
- nullspace verification;
- iterative-refinement before/after residual checks.

No `RUN_TEST` entries were added, removed, or renamed.

## Size Impact

`tests/test_qr.c` is now 3,210 lines after formatting, down from the Day 5
baseline of 3,234 lines.

## Remaining QR Debt

Deferred intentionally:

- `test_qr_refine_ill_conditioned` fixture extraction, because its matrix
  differs from the selected near-duplicate builder;
- generated sin/cos fixture builders;
- tall/economy structured builders;
- diagonal and singleton builders;
- loaded SuiteSparse exact-RHS builders;
- any assertion helper that hides QR proof expectations.

## Validation

Focused affected test:

```sh
make build/test_qr && ./build/test_qr
```

Result: passed. The focused suite ran 73 tests with 0 failures.

Required C quality gate:

```sh
make format && make lint && make test
```

Result: passed after the final semantic-preservation edit.

Additional hygiene checks:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md tests/test_qr.c tests/test_ldlt_csc.c
```

Result: pending final Day 6 hygiene pass.

Final result: passed. `git diff --check` returned cleanly, and the
trailing-whitespace scan found no matches.

## Completion Criteria Mapping

- QR fixture builders extracted: complete.
- Only approved small-fixture call sites updated, with one semantic mismatch
  intentionally left inline: complete.
- Proof intent remains readable at edited call sites: complete.
- Focused QR suite passes: complete.
- Full C quality gate passes: complete.
- Remaining QR proof-owner debt is documented: complete.
