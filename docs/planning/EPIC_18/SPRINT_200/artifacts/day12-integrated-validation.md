# Sprint 200 Day 12 Integrated Validation

## Scope

Day 12 completes the Sprint 200 item 200.6 validation pass for the selected
`sparse_symbolic_lu()` allocation-failure owner proof. The validation covers
the focused selected-owner gate, the broader symbolic allocation-failure gate,
source-list and documentation checks, full formatting, lint, and the full test
suite required for this branch's C and header-adjacent test changes.

## Integrated Validation Results

| Command | Result | Evidence |
| --- | --- | --- |
| `make symbolic-lu-allocation-failure-gate` | PASS | Selected gate reported 3 tests, 0 failures, 0 skipped, and 3054 assertions. |
| `make symbolic-allocation-failure-gate` | PASS | Broader symbolic allocation gate reported 104 tests, 0 failures, 0 skipped, and 4316 assertions. |
| `make source-list-check` | PASS | Source-list guard reported 49 library sources. |
| `make docs-check` | PASS | Doxygen and API docs coverage completed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| `make format` | PASS | Clang-format completed across source, test, benchmark, example, and public header surfaces. |
| `make lint` | PASS | Strict compile, clang-tidy, and cppcheck completed successfully. |
| `make test` | PASS | Full test suite completed with `All tests passed.` |
| `git diff --check` | PASS | Whitespace validation completed before recording this artifact. |

## Focused Gate Result

The focused selected-owner gate remains:

```sh
make symbolic-lu-allocation-failure-gate
```

It validates the registration guard and then runs `test_etree` with
`SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY=1`, limiting the runtime test path to
the three selected symbolic LU allocation-failure tests.

Validated selected-owner behavior:

- allocation-failure status is returned for deterministic failed allocations;
- requested `sym_L` and `sym_U` outputs are cleared before any selected owner
  allocation can fail;
- failed outputs remain free-safe;
- caller-owned matrix and permutation inputs are preserved;
- cleanup remains stable across repeated failed allocations;
- retry after resetting allocation failure succeeds and produces fresh output.

## Risk Register Update

| Risk | Day 12 disposition |
| --- | --- |
| Focused gate drift | Mitigated by `make symbolic-lu-allocation-failure-gate` and the Python registration guard passing. |
| Broader symbolic allocation regression | Mitigated by `make symbolic-allocation-failure-gate` passing after the selected symbolic LU changes. |
| Formatting or source-list drift | Mitigated by `make format`, `make source-list-check`, and `git diff --check` passing. |
| C quality or analyzer regression | Mitigated by `make lint` passing. |
| Runtime regression | Mitigated by `make test` passing. |
| Claim overreach | Retained as a documentation control: the earned claim remains selected `sparse_symbolic_lu()` allocation-failure proof only. |

## Environment Residuals

No Day 12 environment residuals were observed. All required local validation
commands completed successfully.

## Closeout Readiness

Day 12 provides the required integrated validation evidence for item 200.6.
Days 13 and 14 can proceed with review-surface hardening, final evidence
cross-checks, retrospective preparation, and closeout without an open local
validation blocker.
