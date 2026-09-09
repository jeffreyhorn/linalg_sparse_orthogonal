# Sprint 200 Day 10 Focused Gate Evidence

## Scope

Day 10 implements Sprint 200 item 200.5 for the selected
`sparse_symbolic_lu()` allocation-failure owner. The gate is intentionally
selected-owner scoped. It does not claim broad allocation-failure coverage for
all symbolic analysis, direct solvers, sparse matrix construction, platform OOM
behavior, or concurrent allocation-hook use.

## Gate Wiring

| Surface | Day 10 evidence |
| --- | --- |
| Make target | `make symbolic-lu-allocation-failure-gate` |
| Test binary | `build/test_etree` |
| Gate selector | `SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY=1` |
| Registration guard | `tests/test_symbolic_lu_allocation_failure_gate_registration.py` |
| Existing broad symbolic gate | `make symbolic-allocation-failure-gate` remains intact. |

The selected gate runs only the symbolic LU allocation-failure proof tests from
`tests/test_etree.c`:

1. `test_symbolic_lu_allocation_failures_clear_outputs`
2. `test_symbolic_lu_allocation_failures_cleanup_sweep`
3. `test_symbolic_lu_allocation_failures_recover_on_retry`

The registration guard requires the Make target, the environment-gated
`test_etree` entry point, the three selected tests, the exact symbolic LU
failure-site table, the failed-output free-safe assertions, the allocation hook
reset probe, and the fresh retry-output assertion.

## Source And Test List Alignment

No new C test file was added on Day 10. The focused proof reuses the existing
`tests/test_etree.c` binary, so no source-list update was required. The only new
test-support file is a Python registration guard that is invoked directly by
the new Make target.

## Validation

Commands run:

```sh
make format && make symbolic-lu-allocation-failure-gate
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `make format` | PASS | Formatting completed before the focused gate. |
| `make symbolic-lu-allocation-failure-gate` | PASS | 3 tests, 0 failures, 0 skipped, 3054 assertions. |
| `make symbolic-allocation-failure-gate` | PASS | 104 tests, 0 failures, 0 skipped, 4316 assertions. |
| `make lint` | PASS | Strict compile, clang-tidy, and cppcheck completed. |
| `make test` | PASS | Full test suite completed with all tests passed. |
| `git diff --check` | PASS | Whitespace validation completed after documentation updates. |

## Completion Assessment

Item 200.5 is implemented for the selected `sparse_symbolic_lu()` owner. The
selected-owner proof can now be run without the full suite, and missing proof
registration fails clearly through the Python guard.
