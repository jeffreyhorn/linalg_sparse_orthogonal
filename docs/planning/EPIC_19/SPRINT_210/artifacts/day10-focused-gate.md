# Sprint 210 Day 10: Focused Gate Wiring

## Purpose

Wire a repeatable focused validation path for the selected linked-list LDLT
allocation-failure owner proof without implying broad LDLT, CSC LDLT, or
direct-solver allocation-failure coverage.

## Implementation Summary

| File | Change |
| --- | --- |
| `Makefile` | Added `ldlt-linked-list-allocation-failure-gate`. |
| `CMakeLists.txt` | Added CTest labels `ldlt;linked_list;allocation_failure` to `test_ldlt`. |
| `tests/test_ldlt_allocation_failure_gate_registration.py` | Added registration guard for the Make target, CMake label, selected `RUN_TEST(...)` entries, representative failure cases, and key assertions. |

## Focused Command

```sh
make ldlt-linked-list-allocation-failure-gate
```

The target:

1. builds `$(BUILDDIR)/test_ldlt`;
2. runs `python3 tests/test_ldlt_allocation_failure_gate_registration.py`;
3. runs `$(BUILDDIR)/test_ldlt`;
4. prints `ldlt-linked-list-allocation-failure-gate: passed`.

## Guarded Registration Contract

The registration guard requires:

- `.PHONY: ldlt-linked-list-allocation-failure-gate`;
- `ldlt-linked-list-allocation-failure-gate: $(BUILDDIR)/test_ldlt`;
- Makefile invocation of the guard script;
- `add_sparse_test(test_ldlt)` in CMake;
- CTest labels `ldlt;linked_list;allocation_failure`;
- selected linked-list LDLT allocation failure, cleanup, stale-output, and retry
  `RUN_TEST(...)` registrations;
- representative fail-after cases from the Day 6 sweep;
- key assertions for 25-case coverage, failure cleanup, success cleanup,
  stale-output sentinels, retry baseline equality, and hook reset.

## Boundary Notes

The gate name is intentionally explicit:

- selected linked-list LDLT only;
- no CSC LDLT claim;
- no reorder allocation-failure claim;
- no solve/refine/condest allocation-failure claim;
- no broad direct-solver allocation-failure claim.

## Validation

Commands run:

```sh
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make ldlt-linked-list-allocation-failure-gate
```

Focused gate result:

- registration guard passed;
- `95` LDLT tests passed;
- `0` tests failed;
- `0` tests skipped;
- `7781` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.
