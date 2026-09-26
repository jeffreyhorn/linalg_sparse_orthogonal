# Sprint 210 Day 13: Review Hardening

## Scope

Day 13 reviewed the selected no-reorder linked-list LDLT allocation-failure
proof for missing cleanup assertions, brittle gate wiring, stale documentation
ownership, and claim-boundary overreach.

The reviewed surfaces were:

- `src/sparse_ldlt.c`;
- `tests/test_ldlt.c`;
- `tests/test_ldlt_allocation_failure_gate_registration.py`;
- `Makefile`;
- `CMakeLists.txt`;
- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- Sprint 210 working notes and artifacts.

## Finding And Fix

| Finding | Risk | Fix |
| --- | --- | --- |
| Registration guard matched raw `RUN_TEST(...)` substrings. | A commented-out proof-owner registration could satisfy the guard while the executable test binary no longer ran the selected proof. | Hardened `tests/test_ldlt_allocation_failure_gate_registration.py` to require active `RUN_TEST(...)` lines exactly once for each selected linked-list LDLT proof-owner test. |

## Invariant Audit

| Invariant | Evidence | Disposition |
| --- | --- | --- |
| Failure sweep breadth | 25 deterministic fail-after cases in `tests/test_ldlt.c`. | Preserved. |
| Cleanup and free safety | `assert_ldlt_failure_output_free_safe(...)`, `assert_ldlt_success_output_free_safe(...)`. | Preserved. |
| Stale-output suppression | `test_ldlt_linked_list_allocation_failures_clear_stale_outputs`. | Preserved. |
| Caller-input preservation | `assert_ldlt_allocation_failure_input_intact(...)`. | Preserved. |
| Retry-after-reset | all-case retry test plus representative baseline-match retry test. | Preserved. |
| Focused gate wiring | Make target, CMake labels, active test registration guard. | Hardened. |
| Claim boundary | README, INSTALL, maintainer guide, and Sprint 210 artifacts retain selected-owner wording and non-claims. | Preserved. |

## Retained Non-Claims

Day 13 does not widen the selected-owner proof. The following remain outside
the Sprint 210 claim:

- CSC LDLT allocation-failure proof;
- reordered LDLT allocation-failure proof;
- Cholesky, broad direct solvers, QR, SVD, eigensolver, sparse matrix
  construction, conversion, IO, package/install, or generated-tooling
  allocation-failure proof;
- operating-system OOM behavior;
- platform parity, hosted CI proof, package-manager proof, shared-library ABI
  proof, performance proof, release readiness, external-library parity, or
  state-of-the-art reliability support;
- concurrent allocation-hook behavior.

## Validation

Commands run:

```sh
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make ldlt-linked-list-allocation-failure-gate
make docs-check
make support-docs-guard
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 tests/test_ldlt_allocation_failure_gate_registration.py` | PASS | `ldlt-allocation-failure-gate-registration: passed`. |
| `make ldlt-linked-list-allocation-failure-gate` | PASS | `95` LDLT tests, `0` failures, `0` skips, and `7781` assertions. |
| `make docs-check` | PASS | Doxygen and API coverage passed with 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` under separate installed-header policy. |
| `make support-docs-guard` | PASS | `test-support-quick-reference-docs: ok`. |
| `git diff --check` | PASS | Whitespace validation completed after Day 13 hardening. |
