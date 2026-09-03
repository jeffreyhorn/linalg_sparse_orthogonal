# Sprint 195 Day 8: Cleanup and Stale-Output Proof

## Purpose

Strengthen the selected `sparse_symbolic_cholesky()` allocation-failure proof
with cleanup-specific assertions for stale output, free-after-failure safety,
and repeated cleanup safety.

## Cleanup Coverage Added

Day 8 added `assert_symbolic_failure_free_safe(...)` in `tests/test_etree.c`.
The helper asserts that a failed `sparse_symbolic_t` output is empty, then calls
`sparse_symbolic_free(...)` twice and rechecks the empty state after each call.

The helper is now used by:

- `test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr`;
- `test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr`;
- `test_symbolic_cholesky_allocation_failures_clear_partial_state`.

## Assertion Trace

| Failure class | Cleanup assertion |
| --- | --- |
| Empty matrix `col_ptr` allocation | Output remains NULL/zero and free-safe after failure. |
| Non-empty `sym->col_ptr` allocation | Stale scalar fields are cleared before failure publication and remain free-safe. |
| `row_idx` allocation | Published `col_ptr` is cleaned before return and repeated free stays safe. |
| Child and marker workspace allocations | Partial output and temporaries are cleaned before return. |
| Column-row workspace allocations | Caller-visible output remains empty after mid-construction failure. |
| Propagated row-set allocations | Late construction failure clears symbolic output and leaves no stale publication. |

The known-5x5 failure helper still asserts caller-owned matrix data remains
intact before validating free-after-failure safety.

## Harness Drift Guard

`tests/test_symbolic_allocation_failure_gate_registration.py` now requires the
cleanup helper call site so the focused Make gate cannot drift back to only
status-code assertions.

## Diagnostic Notes

The existing private allocation harness does not expose reliable per-test
allocation counters, so Day 8 does not add counter-based leak assertions. The
diagnostic evidence is instead bounded to deterministic allocation failures,
explicit empty-state assertions, repeated cleanup calls, the focused
`symbolic-allocation-failure-gate`, and formatting/diff checks.

## Non-Claims

This proof remains selected-owner-only. It does not claim exhaustive etree,
analysis, LU symbolic, direct-solver, sparse-matrix constructor, OS OOM, or
concurrent allocation-hook cleanup coverage.
