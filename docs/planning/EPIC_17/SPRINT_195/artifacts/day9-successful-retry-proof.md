# Sprint 195 Day 9: Successful Retry Proof

## Purpose

Prove that `sparse_symbolic_cholesky()` can recover after deterministic
allocation failure and complete a later successful call with the same
caller-owned fixture inputs.

## Retry Coverage Added

Day 9 added `test_symbolic_cholesky_allocation_failures_recover_on_retry` to
`tests/test_etree.c`.

The test uses the known-5x5 symbolic fixture because it exercises:

- non-empty `sym->col_ptr` allocation;
- `sym->row_idx` allocation;
- child and marker workspace allocation;
- column-row workspace allocation;
- propagated row-set allocation;
- nontrivial symbolic row output that can be checked exactly.

## Retry Sequence

For each selected fail-after checkpoint, the test:

1. constructs the known-5x5 matrix and computes parent, postorder, and column
   counts before arming the allocation hook;
2. forces `sparse_symbolic_cholesky()` to fail at the selected allocation;
3. resets the allocation hook before any assertion can return;
4. asserts `SPARSE_ERR_ALLOC`;
5. verifies caller-owned matrix entries are intact;
6. verifies the failed symbolic output is empty and remains safe after repeated
   `sparse_symbolic_free(...)`;
7. reruns `sparse_symbolic_cholesky()` with the same fixture data and no
   allocation hook; and
8. compares the retry output against the known-5x5 symbolic oracle.

## Selected Retry Checkpoints

| Fail-after | Allocation class |
| ---: | --- |
| 0 | `sym->col_ptr` |
| 1 | `sym->row_idx` |
| 2 | `child_head` |
| 3 | `child_next` |
| 4 | `marker` |
| 5 | `tmp` |
| 6 | `col_rows` |
| 7 | `col_nrows` |
| 8 | first propagated row-set allocation |

## Ordering Assumptions

The retry proof depends on the existing private allocation-hook convention:
tests reset the hook before arming it and immediately after the selected call,
before any assertion macro can return. The focused gate runs success,
allocation-failure, cleanup, and retry tests in the same `test_etree` process
to catch ordering sensitivity.

## Non-Claims

The retry proof remains scoped to `sparse_symbolic_cholesky()` and the selected
known-5x5 fixture family. It does not claim retry safety for
`sparse_symbolic_lu()`, `sparse_analyze()`, standalone etree/postorder/colcount
helpers, direct solvers, sparse matrix construction, OS OOM behavior, or
concurrent use of the private allocation hook.
