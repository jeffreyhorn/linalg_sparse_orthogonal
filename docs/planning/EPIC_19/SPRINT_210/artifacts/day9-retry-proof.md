# Sprint 210 Day 9: Retry Proof

## Purpose

Prove that selected linked-list LDLT allocation failures do not poison later
factorization attempts, and that successful retry output matches an independent
success baseline for representative and boundary failure sites.

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_ldlt.c` | Added `assert_ldlt_success_outputs_match(...)`, `expect_ldlt_retry_matches_success_baseline(...)`, and `test_ldlt_linked_list_retry_matches_success_baseline`. |

No production source changed on Day 9.

## Baseline Comparison

The Day 9 retry test builds an independent success factor before injecting a
failure. After the injected allocation failure is cleaned and the hook is reset,
the same fixture is factored again and compared against the baseline across:

- `n`, `factor_norm`, and `tol`;
- `L` dimensions, `nnz`, and all physical entries;
- `D` and `D_offdiag`;
- `pivot_size`;
- `perm`.

This is stronger than solve-residual-only retry evidence because it proves the
selected owner returns the same factor metadata and sparse `L` structure after
failure cleanup.

## Retry Cases

| `fail_after` | Named site | Why selected |
| ---: | --- | --- |
| 0 | `D output array` | First selected output allocation. |
| 1 | `D_offdiag output array` | Partial output allocation. |
| 2 | `pivot_size output array` | Final initial output-array allocation. |
| 3 | `working copy entry buffer` | First propagated working-copy setup allocation. |
| 11 | `working copy column-tail scratch` | Last working-copy setup allocation before output `L`. |
| 12 | `L row headers` | First output-`L` shell allocation. |
| 18 | `permutation output array` | Selected permutation output allocation. |
| 19 | `column accumulator workspace` | First selected dense workspace allocation. |
| 24 | `pivot-candidate nonzero list workspace` | Final selected dense workspace allocation. |

The existing all-site retry loop remains in place for every one of the 25
failure sites; Day 9 adds baseline-equality proof for representative and
boundary sites.

## Hook And Cleanup Interaction

For each Day 9 retry case:

1. Build a clean success baseline.
2. Inject the selected allocation failure.
3. Reset the allocation hook before assertions.
4. Verify `SPARSE_ERR_ALLOC`, empty/free-safe output, preserved caller input,
   and hook reset.
5. Retry factorization without the hook.
6. Verify solve residual and full baseline equality.
7. Free both factor outputs and the caller-owned fixture.

## Boundary Notes

The proof remains selected linked-list LDLT only. It does not claim retry
correctness for:

- CSC LDLT;
- reorder allocation failures;
- solve/refine/condest workspaces;
- broad direct-solver allocation-failure recovery;
- matrix-pool allocation behavior outside this selected owner path.

## Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `95` tests passed;
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
