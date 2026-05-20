# Sprint 35 Day 4: Header Cleanup Batch I

## Scope

Apply the Day 3 public-example contract to the first installed-header cleanup batch, with emphasis on the highest-signal truthfulness issue identified on Day 2.

Day 2 already showed that Sprint 35 does **not** inherit a broad installed-header positional-initializer conversion queue. Day 4 therefore focuses on the one header whose public wording no longer matched the implementation:

- `include/sparse_svd.h`

## Main Result

Day 4 closed the highest-priority installed-header truthfulness gap in `include/sparse_svd.h`.

The touched header now does three things consistently:

1. Shows the public example in the Day 3 style-contract form:
   - designated initializer for a non-default path
   - minimal fields only
2. Describes `sparse_svd_compute()` accurately:
   - `opts == NULL` means singular values only
   - `compute_uv = 1, economy = 1` returns thin/economy factors
   - `compute_uv = 1, economy = 0` returns full `U` / `V^T`
3. Describes `sparse_svd_partial()` accurately:
   - approximate singular vectors are supported only for the thin/economy path
   - full-mode vector recovery is not supported in the partial-SVD API

## What Changed

### 1. Usage example now teaches the explicit public contract

The top-level `sparse_svd.h` usage snippet now uses a multi-line designated initializer:

```c
sparse_svd_opts_t opts = {
    .compute_uv = 1,
    .economy = 1,
};
```

and now explicitly tells readers:

- `economy = 1` means thin/economy output
- `economy = 0` requests full `U` / `V^T`

This keeps the example minimal while still naming the current supported branch clearly.

### 2. `sparse_svd_compute()` docs now match shipped behavior

The old public header still claimed that:

- `compute_uv` without economy meant “full SVD not implemented”

That was stale after Sprint 29 Day 3. Day 4 removed that contradiction and replaced it with the current contract:

- `NULL` opts => singular values only
- `compute_uv && economy` => thin/economy vectors
- `compute_uv && !economy` => full orthonormal `U` / `V^T`

### 3. `sparse_svd_partial()` docs now state the real vector-recovery limit

The old comments contradicted themselves:

- one part said singular vectors are recovered when `compute_uv` is set
- another said singular vectors are not computed

The updated docs now say exactly what the implementation does:

- partial SVD supports approximate singular-vector recovery only when
  `compute_uv = 1` and `economy = 1`
- `compute_uv = 1` with `economy = 0` is rejected

## Cross-Header Implication For Day 5

Day 4 reinforces the Day 2 conclusion:

- the remaining header queue is mostly a **wording and consistency** pass
- it is **not** a large syntax conversion batch

That shifts Day 5 toward:

- re-reviewing the remaining touched headers for smaller wording drift
- checking whether any other public header still understates or overstates
  supported behavior
- documenting the residual dependency handoff to README/tutorial cleanup

## Bottom Line

The first installed-header cleanup batch was intentionally narrow and high-value. `include/sparse_svd.h` now teaches the current public SVD contract instead of an obsolete one, and it does so using the public example style rule Sprint 35 defined on Day 3.
