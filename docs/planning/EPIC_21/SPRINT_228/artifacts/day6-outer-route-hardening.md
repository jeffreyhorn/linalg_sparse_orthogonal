# Sprint 228 Day 6: Outer-Route Hardening

## Outcome

The lower-memory per-cell reconstruction route now fails closed at its internal factor boundary. It preserves the accepted ascending-singular-index accumulation order, exact `fabs(value) >= drop_tol` retention rule, row-major coordinate traversal, and one-insert-per-coordinate construction while adding checked factor extents and explicit numerical-failure handling.

The default remains the dense compatibility route on Day 6. Selection promotion and public selector validation are Day 7 work; this change hardens the route that will be promoted.

## Before/after discrepancy record

The 42-cell frozen correctness corpus exposed no value, ordering, duplicate, threshold, rectangular, or cancellation discrepancy between the existing routes. Review of the lower-memory helper did expose latent boundary gaps not reachable from ordinary accepted factors:

| Boundary | Before Day 6 | Day 6 behavior |
| --- | --- | --- |
| Output argument | Dereferenced without a NULL check and not cleared on entry | NULL output pointer returns `SPARSE_ERR_NULL`; every valid output pointer is cleared before validation |
| Factor metadata | Trusted caller-supplied dimensions, mode, and pointers | Requires matching positive dimensions, economy mode, `k=min(m,n)`, valid rank, and all factor pointers |
| Factor extents | Offset products relied on upstream allocation history | Converts `m`, `n`, and `k` to `size_t`; checks sigma, `m*k`, and `n*k` byte extents before indexing or output allocation |
| Singular values and vectors | NaN/Inf could flow into multiplication or sparse insertion | Every used singular value and vector entry must be finite; negative singular values fail closed |
| Arithmetic | Product or running-sum overflow could produce a non-finite stored value | Checks scaled factor, complete term, and next running sum; any non-finite result returns `SPARSE_ERR_NUMERIC` |
| Partial output | Insertion errors destroyed the partial matrix, but numeric failure was not detected | Numeric and insertion failures both destroy the partial matrix and leave the caller's output NULL |
| Retry | Not directly pinned | A failure after the first node insertion is followed by a successful retry from the same factor object |

Zero singular values terminate the useful descending factor range, matching the dense compatibility route and avoiding reads from mathematically unused vectors.

## Maintained internal proof surface

`sparse_svd_lowrank_outer_product()` is declared in `src/sparse_svd_internal.h`. It remains outside the installed public API and exists so the public wrapper and selected tests share the exact hardened implementation. Its contract requires a resolved finite nonnegative threshold and an economy factorization whose metadata agrees with the supplied dimensions.

Ownership is now explicit:

1. the helper clears `*out` before any validation;
2. factor storage remains borrowed and is never modified or freed by the helper;
3. the helper exclusively owns the sparse output while constructing it;
4. any numeric or insertion failure destroys that output; and
5. only complete success transfers the output to the caller.

The public wrapper continues to own and free the SVD factors after the helper returns.

## Focused regressions

Three maintained tests supplement the frozen corpus:

- `test_s228_outer_checked_dimensions_and_arguments` covers NULL output, malformed metadata, missing factors, non-finite thresholds, invalid rank, and impossible factor byte extents. The `IDX_MAX` extent case rejects before dereferencing synthetic one-element pointers or attempting an output allocation under both index widths.
- `test_s228_outer_numeric_failure_destroys_partial_output` inserts the first accepted node, encounters an infinite later factor, requires `SPARSE_ERR_NUMERIC` with NULL output, then repairs that factor and proves a successful structurally valid retry. It also covers a non-finite singular value before construction.
- `test_s228_outer_numeric_product_and_sum_overflow` distinguishes scaled-product overflow, later-cell product overflow after partial construction, and accumulation overflow from two individually finite terms.

Existing square, tall, wide, rank-deficient, threshold-equality, signed-cancellation, extreme finite-scale, and seeded replay tests continue to check numerical results and complete row/column sparse structure for explicit `on` and `off` routes.

## Size and ownership closure

- Sigma storage requires checked `k*sizeof(double)` bytes.
- U storage requires checked `m*k*sizeof(double)` bytes.
- Vt storage requires checked `n*k*sizeof(double)` bytes.
- Validated extents dominate every `s*m+i` and `j*k+s` offset used by the helper.
- The route allocates no reconstruction array and introduces no `m*n` size calculation.
- `sparse_create()` retains responsibility for checked shell/map allocations.
- `sparse_insert()` retains responsibility for node-slab allocation and sorted row/column linking.
- All helper exits either own no output or transfer exactly one complete output.

Deterministic allocation fail-index coverage for shell/maps and node slabs remains scheduled for Days 8–9; Day 6 closes the helper's dimension, numerical, partial-output, and ownership boundaries without changing the allocation subsystem.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | Passed: 126 tests, 17,362 assertions, no failures or skips |
| `make BUILDDIR=build/sprint228-day6-64 SPARSE_IDX_BITS=64 build/sprint228-day6-64/test_svd && ./build/sprint228-day6-64/test_svd` | Passed: 126 tests, 17,362 assertions, no failures or skips |

The full lint, runtime, and documentation gates are recorded in `WORKING_NOTES.md` after final Day 6 validation.
