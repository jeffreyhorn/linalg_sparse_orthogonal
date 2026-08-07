# Day 10 Proof-Owner Cleanup

## Scope

Day 10 keeps the Day 9 partial-SVD corpus proof maintainable without widening
the Sprint 140 claim surface. The cleanup is limited to test/helper ownership
for reusable partial-SVD residual and coordinate-projector checks.

## Cleanup Applied

| Surface | Change |
| --- | --- |
| `tests/test_svd_partial_shared_helpers.h` | Added a focused shared helper header for reusable partial-SVD residual and coordinate-projector checks. |
| `tests/test_svd_partial_helpers.h` | Moved existing reusable residual/projector helper implementations behind the shared helper header. |
| `tests/test_svd_partial_corpus.c` | Removed local duplicate triplet-residual and projector helpers and now calls the shared helper APIs. |

## Ownership Boundary

- The shared helper header owns math-only checks that are useful to more than
  one partial-SVD proof owner.
- `tests/test_svd_partial_helpers.h` continues to own the historical
  broad-partial-SVD test cases.
- `tests/test_svd_partial_corpus.c` continues to own the Sprint 140
  clustered/repeated fixture construction, default/tight-budget scenario setup,
  and fixture-specific expected singular values.
- No public API, fixture manifest semantics, or oracle output contract changed
  on Day 10.

## Validation Plan

Because Day 10 touches `.c` and `.h` files, the required validation gate is:

```sh
make format
make lint
make test
```

Focused checks should also confirm both partial-SVD owners still build and pass:

```sh
make build/test_svd_partial_corpus
./build/test_svd_partial_corpus
make build/test_svd
./build/test_svd
```

The corpus/schema checks remain unchanged from Day 9 and should continue to
pass after helper cleanup.

