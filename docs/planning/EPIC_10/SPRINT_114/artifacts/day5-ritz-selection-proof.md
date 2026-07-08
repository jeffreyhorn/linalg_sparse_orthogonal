# Day 5 Repeated and Clustered Ritz Selection Proof

## Purpose

Day 5 implements the repeated/clustered Ritz selection proof designed on Day 4.
The implementation keeps exact repeated-value behavior at the selector
boundary and proves clustered-but-distinct public behavior through the grow-m
Lanczos surface.

## Implemented Proofs

| Proof | File | Test | Evidence |
|---|---|---|---|
| Repeated `LARGEST` / `SMALLEST` selector behavior | `tests/test_ldlt_backend_dispatch.c` | `test_s114_select_indices_repeated_largest_smallest` | Uses sorted `theta = {-4, -4, -1, 0.5, 0.5, 2, 2, 9}` and asserts exact selected indices for repeated values. |
| Equal-magnitude `NEAREST_SIGMA` selector tie behavior | `tests/test_ldlt_backend_dispatch.c` | `test_s114_select_indices_nearest_sigma_equal_magnitude_ties` | Uses `theta = {-5, -3, -1, 1, 3, 5}` and asserts the current right-endpoint-first tie contract. |
| Clustered public Ritz values | `tests/test_eigs.c` | `test_clustered_largest_ritz_selection_public_values` | Uses a diagonal spectrum with top cluster `{10.0, 9.99999, 9.99998}` and asserts public eigenvalue order/values without claiming vector uniqueness or exact multiplicity behavior. |

## Proof Boundaries

- Exact repeated values are asserted only through `s20_select_indices`.
- Public clustered-spectrum proof uses distinct eigenvalues with visible
  `1e-5` gaps.
- The public test asserts eigenvalue values and ordering only.
- No Ritz selector source movement was performed.
- No public API, install-header, source-list, helper-target, Make, CMake, or
  reviewed CTest registration changed.

## Validation Plan

Day 5 modifies C tests, so the required quality gate is:

```sh
make format && make lint && make test
```

## Completion Criteria

- Repeated values are selected deterministically at the selector boundary.
- Equal-magnitude nearest-sigma ties preserve the current selector contract.
- Clustered-but-distinct public spectra return the intended eigenvalue set.
- Ritz selection movement remains blocked until the Day 10 movement decision
  can consider the full proof stack.
