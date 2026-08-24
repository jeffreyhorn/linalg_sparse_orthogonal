# Sprint 178 Day 8: Failure Coverage Expansion

## Scope

Day 8 expands the selected `sparse_matmul()` allocation-failure regression
coverage from the first workspace allocation to every selected workspace
ownership path frozen by Day 3.

The selected deterministic failure sites are:

| Failure site | Fail-after count | Covered by Day 8 stale-output check |
| --- | ---: | --- |
| `acc` accumulator workspace | `6` | already covered by Day 7 |
| `nz_flag` nonzero marker workspace | `7` | yes |
| `touched` touched-column workspace | `8` | yes |

## Implemented Regression

Day 8 adds:

| Test | File | Failure sites |
| --- | --- | --- |
| `test_matmul_remaining_workspace_allocation_failures_clear_stale_output` | `tests/test_matmul.c` | `nz_flag`, `touched` |

The test reuses a shared helper that:

1. Builds valid `A` and `B` inputs before enabling fail injection.
2. Creates a separate stale output matrix with sentinel value `42.0`.
3. Calls `sparse_matmul()` with `C` initially pointing at the stale matrix.
4. Injects failure at the selected workspace allocation.
5. Asserts `SPARSE_ERR_ALLOC`.
6. Asserts `C == NULL`.
7. Asserts the separate stale matrix remains caller-owned and unchanged.
8. Resets the private hook and retries the same multiplication successfully.
9. Verifies retry dimensions, nonzero count, and numeric product.

## Coverage Result

All selected `sparse_matmul()` workspace allocation sites now have deterministic
failure coverage with:

- error-contract assertion;
- no stale public output publication;
- caller-owned stale matrix preservation;
- hook reset;
- successful retry;
- numeric product verification.

## Boundary

Day 8 does not expand Sprint 178 to adjacent allocation paths. The following
remain out of scope:

- `sparse_create()` output shell allocation;
- `sparse_insert()` node or slab allocation while flushing the product;
- matrix copy, transpose, or CSR conversion allocation paths;
- direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines,
  reorder routines, package/install flows, and generated-report tooling.

## Validation

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

## Handoff

Day 9 should confirm no product cleanup changes are required for the selected
`sparse_matmul()` workspace failures, preserve public error ordering, and
prepare the focused gate work without broadening the allocation-failure claim.
