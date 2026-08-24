# Sprint 178 Day 9: Cleanup And Error Contracts

## Scope

Day 9 reviews the selected `sparse_matmul()` workspace allocation-failure
proof after the Day 7-8 regressions and confirms whether product cleanup fixes
are required.

The selected subsystem remains limited to:

- `sparse_matmul()`;
- workspace allocations after output shell creation;
- `acc`, `nz_flag`, and `touched` allocation failures.

## Cleanup Review

`sparse_matmul()` already follows the Day 4 cleanup invariant for the selected
workspace allocations:

1. Reject `C == NULL` before writing through the output parameter.
2. Clear `*C` before validating input matrices or shape compatibility.
3. Allocate `out` before selected workspace arrays.
4. On selected workspace allocation failure, free any allocated workspace,
   free `out`, and return `SPARSE_ERR_ALLOC`.
5. Publish `out` only after workspace allocation, product construction, and
   workspace cleanup succeed.

The Day 7-8 regressions pass for all selected failure sites, so no product-code
cleanup fix is required on Day 9.

## Error-Contract Regression

Day 9 adds:

| Test | File | Contract |
| --- | --- | --- |
| `test_matmul_error_precedence_clears_stale_output` | `tests/test_matmul.c` | `C == NULL` has `SPARSE_ERR_NULL` precedence; non-`NULL` output pointers are cleared before `A/B` null checks and shape checks; valid retry still succeeds. |

The regression asserts:

- `sparse_matmul(NULL, NULL, NULL)` returns `SPARSE_ERR_NULL`;
- `sparse_matmul(NULL, B, &C)` returns `SPARSE_ERR_NULL` and clears stale `C`;
- `sparse_matmul(A, shape_bad, &C)` returns `SPARSE_ERR_SHAPE` and clears
  stale `C`;
- the caller-owned stale matrix remains unchanged after both rejected calls;
- a valid retry with the same selected fixture inputs returns `SPARSE_OK` and
  produces the expected product.

## Product-Code Decision

No product implementation change was made. The existing `src/sparse_matrix.c`
ordering already matches the Day 4 public-state and cleanup invariants:

- `C == NULL` is rejected before dereference;
- `*C = NULL` happens before input and shape validation;
- selected allocation failures return `SPARSE_ERR_ALLOC`;
- successful publication happens at the end of `sparse_matmul()`.

## Boundary

Day 9 does not broaden the Sprint 178 proof to:

- `sparse_create()` shell allocation;
- `sparse_insert()` product-flush allocation;
- matrix copy, transpose, CSR/CSC conversion, or build helper allocation;
- any direct solver, decomposition, eigensolver, graph, reorder, package, or
  generated tooling path.

## Validation

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

## Handoff

Day 10 should add a focused Make/CTest gate for the selected `sparse_matmul()`
allocation-failure proof and keep the gate wording scoped to the selected
matrix multiply workspace paths.
