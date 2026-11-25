# Day 10 In-Place Atomicity and Alias Closure

## Outcome

`sparse_add_inplace` now has executable coverage for its complete frozen mutation contract. Pointer, shape, unsupported-state, and self-alias failures remain pre-mutation and exact. Once validation succeeds, an insertion allocation failure may leave the scaled destination plus the row-major prefix already processed from B; that destination is structurally valid and reusable or destroyable, while B remains exact.

The implementation's arithmetic behavior did not change. Day 10 makes the existing in-place node-slab allocation reachable through the test injector and documents the already-selected partial-update boundary precisely.

## Validation and alias boundary

The cumulative always-on corpus now covers:

- NULL A and B rejection;
- shape rejection before state validation;
- row, column, inverse-map, and factored-state rejection for either operand;
- exact unchanged A/B topology and metadata on every pre-mutation rejection; and
- `A == B` rejection with the cached norm, nodes, maps, factor metadata, and pool state unchanged.

The self-alias contract remains rejection rather than an implicit scaling shortcut. This avoids reading B through links that the same call is mutating.

## Adversarial success cases

A 4x5 dense-oracle fixture combines shared columns, complete cancellation, A-only and B-only entries, scaling, and a fully empty row. It verifies:

- exact `2*A - B` values and `nnz`;
- strict row/column links and identity maps;
- unchanged B pointers, links, values, pool state, maps, and metadata;
- cleared factor metadata on the mutated destination; and
- copy, lookup, and destruction after the mixed update.

A separate 3x3 `alpha=0` case removes all original A nodes and then inserts `-2*B`, proving that the free list is safely reused while B remains exact.

## Allocation failure and partial-update contract

The public insertion path remains non-injectable. A file-local insertion helper now accepts the existing injectable flag, and only `sparse_add_inplace` selects it. This exposes its one fallible operation—allocating a new node slab—without changing normal `sparse_insert` behavior or public ABI.

The deterministic fixture begins with a full current slab, an overlapping B entry, and a following B-only entry. For `A = 2*A + 3*B`:

| Fail index | Result | Destination state |
| ---: | --- | --- |
| 0 | `SPARSE_ERR_ALLOC` | Scaling and the overlapping prefix are retained: A(0,0)=14, A(1,1)=6, and the missing A(0,1) remains zero. |
| 1 | `SPARSE_OK` | A(0,0)=14, A(0,1)=15, A(1,1)=6; one new slab is owned. |
| 2 | `SPARSE_OK` | Same result, proving one boundary beyond the final injectable allocation. |

At every boundary B matches its exact snapshot. The destination has valid row/column links, identity maps, coherent `nnz`, invalidated norm cache, cleared factor state, and coherent pool ownership. After the injected failure, copy, lookup, out-of-place arithmetic, a zero-input in-place call, and destruction all succeed. A clean pre-call copy immediately retries the original operation successfully once injection is reset.

Repeating the original operation directly on the partially updated destination is intentionally not described as transactional retry: it may apply the retained prefix again. The public header now states that limitation.

## Evidence

| Command | Result |
| --- | --- |
| `make format` | Pass |
| Focused 32-bit arithmetic suite | 49 tests, 79,666 assertions, zero failures/skips |
| Focused 64-bit arithmetic suite | 49 tests, 79,666 assertions, zero failures/skips |
| Focused ASan/UBSan arithmetic suite | 49 tests, 79,666 assertions, zero failures/skips |
| `make lint` | Pass: strict warnings, clang-tidy, and cppcheck |
| `make test` | Pass: complete repository test suite |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `git diff --check` | Pass |

