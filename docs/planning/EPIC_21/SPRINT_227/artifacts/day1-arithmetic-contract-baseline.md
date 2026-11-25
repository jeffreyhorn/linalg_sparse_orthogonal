# Sprint 227 Day 1: Arithmetic Contract and Baseline Inventory

## Decision

The Day 1 baseline is complete, but the arithmetic contract is **not frozen**. Current documentation says all three selected APIs operate in physical index space and warns callers not to use nonidentity permutations. The implementation does not enforce that precondition. A valid row-swap probe returns `SPARSE_OK` from `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` while producing values that differ from independent logical dense arithmetic. Day 2 must choose either explicit pre-mutation rejection or consistent logical arithmetic and then freeze error, ownership, and alias semantics.

No production C/header file changed on Day 1. The baseline source is `fcdeaaee856821e8c801d805c63b5fd40eb89b2f`; the uncommitted Sprint 227 plan and Day 1 records are not part of that code identity.

## API and implementation trace

| Surface | Current implementation | Current public contract | Day 1 finding |
| --- | --- | --- | --- |
| `sparse_add` | Validates three pointers, clears `*C_out`, checks equal shape, creates an identity-permutation output, merges physical sorted rows, drops values with `fabs(value) < 1e-15`, and calls `sparse_insert` for every retained node. | Borrowed inputs, caller-owned output, NULL output after validated-pointer errors, physical-index note, shape/alloc errors. | Inputs are not mutated, but nonidentity permutations are accepted and stripped from the result, creating misleading logical success. Per-node insertion rescans row/column links. |
| `sparse_add_inplace` | Validates pointers/shape, invalidates cached norm and factor state, scales A, then walks B's physical rows and inserts/removes entries in A. It may rescan a row after structural changes. | A may be partially updated after validation on insertion/allocation error; physical-index note; no explicit self-alias rule. | Nonidentity states are accepted. `A == B` is not snapshot-stable because scaling A also changes the borrowed B view before beta is applied. |
| `sparse_matmul` | Validates/clears output, checks inner shape, uses a per-row dense accumulator and touched-column sort, then appends nodes directly with row and column tails into an identity-permutation output. | Borrowed inputs, caller-owned NULL-on-error output, physical-index note, null/shape/alloc errors. | Its output builder is the relevant structural model for addition. Nonidentity states are still accepted and logical values can be wrong. |
| Logical element access | `sparse_get` maps logical row/column through `row_perm` and `col_perm`; `sparse_get_phys` bypasses maps. | Public access distinguishes logical from physical coordinates. | The warning-only arithmetic behavior conflicts with ordinary logical reads unless callers manually uphold an unenforced precondition. |
| Permutation detection | `sparse_matrix_has_non_identity_row_col_perms` checks row, inverse-row, column, and inverse-column maps. | Internal helper, already used by matrix reset logic. | A bounded detection seam exists but is not used by the three selected arithmetic APIs. Day 2 must decide whether it is sufficient for the chosen policy. |

## Current behavior probe

The ignored probe under `build/s227-day1-probe.c` was compiled against the dedicated Release static library. `SPARSE_OK` is 0 and `SPARSE_ERR_ALLOC` is 2.

Inputs:

- A physical values: `[[1,2],[3,4]]`; A logical row map swaps rows, so logical A is `[[3,4],[1,2]]`.
- B logical/physical values: `[[10,20],[30,40]]`.
- M logical/physical values: `[[5,6],[7,8]]`.

| Call | Status | Actual logical output | Independent logical oracle | Baseline verdict |
| --- | ---: | --- | --- | --- |
| `sparse_add(A,B,1,1,&C)` | 0 | `[[11,22],[33,44]]` | `[[13,24],[31,42]]` | Misleading success; physical sum emitted under identity maps. |
| `sparse_add_inplace(Acopy,B,1,1)` | 0 | `[[33,44],[11,22]]` | `[[13,24],[31,42]]` | Misleading success; physical mutation retained A's row map. |
| `sparse_matmul(A,M,&C)` | 0 | `[[19,22],[43,50]]` | `[[43,50],[19,22]]` | Misleading success; physical product emitted under identity maps. |
| `sparse_add_inplace(A,A,2,3)` | 0 | `[[8,16],[24,32]]` | Snapshot expression `[[5,10],[15,20]]` | Self-alias semantics are undocumented and not snapshot-stable. |

These observations establish the baseline defect; they do not choose the Day 2 contract.

## Ownership, error, and mutation inventory

| Condition | `sparse_add` | `sparse_add_inplace` | `sparse_matmul` |
| --- | --- | --- | --- |
| NULL output pointer | `SPARSE_ERR_NULL`; cannot clear output | Not applicable | `SPARSE_ERR_NULL`; cannot clear output |
| NULL input after valid output pointer | `SPARSE_ERR_NULL`; output storage is unchanged because all three pointers are validated before `*C_out` is cleared | `SPARSE_ERR_NULL`; no mutation | `SPARSE_ERR_NULL`; `*C=NULL` because C is validated and cleared before A/B |
| Shape error | `SPARSE_ERR_SHAPE`; `*C_out=NULL` | `SPARSE_ERR_SHAPE`; no mutation | `SPARSE_ERR_SHAPE`; `*C=NULL` |
| Successful output | New caller-owned identity-permutation matrix | A is the caller-owned destination | New caller-owned identity-permutation matrix |
| Allocation/insertion error | Frees temporary C and returns error | Documented partial update after validation | Frees temporary output/workspace and leaves `*C=NULL` |
| Input mutation | A and B read-only | A changes; B nominally read-only, but `A == B` aliases it | A and B read-only |
| Drop threshold | Retains `fabs(value) >= 1e-15` | Removes/skips `fabs(value) < 1e-15` | Retains `fabs(value) >= 1e-15` |
| Factor/cache state | New output starts unfactored with invalid norm cache | Clears factor state and norm cache before arithmetic | New output starts unfactored with invalid norm cache |
| Nonidentity permutations | Warning only; accepted | Warning only; accepted | Warning only; accepted |

## Allocation baseline

The Day 1 fail-after sweep constructed A and B before enabling injection, then invoked `sparse_add` with a stale non-NULL output sentinel.

| Fail-after index | Result | Output |
| ---: | --- | --- |
| 0–5 | `SPARSE_ERR_ALLOC` | NULL |
| 6–7 | `SPARSE_OK` | New result |

The six failures correspond to result row headers, column headers, row permutation, inverse row permutation, column permutation, and inverse column permutation. `sparse_add` uses noninjectable matrix-shell allocation and `sparse_insert` reaches the node pool through the noninjectable node path. Therefore the current sweep cannot force result-shell allocation itself or output-node slab failures. The Sprint 227 builder must expose all owned allocations to deterministic failure testing, matching the stronger matmul pattern.

## Existing test inventory and gaps

| Suite | Current baseline | Covered | Missing for Sprint 227 |
| --- | --- | --- | --- |
| `test_sparse_arith` | 23 tests, 103 assertions, 471 source lines | Identity add/in-place basics, scaling, exact cancellation, shape/null, rectangular add, solver/norm integration | Nonidentity permutations, independent randomized dense oracle, output structural links, self alias, stale output, allocation failures, retry, partial-update assertions |
| `test_matmul` | 41 tests, 372,262 assertions, 1,547 source lines | Dense oracles, row/column links, empty/rectangular/large finite cases, exhaustive selected allocation paths, retries and stale output | Nonidentity permutations on either operand and explicit policy regression |

No current focused test references nonidentity row/column maps for these arithmetic calls. Matmul verifies successful outputs have identity maps, which is correct only after the input policy is resolved.

## Independent dense-oracle fixture freeze

The companion [fixture matrix](day1-dense-oracle-fixtures.tsv) freezes 14 deterministic rows. Expected numerical values are computed in logical dense coordinates, never by `sparse_add`, `sparse_add_inplace`, `sparse_matmul`, or their linked-list output. For permuted cases, Day 2 will map each row to one of two acceptable policy outcomes:

1. produce the recorded logical dense result with valid identity output maps, or
2. return the selected argument error before output allocation or destination mutation.

The matrix includes identity, all-zero stored operands, rectangular shapes, exact cancellation, row-only and column-only maps on either operand, both operands permuted, in-place self alias, wide rows, and seed `0x2270ADD` for replayable randomized coverage.

## Focused baseline execution

Commands:

```sh
cmake -S . -B build/s227-day1-cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/s227-day1-cmake \
  --target test_sparse_arith test_matmul --parallel 4
ctest --test-dir build/s227-day1-cmake --output-on-failure \
  -R '^(test_sparse_arith|test_matmul)$'
/usr/bin/time -l build/s227-day1-cmake/test_sparse_arith
/usr/bin/time -l build/s227-day1-cmake/test_matmul
```

Results:

| Process | Correctness result | Real time | Maximum resident set size | Peak memory footprint |
| --- | --- | ---: | ---: | ---: |
| `test_sparse_arith` | 23/23 tests; 103 assertions; 0 failures | 0.00 s | 815,104 bytes | 569,344 bytes |
| `test_matmul` | 41/41 tests; 372,262 assertions; 0 failures | 0.07 s | 3,526,656 bytes | 3,215,360 bytes |

The CTest wrapper reported both tests passing in 0.83 seconds. These are whole-process local measurements on `Darwin 24.6.0` x86_64 with Apple clang 17.0.0. They are context for regression planning, not isolated addition performance evidence. The static library used by the probe has SHA-256 `235c8cb34a3eacf06717f42ac6846ea8cdb7f2aecfd5be023b13d902995dd072`.

## Day 2 handoff

Day 2 must resolve these questions before code changes:

1. reject every nonidentity row/column or inverse map before mutation, or support logical arithmetic across all three APIs;
2. define error precedence and exact error code for unsupported maps;
3. define whether `sparse_add_inplace(A,A,...)` has snapshot semantics or is rejected;
4. preserve or deliberately revise the documented in-place partial-update contract;
5. specify output-map, factor-state, cache, threshold, NaN/Inf, stale-output, and allocation guarantees;
6. map every frozen fixture to a success result or unchanged-input rejection.

Until that decision is recorded, the warning-only behavior remains a known safety gap and no implementation closure is claimed.
