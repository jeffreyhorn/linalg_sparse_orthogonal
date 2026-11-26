# Sprint 228 Day 5: Differential Acceptance and Promotion Design

## Outcome

The pre-promotion differential corpus is complete. Day 5 adds four bounded seeded fixtures, reconstructs an independent long-double reference from a separately accepted economy-SVD factor set, rejects generated cells in the frozen threshold ambiguity band, and runs both explicit production routes against that reference. The same 123-test focused binary passes under 32- and 64-bit index configurations.

No default or selector behavior changes on Day 5. This artifact fixes the implementation seam and the exact tests that must accompany the production transition on Days 6–8.

## Seeded replay corpus

The generator is a local xorshift64* stream. A fixture is reproduced from its full 64-bit seed, dimensions, density, rank, and tolerance; each diagonal also receives a deterministic nonzero increment so density draws cannot accidentally produce an empty factorization.

| Fixture ID | Seed | Shape | Rank | Requested density | Drop tolerance | Expected output `nnz` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `seeded-square-5-r1` | `0x0000000022800001` | 5×5 | 1 | 40% | `0.2` | 15 |
| `seeded-tall-7x4-r3` | `0x0000000022800002` | 7×4 | 3 | 65% | `1e-8` | 24 |
| `seeded-wide-4x7-r4` | `0x0000000022800003` | 4×7 | 4 | 85% | `0.05` | 24 |
| `seeded-sparse-6-r2` | `0x0000000022800004` | 6×6 | 2 | 25% | `0.5` | 23 |

Generated nonzeros are signed integers with magnitude 1–8. The fixture set spans square, tall, and wide dimensions; rank one, intermediate rank, and maximum accepted rank; sparse through dense inputs; mixed signs; and several positive threshold scales.

For each fixture the oracle:

1. computes an economy SVD with vectors independently of either sparse reconstruction helper;
2. accumulates every requested-rank cell in `long double` and records the sum of absolute terms;
3. computes the frozen per-cell tolerance `64 * DBL_EPSILON * max(sigma[0], sum_abs_terms)`;
4. rejects the fixture if any magnitude lies inside the threshold ambiguity band;
5. constructs the expected retained-coordinate matrix without calling either reconstruction route or using `sparse_insert()` for classification; and
6. runs explicit `off` and `on` routes through the existing logical, normwise, ownership, and internal sparse-structure gates.

Observed maximum relative Frobenius errors were `7.033e-17`, `1.146e-16`, `1.068e-16`, and `9.885e-17`, respectively. Both routes produced the independently expected `nnz` for all four seeds.

The completed pre-promotion harness contains 42 explicit route cells: 18 core Day 3 cells, 16 degenerate/numerical Day 4 cells, and 8 seeded Day 5 cells.

## Index-width confirmation

The ordinary build uses `SPARSE_IDX_BITS=32`. A separate build tree, `build/sprint228-day5-64`, was configured through the maintained Make variable with `SPARSE_IDX_BITS=64`; it did not reuse 32-bit objects.

| Configuration | Focused result |
| --- | --- |
| 32-bit | `test_svd`: 123 tests, 17,333 assertions, no failures or skips |
| 64-bit | `test_svd`: 123 tests, 17,333 assertions, no failures or skips |
| 64-bit identity lane | `test_idx_width_functional`: typedef/header/archive all 64; 5 tests and 115 assertions pass |

The bounded numerical results and sparse structures are identical across index widths. This is selected width-sensitive coverage, not a claim of universal 64-bit algorithm coverage.

## Size and overflow audit

The reconstruction path was audited from public dimensions through factor storage and output construction:

- `rank_k` is an `idx_t` and is rejected unless `1 <= rank_k <= min(m,n)` before factorization.
- Economy factor allocations check `m*k`, `n*k`, and byte multiplication with `sparse_size_mul_overflow()` or equivalent `SIZE_MAX / sizeof(double)` bounds before allocation.
- Dense compatibility reconstruction checks `m*n` with `sparse_size_mul_overflow()` and then checks element count against `SIZE_MAX / sizeof(double)` before `calloc()`.
- Dense accumulator offsets are bounded by that validated `m*n` allocation.
- Outer-route `s*m+i` and `j*k+s` offsets are bounded by the already validated and successfully allocated factor extents; the route adds no `m*n` allocation.
- `sparse_create(m,n)` owns checked shell/map allocation, while `sparse_insert()` owns checked node/slab growth and status propagation.
- Dimensions originate from an existing valid matrix, so neither route converts an unvalidated negative dimension to `size_t`.
- On platforms where a positive 64-bit `idx_t` could exceed `size_t`, factor or matrix-shell checked allocation must reject the object before reconstruction is reachable. Promotion must not weaken those checks.

No unchecked new allocation-size multiplication is required by the promoted outer route. Day 6 must still add explicit finite-term/accumulator checks and preserve all cleanup exits; Day 8 must prove default dispatch and invalid-selector precedence.

## Maintained promotion seam

The production change remains internal to `src/sparse_svd.c`; public declarations and structure layouts do not change.

### Selection

Replace the current two-state fallthrough parser with a checked internal parser:

- environment unset → outer route;
- exact `on` → outer route;
- exact `off` → dense compatibility route;
- empty or any other set value → `SPARSE_ERR_BADARG`.

Parsing occurs after pointer/rank/state/tolerance validation and before SVD work, exactly as frozen on Day 2. The parser returns status plus an internal route enum; invalid text must never silently select a route.

### Output construction

Both routes retain one common owner contract:

1. the caller's `*result` is set to NULL at API entry;
2. factorization owns `sigma`, `U`, and `Vt` until reconstruction returns;
3. the selected route creates and owns one output shell;
4. route failure destroys every partial node and shell and returns no output;
5. route success transfers the shell to the API wrapper;
6. the wrapper frees factors and then transfers the output to the caller.

The maintained outer helper computes each cell once in ascending singular-index order, validates every factor/term/running sum/final cell as finite, applies the common threshold predicate, and inserts at most one node per coordinate. Dense compatibility keeps its checked `m*n` accumulator only behind exact `off`.

### Cleanup transitions

Every transition has one owner:

- parser/validation failure: no factor or output allocation;
- factorization failure: factor routine's documented cleanup, output remains NULL;
- output-shell failure: free factors, return `SPARSE_ERR_ALLOC`;
- insertion or numeric failure: route frees partial output, wrapper frees factors;
- success: wrapper frees factors and publishes exactly one output.

No route may publish before all cells pass, and a retry after failure must start from unchanged input.

## Required selector and precedence tests

These tests are fixed now and land with the corresponding production behavior rather than as intentionally failing Day 5 tests:

| Test | Expected result after promotion |
| --- | --- |
| Selector unset | Outer route, `SPARSE_OK`, accepted output |
| Selector `on` | Outer route, `SPARSE_OK`, accepted output |
| Selector `off` | Dense compatibility route, `SPARSE_OK`, accepted output |
| Selector empty, `ON`, `Off`, whitespace, or arbitrary text | `SPARSE_ERR_BADARG`, NULL output, no SVD work |
| Invalid selector plus all-zero input | `SPARSE_ERR_BADARG`; zero short circuit cannot bypass parsing |
| NaN or either infinity tolerance with valid selector | `SPARSE_ERR_BADARG`, NULL output, before SVD work |
| Invalid rank/state plus invalid selector | Earlier frozen rank/state error wins |
| NaN/infinite matrix data with valid arguments | `SPARSE_ERR_NUMERIC`, NULL output |
| Caller environment initially unset | Test restores unset state after explicit-route invocation |
| Caller environment initially set to `on`, `off`, or invalid text | Test restores the exact original bytes after invocation |
| Sequential route changes | No cached selector; each call observes its current stable environment |

Environment mutation tests remain serial. Concurrent process-environment mutation is outside the API contract.

## Promotion readiness

The differential shape/rank/tolerance/degenerate matrix is complete and both current routes satisfy it. This permits hardening the outer route; it does not yet permit changing the default. Promotion remains blocked on:

- Day 6 finite arithmetic and cleanup hardening;
- Day 7 sparse output-construction hardening;
- Day 8 default/parser implementation and the tests above;
- Days 9–10 failure, sanitizer, and quality evidence; and
- Day 11 frozen memory/timing collection.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | 32-bit pass: 123 tests, 17,333 assertions |
| `make BUILDDIR=build/sprint228-day5-64 SPARSE_IDX_BITS=64 build/sprint228-day5-64/test_svd && ./build/sprint228-day5-64/test_svd` | 64-bit pass: 123 tests, 17,333 assertions |
| `make BUILDDIR=build/sprint228-day5-64 SPARSE_IDX_BITS=64 build/sprint228-day5-64/test_idx_width_functional && ./build/sprint228-day5-64/test_idx_width_functional` | 64-bit identity and functional pass: 5 tests, 115 assertions |

Day 5 changes tests and planning evidence only. Production code and public headers remain unchanged.
