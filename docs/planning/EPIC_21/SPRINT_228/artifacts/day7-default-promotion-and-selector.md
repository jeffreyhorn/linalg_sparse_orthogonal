# Sprint 228 Day 7: Default Promotion and Selector Contract

## Outcome

The hardened per-cell reconstruction route is now the maintained default for
`sparse_svd_lowrank_sparse()`. An ordinary call no longer reaches the retained
`calloc(m*n, sizeof(double))` compatibility accumulator. Economy-SVD factors,
bidiagonalization storage, sparse output shells, maps, and nodes are unchanged
and are not included in that narrower scratch-removal claim.

No public symbol, option structure, or ABI layout changed.

## Route and validation contract

`SPARSE_SVD_LOWRANK_OUTER` remains a maintainer compatibility control:

| Selector state | Route/status |
| --- | --- |
| Unset | Maintained per-cell route |
| Exact `on` | Maintained per-cell route |
| Exact `off` | Dense-accumulator compatibility route |
| Any other set value, including empty | `SPARSE_ERR_BADARG`, output NULL |

The public wrapper now applies the frozen order before decomposition:

1. validate output and clear it;
2. validate input, rank, and original matrix state;
3. reject non-finite `drop_tol`;
4. parse the selector exactly;
5. reject non-finite stored matrix values;
6. compute the SVD, resolve finite nonpositive tolerance with
   `DBL_EPSILON * sigma[0]`, and dispatch the selected route.

Thus malformed selectors cannot be hidden by an all-zero input or non-finite
matrix data, and neither route accepts non-finite public input silently.

## Regression coverage

Day 7 adds maintained tests that prove:

- unset output is identical to explicit `on` output and to the retained
  explicit `off` compatibility result on the pinned analytical fixture;
- empty, case-varied, whitespace-padded, alias-like, and numeric selector
  values fail closed and clear a sentinel output pointer;
- NaN and infinite tolerances fail before selector and SVD work;
- selector validation precedes all-zero short-circuit and matrix-value checks;
- valid selection with non-finite matrix data returns `SPARSE_ERR_NUMERIC`;
- temporary explicit-route tests restore prior unset, valid, and invalid
  environment state exactly, demonstrating no cached or leaked selection;
- the historical corpus comparison invokes exact `off` and `on`, rather than
  accidentally treating the promoted default as the dense baseline.

## Caller and documentation audit

Active call sites in `examples/example_svd_lowrank.c`, `docs/tutorial.md`, and
`docs/cookbook.md` use the public function without depending on the old dense
scratch. No benchmark directly invokes this API. The public header, README,
and maintainer guide now describe the promoted default, exact compatibility
selector, invalid-value behavior, and the boundary between removed route
scratch and retained SVD work storage. Historical sprint evidence remains
unchanged because it accurately records the behavior at that time.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | Passed: 129 tests, 17,485 assertions, no failures or skips |
| `make BUILDDIR=build/sprint228-day7-64 SPARSE_IDX_BITS=64 build/sprint228-day7-64/test_svd && ./build/sprint228-day7-64/test_svd` | Passed: 129 tests, 17,485 assertions, no failures or skips |

The full lint, test, documentation, and whitespace gates are recorded in the
working notes after their final Day 7 run.

## Remaining boundary

Day 7 provides source-level dispatch proof: the only call path to the dense
`m*n` accumulator is the exact `off` branch. Day 8 retains the planned
allocation instrumentation and fail-index work so this structural proof is
supplemented by exact dynamic allocation evidence.
