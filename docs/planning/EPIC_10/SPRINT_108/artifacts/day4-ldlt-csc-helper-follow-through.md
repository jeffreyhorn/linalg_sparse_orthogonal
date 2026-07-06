# Day 4 LDLT CSC Helper Follow-Through

## Purpose

Day 4 implements the Day 3 selected LDLT CSC proof-helper cleanup. The scope is
intentionally limited to one local assertion helper around the existing Sprint
20 residual computation in `tests/test_ldlt_csc.c`.

## Implemented Helper

Added near `s20_solve_residual`:

```c
static void assert_s20_solve_residual_below(const char *label, LdltCsc *F,
                                            const SparseMatrix *A_ref,
                                            double tol);
```

The helper:

- calls the existing `s20_solve_residual` computation;
- fails with the proof label, actual residual, and tolerance;
- increments the test-framework assertion count on success;
- does not change residual construction, solve behavior, fixture setup,
  factorization, or validation semantics.

## Updated Call Sites

Only the Day 3 approved call sites were updated:

- `test_s20_supernodal_with_analysis_kkt_5x5`
- `test_s20_supernodal_with_analysis_kkt_10x10`
- `test_s20_supernodal_with_analysis_random_indefinite_30x30`
- `test_s20_supernodal_heuristic_vs_with_analysis_residuals`

Each call site still shows:

- the fixture identity;
- the scalar/two-pass or with-analysis factorization path;
- `ldlt_csc_validate` where it already existed;
- the explicit `1e-10` tolerance.

## Explicit Non-Changes

The Day 4 change did not touch:

- external dense-reference oracle helpers;
- `ldlt_csc_factor_state_matches`;
- row-adjacency exact-set proof helpers;
- unrelated LDLT solve or dispatch tests;
- public headers;
- implementation sources;
- Makefile or CMake membership;
- CTest registration.

## Before/After Metrics

| Metric | Before Day 4 | After Day 4 |
|---|---:|---:|
| `tests/test_ldlt_csc.c` lines | 3,887 | 3,896 |
| Approved residual call sites with generic `ASSERT_TRUE` threshold checks | 4 | 0 |
| Approved residual call sites with labeled residual failures | 0 | 4 |
| New compiled helper target | 0 | 0 |
| Public or install-header changes | 0 | 0 |

## Remaining LDLT CSC Debt

Remaining LDLT CSC oracle debt is deferred rather than expanded:

- external dense-reference helper decomposition needs a dedicated oracle-lane
  review because it combines skip behavior, subprocess parsing, dense
  comparison, solve reconstruction, and residual proof;
- factor-state comparison is already specific and should not be split without a
  separate structural-equivalence design;
- unrelated solve and dispatch assertions are outside the Sprint 20
  supernodal with-analysis boundary.

## Validation Plan

Because Day 4 changes a `.c` test file, the required validation is:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
make format && make lint && make test
git diff --check
```

## Completion Criteria Status

- The Day 3 selected helper was implemented.
- Only approved call sites were updated.
- Direct CSC proof readability remains visible at call sites.
- No helper target, reviewed test-count, public API, or install-header surface
  changed.
