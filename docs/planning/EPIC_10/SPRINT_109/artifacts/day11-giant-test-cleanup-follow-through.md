# Day 11 Giant-Test Cleanup Follow-Through

## Purpose

Day 11 implements the Day 10 approved giant-test cleanup batch in
`tests/test_qr.c` while preserving QR proof visibility at the updated call
sites.

## Implemented Cleanup

Added one local static helper:

```c
make_qr_exact_rhs(const SparseMatrix *A, idx_t x_len, idx_t b_len,
                  double **x_exact_out, double **b_out)
```

The helper owns only repeated setup:

- allocate `x_exact`;
- allocate `b`;
- fill `x_exact[i] = i + 1`;
- compute `b = A*x_exact`.

It does not hide QR factorization, QR solve/refine calls, rank checks,
residual labels, residual tolerances, reconstruction checks, or QR-vs-LU
comparison assertions.

## Updated Call Sites

The setup helper replaced repeated exact-RHS construction in seven call sites:

| Test | Preserved Proof at Call Site |
|---|---|
| `test_qr_solve_nos4` | QR factorization, solve, rank print, and residual label/tolerance. |
| `test_qr_bcsstk04` | rank assertion, reconstruction assertion, solve, and residual tolerance. |
| `test_qr_west0067` | rank print, solve, and residual tolerance. |
| `test_qr_vs_lu` | QR solve, LU solve, residual comparison, and max-difference assertion. |
| `test_qr_tall_synthetic` | synthetic matrix construction, reconstruction check, solve, and residual tolerance. |
| `test_qr_reorder_nos4_fillin` | fill-in comparison, AMD solve, and residual assertion. |
| `test_qr_refine_nos4` | QR solve, refine call, before/after residual print, and tolerance. |

## Explicit Non-Changes

The cleanup intentionally did not change:

- tiny literal RHS tests;
- overdetermined least-squares RHS values;
- rank-deficient RHS values;
- dense-vs-sparse QR comparison logic;
- QR fixture builders already completed in Sprint 108;
- public headers;
- private production headers;
- CTest registration;
- Make/CMake test target membership.

No new compiled helper target was added.

## Metrics

| Metric | Before Day 11 | After Day 11 |
|---|---:|---:|
| `tests/test_qr.c` lines | 3210 | 3194 |
| local exact-RHS setup helper calls | 0 | 7 |
| repeated exact-RHS fill/matvec blocks in selected QR family | 7 | 0 |
| new test helper targets | 0 | 0 |
| CTest registration changes | 0 | 0 |

The line count uses the Day 10 inventory snapshot for the before value and the
post-format Day 11 working tree for the after value.

## Validation

Focused QR validation:

```sh
make build/test_qr && ./build/test_qr
```

Result:

- passed;
- 73 tests;
- 647 assertions;
- 0 failures;
- 0 skips.

Full required C quality gate:

```sh
make format && make lint && make test
```

Result:

- passed;
- formatting completed;
- strict warning syntax check passed;
- `clang-tidy` completed;
- `cppcheck` completed;
- full `make test` completed with all tests passed.

## Remaining Residuals

Deferred giant-test cleanup remains outside this Day 11 batch:

- QR sequential RHS fill helper for non-exact least-squares/refinement smoke;
- LDLT CSC external dense-reference oracle cleanup;
- per-solver iterative exact-RHS helper families;
- SVD storage-layout proof-loop cleanup.

These remain deferred because each either provides less immediate review value
than the selected QR exact-RHS helper or risks hiding proof logic that should
remain visible at call sites.

## Completion Criteria Status

- Approved Day 10 cleanup batch was implemented.
- Focused QR validation passed.
- Full required C quality gate passed.
- Assertion specificity and failure localization remain at call sites.
- No helper-target, public-header, install-header, or CTest registration drift
  was introduced.
