# Sprint 151 Day 9: Proof-Owner Test Implementation

## Purpose

Implement focused partial-SVD corpus proof-owner tests for the three Sprint
151 selected fixture families and validate that every selected expected-result
row now has executable test coverage.

## Implemented Test Changes

| File | Change |
| --- | --- |
| `tests/test_svd_partial_corpus.c` | Added fixture-key constants for the Sprint 151 selected partial-SVD families. |
| `tests/test_svd_partial_corpus.c` | Added local diagonal matrix builders for rank-deficient projector, sparse low-rank output, and fail-closed convergence fixtures. |
| `tests/test_svd_partial_corpus.c` | Generalized the sorted top-k singular-value error helper for reusable fixture-local comparisons. |
| `tests/test_svd_partial_corpus.c` | Added focused rank-deficient metadata/value and projector/residual tests. |
| `tests/test_svd_partial_corpus.c` | Added focused sparse low-rank output shape, nnz, selected-value, dense-error, and sparse/dense consistency test. |
| `tests/test_svd_partial_corpus.c` | Added focused fail-closed tight-budget and default-budget recovery test. |

No new test binary was added. The existing focused proof owner
`tests/test_svd_partial_corpus.c` remains the maintained corpus proof surface.

## New Focused Tests

| Test | Fixture Key | Expected Rows Covered |
| --- | --- | --- |
| `test_partial_svd_corpus_rankdef_diag6x4_k2_metadata_and_values` | `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | default status, singular values, rank |
| `test_partial_svd_corpus_rankdef_diag6x4_k2_projectors_and_residuals` | `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | left projector, right projector, vector residuals, orthogonality |
| `test_partial_svd_corpus_lowrank_rect5x7_k3_sparse_output` | `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | sparse status, shape, nnz, selected values, dense Frobenius error, sparse-vs-dense Frobenius difference |
| `test_partial_svd_corpus_fail_closed_diag6_k2_recovery` | `partial_svd_fail_closed_diag6_k2_v1` | tight-budget status, no partial arrays, recovery status, default singular values, default vector residuals |

## Assertion Coverage

### Rank-Deficient Rectangular

Executable assertions now cover:

- `sparse_svd_rank(A, 1e-8, &rank)` returns rank `2`;
- default `sparse_svd_partial()` returns `SPARSE_OK`;
- result dimensions are `6x4`, `k=2`;
- `sigma`, `U`, and `Vt` are non-null on success;
- sorted top-2 singular values match `9,6` within `1e-8`;
- left/right coordinate-range projector distances are at most `1e-8`;
- max Av and AtU residuals are at most `1e-8`;
- U and V selected-column orthogonality residuals are at most `1e-8`.

### Sparse Low-Rank Output

Executable assertions now cover:

- dense low-rank and sparse low-rank calls both return `SPARSE_OK`;
- sparse output shape is `5x7`;
- sparse output has exactly `3` retained nonzeros via `sparse_nnz()`;
- selected values `(0,0)=8`, `(1,1)=4`, `(2,2)=2`, and `(3,3)=0` match
  within `1e-10`;
- dense low-rank Frobenius error equals `1.0` within `1e-10`;
- sparse-vs-dense Frobenius difference equals `0.0` within `1e-10`.

### Fail-Closed Convergence

Executable assertions now cover:

- tight-budget `max_iter=1` returns `SPARSE_ERR_NOT_CONVERGED`;
- failed result retains `m=6`, `n=6`, and `k=2`;
- failed result publishes no `sigma`, `U`, or `Vt`;
- default-budget recovery returns `SPARSE_OK`;
- recovered result has non-null `sigma`, `U`, and `Vt`;
- sorted top-2 singular values match `9,6` within `1e-8`;
- recovered max Av and AtU residuals are at most `1e-8`.

## Diagnostics

New focused tests print fixture-keyed diagnostics:

- `partial_svd_rankdef_diag6x4_k2_range_projector_v1:`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1:`
- `partial_svd_fail_closed_diag6_k2_v1:`

The diagnostics report ranks, statuses, residuals, shape, nnz, sparse/dense
differences, and singular-value errors. They do not print raw singular-vector
entries as expected evidence.

## Focused Validation

Commands run before the full C gate:

```sh
make build/test_svd_partial_corpus
./build/test_svd_partial_corpus
make build/test_svd
./build/test_svd
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Results:

- `./build/test_svd_partial_corpus`: `10` tests, `247` assertions, all passed.
- `./build/test_svd`: `114` tests, `2067` assertions, all passed.
- Corpus schema validation passed.
- Partial-SVD oracle generation passed.
- Report index normalization passed with `105` rows.

## Claim Boundaries

The Day 9 proof-owner tests remain fixture-local. They do not claim:

- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness;
- storage or drop-tolerance optimality;
- sparse-output performance;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.

## Full Gate Validation

Because Day 9 changed a `.c` file, the required final quality gate was:

```sh
make format && make lint && make test
```

The full gate passed after the focused validation above.
