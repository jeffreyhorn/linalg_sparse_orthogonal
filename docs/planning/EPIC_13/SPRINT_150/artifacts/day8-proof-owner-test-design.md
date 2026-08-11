# Sprint 150 Day 8: Proof-Owner Test Design

## Purpose

Design focused QR corpus proof-owner tests for the Sprint 150 selected fixture
families. The tests should mirror the executable oracle semantics from Day 7
without expanding the large monolithic QR tests or implying broader QR claims.

## Existing Test Surface

`tests/test_qr_corpus.c` is the right proof owner for Sprint 150.

Current coverage in that file:

- builds `qr_rank_deficient_6x4_nullspace_v1`;
- checks fixture shape and `nnz`;
- checks rank `3` and nullity `1`;
- checks solver-produced nullspace normalized residual;
- checks the deterministic reference null direction residual.

Registration already exists:

- Makefile includes `tests/test_qr_corpus.c` in the test source list.
- CMake registers `add_sparse_test(test_qr_corpus)`.

Day 9 should extend `tests/test_qr_corpus.c` rather than adding a new target.
No Make/CMake registration change should be needed unless the implementation
splits helper code into a new source file.

## Selected Proof Rows

### Rank-Deficient Rectangular QR

Fixtures:

- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`

Proof-owner assertions per fixture:

| Assertion | Source | Tolerance |
| --- | --- | --- |
| shape and `nnz` | Day 5 fixture rows | exact |
| rank | `sparse_qr_rank(&qr, 0.0)` | exact |
| nullity | `sparse_qr_nullspace(&qr, 0.0, NULL, &nullity)` | exact |
| solver-produced nullspace residual | `tf_qr_normalized_matvec_residual()` over each basis vector | `1e-10` |
| projector distance | `max_abs(Z Z^T - z_ref z_ref^T)` | `1e-8` |
| reference direction residual | deterministic reference null vector | `1e-12` advisory assertion |

Reference null vectors:

- `qr_rankdef_duplicate_5x4_v1`: `[0.0, -1.0, 0.0, 1.0]`
- `qr_rankdef_dependent_row_4x3_v1`: `[-1.0, -2.0, 1.0]`

The projector check must normalize both observed and reference vectors and
compare projectors, not raw basis entries. The assertion must continue to pass
if the solver flips the sign of the null vector.

### Underdetermined Minimum-Norm QR

Fixtures:

- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Proof-owner assertions per fixture:

| Assertion | Source | Tolerance |
| --- | --- | --- |
| shape and `nnz` | Day 5 fixture rows | exact |
| solve status | `sparse_qr_solve_minnorm(A, b, x, NULL)` | `SPARSE_OK` |
| residual | `||Ax-b||_2` | `1e-10` |
| solution norm | `||x||_2` | `1e-10` |
| exact values | selected deterministic expected vectors | `1e-10` max component error |

Expected RHS and solution vectors:

| Fixture | RHS | Expected Solution | Expected Norm |
| --- | --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | `[1.0, 1.0]` | `[0.5, 0.5, 0.5, 0.5]` | `1.0` |
| `qr_minnorm_3x6_exact_values` | `[3.0, 4.0, 5.0]` | `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]` | `sqrt(8.4)` |
| `qr_minnorm_5x10_exact_values` | `[1.0, 2.0, 3.0, 4.0, 5.0]` | `[0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | `sqrt(11.0)` |

## Helper Design

Day 9 should add small local helpers to `tests/test_qr_corpus.c`:

- `qr_corpus_assert_shape(A, fixture_key, rows, cols, nnz)`
- `qr_corpus_assert_rankdef_fixture(fixture_key, A, rank, nullity, ref_null, cols)`
- `qr_corpus_projector_distance(vec, ref, n)`
- `qr_corpus_relative_minnorm_residual(A, x, b, rows)`
- `qr_corpus_assert_minnorm_fixture(fixture_key, A, b, expected_x, n, expected_norm)`

The helpers should keep diagnostics fixture-key oriented:

- print fixture key;
- print observed rank/nullity;
- print residual and tolerance;
- print projector distance and tolerance;
- print maximum solution error for minimum-norm fixtures.

Avoid adding a generic TSV parser on Day 9. Source-controlled expected rows are
already validated by `scripts/validate_corpus_schema.py` and exercised by
`scripts/run_corpus_oracle.py`; the focused C proof owner should assert the
same deterministic values directly to stay small and portable.

## Fixture Builder Plan

Reuse existing builders:

- `tf_qr_make_rankdef_duplicate_5x4()`
- `tf_qr_make_dependent_row_4x3()`

Add local static builders in `tests/test_qr_corpus.c` for minimum-norm fixtures
unless an existing helper is already available by Day 9:

- `qr_corpus_make_minnorm_2x4()`
- `qr_corpus_make_minnorm_3x6()`
- `qr_corpus_make_minnorm_5x10()`

Keep builders deterministic and aligned with `scripts/validate_corpus_schema.py`
generator entries.

## Test Case Plan

Day 9 should add these focused tests:

1. `test_qr_corpus_rankdef_duplicate_5x4_shape`
2. `test_qr_corpus_rankdef_duplicate_5x4_rank_nullity_residual_projector`
3. `test_qr_corpus_rankdef_dependent_row_4x3_shape`
4. `test_qr_corpus_rankdef_dependent_row_4x3_rank_nullity_residual_projector`
5. `test_qr_corpus_minnorm_2x4_shape`
6. `test_qr_corpus_minnorm_2x4_status_residual_norm_values`
7. `test_qr_corpus_minnorm_3x6_shape`
8. `test_qr_corpus_minnorm_3x6_status_residual_norm_values`
9. `test_qr_corpus_minnorm_5x10_shape`
10. `test_qr_corpus_minnorm_5x10_status_residual_norm_values`

Do not remove the existing 6x4 seed tests. They remain the Sprint 139 seed
proof and provide continuity for the original maintained QR corpus row.

## Validation Plan

Day 9 should run:

```sh
make build/test_qr_corpus
./build/test_qr_corpus
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Because Day 9 will modify `tests/test_qr_corpus.c`, it must also run:

```sh
make format && make lint && make test
```

All must pass before Day 9 is considered complete.

## Claim Boundary

The proof-owner tests may support fixture-local evidence for the selected
expected rows only. They must not assert or imply:

- broad QR correctness;
- raw QR basis or raw nullspace basis equality;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad minimum-norm or least-squares behavior;
- rank-deficient minimum-norm recovery;
- inconsistent-system behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art claims.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Proof-owner tests are scoped to selected QR families. | Complete | Design extends only `tests/test_qr_corpus.c` for the two selected Sprint 150 fixture families and keeps the 6x4 seed tests. |
| Diagnostics identify the failing family and oracle condition. | Complete | Helper plan requires fixture-key-oriented residual, projector, rank/nullity, and solution-error diagnostics. |
| Registration plan preserves current platform and CI boundaries. | Complete | Existing Makefile and CMake registration already cover `test_qr_corpus`; Day 9 should not add new platform lanes. |
