# Day 8 Partial-SVD Convergence-Budget Proof Design

## Purpose

Day 8 defines the focused compiled proof owner for the Sprint 140 partial-SVD
fixture before implementation. The proof must exercise the selected residual
without turning the large existing SVD owner into an even broader mixed-purpose
surface.

## Ownership Decision

Add a new focused test owner:

```text
tests/test_svd_partial_corpus.c
```

Rationale:

- `tests/test_svd.c` already owns broad full-SVD, partial-SVD, rank,
  pseudoinverse, low-rank, and helper-registration coverage.
- `tests/test_svd_partial_helpers.h` already contains reusable partial-SVD
  checks, but adding a corpus-specific convergence lane there would keep
  expanding an already large helper header.
- A focused corpus owner can map directly to the Day 5 expected rows and Day 7
  oracle semantics.
- Build-system registration will make the lane visible as its own maintained
  test target.

The new owner should not remove or weaken existing `test_svd` registrations.

## Proof Scope

| Scope | Included |
| --- | --- |
| Fixture | `partial_svd_clustered_repeated_diag8x6_k3_v1` |
| Matrix | Generated 8 x 6 sparse diagonal with entries `10`, `10`, `9.999999`, `4`, `1`, structural zero |
| Requested rank | `k=3` |
| Solver call | `sparse_svd_partial` |
| Default-budget options | `compute_uv=1`, `economy=1`, `max_iter=0`, `tol=0.0` |
| Tight-budget options | `compute_uv=1`, `economy=1`, `max_iter=1`, `tol=0.0` |
| Claim scope | Fixture-local partial-SVD clustered/repeated top-k subspace and budget behavior |

## Focused Test List

| Test | Expected behavior | Day 5 rows covered |
| --- | --- | --- |
| `test_partial_svd_corpus_clustered_repeated_default_success` | Default-budget run returns `SPARSE_OK`, publishes `sigma`, `U`, and `Vt`, reports `m=8`, `n=6`, `k=3`, and matches top-k singular values. | `*_default_status`, `*_singular_values` |
| `test_partial_svd_corpus_clustered_repeated_projectors` | Default-budget U and V top-k projectors match the exact first-three-coordinate projectors within `1e-8`. | `*_left_subspace`, `*_right_subspace` |
| `test_partial_svd_corpus_clustered_repeated_residuals` | Returned triplets satisfy max `A*v ~= sigma*u` and `A^T*u ~= sigma*v` residual `<= 1e-8`; U/V orthogonality residuals are `<= 1e-8`. | `*_vector_residual`, `*_orthogonality` |
| `test_partial_svd_corpus_clustered_repeated_tight_budget_fail_closed` | Tight-budget run returns `SPARSE_ERR_NOT_CONVERGED`, preserves dimensions and requested `k`, and publishes no `sigma`, `U`, or `Vt`. | `*_tight_budget_status`, `*_tight_budget_no_partial_arrays` |
| `test_partial_svd_corpus_clustered_repeated_recovery_after_failure` | A default-budget run after the tight-budget failure succeeds and passes the same singular-value and residual checks. | guards retry/recovery interpretation without adding a new oracle row |

Day 9 may merge the first three default-budget checks into one test if the
assertion output stays readable. The fail-closed and recovery behavior should
remain distinct enough that non-convergence cannot be masked by a later pass.

## Helper Ownership Map

| Helper | Owner | Notes |
| --- | --- | --- |
| Fixture builder | `tests/test_svd_partial_corpus.c` | Keep local to the focused owner because the matrix is one corpus fixture. |
| Diagonal matrix insertion | Local helper or existing `tf_svd_make_diag_matrix` if accessible without broad include churn | Prefer reuse only if it keeps the new owner simple. |
| Singular-value max error | Local static helper | Exact fixture values make this small and readable. |
| Triplet residual max | Reuse logic from `tests/test_svd_partial_helpers.h` only if extracted cleanly; otherwise duplicate a small local helper | Avoid including the whole helper header into a new owner if that creates hidden registrations or broad coupling. |
| Projector distance | Local static helper | The exact projector is diagonal on the first three coordinates, so implementation can compute `||P_observed - P_expected||_F` directly. |
| Orthogonality residual | Existing `tf_dense_column_orthogonality_error` and `tf_svd_vt_row_orthogonality_error` from SVD helpers if available | Reuse is acceptable if headers are already designed for helpers, not tests. |
| Budget/failure cleanup | Local helper around `sparse_svd_free` and `sparse_free` | Keep error paths readable. |

## Projector Design

The exact top-k left projector for the 8-row side is diagonal with ones at
rows `0`, `1`, and `2`. The exact top-k right projector for the 6-column side
is diagonal with ones at columns `0`, `1`, and `2`.

For a returned basis matrix `Q` with `k=3`, compute:

```text
P_observed(i,j) = sum_s Q(i,s) * Q(j,s)
P_expected(i,j) = 1 if i == j and i < 3 else 0
distance = sqrt(sum_ij((P_observed(i,j) - P_expected(i,j))^2))
```

This accepts sign flips and basis rotations inside the repeated leading
singular-value block while rejecting a wrong top-k subspace.

## Budget And Partial-Result Rules

| Path | Required assertion |
| --- | --- |
| Tight budget | `sparse_svd_partial` returns `SPARSE_ERR_NOT_CONVERGED`. |
| Tight budget result struct | `m=8`, `n=6`, `k=3`; `sigma == NULL`; `U == NULL`; `Vt == NULL`. |
| Default after failure | Fresh default-budget call returns `SPARSE_OK` and publishes all factor arrays. |
| Non-converged factors | Never inspect or treat failed-run arrays as partial numerical evidence. |

The recovery test should allocate a fresh `sparse_svd_t` result for the
default-budget run so a failed run cannot contaminate a passing run.

## Build-System Touch Points

Day 9 should update:

- `Makefile`: add `$(TESTDIR)/test_svd_partial_corpus.c` near
  `$(TESTDIR)/test_svd.c`.
- `CMakeLists.txt`: add `add_sparse_test(test_svd_partial_corpus)` near
  `add_sparse_test(test_svd)`.

No Windows-specific exclusion is expected because the fixture uses no POSIX
threads or temporary-file APIs.

## Validation Plan

Day 9 touches `.c` and build files, so required validation is:

```sh
make build/test_svd_partial_corpus
./build/test_svd_partial_corpus
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
make format
make lint
make test
```

If CMake registration changes, also run a focused configure/list check:

```sh
cmake -S . -B build-cmake
ctest --test-dir build-cmake -N | rg test_svd_partial_corpus
```

If the full quality gate fails, stop and ask before proceeding.

## Existing Coverage Preservation

- Keep `test_svd` present in Make and CMake.
- Do not remove existing helper tests in `tests/test_svd_partial_helpers.h`.
- Do not reinterpret generated Day 7 partial-SVD oracle rows as compiled solver
  pass evidence.
- Keep SuiteSparse-dependent SVD tests optional/skip-based as they are today.
- Preserve public non-claims until Day 11 documentation updates have compiled
  proof evidence.

## Day 9 Handoff

Day 9 should implement `tests/test_svd_partial_corpus.c` with the five focused
tests above, register it in Make and CMake, and run the validation plan. Any
helper extraction should stay narrow and should not move unrelated full-SVD or
low-rank logic.
