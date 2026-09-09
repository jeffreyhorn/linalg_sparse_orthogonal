# Sprint 201 Day 10 Focused Regression

## Summary

Day 10 ran the selected behavior-preservation checks for the Sprint 201 SVD
review-surface extraction and recorded the final invariant-to-test traceability
table.

No additional C regression test was required. The existing `test_svd` proof
owner already exercises every selected rank, pseudoinverse, and dense low-rank
invariant after the Day 6 and Day 7 helper extraction, and the Day 9 guard
regression covers ownership drift rather than solver behavior.

## Focused Commands

Proof-owner rebuild:

```sh
make build/test_svd
```

Result:

```text
make: `build/test_svd' is up to date.
```

Focused behavior proof:

```sh
./build/test_svd
```

Result:

```text
Tests run:    114
Tests failed: 0
Tests skipped: 0
Assertions:   2067
ALL TESTS PASSED
```

Ownership guard:

```sh
make svd-helper-guard
```

Result:

```text
svd-helper-guard: required files ok
svd-helper-guard: proof-owner registration ok
svd-helper-guard: helper boundary ok
svd-helper-guard: selected cluster ownership ok
svd-helper-guard: header-only registration ok
svd-helper-guard: passed
```

Guard regression:

```sh
python3 tests/test_svd_helper_guard.py
```

Result:

- Passed.

## Invariant-To-Test Traceability

| Invariant | Focused Evidence |
| --- | --- |
| Function signatures remain callable through the same proof-owner wrappers. | `./build/test_svd` ran the selected wrappers by their existing `RUN_TEST(...)` names; `make svd-helper-guard` verified selected registrations exist exactly once. |
| `tests/test_svd.c` remains the proof owner. | `make build/test_svd` and `./build/test_svd` used the existing proof-owner binary; `make svd-helper-guard` verified Makefile and CMake proof-owner registration. |
| Public API and ABI stay unchanged. | `git diff --name-only -- include src CMakeLists.txt build-metadata/library_sources.txt .github` produced no paths. |
| Fixture dimensions remain unchanged. | Selected focused tests passed after extraction: rank full/deficient/nearly-singular/threshold/dependent-row/null, pseudoinverse diagonal/Moore-Penrose/null/rectangular/minimum-norm, and dense low-rank diagonal/error-bound/errors. |
| Fixture values remain unchanged. | `./build/test_svd` preserved selected expected ranks, pseudoinverse values, Moore-Penrose residuals, minimum-norm solution values, and low-rank Frobenius errors. |
| Sparse insertion cleanup and ownership remain unchanged. | Selected helper-owned tests passed through the same fixture constructors and early-return helper paths; no production allocation path changed. |
| Error propagation remains unchanged. | `test_svd_rank_null`, `test_pinv_null`, and `test_lowrank_errors` passed with existing `SPARSE_ERR_NULL` and `SPARSE_ERR_BADARG` expectations. |
| Numerical tolerances remain unchanged. | `test_svd_rank_nearly_singular`, `test_svd_rank_diagonal_threshold_fixture`, `test_pinv_moore_penrose`, `test_pinv_rectangular`, and `test_lowrank_error_bound` passed with existing tolerances. |
| Dense layout remains column-major where selected tests require it. | `test_pinv_diagonal`, `test_pinv_rectangular`, `test_pinv_underdetermined_minnorm_solution`, `test_lowrank_diagonal`, and `test_lowrank_error_bound` passed after movement. |
| Numerical output remains unchanged. | Focused output retained rank `5`, rank `2`, threshold ranks `3/2/1`, dependent-row SVD/QR rank `2/2`, Moore-Penrose residuals below threshold, minimum-norm solution norm `1.000`, and low-rank error values. |
| Deterministic registration order remains unchanged. | `make svd-helper-guard` verified every selected `RUN_TEST(...)` marker remains in `tests/test_svd.c` exactly once; no registration-order edit was made. |
| Cleanup behavior remains unchanged. | `./build/test_svd` completed all selected tests with no failures; extraction moved code without changing cleanup branches or ownership releases. |
| Helper scope remains SVD-test-local. | `make svd-helper-guard` verified `tests/test_svd_helpers.h` is absent from Makefile, CMake, and `build-metadata/library_sources.txt`; no standalone helper test target exists. |

## Explicitly Observed Selected Tests

The focused `./build/test_svd` run included and passed:

- `test_svd_rank_full`
- `test_svd_rank_deficient`
- `test_svd_rank_nearly_singular`
- `test_svd_rank_diagonal_threshold_fixture`
- `test_svd_qr_rank_dependent_row_fixture`
- `test_svd_rank_null`
- `test_pinv_diagonal`
- `test_pinv_moore_penrose`
- `test_pinv_null`
- `test_pinv_rectangular`
- `test_pinv_underdetermined_minnorm_solution`
- `test_lowrank_diagonal`
- `test_lowrank_error_bound`
- `test_lowrank_errors`

## Untested Breadth And Residuals

Day 10 intentionally does not claim:

- broad SVD correctness beyond the existing `test_svd` suite;
- new public API, ABI, or installed-header behavior;
- performance improvement;
- partial-SVD helper or corpus-surface changes;
- sparse low-rank extended cluster ownership changes;
- external-reference, package-manager, platform, or release readiness.

No focused behavior-preservation residual is known for the selected rank,
pseudoinverse, and dense low-rank extraction cluster.

## Completion

Item 201.5 now has focused behavior-preservation evidence for the selected SVD
helper extraction. The existing test suite covered the extraction-sensitive
behavior, and no feature-expanding regression test was added.
