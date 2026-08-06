# Day 9 Partial-SVD Proof Implementation

## Summary

Day 9 implements the focused compiled proof owner designed on Day 8:
`tests/test_svd_partial_corpus.c`. The new owner exercises the selected
Sprint 140 fixture `partial_svd_clustered_repeated_diag8x6_k3_v1` without
weakening existing `test_svd` coverage.

## Implemented Files

| File | Change |
| --- | --- |
| `tests/test_svd_partial_corpus.c` | New focused test owner for the generated 8x6 clustered/repeated partial-SVD fixture. |
| `Makefile` | Registers `test_svd_partial_corpus` in `TEST_SRCS`. |
| `CMakeLists.txt` | Registers `test_svd_partial_corpus` with `add_sparse_test`. |
| `docs/planning/EPIC_12/SPRINT_140/WORKING_NOTES.md` | Records Day 9 implementation notes and validation results. |

## Proof Coverage

| Test | Covered behavior |
| --- | --- |
| `test_partial_svd_corpus_clustered_repeated_default_success` | Default-budget `sparse_svd_partial` succeeds, publishes factors, preserves dimensions, and matches top-k singular values. |
| `test_partial_svd_corpus_clustered_repeated_projectors` | Left and right top-k projectors match the exact first-three-coordinate subspaces. |
| `test_partial_svd_corpus_clustered_repeated_residuals` | Returned triplets satisfy `A*v ~= sigma*u`, `A^T*u ~= sigma*v`, and U/V orthogonality tolerances. |
| `test_partial_svd_corpus_clustered_repeated_tight_budget_fail_closed` | Tight budget returns `SPARSE_ERR_NOT_CONVERGED` and publishes no partial `sigma`, `U`, or `Vt` arrays. |
| `test_partial_svd_corpus_clustered_repeated_recovery_after_failure` | A fresh default-budget call after a tight-budget failure recovers and passes value/residual checks. |

## Claim Boundary

The new proof owner supports only fixture-local partial-SVD evidence for the
generated 8x6 clustered/repeated diagonal matrix with `k=3`.

It does not support broad partial-SVD correctness, raw singular-vector
identity, broad repeated-spectrum coverage, external-library parity,
performance claims, or partial-result guarantees.

## Validation Plan

Required Day 9 validation:

```sh
make build/test_svd_partial_corpus
./build/test_svd_partial_corpus
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
cmake -S . -B build-cmake
ctest --test-dir build-cmake -N | rg test_svd_partial_corpus
make format
make lint
make test
```

Generated outputs under `build/` and `build-cmake/` remain uncommitted.
