# Sprint 139 Day 9: Proof Owner Implementation

## Purpose

Day 9 implements the focused QR proof owner designed on Day 8. The new owner
proves the selected maintained corpus fixture locally without adding more
coverage to the already-large `tests/test_qr.c`.

## Implementation Summary

Added:

- `tests/test_qr_corpus.c`
- `tf_qr_make_rankdef_6x4_nullspace_v1()` in `tests/test_qr_helpers.h`
- `tf_qr_normalized_matvec_residual()` in `tests/test_qr_helpers.h`
- `test_qr_corpus` registration in `Makefile`
- `test_qr_corpus` registration in `CMakeLists.txt`

The focused test executable covers only
`qr_rank_deficient_6x4_nullspace_v1`.

## Focused Tests

| Test | Evidence |
| --- | --- |
| `test_qr_corpus_rankdef_6x4_fixture_shape` | The C fixture builder emits a 6x4 matrix with 14 stored nonzeros. |
| `test_qr_corpus_rankdef_6x4_rank_and_nullity` | `sparse_qr_factor()` succeeds, `sparse_qr_rank(&qr, 0.0) == 3`, and `sparse_qr_nullspace(&qr, 0.0, NULL, &nullity)` reports nullity `1`. |
| `test_qr_corpus_rankdef_6x4_nullspace_residual` | The solver-produced nullspace basis has nonzero norm and normalized residual at most `1e-10`. |
| `test_qr_corpus_rankdef_6x4_reference_direction` | The deterministic structural null-vector direction `[-1, -1, 0, 1]` has normalized residual at most `1e-12`. |

Local focused output:

```text
=== QR Corpus Proof ===

  [PASS] test_qr_corpus_rankdef_6x4_fixture_shape
  [PASS] test_qr_corpus_rankdef_6x4_rank_and_nullity
    qr_rank_deficient_6x4_nullspace_v1 solver nullspace normalized residual = 2.220e-16
  [PASS] test_qr_corpus_rankdef_6x4_nullspace_residual
    qr_rank_deficient_6x4_nullspace_v1 reference direction normalized residual = 0.000e+00
  [PASS] test_qr_corpus_rankdef_6x4_reference_direction

--- Summary ---
Tests run:    4
Tests failed: 0
Tests skipped: 0
Assertions:   83
ALL TESTS PASSED
```

## Canonical Fixture Correction

The first focused run caught a local implementation mismatch: the initial C
fixture helper copied the Day 8 sketch instead of the canonical Sprint 138
generator entries and produced 15 nonzeros. The helper and Day 8 artifact were
corrected to mirror `scripts/validate_corpus_schema.py` exactly:

```text
(0,0)=1, (0,3)=1
(1,1)=1, (1,3)=1
(2,2)=1
(3,0)=1, (3,1)=1, (3,3)=2
(4,1)=1, (4,2)=1, (4,3)=1
(5,0)=1, (5,2)=1, (5,3)=1
```

This is the 14-nonzero fixture whose columns satisfy `c3 = c0 + c1` and whose
reference null-vector direction is `[-1, -1, 0, 1]`.

## Ownership and Non-Claims

The new proof owner is additive:

- `tests/test_qr.c` remains registered and continues to own broad QR
  factorization, rank, nullspace, Q, economy, sparse-mode, reorder, threshold,
  and external-reference checks.
- `tests/test_qr_solve.c` remains registered and continues to own solve,
  least-squares, rank-deficient residual, and minimum-norm checks.
- `test_qr_corpus` owns only this maintained corpus-backed rank/nullity and
  nullspace residual proof.

The new proof does not claim:

- exact QR basis components;
- basis sign or orientation;
- broad QR correctness;
- global rank-threshold policy;
- broad rank-deficient solve, least-squares, or minimum-norm behavior;
- SuiteSparse, LAPACK, NumPy, SciPy, platform, performance, or
  state-of-the-art parity.

## Validation

Focused validation:

```sh
make format
make build/test_qr_corpus
./build/test_qr_corpus
cmake -S . -B build/qr-corpus-proof
cmake --build build/qr-corpus-proof --target test_qr_corpus
./build/qr-corpus-proof/test_qr_corpus
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports --include-solver-qr
```

Focused result:

- Make `test_qr_corpus`: passed, 4 tests, 0 failures, 0 skips, 83 assertions.
- CMake `test_qr_corpus`: passed, 4 tests, 0 failures, 0 skips, 83
  assertions.
- Opt-in solver QR oracle/report generation: passed.

Required full gate:

```sh
make format && make lint && make test
```

Result:

- `make format`: passed.
- `make lint`: passed.
- `make test`: passed, including registered `test_qr_corpus`.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected QR residual has a focused proof owner. | Complete | `tests/test_qr_corpus.c` is registered in Make and CMake and covers the selected corpus fixture. |
| Test failures identify the selected behavior and fixture clearly. | Complete | Test names and diagnostic prints include `qr_rank_deficient_6x4_nullspace_v1` and separate shape, rank/nullity, solver residual, and reference-direction checks. |
| Existing QR coverage remains present or is explicitly transferred. | Complete | `test_qr` and `test_qr_solve` remain registered; no existing `tests/test_qr.c` case was removed or transferred. |
