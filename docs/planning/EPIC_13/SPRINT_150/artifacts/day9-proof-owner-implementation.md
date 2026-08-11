# Sprint 150 Day 9: Proof-Owner Test Implementation

## Purpose

Implement focused QR corpus proof-owner tests for the Sprint 150 selected
fixture families. The implementation keeps proof ownership in the existing
`tests/test_qr_corpus.c` target and mirrors the executable oracle semantics
from Days 6-7 without widening QR claims.

## Implementation Summary

Extended `tests/test_qr_corpus.c` with focused coverage for the selected
rank-deficient rectangular and underdetermined minimum-norm QR families.

The existing Sprint 139 seed tests for
`qr_rank_deficient_6x4_nullspace_v1` remain in place. Day 9 adds local helpers
and fixture builders instead of adding a new Make/CMake target.

## Added Helpers

Added local helpers for:

- fixture-key-oriented shape and `nnz` assertions;
- rank, nullity, solver-produced nullspace residual, and reference-null-vector
  residual checks;
- projector-distance checks that compare subspaces instead of raw basis
  entries;
- minimum-norm residual, solution norm, and exact-value checks;
- deterministic local construction of the selected underdetermined
  minimum-norm fixtures.

The diagnostics remain fixture-key oriented so a failing assertion identifies
the family and oracle condition under test.

## Rank-Deficient Rectangular Proof Coverage

Added executable proof-owner coverage for:

- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`

Each fixture now has focused tests for:

- exact shape and `nnz`;
- exact rank;
- exact nullity;
- normalized nullspace residual;
- projector/subspace distance against the deterministic reference null
  direction;
- reference null-vector residual as an advisory deterministic sanity check.

The projector checks preserve the Day 6 non-claim against raw nullspace basis
identity, sign, orientation, scale, or column-order parity.

## Underdetermined Minimum-Norm Proof Coverage

Added executable proof-owner coverage for:

- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Each fixture now has focused tests for:

- exact shape and `nnz`;
- `sparse_qr_solve_minnorm()` status;
- residual norm;
- solution norm;
- exact deterministic solution values within the selected tolerance.

The tests remain fixture-local and do not claim broad underdetermined QR,
least-squares, rank-deficient minimum-norm, or inconsistent-system behavior.

## Registration

No build-system registration change was required. The existing focused QR
corpus target remains the proof owner:

- Make target: `build/test_qr_corpus`
- CTest target: `test_qr_corpus`

## Validation

Focused proof-owner validation passed:

```sh
make build/test_qr_corpus
./build/test_qr_corpus
```

Result:

- Tests run: `14`
- Tests failed: `0`
- Assertions: `258`

Corpus and oracle checks passed:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
```

The oracle command regenerated the local oracle/report outputs and the
freshness check completed successfully with the existing advisory
`generated_present_unchecked` warnings for generated-local rows.

Because Day 9 modified `tests/test_qr_corpus.c`, the full C quality gate was
run and passed:

```sh
make format
make lint
make test
```

## Claim Boundary

Day 9 supports only selected fixture-family evidence for the source-controlled
Sprint 150 QR corpus rows. It does not claim:

- broad QR correctness;
- raw QR basis or raw nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad minimum-norm or least-squares behavior;
- rank-deficient minimum-norm recovery;
- inconsistent-system behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

## Day 10 Handoff

Report integration can now use executable proof-owner evidence from
`tests/test_qr_corpus.c` for the selected families. Day 10 should design the
QR report/index rows around these bounded proof-owner tests and the
generated-local oracle outputs without converting local generated rows into
source-controlled freshness claims.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Selected QR families have executable proof-owner coverage. | Complete | `tests/test_qr_corpus.c` covers both selected rank-deficient fixtures and all three selected minimum-norm fixtures. |
| Focused QR tests pass locally. | Complete | `./build/test_qr_corpus` passed with 14 tests, 0 failures, and 258 assertions. |
| Failures produce actionable family/oracle diagnostics. | Complete | Added fixture-key-oriented helpers for shape, rank/nullity, residual, projector distance, status, norm, and value checks. |
| C quality gates pass after code changes. | Complete | `make format`, `make lint`, and `make test` passed. |
