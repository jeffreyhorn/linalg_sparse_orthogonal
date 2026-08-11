# Sprint 151 Day 7: Oracle Data Implementation

## Purpose

Implement deterministic partial-SVD oracle-generation inputs for the Sprint
151 metadata batch and verify that generated-local oracle/report rows match
the Day 4 comparison contract.

Day 7 keeps this lane generated-reference and local-only. It does not promote
solver-backed hosted-platform evidence or broad partial-SVD correctness.

## Implemented Oracle Changes

| File | Change |
| --- | --- |
| `scripts/run_corpus_oracle.py` | Replaced the single-fixture partial-SVD generated-reference path with a maintained partial-SVD fixture map covering the Sprint 140 fixture plus the three Sprint 151 fixtures. |
| `scripts/run_corpus_oracle.py` | Added generated observations for rank-deficient projector, sparse low-rank output, and fail-closed convergence expected rows. |
| `scripts/run_corpus_oracle.py` | Added per-fixture partial-SVD configuration strings containing fixture key, generator hash, tolerance policy, and generated-reference proof owner. |
| `scripts/run_corpus_oracle.py` | Extended `comparison_kind=value` parsing to support `selected_values`, parallel to existing `solution_values` vector handling. |

## Maintained Partial-SVD Oracle Fixtures

| Fixture Key | Generated Rows | Oracle Scope |
| --- | ---: | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | 8 | Existing Sprint 140 clustered/repeated top-k, subspace, residual, orthogonality, and fail-closed rows. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | 7 | Rank-deficient rectangular top-2 values, rank, left/right projectors, residuals, orthogonality, and default success. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | 6 | Sparse-output status, shape, retained nnz, selected values, dense Frobenius error, and sparse-vs-dense Frobenius difference. |
| `partial_svd_fail_closed_diag6_k2_v1` | 5 | Tight-budget non-convergence, no partial arrays, recovery success, default top-2 values, and default residuals. |

Total maintained partial-SVD generated-reference rows: `26`.

## Selected-Values Comparator

The sparse-output expected row:

```text
partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values
```

now compares:

```text
selected_values=8,4,2,0;max_abs_error=0
```

against:

```text
selected_values=8,4,2,0
```

with absolute tolerance `1e-10`. The comparator validates vector length,
element-wise maximum absolute error, and optional reported `max_abs_error`.

This extension is intentionally narrow. It does not add broad sparse-output
correctness, storage optimality, drop-tolerance optimality, or performance
claims.

## Generated Output Check

Command run:

```sh
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

Generated files:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

Generated oracle summary:

| Metric | Value |
| --- | ---: |
| Total oracle rows | 29 |
| Partial-SVD oracle rows | 26 |
| Partial-SVD fixtures | 4 |
| Comparison statuses | `pass` only |

The three non-partial-SVD rows are the existing generated-reference first-lane
QR rows emitted by the base corpus oracle command.

## Validation

Commands run:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py
git diff --check
```

Results:

- corpus schema validation passed;
- report index normalization passed with `105` rows;
- oracle freshness check exited successfully with `31` rows and expected
  `generated_present_unchecked` warnings for generated-local rows;
- Python syntax compilation passed;
- whitespace check passed.

No `.c` or `.h` files changed on Day 7, so the C quality gate was not
required.

## Claim Boundaries

Generated Day 7 rows are local-only generated-reference evidence for named
fixtures, commands, source revision, platform, configuration, expected rows,
and generator hashes.

They do not claim:

- solver-backed hosted CI proof;
- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness;
- storage or drop-tolerance optimality;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.

## Day 8 Handoff

Day 8 should design focused proof-owner tests that map each selected
expected-result row to executable assertions. The design should decide whether
to extend `tests/test_svd_partial_corpus.c` directly or add small shared
helpers before Day 9 implementation.
