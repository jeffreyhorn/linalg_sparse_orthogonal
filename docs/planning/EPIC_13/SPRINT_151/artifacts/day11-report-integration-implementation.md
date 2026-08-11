# Sprint 151 Day 11: Report Integration Implementation

## Purpose

Implement the Day 10 report-integration design by making the Sprint 151
partial-SVD normalized report-index expectations executable.

Day 11 keeps generated-local partial-SVD report rows local-only. It does not
promote those rows to hosted CI proof, release proof, platform support,
external-library parity, performance evidence, package/ABI support, or
state-of-the-art support.

## Implemented Changes

| File | Change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Added `SPRINT151_PARTIAL_SVD_ROW_COUNTS` for the selected fixtures. |
| `tests/test_normalize_report_index.py` | Added a `generated_oracle_rows()` helper to keep generated oracle assertions focused. |
| `tests/test_normalize_report_index.py` | Extended generated oracle preservation tests to assert all three Sprint 151 partial-SVD fixture families. |
| `tests/test_normalize_report_index.py` | Added strict freshness coverage for stale Sprint 151 partial-SVD generated oracle rows. |

No report-family contract split was required. The current
`oracle/solver_backed` contract remains the correct normalized destination
because the generated oracle rows carry `solver_family=partial_svd` while also
recording `proof_owner=generated_partial_svd_reference` and
`solver_execution=none` in configuration.

## Normalized Row Coverage Now Tested

| Fixture | Expected Generated Oracle Rows | Tested Fields |
| --- | ---: | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | 7 | `oracle/solver_backed`, generated-local origin, `local_only`, pass status, fixture key, partial-SVD solver family, proof owner, no solver execution, non-claims. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | 6 | `oracle/solver_backed`, generated-local origin, `local_only`, pass status, fixture key, partial-SVD solver family, proof owner, no solver execution, non-claims. |
| `partial_svd_fail_closed_diag6_k2_v1` | 5 | `oracle/solver_backed`, generated-local origin, `local_only`, pass status, fixture key, partial-SVD solver family, proof owner, no solver execution, non-claims. |

The generated oracle preservation test still verifies the older Sprint 140
clustered/repeated row, then additionally proves the selected Sprint 151
families are visible in the normalized index.

## Freshness Behavior Now Tested

The new stale-fixture test:

1. Generates partial-SVD oracle rows into an isolated temporary `build/` tree.
2. Rewrites the generated oracle artifact `source_commit` to `oldcommit`.
3. Normalizes the oracle family and confirms all seven rank-deficient
   rectangular fixture rows are present with the stale commit.
4. Runs default oracle freshness and verifies stale generated rows produce
   warnings.
5. Runs `--strict-generated --check-freshness` and verifies stale generated
   oracle rows produce errors.

This matches the Day 10 rule: generated partial-SVD oracle rows are stricter
than advisory benchmark/report rows because they compare maintained expected
values, but they still remain local-only generated evidence.

## Generated Output Cleanup

The normalizer tests generate oracle outputs in temporary build roots for the
new assertions. The maintained local command also rewrites ignored generated
outputs under `build/corpus/` and `build/corpus-reports/`. These files remain
ignored and are not source-controlled deliverables.

## Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 -m py_compile scripts/run_corpus_oracle.py scripts/normalize_report_index.py scripts/validate_corpus_schema.py tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 scripts/normalize_report_index.py --family oracle --strict-generated --check-freshness
git diff --check
```

Results:

- normalized report-index tests passed;
- Python syntax compilation passed;
- corpus schema validation passed;
- partial-SVD oracle/report generation passed;
- normalized corpus/oracle report-index check passed with `105` rows;
- oracle freshness checks passed with `31` rows while preserving expected
  strict-oracle warnings for generated-present rows whose full freshness
  comparison remains pending;
- whitespace check passed.

No `.c` or `.h` files changed on Day 11, so the C quality gate was not
required.

## Claim Boundaries

Day 11 does not claim:

- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness;
- low-rank storage or drop-tolerance optimality;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- hosted CI proof;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.
