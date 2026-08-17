# Day 6 Expected Rows And Dependency Semantics

## Summary

Day 6 ties the selected `partial-svd-diag6-k2` comparison family to
source-controlled report metadata and focused dependency assertions. The new
metadata row makes the generated study traceable before the normalizer promotes
it into selected comparison freshness.

## Files Changed

| File | Change |
| --- | --- |
| `tests/corpus/manifests/report_families.tsv` | Added the `comparison/partial_svd_diag6_k2` contract row. |
| `tests/test_run_external_comparison.py` | Added metadata assertions for comparison report-family rows and required helper dependency assertions. |
| `docs/planning/EPIC_14/SPRINT_161/WORKING_NOTES.md` | Added Day 6 log entry. |

## Report-Family Metadata

The new report-family row is:

| Field | Value |
| --- | --- |
| `report_family` | `comparison` |
| `subfamily` | `partial_svd_diag6_k2` |
| `row_meaning` | `external_process_dense_reference_comparison` |
| `row_origin` | `generated_local` |
| `status` | `unknown` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` |
| `artifact_pattern` | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| `owner` | `Report maintainer` |
| `introduced_in` | `Sprint 161 Day 6` |

The claim scope remains one local fixture-level partial-SVD diagonal top-k
comparison for `partial_svd_diag6_k2` against the selected source-controlled
dense singular-value reference helper.

## Non-Claim Boundary

The metadata row preserves these boundaries:

- no broad SVD correctness;
- no broad partial-SVD correctness;
- no raw singular-vector identity;
- no vector sign or orientation identity;
- no repeated-spectrum ordering claim;
- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- no hosted CI proof;
- no release proof;
- no platform portability proof;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

## Dependency Semantics

The focused runner test now checks that each target emits a passing required
source-controlled helper dependency row:

| Target | Required Helper |
| --- | --- |
| `qr-minnorm` | `tests/qr_external_dense_reference.py` |
| `qr-compatible-ls` | `tests/qr_external_dense_reference.py` |
| `partial-svd-diag6-k2` | `tests/svd_external_dense_reference.py` |

The optional package rows remain:

| Dependency | Status | Interpretation |
| --- | --- | --- |
| `numpy` | `defer` | Optional package baseline is not selected; not pass evidence. |
| `scipy` | `defer` | Optional package baseline is not selected; not pass evidence. |

Selected comparison freshness must not treat optional dependency defers as
passing generated evidence.

## Fixture And Row-ID Alignment

The selected family is aligned across:

| Surface | Value |
| --- | --- |
| CLI target | `partial-svd-diag6-k2` |
| Fixture key | `partial_svd_diag6_k2` |
| Report subfamily | `partial_svd_diag6_k2` |
| Artifact path | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| Selected row count | `10` |
| Support tier | `local_only` |

The selected row IDs remain those locked in the Day 3 metric contract and
implemented on Day 5.

## Validation

Commands run:

```sh
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/run_external_comparison.py tests/test_run_external_comparison.py scripts/validate_corpus_schema.py
python3 tests/test_run_external_comparison.py
```

Results:

- Corpus schema validation passed for `tests/corpus`.
- Python compile checks passed.
- Focused external-comparison runner tests passed.

No `.c` or `.h` files were modified.

## Day 7 Handoff

Day 7 should design the remaining proof-owner tests before normalizer
integration. The current likely test set is:

- keep the expanded runner test as the target dispatch and metadata owner;
- add normalizer tests for complete, missing, unexpected, duplicate, stale,
  fail, defer, and malformed selected partial-SVD comparison rows;
- avoid C proof-owner changes unless later implementation touches solver
  behavior or fixture helpers.
