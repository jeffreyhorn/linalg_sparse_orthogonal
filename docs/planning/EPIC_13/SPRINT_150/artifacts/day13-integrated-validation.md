# Sprint 150 Day 13: Integrated Validation

## Purpose

Run the integrated Sprint 150 validation lane across corpus schema checks,
focused QR proof-owner tests, oracle/report generation, report-index
normalization, documentation stale-claim checks, and the full C quality gate.

## Validation Summary

| Check | Result | Evidence |
| --- | --- | --- |
| Corpus schema validation | Passed | `python3 scripts/validate_corpus_schema.py` reported `tests/corpus ok`. |
| Focused QR proof owner | Passed | `make build/test_qr_corpus && ./build/test_qr_corpus` passed with 14 tests, 0 failures, 0 skips, and 258 assertions. |
| QR oracle generation | Passed | `python3 scripts/run_corpus_oracle.py --include-solver-qr` regenerated the QR oracle TSV, normalized report index, skip report, and manifest under `build/`. |
| Python script compile | Passed | `python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py scripts/normalize_report_index.py`. |
| Report-index normalization | Passed | `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` reported `78 rows ok`. |
| Oracle freshness | Passed | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check` reported freshness ok for 28 rows. |
| Stale current-doc search | Passed | Focused searches found no stale current-doc QR row-count or Sprint 139-only wording. |
| Whitespace and diff checks | Passed | Trailing-whitespace scan and `git diff --check` produced no findings. |
| Full C quality gate | Passed | `make format && make lint && make test` completed successfully. |

The oracle freshness check retained the expected advisory
`generated_present_unchecked` warnings for generated-local oracle rows. Those
warnings are acceptable because the rows remain local evidence tied to the
recorded command, commit, branch, platform, compiler, configuration, support
tier, and artifact path.

## Focused QR Proof Details

The focused QR proof owner remains `tests/test_qr_corpus.c`. Day 13 re-ran it
against the selected six-fixture Sprint 139/Sprint 150 QR family:

- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Observed focused proof highlights:

- `qr_rank_deficient_6x4_nullspace_v1` retained the normalized nullspace
  residual proof at `2.220e-16`.
- `qr_rankdef_duplicate_5x4_v1` proved shape `5x4`, `nnz=14`, rank `3`,
  nullity `1`, zero nullspace residual, zero projector distance, and zero
  deterministic reference-vector residual.
- `qr_rankdef_dependent_row_4x3_v1` proved shape `4x3`, `nnz=9`, rank `2`,
  nullity `1`, nullspace residual `4.154e-16`, projector distance
  `1.110e-16`, and zero deterministic reference-vector residual.
- `qr_underdetermined_minnorm_2x4` proved minimum-norm solve residual
  `1.570e-16`, solution norm `1`, and max exact-value error `1.110e-16`.
- `qr_minnorm_3x6_exact_values` proved minimum-norm solve residual
  `2.391e-15`, solution norm `2.89827534923789`, and max exact-value error
  `4.441e-16`.
- `qr_minnorm_5x10_exact_values` proved minimum-norm solve residual
  `1.018e-15`, solution norm `3.3166247903554`, and max exact-value error
  `2.220e-16`.

## Oracle And Report Surface

The maintained QR oracle command:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The generated-local QR report surface remains bounded to:

- `oracle_row_count=26`
- `solver_qr_row_count=23`
- `partial_svd_row_count=0`
- `support_tier=local_only`
- the six selected fixture keys listed above

No generated `build/` outputs are source-controlled. The generated-local
oracle/report rows remain evidence for the local command result only.

## Current-Documentation Checks

Day 13 searched current user, maintainer, corpus, and Sprint 150 planning
surfaces for stale current-claim wording, including old
`solver_qr_row_count=3`, Sprint 139-only QR corpus wording, and obsolete
four-test proof expectations. The search produced no current-doc hits.

Historical Sprint 139 planning artifacts and the Sprint 150 Day 1 baseline
artifact can still contain old counts as historical records; Day 13 did not
rewrite historical evidence.

## Claim Boundary

Day 13 validates only the selected fixture-local QR corpus family. It does not
claim:

- broad QR correctness;
- raw QR basis or raw nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm or least-squares behavior;
- SVD-pseudoinverse global-oracle behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

## Day 14 Handoff

Day 14 should close the sprint by reconciling the plan, working notes, and
artifacts; record any deferred reorder/COLAMD QR work for Sprint 151 or later;
and prepare the retrospective around the completed six-fixture maintained QR
corpus family.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Corpus schema validation passes. | Complete | `validate_corpus_schema.py` reported `tests/corpus ok`. |
| Focused QR proof-owner tests pass. | Complete | `test_qr_corpus` passed with 14 tests and 258 assertions. |
| Oracle/report generation and normalization pass. | Complete | QR oracle generation, `78`-row normalization, and `28`-row freshness checks passed. |
| Current docs contain no stale current QR corpus claims. | Complete | Focused stale-claim search produced no current-doc findings. |
| Required C quality gate passes. | Complete | `make format && make lint && make test` passed. |
