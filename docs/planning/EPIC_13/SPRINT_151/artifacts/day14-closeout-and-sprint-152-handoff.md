# Sprint 151 Day 14 Closeout And Sprint 152 Handoff

## Closeout Summary

Sprint 151 expands the maintained partial-SVD corpus from the single Sprint
140 clustered/repeated fixture into a four-fixture, generated-local evidence
surface with source-controlled metadata, expected-result rows, focused proof
owner tests, oracle/report rows, and bounded documentation.

## Completed Partial-SVD Families

| Fixture | Source-Controlled Rows | Generated Oracle Rows | Closure |
| --- | --- | ---: | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Existing Sprint 140 fixture, generator, and expected-result rows | 8 | Retained as the clustered/repeated-spectrum baseline. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Added fixture, generator, and expected-result rows | 7 | Closed rank-deficient rectangular range-projector evidence with singular values, rank, projectors, residuals, and orthogonality. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | Added fixture, generator, and expected-result rows | 6 | Closed sparse low-rank output evidence with status, shape, retained entries, selected values, dense error, and sparse-vs-dense consistency. |
| `partial_svd_fail_closed_diag6_k2_v1` | Added fixture, generator, and expected-result rows | 5 | Closed non-repeated fail-closed convergence evidence with tight-budget non-convergence, no partial arrays, default recovery, values, and residuals. |

The maintained partial-SVD generated-local oracle surface is now `26` rows.
The combined corpus oracle command emits `29` generated rows when the QR rows
are included.

## Validation Baseline

Day 13 is the full validation baseline for Sprint 151:

- `python3 scripts/validate_corpus_schema.py` passed.
- `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus`
  passed with `10` tests and `247` assertions.
- `make build/test_svd && ./build/test_svd` passed with `114` tests and
  `2067` assertions.
- `python3 tests/test_normalize_report_index.py` passed.
- `python3 scripts/run_corpus_oracle.py --include-partial-svd` refreshed
  generated-local oracle/report outputs.
- `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`
  passed with `105` rows.
- `python3 scripts/normalize_report_index.py --family oracle --check-freshness`
  passed with `31` oracle-family rows and expected advisory
  `generated_present_unchecked` warnings.
- `make format && make lint && make test` passed because the branch includes
  C test changes.

Day 14 repeats the final closeout checks and records whether the branch remains
clean except for intentional Sprint 151 changes.

## Claim Boundary

Sprint 151 supports only selected maintained fixture-family evidence for the
four partial-SVD fixtures listed above. It does not claim broad partial-SVD
correctness, raw singular-vector identity, sign/orientation/phase parity,
arbitrary basis ordering, broad sparse-output optimality, convergence-rate
guarantees, portable iteration counts, external-library parity, hosted CI
proof, package/ABI support, performance, or state-of-the-art status.

## Residuals

| Residual | Owner Candidate | Rationale |
| --- | --- | --- |
| Generated-local oracle rows still produce advisory `generated_present_unchecked` freshness warnings. | Sprint 152 | Sprint 152 is explicitly scoped to generated report freshness publication and policy decisions. |
| Strict generated freshness is not yet a promoted CI gate for partial-SVD oracle rows. | Sprint 152 | The current row set is stable enough to evaluate which generated families should become required and which remain advisory. |
| External dense-reference partial-SVD fixtures remain deferred. | Future corpus sprint | Optional data provenance, Windows skip behavior, and broad parity wording would widen Sprint 151 beyond the selected local fixture families. |
| Additional repeated-spectrum partial-SVD families remain deferred. | Future numerical corpus sprint | Sprint 140 already owns the strongest current clustered/repeated seed; more families should be added only with distinct claim value. |
| Broad sparse-output/drop-tolerance optimality remains unclaimed. | Future algorithm sprint | Sprint 151 covers deterministic selected sparse-output behavior, not optimality across drop tolerances or matrix families. |

## Sprint 152 Handoff

Sprint 152 should begin from these concrete inputs:

- Source-controlled Sprint 151 partial-SVD fixture, generator, and expected
  rows are available under `tests/corpus/`.
- Focused proof-owner coverage is in `tests/test_svd_partial_corpus.c`.
- Report-index expectations for Sprint 151 generated partial-SVD rows are in
  `tests/test_normalize_report_index.py`.
- The generated command is
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`.
- The normalized report-index check is
  `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`.
- The current oracle freshness check is
  `python3 scripts/normalize_report_index.py --family oracle --check-freshness`.

Recommended Sprint 152 first decisions:

1. Decide whether partial-SVD generated oracle rows should become required via
   `--require-generated` or remain advisory generated-local evidence.
2. Decide which generated freshness failures should block local closeout,
   hosted CI, both, or neither.
3. Stabilize generated report metadata fields before promoting strict
   freshness: command, commit, branch, platform, compiler, configuration,
   support tier, artifact path, row count, and failure message.
4. Keep generated-local rows local-only unless hosted CI and artifact policy
   explicitly promote them.

## Final Checklist

- [x] Sprint 151 artifacts created for Days 1-14.
- [x] Working notes updated through Day 14.
- [x] Completed families, validation, residuals, and Sprint 152 handoff are
  recorded.
- [x] Generated-report evidence boundary is explicit.
- [x] Final closeout commands are recorded in `WORKING_NOTES.md`.
