# Day 9 Normalizer Semantics Audit

## Scope

Day 9 audits `scripts/normalize_report_index.py` behavior for the selected
hosted Sprint 159 promotion path:

- selected QR and partial-SVD oracle rows from
  `make report-index-oracle-freshness`;
- selected QR minimum-norm comparison rows from
  `make report-index-comparison-freshness`;
- advisory and local-only rows that must remain distinct from reviewed hosted
  pass evidence.

No implementation changes are made in this artifact. Day 10 should use this
audit to tighten normalizer behavior and focused tests.

## Current Row-State Behavior Map

| State or condition | Current selected oracle behavior | Current selected comparison behavior | Interpretation |
| --- | --- | --- | --- |
| Required generated family missing | `--require-generated oracle --check-freshness` exits nonzero with `freshness: error` and remediation pointing to `make report-index-oracle-freshness`. | `--require-generated comparison --check-freshness` exits nonzero with `freshness: error` and remediation pointing to `make report-index-comparison-freshness`. | Correct for hosted reviewed rows: missing selected evidence cannot pass. |
| Generated row stale by `source_commit` | Required oracle rows become `freshness: error` with recorded/current commit diagnostics and artifact remediation. | Required comparison rows become `freshness: error` with recorded/current commit diagnostics and artifact remediation. | Correct for hosted reviewed rows: stale selected evidence cannot pass. |
| Generated row reports `fail` | Required oracle rows become `freshness: error` with fixture key and artifact remediation. | Required comparison rows become `freshness: error` with artifact remediation. | Correct for selected promoted rows. |
| Oracle selected row-count mismatch | `selected_oracle_policy_diagnostics()` emits `oracle_selected_row_count` error when total or solver-family counts differ from `52` and `qr=23`, `partial_svd=26`, `unknown=3`. | Not applicable. | Correct for selected oracle hosted promotion. |
| Oracle selected solver-family missing | Emits `oracle_selected_solver_families` error. | Not applicable. | Correct; prevents a family from silently disappearing. |
| Oracle selected fixture key missing | Emits `oracle_selected_fixture_keys` error against the selected fixture-key allowlist. | Not applicable. | Correct; prevents fixture substitution from looking complete. |
| Comparison selected row-set mismatch | Not applicable. | `selected_comparison_policy_diagnostics()` emits `comparison_selected_rows` error for missing, duplicate, unexpected, or count-mismatched selected IDs. | Correct; selected hosted comparison proof is narrow and explicit. |
| Comparison selected non-pass status | Not applicable. | Emits `comparison_selected_status` error when a selected row status is not `pass`. | Correct; selected comparison rows cannot pass as skip/defer/fail. |
| Comparison selected skip/defer status | Not applicable. | Current code appends a `freshness: defer: comparison_optional_rows` diagnostic for skip/defer rows and the non-pass check also makes selected skip/defer rows errors. | Mostly correct, but diagnostic wording says `optional_rows` even though selected rows are not optional. |
| Source-controlled rows | Advisory: governed by schema and Git review. | Advisory: governed by schema and Git review. | Correct; metadata rows are not generated pass evidence. |
| Local-only advisory generated rows absent | Advisory or warning depending on freshness policy and required-family selection. | Advisory or warning depending on freshness policy and required-family selection. | Correct when unselected; dangerous only if workflow wording implies hosted proof. |
| Optional dependency defers | Not part of oracle selected proof. | NumPy/SciPy dependency rows defer in comparison dependency status, but selected comparison rows still pass against the maintained project and Python baselines. | Correct if docs keep optional defers as context only. |

## Ambiguities And Gaps

| Gap | Current observation | Risk |
| --- | --- | --- |
| Oracle generated-present warning on passing selected gate | `make report-index-oracle-freshness` can pass while emitting warnings such as `generated row exists but strict freshness comparison is pending` for generated oracle rows whose commit matches current HEAD. | Reviewers may read a passing selected hosted gate as only advisory or partially unchecked even though selected oracle policy diagnostics already enforce row counts, solver families, fixture keys, stale commits, and failures. |
| Selected oracle rows still carry `local_only` support tier in generated artifacts | Day 8 summary prints `support_tier=local_only` from generated manifest metadata. | CI job name says reviewed hosted freshness, but row metadata still says local-only. This is acceptable only if Sprint 159 explicitly treats hosted execution as reviewed evidence while deferring source metadata promotion. |
| Comparison skip/defer diagnostic wording | `comparison_optional_rows` is used for skip/defer selected rows, while selected row non-pass behavior is an error. | The word `optional` can blur selected proof rows with optional NumPy/SciPy dependency rows. |
| Comparison selected policy test coverage | Existing tests heavily cover selected oracle row-count, missing, stale, failed, solver-family, and fixture-key cases. Comparison selected-row set and non-pass behavior are present in code but need direct focused tests. | A future comparison row-id change, duplicate row, or selected defer could regress without an obvious unit-level guard. |
| Broad `report_index/missing_generated` row remains easy to overinterpret | The row makes absence explicit and is advisory/local-only. | If uploaded or summarized in the hosted job, it could look like reviewed completeness evidence. Day 8 correctly avoids uploading broad normalized index output. |

## Promoted-Row Semantics Draft

Selected hosted oracle and comparison rows should use these semantics:

| Semantic rule | Required behavior |
| --- | --- |
| Missing selected artifacts | Fail the selected command and hosted job. |
| Missing selected row | Fail with row-set, fixture-key, or selected-count diagnostics. |
| Duplicate or unexpected selected row | Fail with explicit row-set diagnostics. |
| Stale selected row | Fail with recorded/current commit, artifact path, and remediation command. |
| Selected row status `fail`, `skip`, `defer`, `unknown`, or `partial` | Fail; selected hosted rows are pass evidence only when `status=pass`. |
| Optional external dependency unavailable | Remain context only; do not fail if selected maintained comparison rows pass and the optional dependency row is not selected pass evidence. |
| Source-controlled contract rows | Advisory; never counted as generated pass evidence. |
| Local-only unselected generated families | Advisory or warning only; never uploaded or summarized as promoted hosted proof. |
| Passing selected hosted gate | Emit concise selected-row summaries without generic warning wording for rows that have current-commit generated outputs and pass selected policy checks. |

## Test And Fixture Update List

Day 10 should add focused normalizer coverage for the promoted semantics:

1. Add a selected comparison fixture writer that can produce the six selected
   row IDs with current commit metadata.
2. Test selected comparison required freshness passes with exactly the six
   expected selected IDs.
3. Test selected comparison required freshness rejects a missing selected row
   with `comparison_selected_rows` and `row_set_mismatch`.
4. Test selected comparison required freshness rejects duplicate and
   unexpected selected IDs.
5. Test selected comparison required freshness rejects stale selected rows.
6. Test selected comparison required freshness rejects `skip` or `defer` on a
   selected row, with wording that does not call selected rows optional.
7. Adjust selected oracle passing diagnostics so current-commit selected rows
   do not emit generic strict-freshness warning wording on a successful
   required gate.
8. Preserve advisory/local-only tests proving coverage, package,
   source-controlled, and broad report-index rows remain non-promoted.

## Day 10 Implementation Guidance

The safest Day 10 implementation is narrowly scoped to the normalizer and
script tests:

- keep workflow commands unchanged;
- keep broad normalized report-index output out of hosted artifacts;
- add comparison selected-row tests before changing diagnostics;
- change wording/severity only for selected current-commit generated rows
  under required oracle/comparison freshness checks;
- keep stale, missing, failed, row-count, row-set, and fixture-key failures
  hard errors for selected hosted rows.

## Completion Check

- Current selected row states are understood before implementation.
- Missing and stale selected rows already fail required freshness gates.
- Failing selected rows already fail required freshness gates.
- Advisory/local-only source rows remain distinct from hosted pass evidence.
- Day 10 has a concrete test and fixture list for closing the remaining
  ambiguity.
