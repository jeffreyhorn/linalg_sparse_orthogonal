# Sprint 138 Day 11 - Optional Data & Skip Semantics

## Purpose

Day 11 implements explicit optional-data skip/defer semantics so unavailable
external data is visible evidence but cannot be counted as numerical pass
evidence.

This implementation does not add optional external data payloads, downloaded
archives, observed source-controlled oracle rows, public claims, or `.c`/`.h`
changes.

## Optional-Data State Model

| State | Report status | Failure class | Meaning |
| --- | --- | --- | --- |
| `available` | Determined by the configured fixture check. | Empty on pass, otherwise row-specific failure. | Optional data is explicitly configured and the numerical check may run. |
| `unavailable` | `skip` | `skip_optional_unavailable` | Optional data was expected but not present or not readable. |
| `disabled` | `skip` | `skip_optional_unavailable` | Optional data is intentionally disabled for default validation. |
| `deferred` | `defer` | `defer_not_implemented` | Optional-data policy or fixture is intentionally not implemented yet. |

Default validation uses `disabled` and must pass without
`SPARSE_CORPUS_OPTIONAL_DATA_DIR`.

## Implemented Optional-Data Row

| Field | Value |
| --- | --- |
| Optional data key | `suitesparse_rank_deficient_qr_subset_v1` |
| Source | SuiteSparse Matrix Collection |
| Expected location | `$SPARSE_CORPUS_OPTIONAL_DATA_DIR/suitesparse_rank_deficient_qr_subset_v1` |
| Availability state | `disabled` |
| Fixture keys | `qr_rank_deficient_external_*` |
| Validation command | `python3 scripts/run_corpus_oracle.py` |
| Skip interpretation | Optional data disabled or unavailable; no QR behavior was proven. |
| Claim boundary | No SuiteSparse parity or external-library parity; no broad corpus completeness claim. |

## Command Behavior

`scripts/run_corpus_oracle.py` now emits optional-data policy rows in addition
to first-lane oracle rows:

| Output | Meaning |
| --- | --- |
| `build/corpus-reports/skips.tsv` | Current optional-data skip/defer rows. |
| `build/corpus-reports/index.tsv` | Includes optional-data `skip` or `defer` rows alongside oracle comparison rows. |

The optional-data row is not emitted as an oracle `pass`. It is report-policy
evidence only.

## False-Pass Guard

`scripts/validate_corpus_schema.py` now checks optional rows so unavailable,
disabled, or deferred rows:

1. require `skip_reason`;
2. require `defer_reason` when state is `deferred`;
3. reject skip interpretation wording that describes pass evidence;
4. require claim-boundary wording that preserves external-parity non-claims.

## Validation Evidence

Day 11 validation used:

```sh
python3 -B scripts/validate_corpus_schema.py
python3 -B scripts/run_corpus_oracle.py
```

The generated `build/corpus-reports/index.tsv` includes a `skip` row for
`suitesparse_rank_deficient_qr_subset_v1`, and
`build/corpus-reports/skips.tsv` records the same optional-data policy row.

## Non-Claims

The optional-data skip/defer implementation does not claim:

- SuiteSparse parity;
- external-library parity;
- broad corpus completeness;
- QR behavior for skipped optional data;
- release readiness;
- package, platform, performance, coverage, or state-of-the-art status.

## Day 11 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Unavailable optional data produces skip/defer evidence only. | Complete | Optional-data rows emit `skip` or `defer` report rows and never oracle `pass` rows. |
| Default validation passes without optional external data. | Complete | The default optional row is `disabled`; `python3 -B scripts/run_corpus_oracle.py` runs without `SPARSE_CORPUS_OPTIONAL_DATA_DIR`. |
| Skip/defer wording preserves corpus and external-parity non-claims. | Complete | Validator checks skip wording and claim-boundary wording; the optional row states no SuiteSparse/external-library parity and no broad corpus completeness. |
