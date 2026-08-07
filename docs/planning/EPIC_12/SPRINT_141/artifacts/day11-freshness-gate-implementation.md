# Day 11 Freshness Gate Implementation

## Purpose

Day 11 implements the normalized report-index freshness gate designed on Day
10. The gate evaluates normalized rows, emits deterministic diagnostics, and
returns nonzero only for strict or explicitly required freshness failures.

The implementation keeps local measurements advisory by default and does not
promote benchmark, coverage, dead-code, package, or generated local rows into
release proof.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added `--check-freshness`, `--strict-generated`, and `--advisory-ok` CLI flags plus freshness evaluation helpers. | Evaluates normalized rows against the Day 10 state/severity model. |
| `tests/test_normalize_report_index.py` | Added freshness tests for missing, required, stale, advisory, skip/defer, and hard-gate failure paths. | Verifies deterministic diagnostics and exit behavior. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day11-freshness-gate-implementation.md` | Added this implementation artifact. | Records behavior, validation, and handoff. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Updated Day 11 notes. | Keeps sprint evidence current. |

## CLI

The generator now supports:

```sh
python3 scripts/normalize_report_index.py --check-freshness
python3 scripts/normalize_report_index.py --check-freshness --family oracle
python3 scripts/normalize_report_index.py --check-freshness --family oracle --require-generated oracle
python3 scripts/normalize_report_index.py --check-freshness --strict-generated
python3 scripts/normalize_report_index.py --check-freshness --advisory-ok
```

`--check` still validates normalized row construction. `--check-freshness`
adds freshness diagnostics and exit behavior. `--require-generated` is reused:
missing generated rows for a required family become `error` diagnostics.

## Diagnostic Format

Diagnostics are deterministic one-line records:

```text
freshness: <severity>: <row_id>: <state>: <reason>
```

Examples covered by tests:

- missing optional oracle rows without requirement produce `warning`;
- missing required oracle rows produce `error`;
- stale oracle rows produce `warning` by default and `error` under
  `--strict-generated`;
- stale benchmark rows produce `advisory`;
- runtime/backend governance rows produce `defer`;
- sentinel hard-gate rows with generated `status=fail` produce `error`.

## Implemented States

| State | Implementation behavior |
| --- | --- |
| `source_controlled` | Advisory diagnostic governed by schema and Git review. |
| `generated_present_unchecked` | Warning for strict generated families, advisory for local measurement families. |
| `fresh` | Generated row has comparable current source commit for advisory families. |
| `stale` | Generated row source commit differs from current HEAD. Severity follows family policy and strict/required flags. |
| `not_generated` | Warning/advisory by default; error when the family is required. |
| `optional_data_skip` | Emits `skip` with row reason. |
| `deferred` | Emits `defer` with row handoff reason. |
| `unsupported` | Emits `unsupported` unless the caller requires that family. |

## Policy Fallback

Generated native rows often replace the source-controlled contract
configuration with row-specific details. Day 11 adds a freshness-policy
fallback by report family and row meaning so generated oracle, sentinel,
guardrail, benchmark, coverage, dead-code, package, optional-data, and
deferred rows classify consistently even when their row-specific
configuration does not include `freshness_policy=...`.

## Test Coverage

`tests/test_normalize_report_index.py` now verifies:

- missing oracle rows warn without `--require-generated`;
- missing oracle rows error with `--require-generated oracle`;
- runtime/backend rows emit `defer`;
- generated oracle rows with stale `source_commit` warn by default;
- stale generated oracle rows error under `--strict-generated`;
- generated benchmark rows with stale source commit remain advisory;
- generated sentinel hard-gate failure rows error;
- existing index-generation tests for corpus, oracle, runtime, quality, and
  package families continue to pass.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family oracle --no-generated --require-generated oracle --check-freshness
```

Results:

- focused normalized-index tests passed;
- required oracle freshness command returned the expected nonzero errors for
  missing generated oracle rows.

## Day 12 Handoff

Day 12 should update maintainer, benchmark, corpus, package, and report docs
with the new `--check-freshness` workflow, diagnostic interpretation, and
non-claim boundaries. Day 13 can decide whether to add a Make target after
documentation and validation settle.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Stale rows are detected deterministically. | Complete | Stale source-commit tests assert warning/default and error/strict behavior. |
| Check behavior matches the Day 10 severity model. | Complete | Tests cover warning, error, advisory, skip/defer, and hard-gate failure paths. |
| Generated measurement outputs remain local unless explicitly source-owned. | Complete | Benchmark freshness remains advisory; package rows remain source-controlled proof-owner rows. |
