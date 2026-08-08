# Day 8 Benchmark, Sentinel, And Guardrail Index Integration

## Purpose

Day 8 extends the normalized report index to local runtime report families:
canonical benchmark reports, performance sentinels, and large-matrix
guardrails. These rows are useful maintainer evidence, but they remain scoped
to local command, platform, compiler, build mode, backend, and generated
artifact context.

The implementation does not run benchmark targets and does not commit
generated report output.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added benchmark, sentinel, and guardrail generated-row parsers. | Preserves native row IDs and local measurement/guardrail semantics when generated reports exist. |
| `tests/test_normalize_report_index.py` | Added synthetic runtime-report fixture coverage. | Verifies threshold-free benchmark rows, sentinel hard-gate rows, sentinel advisory rows, reviewed guardrail rows, and supplemental skip rows. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day8-benchmark-sentinel-guardrail-index-integration.md` | Added this integration artifact. | Records Day 8 behavior, validation, and handoff. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Updated Day 8 notes. | Keeps sprint evidence current. |

## Row Mapping

| Input | Normalized row behavior |
| --- | --- |
| `build/bench-reports/canonical/index.tsv` | Emits `benchmark_<artifact>_<index-artifact>_v1` rows with `status=advisory`, command, report label, platform, compiler, build mode, thread count, and threshold-free non-claims. |
| `build/bench-reports/sentinels/sentinels.tsv` with `claim_boundary=local_wall_gate` | Emits `sentinel_*` rows under `row_meaning=sentinel_hard_gate`, preserving `pass`, `fail`, or `skip` status from the generated row. |
| `build/bench-reports/sentinels/sentinels.tsv` with other claim boundaries | Emits `sentinel_*` rows under `row_meaning=sentinel_advisory_measurement`, mapping `report` to `status=advisory`. |
| `build/bench-reports/large-matrix-guardrails/index.tsv` | Emits `guardrail_<lane_id>_<index-artifact>_v1` rows, preserving reviewed `pass` rows and supplemental `skip` or advisory report rows. |

All generated runtime rows use
`freshness_status=generated_present_unchecked` until Sprint 141 Day 10/11
freshness gates define current/stale behavior.

## Claim Boundaries

- Canonical benchmark rows remain threshold-free local measurements.
- Sentinel hard-gate rows preserve the existing wall-check status, but do not
  broaden into portable performance proof.
- Sentinel advisory rows preserve backend request, selected backend, fallback,
  dense-kernel, and panel-solver context where present.
- Guardrail reviewed rows remain bounded structural or CSV-shape evidence.
- Guardrail supplemental rows remain opt-in report context or explicit skips.
- Runtime/backend policy closure remains deferred to Sprint 142.

## Test Coverage

The focused test suite now creates synthetic generated report fixtures under a
temporary `build` root:

- one canonical benchmark index row;
- one sentinel S5 hard-gate pass row;
- one sentinel S2 threshold-free report row;
- one reviewed guardrail pass row;
- one supplemental guardrail skip row.

The test verifies normalized family grouping, status mapping, native row IDs,
freshness status, and non-claim boundaries without running benchmark binaries.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
python3 scripts/normalize_report_index.py --check
python3 scripts/validate_corpus_schema.py
```

Results:

- focused normalized-index tests passed;
- deterministic source-controlled check reported `42` rows;
- default generated-discovery check reported `54` rows in the current local
  worktree;
- corpus schema validation passed.

## Day 9 Handoff

Day 9 should apply the same normalized-row pattern to coverage, dead-code,
package, install, CMake package, and pkg-config report families. The main
remaining distinction is that package/install rows need static-first
source-controlled proof-owner semantics, while coverage and dead-code remain
local/advisory unless a reviewed gate promotes them.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Benchmark and sentinel rows are discoverable through the normalized index. | Complete | Synthetic benchmark and sentinel generated rows are parsed and normalized by `tests/test_normalize_report_index.py`. |
| Platform/runtime-specific rows have explicit support/freshness semantics. | Complete | Runtime rows preserve platform/compiler/build/backend context and use `generated_present_unchecked` freshness. |
| Sprint 142 handoff captures rows that need deeper runtime governance. | Complete | Runtime/backend governance remains a validated `status=defer` row from Day 5/6, and Day 8 does not resolve backend policy. |
