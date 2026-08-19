# Sprint 169 Day 10: Documentation Indexing Implementation

## Purpose

Day 10 implements the Day 9 documentation-indexing design. The change makes
selected performance evidence easier to find from the top-level README while
keeping generated reports under ignored `build/` paths and preserving the
boundary between hosted-selected freshness, local sentinel governance, and
threshold-free publication rows.

## Implementation Summary

| Area | Change |
| --- | --- |
| `README.md` first-use boundary | Updated `make performance-sentinels` wording to include S5 wall-check and S6 selected `bench_refactor_csc` local smoke ceiling. |
| `README.md` first-use boundary | Added a compact selected performance evidence path table linking freshness, local smoke-gate, and normalized report-index workflows to detailed docs. |
| `README.md` command list | Updated the `make performance-sentinels` comment to describe S5/S6 hard gates plus S2/S3 threshold-free context. |
| `README.md` benchmark summary | Added `make bench-canonical-report-freshness` as the selected row freshness check and linked generated row interpretation to the benchmark report-index handoff. |
| `benchmarks/README.md` report-index handoff | Added direct instructions for finding the selected canonical `bench_refactor_csc` row and the S6 sentinel row. |
| `docs/maintainer_guide.md` normalized index workflow | Documented that `make bench-canonical-report-freshness` remains authoritative for selected performance freshness and normalized report-index output is secondary navigation. |

## Selected Evidence Path Now Exposed

The README now exposes this path:

| Need | Start here | Detailed interpretation |
| --- | --- | --- |
| Selected hosted/local freshness | `make bench-canonical-report-freshness` | `benchmarks/README.md#report-index-handoff` |
| Local selected regression smoke gate | `make performance-sentinels` | `benchmarks/README.md#report-index-handoff` |
| Cross-report navigation | `python3 scripts/normalize_report_index.py --check-freshness` | `docs/maintainer_guide.md#normalized-report-index-workflow` |

The benchmark docs then identify the exact generated rows:

- selected canonical row:
  `artifact=bench_refactor_csc`,
  `relative_path=bench_refactor_csc.csv`,
  `fixture_or_workload=nos4.mtx`;
- local S6 row:
  `sentinel_id=S6`,
  `matrix_or_fixture=nos4.mtx`,
  `metric=refactor_csc_ms`.

## Generated-Output Policy

The implementation keeps generated report output local:

- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`;
- `build/bench-reports/sentinels/sentinels.tsv`;
- `build/bench-reports/sentinels/manifest.txt`;
- `build/report-index/normalized-index.tsv`.

The README now tells readers to regenerate generated report artifacts before
interpreting rows and to keep interpretation within the recorded fixture,
command, branch, build, and machine context.

## Claim Boundary

The updated docs preserve these distinctions:

- selected canonical publication rows remain threshold-free;
- hosted-selected evidence is freshness and methodology validation only;
- S6 is a local selected-lane smoke ceiling only;
- normalized report-index output is navigation/freshness context, not a
  replacement for focused checks;
- generated local rows are not portable performance, external-library parity,
  broad platform, package/ABI, release, or state-of-the-art evidence.

## Validation

Day 10 changed documentation and planning artifacts only. No `.c` or `.h`
files were modified, so the full C quality gate is not required for this day.

Validation run:

```sh
rg -n "S6|selected performance evidence path|bench-canonical-report-freshness|performance-sentinels|hosted_selected_threshold_free|state-of-the-art|portable performance|performance guarantee|portable speed|release benchmark proof" \
  README.md benchmarks/README.md docs/maintainer_guide.md
git diff --check
```

Results:

- targeted claim scan found only scoped selected-performance, sentinel, and
  non-claim wording;
- `git diff --check` passed.

## Day 10 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected performance evidence is easier to find. | Complete | README now includes a selected performance evidence path table and benchmark-doc links. |
| Docs distinguish hosted evidence from local/generated output. | Complete | README and benchmark docs separate hosted-selected freshness from local S6 smoke-gate rows and ignored generated artifacts. |
| Claim scan finds no unsupported performance broadening. | Complete | Targeted scan found only scoped non-claim wording and selected/local evidence references. |
