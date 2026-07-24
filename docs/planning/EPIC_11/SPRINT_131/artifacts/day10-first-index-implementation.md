# Sprint 131 Day 10 - First Index Implementation

## Purpose

Day 10 re-checks the selected first generated report/index candidate against
the Day 8 coverage architecture and Day 9 dead-code/guardrail architecture,
then either implements it or records a concrete deferral.

## Decision

Accept the existing large-matrix guardrail `index.tsv` as Sprint 131's first
generated report/index artifact without changing its schema.

No source, script, Makefile, benchmark, coverage, dead-code, test, or public
documentation semantics were changed. The existing index is already generated
by `make large-matrix-guardrails`, has stable lane IDs, separates reviewed and
supplemental categories, records explicit supplemental skip rows, and is paired
with a manifest that records freshness anchors.

## Candidate Re-Check

| Requirement | Result | Evidence |
| --- | --- | --- |
| Source inputs are explicit. | Pass | `scripts/large_matrix_guardrails.sh` receives the report directory, three reviewed test binaries, and two benchmark binaries from the Makefile target. |
| Output path is explicit. | Pass | Primary index is `build/bench-reports/large-matrix-guardrails/index.tsv`; manifest is `build/bench-reports/large-matrix-guardrails/manifest.txt`. |
| Schema is deterministic. | Pass | Current columns are `lane_id`, `status`, `category`, `command`, `artifact`, and `notes`. |
| Stable row identity exists. | Pass | Lane IDs are `G1`, `G2`, `G3`, `G4`, `S1`, and `S2`. |
| Reviewed and supplemental rows are separated. | Pass | `G1`-`G4` are `reviewed`; `S1` and `S2` are `supplemental`. |
| Supplemental behavior is explicit. | Pass | Supplemental lanes emit `skip` rows unless `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1` is set. |
| Day 8 coverage boundary is preserved. | Pass | The guardrail index does not claim line coverage or coverage completeness. |
| Day 9 dead-code/guardrail boundary is preserved. | Pass | The index remains a large-matrix structural/report guardrail, not dead-code or cleanup evidence. |
| Benchmark semantics are unchanged. | Pass | `G4` validates bounded CSV shape and structural fill rows; no timing threshold or portable performance claim is introduced. |

## Regeneration Command

Reviewed/default mode:

```bash
make large-matrix-guardrails
```

Supplemental opt-in mode:

```bash
SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails
```

## Day 10 Regeneration Result

Ran:

```bash
make large-matrix-guardrails
```

Result: pass.

Observed generated files:

| Artifact | Role |
| --- | --- |
| `build/bench-reports/large-matrix-guardrails/index.tsv` | Primary generated index. |
| `build/bench-reports/large-matrix-guardrails/manifest.txt` | Run metadata and artifact inventory. |
| `build/bench-reports/large-matrix-guardrails/test_graph.txt` | Reviewed `G3` output. |
| `build/bench-reports/large-matrix-guardrails/test_reorder_nd.txt` | Reviewed `G2` output. |
| `build/bench-reports/large-matrix-guardrails/test_reorder_amd_qg.txt` | Reviewed `G1` output. |
| `build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv` | Reviewed `G4` bounded CSV-shape artifact. |

Observed index rows:

| Lane | Status | Category | Interpretation |
| --- | --- | --- | --- |
| `G1` | `pass` | `reviewed` | qg-AMD wrapper and generated banded structural guardrail. |
| `G2` | `pass` | `reviewed` | ND generated-family and named-matrix structural guardrail. |
| `G3` | `pass` | `reviewed` | Graph partition, separator, and generated-family structural guardrail. |
| `G4` | `pass` | `reviewed` | Bounded `bench_reorder` CSV shape and structural fill rows. |
| `S1` | `skip` | `supplemental` | Full named-matrix reorder report is opt-in. |
| `S2` | `skip` | `supplemental` | qg-AMD/generated-banded report is opt-in. |

Observed manifest freshness anchors:

| Field | Value |
| --- | --- |
| `generated_at_utc` | `2026-07-24T15:35:20Z` |
| `git_commit` | `2e3125a2` |
| `git_branch` | `sprint-131` |
| `supplemental` | `0` |

The generated report directory is a build artifact and remains outside the
Sprint 131 documentation artifact set.

## Touched Files

| File | Change type |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_131/artifacts/day10-first-index-implementation.md` | New Day 10 decision and validation artifact. |
| `docs/planning/EPIC_11/SPRINT_131/WORKING_NOTES.md` | Day 10 working-note update. |

No generated report files were checked into the sprint artifact directory.

## Unchanged Semantics Statement

Day 10 does not change:

- benchmark binaries or benchmark CSV schemas;
- coverage targets, coverage thresholds, or coverage interpretation;
- dead-code workflow, dead-code buckets, or dead-code validation behavior;
- test commands, test assertions, or guardrail lane membership;
- public performance, scalability, memory, solver, corpus, or coverage claims.

The accepted implementation is the current generated large-matrix guardrail
index path, not a normalized cross-report schema.

## Deferrals and Residual Implementation Queue

| Deferred item | Blocker | Dependency | Future owner |
| --- | --- | --- | --- |
| Cross-report normalized index schema | Current report families have different freshness, support-tier, failure, and claim-boundary semantics. | Day 11 freshness policy and Day 12 ownership map. | `report-index-owner` |
| Coverage index generation | Coverage remains tree-mutating and supplemental; rows need Day 8 fields before generation. | Future coverage-specific index design. | `coverage-workflow` |
| Dead-code index generation beyond existing `report.tsv` | Raw `xunused` and `cppcheck` outputs must stay behind classified bucket semantics. | Future decision on whether `report.tsv` gets freshness and owner fields. | `deadcode-workflow` |
| Supplemental large-matrix lane promotion | Supplemental rows are threshold-free and platform-local by design. | Runtime/platform baseline and claim-boundary design. | `large-matrix-guardrails` |
| Generated stale-report scanner | Current artifacts expose freshness anchors but no common scanner exists across report families. | Day 11 freshness policy. | `report-index-owner` |

## Day 11 Handoff

Day 11 should validate recurring freshness and drift behavior against this
accepted first index path:

- missing `index.tsv` or `manifest.txt`;
- mismatched manifest commit or branch;
- supplemental skip rows versus opt-in supplemental runs;
- reviewed lane failure versus stale artifact state;
- report directory cleanup or regeneration behavior.

Day 11 should not broaden the large-matrix guardrail index into coverage,
dead-code, benchmark timing, or corpus parity evidence.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Generated index is deterministic if implemented. | Complete | `make large-matrix-guardrails` regenerated the existing index and manifest with stable lane IDs and explicit reviewed/supplemental rows. |
| No benchmark, coverage, dead-code, or test semantics change silently. | Complete | No source, script, Makefile, benchmark, coverage, dead-code, or test files were edited; unchanged-semantics statement records the boundary. |
| Every deferral has blocker, dependency, and future owner. | Complete | Residual queue records blockers, dependencies, and owners for normalized schema, coverage, dead-code, supplemental, and stale-report scanner work. |
