# Sprint 132 Day 10 - Report Index Handoff and Metadata Validation

## Purpose

Validate Sprint 132 generated benchmark and sentinel metadata against the
Sprint 131 report-index model, then record handoff rules and residual
index-integration gaps.

This artifact validates generated metadata shape and freshness context. It
does not change scripts, benchmarks, guardrails, thresholds, public APIs, or
performance claims.

## Generated Report Inspection

| Report family | Artifact | Inspection result | Freshness result |
| --- | --- | --- | --- |
| Performance sentinels | `build/bench-reports/sentinels/sentinels.tsv` | Header has 20 tab-separated fields and 11 data rows. All rows match header width. | Manifest commit `d348b6ca` and branch `sprint-132` match the current checkout. |
| Performance sentinels | `build/bench-reports/sentinels/manifest.txt` | Manifest records generated time, report dir, git commit/branch, platform, compiler, build mode, `OMP_NUM_THREADS`, and dense backend env vars. | Current for this branch as of the Day 8 generation. |
| Canonical benchmark report | `build/bench-reports/canonical/index.tsv` | Header has 13 tab-separated fields and 4 data rows. All rows match header width. | Manifest commit `d348b6ca` and branch `sprint-132` match the current checkout. |
| Canonical benchmark report | `build/bench-reports/canonical/manifest.txt` | Manifest records generated time, report dir, report label, git commit/branch, platform, compiler, build mode, and `OMP_NUM_THREADS`. | Current for this branch as of the Day 8 generation. |
| Large-matrix guardrails | `build/bench-reports/large-matrix-guardrails/manifest.txt` | Existing build artifact records Sprint 131 guardrail metadata. | Historical/stale for Sprint 132 because manifest branch is `sprint-131`; not used as current Day 10 evidence. |

## Metadata Validation Results

| Field family | Sentinel result | Canonical result | Day 10 interpretation |
| --- | --- | --- | --- |
| Stable row identity | `sentinel_id` distinguishes S5 and S2 rows. | `artifact` and `command` distinguish canonical rows. | Sufficient for current report-family indexing. |
| Status | `pass` for S5 and `report` for S2. | No pass/fail status by design. | Preserves hard-gate versus threshold-free split. |
| Support tier | Explicit `reviewed_thresholded` and `reviewed_threshold_free`. | Implicit through `surface=canonical` and `category=measurement`; no normalized `support_tier` column. | Sentinel is index-ready; canonical needs a normalized support-tier field only if a cross-report index consumes it. |
| Claim boundary | Explicit `local_wall_gate` and `local_threshold_free`. | Documented as threshold-free in README/maintainer guide; no generated `claim_boundary` column. | Sentinel is index-ready; canonical claim boundary remains documentation-backed. |
| Freshness | Manifest records generated time, commit, branch, platform, compiler, build mode, thread context, and backend env vars. | Manifest and index record generated time, commit, branch, platform, compiler, build mode, and thread context. | Freshness anchors are reproducible for affected Sprint 132 reports. |
| Backend context | S5 uses `n/a`; S2 records request `unset`, selected `builtin`, fallback `n/a`, dense kernel `builtin`, and panel solver `batched_panel`. | Direct benchmark CSVs retain backend fields where benchmark-owned; index does not duplicate them. | Backend state remains visible without claiming parity. |
| OpenMP and thread context | Rows include `build_mode=serial` and `omp_num_threads=unset`. | Index and manifest include `build_mode=serial` and `omp_num_threads=unset`. | Runtime context is visible and local-only. |
| Artifact path | Rows name `wall_check.txt` or `bench_chol_csc_nos4.csv`. | Rows name canonical artifact basename and relative path. | Artifact traceability is sufficient for current report-family consumers. |
| Failure meaning | S5 fail remains existing wall-check failure; S2 `report` is threshold-free. | Canonical has no failure status beyond command success. | Failure meanings remain family-specific and should not be flattened. |

## Stale, Missing, Skip, Supplemental, and Unavailable Behavior

| Condition | Current behavior | Index handoff |
| --- | --- | --- |
| Stale generated artifact | Compare manifest git commit/branch to current checkout; Sprint 131 guardrail artifact is historical/stale on `sprint-132`. | Mark stale/historical before using as current evidence; stale is not a solver failure. |
| Missing required sentinel binary | Sentinel script emits `skip` where practical. | Preserve skip row and note; skip is not pass. |
| Missing wall-check baseline | Sentinel script emits S5 skip. | Preserve as missing local gate evidence, not performance success. |
| Missing Cholesky fixture | Sentinel script emits S2 skip. | Preserve as report incompleteness. |
| Supplemental guardrail disabled | Existing guardrail index emits supplemental skips in default mode. | Keep supplemental rows opt-in; do not promote through sentinel/canonical changes. |
| Optional backend unavailable | Sentinel S2 can still report builtin/fallback context where available; unknown states remain explicit in skip/error rows. | Preserve `n/a`, `unknown`, selected, and fallback fields; do not infer availability. |
| Canonical direct backend fields | Backend metadata remains inside benchmark-owned CSVs, not duplicated in the canonical index. | Cross-report index must inspect direct CSV headers or add explicit canonical fields later. |

## Sprint 131 Alignment Table

| Sprint 131 requirement | Sprint 132 status | Evidence | Gap |
| --- | --- | --- | --- |
| Generated timestamp and git commit/branch for freshness. | Met for affected sentinel and canonical reports. | Both manifests record `generated_at_utc`, `git_commit=d348b6ca`, and `git_branch=sprint-132`. | None for affected reports. |
| Platform/compiler/backend/OpenMP context for runtime reports. | Met for sentinels; mostly met for canonical. | Sentinel manifest and rows include runtime/backend context; canonical index/manifest include platform/compiler/build/thread context. | Canonical index does not duplicate backend fields from direct CSVs. |
| Stable row identity. | Met. | S5/S2 sentinel IDs; canonical artifact plus command rows. | Cross-report normalized `report_key` is deferred. |
| Support tier visible. | Met for sentinels; documentation-backed for canonical. | Sentinel `support_tier`; canonical `surface=canonical`, `category=measurement`, README/maintainer non-claim wording. | Add explicit canonical support tier only if normalized index work resumes. |
| Claim boundary visible. | Met for sentinels; documentation-backed for canonical. | Sentinel `claim_boundary`; canonical threshold-free wording in docs. | Add explicit canonical claim boundary only with cross-report schema work. |
| Failure meaning explicit. | Met by report-family policy. | S5 hard gate, S2 report rows, canonical command success, guardrail historical policy. | Automated failure-meaning field remains deferred. |
| Stale/missing/skip behavior explicit. | Met by current policy. | Sentinel skip rows; Sprint 131 freshness labels; Day 9 report-index handoff wording. | Automated stale-report scanner remains deferred. |
| No portable performance, backend parity, OpenMP speedup, or memory claim. | Met. | Day 9 non-claim scan and docs wording. | Ongoing claim-drift review remains maintainer-guide owned. |

## Residual Metadata Queue

| Gap | Blocker | Dependency | Future owner |
| --- | --- | --- | --- |
| Canonical `support_tier` column | Current canonical index is artifact-level and threshold-free; no consumer requires normalized support tiers yet. | Cross-report index consumer or schema validator decision. | `report-index-owner` plus `benchmark-report-owner`. |
| Canonical `claim_boundary` column | Claim boundary is documented but not generated per row. | Decision to make canonical index rows self-describing beyond README/manifest. | `report-index-owner`. |
| Canonical direct backend duplication | Backend fields live in direct benchmark CSVs, not canonical `index.tsv`. | Parser or schema decision for artifact-level versus row-level indexing. | Direct/backend benchmark owner and `report-index-owner`. |
| Automated stale-report scanner | Report families still differ in freshness anchors and status meanings. | Common metadata contract across sentinel, canonical, guardrail, coverage, and dead-code reports. | `report-index-owner`. |
| Large-matrix guardrail Sprint 132 refresh | Existing build artifact is historical from `sprint-131`. | Run `make large-matrix-guardrails` only if Day 11/13 validation selects that surface. | `large-matrix-guardrails`. |
| Supplemental validation | Supplemental guardrail rows remain opt-in and host-sensitive. | Runtime budget and support-tier promotion policy. | `large-matrix-guardrails`. |
| Unavailable optional backend probe rows | No public backend availability probe was added. | Explicit unsupported/unavailable semantics and non-portability policy. | Runtime governance owner. |

## Day 11 Handoff

Day 11 should run focused benchmark/runtime validation for touched surfaces:

- re-run `bash -n scripts/performance_sentinels.sh`
- re-run `bash -n scripts/bench_canonical_report.sh`
- re-run `make performance-sentinels`
- re-run `make bench-canonical-report`
- inspect generated sentinel and canonical headers
- run docs hygiene

Do not run supplemental guardrails or broad benchmark sweeps unless Day 11
explicitly expands validation scope.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Affected generated metadata is reproducible or explicitly deferred. | Complete | Sentinel and canonical reports generated successfully on Day 8 and passed Day 10 schema inspection; stale Sprint 131 guardrail artifact is explicitly deferred. |
| Report rows preserve support-tier and freshness boundaries. | Complete | Sentinel rows carry support tier and claim boundary; sentinel/canonical manifests carry freshness anchors; canonical normalized gaps are recorded. |
| Every index-integration gap has blocker, dependency, and owner. | Complete | Residual metadata queue records blockers, dependencies, and owners for canonical normalization, stale scanning, guardrail refresh, supplemental validation, and backend availability rows. |
