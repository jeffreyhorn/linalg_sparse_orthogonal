# Sprint 163 Retrospective

**Sprint:** 163 - Methodology-Bound Performance Publication
**Duration:** 14 days (Days 1-14 landed on branch `sprint-163`)
**Status:** Complete

## Source Artifact Note

Sprint 163 was executed from the Epic 14 project-plan section for Sprint 163
and lives under `docs/planning/EPIC_14/SPRINT_163/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 project-plan path; `WORKING_NOTES.md` records that
path mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 163 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Inventoried benchmark, sentinel, wall-check, report-index, CI,
      documentation, and package-boundary evidence surfaces.
- [x] Selected the narrow performance-publication surface:
      `make bench-canonical-report` and `make performance-sentinels`.
- [x] Classified canonical benchmark rows as local-only threshold-free
      measurements.
- [x] Classified S5 sentinel rows as the only selected hard local wall-check
      timing gate.
- [x] Classified S2 and S3 sentinel rows as threshold-free backend-context
      report rows, not pass/fail or backend-superiority evidence.
- [x] Defined methodology fields, row-state semantics, repeat/warmup/variance
      rules, and non-superiority caveats for selected rows.
- [x] Added methodology fields to canonical benchmark `index.tsv` output while
      preserving existing leading columns and generated artifact locations.
- [x] Added baseline provenance, repeat semantics, warmup, variance, and
      methodology notes to sentinel `sentinels.tsv` output while preserving
      S5/S2/S3 semantics.
- [x] Updated report-index normalization so benchmark and sentinel
      configuration text preserves the new methodology fields.
- [x] Updated README, benchmark docs, maintainer guide, and report-index schema
      wording for local-only generated performance rows.
- [x] Revalidated selected report generation, normalizer behavior, corpus
      schema, package-boundary non-claims, and documentation wording.
- [x] Published the Sprint 164 API-header handoff.
- [x] Ran final targeted validation. No `.c` or public `.h` files changed, so
      the full C quality gate was not required.

## What Went Well

1. **The sprint kept the evidence surface narrow.** It selected canonical
   benchmark reports and performance sentinels instead of turning the sprint
   into broad benchmark governance or a full hosted performance program.

2. **Hard gates and report rows stayed separate.** S5 remains the only hard
   local timing gate, while canonical rows and S2/S3 sentinel rows are
   threshold-free reports with explicit `local_threshold_free` boundaries.

3. **The methodology fields make generated rows reviewable.** Canonical rows
   now carry status, support tier, claim boundary, fixture/workload, repeat,
   warmup, variance, baseline, threshold, backend context, and methodology
   notes. Sentinel rows now carry baseline provenance and matching methodology
   notes.

4. **The normalizer preserves row meaning.** Report-index output now carries
   the selected methodology fields forward as navigation metadata without
   turning generated local rows into release, hosted, package, ABI, platform,
   or performance proof.

5. **Documentation aligned before closeout.** README, benchmark docs,
   maintainer guide, and report-index schema docs now use the same local-only,
   threshold-free, and non-superiority language.

6. **Package evidence stayed out of performance evidence.** Sprint 162's
   static-first package boundary remained intact, and the static package
   deferral guard was included in Sprint 163 validation because the docs touch
   package/performance separation.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 163 plan lives under Epic 14. The sprint recorded the
   mismatch and proceeded from the current Epic 14 plan.

2. **The publication is intentionally local-only.** The selected outputs are
   useful and reviewable, but they are not hosted CI proof, portable
   performance proof, or state-of-the-art evidence.

3. **Statistical methodology remains limited.** Current selected rows still
   record `warmup=not_recorded` and `variance=not_recorded`; they document
   limits clearly but do not close repeat/variance methodology.

4. **Sensitive wording required repeated review.** Performance publication
   touches terms like proof, superiority, backend, ABI, package, and hosted CI.
   The sprint needed explicit scans and artifacts to keep positive claims from
   exceeding the evidence.

5. **Generated artifacts remain local and ignored.** This is correct for the
   current methodology, but it means reviewers must regenerate `build/`
   outputs instead of inspecting committed report rows.

## Final Metrics

### Validation

| Metric | Sprint 163 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no |
| shell syntax checks | passed |
| canonical benchmark report | passed: `make bench-canonical-report` |
| performance sentinels | passed: `make performance-sentinels` |
| focused normalizer tests | passed: `python3 tests/test_normalize_report_index.py` |
| benchmark/sentinel normalized index | passed: `26` rows |
| corpus schema validation | passed |
| static package deferral guard | passed |
| unsupported-claim wording scan | passed; hits are non-claims or boundaries |
| final `git diff --check` | passed |

### Selected Performance Surface

| Metric | Sprint 163 close state |
| --- | ---: |
| selected report commands | 2 |
| canonical benchmark rows | 4 |
| sentinel rows | 19 |
| S5 hard local wall-gate rows | 3 |
| S2 threshold-free report rows | 8 |
| S3 threshold-free report rows | 8 |
| normalized benchmark/sentinel rows | 26 |
| hosted CI performance-proof rows | 0 |
| portable performance claims | 0 |
| backend superiority claims | 0 |
| state-of-the-art claims | 0 |

### Artifact Package

| Metric | Sprint 163 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| shell report scripts changed | 2 |
| Python report-index scripts changed | 1 |
| public/maintainer/schema docs changed | 4 |
| generated build/report artifacts committed | 0 |

## Closed Claim

Sprint 163 closes this methodology-bound performance publication claim:

The project now has a selected local performance-publication surface for
canonical benchmark reports and performance sentinels. Canonical benchmark rows
are published as local-only threshold-free measurements with explicit
methodology fields. S5 sentinel rows remain the only selected hard local
wall-check timing gate with baseline provenance. S2 and S3 sentinel rows remain
threshold-free backend-context report rows. The normalized report index
preserves these fields for navigation, and public documentation states the
local-only and non-superiority boundaries.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-row-inventory.md](./artifacts/day2-row-inventory.md);
- [day3-surface-selection.md](./artifacts/day3-surface-selection.md);
- [day4-methodology-contract.md](./artifacts/day4-methodology-contract.md);
- [day5-schema-gap-analysis.md](./artifacts/day5-schema-gap-analysis.md);
- [day6-report-implementation-1.md](./artifacts/day6-report-implementation-1.md);
- [day7-report-implementation-2.md](./artifacts/day7-report-implementation-2.md);
- [day8-gate-classification.md](./artifacts/day8-gate-classification.md);
- [day9-benchmark-docs.md](./artifacts/day9-benchmark-docs.md);
- [day10-public-docs.md](./artifacts/day10-public-docs.md);
- [day11-selected-validation.md](./artifacts/day11-selected-validation.md);
- [day12-cross-surface-validation.md](./artifacts/day12-cross-surface-validation.md);
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 164 Readiness

Sprint 164 should begin from these settled Sprint 163 boundaries:

| Starting item | Required posture |
| --- | --- |
| Public API/header docs | Audit for unsupported performance, backend, platform, package, ABI, runtime-loader, or state-of-the-art wording. |
| Benchmark references | Cite Sprint 163 rows only as local methodology-bound evidence. |
| S5 sentinel rows | Keep as hard local wall-check gate rows with baseline provenance. |
| S2/S3 sentinel rows | Keep as threshold-free backend-context rows, not pass/fail or superiority proof. |
| Normalized report index | Treat as navigation metadata, not release proof. |
| Package/install/ABI evidence | Keep separate from performance evidence. |

Recommended Sprint 164 first step:

Audit public headers, generated API reference material, and user-facing API
documentation for language that could imply performance guarantees or broader
platform/package/ABI support than the current evidence supports.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 163 close:

- hosted performance publication proof with runner, compiler, command,
  artifact, and row-state evidence;
- repeat/warmup/variance methodology for selected benchmark rows;
- promotion path for S2/S3 rows if a future sprint wants backend superiority
  evidence;
- package/install, shared-library ABI, runtime-loader, and package-manager
  evidence as separate product surfaces;
- broad benchmark governance beyond the selected canonical report and sentinel
  commands;
- portable performance claims;
- state-of-the-art performance claims;
- external-library performance parity claims;
- OpenMP speedup portability claims.

Still consciously constrained rather than silently solved:

- no hosted CI performance proof from local generated rows;
- no release proof from normalized report-index rows alone;
- no package proof from performance rows;
- no ABI or runtime-loader proof from performance rows;
- no backend superiority proof from S2/S3 context rows;
- no broad platform performance parity claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-row-inventory.md](./artifacts/day2-row-inventory.md)
- [day3-surface-selection.md](./artifacts/day3-surface-selection.md)
- [day4-methodology-contract.md](./artifacts/day4-methodology-contract.md)
- [day5-schema-gap-analysis.md](./artifacts/day5-schema-gap-analysis.md)
- [day6-report-implementation-1.md](./artifacts/day6-report-implementation-1.md)
- [day7-report-implementation-2.md](./artifacts/day7-report-implementation-2.md)
- [day8-gate-classification.md](./artifacts/day8-gate-classification.md)
- [day9-benchmark-docs.md](./artifacts/day9-benchmark-docs.md)
- [day10-public-docs.md](./artifacts/day10-public-docs.md)
- [day11-selected-validation.md](./artifacts/day11-selected-validation.md)
- [day12-cross-surface-validation.md](./artifacts/day12-cross-surface-validation.md)
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md)
- [day14-closeout.md](./artifacts/day14-closeout.md)
