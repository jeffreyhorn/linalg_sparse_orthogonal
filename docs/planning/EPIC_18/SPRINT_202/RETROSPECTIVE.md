# Sprint 202 Retrospective

**Sprint:** 202 - Hosted Selected Benchmark Freshness on One Additional Platform
**Duration:** 14 days (Days 1-14 landed on branch `sprint-202`)
**Status:** Closed for branch-local implementation, validation, and claim
calibration of one macOS hosted selected benchmark freshness lane; hosted
GitHub Actions run evidence remains pending until the branch is pushed and CI
executes

## Source Artifact Note

Sprint 202 was executed from the Epic 18 project-plan section for Sprint 202
and lives under `docs/planning/EPIC_18/SPRINT_202/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint selected exactly one additional hosted selected benchmark freshness
lane: macOS hosted freshness for `SRT-BENCH-REFACTOR-CSC-NOS4`, the existing
`bench_refactor_csc` / `tests/data/suitesparse/nos4.mtx --repeat 1` selected
benchmark row. The work records candidate ranking, selected lane scope,
metadata contract, validator and manifest design, freshness regression
coverage, hosted workflow wiring, local hosted-mode simulation, documentation
calibration, integrated validation, review hardening, and final closeout.

## Definition Of Done Checklist

- [x] Created Sprint 202 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Ranked candidate hosted platforms and benchmark rows before selecting a
      lane.
- [x] Selected macOS hosted freshness for the existing selected benchmark row
      instead of expanding the benchmark matrix.
- [x] Preserved Sprint 192 threshold-free methodology semantics:
      `status=measurement`, `baseline=n/a`, `threshold=n/a`, and
      `not_portable_performance_claim`.
- [x] Added a macOS `selected-performance-freshness` workflow job with a
      selected-only artifact upload.
- [x] Extended selected target manifest metadata to include Linux and macOS
      hosted selected benchmark freshness lanes.
- [x] Added regression coverage for missing artifacts, missing and duplicate
      index rows, malformed timestamps, required selected metadata, path drift,
      hosted metadata drift, and unselected-row promotion.
- [x] Updated README, install, benchmark, corpus, schema, maintainer, and
      residual-queue wording without claiming portable performance,
      timing-threshold success, Linux/macOS parity, Windows selected benchmark
      freshness, package-manager support, release readiness, or
      state-of-the-art performance.
- [x] Ran focused workflow, manifest, docs, selected benchmark freshness,
      Python syntax, and whitespace checks.
- [x] Preserved explicit residuals for hosted CI evidence review and deferred
      benchmark/platform/package/performance claims.

## What Went Well

1. **The lane stayed selected.** The sprint added one macOS hosted selected
   freshness lane for the existing selected benchmark row and avoided broad
   canonical benchmark publication.

2. **Methodology vocabulary remained stable.** The selected row keeps
   threshold-free measurement semantics and the required
   `not_portable_performance_claim` token, so the new lane is freshness
   evidence rather than a timing gate.

3. **Workflow upload scope is narrow.** The macOS job uploads
   `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`, matching the
   selected evidence bundle instead of publishing all benchmark CSVs.

4. **Regression coverage became more drift-sensitive.** The checker and
   workflow tests now catch missing selected artifacts, duplicate selected
   rows, malformed metadata, hosted placeholder metadata, path drift, and
   unselected-row promotion.

5. **Documentation moved with the manifest.** Public and maintainer surfaces
   now describe Linux/macOS hosted selected freshness while keeping explicit
   non-claims adjacent to the evidence.

6. **Residuals were made reviewable.** The residual queue and closeout artifact
   state exactly what hosted evidence must be inspected after push or PR
   creation.

## What Didn't Go Well

1. **Hosted proof cannot be completed before CI runs.** Local hosted-mode
   simulation proves metadata and checker behavior, but it cannot prove the
   GitHub-hosted macOS runner, artifact upload, and summary until the branch is
   pushed.

2. **Several claim surfaces needed coordinated edits.** README, INSTALL,
   benchmark docs, corpus docs, schema docs, maintainer guidance, and residual
   queue wording all had to move together to avoid stale Linux-only language.

3. **Path and artifact semantics are easy to over-broaden.** The sprint needed
   explicit tests and review scans to keep the macOS lane from becoming a broad
   hosted benchmark publication claim.

4. **Windows remains intentionally deferred.** Windows selected benchmark
   freshness has separate shell, path, runtime, and claim risks and remains a
   non-claim after this sprint.

## Final Metrics

### Validation

| Metric | Sprint 202 close state |
| --- | --- |
| selected performance docs guard | passed on Days 10, 11, 12, 13, and 14 |
| selected workflow guard | passed on Days 8, 9, 11, 12, and 14 |
| selected report target manifest test | passed on Days 11, 12, and 14 |
| selected benchmark freshness regression | passed on Days 7, 11, 12, 13, and 14 |
| normalizer report-index checks | passed on Day 12 |
| hosted-mode local simulation | passed on Days 11 and 12 with Sprint 202 macOS hosted metadata |
| Python syntax check | passed on Days 11, 12, 13, and 14 |
| stale claim/path scans | passed on Days 10, 12, and 13 with only intentional non-claim hits |
| final `git diff --check` | passed |
| final full C quality gate | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 202 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| CI workflow files changed | 1 |
| Public documentation files changed | 4 |
| Maintainer documentation files changed | 1 |
| Corpus manifest/schema documentation files changed | 3 |
| Python validation or guard test files changed | 3 |
| Epic residual queue files changed | 1 |
| Production C implementation files changed | 0 |
| Public or internal C header files changed | 0 |
| Makefile targets changed | 0 |
| CMake registration files changed | 0 |
| Benchmark binaries changed | 0 |
| Package recipe or install script files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Platform and row selection completed | 1 |
| Methodology metadata items completed | 1 |
| Workflow lane items implemented | 1 |
| Freshness test items completed | 1 |
| Docs calibration items completed | 1 |
| Local validation items completed | 1 |
| Hosted CI evidence residuals retained | 1 |
| Portable performance claims promoted | 0 |
| Timing-threshold claims promoted | 0 |
| Broad benchmark-family publication claims promoted | 0 |

The count covers Sprint 202 items 202.1 through 202.6.

## Closed Claim

Sprint 202 closes this bounded branch-local claim:

The current branch adds one macOS hosted selected benchmark freshness lane for
`SRT-BENCH-REFACTOR-CSC-NOS4`. The selected row remains
`bench_refactor_csc` over `tests/data/suitesparse/nos4.mtx --repeat 1`; the
macOS workflow job is `selected-performance-freshness`; the selected workflow
artifact is `sprint202-macos-selected-performance-freshness`; the uploaded
bundle is limited to `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`;
the freshness checker remains `scripts/check_bench_canonical_freshness.py
--mode hosted`; the selected target manifest records Linux and macOS hosted
selected freshness; public and maintainer documentation preserve
threshold-free, methodology-bound, non-portable interpretation.

This claim does not include portable performance, timing thresholds, benchmark
speedup, Linux/macOS performance parity, Windows selected benchmark freshness,
broad benchmark-family publication, Homebrew/core readiness, bottles,
Linuxbrew, public package-manager distribution, package or ABI support,
backend superiority, release benchmark readiness, broad platform support, or
state-of-the-art sparse linear algebra performance.

Hosted CI execution remains a residual: the claim is branch-local until a
post-push GitHub Actions run proves the macOS job executed, checked hosted
freshness, uploaded the selected bundle, and summarized the selected row
without overclaiming.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-benchmark-freshness-intake.md](./artifacts/day1-benchmark-freshness-intake.md);
- [day2-platform-row-ranking.md](./artifacts/day2-platform-row-ranking.md);
- [day3-selected-lane-decision.md](./artifacts/day3-selected-lane-decision.md);
- [day4-methodology-metadata-contract.md](./artifacts/day4-methodology-metadata-contract.md);
- [day5-validator-manifest-design.md](./artifacts/day5-validator-manifest-design.md);
- [day6-freshness-validator-implementation.md](./artifacts/day6-freshness-validator-implementation.md);
- [day7-freshness-regression-fixtures.md](./artifacts/day7-freshness-regression-fixtures.md);
- [day8-hosted-workflow-lane.md](./artifacts/day8-hosted-workflow-lane.md);
- [day9-workflow-guard-local-simulation.md](./artifacts/day9-workflow-guard-local-simulation.md);
- [day10-documentation-calibration.md](./artifacts/day10-documentation-calibration.md);
- [day11-focused-freshness-validation.md](./artifacts/day11-focused-freshness-validation.md);
- [day12-integrated-validation-hosted-evidence.md](./artifacts/day12-integrated-validation-hosted-evidence.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-retrospective-inputs.md](./artifacts/day14-closeout-retrospective-inputs.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Hosted macOS selected benchmark freshness evidence | PR or pushed branch CI review | Confirm `selected-performance-freshness` ran on `macos-latest`, captured CPU metadata, generated the canonical report, passed hosted freshness, uploaded the selected bundle, and kept summary claims bounded. |
| Windows selected benchmark freshness remains unclaimed | Future Windows selected benchmark owner | Rank Windows runtime/path/shell risks, add selected workflow and path-normalized checks, update docs, and inspect hosted Windows evidence. |
| Timing thresholds remain unclaimed | Future methodology and performance owner | Define baseline, variance, warmup/repeat policy, tolerance, same-machine comparison semantics, and threshold failure behavior before adding timing gates. |
| Broad benchmark-family publication remains unclaimed | Future benchmark publication owner | Select additional rows deliberately, add manifest identity and upload guards, and calibrate docs before publishing broader benchmark artifacts. |
| Package-manager and package/ABI support remain unclaimed | Future packaging owner | Add and validate package recipes, install evidence, ABI policy, bottles or explicit non-bottle policy, and public support wording. |
| State-of-the-art sparse linear algebra performance remains unclaimed | Future evidence and comparison owner | Provide methodology-bound external comparisons, representative matrices, repeatable performance evidence, and claim-reviewed documentation. |

## Next-Sprint Readiness

Sprint 202 leaves one additional hosted selected benchmark freshness lane ready
for post-push hosted evidence review.

| Future need | Sprint 202 handoff |
| --- | --- |
| Hosted CI review | Use the Day 12 and Day 14 checklist to inspect the macOS workflow run and artifact contents after PR creation. |
| Additional platform freshness | Reuse the candidate-ranking and selected-lane decision pattern, but treat Windows as a separate owner with path and shell risks. |
| Benchmark threshold promotion | Keep the Sprint 192 and Sprint 202 threshold-free metadata until a sprint owns threshold methodology end to end. |
| Documentation maintenance | Keep Linux/macOS hosted selected freshness wording aligned across README, INSTALL, benchmark docs, corpus docs, schema docs, maintainer guide, and selected target manifest. |
| Validator maintenance | Preserve tests for missing/duplicate selected rows, path drift, hosted metadata drift, and unselected-row promotion whenever selected benchmark artifacts change. |

## Final Assessment

Sprint 202 is complete as a branch-local selected benchmark freshness sprint. It
adds one macOS hosted selected lane, keeps artifact publication selected-only,
extends manifest and guard coverage, calibrates public and maintainer wording,
and records the exact hosted CI evidence still required after push. The sprint
does not promote portable performance, timing-threshold, package-manager,
platform-parity, release, or state-of-the-art claims.
