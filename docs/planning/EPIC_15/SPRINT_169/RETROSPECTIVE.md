# Sprint 169 Retrospective

**Sprint:** 169 - Performance Methodology Hardening
**Duration:** 14 days (Days 1-14 landed on branch `sprint-169`)
**Status:** Complete

## Source Artifact Note

Sprint 169 was executed from the active Epic 15 project-plan section for
Sprint 169 and lives under `docs/planning/EPIC_15/SPRINT_169/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 169 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Carried forward the Sprint 168 selected hosted performance lane:
      `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`.
- [x] Audited the canonical report generator, selected freshness checker,
      generated canonical report shape, hosted CI metadata, README benchmark
      wording, benchmark docs, and maintainer-guide references.
- [x] Defined repeat-count, warmup, variance, sample, and threshold semantics
      for the selected performance publication row.
- [x] Normalized selected report metadata with `matrix_size=n=100`,
      `warmup=none_configured`, and
      `variance=not_computed_single_sample`.
- [x] Extended selected-row freshness checking with manifest agreement and
      focused positive/negative regression tests.
- [x] Added a separate S6 local selected-lane regression sentinel without
      adding a threshold to the canonical selected publication row.
- [x] Updated README, benchmark docs, and maintainer docs with selected
      performance evidence paths, report-index handoff wording, S6 sentinel
      scope, hosted proof boundaries, and platform/backend caveats.
- [x] Prepared the hosted evidence review checklist, expected CI summary
      output, artifact review steps, fallback handling, and evidence
      activation rule.
- [x] Ran final focused validation for shell syntax, Python compile, selected
      freshness tests, selected report freshness, performance sentinels,
      normalized report-index checks, targeted claim scans, generated-output
      hygiene, and `git diff --check`.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 169 edits.

## What Went Well

1. **The selected performance lane became methodology-bound.** Sprint 169 kept
   the Sprint 168 `bench_refactor_csc` lane rather than reopening selection,
   then gave its repeat, warmup, variance, matrix-size, threshold, platform,
   runner, build-mode, and backend fields explicit meanings.

2. **Ambiguous statistical fields were replaced with reviewable policy.**
   Days 3-5 changed `warmup` and `variance` from vague non-recorded concepts
   into explicit values: `none_configured` and
   `not_computed_single_sample`.

3. **The schema contract now has regression coverage.** Day 6 added direct
   tests for selected-row identity, row width, required methodology fields,
   manifest agreement, hosted-mode boundaries, and unselected-row local-only
   behavior.

4. **Regression governance stayed separate from publication evidence.** Days
   7-8 added S6 as a local selected-lane smoke ceiling while preserving
   canonical selected performance rows as threshold-free evidence with
   `baseline=n/a` and `threshold=n/a`.

5. **Documentation got a clearer evidence path.** Days 9-11 linked README,
   benchmark documentation, generated report-index interpretation, and the
   maintainer guide so reviewers can find selected freshness, S6 local
   sentinel context, and platform/backend caveats without broadening claims.

6. **Hosted proof remains conditional by design.** Day 13 made the hosted CI
   activation boundary explicit: branch-local hosted-style validation is only
   preflight evidence until PR CI passes and publishes the
   `sprint168-selected-performance-freshness` artifact bundle.

7. **Final validation was focused and repeatable.** Days 12 and 14 ran the
   selected freshness, sentinel, schema, report-index, claim-scan, and hygiene
   checks that actually govern the Sprint 169 surface.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 169 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **The selected report still uses one configured sample.** Sprint 169 made
   this explicit with `configured_repeat_1` and
   `not_computed_single_sample`, but it does not provide variance,
   confidence intervals, warmup-controlled timing, or portable performance
   evidence.

3. **The canonical report bundle still includes unselected rows.** The checker
   and docs keep only `bench_refactor_csc` promoted, but reviewers still need
   to recognize that `bench_chol_csc`, `bench_iterative_reuse`, and
   `bench_eigs_reuse` remain local/advisory.

4. **S6 is intentionally coarse.** The local S6 smoke ceiling catches large
   selected-lane regressions, but it is not a calibrated portable performance
   threshold and should not be used as hosted publication evidence.

5. **Hosted proof is unavailable until PR CI runs.** The branch can validate
   local and hosted-style semantics, but the reviewed hosted claim depends on
   the named CI job passing for the reviewed commit.

6. **Claim scans remain noisy.** Unsupported-claim phrases appear in
   non-claims, stop conditions, and validation command records, so the sprint
   required interpretation rather than treating every match as a defect.

## Final Metrics

### Validation

| Metric | Sprint 169 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| shell syntax checks | passed: `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh` |
| Python compile checks | passed for selected freshness checker, report-index normalizer, and focused tests |
| selected freshness tests | passed: 8 positive/negative cases |
| local selected freshness | passed: `make bench-canonical-report-freshness` |
| hosted-style local metadata validation | passed with hosted selected metadata and hosted checker mode |
| local performance sentinel bundle | passed: `make performance-sentinels` |
| S6 forced-failure smoke check | passed on Day 8 by failing as expected with a near-zero ceiling |
| normalized report-index tests | passed |
| normalized benchmark/sentinel freshness | passed: 27 rows with expected advisory and hard-gate warnings |
| targeted claim scans | passed by inspection; matches were scoped caveats, non-claims, or validation records |
| final `git diff --check` | passed |
| generated build/report/cache artifacts committed | 0 |

### Changed Surface

| Metric | Sprint 169 close state |
| --- | ---: |
| Makefile targets/comments changed | 1 |
| shell report/sentinel scripts changed | 2 |
| Python report/checker scripts changed | 2 |
| Python focused tests added or updated | 2 |
| public/maintainer docs changed | 3 |
| workflow files changed | 0 |
| C source files changed | 0 |
| public header files changed | 0 |
| daily artifacts under `SPRINT_169/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 169 close state |
| --- | ---: |
| selected hosted performance rows | 1 |
| selected fixture scopes | 1 |
| selected canonical freshness checkers | 1 |
| selected canonical freshness test files | 1 |
| local selected-lane regression sentinels | 1 |
| timing thresholds added to canonical selected publication row | 0 |
| portable performance superiority claims added | 0 |
| broad benchmark publication claims added | 0 |
| external-library parity claims added | 0 |
| package, shared-library, or ABI claims added | 0 |
| broad platform parity claims added | 0 |
| state-of-the-art sparse linear algebra performance claims added | 0 |

## Closed Claim

Sprint 169 closes this Epic 15 performance-methodology hardening claim:

The selected Sprint 168 performance lane now has a source-controlled
methodology policy, normalized selected-row schema, manifest agreement,
focused regression tests, a separate local selected-lane large-regression
sentinel, public and maintainer documentation paths, platform/backend caveats,
hosted evidence review instructions, and final focused validation.

The selected publication row remains the `bench_refactor_csc` canonical row on
`tests/data/suitesparse/nos4.mtx --repeat 1`. It is threshold-free, uses
`repeat_semantics=configured_repeat_1`, records
`warmup=none_configured`, records
`variance=not_computed_single_sample`, records `matrix_size=n=100`, and
becomes reviewed hosted evidence only after the named PR CI job passes and
publishes the named artifact bundle for the reviewed commit.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-methodology-intake.md](./artifacts/day1-methodology-intake.md);
- [day2-methodology-audit.md](./artifacts/day2-methodology-audit.md);
- [day3-statistical-policy.md](./artifacts/day3-statistical-policy.md);
- [day4-schema-normalization-design.md](./artifacts/day4-schema-normalization-design.md);
- [day5-policy-implementation.md](./artifacts/day5-policy-implementation.md);
- [day6-schema-regression-tests.md](./artifacts/day6-schema-regression-tests.md);
- [day7-regression-sentinel-design.md](./artifacts/day7-regression-sentinel-design.md);
- [day8-sentinel-implementation.md](./artifacts/day8-sentinel-implementation.md);
- [day9-documentation-indexing-design.md](./artifacts/day9-documentation-indexing-design.md);
- [day10-documentation-indexing.md](./artifacts/day10-documentation-indexing.md);
- [day11-platform-and-backend-caveats.md](./artifacts/day11-platform-and-backend-caveats.md);
- [day12-integrated-local-validation.md](./artifacts/day12-integrated-local-validation.md);
- [day13-hosted-evidence-prep.md](./artifacts/day13-hosted-evidence-prep.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No broad state-of-the-art sparse linear algebra status, portable performance
superiority, broad benchmark-family publication, external-library performance
parity, package-manager distribution, shared-library support, dynamic ABI
stability, runtime-loader behavior, broad platform parity, release benchmark
proof, or solver correctness claim was added.

## Sprint 170 Readiness

| Future need | Sprint 169 handoff |
| --- | --- |
| Shared-library ABI decision | Treat selected performance methodology as unrelated to package, shared-library, dynamic ABI, runtime-loader, or package-manager evidence. |
| Package metadata review | Preserve static-first/package wording boundaries and avoid coupling install proof to performance evidence. |
| Hosted performance evidence | Require the named hosted selected-performance CI job and artifact bundle before citing hosted evidence for the reviewed commit. |
| Local regression governance | Keep S6 local-only and separate from threshold-free canonical publication rows. |
| Generated report publication | Keep benchmark, sentinel, and normalized report outputs ignored unless a later sprint explicitly publishes a reviewed artifact. |
| Claim-scope maintenance | Continue scanning README, benchmark docs, and maintainer docs for performance, ABI, platform, backend, and state-of-the-art overreach. |

## Follow-Up Risks

| Risk | Handling |
| --- | --- |
| Hosted selected-performance CI fails due to GitHub Actions infrastructure. | Rerun CI; do not activate hosted evidence or change claim wording for infra-only failures. |
| Hosted freshness checker rejects metadata. | Treat it as a schema or claim-boundary regression and preserve selected-row-only promotion. |
| S6 timing varies across local machines. | Keep S6 as a coarse local smoke ceiling and avoid portable performance wording. |
| Unselected canonical rows are mistaken for hosted performance evidence. | Continue enforcing local-only unselected rows in checker tests, docs, and PR descriptions. |
| Single-sample timing is overread as statistical evidence. | Preserve `not_computed_single_sample`, threshold-free baseline fields, and no-portable-performance language. |
| Sprint 170 ABI work accidentally borrows performance evidence. | Start Sprint 170 from the Day 14 handoff and keep package/ABI claims independently evidenced. |
