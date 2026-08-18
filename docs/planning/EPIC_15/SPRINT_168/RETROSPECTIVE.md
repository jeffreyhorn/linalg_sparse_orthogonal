# Sprint 168 Retrospective

**Sprint:** 168 - Hosted Performance Publication Lane
**Duration:** 14 days (Days 1-14 landed on branch `sprint-168`)
**Status:** Complete

## Source Artifact Note

Sprint 168 was executed from the active Epic 15 project-plan section for
Sprint 168 and lives under `docs/planning/EPIC_15/SPRINT_168/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path and the title "Hosted Performance Publication Date"; `WORKING_NOTES.md`
records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 168 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Carried forward Sprint 167 selected gap `G167-01`, acceptance gates,
      stop conditions, and performance-publication non-claims.
- [x] Inventoried current benchmark/report owners, canonical report outputs,
      generated-output paths, existing docs wording, and reusable freshness
      conventions.
- [x] Selected one hosted performance publication lane:
      `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`
      through `make bench-canonical-report`.
- [x] Confirmed local runtime suitability and generated report stability for
      the selected lane.
- [x] Extended canonical report metadata with support tier, claim boundary,
      runner context, build flags, CPU model, build mode, thread state,
      repeat semantics, timestamp, branch, commit, baseline, threshold, and
      methodology notes.
- [x] Added a strict selected-row freshness checker,
      `scripts/check_bench_canonical_freshness.py`.
- [x] Added `make bench-canonical-report-freshness` for local selected-row
      freshness validation.
- [x] Added the hosted CI job
      `Linux reviewed hosted selected performance freshness` with bounded
      runtime, hosted metadata, selected freshness validation, summary output,
      and artifact upload.
- [x] Updated README, benchmark docs, and maintainer docs with the selected
      hosted lane and retained non-claims.
- [x] Prepared hosted evidence review expectations, fallback wording, and the
      Sprint 169 methodology-hardening handoff.
- [x] Ran final focused validation and `git diff --check`.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 168 edits.

## What Went Well

1. **The sprint narrowed performance publication to one reviewable lane.**
   Day 3 selected `bench_refactor_csc` on `nos4.mtx --repeat 1` after
   comparing canonical alternatives and rejecting broader benchmark,
   sentinel, and smoke-test promotion paths.

2. **The methodology boundary became source-controlled.** Days 5 and 6 moved
   the selected lane from local benchmark output toward inspectable report
   metadata by adding runner, build, CPU, support-tier, claim-boundary, and
   methodology fields to the canonical index and manifest.

3. **Freshness is enforced by a focused checker.** Day 8 added a checker that
   validates the selected row, required metadata, selected command and
   fixture, threshold-free policy, local versus hosted modes, and manifest
   agreement without treating all canonical benchmark rows as hosted
   evidence.

4. **Hosted CI has an explicit evidence path.** Day 10 added a dedicated
   hosted freshness job that generates the canonical report bundle, checks the
   selected row in hosted mode, prints reviewer-oriented summary lines, and
   uploads the report artifact.

5. **Documentation stayed claim-safe.** Day 11 updated README, benchmark
   documentation, and the maintainer guide while preserving non-claims for
   portable performance, broad benchmark publication, external-library parity,
   package/ABI support, platform parity, release proof, and state-of-the-art
   sparse linear algebra performance.

6. **Closeout separated local validation from hosted proof.** Days 13 and 14
   made it explicit that the hosted evidence claim becomes active only after
   the named PR CI job passes and uploads the named artifact.

## What Didn't Go Well

1. **The initial prompt path was stale.** The request referenced Epic 12 while
   the active Sprint 168 section belongs to Epic 15. The sprint handled this
   by recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **The existing canonical report schema was not sufficient for publication.**
   The selected lane needed additional metadata before it could be reviewed as
   methodology-bound hosted evidence.

3. **The selected report still lacks measured warmup and variance policy.**
   Sprint 168 records those fields and keeps claims threshold-free, but
   Sprint 169 should decide whether they remain explicit non-measured metadata
   or become measured methodology fields.

4. **Hosted proof remains pending until PR CI runs.** Local hosted-mode
   validation can prove the code path and metadata contract, but the actual
   hosted evidence is not available until the branch CI job completes.

5. **The canonical bundle includes unselected rows.** The generator still
   writes four CSV artifacts; Sprint 168 mitigates this with selected-row
   checker logic and documentation, but reviewers must keep unselected rows in
   advisory status.

## Final Metrics

### Validation

| Metric | Sprint 168 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| shell syntax check | passed: `bash -n scripts/bench_canonical_report.sh` |
| Python compile check | passed: `python3 -m py_compile scripts/check_bench_canonical_freshness.py` |
| checker CLI smoke check | passed: `python3 scripts/check_bench_canonical_freshness.py --help` |
| workflow YAML parse | passed with Ruby YAML |
| local selected freshness | passed: `make bench-canonical-report-freshness` |
| hosted-mode local equivalent | passed with Sprint 168 hosted metadata |
| CI summary logic | passed against hosted-style local output |
| targeted claim scan | passed by inspection; matches were non-claims or guarded boundaries |
| final `git diff --check` | passed |
| generated build/report/cache artifacts committed | 0 |

### Changed Surface

| Metric | Sprint 168 close state |
| --- | ---: |
| workflow files changed | 1 |
| Makefile targets changed | 1 |
| shell report scripts changed | 1 |
| Python freshness scripts added | 1 |
| public/maintainer docs changed | 3 |
| C source files changed | 0 |
| public header files changed | 0 |
| daily artifacts under `SPRINT_168/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 168 close state |
| --- | ---: |
| selected hosted performance rows | 1 |
| selected fixture scopes | 1 |
| selected hosted CI jobs | 1 |
| selected freshness Make targets | 1 |
| selected report artifacts uploaded by hosted job | 6 |
| timing regression thresholds added | 0 |
| portable performance superiority claims added | 0 |
| broad benchmark publication claims added | 0 |
| external-library parity claims added | 0 |
| package, shared-library, or ABI claims added | 0 |
| broad platform parity claims added | 0 |
| state-of-the-art sparse linear algebra performance claims added | 0 |

## Closed Claim

Sprint 168 closes this Epic 15 hosted performance publication claim:

The project now has a source-controlled selected hosted performance lane for
the `bench_refactor_csc` canonical benchmark row on
`tests/data/suitesparse/nos4.mtx --repeat 1`, with methodology metadata,
local freshness validation, hosted CI freshness validation, summary output,
artifact upload, and claim-safe public documentation. The lane is
threshold-free and supports only the selected GitHub Actions Linux hosted
evidence path after the named CI job passes.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-benchmark-surface-inventory.md](./artifacts/day2-benchmark-surface-inventory.md);
- [day3-candidate-lane-selection.md](./artifacts/day3-candidate-lane-selection.md);
- [day4-runtime-suitability.md](./artifacts/day4-runtime-suitability.md);
- [day5-methodology-metadata-design.md](./artifacts/day5-methodology-metadata-design.md);
- [day6-metadata-implementation.md](./artifacts/day6-metadata-implementation.md);
- [day7-freshness-design.md](./artifacts/day7-freshness-design.md);
- [day8-freshness-implementation.md](./artifacts/day8-freshness-implementation.md);
- [day9-ci-lane-design.md](./artifacts/day9-ci-lane-design.md);
- [day10-ci-implementation.md](./artifacts/day10-ci-implementation.md);
- [day11-claim-safe-docs.md](./artifacts/day11-claim-safe-docs.md);
- [day12-local-validation.md](./artifacts/day12-local-validation.md);
- [day13-hosted-evidence-prep.md](./artifacts/day13-hosted-evidence-prep.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No broad state-of-the-art sparse linear algebra status, portable performance
superiority, broad benchmark-family publication, external-library performance
parity, package-manager distribution, shared-library support, dynamic ABI
stability, runtime-loader behavior, broad platform parity, release benchmark
proof, or solver correctness claim was added.

## Sprint 169 Readiness

| Future need | Sprint 168 handoff |
| --- | --- |
| Warmup and variance methodology | Decide whether `not_recorded` remains explicit metadata or becomes measured policy. |
| Matrix-size interpretation | Add selected-row matrix-size derivation if the publication contract needs dimension-aware review. |
| Report-index integration | Decide whether selected performance freshness stays as a focused checker or also receives normalized report-index publication. |
| Hosted artifact review | Inspect the first PR artifact for reviewer readability and adjust summary output only if needed. |
| Claim boundary enforcement | Keep selected performance evidence scoped to the named row, fixture, command, platform lane, and threshold-free interpretation. |

## Follow-Up Risks

| Risk | Handling |
| --- | --- |
| Hosted job fails due to GitHub Actions infrastructure. | Use the Day 13 fallback wording and do not activate hosted evidence until the job passes. |
| Hosted job exceeds the 10-minute budget. | Narrow command scope or retain local-only evidence until runtime is redesigned. |
| Hosted metadata freshness fails. | Treat artifacts as diagnostics only until `--mode hosted` passes. |
| Timing values are overread as performance guarantees. | Preserve threshold-free docs and non-claim language. |
| Unselected canonical rows are mistaken for hosted evidence. | Continue enforcing selected-row language in docs, checker output, and PR descriptions. |
