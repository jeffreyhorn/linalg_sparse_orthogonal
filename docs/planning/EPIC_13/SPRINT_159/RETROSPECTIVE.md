# Sprint 159 Retrospective

**Sprint:** 159 - Hosted Oracle And Comparison Freshness Promotion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-159`)
**Status:** Complete

## Source Artifact Note

Sprint 159 was planned from the Epic 14 project-plan section for Sprint 159,
while the requested execution package lives under
`docs/planning/EPIC_13/SPRINT_159/`. The sprint records that path mismatch in
`WORKING_NOTES.md`; the implemented plan, artifacts, and retrospective are in
the requested Epic 13 Sprint 159 path.

## Definition Of Done Checklist

- [x] Created Sprint 159 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Selected only the QR/partial-SVD oracle rows and one QR minimum-norm
      comparison family for hosted report-freshness promotion.
- [x] Measured selected oracle and comparison freshness runtime and artifact
      size before editing CI.
- [x] Added the reviewed Linux hosted freshness job in `.github/workflows/ci.yml`.
- [x] Kept macOS and Windows workflows out of Sprint 159 report-index parity
      claims.
- [x] Split hosted artifacts into `sprint159-oracle-freshness` and
      `sprint159-comparison-qr-minnorm`.
- [x] Added deterministic hosted summaries for selected oracle and comparison
      row counts, pass counts, commit/branch, support tier, fixture, and
      optional dependency context.
- [x] Tightened normalizer semantics so selected current-commit generated rows
      report `fresh` while missing, stale, failed, skipped, deferred,
      duplicate, unexpected, or incomplete selected rows fail clearly.
- [x] Added focused selected-comparison normalizer tests.
- [x] Aligned README, maintainer guide, corpus README, and solver-selection
      docs with the selected hosted evidence surface.
- [x] Preserved non-claims for broad QR, broad partial-SVD, external-library
      parity, broad platform support, package, ABI, performance, release, and
      state-of-the-art evidence.
- [x] Ran and recorded the final targeted validation set.

## What Went Well

1. **The promoted surface stayed narrow.** Sprint 159 promoted only selected
   QR/partial-SVD oracle rows and one QR minimum-norm comparison family instead
   of broad report-index freshness.

2. **Runtime and artifact size were measured before CI changes.** Day 4 showed
   the selected gates fit a 15-minute hosted job budget with small artifacts,
   avoiding a speculative hosted lane.

3. **The hosted reviewer path is inspectable.** The new Linux job runs the two
   maintained Make targets, prints deterministic summaries, and uploads split
   artifact groups with 7-day retention and strict missing-file behavior.

4. **Normalizer semantics now match the claim.** Selected current-commit rows
   report `fresh`; stale, missing, invalid, duplicate, unexpected, skipped, or
   deferred selected rows are no longer easy to confuse with pass evidence.

5. **Comparison coverage caught up to the promoted surface.** The new focused
   tests cover selected comparison complete, missing, stale, duplicate,
   unexpected, failed, and deferred row behavior.

6. **Docs distinguish row metadata from hosted execution.** Generated rows can
   remain fixture-local and local-only in metadata while the reviewed Linux job
   proves selected hosted execution.

7. **Sprint 160 has a concrete handoff.** The next comparison expansion should
   close one additional QR comparison family end to end rather than broadening
   parity language.

## What Didn't Go Well

1. **The plan path was inconsistent.** The prompt referenced Epic 12, the
   Sprint 159 plan section existed in Epic 14, and the requested execution path
   was Epic 13. The sprint had to record and work around that mismatch.

2. **The first normalizer wording was ambiguous.** Before Day 10, selected
   oracle rows could pass the required gate while still emitting generic
   `generated_present_unchecked` warnings. That was corrected, but it required
   a separate semantics audit.

3. **Support-tier wording remains subtle.** The hosted lane provides reviewed
   Linux execution, but generated row metadata still says `local_only`. The
   docs now explain the distinction, yet reviewers still need to read that
   boundary carefully.

4. **No hosted CI result exists until PR execution.** Local validation and YAML
   checks passed, but the actual hosted Ubuntu job must still run on the PR.

5. **The artifact trail is large.** The full 14-day trail is useful, but the
   shortest reviewer path is Day 14 closeout, Day 13 hosted readiness, Day 12
   validation, and Day 10 semantics implementation.

## Final Metrics

### Validation

| Metric | Sprint 159 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no |
| extra C lint run | passed on Day 10: `make lint` |
| selected oracle freshness | passed: `make report-index-oracle-freshness` |
| selected comparison freshness | passed: `make report-index-comparison-freshness` |
| focused normalizer tests | passed: `python3 tests/test_normalize_report_index.py` |
| Python syntax compile | passed |
| docs check | passed: `make docs-check` |
| workflow YAML parse | passed |
| final `git diff --check` | passed |
| trailing-whitespace scans | passed |

### Hosted Evidence Surface

| Metric | Sprint 159 close state |
| --- | --- |
| new hosted jobs | 1 |
| hosted job name | `Linux reviewed hosted oracle/comparison freshness` |
| hosted timeout | 15 minutes |
| hosted selected commands | 2 |
| hosted artifact groups | 2 |
| artifact retention | 7 days |
| selected oracle normalized rows | 54 |
| selected generated oracle pass rows | 52 |
| selected comparison normalized rows | 7 |
| selected generated comparison pass rows | 6 |
| broad report-index artifacts uploaded | 0 |
| macOS/Windows workflow report-index changes | 0 |

### Artifact Package

| Metric | Sprint 159 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| workflow files changed | 1 |
| Python scripts changed | 1 |
| focused Python test files changed | 1 |
| public docs changed | 4 |
| generated build/report artifacts committed | 0 |

## Closed Claim

Sprint 159 closes this hosted report-freshness promotion claim:

The project now has a reviewed Linux hosted CI lane that runs the maintained
selected oracle and QR minimum-norm comparison freshness gates, emits
deterministic reviewer summaries, uploads bounded split artifacts, and uses
normalizer semantics/tests that prevent stale, missing, failing, skipped,
deferred, duplicate, unexpected, or incomplete selected rows from silently
passing. Public and maintainer documentation describe exactly this selected
evidence surface and preserve unsupported-claim boundaries.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-promotion-boundary.md](./artifacts/day1-promotion-boundary.md);
- [day2-family-selection-register.md](./artifacts/day2-family-selection-register.md);
- [day3-runtime-measurement-plan.md](./artifacts/day3-runtime-measurement-plan.md);
- [day4-runtime-budget-evidence.md](./artifacts/day4-runtime-budget-evidence.md);
- [day5-ci-surface-design.md](./artifacts/day5-ci-surface-design.md);
- [day6-hosted-freshness-implementation.md](./artifacts/day6-hosted-freshness-implementation.md);
- [day7-artifact-publication-design.md](./artifacts/day7-artifact-publication-design.md);
- [day8-artifact-publication-implementation.md](./artifacts/day8-artifact-publication-implementation.md);
- [day9-normalizer-semantics-audit.md](./artifacts/day9-normalizer-semantics-audit.md);
- [day10-normalizer-semantics-implementation.md](./artifacts/day10-normalizer-semantics-implementation.md);
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md);
- [day12-local-validation.md](./artifacts/day12-local-validation.md);
- [day13-hosted-readiness.md](./artifacts/day13-hosted-readiness.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 160 Readiness

Sprint 160 should begin from these settled Sprint 159 boundaries:

| Starting item | Required posture |
| --- | --- |
| Hosted report freshness | Linux reviewed hosted evidence exists for selected oracle and selected QR minimum-norm comparison gates only. |
| Oracle artifact publication | Split artifact group `sprint159-oracle-freshness`, 7-day retention, selected rows only plus generated-reference context. |
| Comparison artifact publication | Split artifact group `sprint159-comparison-qr-minnorm`, 7-day retention, six selected rows only plus dependency context. |
| Normalizer semantics | Selected current generated rows report `fresh`; invalid selected rows fail. |
| Optional dependencies | NumPy/SciPy defers remain context and are not pass evidence. |
| Broad report index | Still advisory/local; not uploaded as hosted proof. |
| Platform scope | Linux reviewed hosted execution only; no macOS/Windows report-index parity. |

Recommended Sprint 160 first step:

Choose one additional QR comparison family, preferably an overdetermined
compatible QR least-squares fixture with residual and solution checks against
the source-controlled dense helper. Define exact selected row IDs, artifact
paths, normalizer tests, runtime budget, summary fields, and non-claims before
editing CI.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 159 close:

- broad external-library QR parity;
- broad QR or partial-SVD correctness;
- hosted report-index freshness for unselected families;
- macOS and Windows report-index parity;
- optional NumPy/SciPy promoted comparison pass evidence;
- package-manager, shared-library, ABI, dynamic-loader, and install proof from
  report freshness;
- performance superiority or release evidence;
- hosted generated API HTML publication from Sprint 158;
- state-of-the-art sparse linear algebra claims.

Still consciously constrained rather than silently solved:

- generated row metadata remains fixture-local/local-only while reviewed Linux
  hosted execution is recorded separately;
- generated report files under `build/` remain ignored;
- source-controlled report-family metadata remains advisory context, not pass
  evidence by itself;
- hosted summaries aid review but do not replace the selected generator and
  normalizer commands.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-promotion-boundary.md](./artifacts/day1-promotion-boundary.md)
- [day2-family-selection-register.md](./artifacts/day2-family-selection-register.md)
- [day3-runtime-measurement-plan.md](./artifacts/day3-runtime-measurement-plan.md)
- [day4-runtime-budget-evidence.md](./artifacts/day4-runtime-budget-evidence.md)
- [day5-ci-surface-design.md](./artifacts/day5-ci-surface-design.md)
- [day6-hosted-freshness-implementation.md](./artifacts/day6-hosted-freshness-implementation.md)
- [day7-artifact-publication-design.md](./artifacts/day7-artifact-publication-design.md)
- [day8-artifact-publication-implementation.md](./artifacts/day8-artifact-publication-implementation.md)
- [day9-normalizer-semantics-audit.md](./artifacts/day9-normalizer-semantics-audit.md)
- [day10-normalizer-semantics-implementation.md](./artifacts/day10-normalizer-semantics-implementation.md)
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md)
- [day12-local-validation.md](./artifacts/day12-local-validation.md)
- [day13-hosted-readiness.md](./artifacts/day13-hosted-readiness.md)
- [day14-closeout.md](./artifacts/day14-closeout.md)
