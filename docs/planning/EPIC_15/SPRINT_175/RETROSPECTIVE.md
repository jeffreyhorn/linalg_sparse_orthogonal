# Sprint 175 Retrospective

**Sprint:** 175 - Cross-Platform Report Freshness Promotion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-175`)
**Status:** Complete

## Source Artifact Note

Sprint 175 was executed from the active Epic 15 project-plan section for
Sprint 175 and lives under `docs/planning/EPIC_15/SPRINT_175/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 175 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Inventoried generated report freshness commands, selected report
      families, generated-output staging, and platform support boundaries.
- [x] Built a cross-platform report freshness matrix for Linux, macOS,
      Windows, local-only, hosted, artifact-only, and blocked states.
- [x] Selected macOS selected comparison freshness as the single complete
      promotion lane for the sprint.
- [x] Reconciled Linux selected comparison freshness so the hosted summary and
      artifact upload include the Sprint 174 LU target.
- [x] Added a reviewed macOS selected comparison freshness workflow lane that
      runs `make report-index-comparison-freshness`.
- [x] Added workflow guards for Linux/macOS selected comparison target
      inventory, expected row counts, artifact paths, fail-closed uploads, and
      macOS non-claims.
- [x] Updated README, maintainer guide, benchmark docs, corpus docs, and the
      report-family manifest with bounded Linux/macOS selected comparison
      freshness wording.
- [x] Added report-index manifest tests that preserve generated-local support
      tiers and prevent hosted workflow evidence from being misclassified as
      source-controlled generated-row support.
- [x] Ran selected freshness, external comparison, report-index, workflow,
      package deferral, static package/shared ABI deferral, and diff-hygiene
      checks.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 175 edits.

## What Went Well

1. **The sprint promoted one lane completely.** Sprint 175 avoided a broad
   cross-platform freshness claim and instead closed macOS selected comparison
   freshness end to end.

2. **The selected lane reused the maintained local target.** Linux and macOS
   hosted freshness now run the same `make report-index-comparison-freshness`
   command that maintainers use locally.

3. **The Linux LU inventory mismatch was fixed while staying in scope.** The
   hosted Linux selected comparison summary and artifact upload now include
   `lu-nonsym-square-5`, matching the Sprint 174 local freshness target.

4. **Workflow behavior is source-controlled and testable.** The new
   `tests/test_selected_comparison_workflow.py` guard checks target names, row
   counts, artifact paths, fail-closed upload behavior, and bounded macOS
   wording.

5. **Generated-output staging stayed clean.** The selected comparison reports
   regenerate under `build/comparison/*`, remain ignored by Git, and are
   uploaded only as hosted workflow artifacts.

6. **Documentation separates generated-local rows from hosted evidence.**
   Manifest and docs wording now make it clear that generated comparison rows
   remain `local_only` while workflow artifacts provide Linux/macOS hosted
   selected-artifact evidence.

7. **The final validation matched the changed surface.** The sprint exercised
   selected freshness, comparison runner, report-index normalization, workflow
   guards, package/ABI deferral guards, and diff hygiene.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 175 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **Linux and macOS workflow summaries duplicate target metadata.** The
   workflow guard catches drift, but future target additions still require
   synchronized edits across both workflow scripts and the test.

3. **Artifact upload paths are explicit and repetitive.** This is reviewable
   and fail-closed, but each new selected comparison target needs path updates
   in workflows and tests.

4. **Windows report freshness remains deferred.** Sprint 175 did not design a
   Windows-safe report freshness execution model, so no Windows freshness
   claim was added.

5. **Selected oracle freshness did not move beyond Linux.** The macOS
   promotion covers selected comparison freshness only; oracle freshness still
   has Linux hosted evidence only.

6. **Hosted evidence is not public report publication.** The sprint improved
   hosted CI artifact evidence but did not publish generated reports to a
   stable public URL or release artifact.

7. **Claim scans still require careful interpretation.** Broad platform terms
   appear correctly in non-claims and support-tier boundaries, so scan results
   still need human review.

## Final Metrics

### Validation

| Metric | Sprint 175 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| selected comparison freshness target | passed: `make report-index-comparison-freshness` |
| comparison runner test | passed: `python3 tests/test_run_external_comparison.py` |
| report-index normalization test | passed: `python3 tests/test_normalize_report_index.py` |
| selected comparison workflow guard | passed: `python3 tests/test_selected_comparison_workflow.py` |
| comparison runner self-check | passed: `python3 scripts/run_external_comparison.py --self-check` |
| direct comparison freshness check | passed: `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` |
| selected comparison freshness rows | passed: `normalize-report-index: freshness ok (32 rows)` |
| package-manager deferral guard | passed |
| static package/shared ABI deferral guard | passed |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 175 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 0 |
| GitHub workflow files changed | 2 |
| workflow guard tests added | 1 |
| report-index tests changed | 1 |
| selected comparison hosted platforms | 2 |
| selected comparison targets in hosted Linux/macOS summaries | 4 |
| selected comparison generated rows checked by freshness | 28 |
| selected comparison contract rows checked by freshness | 4 |
| selected comparison rows checked by freshness | 32 |
| selected comparison generated files regenerated locally | 24 |
| public/maintainer docs changed | 4 |
| report-family manifest rows changed | 1 |
| daily artifacts under `SPRINT_175/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 175 close state |
| --- | ---: |
| new reviewed macOS selected comparison freshness lanes | 1 |
| Linux selected comparison hosted inventory reconciliations | 1 |
| Windows report freshness claims added | 0 |
| selected oracle macOS freshness claims added | 0 |
| hosted publication claims for all generated reports added | 0 |
| hosted generated API HTML publication claims added | 0 |
| broad report-index freshness claims added | 0 |
| unselected comparison family freshness claims added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI support claims added | 0 |
| runtime-loader support claims added | 0 |
| release evidence claims added | 0 |
| performance superiority claims added | 0 |
| external-library ecosystem parity claims added | 0 |
| state-of-the-art sparse linear algebra claims added | 0 |

## Closed Claim

Sprint 175 closes this Epic 15 cross-platform report freshness claim:

The project now has a reviewed macOS selected comparison freshness lane in
addition to local freshness and reviewed Linux hosted selected comparison
evidence. The maintained command
`make report-index-comparison-freshness` regenerates the four selected
comparison families, and both Linux and macOS hosted workflows summarize and
upload selected generated artifacts for:

- `qr-minnorm`;
- `qr-compatible-ls`;
- `partial-svd-diag6-k2`;
- `lu-nonsym-square-5`.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-freshness-intake.md](./artifacts/day1-freshness-intake.md);
- [day2-generated-report-inventory.md](./artifacts/day2-generated-report-inventory.md);
- [day3-platform-gap-matrix.md](./artifacts/day3-platform-gap-matrix.md);
- [day4-promotion-decision.md](./artifacts/day4-promotion-decision.md);
- [day5-path-execution-audit.md](./artifacts/day5-path-execution-audit.md);
- [day6-normalization-design.md](./artifacts/day6-normalization-design.md);
- [day7-normalization-implementation.md](./artifacts/day7-normalization-implementation.md);
- [day8-gate-integration.md](./artifacts/day8-gate-integration.md);
- [day9-documentation-tier-update.md](./artifacts/day9-documentation-tier-update.md);
- [day10-report-index-reconciliation.md](./artifacts/day10-report-index-reconciliation.md);
- [day11-cross-platform-claim-review.md](./artifacts/day11-cross-platform-claim-review.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-maintenance-review.md](./artifacts/day13-maintenance-review.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No Windows report freshness, selected oracle macOS freshness, hosted
publication of all generated reports, hosted generated API HTML publication,
broad report-index freshness, unselected comparison family freshness,
package-manager support, shared-library ABI support, runtime-loader behavior,
release evidence, performance superiority, external-library ecosystem parity,
or state-of-the-art sparse linear algebra claim was added.

## Sprint 176 Readiness

| Future need | Sprint 175 handoff |
| --- | --- |
| Epic 15 final validation | Treat Sprint 175 as bounded local plus Linux/macOS selected comparison evidence only. |
| Selected comparison freshness | Run `make report-index-comparison-freshness` before relying on selected local generated comparison rows. |
| Hosted comparison artifacts | Inspect Linux/macOS workflow artifacts for selected generated files; do not treat them as stable public report publication. |
| Windows report freshness | Design a Windows-safe execution model separately; do not infer support from macOS. |
| Selected oracle freshness on macOS | Keep Linux-only hosted oracle wording until a separate macOS oracle lane exists. |
| Workflow maintainability | Factor duplicated Linux/macOS summary logic before adding more selected comparison targets or platforms. |
| Generated comparison output staging | Keep `build/comparison/*` generated outputs ignored and local unless a later sprint explicitly promotes hosted or release publication. |
| Package-manager wording changes | Run `bash scripts/package_manager_deferral_check.sh`. |
| Static package/shared ABI wording changes | Run `bash scripts/static_package_deferral_check.sh`. |
