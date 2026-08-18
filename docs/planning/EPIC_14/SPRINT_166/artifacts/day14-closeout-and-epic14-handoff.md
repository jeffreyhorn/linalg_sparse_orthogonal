# Day 14 Closeout And Epic 14 Handoff

## Purpose

Day 14 closes Sprint 166 by publishing final changed-file notes, validation
commands, known residuals, the Epic 14 retrospective, PR description bullets,
review-risk notes, and next-epic handoff.

## Final Changed Files

Sprint 166 implementation and public-claim changes:

- `.github/workflows/ci.yml`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `docs/solver_selection.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`

Sprint 166 and Epic 14 planning/closeout artifacts:

- `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`
- `docs/planning/EPIC_14/SPRINT_166/PLAN.md`
- `docs/planning/EPIC_14/SPRINT_166/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day1-sprint-intake.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day2-generated-report-evidence-inventory.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day3-solver-package-performance-api-inventory.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day4-validation-baseline-design.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day5-local-validation-baseline.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day6-supplemental-validation-sweep.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day7-hosted-ci-evidence-reconciliation.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day8-public-claim-audit-performance-report.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day9-public-claim-audit-package-abi-windows.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day10-project-plan-reconciliation-part1.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day11-project-plan-reconciliation-part2.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day12-epic14-retrospective-draft.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day13-final-residual-queue-and-closeout-prep.md`
- `docs/planning/EPIC_14/SPRINT_166/artifacts/day14-closeout-and-epic14-handoff.md`

No `.c` files were changed during Sprint 166. The branch includes public-doc,
workflow, and planning-artifact changes, plus Sprint 166 validation records
from earlier days.

## Final Validation Summary

Strongest Sprint 166 validation was recorded before closeout:

- Day 5: `make format`, `make lint`, `make test`, corpus schema validation,
  report normalizer tests, external comparison runner tests, Python compile
  checks, and `git diff --check` passed.
- Day 6: generated API docs, selected oracle freshness, selected comparison
  freshness, report-index checks, package report checks, static package
  deferral, Make install/`pkg-config`, CMake install/export, benchmark report,
  performance sentinel, targeted claim scans, and `git diff --check` passed.
- Day 7: selected comparison freshness and hosted comparison summary/upload
  path checks passed after the workflow update.
- Day 9: `bash scripts/static_package_deferral_check.sh` passed after package
  wording cleanup.

Day 14 final documentation/touched-surface checks:

- `git diff --check`: passed.
- Targeted stale hosted-comparison wording scan: passed with no stale
  QR-minnorm-only hosted artifact wording in current public/workflow surfaces.
- Targeted unsupported package/ABI/platform claim scan: passed with matches
  classified as supported static-first wording, explicit non-claims, guard
  text, or historical Sprint artifacts.
- `.c`/`.h` changed-file scan: no current source/header diff requiring a new
  Day 14 full C quality gate.

## Epic 14 Retrospective

Sprint 166 has enough evidence to publish the final Epic 14 retrospective in
this branch:

- [`docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`](../../EPIC_14_RETROSPECTIVE.md)

The retrospective consumes the Day 12 draft and Day 13 final residual queue,
then records earned claims, retained non-claims, validation evidence,
state-of-the-art assessment, and next-epic handoff.

## Final Residual Queue

The final residual queue remains the Day 13 queue:

- [`day13-final-residual-queue-and-closeout-prep.md`](day13-final-residual-queue-and-closeout-prep.md)

Highest-priority residuals:

| Priority | Residual | Promotion gate |
| --- | --- | --- |
| P0 | Sprint 166 PR-hosted CI confirmation. | PR Linux/macOS/Windows checks pass, or failures are reconciled and fixed before merge. |
| P1 | Hosted performance publication proof. | Reviewed hosted lane runs selected benchmark/sentinel publication checks and uploads methodology-bound artifacts without superiority wording. |
| P1 | Shared-library ABI product design. | Shared-library builds install and pass platform-specific downstream consumers with ABI/loader docs and CI proof, or static-only support is reaffirmed with guards. |
| P1 | Package-manager distribution readiness. | Selected package-manager recipes install, compile/link/run downstream consumers, validate metadata/version behavior, and publish support-tier docs. |
| P1 | Broader public-header cleanup batch. | Declaration-preserving cleanup lands for selected headers, generated docs pass, public docs stay coherent, and required C quality gates pass. |

## PR Description Bullets

Suggested PR summary:

- Add Sprint 166 final-validation plan, working notes, and daily evidence
  artifacts through Epic 14 closeout.
- Publish `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` with earned claims,
  retained non-claims, validation evidence, state-of-the-art assessment, and
  residual handoff.
- Reconcile Epic 14 generated API, hosted generated evidence, QR comparison,
  partial-SVD comparison, Windows package, performance, public-header, and
  static-first package outcomes against the project plan.
- Update reviewed Linux hosted comparison workflow wording, summary, and
  artifact upload paths so selected comparison evidence covers QR min-norm,
  QR compatible least-squares, and partial-SVD diag6 k2 families.
- Tighten public documentation wording so selected comparison rows are local
  generated evidence by default and reviewed Linux hosted evidence only after
  the hosted report-freshness lane runs.
- Tighten package wording around Windows `pkg-config` command execution parity
  while preserving static-first package and ABI non-claims.

Suggested validation:

- `make format`
- `make lint`
- `make test`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_run_external_comparison.py`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py scripts/run_corpus_oracle.py`
- `make docs-check`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `python3 scripts/normalize_report_index.py --check`
- `python3 scripts/normalize_report_index.py --family package --check`
- `python3 scripts/normalize_report_index.py --family package --check-freshness`
- `bash scripts/static_package_deferral_check.sh`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make bench-canonical-report`
- `make performance-sentinels`
- targeted public/workflow claim scans
- `git diff --check`

## Review-Risk Notes

| Risk | Review focus |
| --- | --- |
| Hosted comparison evidence could be overread as broad external parity. | Check workflow, README, maintainer guide, solver-selection docs, corpus docs, and report-index schema wording for selected-family and fixture-local boundaries. |
| Sprint 159 historical hosted scope could conflict with Sprint 160/161 selected-family growth. | Current selected-comparison hosted claims should cite Sprint 166 Day 7. |
| Local generated rows could be mistaken for hosted or release proof. | Check docs distinguish local generated output, source-controlled metadata, advisory rows, selected required rows, and hosted artifact evidence. |
| Package wording could imply shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows support. | Review `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, `sparse.pc.in`, CMake comments, and static package guard results. |
| Performance report wording could imply superiority. | Confirm benchmark/sentinel rows remain local, methodology-bound, and non-superiority unless future hosted methodology exists. |
| Public-header cleanup could be treated as complete for all headers. | Keep Sprint 164 claims limited to `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`; broader header cleanup remains residual. |
| Branch-level hosted CI remains unproven until PR workflows run. | PR description and final closeout should say local validation passed and hosted evidence is pending until CI completes. |

## Next-Epic Handoff

Recommended next-epic starting points:

1. Confirm Sprint 166 PR-hosted CI and reconcile any hosted failures before
   merge.
2. Decide whether hosted performance publication is worth promoting or should
   remain local-only.
3. Decide whether shared-library ABI support is a product goal or should stay
   static-first with stronger guards.
4. Define package-manager distribution scope before claiming ecosystem
   install readiness.
5. Continue declaration-preserving public-header cleanup for the next selected
   high-risk header batch.
6. Add one additional bounded comparison family only with fixture, reference,
   metric, tolerance, row, docs, and validation ownership defined up front.

## Completion Criteria

- Epic 14 final validation and claim recalibration are complete or explicitly
  residualized.
- Public claims and non-claims are evidence-bounded.
- Epic 14 retrospective is published in this branch.
- Residuals have promotion gates rather than vague aspirations.
- Branch is ready for review with validation evidence and closeout handoff,
  subject to PR-hosted CI confirmation after push.
