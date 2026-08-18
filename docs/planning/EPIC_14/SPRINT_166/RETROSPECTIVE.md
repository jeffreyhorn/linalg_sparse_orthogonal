# Sprint 166 Retrospective

**Sprint:** 166 - Epic 14 Final Validation, Claim Recalibration & Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-166`)
**Status:** Complete

## Source Artifact Note

Sprint 166 was executed from the Epic 14 project-plan section for Sprint 166
and lives under `docs/planning/EPIC_14/SPRINT_166/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 project-plan path; `WORKING_NOTES.md` records that
path mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 166 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Inventoried Epic 14 evidence for generated API docs, hosted
      oracle/comparison freshness, QR comparison, partial-SVD comparison,
      Windows package decision, performance publication, public-header/API
      cleanup, and static-first package boundary hardening.
- [x] Designed the final validation baseline and ran the strongest feasible
      local baseline: `make format`, `make lint`, `make test`, corpus schema
      validation, report normalizer tests, comparison runner tests, Python
      compile checks, and `git diff --check`.
- [x] Ran supplemental generated docs, selected report freshness, package,
      install/export, benchmark, performance-sentinel, report-index, and
      targeted claim-boundary checks.
- [x] Reconciled reviewed Linux hosted comparison workflow wording, summary,
      and upload paths with the current selected comparison family set.
- [x] Audited public claim wording for state-of-the-art, external-parity,
      portable-performance, hosted-publication, generated-report,
      package-manager, shared-library, dynamic ABI, runtime-loader, Windows
      parity, and installed-package overreach.
- [x] Reconciled every Epic 14 project-plan item for Sprints 157-166 as
      complete, narrowed, deferred, residualized, or pending PR-hosted
      confirmation with evidence links.
- [x] Published `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`.
- [x] Published the final residual queue, PR description bullets,
      review-risk notes, and next-epic handoff.
- [x] Preserved explicit non-claims for broad sparse linear algebra
      state-of-the-art status, external-library parity, portable performance,
      backend superiority, package-manager distribution, shared-library
      support, dynamic ABI compatibility, runtime-loader behavior, broad
      Windows package parity, Windows Makefile parity, and Windows
      `pkg-config` execution parity.

## What Went Well

1. **The final evidence inventory was broad but bounded.** Days 1-3 pulled in
   Sprints 157-165 without treating earlier closeout language as broader proof
   than the artifacts supported.

2. **The local validation baseline was strong.** Day 5 reran the full local
   quality gate and focused report/corpus Python checks, giving the closeout a
   fresh baseline before public claim work.

3. **Supplemental generated evidence stayed classified.** Day 6 validated
   generated API docs, selected report freshness, package checks, benchmark
   reports, and performance sentinels while keeping generated outputs local
   unless a reviewed hosted path owned them.

4. **The hosted comparison workflow mismatch was closed.** Day 7 updated the
   reviewed Linux hosted comparison lane so its name, summary, and upload
   paths match the selected QR min-norm, QR compatible least-squares, and
   partial-SVD diag6 k2 family set.

5. **Public claim wording became more precise.** Days 8 and 9 clarified when
   selected comparison rows are local generated evidence versus reviewed Linux
   hosted evidence, and tightened Windows `pkg-config` wording to command
   execution parity.

6. **Project-plan reconciliation separated completion from non-claims.** Days
   10 and 11 marked retained product decisions as complete without implying
   hidden support for generated API hosting, Windows package parity,
   shared-library support, dynamic ABI, package-manager distribution, or broad
   numerical parity.

7. **Epic 14 closed with an explicit retrospective and residual queue.** Days
   12-14 produced the retrospective draft, actionable residual queue, final
   Epic 14 retrospective, PR material, and next-epic handoff.

## What Didn't Go Well

1. **The sprint prompt path was stale.** The request referenced Epic 12 while
   the active Sprint 166 plan lives under Epic 14. The sprint recorded the
   mismatch and proceeded from the current Epic 14 plan.

2. **Historical hosted comparison wording needed a later correction.** Sprint
   159 created the first hosted comparison lane, but Sprint 160 and Sprint 161
   later expanded the selected comparison family set. Sprint 166 had to close
   that artifact-scope mismatch explicitly.

3. **Claim scans are noisy by design.** Unsupported-claim terms appear in
   guard checks, non-claims, historical artifacts, and public docs. The sprint
   needed targeted interpretation rather than treating every match as a bug.

4. **Hosted proof still cannot be finalized before PR CI.** Sprint 166 can
   validate local workflow structure and claim wording, but branch-specific
   hosted Linux/macOS/Windows proof remains pending until the PR runs.

5. **Epic 14 deliberately leaves large product surfaces out of scope.** The
   closeout improves evidence discipline, but shared-library support, dynamic
   ABI, runtime-loader behavior, package-manager distribution, hosted
   performance publication, broader header cleanup, and broad competitive
   parity remain future work.

## Final Metrics

### Validation

| Metric | Sprint 166 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| strongest local baseline | passed: `make format && make lint && make test` on Day 5 |
| corpus schema validation | passed |
| report normalizer tests | passed |
| external comparison runner tests | passed |
| Python compile checks | passed |
| generated API docs check | passed: 18 checked-in public headers, 18 reference pages, 18 source pages |
| selected oracle freshness | passed: 54 rows |
| selected comparison freshness | passed: QR min-norm, QR compatible LS, partial-SVD diag6 k2; 25 freshness rows |
| normalized report index | passed: 176 rows |
| package report-index checks | passed: 6 rows and freshness |
| static package deferral guard | passed |
| Make install/`pkg-config` validation | passed: 23 checks, 0 failures |
| CMake install/export validation | passed: 27 checks, 0 failures, 0 skips |
| benchmark report generation | passed |
| performance sentinels | passed |
| targeted claim scans | passed; matches were non-claims, guarded boundaries, or historical artifacts |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 166 close state |
| --- | ---: |
| workflow files changed | 1 |
| public/maintainer docs changed | 6 |
| C source files changed | 0 |
| public header files changed | 0 |
| Sprint 166 daily artifacts | 14 |
| Sprint 166 plan files | 1 |
| Sprint 166 working notes files | 1 |
| Sprint 166 retrospective files | 1 |
| Epic retrospective files published | 1 |
| generated build/docs/report/cache artifacts committed | 0 |

### Claim Governance

| Metric | Sprint 166 close state |
| --- | ---: |
| Epic 14 sprints reconciled | 10 |
| Epic 14 project-plan item groups reconciled | 70 |
| final Epic 14 success criteria reconciled | 8 |
| final residual queue entries | 18 |
| broad state-of-the-art claims added | 0 |
| broad external-library parity claims added | 0 |
| portable performance superiority claims added | 0 |
| shared-library or dynamic ABI claims added | 0 |
| Windows Makefile or Windows `pkg-config` execution parity claims added | 0 |

## Closed Claim

Sprint 166 closes this final Epic 14 validation and claim recalibration claim:

Epic 14 now has a source-controlled final evidence package that inventories and
validates its selected generated API, hosted generated evidence, QR comparison,
partial-SVD comparison, Windows package, performance publication, public-header
coherence, and static-first package outcomes; reconciles every Epic 14
project-plan item against actual evidence; publishes
`docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`; and leaves a prioritized
residual queue with owners, blockers, prerequisites, and promotion gates.

The Sprint 166 branch also updates current public and workflow wording so the
selected comparison hosted evidence claim matches the maintained selected
family set, and so package/Windows wording remains bounded to static-first and
CMake-first evidence.

No broad state-of-the-art sparse linear algebra status, broad external-library
parity, portable performance superiority, backend superiority,
package-manager distribution, shared-library support, dynamic ABI
compatibility, runtime-loader behavior, broad Windows package parity, Windows
Makefile parity, or Windows `pkg-config` command execution parity was added.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-generated-report-evidence-inventory.md](./artifacts/day2-generated-report-evidence-inventory.md);
- [day3-solver-package-performance-api-inventory.md](./artifacts/day3-solver-package-performance-api-inventory.md);
- [day4-validation-baseline-design.md](./artifacts/day4-validation-baseline-design.md);
- [day5-local-validation-baseline.md](./artifacts/day5-local-validation-baseline.md);
- [day6-supplemental-validation-sweep.md](./artifacts/day6-supplemental-validation-sweep.md);
- [day7-hosted-ci-evidence-reconciliation.md](./artifacts/day7-hosted-ci-evidence-reconciliation.md);
- [day8-public-claim-audit-performance-report.md](./artifacts/day8-public-claim-audit-performance-report.md);
- [day9-public-claim-audit-package-abi-windows.md](./artifacts/day9-public-claim-audit-package-abi-windows.md);
- [day10-project-plan-reconciliation-part1.md](./artifacts/day10-project-plan-reconciliation-part1.md);
- [day11-project-plan-reconciliation-part2.md](./artifacts/day11-project-plan-reconciliation-part2.md);
- [day12-epic14-retrospective-draft.md](./artifacts/day12-epic14-retrospective-draft.md);
- [day13-final-residual-queue-and-closeout-prep.md](./artifacts/day13-final-residual-queue-and-closeout-prep.md);
- [day14-closeout-and-epic14-handoff.md](./artifacts/day14-closeout-and-epic14-handoff.md);
- [../EPIC_14_RETROSPECTIVE.md](../EPIC_14_RETROSPECTIVE.md).

## Epic 14 Closeout State

| Success criterion | Sprint 166 close state |
| --- | --- |
| Generated API reference publication is no longer ambiguous. | Complete: generated HTML remains local-only by decision, and source-header-first Doxygen coverage is maintained. |
| Selected generated oracle/comparison evidence has a reviewed hosted path. | Complete for selected Linux hosted path, subject to PR-hosted CI confirmation for this branch. |
| One bounded QR comparison family and one bounded partial-SVD comparison family are published with normalized freshness checks. | Complete with fixture-local non-parity wording. |
| Windows package parity is implemented for selected scope or retained as an explicit test-backed non-claim. | Complete as retained non-claim for Windows Makefile and Windows `pkg-config` execution parity. |
| Performance reporting has methodology-bound publication without superiority overclaiming. | Complete for local generated benchmark/sentinel reports. |
| Public headers and API docs remain declaration-preserving and coherent. | Complete for the selected Sprint 164 header batch. |
| Static-first package and ABI non-claims are hardened. | Complete. |
| Final state-of-the-art assessment maps every positive claim to recurring evidence and rejects unsupported broad claims. | Complete in `EPIC_14_RETROSPECTIVE.md`. |

## Next-Epic Readiness

Recommended next-epic starting points:

| Starting item | Required posture |
| --- | --- |
| Sprint 166 PR-hosted CI | Confirm PR Linux/macOS/Windows results before treating hosted branch evidence as final. |
| Hosted performance publication | Decide whether to add a reviewed hosted performance/report lane or retain local-only performance rows. |
| Shared-library ABI design | Treat shared-library support as a product design, ABI, loader, install, docs, and CI project, not a small build flag change. |
| Package-manager readiness | Select exact package-manager scope and provenance before claiming distribution support. |
| Public-header cleanup | Continue the Sprint 164 declaration-preserving process for the next selected header batch. |
| Additional comparison family | Add one bounded family only with fixtures, references, metrics, tolerances, row IDs, docs, and validation ownership defined up front. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 166 close:

- Sprint 166 branch-specific hosted CI confirmation until PR checks run;
- hosted performance publication proof;
- shared-library ABI product design;
- package-manager distribution readiness;
- broader public-header cleanup;
- additional bounded QR/SVD/partial-SVD comparison families;
- macOS/Windows generated report-freshness parity;
- hosted generated API HTML publication;
- generated `sparse_version.h` API-doc representation;
- statistical benchmark methodology for superiority claims;
- backend superiority or OpenMP speedup proof;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad report-index freshness for all generated families;
- optional NumPy/SciPy promoted pass evidence;
- table-wide README/API index reshaping and tutorial expansion;
- maintained declaration-preservation helper;
- broad external-library parity and state-of-the-art positioning.

Still consciously constrained rather than silently solved:

- no unqualified state-of-the-art sparse linear algebra claim;
- no broad external-library or ecosystem parity;
- no broad QR, SVD, or partial-SVD correctness;
- no proof from advisory, deferred, optional dependency, or
  source-controlled metadata rows alone;
- no hosted proof for local-only generated API HTML, benchmark/sentinel rows,
  or unselected generated report families;
- no portable performance, backend superiority, OpenMP speedup portability, or
  state-of-the-art performance evidence;
- no package-manager distribution;
- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader behavior;
- no static/shared package selector UX;
- no Windows Makefile install/uninstall parity;
- no Windows `pkg-config` command execution parity;
- no broad Windows package or platform parity.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [../EPIC_14_RETROSPECTIVE.md](../EPIC_14_RETROSPECTIVE.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-generated-report-evidence-inventory.md](./artifacts/day2-generated-report-evidence-inventory.md)
- [day3-solver-package-performance-api-inventory.md](./artifacts/day3-solver-package-performance-api-inventory.md)
- [day4-validation-baseline-design.md](./artifacts/day4-validation-baseline-design.md)
- [day5-local-validation-baseline.md](./artifacts/day5-local-validation-baseline.md)
- [day6-supplemental-validation-sweep.md](./artifacts/day6-supplemental-validation-sweep.md)
- [day7-hosted-ci-evidence-reconciliation.md](./artifacts/day7-hosted-ci-evidence-reconciliation.md)
- [day8-public-claim-audit-performance-report.md](./artifacts/day8-public-claim-audit-performance-report.md)
- [day9-public-claim-audit-package-abi-windows.md](./artifacts/day9-public-claim-audit-package-abi-windows.md)
- [day10-project-plan-reconciliation-part1.md](./artifacts/day10-project-plan-reconciliation-part1.md)
- [day11-project-plan-reconciliation-part2.md](./artifacts/day11-project-plan-reconciliation-part2.md)
- [day12-epic14-retrospective-draft.md](./artifacts/day12-epic14-retrospective-draft.md)
- [day13-final-residual-queue-and-closeout-prep.md](./artifacts/day13-final-residual-queue-and-closeout-prep.md)
- [day14-closeout-and-epic14-handoff.md](./artifacts/day14-closeout-and-epic14-handoff.md)
