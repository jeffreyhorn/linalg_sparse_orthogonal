# Sprint 156 Day 2 Evidence Inventory

## Purpose

Day 2 converts the Sprint 147-155 source inventory into a final Epic 13
evidence matrix. The goal is to identify what evidence exists, what support
tier it belongs to, which claims it can support, which gaps remain, and which
Sprint 156 validation day must reconcile it before final closeout.

## Evidence Tier Legend

| Tier | Meaning For Sprint 156 |
| --- | --- |
| Reviewed | Evidence expected to be enforced or reviewed through repository gates, PR CI, or maintained proof-owner commands. |
| Supplemental | Useful confidence evidence that does not by itself widen public support claims. |
| Local | Evidence recorded from local commands or artifacts; must not be treated as hosted multi-platform proof. |
| Generated-local | Generated rows or reports produced locally and meaningful only with the selected freshness/normalization command. |
| Source-controlled | Checked-in metadata, fixtures, or docs that describe intent but are not fresh generated proof by themselves. |
| Staged | Known work that remains outside the reviewed claim until promoted. |
| Deferred | Explicitly not closed in Epic 13; must carry owner, blocker, and promotion gate forward. |

## Final Evidence Matrix

| Area | Sprint Source | Evidence Available | Tier | Claim It Can Support | Sprint 156 Follow-Up |
| --- | --- | --- | --- | --- | --- |
| Epic 13 baseline and claim gates | Sprint 147 | Selected gap register, candidate claim register, Windows/corpus/report/package/comparison evidence gates, quality surface map, and public claim freeze audit. | Source-controlled planning evidence | Epic 13 had an evidence contract before implementation sprints began. | Day 10 should compare final public claims against the Sprint 147 freeze and Day 6 claim targets. |
| Windows staged test portability | Sprint 148 | Ported/promoted staged Windows CMake CTest coverage, updated expected Windows count, and documented retained staged blockers. | Reviewed after PR CI; local during sprint closeout | Windows reviewed CMake-first test surface is broader than pre-Epic 13. | Day 6 must reconcile final Windows CI count and staged exclusions. |
| Windows install-validation parity decision | Sprint 149 | Reviewed Windows CMake install/downstream validation lane, metadata checks, exact-version package checks, and support wording. | Reviewed after PR CI; local during sprint closeout | Windows supports the maintained CMake-first static package install/downstream proof. | Day 5 and Day 6 must confirm current workflow and docs still preserve Windows Makefile and `pkg-config` non-claims. |
| QR maintained corpus | Sprint 150 | Six-fixture QR corpus family, proof-owner test coverage, expected report rows, documentation alignment, and generated-local report behavior. | Reviewed tests plus generated-local reports | QR has bounded maintained corpus coverage for named fixture families. | Day 7 must check current QR corpus/report rows and freshness expectations. |
| Partial-SVD maintained corpus | Sprint 151 | Four-fixture partial-SVD corpus family, proof-owner test coverage, expected report rows, documentation alignment, and residual repeated-spectrum deferrals. | Reviewed tests plus generated-local reports | Partial-SVD has bounded maintained corpus coverage for named fixture families. | Day 7 must check current partial-SVD corpus/report rows and freshness expectations. |
| Generated report freshness | Sprint 152 | Selected oracle freshness policy, report normalization semantics, `make report-index-oracle-freshness`, and deferred non-selected report families. | Generated-local with source-controlled policy | Selected oracle report rows can be cited only after the required freshness gate. | Day 7 must run or review the selected freshness commands and identify stale/non-selected families. |
| Static-first package and ABI posture | Sprint 153 | Product decision to keep static-first packaging, install/export proof, downstream proof, CMake/shared-library rejection behavior, and package/ABI docs. | Reviewed/local package proof depending on platform lane | Static-first package contract is maintained; shared-library ABI remains deferred. | Day 5 must validate package metadata and install/downstream proof; Day 10 must audit package/ABI wording. |
| External comparison harness | Sprint 154 | First narrow `qr-minnorm` external-process dense-reference comparison, output schema, dependency policy, generated comparison report rows, and study artifact. | Generated-local/local comparison evidence | One named QR minimum-norm fixture has a narrow local comparison study. | Day 8 must verify comparison freshness/provenance and block broad ecosystem parity wording. |
| Adoption and API documentation | Sprint 155 | Reworked tutorial, API reference index, maintainer guide policy, selected header cleanup, and declaration-preservation scans. | Documentation evidence plus reviewed full C gate after public-header edits | Adoption path and selected public header docs are more coherent without declaration drift. | Day 9 must reconcile docs/API/header surfaces; Day 10 must audit final claims. |

## Deliverable Coverage

| Sprint 156 Deliverable | Current Evidence Source | Coverage Status |
| --- | --- | --- |
| Final Epic 13 evidence inventory | This Day 2 artifact plus Sprint 147-155 retrospectives and artifacts | Started and ready for Day 3 validation mapping |
| Final validation package | Sprint 147 quality map, Sprint 148-155 validation artifacts, and current repo commands | Needs Day 3-5 command matrix and execution/review |
| Public claim/non-claim audit | Sprint 147 claim freeze, Sprint 155 claim hygiene handoff, public docs | Needs Day 10 final audit |
| Residual queue with promotion gates | Prior sprint residual sections and Day 2 missing/stale list | Needs Day 11 consolidation |
| Epic 13 retrospective | Sprint 147-155 retrospectives plus Sprint 156 validation artifacts | Needs Day 12 draft |
| Next-epic handoff | Sprint 155 handoff and Day 11 residual queue | Needs Day 13-14 reconciliation |

## Missing Or Stale Evidence List

| Gap | Current Status | Why It Matters | Follow-Up |
| --- | --- | --- | --- |
| Final hosted CI reconciliation | Not captured in Day 2 | Several sprint retrospectives closed before PR CI existed or before final merge evidence was available. | Day 6 should reconcile final GitHub Actions outcomes where available. |
| Final local quality baseline | Not yet run for Sprint 156 | Day 2 is inventory-only; final closeout still needs current repository validation. | Day 3 should define commands; Day 4 should run the selected local baseline. |
| Package/install proof freshness | Needs current confirmation | Package metadata and install behavior can drift after Sprint 153. | Day 5 should run or review Make/CMake install and downstream proof. |
| QR and partial-SVD generated rows | Need freshness interpretation | Source-controlled rows and generated-local rows are not equivalent to fresh report evidence. | Day 7 should check corpus/report freshness and stale rows. |
| Comparison report freshness | Needs current confirmation | Sprint 154 comparison evidence is narrow and generated-local. | Day 8 should check comparison outputs, dependency status, and report freshness. |
| Generated API HTML refresh | Deferred from Sprint 155 | Current `docs/api/html/` is not a complete fresh reference surface. | Day 9 should preserve the publication-policy boundary; Day 11 should retain residual if not promoted. |
| Remaining public-header cleanup | Deferred from Sprint 155 | Header docs outside the selected batch may still have cleanup debt. | Day 9 should document residual; Day 11 should assign promotion gate. |
| Final public claim audit | Not yet complete | The final closeout must prevent support widening across all public docs. | Day 10 owns final scan and correction list. |

## Validation Follow-Up Queue

| Day | Validation Question | Evidence Inputs |
| --- | --- | --- |
| Day 3 | Which exact commands and skip/defer semantics make up the final validation matrix? | Sprint 147 quality map, Sprint 148-155 validation summaries, current Make/CMake targets |
| Day 4 | Does the current local baseline pass? | `make format`, `make lint`, `make test` when required, docs-only checks when not |
| Day 5 | Does static-first package/install evidence still match docs and metadata? | `tests/test_install.sh`, CMake install/downstream proof, `INSTALL.md`, `sparse.pc.in`, CMake package files |
| Day 6 | What platform evidence is reviewed, supplemental, staged, or deferred? | GitHub Actions workflows, Windows CTest count, Sprint 148-149 platform artifacts |
| Day 7 | Are QR/partial-SVD corpus rows current and bounded? | Corpus tests, oracle report rows, freshness normalization commands |
| Day 8 | Is the `qr-minnorm` comparison study fresh and narrow? | Comparison scripts, generated comparison outputs, Sprint 154 study artifact |
| Day 9 | Do adoption/API docs align with final evidence? | README, tutorial, cookbook, API reference, maintainer guide, public headers |
| Day 10 | Are all public claims evidence-bound or explicit non-claims? | Public docs, support docs, package docs, report docs, header comments |
| Day 11 | Which residuals need next-epic owners and gates? | Prior residual sections, missing/stale evidence list, Days 3-10 findings |

## Day 2 Completion Check

- Sprint 147 baseline and claim-target evidence is inventoried.
- Sprint 148 and Sprint 149 Windows portability/install evidence is
  inventoried.
- Sprint 150 and Sprint 151 QR/partial-SVD corpus evidence is inventoried.
- Sprint 152 report freshness evidence is inventoried.
- Sprint 153 package/ABI evidence and static-first boundaries are inventoried.
- Sprint 154 comparison-study evidence is inventoried.
- Sprint 155 adoption/API/header evidence is inventoried.
- Missing or stale evidence is listed.
- Validation follow-up queue is ready for Day 3.
