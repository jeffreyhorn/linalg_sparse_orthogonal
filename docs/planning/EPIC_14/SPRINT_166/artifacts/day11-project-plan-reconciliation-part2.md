# Day 11 Project Plan Reconciliation Part 2

## Scope

Day 11 reconciles Epic 14 project-plan items for Sprints 162 through 166
against their closeout artifacts, retrospectives where available, and the
Sprint 166 evidence gathered through Day 11.

This artifact completes the project-plan reconciliation started on Day 10.
It treats retained non-claims and explicit product decisions as valid closure
only when the sprint evidence says that was the selected outcome.

## Sprint-Level Reconciliation

| Sprint | Project-plan goal | Close state | Evidence | Current interpretation |
| --- | --- | --- | --- | --- |
| 162 | Decide and close the remaining Windows package parity gap for `pkg-config` and Makefile support without confusing it with CMake install validation. | Complete with retained non-claim product decision. | [`SPRINT_162/artifacts/day14-closeout.md`](../../SPRINT_162/artifacts/day14-closeout.md), [`SPRINT_162/RETROSPECTIVE.md`](../../SPRINT_162/RETROSPECTIVE.md). | Windows package support remains CMake-first and static-first. Windows Makefile install/uninstall parity and Windows `pkg-config` command execution parity are guarded non-claims. |
| 163 | Publish methodology-bound performance/report artifacts for selected canonical benchmark and sentinel rows while preserving non-superiority claims. | Complete with local-only methodology boundary. | [`SPRINT_163/artifacts/day14-closeout.md`](../../SPRINT_163/artifacts/day14-closeout.md), [`SPRINT_163/RETROSPECTIVE.md`](../../SPRINT_163/RETROSPECTIVE.md). | Canonical benchmark and sentinel rows became more reviewable, but they do not provide hosted performance proof, superiority evidence, or state-of-the-art performance claims. |
| 164 | Complete a declaration-preserving public-header cleanup batch and keep user-facing API docs coherent. | Complete for the selected public-header batch. | [`SPRINT_164/artifacts/day14-closeout.md`](../../SPRINT_164/artifacts/day14-closeout.md), [`SPRINT_164/RETROSPECTIVE.md`](../../SPRINT_164/RETROSPECTIVE.md). | `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h` were cleaned with declaration-preservation proof; broader header cleanup remains residual. |
| 165 | Harden the static-first package boundary so shared-library, dynamic ABI, runtime-loader, and package-manager non-claims cannot drift. | Complete with hardened static-first non-claim guards. | [`SPRINT_165/artifacts/day14-closeout-and-handoff.md`](../../SPRINT_165/artifacts/day14-closeout-and-handoff.md), [`SPRINT_165/RETROSPECTIVE.md`](../../SPRINT_165/RETROSPECTIVE.md), [`day9-public-claim-audit-package-abi-windows.md`](day9-public-claim-audit-package-abi-windows.md). | Static archive package proof is stronger. Shared-library support, dynamic ABI compatibility, runtime-loader behavior, package-manager distribution, Windows Makefile parity, and Windows `pkg-config` execution parity remain explicit residual product decisions. |
| 166 | Validate final Epic 14 state, recalibrate public claims, and publish closeout. | In progress; evidence inventory, validation, hosted reconciliation, claim audit, and project-plan reconciliation are complete through Day 11. | [`day1-sprint-intake.md`](day1-sprint-intake.md) through [`day11-project-plan-reconciliation-part2.md`](day11-project-plan-reconciliation-part2.md). | Remaining work is Epic retrospective drafting, final residual queue publication, final closeout, and PR-level hosted CI confirmation. |

## Sprint 162 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Windows Package Audit. | Complete. | Sprint 162 Day 2 audit and Day 14 closeout. | The sprint separated Windows CMake install proof from Unix Make/`pkg-config` proof. |
| 2 | Product Decision. | Complete with retained non-claim decision. | Sprint 162 Day 4 product decision and Day 14 closeout. | The selected decision was to retain Windows Makefile and Windows `pkg-config` execution parity as non-claims. |
| 3 | Selected Proof. | Complete as stronger rejection guard. | Sprint 162 Day 5-7 implementation artifacts and Day 14 closeout. | Guard work strengthened unsupported-surface checks instead of adding unselected Windows package execution. |
| 4 | CI Alignment. | Complete. | Sprint 162 Day 8 artifact and Day 14 closeout. | Windows workflow wording now labels CMake package proof and metadata-only `sparse.pc` inspection. |
| 5 | Downstream Consumer Evidence. | Complete for CMake-first Windows package evidence. | Sprint 162 Day 9 artifact and Day 14 closeout. | Exact-version and downstream behavior remain CMake package evidence, not `pkg-config` command execution proof. |
| 6 | Docs Alignment. | Complete. | Sprint 162 Day 11 artifact and Day 14 closeout. | README, INSTALL, and maintainer wording preserve retained non-claims. |
| 7 | Validation And Closeout. | Complete. | Sprint 162 Day 10, Day 12, Day 13, and Day 14 artifacts. | Static deferral, install, CMake install, whitespace, and changed-file checks passed. |

## Sprint 163 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Surface Selection. | Complete. | Sprint 163 Day 2-3 artifacts and Day 14 closeout. | Selected canonical benchmark and sentinel report rows only. |
| 2 | Methodology Contract. | Complete. | Sprint 163 Day 4 artifact and Day 14 closeout. | Methodology fields define platform, compiler/build context, repeats, variance, thresholds, and caveats. |
| 3 | Report Enhancements. | Complete. | Sprint 163 Day 5-7 artifacts and Day 14 closeout. | Benchmark/sentinel reports now emit additional methodology metadata. |
| 4 | Gate Classification. | Complete. | Sprint 163 Day 8 artifact and Day 14 closeout. | S5 remains a hard local wall-check gate; S2/S3 remain threshold-free backend-context reports. |
| 5 | Docs Alignment. | Complete. | Sprint 163 Day 9-10 artifacts and Day 14 closeout. | Public, benchmark, maintainer, and schema docs describe local-only non-superiority boundaries. |
| 6 | Validation. | Complete. | Sprint 163 Day 11-12 artifacts and Day 14 closeout. | Selected report scripts, generated reports, normalizer, schema, and static package guard checks passed. |
| 7 | Closeout. | Complete. | Sprint 163 Day 13-14 artifacts and retrospective. | Hosted performance publication, superiority methodology, and broader benchmark claims remain residual. |

## Sprint 164 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Header Selection. | Complete. | Sprint 164 Day 2 artifact and Day 14 closeout. | Selected `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`. |
| 2 | Declaration Baseline. | Complete. | Sprint 164 Day 3-4 artifacts and Day 14 closeout. | Normalized declaration capture established the pre-edit checksum. |
| 3 | Comment Cleanup. | Complete for selected headers. | Sprint 164 Day 5-7 artifacts and Day 14 closeout. | Cleanup covered ownership, lifetime, status/error, output-buffer, option/result, and backend wording. |
| 4 | Cross-Link Cleanup. | Complete. | Sprint 164 Day 8 and Day 11 artifacts and Day 14 closeout. | README, tutorial, and solver-selection wording were aligned where they contradicted selected headers. |
| 5 | Declaration Preservation. | Complete. | Sprint 164 Day 10 and Day 14 artifacts. | Final checksum matched the baseline and the normalized declaration diff had no output. |
| 6 | Generated Reference Check. | Complete. | Sprint 164 Day 9 and Day 14 artifacts. | `make docs-check` passed under the Sprint 158 generated API policy. |
| 7 | Validation And Closeout. | Complete. | Sprint 164 Day 12-14 artifacts and retrospective. | Full public-header gate passed; broader non-selected header cleanup remains residual. |

## Sprint 165 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Package Metadata Audit. | Complete. | Sprint 165 Day 2 artifact and Day 14 closeout. | CMake package files, `sparse.pc`, install scripts, and CI checks were audited for unsupported wording and metadata. |
| 2 | Static Deferral Guard. | Complete. | Sprint 165 Day 3-4 artifacts and Day 14 closeout. | `BUILD_SHARED_LIBS=ON` rejection and shared metadata checks were strengthened. |
| 3 | ABI Non-Claim Audit. | Complete. | Sprint 165 Day 5-6 artifacts and Day 14 closeout. | Public docs and package wording distinguish exact package metadata from dynamic ABI support. |
| 4 | Downstream Proof Refresh. | Complete. | Sprint 165 Day 7-8, Day 10-11 artifacts, and Day 14 closeout. | Make install/`pkg-config` and CMake install/export downstream proof were refreshed for static archive behavior. |
| 5 | Package Docs Alignment. | Complete. | Sprint 165 Day 9 artifact and Day 14 closeout. | README, INSTALL, maintainer guide, CMake comments, and package wording align to static-first support. |
| 6 | Validation. | Complete. | Sprint 165 Day 10-12 artifacts and Day 14 closeout. | Full quality gate, static package deferral, install scripts, CMake install scripts, and package report-index checks passed. |
| 7 | Closeout. | Complete. | Sprint 165 Day 13-14 artifacts and retrospective. | Shared-library, dynamic ABI, runtime-loader, package-manager, Windows Makefile, and Windows `pkg-config` parity remain future product scope. |

## Sprint 166 Item Reconciliation

| Item | Planned work | Close state through Day 11 | Evidence | Remaining work |
| --- | --- | --- | --- | --- |
| 1 | Final Evidence Inventory. | Complete. | Day 1, Day 2, and Day 3 artifacts. | None. |
| 2 | Full Validation Baseline. | Complete. | Day 4, Day 5, and Day 6 artifacts. | Re-run only if later touched surfaces require it. |
| 3 | Hosted CI Reconciliation. | Complete locally; PR hosted confirmation pending. | Day 7 artifact and `.github/workflows/ci.yml` update. | Confirm hosted CI after PR push. |
| 4 | Claim Audit. | Complete through public performance/report and package/ABI/Windows surfaces. | Day 8 and Day 9 artifacts. | Re-scan if Days 12-14 edit public claim surfaces. |
| 5 | Project Plan Reconciliation. | Complete through Day 11. | Day 10 and Day 11 artifacts. | None unless Day 12-14 uncover contradictions. |
| 6 | Epic 14 Retrospective. | Pending. | Planned for Day 12 and final Sprint 166 retrospective. | Draft Epic 14 retrospective and Sprint 166 retrospective. |
| 7 | Residual Queue. | Pending. | Seeded by Day 10 and Day 11 registers. | Publish final residual queue and closeout prep on Day 13/14. |

## Final Success-Criteria Status Draft

| Epic 14 success criterion | Draft status | Evidence | Boundary |
| --- | --- | --- | --- |
| Generated API reference publication is no longer ambiguous. | Complete. | Sprint 158 closeout, Sprint 164 generated-reference check, Sprint 166 Day 10 reconciliation. | Generated HTML is ignored/local-only; source headers plus Doxygen coverage checks are the maintained evidence. |
| Selected generated oracle/comparison evidence has a reviewed hosted path. | Complete for selected Linux hosted path, pending PR-hosted confirmation for current Sprint 166 workflow edits. | Sprint 159 closeout, Sprint 160 and 161 comparison-family closeouts, Sprint 166 Day 7 reconciliation. | This does not imply broad report-index freshness, unselected-family proof, or macOS/Windows report-freshness parity. |
| One bounded QR comparison family and one bounded partial-SVD comparison family are published with normalized freshness checks. | Complete. | Sprint 160 and 161 closeouts, `make report-index-comparison-freshness` records in Sprint 166 Day 6 and Day 7. | Claims are fixture-local and comparison-family-specific. |
| Windows package parity is either implemented for selected scope or retained as an explicit test-backed non-claim. | Complete as retained non-claim. | Sprint 162 closeout, Sprint 165 package-boundary closeout, Sprint 166 Day 9 audit. | Windows support remains CMake-first; Windows Makefile and Windows `pkg-config` execution parity are not claimed. |
| Performance reporting has methodology-bound publication without superiority overclaiming. | Complete for local generated report publication. | Sprint 163 closeout and Sprint 166 Day 8 audit. | No hosted performance proof, portable performance guarantee, backend superiority, or state-of-the-art performance claim. |
| Public headers and API docs remain declaration-preserving and coherent. | Complete for selected header batch. | Sprint 164 closeout and Sprint 166 Day 3 inventory. | Broader header cleanup remains residual; generated HTML remains local-only. |
| Static-first package and ABI non-claims are hardened. | Complete. | Sprint 165 closeout and Sprint 166 Day 9 static package deferral check. | Static package proof does not imply shared-library support, dynamic ABI compatibility, runtime-loader behavior, package-manager distribution, or static/shared selector UX. |
| The final state-of-the-art assessment maps every positive claim to recurring evidence and rejects unsupported broad claims. | In progress. | Sprint 166 Day 8-11 claim audits and reconciliation artifacts. | Final wording belongs to Day 12 retrospective draft, Day 13 residual queue, and Day 14 closeout. |

## Remaining Sprint 166 Work

After Day 11, the remaining Sprint 166 work is reduced to:

1. Draft the Epic 14 retrospective with earned claims, retained non-claims,
   validation evidence, and state-of-the-art assessment.
2. Publish the final residual queue with owners, blockers, prerequisites, and
   promotion gates.
3. Prepare the Sprint 166 retrospective and PR closeout.
4. Confirm PR-level hosted CI after the branch is pushed.

## Validation

- Documentation/planning artifact only for Day 11.
- No `.c` or `.h` files were modified for this Day 11 reconciliation.
- `git diff --check` passed after the artifact and working-notes update.
