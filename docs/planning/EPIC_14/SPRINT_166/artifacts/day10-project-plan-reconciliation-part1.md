# Day 10 Project Plan Reconciliation Part 1

## Scope

Day 10 reconciles Epic 14 project-plan items for Sprints 157 through 161
against their sprint plans, working notes, daily artifacts, retrospectives,
and Sprint 166 claim-audit updates.

This artifact does not rewrite the historical sprint closeouts. It records
their current close state for final Epic 14 closeout and separates completed
work from narrowed claims, explicit product decisions, and residualized work.

## Sprint-Level Reconciliation

| Sprint | Project-plan goal | Close state | Evidence | Current interpretation |
| --- | --- | --- | --- | --- |
| 157 | Establish the Epic 14 baseline, selected residuals, evidence contract, quality surface, risks, and handoff. | Complete. | [`SPRINT_157/artifacts/day14-sprint-closeout-and-sprint158-handoff.md`](../../SPRINT_157/artifacts/day14-sprint-closeout-and-sprint158-handoff.md), [`SPRINT_157/RETROSPECTIVE.md`](../../SPRINT_157/RETROSPECTIVE.md). | Planning and evidence-gate sprint only; it selected targets and non-goals without claiming implementation closure. |
| 158 | Close generated API HTML publication with a committed/publication decision or guarded local-only decision. | Complete with explicit product decision. | [`SPRINT_158/artifacts/day14-closeout-handoff.md`](../../SPRINT_158/artifacts/day14-closeout-handoff.md), [`SPRINT_158/RETROSPECTIVE.md`](../../SPRINT_158/RETROSPECTIVE.md). | Generated API HTML remains ignored and local-only by design; source headers and Doxygen coverage checks are the maintained API-doc evidence. |
| 159 | Promote selected hosted oracle and comparison freshness evidence. | Complete for the selected Sprint 159 hosted scope; later comparison-scope expansion was reconciled by Sprint 166 Day 7. | [`SPRINT_159/artifacts/day14-closeout.md`](../../SPRINT_159/artifacts/day14-closeout.md), [`SPRINT_159/RETROSPECTIVE.md`](../../SPRINT_159/RETROSPECTIVE.md), [`day7-hosted-ci-evidence-reconciliation.md`](day7-hosted-ci-evidence-reconciliation.md). | Sprint 159 promoted selected oracle and QR min-norm comparison freshness. Current Epic 14 closeout must use the Sprint 166 Day 7 hosted workflow update for the broader selected-comparison artifact scope. |
| 160 | Add one bounded QR comparison family and integrate it into selected freshness evidence. | Complete. | [`SPRINT_160/artifacts/day14-closeout.md`](../../SPRINT_160/artifacts/day14-closeout.md), [`SPRINT_160/RETROSPECTIVE.md`](../../SPRINT_160/RETROSPECTIVE.md), [`day7-hosted-ci-evidence-reconciliation.md`](day7-hosted-ci-evidence-reconciliation.md). | `qr-compatible-ls` closed one fixture-local QR comparison gap. It does not support broad QR correctness, external parity, or portable performance claims. |
| 161 | Add one bounded partial-SVD comparison family and integrate it into selected freshness evidence. | Complete. | [`SPRINT_161/artifacts/day14-closeout.md`](../../SPRINT_161/artifacts/day14-closeout.md), [`SPRINT_161/RETROSPECTIVE.md`](../../SPRINT_161/RETROSPECTIVE.md), [`day7-hosted-ci-evidence-reconciliation.md`](day7-hosted-ci-evidence-reconciliation.md). | `partial-svd-diag6-k2` closed one fixture-local partial-SVD comparison gap. It does not support broad SVD correctness, raw vector identity, repeated-spectrum ordering, or external-library parity claims. |

## Sprint 157 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Baseline Inventory. | Complete. | Sprint 157 Day 1-6 artifacts and Day 14 closeout. | Baseline covered code, tests, docs, generated artifacts, package, ABI, and platform surfaces. |
| 2 | Residual Selection. | Complete. | Sprint 157 Day 7-8 artifacts and Day 14 closeout. | Residuals were converted into selected Epic 14 targets plus explicit long-horizon non-goals. |
| 3 | Evidence Contract. | Complete. | Sprint 157 Day 9 artifact and Day 14 closeout. | Evidence templates became the rule set for later claim audits and final closeout. |
| 4 | Claim Target Register. | Complete. | Sprint 157 Day 11 artifact and Day 14 closeout. | Accepted claims and rejected broad claims were documented before implementation sprints. |
| 5 | Quality Surface Map. | Complete. | Sprint 157 Day 10 artifact and Day 14 closeout. | Validation command ownership was mapped by change type and evidence family. |
| 6 | Risk And Handoff. | Complete. | Sprint 157 Day 12 artifact, `WORKING_NOTES.md`, and Day 14 closeout. | Sprint 158 started from a concrete generated API-docs handoff. |
| 7 | Closeout. | Complete. | Sprint 157 Day 13-14 artifacts and retrospective. | Sprint 157 left no unresolved planning item; implementation residuals were handed to Sprints 158-166. |

## Sprint 158 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Doxygen Baseline. | Complete. | Sprint 158 artifacts and retrospective. | Generated API output, warning state, ignored-output policy, and header coverage were inventoried. |
| 2 | Publication Decision. | Complete with explicit local-only product decision. | Sprint 158 Day 14 closeout and retrospective. | `docs/api/html/` remains ignored and not source-controlled. This is not a hosted-publication completion claim. |
| 3 | Coverage Check. | Complete. | Sprint 158 closeout validation and `make docs-check`/`make api-docs-coverage` records. | Source-header-first API coverage became the maintained recurring evidence. |
| 4 | Warning Triage. | Complete. | Sprint 158 closeout validation. | Doxygen warning checks are part of the recurring docs gate. |
| 5 | Docs Alignment. | Complete. | Sprint 158 docs changes and retrospective. | Public docs were aligned to local-only generated HTML and source-header-first authority. |
| 6 | Validation. | Complete. | Sprint 158 retrospective validation record. | Full public-header quality gate was run because headers/docs were touched. |
| 7 | Closeout. | Complete. | Sprint 158 Day 14 closeout and retrospective. | Hosted generated API HTML publication and committed generated HTML remain explicit non-selected residuals. |

## Sprint 159 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Family Selection. | Complete. | Sprint 159 Day 14 closeout and retrospective. | Selected hosted report-freshness ownership was scoped to selected oracle rows and selected comparison rows available at the time. |
| 2 | Runtime Budget. | Complete. | Sprint 159 artifacts and retrospective. | Hosted report work stayed bounded to reviewed Linux freshness lanes and retained advisory/local-only rows outside the claim. |
| 3 | CI Implementation. | Complete for original Sprint 159 scope; current selected-comparison scope corrected later. | Sprint 159 Day 14 closeout, Sprint 159 retrospective, Sprint 166 Day 7 artifact. | Sprint 159 implemented hosted selected oracle and QR min-norm comparison freshness. Sprint 166 Day 7 updated the hosted comparison lane naming, summary, and upload paths for later selected families. |
| 4 | Artifact Publication. | Complete for original Sprint 159 scope; later expanded by Sprint 166 Day 7. | Sprint 159 Day 14 closeout and Sprint 166 Day 7 artifact. | The current hosted artifact claim is only valid for selected comparison families after the Day 7 workflow reconciliation. |
| 5 | Normalizer Semantics. | Complete. | Sprint 159 validation records and normalizer tests. | Selected rows fail when skipped/deferred; advisory and optional rows remain non-proof context. |
| 6 | Docs Alignment. | Complete with later wording refinement. | Sprint 159 retrospective, Sprint 166 Day 8 artifact. | Sprint 166 Day 8 clarified that selected comparison rows are local generated evidence by default and hosted evidence only after the reviewed Linux lane runs. |
| 7 | Validation And Closeout. | Complete. | Sprint 159 Day 14 closeout and retrospective. | Broad report-index freshness, macOS/Windows report-index parity, unselected generated families, and hosted generated API HTML remained residualized. |

## Sprint 160 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Target Selection. | Complete. | Sprint 160 Day 14 closeout and retrospective. | Selected `qr-compatible-ls` as one bounded QR comparison family. |
| 2 | Metric Contract. | Complete. | Sprint 160 artifacts and retrospective. | Metrics stayed fixture-local and did not claim broad QR parity. |
| 3 | Harness Extension. | Complete. | Sprint 160 validation records. | Descriptor-backed comparison runner generated the selected QR compatible least-squares rows. |
| 4 | Focused Tests. | Complete. | Sprint 160 retrospective validation record. | Runner and normalizer tests covered selected row behavior. |
| 5 | Report Integration. | Complete, with hosted artifact naming residual later closed. | Sprint 160 Day 14 closeout and Sprint 166 Day 7 artifact. | Selected comparison freshness expanded locally; Day 7 aligned hosted artifact naming and upload paths with current selected families. |
| 6 | Docs Alignment. | Complete. | Sprint 160 retrospective and Sprint 166 Day 8 artifact. | Public claim wording remained bounded to fixture-local evidence. |
| 7 | Validation And Closeout. | Complete. | Sprint 160 Day 14 closeout and retrospective. | Partial-SVD comparison publication was handed to Sprint 161. |

## Sprint 161 Item Reconciliation

| Item | Planned work | Close state | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | Target Selection. | Complete. | Sprint 161 Day 14 closeout and retrospective. | Selected `partial-svd-diag6-k2` as the first bounded partial-SVD comparison family. |
| 2 | Metric Contract. | Complete. | Sprint 161 Day 14 closeout and retrospective. | Metrics used singular values, residual norm, orthogonality, and projector-diagonal checks rather than raw vector identity. |
| 3 | Harness Extension. | Complete. | Sprint 161 validation records. | Descriptor-backed comparison runner generated the selected partial-SVD rows. |
| 4 | Focused Tests. | Complete. | Sprint 161 retrospective validation record. | Runner and normalizer tests covered partial-SVD selected row behavior and optional dependency non-proof states. |
| 5 | Report Integration. | Complete, with hosted artifact scope reconciled later. | Sprint 161 Day 14 closeout and Sprint 166 Day 7 artifact. | Selected comparison freshness includes QR and partial-SVD families; current hosted upload paths include all selected comparison directories. |
| 6 | Docs Alignment. | Complete. | Sprint 161 retrospective and Sprint 166 Day 8 artifact. | Docs keep the partial-SVD claim fixture-local and reject broad SVD/external parity wording. |
| 7 | Validation And Closeout. | Complete. | Sprint 161 Day 14 closeout and retrospective. | Sprint 162 package parity work was correctly separated from comparison evidence. |

## Narrowed, Deferred, And Residualized Register

| Topic | State after Sprint 157-161 reconciliation | Owner/evidence |
| --- | --- | --- |
| Hosted generated API HTML publication. | Deferred/not selected. Sprint 158 closed with a local-only generated HTML product decision. | Sprint 158 Day 14 closeout and retrospective. |
| Committed `docs/api/html/` output. | Deferred/not selected. Source headers plus coverage checks remain the maintained API-doc evidence. | Sprint 158 Day 14 closeout and retrospective. |
| Broad report-index freshness. | Residualized. Only selected hosted/generated report rows may be used as evidence. | Sprint 159 Day 14 closeout; Sprint 166 Day 8 docs wording. |
| Unselected generated oracle/comparison families. | Residualized/advisory. They cannot be read as hosted proof or release proof. | Sprint 159-161 retrospectives. |
| macOS/Windows report-index freshness parity. | Residualized. Reviewed Linux hosted report-freshness evidence does not imply cross-platform report-freshness parity. | Sprint 159 Day 14 closeout and retrospective. |
| Hosted comparison artifact naming/scope mismatch. | Closed by Sprint 166 Day 7. | `day7-hosted-ci-evidence-reconciliation.md`. |
| Broad QR correctness or external-library parity. | Not claimed. Sprint 160 closed one fixture-local QR compatible least-squares comparison family only. | Sprint 160 Day 14 closeout and retrospective. |
| Broad SVD/partial-SVD correctness or raw singular-vector identity. | Not claimed. Sprint 161 closed one fixture-local diagonal top-k comparison family only. | Sprint 161 Day 14 closeout and retrospective. |
| Package, ABI, Windows package parity, performance publication, public-header coherence, and static-first boundary work. | Out of Day 10 scope; reconciled in Day 11 for Sprints 162-166. | Sprint 162-166 artifacts. |

## Current Claim Reconciliation

- Sprint 157 is a planning baseline and evidence-contract completion, not an
  implementation claim.
- Sprint 158 supports the claim that generated API coverage is guarded through
  source-header-first checks. It does not support hosted or source-controlled
  generated HTML publication.
- Sprint 159 supports reviewed Linux hosted selected oracle freshness and the
  hosted comparison freshness scope selected at that time. Current selected
  comparison hosted evidence must cite Sprint 166 Day 7 because Sprints 160
  and 161 later expanded the selected comparison families.
- Sprint 160 supports one bounded QR compatible least-squares comparison
  family. It does not support broad QR correctness, broad external parity, or
  portable performance claims.
- Sprint 161 supports one bounded partial-SVD diagonal top-k comparison
  family. It does not support broad SVD correctness, vector identity,
  repeated-spectrum ordering, broad external parity, package proof, or
  performance claims.

## Completion Check

- Sprint 157 through 161 project-plan items have explicit close states.
- Completed items include supporting evidence links.
- Product decisions and residuals are not presented as completed
  implementation proof.
- The only Sprint 159-161 hosted comparison claim used for final closeout is
  the reconciled selected-comparison hosted surface from Sprint 166 Day 7.

## Validation

- Documentation/planning artifact only for Day 10.
- No `.c` or `.h` files were modified for this Day 10 reconciliation.
- `git diff --check` passed after the artifact and working-notes update.
