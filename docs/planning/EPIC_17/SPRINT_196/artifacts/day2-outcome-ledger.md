# Sprint 196 Day 2 Artifact: Outcome Ledger

**Date:** 2026-09-03
**Sprint item coverage:** 196.1
**Day 2 goal:** Reconcile Sprint 187 through Sprint 195 outcomes, validation
records, decisions, residuals, conflicts, and supersessions before public claim
or project-plan status edits begin.

## Summary

Day 2 converts the Day 1 evidence inventory into an outcome ledger. Each prior
Epic 17 sprint now has a reconciled status, evidence anchor, retained residual
state, and claim-boundary note for later Sprint 196 documentation work.

The core finding is that Epic 17 closed several selected, bounded gaps, but it
did not close broad package-manager distribution, broad Windows parity, broad
external-library parity, portable performance, dynamic ABI/shared-library
support, exhaustive reliability, release readiness, or unqualified
state-of-the-art status.

## Sprint Outcome Ledger

| Sprint | Planned closure | Reconciled outcome | Evidence anchor | Status class |
| --- | --- | --- | --- | --- |
| 187 | Epic 17 baseline, gap ledger, acceptance gates, and handoffs | Built the gap ledger, residual reconciliation, closure selection, acceptance gates, quality map, and Sprint 188-195 handoffs. | `docs/planning/EPIC_17/SPRINT_187/artifacts/day14-closeout-summary.md`; `docs/planning/EPIC_17/SPRINT_187/RETROSPECTIVE.md` | Complete |
| 188 | Homebrew proof completion | Hardened local proof material and package guards, but kept support unavailable while approved standalone license metadata is missing. | `docs/planning/EPIC_17/SPRINT_188/artifacts/day14-closeout-summary.md`; `docs/planning/EPIC_17/SPRINT_188/RETROSPECTIVE.md` | Complete with guarded residual |
| 189 | PowerShell validation ownership | Added source-controlled PowerShell validation ownership, local unavailable semantics, hosted fail-closed wiring, tests, and docs. | `docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md`; `docs/planning/EPIC_17/SPRINT_189/RETROSPECTIVE.md` | Complete with hosted evidence pending at closeout |
| 190 | Windows selected report freshness decision | Added one bounded selected Cholesky Windows workflow path and target-specific freshness validation, while retaining pending hosted evidence/manifest promotion residuals. | `docs/planning/EPIC_17/SPRINT_190/artifacts/day14-sprint-closeout.md`; `docs/planning/EPIC_17/SPRINT_190/RETROSPECTIVE.md` | Complete with residual narrowed |
| 191 | Bounded external comparison family | Added one local-only `qr-incompatible-ls` selected comparison family with fixture, runner, manifest, docs, tests, and freshness diagnostics. | `docs/planning/EPIC_17/SPRINT_191/artifacts/day14-closeout-and-handoff.md`; `docs/planning/EPIC_17/SPRINT_191/RETROSPECTIVE.md` | Complete |
| 192 | Methodology-bound performance evidence lane | Added one selected Linux hosted `bench_refactor_csc` evidence lane with exact three-file bundle and threshold-free methodology metadata. | `docs/planning/EPIC_17/SPRINT_192/artifacts/day14-closeout-and-handoff.md`; `docs/planning/EPIC_17/SPRINT_192/RETROSPECTIVE.md` | Complete |
| 193 | Selected large review-surface reduction | Extracted one selected QR external-reference rank/nullspace/threshold cluster into a guarded helper while preserving behavior. | `docs/planning/EPIC_17/SPRINT_193/artifacts/day14-closeout.md`; `docs/planning/EPIC_17/SPRINT_193/RETROSPECTIVE.md` | Complete |
| 194 | Adoption and API coherence simplification | Consolidated support/readiness truth, improved installed-consumer routing, normalized diagnostics wording, and reduced selected header narrative. | `docs/planning/EPIC_17/SPRINT_194/artifacts/day14-closeout-handoff.md`; `docs/planning/EPIC_17/SPRINT_194/RETROSPECTIVE.md` | Complete |
| 195 | Selected reliability and failure-path proof | Proved selected `sparse_symbolic_cholesky()` allocation-failure cleanup, stale-output suppression, retry, focused gate, and claim boundaries. | `docs/planning/EPIC_17/SPRINT_195/artifacts/day14-closeout-review-package.md`; `docs/planning/EPIC_17/SPRINT_195/RETROSPECTIVE.md` | Complete |

## Topic Ledger

| Topic | Completed evidence | Residual, pending, or deferred evidence | Later Sprint 196 claim rule |
| --- | --- | --- | --- |
| Package management | Local Homebrew proof path and guards are stricter; static package install validation remains covered. | Approved root license metadata and exact Homebrew license identifier are missing; package-manager support remains unclaimed. | Say proof material exists and is guarded; do not say Homebrew or package-manager install support exists. |
| Windows and PowerShell | PowerShell validation owner exists; one selected Cholesky Windows workflow path exists. | Local `pwsh` is unavailable here; Sprint 190 hosted evidence and manifest promotion were pending at closeout; broad Windows parity remains unclaimed. | Separate hosted evidence, local unavailable semantics, selected workflow paths, and broad Windows non-claims. |
| External comparison | `qr-incompatible-ls` landed as one bounded local-only selected family. | Windows QR promotion, optional package baselines, broader QR parity, generated local artifacts, and review-volume follow-ups remain residual. | Keep comparison claims selected-target and fixture scoped. |
| Performance | One Linux hosted selected benchmark evidence lane landed with exact bundle and methodology metadata. | Timing threshold, portable performance, Windows/macOS freshness, unselected CSV publication, and release benchmark claims remain residual. | Say measurement evidence, not superiority or threshold enforcement. |
| Maintainability/review surface | Selected QR external-reference helper extraction is complete and guarded. | Other QR clusters, helper dependency tracking, helper-size split, and unrelated warning hygiene remain residual. | Say one selected review surface was reduced; do not imply broad test-suite simplification. |
| Adoption/API coherence | Support/readiness matrix and user-doc routing are clearer; selected headers are less narrative-heavy. | Markdown link target, package distribution, shared library/dynamic ABI, broad Windows parity, portable performance, and further header cleanup remain residual. | Treat `INSTALL.md` as support/readiness owner and avoid duplicated support claims. |
| Reliability | Selected symbolic Cholesky output allocation failure path is deterministically tested and guarded. | Other symbolic/analyze/etree/direct/matrix owners, OS OOM, concurrency, and hosted gate ownership remain residual. | Say selected-owner proof only. |

## Completed Outcomes

- Sprint 187 completed planning, gap-ledger, acceptance-gate, quality-map, and
  handoff work.
- Sprint 188 completed proof hardening, package guard alignment, package docs,
  and validation around the current unavailable Homebrew proof state.
- Sprint 189 completed source-controlled PowerShell validation ownership.
- Sprint 190 completed a bounded Windows selected Cholesky workflow path and
  target-specific freshness tooling.
- Sprint 191 completed one bounded local-only external comparison family.
- Sprint 192 completed one threshold-free selected performance evidence lane.
- Sprint 193 completed one selected QR review-surface reduction.
- Sprint 194 completed selected adoption and API coherence simplification.
- Sprint 195 completed one selected reliability/failure-path proof.

## Guarded, Pending, Deferred, And Residual Outcomes

| Category | Outcome |
| --- | --- |
| Guarded residual | Homebrew proof success and package-manager support wait for approved standalone root license metadata and exact Homebrew license identifier. |
| Environment residual | Local PowerShell execution remains unavailable on this machine unless `pwsh` is installed. |
| Hosted evidence pending at closeout | Sprint 189 hosted PowerShell validation and Sprint 190 hosted selected Cholesky artifact/freshness evidence were PR-CI-owned at their closeouts. |
| Promotion pending at closeout | Sprint 190 selected Cholesky Windows manifest promotion required hosted evidence review before changing selected metadata. |
| Future comparison residuals | Windows QR incompatible freshness, optional package baselines, broader QR least-squares parity, generated local comparison evidence, and comparison review-volume cleanup. |
| Future performance residuals | Timing thresholds, portable performance evidence, Windows/macOS selected benchmark freshness, unselected CSV publication, and release benchmark claims. |
| Future maintainability residuals | Additional QR cluster extraction, helper dependency tracking, possible helper split, and unrelated warning hygiene. |
| Future adoption/API residuals | Markdown link-check target, package distribution, shared-library/dynamic ABI, broad Windows parity, portable performance, and remaining declaration-adjacent header detail. |
| Future reliability residuals | `sparse_symbolic_lu()`, `sparse_analyze()`, helper-level etree routines, direct solvers, matrix construction, OS OOM, concurrent allocation hooks, and hosted gate ownership. |

## Conflicts And Supersessions

| Surface | Conflict risk | Day 2 resolution |
| --- | --- | --- |
| Sprint 188 metadata implementation | The original plan allowed implementation if metadata was selected. | Final evidence supersedes that path: metadata was not implemented because license terms must not be invented. |
| Package docs | Local proof material could be interpreted as support. | Later claim edits must keep proof material separate from user-facing package-manager support. |
| Sprint 189 hosted wiring | A hosted PowerShell job could be confused with Windows report evidence. | Keep PowerShell validation ownership separate from report freshness. |
| Sprint 190 selected workflow | The selected Cholesky lane could be interpreted as fully promoted Windows selected freshness. | Record it as a bounded workflow path with hosted evidence/manifest-promotion residuals unless later evidence proves promotion. |
| Sprint 191 comparison family | One fixture could be interpreted as QR or external-library parity. | Keep wording local-only and fixture-bound. |
| Sprint 192 performance lane | Hosted benchmark artifacts could be interpreted as performance superiority. | Keep wording threshold-free, methodology-bound, and non-portable. |
| Sprint 193 extraction | One helper move could be interpreted as broad maintainability cleanup. | Keep claim to one selected QR external-reference review surface. |
| Sprint 194 support matrix | Support/readiness consolidation could be interpreted as new support. | Treat it as claim routing and calibration, not feature promotion. |
| Sprint 195 reliability proof | One owner proof could be interpreted as broad allocation-failure reliability. | Keep proof scoped to selected symbolic Cholesky output allocation. |
| Epic 17 final outcome | Selected closures could be interpreted as state-of-the-art status. | Preserve explicit non-claims and state-of-the-art assessment boundaries. |

## 196.1 Acceptance Evidence

| Completion criterion | Evidence |
| --- | --- |
| Every Sprint 187-195 outcome has a status and evidence link. | The sprint outcome ledger above lists a status class and evidence anchor for each sprint. |
| Deferred and residual outcomes are separated from completed outcomes. | The guarded/pending/deferred/residual table separates non-closed work from completed outcomes. |
| Conflicting or superseded wording is identified before documentation edits. | The conflict and supersession table records the exact wording risks to handle in later claim calibration. |

## Validation

- `sed -n '65,115p' docs/planning/EPIC_17/SPRINT_196/PLAN.md`
- `sed -n '338,380p' docs/planning/EPIC_17/PROJECT_PLAN.md`
- `rg -n "^\\| 19[0-9]\\.[0-9]|^\\| 18[7-9]\\.[0-9]" docs/planning/EPIC_17/PROJECT_PLAN.md`
- Reviewed Sprint 187-195 Day 14 closeout artifacts.
- Reviewed Sprint 187-195 retrospective status and residual sections.

Day 2 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required for this day.
