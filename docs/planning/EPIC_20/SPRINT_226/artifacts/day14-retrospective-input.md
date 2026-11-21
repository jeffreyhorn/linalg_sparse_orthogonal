# Sprint 226 Day 14: Retrospective Input

This is input for the later Sprint 226 retrospective, not a release decision. The [plan](../PLAN.md) allocated 140 estimated hours over six items and fourteen 10-hour days; actual labor was not tracked. [Working notes](../WORKING_NOTES.md) and the [closeout index](day14-closeout-evidence-index.md) carry the daily receipts.

| Planned item | Achieved in this branch | Remaining outcome |
| --- | --- | --- |
| 226.1 Candidate Freeze (20 h) | Inventoried version, source, license, platform and approval prerequisites; wrote a blocked freeze rule. | No approved changelog, named release/rollback owners or clean selected SHA. **Blocked.** |
| 226.2 Reproducible Source Archive (28 h) | Two independent same-host base archives matched byte for byte; extraction/source map and 25-file CMake static install map retained. | Rebuild and approve two archives and installed map for the final candidate. **Blocked product outcome.** |
| 226.3 Candidate Platform Proof (26 h) | Local exploratory Make/CMake/width checks and historical hosted Linux/macOS/Windows receipts reconciled. | No exact-candidate hosted run; historical Linux parity interrupted; provider aggregate incomplete. **Blocked.** |
| 226.4 First-Use Audit (24 h) | Clean-prefix README/INSTALL/example/API route exercised locally; version/width, all installed headers, links and package metadata audited; README/INSTALL corrected. | Repeat from committed candidate archive and have docs/API owners sign packaged wording. **Local deliverable complete, candidate acceptance blocked.** |
| 226.5 Readiness Decision (22 h) | Technical [BLOCKED decision](day12-static-candidate-readiness-decision.md), evidence index, approval register and rollback proposal written. | No release-owner signature, accepted timing scope, asset destination or final go/no-go. **Blocked.** |
| 226.6 Epic Closeout (20 h) | [Epic reconciliation](day13-epic20-completion-reconciliation.md), [claim ledger](day13-validation-and-claim-ledger.md) and [residual handoff](day14-final-epic20-closeout-and-handoff.md) completed locally. | Retrospective drafting and next-epic owner adoption remain subsequent work; no release claim inferred. |

## Lessons and unresolved risks

1. **The exact-source rule prevented a false release claim.** The old base tarball passed same-host reproducibility, static install and local consumers, but the documentation fixes and sprint records were made afterward. Its digest cannot authenticate the intended package.
2. **Installed first use is now concrete.** Clean-prefix Make and CMake instructions were followed through a maintained consumer; Make versus CMake metadata scope and 19-header coverage were corrected. A future candidate run should preserve this end-to-end route.
3. **Historical hosted green subsets did not close the platform gate.** Linux parity ended mid-suite, and no hosted lane tested the selected candidate. Adopted static scope and exact SHA must precede platform signoff.
4. **Timing and external evidence remained independent blockers.** Missing accepted baseline prevents a Sprint 225 measurement; incomplete CHOLMOD provenance and rights prevent an observed external cell. Neither can be recast as a passing performance or parity result.
5. **Ownership and publication policy remain the critical path.** Release, product, build, CI, API/docs, license and rollback roles have no named signatures. A technical no-go record is useful, but cannot authorize tag, asset or claim publication.

**Next action:** name the release and rollback owners, approve version/changelog/static scope, commit and review the intended source, select a clean candidate SHA, then execute the [ordered handoff gates](day14-final-epic20-closeout-and-handoff.md#residual-handoff-with-acceptance-gates). Keep the current status **BLOCKED** until actual receipts and approvals exist.
