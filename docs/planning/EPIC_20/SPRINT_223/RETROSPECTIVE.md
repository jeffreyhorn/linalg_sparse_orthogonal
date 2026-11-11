# Sprint 223 Retrospective

**Sprint:** 223 - Selected Windows Cholesky Freshness

**Duration:** 14 planned days on branch `sprint-223`

**Status:** The selected Cholesky fixture earned a bounded Windows hosted-freshness claim on the Day 14 source head. Any later commit, including this retrospective, requires a new exact-head hosted audit before the claim is current again.

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) allocates 140 estimated hours to six items. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md), [Day 14 decision](artifacts/day14-exact-scope-decision.md), and [PR #247 evidence ledger](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247) identify the target, gates, corrections, and run artifacts. The hours below are planning estimates, not measured engineering time or CI duration. The intended result was freshness for `cholesky-spd-tridiag-5` on a selected Windows MSVC lane, with six passing comparison rows, matching provenance and contract metadata, negative guards, and a narrow public sentence. The final Day 14 head `69301af0557a52a62efe89adb2030a12ee2607a9` met those conditions; the retrospective commit must be checked afresh.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 223.1 Hosted Run Intake | 20 h | Identified the exact target and inspected a current `master` Windows selected job, six-file artifact, digest, commands, and provenance. That intake artifact became historical after branch edits. |
| 223.2 Row And Artifact Audit | 30 h | Traced all six unique ordered rows to the six uploaded files, project/baseline observations, required dependencies, source revision, Windows AMD64 platform and VS 2022 Release probe. Found the old generated `local_only` contradiction. |
| 223.3 Atomic Promotion | 26 h | Aligned the one Cholesky selected-manifest row, target-scoped generator tier/configuration, workflow tuple, summary, public wording and exclusions. Other targets and local runs stayed within their existing scope. |
| 223.4 Negative Guards | 24 h | Added regressions rejecting wrong/missing/extra upload paths and files, row/path changes, dirty or mismatched provenance, wrong platform/compiler/tier, and claim/non-claim contradictions. |
| 223.5 Hosted Recheck | 22 h | Rejected an initial PR merge-SHA artifact, corrected the checkout and generator provenance, then audited later exact-head Windows artifacts. The Day 14 Windows selected job and matching macOS selected job passed. |
| 223.6 Scope Sign-Off | 18 h | Applied thirteen promotion criteria, recorded final file hashes and local checks, approved one sentence, and retained the residual queue and Sprint 224 handoff. |
| **Total** | **140 h** | **One selected fixture promoted on the audited Day 14 head; no broad Windows claim.** |

## What Went Well

1. **The selection stayed exact.** The manifest ID `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, target `cholesky-spd-tridiag-5`, six required file names and six row IDs were frozen at intake. The final audit found no missing, duplicate, extra or wrong-target row. Other Windows comparison targets did not inherit this promotion.
2. **The artifact remained independently inspectable.** The Day 14 [Windows run 42938412111](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42938412111), [selected MSVC job 130254082184](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42938412111/job/130254082184), and [artifact 13819243550](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42938412111/artifacts/13819243550) share source SHA `69301af0557a52a62efe89adb2030a12ee2607a9`. The downloaded 4,609-byte ZIP matched GitHub's SHA-256 digest `1601914db624d504a77c8bb23ef840598bb41853b580f47b7633b513f01557d8`; its six members were hashed individually in the PR ledger. All six rows passed and the selected normalizer returned `freshness ok (17 rows)` on the extracted artifact.
3. **Contradictions were resolved across surfaces.** The hosted rows, manifest and summary agree on `windows-amd64`, `cmake-probe:Visual Studio 17 2022:Release`, `source_branch=sprint-223`, a clean worktree and `support_tier=hosted_selected`. The source-controlled dense Cholesky helper is identified as the baseline. Optional NumPy and SciPy package rows remain deferred, so no third-party package parity is inferred.
4. **Negative tests protected the boundary.** The Day 9 and 10 guards enforce exact upload membership, row paths and cross-file metadata. Local generator runs still emit `local_only` and do not borrow hosted proof. The Windows PowerShell structural suite and hosted execution covered their respective checks; generated API HTML remained local-only.
5. **The final claim was tied to the tested revision.** The PR description holds the exact head, run/job/artifact identifiers, ZIP and file hashes, thirteen-gate result and stale-evidence rule. Public documentation uses the single selected sentence and explicit non-claims.

## What Didn't Go Well

1. **The first PR artifact was stale by construction.** The selected Windows job initially checked out the PR merge revision and reported an unknown source branch. Its numerical pass could not establish exact-head freshness. The job was changed to check out the PR head, and the generator now verifies its SHA and branch. The rejected artifact remains in the [Day 11 ledger](artifacts/day11-hosted-candidate-recheck.md).
2. **A shared job name broke macOS selected comparison.** The new Windows-specific generator gate first keyed on `GITHUB_JOB=selected-comparison-freshness`, a name also used on macOS. The Day 13 macOS check failed despite the Windows selected job passing. The generator now recognizes genuine non-Windows contexts as local-only and still rejects malformed Windows-like contexts. The subsequent [macOS selected job](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42938412068/job/130254082156) passed on the Day 14 head.
3. **Two local guard regressions needed refinement.** A missing-study diagnostic initially triggered on benchmark-only strict checks; a new hosted metadata check initially assumed every legacy contract had `target_key`. Both were scoped to the selected Cholesky path, and the full normalizer suite passed. These failed first runs are documented in [working notes](WORKING_NOTES.md).
4. **Each documentation commit ages hosted proof.** Day 11, Day 12 and Day 13 artifacts were valid for their respective commits but could not prove the later Day 14 source. This retrospective likewise changes the source head. The PR ledger must be replaced with a new exact-head audit after this commit; otherwise promotion must be re-deferred. This is a real custody constraint, not a waiver of the freshness gate.
5. **Native Windows checks are hosted-only here.** Local PowerShell structural tests passed, but the local machine did not have `pwsh` or MSVC. The hosted selected Windows job supplies the native execution evidence. A local `make report-index-comparison-freshness` pass does not by itself prove Windows freshness.

## Final Metrics And Claim

| Metric | Day 14 audited result |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Selected target | `cholesky-spd-tridiag-5` only |
| Windows source head | `69301af0557a52a62efe89adb2030a12ee2607a9` on `sprint-223`, clean artifact worktree |
| Hosted Windows evidence | Successful run `42938412111`, selected job `130254082184`, downloadable artifact `13819243550` |
| Uploaded evidence | Exactly six files, six ordered study rows, six `pass`, no non-passing selected row; API/download ZIP digest matched |
| Hosted platform/compiler | `windows-amd64`; Visual Studio 17 2022 Release CMake probe |
| Cross-platform selected job | Day 14 macOS selected job `130254082156` passed on the same SHA; Linux selected evidence predates this Windows promotion and remains independently scoped |
| Final local checks | Selected manifest, workflow, generator, normalizer, Windows structural, corpus schema, Python compile and `git diff --check` passed; `make report-index-comparison-freshness` yielded 46 fresh rows; `make docs-check` covered 18 public header pages |
| C quality gate | No `.c` or `.h` files changed, so `make format && make lint && make test` was not required |

The approved sentence, **for the audited Day 14 head only**, is: “Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense Cholesky reference helper.” It does not establish broad Cholesky correctness, other Windows report freshness, CHOLMOD or other external-library parity, package-manager or shared-library ABI support, performance superiority, release readiness or state-of-the-art status. The helper is source-controlled, not a CHOLMOD package. If the retrospective commit lacks its own successful exact-head job and artifact audit, the current-head decision is re-defer until that audit is complete.

## Daily Evidence

- [Day 1 exact target and hosted intake](artifacts/day1-exact-target-and-hosted-intake.md)
- [Day 2 MSVC run and six-file artifact intake](artifacts/day2-hosted-msvc-run-intake.md)
- [Day 3 row and artifact traceability](artifacts/day3-row-and-artifact-traceability.md)
- [Day 4 generator, platform and claim audit](artifacts/day4-generator-platform-and-claim-audit.md)
- [Day 5 thirteen-gate promotion decision](artifacts/day5-promotion-gate-decision.md)
- [Day 6 manifest and generator candidate](artifacts/day6-manifest-generator-candidate.md)
- [Day 7 documentation and workflow guard](artifacts/day7-documentation-and-workflow-guard.md)
- [Day 8 atomic review and negative matrix](artifacts/day8-atomic-candidate-and-negative-matrix.md)
- [Day 9 artifact, row and path guards](artifacts/day9-artifact-row-path-guards.md)
- [Day 10 metadata and claim guards](artifacts/day10-metadata-and-claim-guards.md)
- [Day 11 hosted candidate recheck](artifacts/day11-hosted-candidate-recheck.md)
- Day 12 final proposed commit recheck: [working notes](WORKING_NOTES.md) and [PR ledger](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247)
- [Day 13 hosted closeout and local gates](artifacts/day13-hosted-closeout-and-local-gates.md)
- [Day 14 exact-scope decision](artifacts/day14-exact-scope-decision.md)

## Residual Handoff

Sprint 224 must independently audit `qr-incompatible-ls` on its own exact-head Windows run and six-file artifact. It must not reuse Cholesky evidence or extend the Cholesky sentence to QR. Other Windows solver/report targets, broad cross-platform parity, package/ABI distribution, CHOLMOD comparison, timing/ranking and state-of-the-art claims remain outside Sprint 223. Keep the Sprint 220 API docs local-only guard. After this retrospective is pushed, refresh the PR ledger with the new head's hosted run and artifact; an old artifact is historical, however successful its own checks were.
