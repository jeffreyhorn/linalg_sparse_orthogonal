# Sprint 224 Retrospective

**Sprint:** 224 - Selected Windows QR Incompatible Freshness

**Duration:** 14 planned days on branch `sprint-224`

**Status:** The named QR incompatible least-squares comparison earned a bounded Windows hosted-freshness decision on the Day 14 source head. This retrospective changes that head; its own exact-head hosted run and six-file audit are required before the PR's current-head attestation can be renewed.

## Source and Definition of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md) allocate 140 estimated hours to six items and fourteen 10-hour days. These are planning estimates, not measured labor or CI duration. [Working notes](WORKING_NOTES.md), the [Day 14 decision and handoff](artifacts/day14-qr-final-decision-and-handoff.md), and [PR #248's evidence record](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248) track the actual gates and artifacts. The intended result was six fresh, passing comparison rows for the single `qr_overdetermined_incompatible_4x2` fixture on a reviewed Windows/MSVC selected lane, with exact source provenance, negative guards, and one narrow public claim. It was not a general Windows QR solver or package-parity release.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 224.1 Hosted QR Collection | 20 h | Froze the manifest/row/file contract and downloaded the existing Windows QR artifact. Its six files were intact, but the generated rows remained `local_only`; intake could not itself authorize promotion. |
| 224.2 Selected Row Audit | 30 h | Matched six ordered rows to the manifest and eight project/baseline observations; recomputed ZIP/file hashes, source and runtime provenance, 4×2 solution, residual, norms and deltas. Applied thirteen Sprint 208 gates to determine candidate eligibility. |
| 224.3 Atomic Metadata Update | 26 h | Aligned only the QR incompatible selected manifest tuple, guarded hosted generator, explicit PR-head Windows checkout, strict normalizer and bounded maintained documentation. The other QR targets and Cholesky retained separate ownership. |
| 224.4 Failure Fixtures | 24 h | Added malformed archive/upload/path/row/provenance/claim/workflow mutations and confirmed valid named QR evidence still passes while contradictions fail closed. |
| 224.5 Hosted Recheck | 22 h | Audited successful exact-head Windows candidate and proposed-head jobs, then the Day 14 committed head. Each later source edit made earlier artifacts historical. |
| 224.6 Exact-Scope Review | 18 h | Passed affected local gates, recorded the final 13-gate attestation, approved the one selected sentence for the Day 14 tested head, and handed the independent timing gate to Sprint 225. |
| **Total** | **140 h** | **Named selected QR freshness approved on audited Day 14 head; this retrospective commit requires a fresh audit.** |

## What Went Well

1. **The evidence target stayed narrow.** The selected contract named `SRT-COMP-QR-INCOMPATIBLE-LS`, target `qr-incompatible-ls`, the one incompatible 4×2 fixture, six flat files and six ordered `comparison_qr_overdetermined_incompatible_4x2_*_v1` rows. No Cholesky row, compatible QR or minimum-norm QR target was promoted by association.
2. **The Windows result was independently inspectable.** At Day 14 head `a963a75028cb1a75ec9cb336a7bb831045086d5b`, [Windows run 43770537114](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43770537114), [selected QR job 132985230692](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43770537114/job/132985230692), and [artifact 14183169633](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43770537114/artifacts/14183169633) agreed on the exact PR-head SHA. The successful job checked out that SHA, built with VS 2022 x64 Release, generated the comparison, ran selected freshness and uploaded the bundle. The API, upload log and independently downloaded ZIP shared SHA-256 `4a158851071d1c2edc4af3a546f62e1ddc2ad562f03d53952ef2e91e6016da94`. CRC, six-file membership, byte comparisons and per-file hashes passed; the PR attestation records each hash.
3. **The numerical result was checked beyond row labels.** Both sides yielded solutions within `1e-10` of `[2, -1]`. Independent reconstruction of the fixture reproduced residual norm `sqrt(3)` and solution norm `sqrt(5)`; the maximum component delta was `2.2204460492503131e-16`, below the selected `1e-10` tolerance. Six `hosted_selected` study rows and four observations per side passed, and the downloaded bundle passed the selected normalizer with `freshness ok (17 rows)`.
4. **Cross-file and negative guards constrained promotion.** Manifest, study, summary, commands, helper identity, branch, clean worktree, `windows-amd64` and compiler fields agreed. Mutation tests rejected wrong files, paths, order, source state, baseline metadata, local-only contradictions, broad wording and workflow drift. The QR selected path was added without weakening the independent Cholesky profile or local-only API-documentation guard.
5. **Claim wording followed proof.** README, INSTALL, maintainer, corpus and schema surfaces use the same bounded sentence and point to the PR for the post-commit attestation. The Day 14 decision applied all thirteen gates and kept broad Windows QR, rank policy, external packages, ABI, performance superiority and state-of-the-art claims outside the artifact.

## What Did Not Go Well

1. **The intake artifact contained a real contract contradiction.** The existing Windows QR job passed numerically, but the rows still said `local_only` and the summary denied hosted/Windows freshness. This required a target-scoped generator and metadata alignment before the observed artifact could support a claim. A successful collection job alone was insufficient.
2. **Evidence aged with every source commit.** The Day 11 candidate `eb2e3b7b…` and Day 12 proposed head `07b7f831…` each had a valid hosted artifact for its own revision. The Day 12/13 evidence notes and final wording changed the source SHA, requiring the new Day 14 run. This retrospective likewise invalidates the current-head status of the Day 14 artifact. The PR body, rather than a self-referential Git file, is the post-commit run/hash ledger.
3. **The validator needed more than a positive fixture.** Artifact, row-order, Windows path, provenance and broad-claim mutations exposed gaps that the happy path could not reveal. They were closed in the QR selected profile and regression suite before the hosted sign-off.
4. **Native Windows validation was available only on hosted runners.** The local PowerShell structural guard passed, but this macOS workstation did not provide MSVC or native PowerShell execution. The hosted Windows selected job supplied the platform-specific proof; local checks did not stand in for it.
5. **The scope is intentionally smaller than general QR support.** The source-controlled dense reference helper and one incompatible fixture cannot demonstrate raw QR basis identity, global rank-threshold behavior, external-package equivalence, broad Windows report parity or timing performance. Those remain separate evidence problems, not implicit Sprint 224 successes.

## Final Metrics and Claim

| Metric | Day 14 audited result |
| --- | --- |
| Planned effort | 140 h across six items and fourteen 10-hour days; actual hours not tracked |
| Selected target | `SRT-COMP-QR-INCOMPATIBLE-LS` / `qr-incompatible-ls` only |
| Audited source | `a963a75028cb1a75ec9cb336a7bb831045086d5b`, `sprint-224`, clean artifact worktree |
| Windows evidence | Successful selected job `132985230692` in run `43770537114`; downloadable six-file artifact `14183169633` |
| Artifact and rows | ZIP API/upload/download digest matched; six required files, six ordered passing hosted rows, four passing observations per side |
| Platform and compiler | `windows-amd64`, Visual Studio 17 2022 Release CMake probe |
| Numerical check | Solution `[2, -1]`, residual `sqrt(3)`, solution norm `sqrt(5)`, maximum cross-side delta `2.2204460492503131e-16` |
| Selected normalizer | `freshness ok (17 rows)` on downloaded final-head bundle |
| Final local checks | Selected manifest, generator, workflow, full normalizer, Windows structural, support-doc and API-doc suites and `git diff --check` passed |
| C quality trigger | No `.c` or `.h` file changed during this closeout, so `make format && make lint && make test` was not required |

For the audited Day 14 head, the approved sentence was: **“Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.”** The Windows portion is supported by the exact named artifact and the separate reviewed Linux/macOS selected lanes; this sentence is not a global solver or external-library claim. The [PR #248 attestation](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248) carries the final-head IDs, exact file hashes, thirteen PASS results and stale-evidence rule. After this retrospective is committed, that attestation must be refreshed for the new head before the current-head Windows claim is considered renewed.

## Daily Evidence

- [Day 1 exact QR target and intake](artifacts/day1-exact-qr-target-and-intake.md)
- [Day 2 hosted MSVC QR intake](artifacts/day2-hosted-msvc-qr-intake.md)
- [Day 3 artifact and row audit](artifacts/day3-qr-artifact-row-audit.md)
- [Day 4 numerical and provenance audit](artifacts/day4-qr-numerical-and-provenance-audit.md)
- [Day 5 promotion gate decision](artifacts/day5-qr-promotion-gate-decision.md)
- [Day 6 manifest and generator candidate](artifacts/day6-qr-manifest-generator-candidate.md)
- [Day 7 documentation and workflow candidate](artifacts/day7-qr-docs-workflow-candidate.md)
- [Day 8 atomic alignment and negative matrix](artifacts/day8-qr-atomic-alignment-and-negative-matrix.md)
- [Day 9 artifact, row and path regressions](artifacts/day9-qr-artifact-row-path-regressions.md)
- [Day 10 provenance and claim regressions](artifacts/day10-qr-provenance-claim-regressions.md)
- [Day 11 hosted candidate audit](artifacts/day11-hosted-qr-candidate-audit.md)
- [Day 12 proposed-head recheck](artifacts/day12-final-proposed-head-qr-recheck.md)
- [Day 13 hosted and local validation closeout](artifacts/day13-hosted-local-validation-closeout.md)
- [Day 14 selected decision and handoff](artifacts/day14-qr-final-decision-and-handoff.md)

## Residual Handoff

Sprint 225 owns the independent one-workload canonical `bench_refactor_csc`/`nos4` timing gate. It should retain strict source-head and artifact custody while keeping timing and performance rankings separate from Sprint 224's threshold-free QR correctness/freshness result. Other QR targets, broad Windows solver parity, external-package comparison, package/ABI support and state-of-the-art claims remain outside this sprint. After the retrospective commit is pushed, rerun and independently audit the selected Windows QR bundle for that new SHA and update the PR attestation; a Day 14 artifact cannot establish freshness for a later commit.
