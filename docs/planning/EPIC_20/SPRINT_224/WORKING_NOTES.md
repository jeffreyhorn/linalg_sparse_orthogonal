# Sprint 224 Working Notes

## Day 1: Freeze the QR selection contract (2026-11-15)

**State:** Day 1 source intake complete. Selected Windows `qr-incompatible-ls` freshness is **blocked pending exact-commit hosted MSVC run, selected job and six-file artifact inspection**. The manifest remains Linux/macOS-only with `local_only` support; no public QR Windows claim was changed.

**Source revision:** Local `master`, `origin/master` and `sprint-224` were `44e3b84dab97a10b2727cc026e7124a9a5415421` at intake. The local branch has untracked Sprint 224 planning files; future hosted evidence must report its own clean worktree and tested commit. A historical master/Cholesky result cannot establish QR freshness.

**Evidence ledger:** [Day 1 exact QR target and intake contract](artifacts/day1-exact-qr-target-and-intake.md) records the selected manifest row, six paths, six ordered study row IDs, QR fixture expectations, Windows workflow commands, source hashes, thirteen promotion gates and Day 2 run/artifact checklist. Sprint 208 criteria were adapted to the QR job and the actual hosted event/branch. Sprint 209's collection lane is present; its re-deferral remains in force until evidence is reviewed.

| Check or command | Day 1 outcome |
| --- | --- |
| `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `git rev-parse master`; `git rev-parse origin/master` | Branch and revision recorded; planning files untracked. |
| Read Sprint 208 criteria, Sprint 209 retrospective and Sprint 223 handoff | Thirteen gates, QR re-deferral and separate Cholesky scope recorded. |
| Inspect selected manifest, QR generator and Windows workflow | Exact QR job and six-file upload exist; manifest still excludes Windows; generator still emits `local_only` for QR. |
| `shasum -a 256` on workflow, manifest, generator, normalizer, Windows guard and prerequisite documents | Source hashes retained in Day 1 artifact. |
| Hosted QR run/job/artifact inspection | **Not yet performed; Day 2 gate.** |

**Known candidate gaps:** QR's default `actions/checkout@v4` has no explicit PR-head `ref`, while Cholesky's reviewed lane does. The QR generator lacks the Cholesky-style guarded hosted tier and source-branch override. Day 2 must inspect actual run/checkout provenance before any edit; Days 6–10 own narrow fixes and rejection fixtures if Day 5 authorizes a candidate. These observations do not assert that the QR hosted job failed.

**Owners and next gates:** Windows/report owner obtains and inspects the exact QR hosted run, selected job and downloadable ZIP on Day 2. Artifact/report owner checks six files, hashes, rows, observations and provenance on Days 3–5. Workflow/generator and manifest/docs owners align one target only after that audit. Guard owner adds negative fixtures on Days 8–10. Final source head must receive a fresh hosted run before Day 14 sign-off.

**Failure log:** No Day 1 read-only check failed. The open blocker is **uninspected hosted QR evidence**, not a known solver failure. Missing, stale or contradictory hosted proof requires re-deferral, with Linux/macOS selection and the Windows non-claim preserved.

## Day 2: Collect the hosted MSVC QR artifact (2026-11-15)

**State:** The intake revision has a successful hosted Windows QR collection job and a downloaded six-file ZIP. Selected Windows QR freshness remains **blocked** because generated QR rows retain `local_only` and the summary retains `no hosted CI proof` / `no Windows report freshness`; the final changed head will also need its own hosted recheck.

**Evidence ledger:** [Day 2 hosted MSVC QR intake](artifacts/day2-hosted-msvc-qr-intake.md) records run/job/artifact IDs, URLs, timing, commands, upload digest, independent ZIP hash, exact membership, per-file sizes and hashes, preliminary provenance, blockers and Day 3–5 ownership. The local temporary ZIP and extracted files are audit copies, not committed evidence.

| Evidence | Day 2 result |
| --- | --- |
| Run | [43762862057](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057), `push` on `master`, head `44e3b84dab97a10b2727cc026e7124a9a5415421`, completed successfully. This equals local `sprint-224` HEAD at intake, before later source edits. |
| Selected job | [132961326705](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/job/132961326705), Windows QR incompatible MSVC job, succeeded after build, generator, selected normalizer and upload. |
| Artifact | [14181192246](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts/14181192246), named QR artifact, 4620 bytes, unexpired and downloadable; ZIP/API/upload digest `sha256:09755ddbdc837e2b5587e15a200836c47c3ae6bc105fd3a25cdc8989a6baa69b`. |
| Files | Exactly six required members; ZIP CRC and byte comparisons passed. Per-file hashes are in the Day 2 ledger. |
| Preliminary rows | Six study rows have `status=pass`, Windows AMD64/VS 2022 Release provenance and a clean `master` checkout, but `support_tier=local_only`. |

**Next gates:** Day 3 traces each row to exact observation files and checks row IDs and hashes. Day 4 checks numerical tolerances and manifest/summary provenance. Day 5 applies all promotion gates. Days 6–10 own a QR-only metadata/guard candidate if eligible, and Days 11–14 must inspect a new artifact from the final proposed source head. A later PR run must also prove PR-head rather than merge-checkout provenance.

**Failure log:** No Day 2 command failed. Hosted collection succeeded, while selected promotion remains blocked by the QR `local_only`/non-claim contract and the need for a final-head recheck. The Cholesky result is separate and supplies no QR evidence.

## Day 3: Verify archive bytes and study rows (2026-11-15)

**State:** [Day 3 QR artifact and row audit](artifacts/day3-qr-artifact-row-audit.md) passed for the Day 2 downloaded ZIP on intake commit `44e3b84dab97a10b2727cc026e7124a9a5415421`. The ZIP SHA-256 matches the artifact API digest, its six flat members and extracted bytes match the selected file contract, and all six ordered study row IDs match the selected manifest without missing, duplicate, extra or wrong-target rows.

**Traceability:** All six study rows have `status=pass` and expected metric-specific reasons. Commit, `master` branch, clean worktree, fixture, Windows-style study path and generated timestamp agree with the manifest and upload timing. Project and baseline TSVs each have four passing observation rows. Every study project/baseline value exactly matches its corresponding observation; the maximum-delta row reuses the two solution-value observations. The artifact and row-to-observation table with six per-file hashes is in the Day 3 ledger.

**Checks:** Recomputed ZIP and per-file SHA-256, compared ZIP and extracted bytes, parsed the source-controlled selected manifest and three TSVs, checked six-row order/uniqueness, status reasons, source fields, artifact path, timestamp and exact observation strings. All Day 3 checks passed. No `.c`, `.h`, generator, manifest, workflow or public-doc file changed.

**Next gates and failure log:** Day 4 will recompute numerical deltas/tolerances and audit generator, baseline, platform and summary provenance. Day 5 will decide whether a QR-only promotion candidate is eligible. No structural or value-trace discrepancy was found on Day 3. Promotion remains **blocked** by `local_only`, the explicit Windows/hosted non-claims and the need for a final-head hosted rerun.

## Day 4: Reconcile numerical and generator provenance (2026-11-15)

**State:** [Day 4 numerical and provenance audit](artifacts/day4-qr-numerical-and-provenance-audit.md) passed for the Day 2 Windows QR bundle. Recomputing the source-controlled 4×2 fixture gives the expected nonzero residual `sqrt(3)` and solution `[2, -1]`. Both observed solutions reproduce the reported residual and solution norm; the maximum project–baseline component difference is `2.2204460492503131e-16`, below the `1e-10` limit.

**Provenance:** Manifest, six study rows and summary agree on target/fixture, source SHA/branch, clean worktree, Windows AMD64/VS 2022 Release, generator configuration, baseline command/helper/Python runtime and project command. Required `python3` and source-controlled helper dependencies pass; optional NumPy/SciPy rows remain deferred. The generated claim and non-claims match the QR generator's current local-only scope.

**Checks:** Parsed the six downloaded files as UTF-8; compared observation and study values, recomputed matrix residual and vector norms, recalculated four numerical deltas, checked all applicable `1e-10` tolerances and source-controlled fixture values, and reconciled fifteen shared manifest/study provenance fields plus summary and dependency states. All checks passed; no code, manifest, workflow or public claim changed.

**Next gate and failure log:** No numerical or cross-file provenance discrepancy was found. The generated QR evidence is still `local_only` with explicit `no hosted CI proof` and `no Windows report freshness`, while the selected manifest still excludes Windows. Day 5 must apply the promotion gates and choose a QR-only atomic candidate or re-deferral; a later final-head hosted run remains mandatory.

## Day 5: Decide QR promotion eligibility (2026-11-15)

**Decision:** [Day 5 QR promotion gate ledger](artifacts/day5-qr-promotion-gate-decision.md) authorizes preparation of a **QR-only atomic candidate**, not public Windows QR promotion. The exact intake-commit hosted run, selected job, downloaded six-file ZIP, row audit and numerical/provenance audit satisfy the mandatory hosted intake gates. Current `local_only` generator/manifest fields and Windows/hosted non-claims are promotion failures requiring synchronized changes, guard coverage and a later final-head hosted recheck.

**Gate tally:** Seven intake passes (1–7), gate 8 partial, gates 9–12 fail for promotion, gate 13 passes the current broad non-claim boundary. The complete thirteen-gate table, affected-file/owner map and conditional one-sentence claim are in the Day 5 ledger. No selected manifest, workflow, generator or public documentation was promoted today.

**Checks:** Selected-target normalizer on byte-copied Day 2 bundle: `freshness ok (17 rows)` with six fresh QR generated rows for intake HEAD `44e3b84dab97a10b2727cc026e7124a9a5415421`. Current manifest, selected comparison workflow, Windows PowerShell structural and full normalizer test scripts all passed. Day 3 and Day 4 artifact/numerical audits passed. No `.c` or `.h` file changed.

**Handoff:** Day 6 may align only `SRT-COMP-QR-INCOMPATIBLE-LS` and guarded generator metadata. Day 7 owns bounded docs/workflow wording; Days 8–10 own negative fixtures. Days 11–14 must obtain and inspect an artifact from the final changed source head, including PR-head checkout provenance. Any unreconciled hosted field, accidental local-to-hosted promotion, failing guard or missing final artifact returns the decision to re-deferral.

## Day 6: Align the selected manifest and generator (2026-11-15)

**State:** [Day 6 QR manifest/generator candidate](artifacts/day6-qr-manifest-generator-candidate.md) aligns the single `SRT-COMP-QR-INCOMPATIBLE-LS` row with the named Windows workflow, QR job, artifact, platform and `hosted_selected` support tier. This is a candidate for final-head validation, not an approved public Windows QR freshness claim. A parsed before/after check found no change to any other manifest row.

**Generator boundary:** Only the exact QR Windows hosted job with matching repository, event, platform, checked-out SHA/branch and clean worktree can emit `hosted_selected`. Other QR runs retain `local_only` and original hosted/Windows denials. Hosted wording is limited to the fixture and selected Windows lane; broad QR, least-squares, rank, package, ABI and performance exclusions remain. Six required files and six ordered row IDs are unchanged.

**Checks:** Selected manifest, generator, selected comparison workflow, Windows PowerShell structural and normalizer Python suites passed; `git diff --check` passed. The existing local QR bundle is stale against current HEAD and is not used as promotion evidence. No `.c` or `.h` file changed.

**Handoff:** Day 7 must give the QR job an explicit PR-head checkout and source SHA/branch variables, then align bounded public wording. Days 8–10 must extend strict hosted normalizer checks and negative fixtures to QR. The intake Windows artifact predates these edits; Days 11–14 must inspect a new exact-head hosted bundle before selected Windows QR freshness can be approved. Any failed gate requires re-deferral.

## Day 7: Align docs and workflow ownership (2026-11-15)

**State:** [Day 7 documentation/workflow candidate](artifacts/day7-qr-docs-workflow-candidate.md) synchronizes README, INSTALL, maintainer, corpus and schema wording with the one selected QR manifest sentence. Each public surface labels that sentence conditional on a final changed-head hosted MSVC job and audited six-file artifact. Broad QR, least-squares, Windows report, rank, package/ABI, performance and state-of-the-art claims remain excluded.

**Workflow:** The QR Windows job now uses explicit PR-head or push-SHA checkout, with matching `SPARSE_SELECTED_SOURCE_SHA` and `SPARSE_SELECTED_SOURCE_BRANCH` values. The generator, selected normalizer, six-file fail-closed upload, artifact name, Windows runner and 20-minute timeout remain scoped to the QR job. The selected workflow and Windows PowerShell guards require the provenance tokens; mutation tests remove each and fail.

**Checks:** Selected workflow, Windows structural, manifest and generator test scripts passed; `make support-docs-guard`, `make docs-check`, the pinned-PyYAML API-docs local-only policy and `git diff --check` passed. Direct local PowerShell validation reported `UNAVAILABLE` because `pwsh` is absent; the structural suite passed and this is not hosted proof. No `.c` or `.h` file changed.

**Handoff:** Day 8 owns strict QR hosted normalizer alignment and rejection-case freeze. The conditional sentence remains unaccepted until a new final changed-head hosted QR artifact passes the Days 11–14 audit. The Cholesky selected lane and local-only API-docs guard remain separate.

## Day 8: Finish atomic alignment and specify rejections (2026-11-15)

**State:** [Day 8 atomic candidate and negative-fixture matrix](artifacts/day8-qr-atomic-alignment-and-negative-matrix.md) closes the source-level alignment for item 224.3. The QR-only selected manifest, guarded generator, exact-head Windows job, six-file upload, maintained conditional docs and normalizer now use the same fixture, support tier and claim boundary. This is still a candidate; the final changed-head hosted MSVC artifact remains required before accepting Windows QR freshness.

**Normalizer:** Extended the strict selected hosted cross-file validator from Cholesky to QR incompatible, with exact QR configuration, baseline helper, Windows compiler, fixture, claim/non-claim and summary expectations. A Windows QR study downgraded to `local_only` now fails closed; non-Windows local QR studies remain supported. Added a positive QR hosted fixture and first metadata mutation checks. Cholesky remained on its own expected-value profile.

**Checks:** `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_run_external_comparison.py`, `python3 tests/test_normalize_report_index.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_validate_windows_powershell.py`, `make support-docs-guard` and `git diff --check` passed. A parsed atomic audit confirmed one changed manifest row, three aligned workflow lanes and the same conditional claim on all five maintained text surfaces. No `.c` or `.h` file changed.

**Handoff:** The frozen matrix names missing/extra/renamed files, stale SHA, wrong/duplicate rows, wrong path/compiler/baseline identity, contradictory hosted/local-only wording and workflow job/upload drift with expected diagnostics. Days 9–10 must turn remaining planned entries into negative fixtures and close any guard gap they expose. Days 11–14 must audit a new exact-head Windows QR bundle; no Day 8 local check substitutes for that evidence.

## Day 9: Add artifact and row failure fixtures (2026-11-15)

**State:** [Day 9 QR artifact, row and path regressions](artifacts/day9-qr-artifact-row-path-regressions.md) exercise the exact six-file bundle, six ordered rows, source state, and Windows path forms. The valid hosted QR fixture passes; malformed bundles fail with the intended diagnostic. This remains a local source candidate pending the final changed-head hosted MSVC artifact.

**Guard changes:** The normalizer applies strict selected-file membership to QR, including unexpected sidecars, and rejects out-of-order QR study rows. The selected workflow test parses the upload path block as an ordered list, catching extra and duplicate entries. Upload mutations cover renamed, absolute Windows, backslash, mixed-separator, extra-directory and backup-suffix paths; normalizer mutations check the accepted study-path forms and reject near matches.

**Checks:** The full normalizer, selected workflow, Windows structural, selected manifest and comparison generator Python suites passed. The focused Day 9 hosted mutations passed separately. `git diff --check` passed. No `.c` or `.h` file changed.

**Handoff:** Day 10 must finish the provenance and claim mutation matrix, including wrong baseline identity, contradictory support wording, job/source wiring and Cholesky substitution. Days 11–14 still require a fresh final-head hosted QR job and audited six-file ZIP before any Windows QR freshness decision.

## Day 10: Add provenance and claim failure fixtures (2026-11-15)

**State:** [Day 10 QR provenance and claim rejection ledger](artifacts/day10-qr-provenance-claim-regressions.md) completes item 224.4's negative-case matrix. A valid hosted QR fixture passes; wrong row, manifest, summary, generator context and cross-target evidence fail closed. The QR Windows claim remains conditional pending final-head hosted MSVC evidence.

**Guard change:** A QR hosted summary containing an appended `no hosted CI proof` or `no Windows report freshness` denial, or an explicit broad/all-Windows QR claim, now fails even when its selected scope and non-claim line otherwise match. Row, manifest and summary mutations verify field-level diagnostics for platform, compiler, generator configuration, baseline identity/runtime/helper, source branch, provenance and bounded claims. A substituted Cholesky study is rejected by QR selection while the independent Cholesky fixture continues to pass.

**Checks:** The full normalizer, comparison generator, selected manifest, comparison workflow and Windows PowerShell structural Python suites passed. `make support-docs-guard`, `make docs-check` and `git diff --check` passed. No `.c` or `.h` file changed.

**Handoff:** Day 11 must run or locate a new exact-head hosted QR job, inspect the six-file artifact and compare its provenance with the candidate source revision. Local regression fixtures and the earlier intake ZIP do not establish final Windows QR freshness.

## Day 11: Run and audit the hosted QR candidate (2026-11-15)

**State:** [Day 11 hosted QR candidate audit](artifacts/day11-hosted-qr-candidate-audit.md) passes for candidate commit `eb2e3b7b4cc9a5dc5528c49e732fcc5f5aec7a38`. Draft [PR #248](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248) triggered [Windows run 43768694528](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528). The [selected QR job 132979511893](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528/job/132979511893) and the overall run succeeded; the job explicitly checked out the PR-head SHA, built with VS 2022/MSVC, generated the QR report, passed selected freshness and uploaded [artifact 14183083856](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528/artifacts/14183083856).

**Evidence:** The artifact API, upload log and independently downloaded ZIP agree on digest `sha256:e47d2d06f09686ef8fecfc28f0d648732a96855289049913c3139ffb5faf0675`. Exactly six flat files passed CRC and byte-for-byte extraction checks. Six ordered passing `hosted_selected` rows, four project and four baseline observations, required/optional dependency statuses, manifest, summary, Windows paths, source/branch/clean-tree fields and numerical tolerances agree. The downloaded bundle passed the local selected normalizer at candidate HEAD: `freshness ok (17 rows)`. File sizes, hashes, IDs and numerical worksheet are in the Day 11 ledger.

**Failure log and next gate:** No selected-job or artifact failure remains. The Day 11 evidence ledger is a post-run branch change, so Day 12 must freeze the final proposed commit and rerun the selected hosted QR job for that new SHA. The Day 11 ZIP proves only `eb2e3b7b…`; it cannot be promoted as final-head evidence after source drift.
