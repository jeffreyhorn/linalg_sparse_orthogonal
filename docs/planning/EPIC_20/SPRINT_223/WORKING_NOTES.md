# Sprint 223 Working Notes

## Day 1: Freeze the exact Windows target (2026-11-10)

**State:** Intake complete; selected Windows Cholesky freshness remains **blocked pending current hosted run, job and six-file artifact inspection**. No selected manifest, generator, workflow or public claim was changed on Day 1.

**Source revision:** `master`, `origin/master` and `sprint-223` HEAD all point to `7fc86594cb45477b58cd29706c5f87e005ca99c1` at intake. The local `sprint-223` working tree has untracked planning files. A hosted proof must use the actual tested commit and report a clean source tree; this local untracked state is not a generated report row.

**Evidence ledger:** [Day 1 exact target and hosted intake contract](artifacts/day1-exact-target-and-hosted-intake.md) freezes the selected target, current manifest semantics, six required files and row IDs, workflow commands, expected platform/compiler, thirteen promotion gates, source hashes and Day 2 checklist. Sprint 208's historical master artifact is background context only. Sprint 222's local CHOLMOD evidence is outside this Windows task.

| Check or command | Day 1 outcome |
| --- | --- |
| `git status --short --branch`; `git rev-parse HEAD`; `git rev-parse master`; `git rev-parse origin/master` | Branch and revision recorded; planning files untracked. |
| Read Sprint 208 criteria, Sprint 216 RQ-04, Sprint 222 handoff and Sprint 220 guard outcome | One-target promotion criteria and local-only API-docs boundary recorded. |
| Inspect `.github/workflows/windows-ci.yml` and `tests/corpus/manifests/selected_report_targets.tsv` | Bounded Windows job/six-file upload exists; selected manifest remains Linux/macOS with `local_only` support. |
| `shasum -a 256` on workflow, manifest and prerequisite documents | Source hashes retained in Day 1 inventory. |
| Hosted run/job/artifact inspection | **Not yet performed; Day 2 gate.** |

**Owners and next gates:** Windows/report owner obtains exact-commit hosted MSVC run and downloadable artifact on Day 2. Report owner audits six files, rows, hashes, platform and generator on Days 3-5. Manifest/generator/documentation owners align the exact claim only after that audit. Guard owner adds negative fixtures on Days 8-10. Final-commit hosted rerun and reviewer-visible IDs are required before Day 14 sign-off.

**Failure log:** No Day 1 read-only check failed. The open blocker is absence of newly inspected hosted evidence, not a claim that the job or solver failed. The current manifest's Windows omission and `local_only` tier are intentional until all promotion gates pass. If a required artifact cannot be obtained, record re-deferral rather than promote from historical proof.

## Day 2: Inspect the hosted MSVC run (2026-11-10)

**State:** Current source-commit `master` run, selected job and six-file artifact inspected and downloaded; **selected Windows promotion still blocked** pending row/metadata audit and final Sprint 223 commit recheck. The full [Day 2 hosted intake ledger](artifacts/day2-hosted-msvc-run-intake.md) records run/job/artifact IDs, URLs, times, log commands, upload digest, exact extracted membership, byte sizes and SHA-256 hashes.

| Evidence | Day 2 result |
| --- | --- |
| Run | `42927368282`, `push` on `master`, head `7fc86594cb45477b58cd29706c5f87e005ca99c1`, completed successfully. |
| Selected job | `130217136498`, `Windows selected Cholesky comparison freshness (MSVC)`, succeeded after MSVC Release build, comparison, selected normalizer and upload. |
| Artifact | `13814213527`, `sprint190-windows-selected-comparison-cholesky`, unexpired and downloadable; API ZIP digest `sha256:469f9e6290401477df6d172cd08cdca1f5db7f7805999ac446b5a206a5defa91`. |
| Download | `/tmp/sprint223-day2-cholesky.csqbWpIx/`; exactly six extracted files, all independently hashed. |
| Log proof | Comparison reported pass; normalizer reported `freshness ok (17 rows)`; upload log reported six files and the matching artifact digest. |

**Next gates and owners:** Report owner audits exact rows, statuses, SHA/branch/clean state, paths and file linkage on Day 3; generator/platform owner checks Windows compiler, support tier and non-claims on Day 4; Day 5 applies all thirteen gates. The downloaded `master` artifact predates Sprint 223 source edits and cannot satisfy Days 11-12 final-commit hosted proof.

**Failure log:** No Day 2 hosted intake or download failure. Row content and generated claim semantics are deliberately **not yet accepted**. If the temporary copy is lost, redownload by the retained run/artifact ID; do not replace it with a different run while keeping these hashes.

## Day 3: Audit files and target rows (2026-11-10)

**State:** [Day 3 row and artifact traceability](artifacts/day3-row-and-artifact-traceability.md) passed for the downloaded `master` artifact: exactly six unique, manifest-owned Cholesky study rows in order, all `pass` with expected reasons and run-head provenance. No selected Windows claim is promoted.

| Audit | Result |
| --- | --- |
| Six extracted files | Byte sizes and SHA-256 values still match Day 2; no files rewritten. |
| `study.tsv` | Six expected IDs, zero missing/duplicate/extra/wrong-target rows; all statuses and reasons match the metric contract. |
| Provenance/path | All six rows use run SHA `7fc86594cb45477b58cd29706c5f87e005ca99c1`, `master`, `clean`, Windows platform/compiler and the selected backslash artifact path. |
| Observation trace | Four project and four baseline observation values match the selected study rows; `manifest.tsv` paths and summary count agree. |
| Dependencies | Required Python/helper rows pass; optional NumPy/SciPy rows defer and provide no package-baseline proof. |

**Source input:** `cholesky-spd-tridiag-5` is constructed from source-controlled generator entries and the dense baseline helper, not from an uploaded `.mtx` file. The artifact contains summarized observations and commands; the source commit identifies the input implementation. Source hashes and the row-to-file table are in the Day 3 ledger.

**Discrepancy log and next gate:** No row/file inconsistency. Generated `support_tier=local_only` and `no hosted CI proof` / `no Windows report freshness` wording remain in the downloaded evidence. Day 4 owns the platform/generator and claim-semantic reconciliation. This older `master` run cannot serve as the final Sprint 223 branch-commit proof.

## Day 4: Audit generator and platform metadata (2026-11-10)

**State:** The [Day 4 platform and claim audit](artifacts/day4-generator-platform-and-claim-audit.md) found coherent Windows AMD64, Visual Studio 2022 Release, source-commit, path and helper identities across the hosted log, `manifest.tsv`, six study rows and summary. It also confirmed the unresolved promotion contradiction: the selected manifest lists Linux/macOS only, while generated configuration and all six study rows say `support_tier=local_only` and the summary disclaims hosted CI proof and Windows report freshness.

| Gate | Day 4 result |
| --- | --- |
| Hosted compiler/platform | `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release` match the selected MSVC job's configure/build/probe commands. |
| Generated provenance | One run SHA, `master`, `clean`, fixture and target across manifest/rows; summary agrees. |
| Baseline/dependencies | Source-controlled dense Cholesky helper is the required baseline; NumPy/SciPy remain optional deferred rows. |
| Support and non-claims | **Blocked for promotion:** manifest and generator remain local-only; generated non-claims deny the bounded hosted proof. |

**Correction map:** The Day 4 ledger maps each selected manifest, generated row/configuration/summary, test, workflow and public-doc field to a narrow future candidate. Generator changes must be target- and hosted-context-specific, with local/unselected paths failing closed; manifest-only promotion would be contradictory. A final commit needs its own hosted run/artifact and run-branch provenance check.

**Failure log and next gate:** No cross-file identity mismatch or missing Day 2 artifact. The support-tier/non-claim contradiction is a real policy blocker, not a solver failure. Day 5 applies the thirteen promotion criteria and decides whether to prepare an atomic candidate or re-defer. No source-controlled metadata or public claim changed on Day 4.

## Day 5: Promotion decision from current evidence (2026-11-10)

**Decision:** [Day 5 thirteen-gate ledger](artifacts/day5-promotion-gate-decision.md) authorizes **preparing** one bounded selected Windows Cholesky candidate. It does **not** promote the claim. Gates 1-7 pass for the current `master` intake artifact; gate 8 is partial; gates 9-12 fail for current manifest/generator/docs; gate 13 retains broad non-claims. The existing hosted run is not final Sprint 223 branch-commit evidence.

| Check | Outcome |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | PASS; current re-deferral and future-promotion contract tests. |
| `python3 tests/test_selected_comparison_workflow.py` | PASS; bounded workflow and upload guards. |
| `python3 tests/test_normalize_report_index.py` | PASS; current path, freshness and row tests. |
| Day 2-4 six-file, row, platform and claim audits | Hosted input evidence coherent; promotion wording still contradictory. |

**Frozen candidate:** Only `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`; append the Windows workflow/job/artifact/platform tuple, use target-scoped `hosted_selected` semantics, retain broad non-claims, and propose exactly the sentence in the Day 5 ledger. Days 6-8 own the atomic change; Days 8-10 own negative guards; Days 11-12 must produce a new hosted run and six-file artifact for the final commit. A missing artifact, stale SHA/branch or remaining `local_only`/Windows denial returns the decision to re-deferral.

**Failure log:** No local test failure. Current gates 9-12 are known promotion failures by design. They are not silently waived by the successful `master` run. No selected manifest, generator, workflow or public claim changed on Day 5.

## Day 6: Align selected manifest and generator (2026-11-10)

**State:** The [Day 6 field-level contract and candidate ledger](artifacts/day6-manifest-generator-candidate.md) records the proposed one-row selected Windows Cholesky manifest update and matching target-scoped hosted generator output. The manifest now aligns Linux/macOS/Windows workflow, job, artifact and platform positions. The generator is designed to emit `hosted_selected` in study rows and configuration, a bounded hosted summary and non-claims that preserve broad exclusions when the exact Windows CI selected job runs; hosted output has not yet been observed for this candidate. Local invocations and other targets remain local-only. This is still a **candidate** pending Day 7 docs, Days 8-10 negative guards and Days 11-12 exact-final-commit hosted proof.

| Check | Outcome |
| --- | --- |
| Selected manifest, workflow and PowerShell structural regression scripts | PASS for the one-target candidate and other-target exclusions. |
| Local external-comparison generator regression script | PASS, including exact selected-hosted context and local isolation. |
| Corpus schema and report-index normalizer regression scripts | PASS; six-file and six-row contract unchanged. |
| `git diff --check` and Python compile check | PASS. |
| Native PowerShell / hosted Windows run | Pending: `pwsh` is absent locally; final candidate has not run on hosted MSVC. |

**Failure log and next gate:** No candidate test contradiction remains locally. The earlier `master` artifact still says `local_only` and is retained as intake evidence only; it cannot prove the new generator output. Day 7 aligns documentation, and later hosted proof must match the final commit or the candidate returns to re-deferral.

## Day 7: Align documentation and workflow guard (2026-11-10)

**State:** The [Day 7 documentation and workflow ledger](artifacts/day7-documentation-and-workflow-guard.md) records a single exact proposed Cholesky claim across README, INSTALL, corpus README, report-index schema and maintainer guide. All five explicitly keep the claim at candidate status pending a final-commit hosted MSVC job and audited six-file artifact. Windows QR, other report families, package/ABI, external-library, timing, and broad correctness claims remain excluded. Historical Sprint 182/208 evidence stays historical.

| Check | Outcome |
| --- | --- |
| Selected manifest and Windows workflow structural/claim-boundary tests | PASS; exact candidate sentence is required and a drift mutation is rejected. |
| `make support-docs-guard`; `make docs-check` | PASS; support matrix and Doxygen coverage remain valid. |
| `make api-docs-local-only API_DOCS_PYTHON=/tmp/s223-day7-api-venv/bin/python` | PASS; generated API HTML remains ignored local output and structured workflow policy finds no publication path. |
| Windows upload inspection | Six exact Cholesky paths, `if-no-files-found: error`, no changed destination; QR lane remains separate. |
| Native PowerShell and exact-final-commit hosted artifact | Pending: no local `pwsh`, and the Day 2 artifact predates the candidate. |

**Next gates:** Days 8-10 close malformed-artifact and contradictory-metadata negatives. Days 11-12 must rerun and audit the final candidate commit on the hosted selected Windows lane. An absent or mismatched artifact returns the candidate to re-deferral.

## Day 8: Atomic candidate review and rejection design (2026-11-10)

**Decision:** Keep the one-target candidate, with acceptance held. The [Day 8 atomic review and negative-test matrix](artifacts/day8-atomic-candidate-and-negative-matrix.md) checks the selected manifest, generator/study/summary, normalizer, Windows workflow and five claim surfaces as one change. A new focused generator test compares selected manifest non-claims, hosted configuration tier, study context and generated summary. The selected manifest diff touches only Cholesky and leaves row order, six files and six row IDs intact.

**Checks:** Selected-manifest, workflow, comparison generator, report-index normalizer, Windows PowerShell structural regression and corpus schema scripts passed; Python syntax and `git diff --check` passed. No `.c` or `.h` file changed. Native `pwsh` and MSVC hosted execution remain unavailable locally.

**Handoff:** The Day 8 matrix assigns 13 negative fixtures to existing checks, Day 9 artifact/row/path work, Day 10 metadata/claim work, or Days 11-12 exact hosted audit. Confirmed gaps are extra upload-path acceptance by presence-only workflow checks and loss of an in-row `artifact_path` during normalization; malformed generated metadata also lacks full cross-file corroboration. The final-commit hosted evidence and PR merge-versus-head SHA distinction remain open. No public Windows freshness claim is accepted yet.

## Day 9: Artifact, row and path guards (2026-11-10)

**State:** The [Day 9 guard and diagnostic ledger](artifacts/day9-artifact-row-path-guards.md) records exact six-path Windows upload enforcement, missing/renamed/extra selected output file rejection, source-row path validation before normalization, dirty hosted worktree rejection, and selected row-set mutations. The valid six-file Cholesky fixture and Windows absolute/backslash/mixed path spellings still pass. The one-target candidate remains unaccepted until final-commit hosted evidence is audited.

**Checks:** The normalizer, selected workflow, Windows PowerShell structural, comparison generator, selected manifest and corpus schema suites passed. Python syntax and `git diff --check` passed. No `.c` or `.h` file changed; native PowerShell/MSVC remain hosted checks.

**Failure log:** The first normalizer run failed because the new missing-study diagnostic also fired during a benchmark-only strict check. I restricted that diagnostic to an explicitly selected Cholesky target; the complete normalizer suite then passed. No remaining local guard failure was observed. Day 10 owns platform/compiler and claim-field corroboration; Days 11-12 own exact hosted run/job/artifact provenance.

## Day 10: Metadata and claim guards (2026-11-10)

**State:** The [Day 10 metadata and claim ledger](artifacts/day10-metadata-and-claim-guards.md) records fail-closed hosted Cholesky checks for platform, CMake generator, support tier, configuration, claim and non-claim text, and cross-file manifest/summary provenance. The unmutated hosted fixture passes. Local-only evidence and other selected targets retain their prior boundaries. Exact Windows workflow, job and artifact tuple mutations now have explicit manifest regressions.

**Checks:** Normalizer, selected manifest, selected comparison workflow, Windows PowerShell structural, comparison generator, corpus schema and API docs workflow policy suites passed; Python compile and `git diff --check` passed. The API docs test used a temporary venv with pinned PyYAML 6.0.3. No C or header file changed. Hosted MSVC run and final-commit artifact audit remain for Days 11–12.

**Failure log:** A first normalizer run exposed legacy contracts without `target_key`. The new guard now uses the established Cholesky subfamily; the full normalizer suite passes.

## Day 11: First hosted candidate recheck (2026-11-10/11 UTC)

**State:** [Day 11 hosted candidate evidence](artifacts/day11-hosted-candidate-recheck.md) preserves the first PR run, whose numeric job passed but whose artifact recorded the pull-request merge SHA and an unknown branch. I fixed the selected job's checkout and generator provenance, pushed candidate `66251f7f65457aae5037c32edcdbef878679e4f3`, and audited successful Windows run `42935352366`, selected job `130243952963`, and downloadable six-file artifact `13816889534`. The ZIP digest, per-file hashes, six passing rows, source SHA/branch, Windows platform, VS 2022 Release generator, bounded claims, and offline selected-target normalization all agree.

**Checks:** Generator, normalizer, selected workflow, selected manifest and Windows PowerShell structural suites passed locally; the hosted selected job and full Windows workflow completed successfully. No C or header file changed. Draft PR #247 supplies the hosted trigger.

**Handoff:** Day 11 proves its candidate commit. This evidence ledger itself changes the branch after that commit, so Day 12 must freeze and rerun the final proposed commit; no old or merge-checkout artifact counts as that proof.

## Day 12: Final proposed commit recheck (2026-11-11 UTC)

**State:** The reviewer-visible [PR #247 Day 12 evidence ledger](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247) records successful Windows run `42935926249`, selected job `130245874636`, artifact `13819095161`, and ZIP digest `sha256:88cb511e8f15a6c2030b178750d0002b477f4a97c69ac212d33615873770ed95` for clean head `a312443def9fbb0bea864f3276b8ac0e6940b6bf`. The six-file, six-row, platform, compiler, claim, observation and offline normalizer audit passed. Day 12 left the branch unchanged to keep proof exact.

**Handoff:** Any subsequent commit makes that artifact historical. Recheck the new head before final sign-off.

## Day 13: Hosted closeout and local gates (2026-11-11 UTC)

**State:** The [Day 13 gate ledger](artifacts/day13-hosted-closeout-and-local-gates.md) confirms Day 12 artifact accessibility and six per-file hashes, records the complete affected local check transcript, and drafts the exact bounded sign-off sentence with non-claims. It does not approve the claim. Publishing these notes changes the branch head, so a fresh hosted audit must follow.

**Checks:** Selected manifest, comparison workflow, normalizer, generator, Windows PowerShell structural and corpus schema suites passed. `make report-index-comparison-freshness` passed with 46 fresh rows, `make docs-check` passed with 18 public header pages, and `git diff --check` was clean. Local Cholesky output remained `local_only`. No C or header file changed.
