# Sprint 223 Day 5: Promotion Gate Decision

## Decision

**Prepare one bounded promotion candidate; do not promote selected Windows Cholesky freshness yet.** The current `master` hosted run, selected MSVC job, six-file artifact and selected rows pass the intake gates. The source-controlled manifest, generated support tier/non-claims and public wording do not yet agree with a promoted Windows claim. Normalizer path coverage is substantial but the full negative matrix is not closed. Days 6-10 may build and guard an atomic candidate; Days 11-12 must rerun its exact final commit on the hosted lane. If a mandatory hosted gate later fails, retain Windows exclusion and record re-deferral.

This is a decision to **prepare** a candidate, not an acceptance of the old `master` artifact as proof for the future Sprint 223 branch commit. No selected manifest, generator, workflow or public claim was changed on Day 5.

## Frozen intake evidence

| Evidence | Exact identity |
| --- | --- |
| Source at intake | `7fc86594cb45477b58cd29706c5f87e005ca99c1`, `master` push |
| [Windows CI run](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282) | ID `42927368282`, completed successfully, created `2026-11-10T23:56:10Z` |
| [Selected Cholesky job](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/job/130217136498) | ID `130217136498`, `Windows selected Cholesky comparison freshness (MSVC)`, successful |
| [Six-file artifact](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/artifacts/13814213527) | ID `13814213527`, `sprint190-windows-selected-comparison-cholesky`, unexpired/downloadable, ZIP digest `sha256:469f9e6290401477df6d172cd08cdca1f5db7f7805999ac446b5a206a5defa91` |
| Extracted file proof | Six exact names and SHA-256 values in the [Day 2 ledger](day2-hosted-msvc-run-intake.md); six exact selected rows and cross-file checks in the [Day 3 ledger](day3-row-and-artifact-traceability.md) |
| Generator/claim proof | Field-by-field contradiction and correction map in the [Day 4 ledger](day4-generator-platform-and-claim-audit.md) |

The old `master` run is relevant because its source commit equals Day 1's base HEAD, and its selected job/artifact are current and inspectable. The uncommitted Sprint 223 branch already differs in working-tree content, so the artifact is **intake proof only**. It cannot be relabeled as a final branch artifact or used to assert final-commit freshness.

## Thirteen-gate evaluation

| Sprint 208 gate | Day 5 verdict | Evidence and remaining condition |
| ---: | --- | --- |
| 1. Current hosted run inspected | **Pass for intake** | Run `42927368282` is the latest inspected `master` Windows CI push at intake SHA; URL, event, timestamps and success recorded. Final branch commit still needs a new run. |
| 2. Exact selected job succeeded | **Pass for intake** | Job `130217136498` succeeded; its log shows MSVC Release configure/build, selected comparison, normalizer and upload. |
| 3. Artifact available | **Pass for intake** | Artifact `13814213527` is unexpired, downloaded and identified by size, expiry and ZIP digest. |
| 4. Artifact membership exact | **Pass for intake** | Exactly `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv`; no extra extracted files. |
| 5. Row identity exact | **Pass for intake** | Six unique `comparison_cholesky_spd_tridiag_5_*` IDs equal the selected manifest set in order. |
| 6. Row status/provenance clean | **Pass for intake** | Every row is `pass`, has the metric's expected reason, run SHA, `source_branch=master` and `worktree_state=clean`. Future PR/branch run needs its own branch-provenance check. |
| 7. Platform/compiler bounded | **Pass for intake** | All six rows and `manifest.tsv` say `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release`, consistent with job commands. |
| 8. Path and negative guards | **Partial** | Existing normalizer tests cover backslash, mixed separator, absolute Windows suffix, near-match, stale, failed, wrong-target, duplicate and changed-row-set cases; workflow tests reject broad/missing uploads. Cholesky-specific missing/renamed file, extra-file and artifact-path mismatch regressions still need explicit closure in Days 8-10. |
| 9. Selected manifest Windows metadata | **Fail** | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` still lists Linux/macOS workflow file, job, artifact and platform only. |
| 10. Support tier and claim agreement | **Fail** | Selected manifest and six generated rows say `local_only`; generated configuration also embeds `support_tier=local_only`. A hosted-selected manifest alone would contradict them. |
| 11. Generated evidence wording | **Fail** | Six study rows and `summary.md` still contain `no hosted CI proof` and `no Windows report freshness`, and summary says “local generated study.” |
| 12. Public/maintainer docs agree | **Fail for promotion** | Current docs accurately describe bounded workflow proof and re-deferral; they do not yet describe a source-controlled promoted Windows selected target. |
| 13. Broad non-claims retained | **Pass for current boundary** | Current manifest and generated artifact exclude broad Windows, package/ABI, performance, release, external-library parity and state-of-the-art claims. Candidate must retain these exclusions while narrowing the Windows wording to `no broad Windows report freshness`. |

**Gate tally:** seven intake passes (1-7), one partial guard gate (8), four promotion failures (9-12), and one current-boundary pass (13). This tally is not a promoted status: all thirteen must pass on the final candidate and exact hosted commit.

## Candidate scope frozen for Days 6-12

- **One selected row:** `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, target `cholesky-spd-tridiag-5`, six file names and six row IDs from the Day 1 contract. Do not add QR, oracle, benchmark, CHOLMOD or unselected comparison targets.
- **One Windows metadata tuple:** append `.github/workflows/windows-ci.yml`, job key `selected-comparison-freshness`, artifact name `sprint190-windows-selected-comparison-cholesky`, platform `windows` in aligned third positions after Linux/macOS.
- **One candidate support tier:** `hosted_selected` for this selected target's verified hosted lane. Local generator invocations and unrelated targets must not inherit hosted status. Keep `study.tsv` tier and configuration consistent with the selected manifest.
- **One proposed selected-manifest sentence:** “Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense Cholesky reference helper.” This exact wording is already encoded as the future-promotion contract in `tests/test_selected_report_targets_manifest.py`; it remains **proposed**, not an approved public claim.
- **One explicit Windows non-claim:** `no broad Windows report freshness`, replacing the contradictory `no Windows report freshness` only for the promoted target/lane. Keep broad Cholesky, package/ABI, performance and state-of-the-art exclusions.
- **Final proof rule:** compare the final run's head SHA, branch/event, clean worktree, job success, unexpired six-file artifact, row set, file hashes, platform/compiler and generated claim fields against the final commit. Re-run if any tracked source or documentation changes after that proof. A missing artifact blocks promotion.

## Change owners and stop conditions

| Owner/surface | Candidate work | Stop condition |
| --- | --- | --- |
| Report/manifest owner | Align the exact selected target row, generated tier/configuration/summary, normalized row and six-file contract together. | Any `local_only` or contradictory Windows/hosted non-claim remains in the proposed hosted artifact. |
| Workflow/guard owner | Keep the bounded selected MSVC job and exact upload; add missing/renamed/extra/path-mismatch and wrong-platform negatives. | Any malformed artifact or row can satisfy selected freshness. |
| Docs owner | Align README, INSTALL, corpus/schema, maintainer and planning wording to the exact candidate sentence and explicit non-claims. | A broader Windows, external package, ABI, timing or superiority claim appears. |
| Hosted evidence owner | Recheck the candidate and then the final proposed commit, retaining run/job/artifact IDs and downloadable bytes. | Job fails, artifact is unavailable, SHA/branch is stale or generated semantics contradict the selected manifest. |

The Sprint 220 structured API-docs workflow guard stays in force, and generated API HTML remains local-only. Sprint 222's blocked CHOLMOD pilot supplies no evidence for this candidate. If source-controlled and hosted proof cannot be made coherent, keep the manifest Linux/macOS-only and record the unresolved gate in the sprint closeout.

## Day 5 checks

`python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py` and `python3 tests/test_normalize_report_index.py` all passed against the current source. They verify the **current re-deferral contract** and existing negative paths, not the proposed promotion. The Day 2-4 read-only artifact audits also passed. No `.c` or `.h` files or implementation files changed on Day 5; final candidate tests and hosted proof remain future gates.
