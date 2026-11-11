# Sprint 223 Day 8: Atomic Candidate Review and Negative-Test Matrix

## Decision

**Keep the one-target promotion candidate; hold acceptance.** The 26-hour item 223.3 candidate now has aligned selected manifest, target-specific generator metadata, public/maintainer wording, and Windows workflow identity. Focused tests pass. The old `master` artifact is still only intake evidence. Days 9-10 must close the rejection gaps below, and Days 11-12 must audit a hosted artifact from the exact final tested commit before the proposed claim can be accepted. No broad Windows freshness claim is made by this Day 8 review.

The selected contract is `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, target `cholesky-spd-tridiag-5`, six `comparison_cholesky_spd_tridiag_5_*` rows and the six exact file names in the selected manifest. The third workflow tuple is `.github/workflows/windows-ci.yml` / `selected-comparison-freshness` / `sprint190-windows-selected-comparison-cholesky` / `windows`. The generator's selected Windows CI context produces `hosted_selected` in `study.tsv` and its configuration, a fixture-local hosted claim and the bounded non-claims. A local invocation still produces `local_only`. The five current documentation surfaces quote the same proposed manifest claim and state the final-commit proof condition.

## One-diff audit

| Surface | Day 8 observation |
| --- | --- |
| Selected manifest | Only the Cholesky row differs from HEAD; changed fields are support tier, the four aligned workflow columns, claim scope and non-claims. Row order, six required files, six row IDs, target key and generator command are unchanged. |
| Generator and normalized rows | The hosted context requires the selected job, workflow, repository, Windows platform and checked-out SHA. The new focused test compares the selected manifest non-claims to a synthetic hosted study context and `summary.md`; it confirms the configuration tier, generated claim, summary and broad exclusions agree. The normalizer accepts the current selected row set and Windows path spellings in its existing regression suite. |
| Workflow | The selected job still runs the one Cholesky target and selected normalizer, then names exactly six upload paths with `if-no-files-found: error`. No workflow command, upload destination or publication path changed in the candidate diff. |
| Public and maintainer docs | README, INSTALL, corpus README, report-index schema and maintainer guide quote the same proposed sentence; the Windows claim guard requires that exact sentence on all five surfaces. QR, oracle, benchmark, package/ABI, external-library and performance exclusions remain. |
| API docs boundary | Day 7's generated-API local-only guard passed; Day 8 did not change its workflow or publication surface. |

This is a coherent **candidate contract**. It is not yet a coherent **observed final-commit artifact**: no hosted Windows run has executed this branch candidate, and the normalizer does not currently corroborate every source file and generated metadata field. Those are explicit blockers, not waived checks.

## Negative-test matrix for item 223.4

“Current” means an existing focused rejection or audit rule. “Add” means a Day 9 or Day 10 regression and implementation gate. Proposed diagnostic labels below name the intended failure class; their final spelling can be fixed by the implementing test.

| # | Mutation / fixture | Contract rule and observable rejection | State and owner |
| ---: | --- | --- | --- |
| 1 | `study.tsv` or `manifest.tsv` source SHA differs from tested HEAD | Reject stale provenance with `source_commit does not match current HEAD`; cross-check both files and hosted run head SHA. | Study stale check exists; add cross-file and hosted audit on Days 9-10/11-12. |
| 2 | Remove `study.tsv` | Reject absent selected output as `missing_artifact_file` or required-generated missing, with target and expected path. | Current normalizer reports missing generated rows; Day 9 makes six-file diagnosis explicit. |
| 3 | Remove or rename one of the other five required files | Reject as `comparison_required_files: missing_artifact_file`, naming the exact file. | Current check catches missing file when study exists; Day 9 adds target-specific rename fixture. |
| 4 | Add a seventh file to the selected upload or downloaded archive | Reject upload path-set drift and archive membership drift; expected six names, observed extra name. | Current workflow checks required-path presence but not equality; Day 9 adds exact-set rejection and artifact audit. |
| 5 | Rename the uploaded artifact or use another artifact ID | Reject with selected job's expected artifact name, and fail hosted artifact-name/ID audit. | Current workflow/PowerShell structural test checks name; Day 9 adds artifact-identity mutation. |
| 6 | Put a wrong `artifact_path` inside a study row while leaving the physical file at the selected path | Reject as `artifact_path_mismatch`, recording row ID, recorded path and selected path. Windows backslashes and mixed separators may normalize to the exact selected path; near-matches must fail. | Current normalizer replaces the in-row path with the physical path before policy checks, so this is a confirmed gap for Day 9. |
| 7 | Change physical selected path or use `cholesky_spd_tridiag_50`, wrong suffix, or another target path | Reject with `row_set_mismatch` or exact-path diagnostic; no substring match. | Current path and near-match tests cover filtering; Day 9 adds end-to-end path mismatch fixture. |
| 8 | Delete, duplicate, add, or retarget a selected row | Reject as `row_set_mismatch` with missing/duplicate/unexpected IDs; duplicate normalized ID may fail earlier. | Current row-set tests cover stale, wrong target, duplicate and unexpected Windows-path rows; Day 9 extends exact six-row artifact fixture. |
| 9 | Set hosted row `platform` or compiler to Linux, macOS, unknown, or another Windows generator | Reject as selected Windows `platform_mismatch` / `compiler_mismatch` against selected job and manifest. | Generator context check exists, but post-generation rows are not corroborated; Day 10. |
| 10 | Set hosted study `support_tier=local_only`, configuration tier `local_only`, or restore `no hosted CI proof` / `no Windows report freshness` | Reject as `support_tier_mismatch` / `claim_contradiction`; six rows and summary must agree with the selected hosted contract. | Manifest/generator candidate test checks construction; Day 10 adds generated-file mutation and normalizer/hosted evidence checks. |
| 11 | Mutate the selected manifest's job, artifact, support tier, claim or Windows non-claim | Reject by exact selected manifest contract or claim-boundary marker. | Current manifest and PowerShell tests cover these fields; keep as Day 10 negative control. |
| 12 | Make the hosted source tree dirty or remove required run/job/artifact provenance | Reject as `dirty_worktree` / missing hosted evidence. A passing local normalizer alone cannot establish hosted freshness. | Day 10 adds local metadata negative; Days 11-12 inspect the actual run, job, branch/event, SHA, clean state and artifact. |
| 13 | Use a pull-request merge SHA as proof for a different branch-head commit | Reject `head_sha_mismatch`; audit the exact tested checkout SHA separately from PR head SHA and require the final intended commit. | Explicit Days 11-12 hosted gate. Existing PR checkout may be detached, so branch identity cannot be inferred from `git branch --show-current` alone. |

The acceptance fixture for these mutations is a clean selected Windows job with the expected artifact name, exactly six files and six distinct passing row IDs, matching source/run SHA, Windows MSVC Release metadata, coherent support tier/claim/non-claims and no broader publication route. A mutation must fail for its intended condition, not merely because an unrelated fixture field was broken.

## Verification and handoff

`python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_run_external_comparison.py`, `python3 tests/test_normalize_report_index.py`, `python3 tests/test_validate_windows_powershell.py`, `python3 scripts/validate_corpus_schema.py`, Python compile checks, and `git diff --check` pass locally. The new generator test exercises the selected manifest, hosted study context and generated summary together. Day 7's support-docs, Doxygen and API local-only checks remain the last checks for the unchanged publication boundary. Native PowerShell and MSVC execution require hosted Windows.

Day 9 owns artifact membership, path and row fixtures. Day 10 owns platform/compiler, support-tier and claim coherence. If those guards or the eventual hosted audit contradict this candidate, hold or reverse the candidate rather than describing it as promoted.
