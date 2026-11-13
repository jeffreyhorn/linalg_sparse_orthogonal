# Sprint 223 Day 14: Exact-Scope Decision

Date: 2026-11-11 UTC. Planned Day 14 effort: 10 hours for item 223.6.

## Decision rule and source-custody boundary

Approve **only** this sentence after the final source commit has a successful exact-head selected Windows job and a downloaded, audited six-file artifact:

> Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense Cholesky reference helper.

The source-controlled selected manifest, generated hosted wording and public documentation now carry that narrow wording. The reviewer-visible [PR #247 evidence ledger](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247) is the final authority for the **current** head SHA, run, job, artifact, digest and file hashes. This file is part of the final source revision, so it cannot contain the result of a CI run that occurs only after it is committed. If the PR ledger lacks an exact-head audit, the decision is **re-defer**, regardless of an older successful run. After a matching run and audit, the decision is **approve the sentence above only**. Do not use an old artifact to bridge a source change.

## Thirteen-gate review

| Gate | Required final evidence | Day 14 disposition before final push |
| ---: | --- | --- |
| 1 | Current PR hosted run, event, head SHA and timestamps | Exact-head run required after this decision file is committed; PR ledger owns final ID |
| 2 | Named selected Cholesky MSVC job succeeds | Prior current-head job `130248750497` passed; final-head job required |
| 3 | Artifact downloadable, unexpired, digest verified | Prior artifact `13818018650` passed; final-head artifact required |
| 4 | Exactly six archive members, no extra or missing file | Day 13 audited six members; exact final archive must be rechecked |
| 5 | Six unique ordered selected row IDs | Day 13 audited six passing IDs; final rows must be rechecked |
| 6 | All rows pass and source SHA/branch/worktree match | Day 13 passed for `976347cb`; final SHA must replace it |
| 7 | `windows-amd64` and VS 2022 Release CMake probe | Day 13 passed; final rows/manifest must retain identity |
| 8 | Negative guards reject wrong paths, files, rows and provenance | Days 8–10 and Day 11 regressions pass locally |
| 9 | Only selected Cholesky manifest row names Windows workflow/job/artifact | Selected manifest, workflow and PowerShell guard suites pass |
| 10 | Hosted tier and configuration agree on selected Windows row | Generator/normalizer tests pass; local runs stay `local_only` |
| 11 | Generated claim and non-claims agree with hosted summary | Hosted row/manifest/summary audit passed on Day 13; final artifact required |
| 12 | Public docs state exact bounded claim and exclusions | README, INSTALL, maintainer guide and corpus/schema docs aligned in this revision |
| 13 | No broad Windows, external parity, package/ABI, CHOLMOD, performance or state-of-the-art inference | Manifest, generated non-claims, docs and boundary guards retain these exclusions |

The Day 13 commit's [Windows selected run](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42936797439) and [artifact](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42936797439/artifacts/13818018650) are historical after Day 14 source edits. The PR ledger must list all thirteen gates as passed against the final head before the narrow sentence is accepted. A failed macOS selected comparison check also blocks the three-platform wording.

## Failure log and corrective evidence

1. The first Day 11 PR artifact used a merge checkout SHA and `source_branch=unknown`; it was rejected for exact-head proof. The selected job now checks out the PR head and the generator verifies the head SHA/branch. Day 11's second run and Day 12/13 runs passed that provenance audit.
2. The Day 13 head initially had a failing macOS selected comparison job: its shared `selected-comparison-freshness` job name triggered Windows-only hosted-context validation. The generator now returns local-only for genuine Linux/macOS contexts while remaining strict for Windows. The regression suite passed; the subsequent [macOS selected job 130252024716](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42937785384/job/130252024716) passed on corrective commit `776e5f8e`.
3. Earlier local normalizer test failures from a missing legacy `target_key` and an overbroad missing-study check were corrected in Days 9–10. No failing local check is being counted as pass evidence.

## Day 14 local checks

```text
python3 tests/test_selected_report_targets_manifest.py     PASS
python3 tests/test_selected_comparison_workflow.py         PASS
python3 tests/test_run_external_comparison.py              PASS
python3 tests/test_normalize_report_index.py               PASS
python3 tests/test_validate_windows_powershell.py          PASS (structural; native pwsh supplied by hosted lane)
python3 scripts/validate_corpus_schema.py                  PASS
make report-index-comparison-freshness                      PASS; 46 fresh rows
make docs-check                                             PASS; 18 public header pages
python3 -m py_compile scripts/run_external_comparison.py tests/test_run_external_comparison.py  PASS
git diff --check                                            PASS
```

The local Cholesky generator still emitted six `local_only` rows with `no hosted CI proof` and `no Windows report freshness`; it did not borrow the Windows hosted tier. No `.c` or `.h` file changed, so the C format/lint/test gate was not required. The native Windows job, not the local PowerShell structural suite, is the execution proof.

## Planned-scope ledger and residual queue

| Item | Planned hours | Days | Outcome |
| --- | ---: | --- | --- |
| 223.1 Hosted Run Intake | 20 | 1–2 | Exact target and historical hosted intake recorded |
| 223.2 Row And Artifact Audit | 30 | 3–5 | Six files/rows audited; candidate decision made |
| 223.3 Atomic Promotion | 26 | 6–8 | Manifest, generator, workflow and docs aligned for one target |
| 223.4 Negative Guards | 24 | 8–10 | File/path/row/provenance/claim contradictions fail closed |
| 223.5 Hosted Recheck | 22 | 11–13 | Candidate and exact-head hosted artifacts audited; stale evidence rejected |
| 223.6 Scope Sign-Off | 18 | 13–14 | Bounded decision rule and final gate review recorded |
| **Sprint 223** | **140** | **14 days × 10 hours planned** | **Within the Epic 20 limit of 168 hours** |

This is a planned-scope accounting, not a claim of measured wall-clock engineering time. The residual queue excludes broad Windows report freshness, other solver/platform parity, package-manager and shared-library ABI support, CHOLMOD, performance ranking and state-of-the-art claims. [Sprint 224](../../PROJECT_PLAN.md#sprint-224-selected-windows-qr-incompatible-freshness) takes the separate `qr-incompatible-ls` Windows selection through its own hosted six-file audit and atomic promotion; Sprint 223's Cholesky artifact is not QR proof. API docs HTML remains local-only under the Sprint 220 guard. Any final-head CI failure or missing artifact returns this decision to re-deferral and requires correcting the source and rerunning all affected gates.
