# Sprint 223 Day 13: Hosted Closeout and Local Gates

Date: 2026-11-10/11 UTC. Planned 10 hours: item 223.5 (2 h), item 223.6 (8 h).

## Hosted evidence closeout

At the start of Day 13, the clean PR head was `a312443def9fbb0bea864f3276b8ac0e6940b6bf`. [Windows run 42935926249](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935926249) and [selected Cholesky job 130245874636](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935926249/job/130245874636) succeeded for that head. [Artifact 13819095161](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935926249/artifacts/13819095161) remained downloadable and unexpired on Day 13; its API digest is `sha256:88cb511e8f15a6c2030b178750d0002b477f4a97c69ac212d33615873770ed95`, matching the downloaded ZIP. It expires `2027-02-09T01:02:07Z`.

The ZIP has exactly six files. Their SHA-256 hashes are tied to the `a312443d` source commit by the manifest and all six study rows:

| Member | SHA-256 |
| --- | --- |
| `project_observations.tsv` | `1df661119e985947fb9ab890c81ea3b55554aad16254c15a3f376fa7cc9d9b99` |
| `baseline_observations.tsv` | `7806a9bf4eb8ada7e537ff53582dce8b5d0d3d35898a69e136db92f0fb2b5b8f` |
| `dependency_status.tsv` | `53bbd048f54089ef51a0c0de928fd99a51d5b375ac71dc59809d9554769c9726` |
| `study.tsv` | `6e8ee4271d9518c87eb1dedce1bb7d34ad0b669253f64345dee4a791abd211af` |
| `summary.md` | `5d63cd014fdfd826868e78b49b5cb6aecae7f72f48988d04c03db15c050fb73b` |
| `manifest.tsv` | `5253156fc02a82315f36a31655bab64da6203dd56ac17d16c683c3e7ee3e80b4` |

The reviewer-visible [PR #247 description](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247) contains the full Day 12 run, job, artifact, timestamp, row and cross-file audit. This Day 13 document changes the branch head when committed. **The `a312443d` artifact then becomes historical; it cannot prove the new head.** A fresh selected hosted job and six-file audit must be recorded in the PR description before sign-off. The PR remains draft.

## Local quality-gate transcript

All commands below completed with exit code 0 on `a312443d`:

```text
python3 tests/test_selected_report_targets_manifest.py    test-selected-report-targets-manifest: ok
python3 tests/test_selected_comparison_workflow.py        test-selected-comparison-workflow: ok
python3 tests/test_normalize_report_index.py              test-normalize-report-index: ok
python3 tests/test_run_external_comparison.py             test-run-external-comparison: ok
python3 tests/test_validate_windows_powershell.py         test-validate-windows-powershell: ok
python3 scripts/validate_corpus_schema.py                 validate-corpus-schema: ... ok
make report-index-comparison-freshness                     passed; normalize-report-index: freshness ok (46 rows)
make docs-check                                            api-docs-coverage: PASS (18 public headers/pages)
git diff --check                                           clean
```

The local `pwsh` executable is unavailable; the Python structural suite passed and the hosted selected MSVC/PowerShell job supplied native execution. The generated comparison target produced six Cholesky `local_only` rows with `no hosted CI proof` and `no Windows report freshness`; local output did not inherit the hosted tier. No `.c` or `.h` file changed, so the C format/lint/test gate was not triggered by Sprint 223 Day 13.

## Draft claim and non-claim matrix

**Draft, not approved:** “Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense Cholesky reference helper.” This is the exact selected-manifest sentence. Day 14 must reapply the thirteen promotion criteria to the then-current PR head and artifact before accepting it.

| Surface | Accepted boundary for review |
| --- | --- |
| Fixture/reference | Only `cholesky-spd-tridiag-5` and its source-controlled dense Cholesky helper |
| Windows freshness | Only the selected job and its exact six-file artifact, contingent on current-head audit |
| Other selected targets | QR incompatible remains evidence collection; no Windows freshness for other comparison, oracle or benchmark rows |
| Numerical breadth | No broad Cholesky, SPD, reordering, CSC/list parity, fill or external-library ecosystem parity |
| Distribution/performance | No package-manager, shared-library ABI, performance superiority or state-of-the-art claim |
| API docs | Generated API HTML remains local-only under the separate workflow policy |

Any source, workflow, test or documentation change after an audited run makes that run stale for final-head sign-off. The next reviewer-visible audit must name the new head SHA, run/job/artifact URLs, artifact digest, six file hashes, row set, platform, compiler, tier, claim and non-claims.
