# Sprint 223 Day 11: Hosted Candidate Recheck

Date: 2026-11-10/11 UTC. Item 223.5, first 10 hours planned.

## Outcome

The first PR job passed numerically but **failed exact-candidate provenance**. I preserved its identity below, corrected the selected Windows job to check out the PR head, and reran the candidate. The second run, [42935352366](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935352366), completed successfully. Its [selected Cholesky job 130243952963](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935352366/job/130243952963) checked out candidate commit `66251f7f65457aae5037c32edcdbef878679e4f3`, passed generation and selected-target freshness (`17 rows` in the normalized output), and uploaded [artifact 13816889534](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42935352366/artifacts/13816889534). This is Day 11 candidate evidence. Day 12 must rerun the final proposed commit after this ledger is committed.

The run event is `pull_request` for [draft PR #247](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/247), branch `sprint-223`. Run head SHA, checkout ref, `SPARSE_SELECTED_SOURCE_SHA`, generated manifest and all study rows agree on `66251f7f65457aae5037c32edcdbef878679e4f3`. The job log shows `HEAD is now at 66251f7` and the artifact upload finalized at `2026-11-11T00:58:13Z`. The full Windows CI workflow completed successfully at `2026-11-11T00:59:57Z`.

## Preserved failed candidate

The earlier [run 42934788118](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42934788118), [job 130242046067](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42934788118/job/130242046067) and [artifact 13817277790](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42934788118/artifacts/13817277790) succeeded as CI, but the artifact's source commit was PR merge SHA `067856c144e5c3b10350582536b54e6d4092494b` and source branch was `unknown`; the PR head was `e6272d8505137462707b41d0c0ea6c8d6218ee27`. Its ZIP digest was `sha256:d2750ddc8b3ed7d4c51bac6489c40e4641a547d2f19261907a0a69ac2535ce3a`. It is rejected as exact-head evidence and is not substituted for the rerun.

The fix pins only the selected Cholesky job's checkout to `${{ github.event.pull_request.head.sha || github.sha }}`, carries the trusted head SHA and branch into the generator, and rejects unknown hosted branches in normalization. Pull-request merge SHA remains distinct from the report source SHA; push events still require source SHA to equal `GITHUB_SHA`. Generator, normalizer, workflow and manifest regressions passed before the push.

## Download and custody

GitHub reports artifact `13816889534` as unexpired and downloadable, created `2026-11-11T00:58:12Z`, expiring `2027-02-09T00:57:27Z`, size `4612` bytes. Downloaded ZIP SHA-256 `00933fc0fc9cb8983825e58422084432b2837fb879793ae4a2c7891dc2b3f115` equals the API digest. `unzip -Z1` lists exactly the six required names; extraction is retained at `/tmp/sprint223-day11-fixed.TSlszNYV/` and the ZIP at `/tmp/sprint223-day11-fixed-artifact.zip`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 470 | `1df661119e985947fb9ab890c81ea3b55554aad16254c15a3f376fa7cc9d9b99` |
| `baseline_observations.tsv` | 497 | `7806a9bf4eb8ada7e537ff53582dce8b5d0d3d35898a69e136db92f0fb2b5b8f` |
| `dependency_status.tsv` | 477 | `53bbd048f54089ef51a0c0de928fd99a51d5b375ac71dc59809d9554769c9726` |
| `study.tsv` | 13049 | `77f27fd8ac685a1dbf497d370094158cdbdf1bd27f1f52dc0afdf54ce4aeab60` |
| `summary.md` | 2546 | `cf51a748611289fe37335ee44c19ee640fa70dfa109eae6b05870460b7f7c495` |
| `manifest.tsv` | 1882 | `aeff8fd5d1587375258f7af91dfa3131fc628850bf3ecfe379ec455c8df1f51d` |

## Row, platform, generator and claim audit

The six unique study row IDs match the selected manifest's ordered `expected_row_ids`; all have `status=pass`, matching `source_commit`, `source_branch=sprint-223`, `worktree_state=clean`, `platform=windows-amd64`, `compiler=cmake-probe:Visual Studio 17 2022:Release`, `support_tier=hosted_selected`, and configuration `stage=sprint183_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=hosted_selected`. The exact claim is fixture-local on the selected Windows hosted lane. Non-claims retain broad Windows, package/ABI, performance and state-of-the-art exclusions. The summary records the same source, branch and non-claims.

The project and baseline observation files each have four passing rows. Project/baseline status, residual norm, solution norm and solution vector values match their corresponding study fields. Both required dependency rows pass; optional NumPy and SciPy rows remain deferred and are not proof. The selected source-controlled fixture and baseline helper remain the same as in the [Day 3 traceability audit](day3-row-and-artifact-traceability.md).

After placing the downloaded six files under an isolated `build/comparison/cholesky_spd_tridiag_5` tree, I ran:

```text
python3 scripts/normalize_report_index.py --build-root /tmp/sprint223-day11-audit-tree/build --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5 --output /tmp/sprint223-day11-audit-tree/normalized.tsv
normalize-report-index: freshness ok (17 rows)
```

The local generator, normalizer, selected workflow, selected manifest and Windows PowerShell structural suites passed after the provenance fix. Native PowerShell/MSVC execution was supplied by the successful hosted job. No C or header file changed.

## Handoff

This ledger changes the branch after the checked candidate `66251f7f`. Day 12 must commit the final proposed source revision, rerun the selected hosted job for that exact head SHA, redownload its artifact, and repeat the digest, row, platform, generator and claim audit. Any later source edit makes that final proof stale.
