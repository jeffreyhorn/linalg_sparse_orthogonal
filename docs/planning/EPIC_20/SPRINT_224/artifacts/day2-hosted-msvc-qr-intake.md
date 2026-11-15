# Sprint 224 Day 2: Hosted MSVC QR Artifact Intake

## Decision

**The exact intake source revision has a successful hosted Windows QR collection run and a downloadable, six-file artifact. Selected Windows QR freshness remains blocked.** The run is a `master` push at `44e3b84dab97a10b2727cc026e7124a9a5415421`, the same committed revision as local `sprint-224` HEAD at intake. The branch's untracked planning files are not part of that run. Its QR study still records `support_tier=local_only`, and its summary retains `no hosted CI proof` and `no Windows report freshness`. Days 3–5 must audit the rows and values; Days 6–10 must resolve eligible metadata/guard gaps; Days 11–14 must repeat the hosted check on the final changed head before any promotion.

## Exact-commit run and selected job

| Evidence | Observed value |
| --- | --- |
| Run | [Windows CI run 43762862057](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057) |
| Event and branch | `push` on `master` |
| Run head / local intake HEAD | `44e3b84dab97a10b2727cc026e7124a9a5415421` |
| Run created / started / updated | `2026-11-15T21:17:35Z` / `2026-11-15T21:17:35Z` / `2026-11-15T21:19:54Z` |
| Run result | `completed`, `success` |
| Selected job | [Job 132961326705](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/job/132961326705), `Windows selected QR incompatible comparison freshness (MSVC)` |
| Job timing | Started `2026-11-15T21:17:37Z`; completed `2026-11-15T21:18:19Z` |
| Job result | `completed`, `success` |

The selected job log shows checkout of `44e3b84dab97a10b2727cc026e7124a9a5415421` from `origin/master`, then:

1. `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`.
2. `cmake --build build --config Release --target sparse_lu_ortho`.
3. `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib` — log reports `project-vs-baseline comparison passed` and writes the six QR files.
4. `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` — log reports `freshness ok (17 rows)`.
5. Upload of the QR-only six-file artifact — log reports 4620 bytes, the digest below and the same artifact ID as the API.

This push run proves the actual `master` checkout for this revision. It does **not** prove that the QR job's default checkout will use the PR source head on a later `pull_request` event; the Day 1 PR-head concern remains open until a PR run is inspected or the job is corrected.

## Artifact metadata and download

| Evidence | Observed value |
| --- | --- |
| Artifact | [Artifact 14181192246](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts/14181192246), `sprint209-windows-selected-comparison-qr-incompatible` |
| Workflow artifact head / branch | `44e3b84dab97a10b2727cc026e7124a9a5415421` / `master` |
| Created / expires | `2026-11-15T21:18:16Z` / `2027-02-13T21:17:35Z` |
| Availability | API reports `expired=false`; `gh run download` and the ZIP API download both succeeded. |
| Size | 4620 bytes |
| API digest | `sha256:09755ddbdc837e2b5587e15a200836c47c3ae6bc105fd3a25cdc8989a6baa69b` |
| Independently downloaded ZIP SHA-256 | `09755ddbdc837e2b5587e15a200836c47c3ae6bc105fd3a25cdc8989a6baa69b` — matches the API and upload log. |
| Temporary local audit copy | `/tmp/sprint224-day2-qr.zip` and `/tmp/sprint224-day2-qr/`; these paths are local audit copies, not committed evidence. |

The ZIP passed its archive CRC check and contains exactly these six flat members. `gh run download` extracted bytes match the ZIP members byte for byte.

| ZIP member | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 492 | `a1ef353db22039bba57ba90ec5ec89d58c9f5ba26079179e492a3f0f7843e5c7` |
| `baseline_observations.tsv` | 502 | `eb43a2d6932783690c8fd5eabe7a8b54cca5d8c66c2ab1d3de5885fc0e880aa6` |
| `dependency_status.tsv` | 475 | `6da38e9d835253a3ac377403447c647b6674647ef58433a43ea08716055afc38` |
| `study.tsv` | 12632 | `5764b145ef9c5e80c47f73429cb11b724687c5a4c533b62a74a7707a354bb27d` |
| `summary.md` | 2441 | `d0fea4be3f97a00648a2ae31bba0e3781840c38d31f59e9b68f5331720b64e5e` |
| `manifest.tsv` | 1857 | `38a0d954a985bfebc5ee4e5b31aaa1452e93716af652448bb3386138a37acd71` |

## Preliminary content and promotion boundary

The manifest says `target=qr-incompatible-ls`, `source_commit=44e3b84dab97a10b2727cc026e7124a9a5415421`, `source_branch=master`, `worktree_state=clean`, `platform=windows-amd64` and `compiler=cmake-probe:Visual Studio 17 2022:Release`. It records `stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`. The study has six rows with `status=pass`, but all six use `support_tier=local_only`; the summary explicitly denies hosted and Windows report freshness. These preliminary facts establish a bounded hosted **collection** result, not permission to change the selected manifest or public claim. Day 3 owns exact row/observation tracing; Day 4 owns numerical and cross-file metadata audit; Day 5 owns promotion eligibility.

## Commands and failure log

```sh
gh run list --workflow 'Windows CI' --branch master --limit 15 --json databaseId,headSha,event,status,conclusion,createdAt,url
gh run view 43762862057 --json databaseId,event,headSha,headBranch,createdAt,updatedAt,status,conclusion,jobs
gh run view --job 132961326705 --log
gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43762862057/artifacts
gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/artifacts/14181192246
gh run download 43762862057 -n sprint209-windows-selected-comparison-qr-incompatible -D /tmp/sprint224-day2-qr
gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/artifacts/14181192246/zip > /tmp/sprint224-day2-qr.zip
shasum -a 256 /tmp/sprint224-day2-qr.zip
```

No Day 2 command failed. The **open promotion blockers** are intentional local-only generated metadata, absence of a QR-selected Windows manifest entry, lack of a PR-head checkout guarantee for a future pull request, and the required later exact-head recheck. No unrelated Cholesky artifact was used. If subsequent work changes the source commit, this `master` artifact becomes historical for that new head.
