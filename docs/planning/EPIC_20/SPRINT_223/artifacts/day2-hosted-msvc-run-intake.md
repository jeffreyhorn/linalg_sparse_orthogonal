# Sprint 223 Day 2: Hosted MSVC Cholesky Run Intake

## Intake decision

**A current source-commit `master` Windows run, selected job and exact six-file artifact were found and downloaded. Selected Windows Cholesky freshness remains unpromoted.** The hosted run is at the Day 1 source HEAD, `7fc86594cb45477b58cd29706c5f87e005ca99c1`; it predates the uncommitted Sprint 223 plan and notes. It is valid intake evidence for the old source commit, not evidence for a future final Sprint 223 commit. Day 3-5 must audit the rows and generated semantics, and Days 11-12 require a new exact-final-commit hosted recheck.

## Run and job identity

| Field | Observed value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `42927368282` |
| Run URL | <https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282> |
| Event, branch | `push`, `master` |
| Head SHA | `7fc86594cb45477b58cd29706c5f87e005ca99c1` |
| Run created, updated (UTC) | `2026-11-10T23:56:10Z`, `2026-11-10T23:58:54Z` |
| Run status, conclusion | `completed`, `success` |
| Selected job ID | `130217136498` |
| Selected job | `Windows selected Cholesky comparison freshness (MSVC)` |
| Job URL and logs | <https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/job/130217136498> |
| Job started, completed (UTC) | `2026-11-10T23:56:13Z`, `2026-11-10T23:56:55Z` |
| Job status, conclusion | `completed`, `success` |

The job log confirms `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`, followed by `cmake --build build --config Release --target sparse_lu_ortho`. The build produced `build/Release/sparse_lu_ortho.lib`. It then ran `python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib`. The log reported `cholesky-spd-tridiag-5 project-vs-baseline comparison passed`. The subsequent selected normalizer command used `--family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` and reported `freshness ok (17 rows)`. This log result is an intake observation; Day 3 will independently inspect the six selected rows.

## Artifact identity and download

| Field | Observed value |
| --- | --- |
| Artifact ID | `13814213527` |
| Name | `sprint190-windows-selected-comparison-cholesky` |
| URL | <https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/artifacts/13814213527> |
| API size | `4593` bytes |
| API upload digest | `sha256:469f9e6290401477df6d172cd08cdca1f5db7f7805999ac446b5a206a5defa91` |
| Created, updated (UTC) | `2026-11-10T23:56:52Z`, `2026-11-10T23:56:52Z` |
| Expires (UTC), expired | `2027-02-08T23:56:10Z`, `false` |
| Download | `gh run download 42927368282 --name sprint190-windows-selected-comparison-cholesky --dir /tmp/sprint223-day2-cholesky.csqbWpIx` exited `0` |

The upload log independently reported six uploaded files, the same ZIP digest, the artifact ID and its URL. The API digest describes the uploaded ZIP; the table below hashes each **extracted file**. The downloaded bytes are retained in the temporary audit directory above for Days 3-5; this PR does not include the generated artifact.

| Extracted file | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 470 | `1df661119e985947fb9ab890c81ea3b55554aad16254c15a3f376fa7cc9d9b99` |
| `baseline_observations.tsv` | 497 | `7806a9bf4eb8ada7e537ff53582dce8b5d0d3d35898a69e136db92f0fb2b5b8f` |
| `dependency_status.tsv` | 477 | `53bbd048f54089ef51a0c0de928fd99a51d5b375ac71dc59809d9554769c9726` |
| `study.tsv` | 12827 | `fe688714a23b27634cc282ceb5f62b1e73e84a757e33d91cb741888aa02b82f0` |
| `summary.md` | 2547 | `42cc909bc2dc9e036874e2f7028abad51d3cb61f2273d7f742e7638ca011b07a` |
| `manifest.tsv` | 1873 | `4847d4e01741674813539bae0e9c89ce9c510b154fbbcc99a5333e44b9d18cfa` |

The extracted member count is **six**, with exactly the six required basenames from the [Day 1 contract](day1-exact-target-and-hosted-intake.md). There were no extra extracted files or directories. This proves artifact membership and current download access; it does not by itself prove every row's identity, status, source or platform fields.

## Day 2 boundary and next gate

The artifact belongs to the `master` push at the Day 1 source commit. No `sprint-223` branch commit has been pushed or hosted. The selected manifest still lists only Linux/macOS and `local_only`; Day 2 changes neither. Day 3 must inspect all six `study.tsv` rows, their statuses, provenance and file paths against the hashes above. Day 4 must reconcile generator/platform and claim fields. A source edit after a hosted proof requires a new hosted run for the final commit before any promotion.

Commands used: `gh run list --workflow 'Windows CI' --limit 20 --json ...`; `gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/jobs?per_page=100`; `gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/42927368282/artifacts?per_page=100`; `gh run view 42927368282 --job 130217136498 --log`; `gh run download 42927368282 --name sprint190-windows-selected-comparison-cholesky --dir /tmp/sprint223-day2-cholesky.csqbWpIx`; and a read-only Python SHA-256 inventory of the extracted files.
