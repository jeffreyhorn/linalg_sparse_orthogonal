# Sprint 224 Day 11: Hosted Windows QR candidate audit

## Candidate run and decision

The Day 1–10 source candidate was committed as `eb2e3b7b4cc9a5dc5528c49e732fcc5f5aec7a38`, pushed on `sprint-224`, and opened as [draft PR #248](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248) so the pull-request workflow could exercise the exact candidate. The [Windows CI run 43768694528](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528) completed successfully. Its [selected QR job 132979511893](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528/job/132979511893) completed successfully and uploaded [artifact 14183083856](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43768694528/artifacts/14183083856).

**Day 11 result: the candidate hosted QR job and independently downloaded six-file artifact pass the selected fixture audit.** This establishes evidence for candidate commit `eb2e3b7b…`; it does not waive Day 12's final proposed-head recheck. Recording this ledger changes the branch after the tested commit, so the branch's final head must receive its own hosted run before final sign-off.

| Field | Observed value |
| --- | --- |
| Event and PR | `pull_request`, draft PR #248 targeting `master` |
| Run head / branch | `eb2e3b7b4cc9a5dc5528c49e732fcc5f5aec7a38` / `sprint-224` |
| Run | `43768694528`, created `2026-11-15T22:20:01Z`, completed successfully |
| Selected job | `132979511893`, started `22:20:04Z`, completed `22:20:44Z`, `success` |
| Artifact | `14183083856`, `sprint209-windows-selected-comparison-qr-incompatible`, created `22:20:41Z`, 4687-byte ZIP, `expired=false`, API expiry `2027-02-13T22:20:01Z` |
| API digest | `sha256:e47d2d06f09686ef8fecfc28f0d648732a96855289049913c3139ffb5faf0675` |
| Independently downloaded ZIP SHA-256 | `e47d2d06f09686ef8fecfc28f0d648732a96855289049913c3139ffb5faf0675` |
| Local audit copies | `/tmp/sprint224-day11-qr.zip`, `/tmp/sprint224-day11-qr/`, `/tmp/sprint224-day11-audit-build/`; these are ignored temporary copies, not committed proof files |

## Job and checkout trace

The selected job's checkout log fetched and checked out `eb2e3b7b4cc9a5dc5528c49e732fcc5f5aec7a38` explicitly, matching `SPARSE_SELECTED_SOURCE_SHA`, the run's PR head, the artifact metadata and the study/manifest source commit. The job configured with Visual Studio 17 2022 x64, built `sparse_lu_ortho` in Release, and ran the QR incompatible comparison generator successfully. The generator logged `external-comparison: qr-incompatible-ls project-vs-baseline comparison passed`. Its selected-target normalizer logged `freshness ok (17 rows)`. The upload step logged 4687 bytes, the same ZIP digest as the artifact API and independent download, and the same artifact ID. All selected job steps, including upload, concluded `success`.

## Six-file ZIP and byte audit

The ZIP passed CRC validation and contains exactly the six flat names in the selected manifest, with no duplicate, missing, extra or wrong-subfamily member. `gh run download` extracted bytes match the independently downloaded ZIP member bytes exactly.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 492 | `a1ef353db22039bba57ba90ec5ec89d58c9f5ba26079179e492a3f0f7843e5c7` |
| `baseline_observations.tsv` | 502 | `eb43a2d6932783690c8fd5eabe7a8b54cca5d8c66c2ab1d3de5885fc0e880aa6` |
| `dependency_status.tsv` | 475 | `6da38e9d835253a3ac377403447c647b6674647ef58433a43ea08716055afc38` |
| `study.tsv` | 13190 | `c02b3345f1e6adf42678998b2ff7bbd8f1244466c4093e74679355ce93661a40` |
| `summary.md` | 2496 | `05695417de1b558c6f2c6f9ed087405b5387a26978d2dc8d626ca5c2c5570a24` |
| `manifest.tsv` | 1866 | `69b61d95bd6e629aef57933577c5670ec65eec0f49e3d561e93c36f2dfe7eb79` |

## Row, provenance and numerical audit

The study has the six ordered `comparison_row_id` values in `SRT-COMP-QR-INCOMPATIBLE-LS`, each once with `status=pass`, `support_tier=hosted_selected`, fixture `qr_overdetermined_incompatible_4x2` and QR-only claim/non-claim fields. The manifest and all six rows agree on commit, branch `sprint-224`, clean worktree, `windows-amd64`, `cmake-probe:Visual Studio 17 2022:Release`, generated time `2026-11-15T22:20:37+00:00`, QR generator configuration, source-controlled dense QR helper, Python runtime, baseline command, project version and project probe command. Summary provenance repeats the target, fixture, commit, branch, clean worktree, platform, compiler and both commands. Its scope is selected and fixture-local; it excludes broad Windows report freshness, performance superiority and state-of-the-art claims. The five generated file paths normalize from Windows backslashes to the exact selected QR directory.

The project and baseline observation files contain four passing rows each. Every corresponding study value exactly equals the appropriate observation string. The status rows report `SPARSE_SUCCESS` and `success`. For the source-controlled 4×2 incompatible fixture, both observed solutions are within `1e-10` of `[2, -1]`; recomputed `||Ax-b||₂` is `sqrt(3)` and solution norm is `sqrt(5)` within tolerance. The maximum project–baseline solution-component delta is `2.2204460492503131e-16`, below `1e-10`; every numeric study delta is within its reported tolerance. Required interpreter and QR helper dependencies pass; optional NumPy and SciPy rows remain deferred. No Cholesky observation was used.

Running the selected normalizer again on byte-copied downloaded files at this local checkout returned `normalize-report-index: freshness ok (17 rows)` and six fresh generated QR rows for the same HEAD. The archive/row/provenance/numerical checks were also repeated independently against the downloaded bytes.

## Commands, failures and next gate

The audit used `gh api` for run, job and artifact metadata, the selected job log and the raw ZIP; `gh run download` for an independent extraction; `shasum -a 256`, `unzip -t`, source-controlled manifest and TSV parsing, byte comparisons, numerical recomputation and `python3 scripts/normalize_report_index.py --build-root /tmp/sprint224-day11-audit-build --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`.

No selected-job, artifact or Day 11 audit failure was found. A first attempt to fetch an in-progress run log was unavailable until the run finished; the completed job log was then inspected. Day 12 must freeze and recheck the final proposed commit, including this evidence ledger, and must not reuse the Day 11 ZIP as proof for a later source revision.
