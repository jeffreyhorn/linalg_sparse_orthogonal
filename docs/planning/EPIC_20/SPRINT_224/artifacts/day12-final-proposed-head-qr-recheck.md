# Sprint 224 Day 12: Proposed-head Windows QR recheck

## Frozen revision and hosted result

The Day 11 evidence notes were committed and pushed as proposed head `07b7f831ffee04f90a25f6cd11182b141eb38f07`. Draft [PR #248](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/pull/248) triggered a new [Windows CI pull-request run 43769185219](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219) on that exact SHA. The overall run and its [selected QR job 132981018706](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219/job/132981018706) concluded `success`: checkout, Visual Studio 2022 x64 configure, Release library build, QR report generation, selected-target freshness and six-file upload all passed. This is distinct from the Day 11 run at `eb2e3b7b…`.

| Evidence field | Day 12 value |
| --- | --- |
| PR / event / branch | Draft PR #248 / `pull_request` / `sprint-224` |
| Tested source SHA | `07b7f831ffee04f90a25f6cd11182b141eb38f07` |
| Run | `43769185219`, created `2026-11-15T22:25:11Z` |
| Selected job | `132981018706`, started `22:25:14Z`, completed `22:25:50Z`, `success` |
| Artifact | [14183482977](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/43769185219/artifacts/14183482977), `sprint209-windows-selected-comparison-qr-incompatible`, created `22:25:47Z`, 4688-byte ZIP, `expired=false`, expires `2027-02-13T22:25:11Z` |
| API/upload ZIP digest | `sha256:36cdbf8ba494b1fa531d7ef4016317691e52da7d9266327cabf54d7e69050ec4` |
| Independently downloaded ZIP SHA-256 | `36cdbf8ba494b1fa531d7ef4016317691e52da7d9266327cabf54d7e69050ec4` |
| Local audit copies | `/tmp/sprint224-day12-qr.zip`, `/tmp/sprint224-day12-qr/`, `/tmp/sprint224-day12-audit-build/`; temporary only |

The job log explicitly shows `actions/checkout` fetching and checking out `07b7f831ffee04f90a25f6cd11182b141eb38f07`, and the job environment's `SPARSE_SELECTED_SOURCE_SHA` matches it. The generator logged `external-comparison: qr-incompatible-ls project-vs-baseline comparison passed`; the normalizer logged `freshness ok (17 rows)`. The upload log reports 4688 bytes, the digest above and artifact ID `14183482977`.

## ZIP and file hashes

The raw API ZIP passed CRC validation and contains exactly the six flat filenames required by `SRT-COMP-QR-INCOMPATIBLE-LS`. A separate `gh run download` extraction matches every ZIP member byte for byte. The artifact API, upload log and independent ZIP hash agree.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `project_observations.tsv` | 492 | `a1ef353db22039bba57ba90ec5ec89d58c9f5ba26079179e492a3f0f7843e5c7` |
| `baseline_observations.tsv` | 502 | `eb43a2d6932783690c8fd5eabe7a8b54cca5d8c66c2ab1d3de5885fc0e880aa6` |
| `dependency_status.tsv` | 475 | `6da38e9d835253a3ac377403447c647b6674647ef58433a43ea08716055afc38` |
| `study.tsv` | 13190 | `26673ba972db21c065de112431bd29164efedae8471d3f8e3ba697ef0a6f2d7e` |
| `summary.md` | 2496 | `d81e8d67bfc677c7908e50702a1f3a10df6eba9b328f469683b08823f7d8a4d3` |
| `manifest.tsv` | 1866 | `5d3b3ce15ba304322a82dee46442756d38aa7fbf5da539e3ecb8c196e5b1394b` |

## Row and cross-file audit

The six `study.tsv` row IDs match the selected manifest in this order, each once with `status=pass` and `support_tier=hosted_selected`:

1. `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`
2. `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1`
3. `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1`
4. `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1`
5. `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1`
6. `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1`

All study rows and `manifest.tsv` report source commit `07b7f831ffee04f90a25f6cd11182b141eb38f07`, branch `sprint-224`, clean worktree, `windows-amd64`, compiler `cmake-probe:Visual Studio 17 2022:Release`, generated time `2026-11-15T22:25:43+00:00`, exact QR target/fixture, `stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=hosted_selected`, source-controlled dense QR baseline identity and helper, Python runtime, and matching baseline/project commands. The summary repeats target, fixture, source, branch, worktree, platform, compiler and both commands; its narrow hosted scope and non-claims agree with the rows. Five Windows-style generated paths normalize to the selected QR directory.

Both observation files have four passing rows. Their status and numerical strings match the corresponding study values exactly. The required interpreter and source-controlled QR helper pass; optional NumPy and SciPy entries remain deferred. For the source-controlled incompatible 4×2 fixture, both solutions remain within `1e-10` of `[2, -1]`, recomputed residual norm is `sqrt(3)`, solution norm is `sqrt(5)`, and the maximum project–baseline solution-component difference is `2.2204460492503131e-16`, within the selected `1e-10` tolerance. No Cholesky result or external package parity is inferred.

The downloaded bytes were copied into a temporary build tree and checked at the tested local HEAD with `python3 scripts/normalize_report_index.py --build-root /tmp/sprint224-day12-audit-build --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`. It returned `normalize-report-index: freshness ok (17 rows)` with all six generated QR rows fresh. A separate byte/provenance/numerical audit passed.

## Staleness rule and disposition

This artifact proves only revision `07b7f831ffee04f90a25f6cd11182b141eb38f07`. Any later commit, workflow/generator change or evidence-note commit creates a different source head and requires another exact-head selected Windows QR run and six-file inspection before approval. The uncommitted Day 12 ledger itself is a post-evidence working-tree change; committing it later will make this run historical for the new head. Missing, stale, expired, contradictory or failed hosted evidence blocks selected Windows QR freshness approval.

No selected-job or downloaded-artifact failure was found. Day 13 must confirm artifact accessibility, rerun the affected local checks and compare the branch/PR head with the tested SHA before drafting a named-target disposition.
