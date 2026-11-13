# Sprint 223 Day 1: Exact Target and Hosted Intake Contract

## Decision at intake

**Selected Windows Cholesky freshness is blocked pending current hosted evidence.** Day 1 inventories the target and workflow. It does not inspect a hosted run, download an artifact or change selected metadata. The Sprint 222 local CHOLMOD pilot is a different comparison and supplies no Windows evidence.

The source revision at intake is `7fc86594cb45477b58cd29706c5f87e005ca99c1` for local `master`, `origin/master` and `sprint-223` HEAD. The branch has untracked Sprint 223 planning files, so this working tree is not a clean hosted report input. A future hosted row must match the **actual** run head SHA and branch; the Sprint 208 `source_branch=master` criterion describes its historical master run and must not be copied blindly onto a pull-request run. Day 2 must capture the run event and branch, then apply the corresponding freshness policy.

## Frozen target and current source-controlled state

| Field | Day 1 value |
| --- | --- |
| Selected manifest ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Family/subfamily/key | `comparison` / `cholesky_spd_tridiag_5` / `cholesky-spd-tridiag-5` |
| Meaning | Freshness of one SPD tridiagonal Cholesky comparison against the selected source-controlled dense Cholesky reference helper |
| Generator command in manifest | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| Study artifact pattern | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Required file count and expected row count | Six files; six rows |
| Current support/freshness | `local_only` / `generated_compare_inputs` |
| Current workflow metadata | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml`, jobs `generated-report-freshness;selected-comparison-freshness`, artifacts `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness`, platforms `linux;macos` |
| Current non-claim | Includes `no Windows report freshness`; Windows is not selected in this manifest row |
| Owner | Report maintainer |

The exact uploaded files, all under `build/comparison/cholesky_spd_tridiag_5/`, are:

1. `project_observations.tsv`
2. `baseline_observations.tsv`
3. `dependency_status.tsv`
4. `study.tsv`
5. `summary.md`
6. `manifest.tsv`

The six manifest-owned `study.tsv` row IDs are:

1. `comparison_cholesky_spd_tridiag_5_project_status_v1`
2. `comparison_cholesky_spd_tridiag_5_baseline_status_v1`
3. `comparison_cholesky_spd_tridiag_5_residual_norm_v1`
4. `comparison_cholesky_spd_tridiag_5_solution_norm_v1`
5. `comparison_cholesky_spd_tridiag_5_solution_values_v1`
6. `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1`

## Hosted workflow path to inspect

The relevant workflow is [Windows CI](../../../../../.github/workflows/windows-ci.yml), triggered by `push` and `pull_request` targeting `main` or `master`. The selected job key is `selected-comparison-freshness`; its displayed name is `Windows selected Cholesky comparison freshness (MSVC)`. It runs on `windows-2022` with a 20-minute job timeout. Its exact sequence is:

1. Check out the run's head commit.
2. Configure `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` in PowerShell.
3. Build `sparse_lu_ortho` in Release with `cmake --build build --config Release --target sparse_lu_ortho`.
4. Run `scripts/run_external_comparison.py` for `cholesky-spd-tridiag-5` using `--probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib`.
5. Run `scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`.
6. Upload `sprint190-windows-selected-comparison-cholesky` through `actions/upload-artifact@v4` with `if-no-files-found: error` and the six named paths above.

The expected generated platform and compiler fields from Sprint 208 are `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release`. They must be verified in the downloaded rows, not inferred from the YAML. The selected workflow is a hosted evidence producer, while the selected manifest row still excludes Windows. The generator has `local_only` support-tier output at `scripts/run_external_comparison.py:1654`, so the eventual generated wording requires direct inspection before promotion. The Sprint 220 structured API-docs workflow guard remains a separate local-only publication boundary.

## Thirteen promotion gates to carry forward

| Gate | Required Day 2-14 proof |
| ---: | --- |
| 1 | Current relevant hosted run ID, URL, event, head SHA, timestamps and success. |
| 2 | Exact selected MSVC job succeeded. |
| 3 | Named artifact is present, unexpired, downloadable and identified by ID, size, expiry and digest. |
| 4 | Archive contains exactly the six required files. |
| 5 | Study has exactly the six selected row IDs. |
| 6 | Every selected row passes and has coherent source commit, branch and clean worktree. |
| 7 | Rows record the bounded Windows platform and Visual Studio 2022 Release compiler. |
| 8 | Normalizer rejects near-match paths, missing/duplicate/extra/wrong/stale rows and artifact mismatches while accepting Windows path separators. |
| 9 | Selected manifest Windows workflow/job/artifact/platform entries align by position. |
| 10 | Selected manifest support tier, claim scope and non-claims agree with hosted evidence. |
| 11 | Generated rows and summary have no contradictory `local_only`, `no hosted CI proof` or `no Windows report freshness` claim for this exact lane. |
| 12 | README, INSTALL, corpus/schema, maintainer and planning wording agree. |
| 13 | Broad Windows, package/ABI, timing, external parity and state-of-the-art claims remain excluded. |

These are the [Sprint 208 criteria](../../../EPIC_19/SPRINT_208/artifacts/day4-promotion-criteria.md), refined to require exact run-head alignment on the final Sprint 223 branch commit as called for by the [Sprint 222 handoff](../../SPRINT_222/artifacts/day14-sprint223-handoff.md). The [Sprint 216 RQ-04 residual](../../../EPIC_19/SPRINT_216/RESIDUAL_QUEUE.md) additionally requires six current files/rows and aligned generator/non-claim semantics. A failed gate leaves selected Windows metadata absent.

## Day 2 hosted intake checklist

- Identify an exact-commit run for the branch and record run ID, URL, event, head SHA, branch, creation/completion times and conclusion.
- Identify the `selected-comparison-freshness` job ID, displayed name, conclusion and log URL; confirm the observed commands.
- Identify artifact ID, name, URL, size, digest, expiry and download result; retain the downloaded six-file archive in a temporary audit directory.
- Hash each file and compare archive membership, row IDs, row status, source SHA/branch, clean-worktree field, platform and compiler to this contract.
- Keep promotion blocked if the current run or artifact is missing, wrong, stale or inaccessible. Record the failure without substituting Sprint 208's historical run.

## Day 1 source inventory

| File | SHA-256 |
| --- | --- |
| `.github/workflows/windows-ci.yml` | `6ba5f17054e9cc852a481f541ce285739269c57e0a263aa7ce71b4d1af9e5a84` |
| `tests/corpus/manifests/selected_report_targets.tsv` | `58ad01ca9c2873a9d17b3204671c37589339a395029eaaec5bcc285710fe1b33` |
| Sprint 208 Day 4 criteria | `63dcb467b98e34dd28650cfa688a3d868d79b7abc36bee2f3ddd94b8de9a4c43` |
| Sprint 216 residual queue | `45c51ded2d71a1d035189bfe1312f00977a0319a7776726973e4a2e53b4f1f2c` |
| Sprint 222 Day 14 handoff | `a0e79cbbce7a013e1391229c30fdd49d9a6641140c5fd032261c741ab6f633e8` |

Day 1 performed read-only source and workflow inspection plus this ledger. It did not run MSVC, download a hosted artifact, validate generated rows, or change the selected manifest.
