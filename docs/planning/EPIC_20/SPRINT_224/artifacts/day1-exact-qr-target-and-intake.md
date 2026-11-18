# Sprint 224 Day 1: Exact QR Target and Hosted Intake Contract

## Intake decision

**Selected Windows QR incompatible freshness remains blocked.** Day 1 freezes the source-controlled target and the hosted proof required to change it. No QR hosted run or ZIP was downloaded or inspected for this sprint, and the existing Cholesky artifact is not QR evidence. The current selected manifest remains Linux/macOS-only with `local_only` support. The Sprint 209 collection lane is an evidence producer, not a promotion decision.

At intake, local `master`, `origin/master`, and `sprint-224` all point to `44e3b84dab97a10b2727cc026e7124a9a5415421` (the PR #247 merge). The local working tree contains untracked Sprint 224 planning files. A later hosted proof must identify the actual tested QR commit and report a clean hosted worktree; this local planning state is not a generated result. Sprint 208's historical `master`-branch criterion must be adapted to the event and branch of the QR run rather than copied literally onto a pull-request run.

## Frozen source-controlled target

| Field | Day 1 value |
| --- | --- |
| Selected manifest ID | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Family / subfamily / key | `comparison` / `qr_incompatible_ls` / `qr-incompatible-ls` |
| Fixture and operation | `qr_overdetermined_incompatible_4x2`; incompatible least-squares solve |
| Generator command in manifest | `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` |
| Study path | `build/comparison/qr_incompatible_ls/study.tsv` |
| Required files / expected rows | Six files / six rows |
| Selection and support | `reviewed_cross_platform_selected`; `local_only`; `generated_compare_inputs` freshness policy |
| Selected workflow metadata | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml`; jobs `generated-report-freshness;selected-comparison-freshness`; artifacts `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness`; platforms `linux;macos` |
| Current claim | Freshness for the named fixture against the selected source-controlled dense reference helper, with `no Windows report freshness` retained |
| Owner | Report maintainer |

The workflow upload is restricted to these six paths under `build/comparison/qr_incompatible_ls/`, in this order:

1. `project_observations.tsv`
2. `baseline_observations.tsv`
3. `dependency_status.tsv`
4. `study.tsv`
5. `summary.md`
6. `manifest.tsv`

The manifest owns exactly these `study.tsv` row IDs, in this order:

1. `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`
2. `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1`
3. `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1`
4. `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1`
5. `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1`
6. `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1`

The generator's QR fixture is 4×2 with RHS `[1, -2, 2, 5]`, expected solution `[2, -1]`, expected solution norm about `2.2360679774997898`, and deliberately nonzero expected residual norm about `1.7320508075688772`. These are fixture expectations, not an observed Windows result. The selected row IDs, statuses, values and tolerances must be read from the hosted files before making any freshness claim.

## Hosted Windows path and known candidate gaps

The `Windows CI` workflow runs on pushes to `main`/`master` and pull requests targeting them. The job key is `selected-qr-incompatible-comparison-freshness`, displayed as `Windows selected QR incompatible comparison freshness (MSVC)`. It uses `windows-2022` and a 20-minute timeout. Its source-controlled sequence is:

1. `actions/checkout@v4`.
2. `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`.
3. `cmake --build build --config Release --target sparse_lu_ortho`.
4. `python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib`.
5. `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls`.
6. Upload `sprint209-windows-selected-comparison-qr-incompatible` with `actions/upload-artifact@v4`, `if-no-files-found: error`, and the six paths above.

**Candidate gaps for Days 2 and 6–10:** Unlike the selected Cholesky job, this QR job has no explicit PR-head checkout reference or `SPARSE_SELECTED_SOURCE_SHA` / `SPARSE_SELECTED_SOURCE_BRANCH` environment. A pull-request artifact may therefore describe a merge checkout rather than the PR source head; inspect the actual job and report before deciding. `scripts/run_external_comparison.py` currently applies `hosted_selected` only to the selected Cholesky context and otherwise emits `local_only`; the QR report's `no hosted CI proof` and `no Windows report freshness` wording is incompatible with promotion until a narrow, guarded QR-hosted path is implemented and reviewed. These are source-contract gaps, not claims that a hosted QR job failed. The target's expected Windows platform/compiler, if proved by the downloaded rows, are `windows-amd64` and `cmake-probe:Visual Studio 17 2022:Release`.

## Promotion gates adapted to QR

| Gate | Required proof before a Windows QR claim |
| ---: | --- |
| 1 | Current relevant hosted run ID, URL, event, source head, timestamps and success. |
| 2 | Exact selected QR MSVC job succeeded on the intended source revision. |
| 3 | Named QR artifact has ID, size, expiry and API digest; it is unexpired and downloadable. |
| 4 | ZIP digest matches the API digest and contains exactly the six required files. |
| 5 | `study.tsv` has exactly the six manifest-owned QR row IDs. |
| 6 | Every row passes and has coherent commit, branch, clean worktree, observation values and tolerances. |
| 7 | Study, manifest and summary establish Windows AMD64, VS 2022 Release, baseline identity and generator provenance. |
| 8 | Path/row guards reject near matches, stale/wrong/duplicate/extra rows and wrong files while accepting Windows path forms. |
| 9 | Only the QR manifest row gains platform-aligned Windows workflow, job and artifact fields. |
| 10 | Manifest support tier, claim scope and non-claims agree with hosted QR data. |
| 11 | Hosted QR rows and summary have no contradictory `local_only`, `no hosted CI proof` or `no Windows report freshness` markers. |
| 12 | README, INSTALL, corpus/schema, maintainer and planning surfaces use one bounded QR sentence. |
| 13 | Other QR/Cholesky targets, broad Windows solver or report freshness, package/ABI, timing and state-of-the-art claims remain excluded. |

## Day 2 intake and ownership

| Owner | Next evidence/action |
| --- | --- |
| Windows/report owner | Obtain a QR run tied to a known commit; record event, source/checkout SHA, branch, selected job conclusion, artifact ID/URL/size/expiry/digest and logs. A branch-only push cannot trigger this workflow; a pull request or qualifying base-branch run is needed. |
| Artifact/report owner | Download ZIP, verify digest and six files, then trace study rows to observations, manifest and summary on Days 3–5. |
| Workflow/generator owner | Resolve any PR merge-checkout and QR `local_only` mismatch only after hosted intake; preserve exact-target guard coverage. |
| Manifest/docs owner | Keep Windows QR unselected until every promotion gate passes; align only the named target on Days 6–8 if eligible. |
| Guard/review owner | Add negative fixtures on Days 8–10 and recheck the final proposed head on Days 11–14. |

**Failure log:** Day 1 source inspection completed without a failed check. Hosted run, job, artifact and numerical results are **not yet inspected**. If the QR artifact is absent, inaccessible, stale or inconsistent, record the blocker and re-defer instead of using the Cholesky ZIP or local QR run.

## Day 1 source inventory

| Source file | SHA-256 at intake |
| --- | --- |
| `.github/workflows/windows-ci.yml` | `eae2720c95f3458b5939bbe0de10bf68510ea1e643e3a3b2ccbf217847963f5d` |
| `tests/corpus/manifests/selected_report_targets.tsv` | `2a03981b013aaff8a8aabeebf06cc3a45796918ae63b96c589b1597121d69109` |
| `scripts/run_external_comparison.py` | `418b26fa07eed9dde7367dd0699fd00503a9f6ec0852871ec4d05d40d84dae4b` |
| `scripts/normalize_report_index.py` | `2cda24177d2cfdfb8cdef89f391476b743492eff230bc722ffd514c9c65e558d` |
| `scripts/validate_windows_powershell.py` | `bbee4fc879dc6e7db30d17f52cede01521269cea9e49ad2630efe8bcd0394637` |
| Sprint 208 Day 4 criteria | `63dcb467b98e34dd28650cfa688a3d868d79b7abc36bee2f3ddd94b8de9a4c43` |
| Sprint 209 retrospective | `d2b95e6af08fbdc99bfa67429c7fcf13f05bdb9fc218939aabe447569b1ef338` |
| Sprint 223 retrospective | `1c581598c5eedafc75780696a81991c47d7f8ef5cc0956f5482933eec4e1a9fa` |

Day 1 performed source and workflow inspection. It did not modify the selected manifest, run MSVC, inspect hosted QR evidence, or authorize Windows QR promotion.
