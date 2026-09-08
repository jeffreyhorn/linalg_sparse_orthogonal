# Sprint 199 Working Notes

## Sprint Scope

Sprint 199 promotes the guarded selected Windows Cholesky comparison freshness
lane only if hosted evidence, selected target manifest metadata, workflow
guards, and documentation agree. The selected target is
`cholesky-spd-tridiag-5`.

## Item Checklist

| Item | Description | Owner artifacts | Status |
| --- | --- | --- | --- |
| 199.1 | Inspect hosted Windows artifacts for `cholesky-spd-tridiag-5`, row IDs, artifact paths, and workflow metadata. | Day 1 intake; Day 2 hosted evidence inventory; Day 3 evidence semantics; GitHub Actions run artifacts | Complete for evidence semantics: hosted Windows CI run `34269219871` proves one successful MSVC selected Cholesky comparison freshness lane for the exact target, artifact, expected rows, and commit, with retained non-claims and promotion blockers recorded. |
| 199.2 | Promote selected target metadata only if hosted evidence proves exact target and platform scope. | `tests/corpus/manifests/selected_report_targets.tsv`; Day 4 decision artifact | Re-deferred for Day 4: hosted evidence proves the exact Windows Cholesky lane, but the manifest remains `workflow_platforms=linux;macos` until path-normalization tests, generated support metadata, and generated non-claim wording are aligned. |
| 199.3 | Add or update tests for Windows path normalization, selected-target filtering, missing rows, and stale artifacts. | `scripts/normalize_report_index.py`; `tests/test_normalize_report_index.py`; Day 5-Day 7 artifacts | Complete for normalizer hardening: path separator/suffix tests, near-match rejection, wrong-target row-set diagnostics, stale Windows-path diagnostics, and selected-target CLI misuse coverage are in place; platform/compiler policy remains tied to future manifest promotion. |
| 199.4 | Align Windows workflow, PowerShell validation, artifact names, and selected freshness commands. | `.github/workflows/windows-ci.yml`; `scripts/validate_windows_powershell.py`; `tests/test_validate_windows_powershell.py`; Day 8-Day 10 artifacts | Complete for integrated gate ownership: Day 8 confirmed workflow alignment; Day 9 added direct PowerShell guard tests for generator target drift, selected artifact-name drift, and fail-closed upload behavior; Day 10 confirmed selected manifest, workflow, PowerShell, normalizer, and local comparison freshness gates agree. |
| 199.5 | Update README, INSTALL, corpus docs, and maintainer guide with promoted or re-deferred selected Windows claim. | `README.md`; `INSTALL.md`; `tests/corpus/README.md`; `docs/maintainer_guide.md`; `docs/planning/EPIC_18/PROJECT_PLAN.md`; Day 11-Day 12 artifacts | Complete for re-deferred disposition: README, INSTALL, corpus docs, maintainer guide, and Epic planning status now say Sprint 199 reviewed the exact hosted Windows Cholesky path and re-deferred selected freshness promotion until selected manifest, generated support tier, and generated non-claim wording are promoted together. |
| 199.6 | Run selected manifest, workflow, PowerShell, normalizer, freshness, docs, and applicable C quality gates. | Day 10 gate integration; Day 11 public docs; Day 12 maintainer/planning alignment; Day 13 validation artifact; Day 14 closeout artifact | Complete. Selected manifest, workflow, normalizer, external comparison, PowerShell test, docs, comparison freshness, package/static deferral, and whitespace gates passed. Local `make windows-powershell-validate` exits `2` when `pwsh` is unavailable, which is expected unavailable evidence. No `.c` or `.h` files changed, so the full C quality trio was not required. |

## Evidence Ledger

| Date / Day | Evidence | Result | Notes |
| --- | --- | --- | --- |
| Day 1 | Sprint 199 project-plan review | Complete | Items 199.1 through 199.6 mapped to owner artifacts and sprint days. |
| Day 1 | `tests/corpus/manifests/selected_report_targets.tsv` review | Current state identified | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` remains `workflow_platforms=linux;macos` and `support_tier=local_only`; Windows is not promoted in source metadata. |
| Day 1 | `.github/workflows/windows-ci.yml` review | Guarded workflow path exists | `selected-comparison-freshness` runs `cholesky-spd-tridiag-5`, target-specific freshness, and uploads `sprint190-windows-selected-comparison-cholesky`. |
| Day 1 | `gh run list --workflow windows-ci.yml --limit 5` | Latest runs found | Latest `master` run `34269219871` at `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` completed successfully on 2026-09-08. |
| Day 1 | `gh run view 34269219871` | Selected job found | `Windows selected Cholesky comparison freshness (MSVC)` completed successfully, alongside CMake install/downstream, reviewed CMake subset, and PowerShell validation jobs. |
| Day 1 | `gh run download 34269219871 --name sprint190-windows-selected-comparison-cholesky` | Exit `0` | Artifact contains `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. Row-level semantics are reserved for Day 2/Day 3. |
| Day 1 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest invariants pass for current unpromoted Windows metadata. |
| Day 1 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow contract and selected Cholesky path guard pass. |
| Day 1 | `python3 tests/test_validate_windows_powershell.py` | Passed | PowerShell validator tests pass, including fake PowerShell and local unavailable behavior. |
| Day 1 | `python3 tests/test_normalize_report_index.py` | Passed | Existing normalizer regression tests pass. |
| Day 1 | `python3 tests/test_run_external_comparison.py` | Passed | Existing external comparison generator tests pass. |
| Day 1 | `make windows-powershell-validate` | Exit `2` | Structural checks pass, but local `pwsh` is unavailable; this is environment residual evidence, not pass evidence. |
| Day 2 | `gh run view 34269219871` | Passed | Run `34269219871` is `Windows CI` on `master` at `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`; selected job `Windows selected Cholesky comparison freshness (MSVC)` succeeded from 2026-09-08T19:29:05Z to 2026-09-08T19:29:45Z. |
| Day 2 | `gh api repos/.../actions/runs/34269219871/artifacts` | Passed | Artifact `sprint190-windows-selected-comparison-cholesky` has ID `10073117703`, size `4592`, created/updated 2026-09-08T19:29:41Z, and is not expired. |
| Day 2 | Hosted artifact file inventory | Complete | Artifact contains exactly the expected six files: project observations, baseline observations, dependency status, study, summary, and manifest. |
| Day 2 | Hosted `study.tsv` row inventory | Complete | Six rows are present, all expected row IDs are present, no unexpected rows are present, every row has `status=pass`, `platform=windows-amd64`, `source_branch=master`, clean worktree, and source commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`. |
| Day 2 | Hosted path review | Gap identified | Hosted generated rows use backslash artifact path `build\comparison\cholesky_spd_tridiag_5\study.tsv`, while the selected manifest currently uses forward slashes. Day 5-Day 7 must harden path normalization before promotion. |
| Day 2 | Hosted support-tier review | Gap identified | Hosted rows still carry `support_tier=local_only` and non-claims including `no Windows report freshness`; Day 3-Day 4 must decide whether and how to promote manifest/support wording. |
| Day 3 | Hosted evidence semantics review | Complete | The hosted artifact proves selected Windows Cholesky comparison freshness only for `cholesky-spd-tridiag-5`, Windows MSVC/CMake, run `34269219871`, artifact `sprint190-windows-selected-comparison-cholesky`, and commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`. |
| Day 3 | Promotion threshold checklist | Complete | Promotion requires exact target identity, expected file set, expected row IDs, all-pass statuses, clean hosted provenance, target-specific freshness command, Windows path filtering coverage, reconciled support/non-claim metadata, and unchanged broad non-claims. |
| Day 3 | Evidence ambiguity classification | Complete | `support_tier=local_only`, summary text saying `no hosted CI proof`, and backslash artifact paths are blockers for direct manifest promotion; optional NumPy/SciPy deferred rows and PowerShell ownership are not pass evidence. |
| Day 4 | Manifest promotion decision | Re-deferred | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` stays `workflow_platforms=linux;macos` because Day 3 threshold criteria are not all met in source-controlled tests and metadata. |
| Day 4 | Manifest changed-surface review | Complete | No manifest edit was made; impacted future surfaces are `tests/corpus/manifests/selected_report_targets.tsv`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, generator summary/non-claim text, README, INSTALL, corpus docs, and maintainer guide. |
| Day 5 | Windows artifact path normalization tests | Passed | `tests/test_normalize_report_index.py` now covers forward-slash, backslash, mixed-separator, and absolute Windows suffix paths for selected Cholesky filtering. |
| Day 5 | Near-match selected artifact rejection tests | Passed | Added cases for similarly named Cholesky paths, suffix extensions, and absolute Windows near matches that must not match the selected artifact. |
| Day 5 | `python3 tests/test_normalize_report_index.py` | Passed | Focused normalizer regression suite passed after Day 5 test additions. |
| Day 6 | Wrong-target selected comparison diagnostics | Passed | `normalize_report_index.py` now emits a target-specific `comparison_selected_rows` row-set mismatch when generated comparison rows exist but none match the requested selected artifact. |
| Day 6 | Missing/stale selected Cholesky diagnostics review | Complete | Existing selected Cholesky tests cover missing generated family, stale rows, failed rows, dependency-only row gaps, and Windows backslash stale rows with target-specific remediation. |
| Day 6 | `python3 tests/test_normalize_report_index.py` | Passed | Focused normalizer regression suite passed after the wrong-target diagnostic change. |
| Day 7 | Selected-target CLI hardening | Passed | Added a regression test that unknown selected comparison target keys fail clearly through `selected_report_targets.tsv` validation and remediation. |
| Day 7 | `python3 tests/test_normalize_report_index.py` | Passed | Focused normalizer suite passed with Day 5-Day 7 path, wrong-target, stale, and CLI coverage. |
| Day 7 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest invariants still pass with Windows unpromoted. |
| Day 7 | `make report-index-comparison-freshness` | Passed | Selected local comparison outputs regenerated and freshness check passed with 46 rows; generated `build/` outputs remain local artifacts. |
| Day 8 | Windows workflow alignment review | Complete | `.github/workflows/windows-ci.yml` already uses the exact selected target, MSVC CMake generator/arch/config, static library path, target-specific freshness command, selected artifact name, and six selected upload paths. |
| Day 8 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow contract confirms selected Windows Cholesky target, freshness command, artifact name, and bounded upload paths. |
| Day 8 | `python3 tests/test_validate_windows_powershell.py` | Passed | PowerShell ownership and claim-boundary tests pass with the manifest still unpromoted for Windows. |
| Day 8 | `python3 tests/test_normalize_report_index.py` | Passed | Normalizer tests still pass after workflow alignment review. |
| Day 9 | PowerShell guard ownership tests | Passed | Added tests proving generator target drift, selected artifact-name drift, and missing `if-no-files-found: error` fail clearly. |
| Day 9 | Retained non-claim guard review | Complete | Existing claim-boundary markers still preserve Windows Makefile, `pkg-config`, broad report freshness, package-manager, shared-library, dynamic ABI, and local unavailable PowerShell non-claims. |
| Day 9 | `python3 tests/test_validate_windows_powershell.py` | Passed | Focused PowerShell ownership test suite passed after Day 9 guard additions. |
| Day 9 | `make windows-powershell-validate` | Exit `2` | Structural checks passed; local `pwsh` is unavailable, so the wrapper records unavailable local PowerShell as non-pass evidence. |
| Day 10 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest invariants still agree with the Day 4 Windows re-deferral. |
| Day 10 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Windows workflow contract remains bounded to `cholesky-spd-tridiag-5`, selected artifact upload paths, and target-specific freshness command. |
| Day 10 | `python3 tests/test_validate_windows_powershell.py` | Passed | PowerShell ownership and claim-boundary tests pass, including local unavailable and hosted-required guard behavior. |
| Day 10 | `python3 tests/test_normalize_report_index.py` | Passed | Normalizer selected-target, Windows path, stale/missing row, wrong-target, and CLI misuse regression tests pass. |
| Day 10 | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Passed | Selected Cholesky rows are fresh to current `HEAD`; output remains advisory and selected-target scoped. |
| Day 10 | `make report-index-comparison-freshness` | Passed | Regenerated selected local comparison outputs and completed with `passed (local-only generated comparison freshness)`. |
| Day 11 | README public report-index wording | Updated | Public wording now records Sprint 199 reviewed hosted CI evidence for the exact Windows Cholesky path and kept Windows promotion re-deferred on selected metadata and generated semantics. |
| Day 11 | INSTALL support readiness matrix and platform row | Updated | User-facing Windows selected Cholesky status now says reviewed-hosted-path-but-re-deferred metadata status. |
| Day 11 | `tests/corpus/README.md` selected target interpretation | Updated | Corpus docs now keep the selected manifest as positive authority and record that the source manifest still does not list `windows`. |
| Day 11 | `make docs-check` | Passed | Doxygen generation and API docs coverage completed after public wording edits. |
| Day 11 | `python3 tests/test_validate_windows_powershell.py` | Passed | Initial exact-marker failures in README/corpus wording were corrected; final run passed claim-boundary and selected workflow guard coverage. |
| Day 11 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest invariants still pass with Windows unpromoted. |
| Day 11 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Selected Windows Cholesky workflow contract still passes after docs edits. |
| Day 11 | `git diff --check` | Passed | No whitespace errors after Day 11 edits. |
| Day 12 | `docs/maintainer_guide.md` selected comparison guidance | Updated | Maintainer guidance now records Sprint 199 reviewed the exact hosted Windows Cholesky path and re-deferred selected Windows freshness promotion on selected metadata and generated-claim blockers. |
| Day 12 | `docs/planning/EPIC_18/PROJECT_PLAN.md` status row | Updated | Sprint 199 is no longer listed as pending future execution; it is in progress with Windows promotion re-deferred and cites Day 1-Day 12 artifacts. |
| Day 12 | Residual queue | Recorded | Broad Windows report freshness, QR incompatible Windows comparison freshness, Windows oracle freshness, Windows benchmark freshness, selected metadata promotion, and unavailable local PowerShell evidence remain owner-scoped residuals. |
| Day 12 | `make docs-check` | Passed | Doxygen generation and API docs coverage completed after maintainer/planning edits. |
| Day 12 | `python3 tests/test_validate_windows_powershell.py` | Passed | Claim-boundary and selected workflow guard coverage still passes after maintainer guidance updates. |
| Day 12 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest invariants still pass with Windows unpromoted. |
| Day 12 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Selected Windows Cholesky workflow contract still passes after maintainer/planning edits. |
| Day 12 | `git diff --check` | Passed | No whitespace errors after Day 12 edits. |
| Day 13 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest invariants still agree with Windows re-deferral. |
| Day 13 | `python3 tests/test_normalize_report_index.py` | Passed | Selected-target filtering, Windows path normalization, missing/stale diagnostics, and CLI misuse coverage pass. |
| Day 13 | `python3 tests/test_run_external_comparison.py` | Passed | External comparison runner tests pass. |
| Day 13 | `python3 tests/test_validate_windows_powershell.py` | Passed | Workflow markers, claim boundaries, fake PowerShell pass paths, local unavailable behavior, and hosted-required fail-closed behavior pass. |
| Day 13 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Windows selected Cholesky workflow remains target/artifact/path scoped. |
| Day 13 | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Passed | Six selected Cholesky generated rows were fresh to current `HEAD`. |
| Day 13 | `make report-index-comparison-freshness` | Passed | Regenerated selected local comparison outputs and passed with local-only freshness wording. |
| Day 13 | `make docs-check` | Passed | Doxygen generation and API docs coverage completed. |
| Day 13 | `make windows-powershell-validate` | Exit `2` | Structural checks passed; local `pwsh` unavailable remains unavailable local evidence, not pass evidence. |
| Day 13 | `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager public non-claims and selected Homebrew local proof boundary remain intact. |
| Day 13 | `bash scripts/static_package_deferral_check.sh` | Passed | Static package deferral and Windows package non-claim wording remain intact. |
| Day 13 | Generated artifact scan | Complete | `build/` and `docs/api/` are ignored generated trees; untracked source candidates are only Sprint 199 planning files/artifacts. |
| Day 13 | `git diff --check` | Passed | No whitespace errors after integrated validation. |
| Day 14 | Sprint 199 artifact consistency scan | Complete | Day 1-Day 14 artifacts consistently support reviewed hosted Windows Cholesky evidence with selected Windows freshness promotion re-deferred. |
| Day 14 | Final item status | Complete | Items 199.1 through 199.6 are finalized in the closeout artifact. |
| Day 14 | Generated artifact review | Complete | `build/` and `docs/api/` remain ignored generated outputs and are not source-controlled staging candidates. |
| Day 14 | Retained non-claim scan | Complete | No broad Windows freshness, Windows oracle/benchmark freshness, QR incompatible Windows freshness, package/ABI, performance, release, or state-of-the-art claim is introduced. |
| Day 14 | `python3 tests/test_validate_windows_powershell.py` | Passed | Claim-boundary markers, selected workflow guards, and PowerShell availability semantics pass after closeout edits. |
| Day 14 | `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest invariants still agree with Windows re-deferral. |
| Day 14 | `python3 tests/test_selected_comparison_workflow.py` | Passed | Windows selected Cholesky workflow contract remains scoped. |
| Day 14 | `make docs-check` | Passed | Doxygen generation and API docs coverage completed after closeout edits. |
| Day 14 | `git diff --check` | Passed | No whitespace errors after closeout edits. |

## Owner Surface Inventory

| Surface | Owner files | Day 1 state |
| --- | --- | --- |
| Selected target manifest | `tests/corpus/manifests/selected_report_targets.tsv` | Cholesky target is selected for Linux/macOS only; Windows metadata promotion remains pending. |
| Corpus docs | `tests/corpus/README.md` | Own selected report target interpretation and retained non-claims. |
| Windows workflow | `.github/workflows/windows-ci.yml` | Contains one bounded `selected-comparison-freshness` job for `cholesky-spd-tridiag-5`. |
| PowerShell validator | `scripts/validate_windows_powershell.py`; `tests/test_validate_windows_powershell.py` | Owns workflow snippet parsing, selected workflow markers, manifest reference checks, and claim-boundary anchors. |
| Comparison generator | `scripts/run_external_comparison.py`; `tests/test_run_external_comparison.py` | Owns `cholesky-spd-tridiag-5` generation and CMake probe behavior. |
| Report normalizer | `scripts/normalize_report_index.py`; `tests/test_normalize_report_index.py` | Owns selected-target freshness filtering, missing/stale diagnostics, and generated row normalization. |
| Local comparison freshness gate | `Makefile` target `report-index-comparison-freshness` | Regenerates selected local comparison outputs and checks all manifest-selected comparison rows. |
| Public support docs | `README.md`; `INSTALL.md` | Describe Sprint 199 reviewed hosted evidence for the guarded Windows selected Cholesky workflow path and re-deferred selected freshness promotion. |
| Maintainer guidance | `docs/maintainer_guide.md` | States the Windows path remains guarded workflow evidence until selected metadata, generated support tier, generated non-claim wording, and claim contract are promoted together. |
| Prior sprint evidence | `docs/planning/EPIC_17/SPRINT_190/*`; `docs/planning/EPIC_17/SPRINT_191/*`; `docs/planning/EPIC_18/SPRINT_197/*` | Sprint 190 created the bounded workflow path; Sprint 191 kept selected comparison scope narrow; Sprint 197 selected Windows freshness promotion criteria for Epic 18. |

## Validation Matrix

| Validation | Trigger | Day 1 baseline |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Manifest metadata edits | Passed on Day 13. |
| `python3 tests/test_selected_comparison_workflow.py` | Windows workflow selected comparison edits | Passed on Day 13. |
| `python3 tests/test_validate_windows_powershell.py` | Windows workflow, PowerShell, selected manifest, or claim-boundary edits | Passed on Day 13. |
| `make windows-powershell-validate` | Manual PowerShell ownership check | Exit `2` locally because `pwsh` is unavailable after structural checks pass; hosted `--require-pwsh` owns pass evidence. |
| `python3 tests/test_normalize_report_index.py` | Normalizer filtering, path, missing-row, or stale-diagnostic edits | Passed on Day 13. |
| `python3 tests/test_run_external_comparison.py` | External comparison generator or CMake probe edits | Passed on Day 13. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Selected Cholesky report-index freshness edits | Passed on Day 13. |
| `make report-index-comparison-freshness` | Selected comparison generated-output or manifest freshness edits | Passed on Day 13. |
| `make docs-check` | Public or maintainer documentation edits | Passed on Day 13. |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager wording or proof-boundary edits | Passed on Day 13. |
| `bash scripts/static_package_deferral_check.sh` | Static package, shared-library, ABI, or Windows package non-claim wording | Passed on Day 13. |
| `make format && make lint && make test` | Any `.c` or `.h` change | Not required for Sprint 199 closeout because no `.c` or `.h` files are modified. |

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Hosted workflow success is treated as manifest promotion without row review. | Windows selected freshness could be overclaimed from job status alone. | Day 2 and Day 3 must review artifact files, row IDs, paths, timestamps, and semantics before Day 4 manifest changes. |
| Windows path separators break selected artifact filtering. | Freshness diagnostics could drop valid Windows rows or skip stale evidence. | Days 5 through 7 add focused normalizer coverage for backslashes, suffix matching, missing rows, and stale artifacts. |
| PowerShell validation is confused with report freshness. | Workflow snippet parseability could be mistaken for generated-report evidence. | Keep PowerShell ownership separate from the selected comparison freshness job and record local `pwsh` absence as unavailable evidence. |
| Public docs promote broad Windows report freshness. | Users may infer unsupported oracle, benchmark, QR, or broad report freshness support. | Day 11 and Day 12 docs must retain explicit non-claims outside `cholesky-spd-tridiag-5`. |
| Generated artifacts are committed. | Source control gains transient report bundles. | Day 13 and Day 14 must scan generated `build/` and downloaded artifact outputs before staging. |

## Open Questions

1. Does the downloaded hosted artifact from run `34269219871` contain all six
   expected row IDs with current commit metadata? Day 2 answer: yes, all six
   expected row IDs are present for commit
   `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`.
2. Does the hosted artifact path match the selected manifest
   `artifact_pattern` exactly after Windows path normalization? Day 2 answer:
   row paths use backslashes and require explicit normalizer coverage before
   promotion.
3. Should `workflow_platforms` for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` become
   `linux;macos;windows`, or should Windows remain re-deferred with a narrower
   hosted-evidence note? Day 3 answer: promotion is semantically supportable
   only after path-filter coverage and support/non-claim metadata are aligned;
   otherwise re-defer with the exact blockers. Day 4 decision: re-defer; do
   not add `windows` yet.
4. Are additional normalizer tests needed for absolute Windows paths emitted
   by `run_external_comparison.py` under MSVC/CMake? Day 3 answer: yes,
   backslash, mixed-separator, absolute-path suffix, and near-match rejection
   tests are required before relying on selected artifact filtering. Day 5
   update: these path-shape tests now exist for selected Cholesky generated-row
   matching.
5. Which public wording is acceptable if the claim is promoted: selected
   Windows Cholesky comparison freshness only, or guarded hosted evidence only?
   Day 3 answer: only selected Windows Cholesky comparison freshness for
   `cholesky-spd-tridiag-5` is supportable; broad Windows report freshness,
   oracle, benchmark, package, performance, release, and state-of-the-art
   wording must remain non-claims.

## Retained Non-Claims

Sprint 199 must not claim broad Windows report freshness, Windows selected
oracle freshness, Windows selected benchmark freshness, QR incompatible
Windows comparison promotion, Linux/macOS promotion changes, general
package-manager support, Homebrew/core readiness, bottles, Linuxbrew support,
shared-library package support, dynamic ABI compatibility, runtime-loader
behavior, performance superiority, external-library parity, or
state-of-the-art status.

## Day Log

### Day 1: Promotion Intake and Evidence Map

- Created the Sprint 199 working-notes scaffold.
- Mapped project-plan items 199.1 through 199.6 to owner artifacts.
- Inventoried selected Windows Cholesky freshness owner surfaces across the
  selected target manifest, workflow, PowerShell validator, generator,
  normalizer, public docs, maintainer docs, and prior sprint artifacts.
- Confirmed the current selected target manifest still excludes Windows for
  `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
- Confirmed the Windows workflow contains the bounded
  `selected-comparison-freshness` job for `cholesky-spd-tridiag-5`.
- Identified the latest successful `master` Windows workflow run
  `34269219871` at commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`.
- Downloaded the selected Windows Cholesky artifact from that run and confirmed
  the six expected files are present.
- Ran deterministic local baseline checks and recorded the expected local
  PowerShell unavailable result.

### Day 2: Hosted Artifact Inventory

- Inspected Windows workflow run `34269219871` and recorded workflow name,
  event, branch, commit, job list, job timing, and success conclusion.
- Queried GitHub artifact metadata and recorded selected artifact ID
  `10073117703`, name, size, timestamps, and non-expired status.
- Inspected the downloaded selected Cholesky artifact file set and confirmed
  the six expected files are present.
- Parsed `study.tsv` and confirmed all six expected Cholesky row IDs are
  present, all rows pass, all rows are `windows-amd64`, and all rows reference
  the clean `master` commit
  `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`.
- Recorded two Day 2 promotion gaps: hosted rows still use backslash artifact
  paths that need explicit normalizer coverage, and generated support metadata
  remains `local_only` with `no Windows report freshness` non-claims.

### Day 3: Evidence Semantics

- Classified the hosted artifact as proof of one selected Windows MSVC/CMake
  Cholesky comparison freshness lane, not proof of broad Windows report
  freshness or any unselected target family.
- Confirmed the evidence semantics depend on the exact target key
  `cholesky-spd-tridiag-5`, expected row IDs, all-pass row statuses, selected
  workflow job, selected artifact name, clean `master` commit provenance, and
  target-specific freshness command.
- Identified three direct promotion blockers for Day 4 to resolve or re-defer:
  backslash artifact paths need normalizer coverage, generated rows still say
  `support_tier=local_only`, and generated summary/non-claim text still says
  `no hosted CI proof` and `no Windows report freshness`.
- Recorded that optional NumPy/SciPy dependency defers, PowerShell workflow
  validation, CMake install/downstream jobs, and reviewed CMake consumer jobs
  are context only for this claim and cannot be counted as selected freshness
  pass evidence.
- Defined the Day 4 threshold: promote only if the selected manifest, generated
  metadata, workflow evidence, target-specific filtering, and public claim
  wording can all agree on the same narrow Windows Cholesky scope.

### Day 4: Manifest Decision

- Applied the Day 3 promotion threshold to
  `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
- Kept `tests/corpus/manifests/selected_report_targets.tsv` unchanged:
  `workflow_platforms` remains `linux;macos` and the non-claim still includes
  `no Windows report freshness`.
- Recorded the decision as a re-deferral rather than a rejection of the hosted
  evidence. The hosted artifact is valid input, but the source-controlled
  normalizer/path tests and generated metadata are not yet aligned enough to
  make the manifest authoritative for Windows.
- Identified the exact promotion prerequisites for later sprint days:
  Windows separator and suffix filtering tests, missing/stale selected-row
  diagnostics, generated support-tier semantics, generated summary non-claim
  wording, and public/maintainer documentation wording.
- Confirmed no unreviewed target, platform, oracle lane, benchmark lane,
  package/ABI surface, performance claim, release claim, or broad Windows
  freshness surface was promoted.

### Day 5: Windows Path Normalization Tests

- Reviewed `selected_comparison_generated_rows()` and confirmed the current
  implementation normalizes backslashes to forward slashes before selected
  artifact comparison.
- Expanded `test_selected_comparison_generated_rows_match_windows_artifact_paths`
  to cover forward-slash relative paths, backslash relative paths,
  mixed-separator paths, and absolute Windows paths ending in the selected
  artifact pattern.
- Added
  `test_selected_comparison_generated_rows_reject_near_match_artifact_paths`
  so similarly named Cholesky artifacts, extension suffixes, and absolute
  Windows near matches cannot pass selected-target filtering.
- Added the new test to the direct `main()` runner used by
  `python3 tests/test_normalize_report_index.py`.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Reduced the Day 4 path-normalization promotion blocker for row filtering.
  Day 6 still owns missing-row, stale-artifact, wrong-target, and
  wrong-platform diagnostic expansion.

### Day 6: Freshness Diagnostics

- Reviewed selected comparison diagnostics for missing selected rows, stale
  rows, failed rows, dependency-only rows, duplicate rows, and wrong-target
  generated artifacts.
- Patched `selected_comparison_policy_diagnostics()` so a selected-target
  freshness check fails when generated comparison rows exist but no generated
  rows match the selected target artifact. This prevents a QR-only comparison
  artifact from satisfying `--selected-target cholesky-spd-tridiag-5`.
- Added
  `test_selected_comparison_target_freshness_rejects_wrong_target_rows`, which
  checks the new target-specific row-set mismatch, selected Cholesky target ID,
  `observed=0`, missing Cholesky row IDs, selected artifact diagnostic, and
  selected-target remediation command.
- Reconfirmed existing coverage for selected Cholesky missing generated
  family, stale rows, failed rows, dependency-only rows, and Windows backslash
  stale-row diagnostics.
- Ran `python3 tests/test_normalize_report_index.py`; it passed.
- Classified wrong platform/compiler validation as a remaining metadata-policy
  residual: the current selected manifest has not promoted Windows yet and
  does not provide an authoritative expected compiler tuple. Day 7 should
  avoid adding false platform policy unless the manifest claim surface is
  updated at the same time.

### Day 7: Normalizer Hardening

- Reviewed Day 5 and Day 6 normalizer changes against the Sprint 199
  completion criteria.
- Confirmed `selected_comparison_generated_rows()` normalizes Windows
  backslashes and preserves exact-or-directory-suffix matching semantics.
- Confirmed selected-target wrong-artifact rows now produce a target-specific
  row-set mismatch instead of silently passing as unrelated generated
  comparison evidence.
- Added `test_selected_target_unknown_key_fails_clearly` so unknown
  `--selected-target` values fail through selected manifest validation with
  the schema remediation command.
- Reconfirmed the existing guard that `--selected-target` cannot be supplied
  without `--check-freshness`.
- Ran `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_selected_report_targets_manifest.py`, and
  `make report-index-comparison-freshness`; all passed.
- Left platform/compiler rejection semantics out of the normalizer until the
  selected target manifest owns an explicit Windows platform/compiler contract.
  This keeps Day 7 scoped to selected comparison freshness behavior and avoids
  creating an implicit broad Windows policy.

### Day 8: Workflow Alignment

- Reviewed `.github/workflows/windows-ci.yml` selected comparison freshness
  job against the Day 4 re-deferral and Day 7 normalizer hardening.
- Confirmed the workflow remains bounded to `cholesky-spd-tridiag-5` and uses
  `Visual Studio 17 2022`, `x64`, `Release`, and
  `build/Release/sparse_lu_ortho.lib`.
- Confirmed the freshness command uses
  `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`.
- Confirmed the upload artifact remains
  `sprint190-windows-selected-comparison-cholesky` and uploads only the six
  selected Cholesky comparison files under
  `build/comparison/cholesky_spd_tridiag_5/`.
- Made no workflow edit because the current guarded workflow path already
  matches the re-deferred manifest state and does not imply promoted Windows
  selected freshness.
- Ran `python3 tests/test_selected_comparison_workflow.py`,
  `python3 tests/test_validate_windows_powershell.py`, and
  `python3 tests/test_normalize_report_index.py`; all passed.

### Day 9: PowerShell Guard

- Reviewed `scripts/validate_windows_powershell.py` selected Cholesky
  ownership constants and workflow-structure checks.
- Confirmed the validator already enforces the exact selected generator
  command, selected freshness command, upload artifact name, six upload paths,
  `if-no-files-found: error`, `windows-2022`, and `timeout-minutes: 20`.
- Added direct regression tests for generator target drift, artifact-name
  drift, and upload fail-open drift in `tests/test_validate_windows_powershell.py`.
- Reconfirmed existing tests for selected-target drift, broad upload paths,
  missing required upload files, forbidden selected report freshness outside
  the bounded lane, hosted validation wiring, PowerShell parse behavior, and
  retained claim-boundary wording.
- Ran `python3 tests/test_validate_windows_powershell.py`; it passed locally.
- Ran `make windows-powershell-validate`; it completed structural checks and
  exited `2` because local `pwsh` is unavailable. This remains unavailable
  local evidence, not pass evidence.
- Kept the guard aligned with Day 4 re-deferral: the Windows Cholesky workflow
  remains bounded guarded evidence while the selected manifest still has no
  Windows `workflow_platforms` entry.

### Day 10: Gate Integration

- Connected the selected target manifest, workflow contract, PowerShell guard,
  normalizer diagnostics, and local generated comparison freshness into one
  Day 10 gate record.
- Confirmed `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` remains intentionally
  re-deferred for Windows: `workflow_platforms=linux;macos`,
  `support_tier=local_only`, and `no Windows report freshness` are still the
  source-controlled claim boundary.
- Ran `python3 tests/test_selected_report_targets_manifest.py`,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 tests/test_validate_windows_powershell.py`, and
  `python3 tests/test_normalize_report_index.py`; all passed.
- Ran the narrow selected-target freshness command for
  `cholesky-spd-tridiag-5`; it passed and reported the six selected Cholesky
  generated rows as fresh to current `HEAD`.
- Ran `make report-index-comparison-freshness`; it regenerated selected local
  comparison outputs and passed with local-only freshness wording.
- Recorded pass behavior, guarded failure behavior, retained non-claims, and
  residuals in `artifacts/day10-gate-integration.md`.

### Day 11: Public Documentation Calibration

- Updated README report-index, QR/report freshness, and install handoff
  wording to reflect the Sprint 199 reviewed-but-re-deferred Windows Cholesky
  disposition.
- Updated INSTALL support readiness and supported-platform wording so the
  Windows selected Cholesky row no longer reads as pending hosted evidence
  review. It now records reviewed hosted path evidence with re-deferred
  metadata/generated-claim promotion.
- Updated `tests/corpus/README.md` so selected target documentation keeps the
  manifest as positive authority and states the manifest still does not list
  `windows` for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
- Preserved explicit non-claims for broad Windows report freshness, Windows
  oracle freshness, Windows benchmark freshness, unselected Windows comparison
  families, Windows Makefile parity, Windows `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, and broad Windows parity.
- Added `artifacts/day11-public-docs.md` with updated surfaces, retained
  non-claims, owner surfaces, and validation plan.
- Ran `make docs-check`; it passed.
- Ran `python3 tests/test_selected_report_targets_manifest.py` and
  `python3 tests/test_selected_comparison_workflow.py`; both passed.
- Ran `python3 tests/test_validate_windows_powershell.py`; after restoring
  exact guarded marker line breaks in README and corpus docs, the final run
  passed.
- Ran `git diff --check`; it passed.

### Day 12: Maintainer and Planning Alignment

- Updated `docs/maintainer_guide.md` selected comparison freshness guidance so
  the Windows `cholesky-spd-tridiag-5` path is described as reviewed hosted
  evidence with selected Windows freshness promotion re-deferred.
- Updated maintainer residual wording so `windows` stays out of selected target
  `workflow_platforms` until selected metadata, generated support tier,
  generated non-claim wording, and the claim contract are promoted together.
- Updated the repeated normalized report-index workflow guidance with the same
  reviewed-but-re-deferred Sprint 199 interpretation.
- Updated `docs/planning/EPIC_18/PROJECT_PLAN.md` so Sprint 199 is no longer
  listed as pending future execution. The row now cites Day 1-Day 12 artifacts
  and explicitly keeps selected Windows freshness re-deferred.
- Added `artifacts/day12-maintainer-planning-alignment.md` with maintainer
  guidance changes, planning status rationale, item status, residual queue,
  and validation plan.
- Ran `make docs-check`; it passed.
- Ran `python3 tests/test_validate_windows_powershell.py`,
  `python3 tests/test_selected_report_targets_manifest.py`, and
  `python3 tests/test_selected_comparison_workflow.py`; all passed.
- Ran `git diff --check`; it passed.
- Kept 199.6 open for Day 13 integrated validation and Day 14 closeout.

### Day 13: Integrated Validation

- Ran the full Sprint 199 focused Python validation set:
  `python3 tests/test_selected_report_targets_manifest.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_run_external_comparison.py`,
  `python3 tests/test_validate_windows_powershell.py`, and
  `python3 tests/test_selected_comparison_workflow.py`; all passed.
- Ran the narrow selected-target freshness command for
  `cholesky-spd-tridiag-5`; it passed and reported the six selected Cholesky
  rows fresh to current `HEAD`.
- Ran `make report-index-comparison-freshness`; it regenerated selected local
  comparison output and passed with local-only freshness wording.
- Ran `make docs-check`; it passed.
- Ran `make windows-powershell-validate`; it exited `2` after structural
  checks passed because local `pwsh` is unavailable. This remains unavailable
  local evidence, not pass evidence.
- Ran `bash scripts/package_manager_deferral_check.sh` and
  `bash scripts/static_package_deferral_check.sh`; both passed.
- Confirmed `git diff --name-only` contains no `.c` or `.h` changes, so
  `make format && make lint && make test` is not required for Day 13.
- Confirmed generated `build/` and `docs/api/` outputs are ignored and not
  source-controlled staging candidates.
- Added `artifacts/day13-integrated-validation.md` with command results,
  generated artifact review, quality-gate decision, and residuals.

### Day 14: Closeout Review

- Reviewed the full Sprint 199 artifact set from Day 1 through Day 14 for
  consistency with the Day 4 manifest re-deferral, Day 10 gate integration,
  Day 11 public docs, Day 12 maintainer/planning alignment, and Day 13
  validation results.
- Finalized items 199.1 through 199.6 as complete for the reviewed
  re-deferred disposition.
- Confirmed Sprint 199 closes with selected Windows Cholesky freshness
  promotion re-deferred, not rejected and not promoted.
- Confirmed retained non-claims still cover broad Windows report freshness,
  Windows oracle freshness, Windows benchmark freshness, QR incompatible
  Windows comparison freshness, unselected Windows comparison families,
  package-manager support, shared-library support, dynamic ABI support,
  performance superiority, release readiness, and state-of-the-art status.
- Confirmed generated `build/` and `docs/api/` outputs remain ignored and
  should not be staged.
- Added `artifacts/day14-closeout-review.md` with final disposition, item
  status, source-controlled evidence set, retained non-claims, generated
  artifact review, retrospective inputs, and validation reference.
- Ran `python3 tests/test_validate_windows_powershell.py`,
  `python3 tests/test_selected_report_targets_manifest.py`, and
  `python3 tests/test_selected_comparison_workflow.py`; all passed.
- Ran `make docs-check`; it passed.
- Ran `git diff --check`; it passed.
