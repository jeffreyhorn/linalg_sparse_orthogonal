# Sprint 223 Day 7: Documentation and Workflow Guard Alignment

## Candidate status

Day 5 authorized preparation of one bounded candidate, and Day 6 aligned its selected manifest and generator. Day 7 carries that contract through the public and maintainer documentation. **The claim is proposed, not accepted** until a successful selected Windows job and its exact six-file artifact are audited for the final Sprint 223 commit. The earlier `master` artifact is intake evidence for the workflow path, not final-commit proof.

Exact proposed claim, shared by the selected manifest and all five documentation surfaces:

> Selected Cholesky SPD tridiagonal solve comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense Cholesky reference helper.

The claim covers the selected fixture and source-controlled dense helper only. The candidate manifest lists `.github/workflows/windows-ci.yml`, job `selected-comparison-freshness`, artifact `sprint190-windows-selected-comparison-cholesky` and platform `windows` after the existing Linux/macOS entries. Its `hosted_selected` tier describes the proposed selected hosted contract; a local generator invocation still produces `local_only` evidence.

## Claim and non-claim matrix

| Surface or topic | Day 7 interpretation | Gate |
| --- | --- | --- |
| README report interpretation | Quotes the exact candidate sentence and says final-commit hosted proof is pending. | Windows claim-boundary guard checks exact sentence and retained exclusions. |
| INSTALL support-readiness matrix | Labels the Windows Cholesky row **candidate**; gives the same proposed sentence and pending artifact condition. | Support-docs guard and Windows claim-boundary guard. |
| Corpus README and report-index schema | Describe one selected Cholesky manifest row with aligned Windows workflow/job/artifact/platform metadata and the same candidate sentence. | Corpus schema, selected-target and claim-boundary guards. |
| Maintainer guide and planning notes | Describe the same candidate, retain historical Sprint 182/208 deferral as history, require final-commit artifact review before acceptance. | Windows claim-boundary guard and this sprint's Day 5-7 ledgers. |
| Windows QR incompatible comparison | Evidence-collection workflow only; its selected manifest row remains Linux/macOS-only and `local_only`. | Selected-target and Windows workflow guards. |
| Other Windows report families | No selected Windows oracle, benchmark or unselected comparison freshness; no broad report-index freshness. | Selected-target allowlist and bounded workflow upload checks. |
| Numerical and ecosystem claims | No broad Cholesky or SPD correctness, CSC parity, external-library parity, timing/performance superiority or state-of-the-art status. | Manifest and generated non-claims; docs claim boundaries. |
| Package and ABI | No package-manager or shared-library ABI proof from this report. | INSTALL matrix and manifest non-claims. |
| Generated API HTML | Remains ignored local output under `docs/api/`, with no CI upload or publication. | `make api-docs-local-only` and structured workflow policy. |

## Workflow and guard inspection

The selected Windows job still configures/builds the MSVC Release library, runs only `cholesky-spd-tridiag-5`, invokes the selected report-index normalizer, and uploads exactly these six named paths under `build/comparison/cholesky_spd_tridiag_5/`: `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. Its upload step retains `if-no-files-found: error`. No Windows workflow command, destination, or upload path changed on Day 7.

The Windows PowerShell validator now requires the exact proposed sentence on README, INSTALL, corpus README, report-index schema and maintainer guide after whitespace normalization. A mutation of that sentence fails with a specific diagnostic. Existing broad-promotion, QR-promotion and wrong-target rejection tests remain active. The Sprint 220 generated-API workflow policy still checks 111 job/step records; it reports no generated HTML publication route.

## Verification and remaining gates

Day 7 checks: `python3 tests/test_validate_windows_powershell.py`, `python3 tests/test_selected_report_targets_manifest.py`, `python3 tests/test_selected_comparison_workflow.py`, `make support-docs-guard`, `make docs-check`, `make api-docs-local-only API_DOCS_PYTHON=/tmp/s223-day7-api-venv/bin/python`, and `git diff --check`. The API guard uses a temporary Python environment with the pinned PyYAML dependency; generated Doxygen HTML stays ignored. Local `pwsh` is unavailable, so the native PowerShell parse proof remains assigned to hosted Windows CI.

Days 8-10 still own adversarial artifact/metadata rejection cases. Days 11-12 still own the exact-final-commit hosted job and downloadable six-file artifact audit. If those gates fail, the candidate must be re-deferred and the candidate wording revisited; the Day 2 `master` artifact cannot substitute.
