# Sprint 224 Day 6: QR manifest and generator candidate

## Decision and evidence boundary

Day 5 permitted a **candidate** based on the independently inspected Windows MSVC QR intake artifact from source commit `44e3b84dab97a10b2727cc026e7124a9a5415421` (run `43762862057`, job `132961326705`, artifact `14181192246`). Its six-file archive, row order, observations, numerics and cross-file provenance passed the Day 3–5 audits. The artifact itself was generated as `local_only`; it does not establish that these edits passed a hosted run. The selected Windows claim remains conditional on the final changed-head run and audit.

## Field-level manifest change

Only `SRT-COMP-QR-INCOMPATIBLE-LS` changed in `tests/corpus/manifests/selected_report_targets.tsv` (verified by comparing every parsed row with `HEAD`).

| Field | Before | Day 6 candidate |
| --- | --- | --- |
| `support_tier` | `local_only` | `hosted_selected` |
| `workflow_file` | `ci.yml;macos-ci.yml` | same two, then `.github/workflows/windows-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` | same two, then `selected-qr-incompatible-comparison-freshness` |
| `workflow_artifact` | Linux and macOS Sprint 175 selected artifacts | same two, then `sprint209-windows-selected-comparison-qr-incompatible` |
| `workflow_platforms` | `linux;macos` | `linux;macos;windows` |
| `claim_scope` | Named fixture against the selected dense helper, without platform wording | Named fixture on reviewed Linux, macOS and Windows hosted lanes against that helper |
| `non_claims` | `no Windows report freshness` | `no broad Windows report freshness`; all other exclusions retained |

The generator now admits `hosted_selected` for QR only under the exact `selected-qr-incompatible-comparison-freshness` job, Windows platform and runner, `Windows CI` workflow, expected repository, push/PR event, exact checked-out SHA and branch, and clean worktree. A local QR run still emits `local_only`, local summary/claim scope and the original hosted/Windows denials. The hosted QR variant uses one named fixture and one selected Windows lane. The generator also carries the manifest's raw-basis and Q-orientation exclusions in its QR non-claims; broad QR, least-squares, rank, ecosystem, package, ABI, performance and state-of-the-art exclusions remain.

The Cholesky target, its job mapping and metadata transformation remain separate. The six QR file names, six ordered row IDs, artifact path, target key and generator command were not changed. Manifest, Windows structural validator and generator tests now check the candidate tuple and reject wrong workflow/job/artifact/platform metadata. The selected normalizer's existing test suite passes; its strict hosted cross-file validation is still Cholesky-specific and must be extended for QR in Days 8–10 before any public promotion.

## Checks and remaining gates

- `python3 tests/test_selected_report_targets_manifest.py`: pass.
- `python3 tests/test_run_external_comparison.py`: pass, including QR hosted context and narrow wording assertions.
- `python3 tests/test_selected_comparison_workflow.py`: pass.
- `python3 tests/test_validate_windows_powershell.py`: pass (the expected `pwsh`-missing negative fixture prints a failure diagnostic internally).
- `python3 tests/test_normalize_report_index.py`: pass.
- `git diff --check`: pass.
- Parsed-manifest before/after comparison: only the QR target row changed, in the seven fields above.
- A direct freshness check against the local `build/comparison/qr_incompatible_ls/study.tsv` fails as expected: that leftover bundle records commit `b56df58ba91ec4af66393ecd2318043e46adc763`, not current HEAD. It is not Day 6 evidence.

The present QR Windows workflow still uses default checkout and does not supply `SPARSE_SELECTED_SOURCE_SHA` or `SPARSE_SELECTED_SOURCE_BRANCH`. With this generator, that job cannot produce hosted-selected output until Day 7 wires an explicit PR-head checkout and source variables. Days 8–10 must add QR hosted normalizer and negative path/row/provenance fixtures. Days 11–14 must inspect a new exact-head Windows run and six-file artifact. If these gates fail, revert the QR candidate to the Linux/macOS-only selection and `local_only` wording.
