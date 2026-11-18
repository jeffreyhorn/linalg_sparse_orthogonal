# Sprint 224 Day 7: QR documentation and workflow ownership candidate

## Decision boundary

The selected QR manifest and guarded generator changes from Day 6 are now reflected in the maintained public and maintainer surfaces. The Windows QR freshness sentence below is **conditional**. The intake artifact came from commit `44e3b84dab97a10b2727cc026e7124a9a5415421` before these edits. The final changed-head hosted MSVC job and its six-file archive must be independently audited before the sentence becomes an accepted claim.

> Selected QR incompatible least-squares comparison rows are fresh for the named fixture on reviewed Linux, macOS, and Windows hosted lanes against the selected source-controlled dense reference helper.

The same sentence is carried in `tests/corpus/manifests/selected_report_targets.tsv`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` and `tests/corpus/schemas/report_index_fields.md`. The Windows claim guard checks that sentence in all five maintained text surfaces and rejects unsupported blanket promotion wording. These docs identify the current state as a candidate pending a final changed-head audit; they do not cite the intake bundle as final proof.

## Claim and non-claim matrix

| Surface | Selected QR statement | Retained boundary |
| --- | --- | --- |
| Selected manifest | One named `qr-incompatible-ls` fixture; Linux, macOS and Windows workflow tuple | No broad QR/least-squares parity, QR basis or Q orientation identity, global rank policy, external package parity, broad Windows freshness, package-manager or ABI support, performance or state-of-the-art claim. |
| Generated QR rows/summary | `hosted_selected` only in the exact Windows QR job with matching source SHA/branch and clean tree; local runs remain `local_only` | Hosted wording refers only to the selected Windows lane. Broad exclusions remain in the generated non-claims. |
| Public README and INSTALL | Same conditional selected sentence; explicit final-head audit requirement | No broad Windows report, package, ABI, performance or solver coverage claim. |
| Maintainer guide, corpus README and schema | Same conditional selected sentence; manifest is the selection authority | Other report targets and the Sprint 182 broad Windows deferral remain separate. |
| Windows workflow | Exact QR job, selected normalizer, fail-closed six-file upload | Cholesky job unchanged; no oracle, benchmark, generated API docs publication or broad artifact upload. |

## Workflow to artifact correspondence

`selected-qr-incompatible-comparison-freshness` remains pinned to `windows-2022` and a 20-minute job budget. It now sets `SPARSE_SELECTED_SOURCE_SHA` to `${{ github.event.pull_request.head.sha || github.sha }}` and `SPARSE_SELECTED_SOURCE_BRANCH` to `${{ github.head_ref || github.ref_name }}`. Its `actions/checkout@v4` step checks out the same PR-head or push SHA. The generator's exact-job/repository/platform/event/SHA/branch/clean-worktree guard uses those values. The job still configures and builds the MSVC Release static library, runs only `qr-incompatible-ls`, checks selected-target freshness, and uploads only these flat members under `build/comparison/qr_incompatible_ls/`:

1. `project_observations.tsv`
2. `baseline_observations.tsv`
3. `dependency_status.tsv`
4. `study.tsv`
5. `summary.md`
6. `manifest.tsv`

The upload artifact remains `sprint209-windows-selected-comparison-qr-incompatible` with `if-no-files-found: error`. The Windows PowerShell and selected workflow guards verify the three source-provenance tokens and six upload paths; mutation tests remove each provenance token and require failure. The Cholesky lane and Sprint 220 local-only generated API docs rule were not changed.

## Checks and remaining work

- `python3 tests/test_selected_comparison_workflow.py`: pass.
- `python3 tests/test_validate_windows_powershell.py`: pass, including the expected local missing-`pwsh` negative fixture.
- `python3 tests/test_selected_report_targets_manifest.py`: pass.
- `python3 tests/test_run_external_comparison.py`: pass.
- `make support-docs-guard`: pass.
- `make docs-check`: pass (`api-docs-coverage` confirms all 18 public headers have generated reference and source pages).
- `API_DOCS_PYTHON=/tmp/sprint224-api-docs-venv/bin/python bash scripts/check_api_docs_local_only.sh`: pass with pinned PyYAML 6.0.3; structured policy checked 111 job/step records.
- `git diff --check`: pass.
- Direct `python3 scripts/validate_windows_powershell.py` completed structural checks but returned `UNAVAILABLE: pwsh not found` locally. This is not a hosted validation result.

Day 8 must finish the atomic candidate by extending strict hosted cross-file normalizer checks to QR and specifying the adversarial rejection cases. Days 9–10 add negative fixtures. Days 11–14 must inspect a new exact-head hosted job and six-file artifact. Any failed gate returns the QR target to explicit re-deferral rather than treating this conditional wording as accepted proof.
