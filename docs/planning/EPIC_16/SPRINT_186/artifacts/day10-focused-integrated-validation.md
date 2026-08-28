# Sprint 186 Day 10: Focused Integrated Validation

## Purpose

Run the focused integrated validation queue from the Day 9 matrix before the
Day 11 full repository quality gate. Day 10 covers documentation, generated
API, package/provider status, selected manifest, workflow, report freshness,
and selected external comparison checks.

## Validation Results

| Command | Result | Evidence |
| --- | --- | --- |
| `git diff --check` | Pass | No whitespace errors were reported. |
| `make api-docs-validate` | Pass | Doxygen generated local HTML; API docs coverage reported 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages; local-only generated-output guard passed. |
| `make api-docs-freshness` | Pass | Re-ran the generated API validation path and confirmed `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` remain ignored local output with no tracked, staged, or visible non-ignored generated API files. |
| `make qr-header-docs-guard` | Pass | Header sections, declarations, unsupported-claim absence, and docs alignment passed. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Static package boundary, BUILD_SHARED_LIBS rejection, static target declaration, install metadata, support wording, Windows package non-claims, and workflow non-execution checks passed. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Package-manager deferral record, selected Homebrew local proof boundary, package metadata neutrality, and public non-claims passed. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus manifests and schemas validated. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Selected target manifest validation and Windows deferral drift checks passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Linux/macOS selected comparison workflow scope and Windows deferral guards passed. |
| `python3 tests/test_normalize_report_index.py` | Pass | Normalizer tests for selected freshness, package rows, deferred rows, optional rows, and manifest expectations passed. |
| `make report-index-oracle-freshness` | Pass | Selected local oracle output regenerated; freshness check passed with 54 normalized rows. |
| `make report-index-comparison-freshness` | Pass | Selected comparison outputs regenerated for QR minimum-norm, QR compatible least-squares, partial-SVD diagonal, LU nonsymmetric square, and Cholesky SPD tridiagonal targets; freshness check passed with 39 normalized rows. |
| `python3 tests/test_run_external_comparison.py` | Pass | External comparison runner tests passed. |

## Claim Coverage

| Claim family | Day 10 coverage |
| --- | --- |
| Generated API local-only status | `make api-docs-validate` and `make api-docs-freshness` passed; generated HTML remains ignored local output. |
| QR header coherence | `make qr-header-docs-guard` passed without editing public headers. |
| Static-first package boundary | `bash scripts/static_package_deferral_check.sh` passed. |
| Package-manager non-support and Homebrew proof path | `bash scripts/package_manager_deferral_check.sh` passed. |
| Selected report target authority | Corpus schema, selected target manifest, workflow guard, and normalizer tests passed. |
| Windows report freshness deferral | Selected manifest and workflow guard tests passed; no Windows selected freshness claim was added. |
| Selected oracle freshness | `make report-index-oracle-freshness` passed for the selected QR and partial-SVD oracle rows. |
| Selected comparison freshness | `make report-index-comparison-freshness` passed for selected QR, partial-SVD, LU, and Cholesky comparison rows. |

## Generated Output Handling

Day 10 regenerated ignored local outputs under:

- `docs/api/html/`
- `build/corpus/`
- `build/corpus-reports/`
- `build/report-index/`
- `build/comparison/`

These outputs are local validation artifacts and are not intended to be staged
or committed. The generated API local-only guard confirmed no tracked, staged,
or non-ignored generated API files. Python cache files created under
`scripts/__pycache__/` were removed after validation.

## Residuals Preserved

| Residual | Day 10 handling |
| --- | --- |
| R186-PKG-LICENSE | Remains active. Package/provider guards passed, but full Homebrew proof success remains blocked until approved standalone license metadata exists or an alternate formula license strategy is selected. |
| R186-WIN-PWSH | Remains active. Local `pwsh` is unavailable, so PowerShell parse/workflow validation still needs a suitable environment or hosted validation owner. |
| R186-WIN-REPORT-FRESHNESS | Remains active. Selected manifest and workflow guards preserve the formal Windows report freshness deferral. |
| R186-HOSTED-API | Remains active. Generated API HTML remains local-only; no hosted, retained artifact, or committed generated-output path was selected. |
| R186-BROAD-COMPARISON | Remains active. Selected comparison freshness passed for named bounded families only. |

## Day 11 Readiness

Day 10 focused validation passed with no in-scope fixes required. Day 11 can
run the C-adjacent review-surface guards and final broad quality gate:

```sh
make matmul-allocation-failure-gate
make ldlt-csc-helper-guard
make source-list-check
make format && make lint && make test
git diff --check
```

## Validation

Day 10 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

Required validation:

```sh
git diff --check
```
