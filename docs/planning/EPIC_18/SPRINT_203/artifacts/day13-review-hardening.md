# Day 13: Review Hardening

## Purpose

Day 13 audited the Sprint 203 diff for unrelated churn, stale wording,
accidental metadata/workflow promotion, documentation guard completeness, and
reviewer-facing evidence clarity.

## Hardening Change

The review found one documentation guard gap: Day 11 added the QR incompatible
Windows selected freshness boundary to
`tests/corpus/schemas/report_index_fields.md`, but
`scripts/validate_windows_powershell.py` did not yet enforce that schema
marker. Day 13 added the schema file to `CLAIM_BOUNDARY_MARKERS`.

## Changed-File Inventory

| Category | Files |
| --- | --- |
| Public/user docs | `README.md`; `INSTALL.md` |
| Maintainer docs | `docs/maintainer_guide.md` |
| Corpus/schema docs | `tests/corpus/README.md`; `tests/corpus/schemas/report_index_fields.md` |
| Windows claim-boundary guard | `scripts/validate_windows_powershell.py` |
| Normalizer tests | `tests/test_normalize_report_index.py` |
| Workflow guard tests | `tests/test_selected_comparison_workflow.py` |
| Manifest guard tests | `tests/test_selected_report_targets_manifest.py` |
| Sprint planning/evidence | `docs/planning/EPIC_18/SPRINT_203/PLAN.md`; `WORKING_NOTES.md`; Day 1 through Day 13 artifacts |

## Unchanged Promotion Surfaces

| Surface | Day 13 status |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Unchanged; no QR incompatible Windows workflow lane or upload path was added. |
| `.github/workflows/ci.yml` | Unchanged. |
| `.github/workflows/macos-ci.yml` | Unchanged. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Unchanged; `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`. |
| `scripts/run_external_comparison.py` | Unchanged. |
| `scripts/normalize_report_index.py` | Unchanged. |

## Guard Coverage Checklist

| Guard | Coverage |
| --- | --- |
| Manifest guard | Verifies QR incompatible row ids, required files, retained non-claims, no Windows platform, no Windows workflow file, and no reused Cholesky artifact. |
| Workflow guard | Rejects accidental Windows `qr-incompatible-ls` generator command, equals-form target command, selected freshness command, artifact name, QR subfamily token, or QR upload path. |
| Normalizer guard | Covers Windows-style QR artifact path matching, near-match rejection, stale rows, dependency-only rows, duplicate rows, unexpected rows, and wrong-target diagnostics. |
| Documentation guard | Enforces QR incompatible Windows selected freshness re-deferral markers across README, INSTALL, maintainer guide, corpus README, and report-index schema docs. |

## Reviewer Evidence Summary

Sprint 203 currently provides local selected QR incompatible generator proof,
local selected freshness proof, normalizer path/diagnostic regression coverage,
manifest/workflow re-deferral guards, and claim-safe documentation. It does not
provide hosted Windows/MSVC QR incompatible proof or inspected hosted Windows
QR artifacts. The correct reviewer interpretation is therefore re-deferral, not
promotion.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed; claim-boundary coverage now reports five files. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `python3 scripts/validate_windows_powershell.py` | Structural and claim-boundary checks passed; exited `2` because local `pwsh` is unavailable. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `git diff --check -- README.md INSTALL.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md scripts/validate_windows_powershell.py docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed. |
| `git diff -- .github/workflows tests/corpus/manifests` | Empty diff. |
| `git diff --name-only \| sort` | Reviewed changed-file inventory. |

## Residuals

| Residual | Review disposition |
| --- | --- |
| Hosted Windows/MSVC `qr-incompatible-ls` proof | Still absent; blocks Windows QR selected freshness promotion. |
| Hosted Windows QR artifact inspection | Still absent; blocks workflow/manifest promotion. |
| Local `pwsh` availability | Still unavailable; structural checks pass and this remains an environment residual. |
| Broad Windows report freshness | Still not promoted. |

## Promotion Boundary

No Day 13 change promotes Windows QR incompatible selected freshness. The only
implementation change is additional documentation guard coverage for the
already documented schema boundary.
