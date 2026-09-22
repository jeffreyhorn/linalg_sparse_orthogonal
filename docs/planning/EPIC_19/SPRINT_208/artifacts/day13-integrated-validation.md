# Sprint 208 Day 13 Integrated Validation

## Scope

Day 13 ran the integrated Sprint 208 validation matrix for the selected
Windows Cholesky freshness re-deferral path. The validation confirms that the
manifest, schema, workflow, PowerShell, normalizer, freshness, documentation,
and claim-boundary guards agree with the current branch state.

No `.c` or `.h` files were modified by Sprint 208, so the C quality gate trio
(`make format && make lint && make test`) was not required for Day 13.

## Validation Commands

| Command | Result | Notes |
| --- | --- | --- |
| `make windows-powershell-guard` | Passed | Structural Windows workflow and PowerShell checks passed. The guard also exercised the expected local `--require-pwsh` failure path because `pwsh` is not installed locally; this is not hosted pass evidence. |
| `python3 tests/test_normalize_report_index.py` | Passed | `test-normalize-report-index: ok`. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | `test-selected-report-targets-manifest: ok`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema validation completed without diagnostics. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | `test-selected-comparison-workflow: ok`. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | `test-validate-windows-powershell: ok`. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py` | Passed | Python compilation completed without diagnostics. |
| `make docs-check` | Passed | Doxygen generated local API docs, and `api-docs-coverage` checked 18 checked-in public headers plus the generated `sparse_version.h` exclusion. |
| `make support-docs-guard` | Passed | `test-support-quick-reference-docs: ok`. |
| `make report-index-comparison-freshness` | Passed | Generated local comparison evidence and reported freshness ok for 46 rows. |
| `rg -n "Sprint 199 reviewed\|Sprints 208 through 216\|208-216 \\\| Pending\|Windows selected (Cholesky\|comparison\|report) freshness is promoted\|PowerShell validation proves Windows report freshness\|hosted Windows report freshness is promoted\|Windows Cholesky selected freshness promoted" docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_19/PROJECT_PLAN.md README.md INSTALL.md` | Passed | Search returned no matches. |

## Generated Artifact Audit

`git status --short` showed only the expected modified Sprint 208 docs,
scripts, and Python tests plus the untracked `SPRINT_208` planning directory.

`git status --ignored --short` showed ignored generated or local outputs:

- `.claude/`
- `.swp`
- `archive/sparse_lu`
- `build/`
- `cmake-build/`
- `docs/api/`
- `scripts/__pycache__/`
- `tests/__pycache__/`

These are ignored local artifacts and were not staged as Sprint 208 evidence.

## Claim Boundary Result

Day 13 did not promote selected Windows Cholesky freshness. The selected
manifest remains source-controlled as Linux/macOS metadata only, with
`windows` absent from `workflow_platforms` and `no Windows report freshness`
retained as a required non-claim.

Retained non-claims:

- broad Windows report freshness;
- Windows selected oracle or benchmark freshness;
- Windows QR incompatible selected freshness;
- package-manager support;
- shared-library or dynamic ABI support;
- portable performance, release, external-library parity, or state-of-the-art
  evidence.

## Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Item 208.6 has current validation evidence. | Met. The selected Sprint 208 validation matrix passed for the re-deferral path. |
| Required commands pass or blockers are documented with exact failing output. | Met. Commands passed. The only local limitation is missing `pwsh`, which is explicitly covered as unavailable local evidence by the guard suite. |
| No generated proof artifacts or unsupported claim changes remain untracked. | Met. Generated outputs remain ignored; untracked Sprint 208 files are planned artifacts. |

