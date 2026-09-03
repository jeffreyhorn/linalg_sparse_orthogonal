# Sprint 196 Day 6 Artifact: Maintainer and API Documentation Recalibration

**Date:** 2026-09-03
**Sprint item coverage:** 196.2, 196.4, with inputs to 196.5 and 196.6
**Day 6 goal:** Align maintainer, API, report, and planning-adjacent
documentation with the public claim calibration from Day 5 while preserving
deeper evidence ownership.

## Summary

Day 6 updated maintainer/report documentation to match the public
`guarded-workflow` interpretation for the Sprint 190 Windows selected Cholesky
path. It also added a maintainer-facing Epic 17 evidence ownership map so
future claim reviews can find the right gate owner without using README as a
policy dump.

API documentation was audited and left unchanged because it already keeps
generated HTML local-only and avoids unsupported package, platform, ABI,
performance, release, hosted documentation, and state-of-the-art claims.

## Changed Files

| File | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Added an Epic 17 evidence ownership table for package, Windows/PowerShell, selected comparison, selected performance, review-surface, adoption/API, and reliability proof owners. |
| `docs/maintainer_guide.md` | Clarified that Sprint 190 Windows Cholesky is guarded workflow evidence until hosted evidence, selected metadata, support tier, and claim contract are reviewed together. |
| `tests/corpus/README.md` | Aligned selected report freshness wording with the guarded Windows Cholesky workflow interpretation and manifest-promotion requirement. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Recorded Day 6 edits, gate-owner map, planning-adjacent notes, and validation plan. |

## API Documentation Audit

| Surface | Result |
| --- | --- |
| `docs/api_reference.md` generated HTML section | Already states generated API HTML is local-only, ignored, and current only for the branch/checkout where freshness just passed. |
| Hosted docs claim | No hosted generated API publication claim found. |
| ABI/package/platform claim | Existing claim-boundary section rejects dynamic ABI compatibility, shared-library support, package-manager distribution, broad platform parity, external-library parity, portable performance, and state-of-the-art coverage. |
| Public headers | Not edited. Header changes were avoided because no concrete header overclaim was found and header edits would require the full C/header gate. |

## Maintainer Gate-Owner Map

| Evidence family | Maintainer owner | Required/focused gate |
| --- | --- | --- |
| Package/Homebrew proof | Package scripts, INSTALL, packaging Homebrew notes, Sprint 188 artifacts | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; install/CMake install tests when install files change |
| Windows/PowerShell ownership | Windows workflow, PowerShell validator, selected target manifest, claim-boundary markers | `make windows-powershell-guard`; `make windows-powershell-validate`; hosted Windows `--require-pwsh` job |
| Selected comparison freshness | Selected target manifest, corpus docs, comparison runner, normalizer | `make report-index-comparison-freshness`; selected manifest, runner, and normalizer tests |
| Selected performance evidence | Benchmark docs, selected target manifest, canonical freshness checker | `make bench-canonical-report-freshness`; selected performance and bench freshness tests |
| Review-surface reduction | QR proof owner, extracted helper, helper guard scripts/tests | `make qr-external-ref-helper-guard`; helper guard tests; full C gate after header/test edits |
| Adoption/API coherence | README, INSTALL, user docs, API reference, public headers | `make docs-check`; `make api-docs-freshness`; install/docs guards; full C gate after header edits |
| Reliability/failure-path proof | Selected implementation owner, focused tests, Make/CTest labels, claim docs | `make symbolic-allocation-failure-gate`; registration guard; full C gate after code/header edits |

## Residual Interpretation

- Package-manager support remains unclaimed until the package/provider proof
  and approved license metadata residual closes.
- Windows selected Cholesky remains a guarded workflow path until hosted
  evidence and selected target manifest metadata are promoted together.
- Local missing `pwsh` remains environment residual evidence, not a pass.
- Selected comparison and selected performance evidence stay target/row scoped.
- Reliability proof remains selected-owner scoped.
- Generated API HTML remains local-only and not hosted/release evidence.
- State-of-the-art status remains unclaimed.

## Validation Results

The changed files are documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required for Day 6.

- `git diff --check`: passed.
- `make windows-powershell-guard`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_normalize_report_index.py`: passed.
- `make docs-check`: passed.
- `make api-docs-freshness`: passed.

Generated-output hygiene:

- `docs/api/html/` was regenerated by the docs checks and remains ignored.
- Generated `scripts/__pycache__/` files from Python validation imports were
  removed after validation.
