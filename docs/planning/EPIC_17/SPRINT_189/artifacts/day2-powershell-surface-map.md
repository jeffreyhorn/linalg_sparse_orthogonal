# Sprint 189 Day 2: PowerShell Surface Map

## Purpose

Build the detailed owner-surface map for Sprint 189 PowerShell validation
ownership. The audit identifies workflow snippets, report-adjacent guards,
artifact names, selected report manifest assumptions, documentation surfaces,
and Day 3 command-design requirements.

## Windows Workflow Audit

| Workflow surface | Location | Classification | Validation expectation |
| --- | --- | --- | --- |
| Workflow-level Windows non-claim comment | `.github/workflows/windows-ci.yml` header | Required validation input | Preserve the Sprint 182 report freshness deferral and CMake-first/static-first boundary. |
| `build-and-test` job | `.github/workflows/windows-ci.yml` | Required validation input | Keep hosted `windows-2022`, CMake configure/build, `ctest -N`, full `ctest`, and `shell: pwsh` ownership. |
| CMake configure step | `build-and-test` | PowerShell snippet | Parse or dry-run command text when `pwsh` exists; guard `shell: pwsh`. |
| CMake build step | `build-and-test` | PowerShell snippet | Parse or dry-run command text when `pwsh` exists; guard `shell: pwsh`. |
| CTest surface inspection step | `build-and-test` | Multi-line PowerShell snippet | Validate syntax/parseability and anchors for `EXPECTED_WINDOWS_CTEST_COUNT`, `Total Tests:`, and reviewed-test output text. |
| CTest execution step | `build-and-test` | PowerShell snippet | Parse or dry-run command text when `pwsh` exists; guard `shell: pwsh`. |
| `install-and-downstream` job | `.github/workflows/windows-ci.yml` | Required validation input | Keep hosted `windows-2022`, static-first CMake install/downstream proof, and `shell: pwsh` ownership. |
| Install/downstream proof step | `install-and-downstream` | Multi-line PowerShell snippet | Validate syntax/parseability and anchors for static `.lib`, installed headers, CMake package metadata, metadata-only `sparse.pc`, downstream consumers, exact-version pass, and mismatch-version rejection. |

## Report and Artifact Audit

The selected report manifest currently has 7 rows and no row lists `windows`
in `workflow_platforms`. Sprint 189 must not add Windows report platforms or
selected Windows report uploads.

| Manifest surface | Current value | Classification | Sprint 189 decision |
| --- | --- | --- | --- |
| `workflow_file` | `.github/workflows/ci.yml`; `.github/workflows/macos-ci.yml` | Required guard input | Do not add `.github/workflows/windows-ci.yml` for selected report freshness in Sprint 189. |
| `workflow_job` | `generated-report-freshness`; `selected-comparison-freshness`; `hosted-performance-freshness` | Required guard input | Keep Windows workflow free of selected report freshness jobs. |
| `workflow_artifact` | `sprint159-oracle-freshness`; `sprint175-linux-selected-comparison-freshness`; `sprint175-macos-selected-comparison-freshness`; `sprint168-selected-performance-freshness` | Required guard input | Guard that Windows CI does not upload these selected report artifacts. |
| `workflow_platforms` | `linux`; `linux;macos` | Required guard input | Keep `windows` absent while Sprint 182 deferral is active. |
| `generator_command` | `make report-index-oracle-freshness`; `python3 scripts/run_external_comparison.py --target ...`; `make bench-canonical-report-freshness` | Retained non-goal for Windows | Do not run selected report generator commands in Windows CI during Sprint 189. |
| `artifact_pattern` and `required_files` | Generated local `build/` report paths | Retained non-goal for Windows | Do not publish or require selected Windows report artifacts in Sprint 189. |

## Existing Guard Audit

| Guard | Current role | Day 2 result |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Validates corpus manifest/schema shape. | Passed on Day 1 and remains a required baseline check. |
| `python3 tests/test_selected_report_targets_manifest.py` | Fails if selected manifest rows list `windows` while the Sprint 182 deferral is active. | Passed on Day 1; must remain part of Sprint 189 validation. |
| `python3 tests/test_selected_comparison_workflow.py` | Fails if Windows CI runs selected report freshness commands or uploads selected report artifacts while deferral is active. | Passed on Day 1; must remain part of Sprint 189 validation. |

## Documentation Audit

| Documentation surface | Current claim boundary | Sprint 189 implication |
| --- | --- | --- |
| `README.md` | Windows remains CMake-first; Windows report freshness is formally deferred. | Future docs may mention PowerShell validation ownership only, not report freshness. |
| `INSTALL.md` | Windows install path is CMake-only; no Windows Makefile or `pkg-config` execution parity. | Keep install support wording separate from report workflow validation. |
| `docs/maintainer_guide.md` | Local unavailable PowerShell checks are environment residuals, not pass evidence. | Day 10 should document the new command and local unavailable behavior. |
| Sprint 182 deferral artifact | Windows report freshness remains formally deferred. | Sprint 189 must not edit this into a promotion record. |
| Sprint 187 Windows acceptance gates | Sprint 189 closes PowerShell validation ownership only. | Day 13 should audit closure against these gates. |

## Classification Summary

| Surface family | Classification |
| --- | --- |
| `.github/workflows/windows-ci.yml` PowerShell blocks | Required validation input. |
| `shell: pwsh` declarations in Windows workflow steps | Required validation input. |
| Existing selected report manifest rows | Required guard input and retained Windows non-promotion evidence. |
| Linux/macOS selected report artifact names | Required guard input for Windows non-upload checks. |
| Report generators and generated report paths | Retained non-goal for Windows during Sprint 189. |
| Local `pwsh` availability | Local optional evidence; absence must be explicit unavailable evidence. |
| Hosted Windows `pwsh` execution | Hosted evidence for validation ownership once wired. |
| README, INSTALL, maintainer guide | Documentation surfaces for validation ownership and retained non-claims. |

## Day 3 Command-Design Requirements

The validation command should:

1. Have one stable invocation suitable for local maintainers and hosted
   Windows CI.
2. Parse or dry-run the selected `.github/workflows/windows-ci.yml`
   PowerShell snippets when `pwsh` is available.
3. Return explicit unavailable/skip evidence when local `pwsh` is absent,
   without treating absence as pass evidence.
4. Fail closed in hosted Windows CI when `pwsh` exists but parsing or guard
   checks fail.
5. Guard the Windows workflow `shell: pwsh` declarations for selected steps.
6. Reuse existing schema, manifest, and workflow guard expectations where
   practical.
7. Keep selected report freshness commands and selected artifact upload names
   out of Windows CI while Sprint 182 deferral remains active.
8. Keep `windows` out of selected report manifest `workflow_platforms` during
   Sprint 189.
9. Emit diagnostics that distinguish syntax validation, ownership validation,
   local unavailable state, and retained freshness non-claims.

## Day 2 Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
