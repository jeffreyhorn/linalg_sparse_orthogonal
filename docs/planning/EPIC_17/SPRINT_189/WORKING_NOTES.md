# Sprint 189 Working Notes

## Sprint Goal

Close the PowerShell validation environment gap by adding an owned local/hosted
validation command for Windows report workflow material.

## Branch Baseline

- Branch: `sprint-189`
- Starting point: current `master` after PR #209 merge.
- Epic 17 owner gap: PowerShell validation ownership for Windows report
  workflow material.
- Sprint 189 plan status: day-by-day plan exists at
  `docs/planning/EPIC_17/SPRINT_189/PLAN.md`.

## Planning Source

| Field | Value |
| --- | --- |
| Project plan | `docs/planning/EPIC_17/PROJECT_PLAN.md` |
| Section | `Sprint 189: PowerShell Validation Ownership` |
| Sprint duration | 14 days, approximately 166 hours |
| Acceptance gate source | `docs/planning/EPIC_17/SPRINT_187/artifacts/day8-windows-acceptance-gates.md` |
| Prior deferral source | `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` |
| Workflow guard source | `tests/test_selected_comparison_workflow.py` |
| Manifest guard source | `tests/test_selected_report_targets_manifest.py` |

## Sprint 189 Item Boundaries

| Item | Name | Sprint 189 interpretation |
| --- | --- | --- |
| 189.1 | PowerShell Surface Audit | Inventory Windows workflow PowerShell snippets, report-adjacent scripts, artifact names, selected report manifests, and claim boundaries. |
| 189.2 | Validation Command Design | Define an owned command that validates selected PowerShell workflow material locally when `pwsh` exists and in hosted Windows CI. |
| 189.3 | Hosted CI Wiring | Add hosted Windows validation wiring without promoting Windows report freshness or artifact publication. |
| 189.4 | Guard Tests | Add checks that fail on stale validation ownership, unsupported artifact names, or accidental report freshness promotion. |
| 189.5 | Documentation Update | Explain the PowerShell validation owner, local unavailable behavior, hosted evidence, and retained non-claims. |
| 189.6 | Validation | Run Windows-adjacent guards, manifest/schema tests, docs checks, and the full C gate if `.c` or `.h` files change. |

## Day 1 PowerShell Validation Baseline

Day 1 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day1-powershell-validation-intake.md`
as the baseline inventory and owner-surface record.

### Day 1 Owner Surface Inventory

| Surface | Current owner files | Day 1 finding |
| --- | --- | --- |
| Windows hosted workflow | `.github/workflows/windows-ci.yml` | Present. Uses `shell: pwsh` for the reviewed CMake configure/build, CTest, and install/downstream lanes. |
| Windows report freshness deferral | `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` | Present. Sprint 189 must keep report freshness deferred while adding validation ownership. |
| Windows acceptance gates | `docs/planning/EPIC_17/SPRINT_187/artifacts/day8-windows-acceptance-gates.md` | Present. Defines Sprint 189 as PowerShell validation ownership only, not report freshness promotion. |
| Selected report manifest | `tests/corpus/manifests/selected_report_targets.tsv` | Present with 7 selected rows and no Windows workflow platform rows. |
| Manifest/schema guards | `scripts/validate_corpus_schema.py`, `tests/test_selected_report_targets_manifest.py` | Present and passing. |
| Workflow freshness guard | `tests/test_selected_comparison_workflow.py` | Present and passing. Guards Windows report freshness deferral and selected artifact non-uploads. |
| Maintainer docs | `docs/maintainer_guide.md` | Present. Documents Windows report freshness deferral and warns that local PowerShell unavailability is not evidence. |
| User docs | `README.md`, `INSTALL.md` | Present. Keep Windows CMake-first/static-first and report freshness non-claims. |

### Day 1 Local Tool Snapshot

| Tool | Local status | Sprint 189 impact |
| --- | --- | --- |
| `pwsh` | Not found on local `PATH`. | Local PowerShell validation must report unavailable/skip evidence unless Day 7 adds a mockable self-test path. |
| `powershell` | Not found on local `PATH`. | Windows PowerShell is not locally available as a fallback command. |
| `python3` | Available at `/usr/local/bin/python3`. | Local schema, manifest, and workflow guards can run. |
| `gh` | Available at `/usr/local/bin/gh`. | Hosted workflow dispatch and run inspection are possible if the sprint reaches CI execution. |

### Day 1 Baseline Command Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Exit `0` | Corpus schema baseline is valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Exit `0` | Selected target manifest baseline is valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Exit `0` | Existing workflow guard keeps Windows report freshness deferred. |
| Selected manifest Windows platform scan | 0 rows | No selected report row lists `windows` in `workflow_platforms`. |
| `command -v pwsh` | Not found | Local PowerShell validation is currently unavailable and must not be treated as pass evidence. |

### Day 1 Non-Claim Baseline

Sprint 189 may close PowerShell validation ownership. It must not claim:

- Windows report freshness;
- selected Windows report artifact publication;
- broad Windows platform parity;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library package support;
- dynamic ABI, DLL/import-library, runtime-loader, or ABI stability support;
- portable performance or state-of-the-art evidence from Windows reports.

### Day 1 Risks

| Risk | Mitigation |
| --- | --- |
| Local `pwsh` absence could be mistaken for a passed validation command. | The command contract must emit explicit unavailable/skip evidence and docs must say that local absence is not hosted proof. |
| Hosted wiring could accidentally run selected report freshness commands. | Keep Sprint 182 deferral guards active and add ownership checks without report generation or upload promotion. |
| Workflow artifact names could drift separately from selected report manifest rows. | Day 2 must inventory artifact names and Day 6/Day 9 must guard the selected assumptions. |
| A PowerShell parse command could validate syntax but be overread as Windows report freshness. | Docs and guards must separate syntax/ownership validation from generated artifact freshness evidence. |
| C/header edits are not expected but would expand validation requirements. | Any `.c` or `.h` change requires `make format && make lint && make test`. |

### Day 1 Open Questions

| Question | Day 1 disposition |
| --- | --- |
| Which PowerShell snippets are selected for owned validation? | Open for Day 2 surface audit. |
| Should the owner command be shell, Python, Make, or a combination? | Open for Day 3 command design. |
| Should hosted Windows CI run the validation command in the existing workflow or a separate job? | Open for Day 8 hosted wiring. |
| How should local unavailable output be represented in exit codes and docs? | Open for Day 3 command design. |
| Should Sprint 189 change selected report manifest metadata? | No on Day 1. Manifest Windows platforms remain absent; report freshness promotion belongs to Sprint 190. |

### Day 1 Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 2 PowerShell Surface Audit

Day 2 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day2-powershell-surface-map.md`
as the detailed PowerShell surface map and Day 3 command-design handoff.

### Day 2 Workflow Findings

| Workflow surface | Finding |
| --- | --- |
| `.github/workflows/windows-ci.yml` header | Preserves the CMake-first/static-first Windows boundary and Sprint 182 report freshness deferral. |
| `build-and-test` job | Runs on `windows-2022` and uses `shell: pwsh` for CMake configure, CMake build, CTest inventory, and full CTest execution. |
| `install-and-downstream` job | Runs on `windows-2022` and uses one multi-line `shell: pwsh` step for CMake install/downstream validation. |
| Selected report freshness commands | Not present as Windows workflow run steps while Sprint 182 deferral remains active. |
| Selected report artifact uploads | Not present in Windows workflow while Sprint 182 deferral remains active. |

### Day 2 Report and Artifact Findings

The selected report target manifest has 7 rows and no `windows` platform rows.
Current selected workflow artifacts are Linux/macOS scoped:

- `sprint159-oracle-freshness`;
- `sprint175-linux-selected-comparison-freshness`;
- `sprint175-macos-selected-comparison-freshness`;
- `sprint168-selected-performance-freshness`.

Sprint 189 should guard that Windows CI does not upload those selected report
artifacts and does not run selected report freshness commands. Windows report
freshness promotion belongs to Sprint 190, not the Sprint 189 validation owner.

### Day 2 Documentation Findings

| Surface | Finding |
| --- | --- |
| `README.md` | Keeps Windows CMake-first and report freshness formally deferred. |
| `INSTALL.md` | Keeps Windows install support CMake-only and avoids Makefile/`pkg-config` execution parity claims. |
| `docs/maintainer_guide.md` | States local unavailable PowerShell checks are environment residuals rather than pass evidence. |
| Sprint 182 deferral artifact | Keeps Windows report freshness formally deferred. |
| Sprint 187 acceptance gates | Define Sprint 189 as validation ownership, not freshness promotion. |

### Day 2 Command-Design Handoff

Day 3 should design one stable validation command that parses or dry-runs the
selected Windows workflow PowerShell snippets when `pwsh` exists, reports an
explicit unavailable state when local `pwsh` is absent, fails closed in hosted
Windows CI, guards selected `shell: pwsh` declarations, reuses existing
manifest/workflow checks, and keeps selected Windows report freshness
unpromoted.

### Day 2 Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 3 Validation Command Design

Day 3 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day3-validation-command-design.md`
as the command contract and Day 4 implementation checklist.

### Day 3 Command Shape Decision

| Layer | Selected name | Purpose |
| --- | --- | --- |
| Script | `python3 scripts/validate_windows_powershell.py` | Own structural workflow/report checks and PowerShell parse validation. |
| Make target | `make windows-powershell-validate` | Provide the stable maintainer entry point. |
| Hosted CI invocation | `python3 scripts/validate_windows_powershell.py --require-pwsh` | Fail closed on hosted Windows if `pwsh` is missing or parsing fails. |

The script will use Python because existing workflow and manifest guards are
Python-based, and the Make target will expose the stable local maintainer
command.

### Day 3 Exit-Code Contract

| Condition | Exit | Interpretation |
| --- | ---: | --- |
| `pwsh` exists and all checks pass | `0` | PowerShell snippets parsed and structural ownership checks passed. |
| `pwsh` is absent in local/default mode and structural checks pass | `2` | Local unavailable evidence only; not pass evidence. |
| Structural checks fail | `1` | Validation failure. |
| `pwsh` exists but parsing fails | `1` | Validation failure. |
| `--require-pwsh` is used and `pwsh` is absent | `1` | Hosted validation failure. |

### Day 3 Required Structural Checks

The validation command should fail if the Windows workflow is missing, the
Sprint 182 deferral comment disappears, reviewed Windows jobs are missing,
selected Windows steps lose `shell: pwsh`, selected jobs stop using
`windows-2022`, selected report freshness commands appear in Windows CI,
selected report upload artifacts appear in Windows CI, selected manifest rows
list `windows`, or the Sprint 182 deferral artifact loses marker text.

### Day 3 Parse Strategy

When `pwsh` is available, the command should parse selected workflow `run`
blocks through `[scriptblock]::Create(...)` in no-profile/non-interactive mode.
It must not execute CMake, CTest, report generators, uploads, or generated
artifact commands as part of parse validation.

### Day 3 Day-4 Handoff

Day 4 should implement `scripts/validate_windows_powershell.py`, add the
`windows-powershell-validate` Make target, emit stable pass/fail/unavailable
diagnostics, and rerun the existing schema, manifest, and workflow guards.

### Day 3 Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 4 Local Validation Command Scaffold

Day 4 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day4-local-command-scaffold.md`
as the implementation record for the initial PowerShell validation command.

### Day 4 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added the owned validation script for selected Windows CI PowerShell workflow material. |
| `Makefile` | Added `windows-powershell-validate` as the stable maintainer entry point. |

### Day 4 Command Behavior

| Invocation | Result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural checks pass locally, then missing `pwsh` is reported as unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | The Make target exposes the same local unavailable state. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing `pwsh`. |

### Day 4 Structural Coverage

The script now checks the Windows workflow deferral comment, selected jobs,
`windows-2022` runner ownership, selected `shell: pwsh` declarations, selected
step command anchors, absence of selected report freshness commands/artifact
names in Windows CI, the Sprint 182 deferral marker, and absence of `windows`
from selected report manifest workflow platforms.

### Day 4 Parse Path

When `pwsh` is available, selected `run` blocks are written to temporary
snippets and parsed with `[scriptblock]::Create(...)`. The script does not
execute CMake, CTest, report generators, uploads, or generated artifact
commands during parse validation.

### Day 4 Day-5 Handoff

Day 5 should expand workflow snippet validation with tighter command-reference
checks and focused drift coverage around selected Windows steps and
`shell: pwsh` ownership.

### Day 4 Validation

Day 4 changed a Python script, the Makefile, and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 5 Workflow Snippet Validation

Day 5 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day5-workflow-snippet-coverage.md`
as the workflow-snippet validation coverage record.

### Day 5 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added detection for unowned `shell: pwsh` steps in the Windows workflow. |
| `tests/test_validate_windows_powershell.py` | Added focused tests for current workflow ownership, shell drift, command-token drift, unowned PowerShell steps, forbidden report freshness commands, deferral record validation, manifest deferral validation, and local unavailable semantics. |

### Day 5 Owned PowerShell Steps

The validator owns all five current `shell: pwsh` steps in
`.github/workflows/windows-ci.yml`:

| Job | Step |
| --- | --- |
| `build-and-test` | `Run enforced reviewed CMake configure path (MSVC, x64)` |
| `build-and-test` | `Run enforced reviewed CMake build path (Release)` |
| `build-and-test` | `Inspect enforced Windows reviewed consumer CTest surface (ctest -N)` |
| `build-and-test` | `Run enforced reviewed CMake execution path (ctest)` |
| `install-and-downstream` | `Run reviewed CMake install/downstream validation proof` |

Any new `shell: pwsh` step in Windows CI now fails validation until it is
intentionally added to the selected ownership list or moved out of scope.

### Day 5 Drift Coverage

| Drift | Guard result |
| --- | --- |
| Selected step changes shell. | Fails with `must declare shell: pwsh`. |
| Selected configure command anchor changes. | Fails with the missing command token. |
| New unowned PowerShell step appears. | Fails with `windows workflow has unowned PowerShell steps`. |
| Windows CI gains selected report freshness command or artifact name. | Fails with selected report freshness non-promotion diagnostics. |
| Deferral record or selected manifest violates Windows deferral. | Fails through deferral/manifest validation. |

### Day 5 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | New workflow ownership and drift tests pass without local `pwsh`. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural checks pass, then missing local `pwsh` is unavailable evidence. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest still keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects Windows selected report freshness commands/uploads. |

Day 5 changed Python tests, a Python script, and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 6 Report Artifact and Manifest Validation

Day 6 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day6-report-artifact-guards.md`
as the report artifact and manifest validation record.

### Day 6 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added selected report manifest reference validation. |
| `scripts/validate_windows_powershell.py` | Derives selected report freshness blocker tokens from manifest `generator_command`, `workflow_job`, and `workflow_artifact` fields. |
| `tests/test_validate_windows_powershell.py` | Added drift tests for manifest-derived artifact blockers and missing workflow-file references. |

### Day 6 Manifest Reference Checks

The PowerShell validator now verifies that selected report manifest rows have
expected selected families, non-empty workflow metadata, valid artifact/platform
cardinality, existing referenced workflow files, sprint-scoped selected
artifact names, and no `windows` workflow platform while the Sprint 182
deferral is active.

### Day 6 Artifact Guarding

Current selected workflow artifacts remain Linux/macOS scoped:

- `sprint159-oracle-freshness`;
- `sprint175-linux-selected-comparison-freshness`;
- `sprint175-macos-selected-comparison-freshness`;
- `sprint168-selected-performance-freshness`.

Because the validator derives blocker tokens from the manifest, any future
selected report artifact added to the manifest is automatically forbidden in
Windows CI while Windows report freshness remains deferred.

### Day 6 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Manifest-derived artifact blockers and workflow-reference checks behave as expected. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural, report, and manifest checks pass; local `pwsh` remains unavailable evidence. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest still keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

Day 6 changed Python tests, a Python script, and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 7 Local PowerShell Parse Path

Day 7 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day7-local-pwsh-path.md`
as the local available/unavailable path evidence record.

### Day 7 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Flushed pass/fail/unavailable diagnostics to keep redirected logs ordered. |
| `scripts/validate_windows_powershell.py` | Cleaned the workflow job block line-slice helper while preserving behavior. |
| `tests/test_validate_windows_powershell.py` | Added fake-`pwsh` tests for selected snippet parse success, parse failure diagnostics, and full `main()` available-path success. |

### Day 7 Local Behavior Evidence

This local environment still has no `pwsh` executable on `PATH`, so default
local validation returns unavailable evidence after structural checks pass.

| Invocation | Expected result | Interpretation |
| --- | ---: | --- |
| `python3 scripts/validate_windows_powershell.py` | `2` | Local unavailable evidence; not proof success. |
| `make windows-powershell-validate` | `2` | Stable maintainer entry point exposes the same unavailable state. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | `1` | Hosted/fail-closed mode rejects missing `pwsh`. |

### Day 7 Fake PowerShell Coverage

The focused validator test now creates a temporary fake `pwsh` executable to
exercise the available path without requiring PowerShell on all developer
machines. The fake path confirms the validator passes `-NoProfile`,
`-NonInteractive`, and `SPARSE_PWSH_SNIPPET`, parses all five selected
workflow snippets, reports parse subprocess failures clearly, and allows
`--require-pwsh` to succeed when `pwsh` is available.

### Day 7 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Fake available, fake failure, and local unavailable paths behave as expected. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |

Day 7 changed Python tests, a Python script, and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 8 Hosted Windows CI Wiring

Day 8 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day8-hosted-windows-lane.md`
as the hosted Windows validation lane evidence record.

### Day 8 Implementation Summary

| Surface | Change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added a dedicated `powershell-validation` job on `windows-2022`. |
| `.github/workflows/windows-ci.yml` | The job runs `python scripts/validate_windows_powershell.py --require-pwsh` under `shell: cmd`. |
| `scripts/validate_windows_powershell.py` | Added hosted validation wiring checks for job id, runner, command, and shell. |
| `tests/test_validate_windows_powershell.py` | Added drift tests for missing fail-closed mode, wrong hosted runner, and accidental `shell: pwsh` on the hosted validation step. |

### Day 8 Hosted Job Contract

| Field | Value |
| --- | --- |
| Job id | `powershell-validation` |
| Runner | `windows-2022` |
| Command | `python scripts/validate_windows_powershell.py --require-pwsh` |
| Shell | `cmd` |

The hosted job validates selected Windows PowerShell workflow ownership and
snippet parseability. It does not run report generation, upload selected
report artifacts, or promote Windows report freshness.

### Day 8 Guarded Drift

| Drift | Guard result |
| --- | --- |
| Hosted validation command loses `--require-pwsh`. | Fails with hosted command diagnostic. |
| Hosted validation job moves off `windows-2022`. | Fails with hosted runner diagnostic. |
| Hosted validation step changes to `shell: pwsh`. | Fails because the hosted validator invocation must stay `shell: cmd`. |
| Windows workflow gains selected report freshness command or selected artifact name. | Fails through selected freshness non-promotion checks. |

### Day 8 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Hosted wiring, fake PowerShell, and local unavailable paths behave as expected. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |

Day 8 changed a workflow file, Python tests, a Python script, and planning
documentation but no `.c` or `.h` files. The full C quality gate is not
required.

## Day 9 Ownership Guard Tests

Day 9 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day9-ownership-guard-tests.md`
as the ownership guard and claim-boundary evidence record.

### Day 9 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added claim-boundary checks across README, INSTALL, maintainer guide, and corpus README. |
| `scripts/validate_windows_powershell.py` | Added unsupported Windows/PowerShell promotion phrase detection. |
| `scripts/validate_windows_powershell.py` | Added a Windows workflow `actions/upload-artifact` guard while selected Windows report evidence is absent. |
| `tests/test_validate_windows_powershell.py` | Added tests for docs claim anchors, claim promotion drift, Windows artifact-publication drift, and local unavailable non-pass wording. |

### Day 9 Guarded Drift

| Drift | Guard result |
| --- | --- |
| Windows workflow adds `actions/upload-artifact`. | Fails while selected Windows report evidence is absent. |
| README loses the Sprint 182 Windows report freshness deferral marker. | Fails with a missing non-claim marker diagnostic. |
| Maintainer docs say PowerShell validation proves Windows report freshness. | Fails with unsupported Windows/PowerShell claim diagnostic. |
| Local unavailable output loses the non-pass evidence sentence. | Fails the focused validator test. |
| Hosted validation wiring loses `--require-pwsh`, `windows-2022`, or `shell: cmd`. | Continues to fail through Day 8 hosted wiring checks. |

### Day 9 Claim Boundary

The validator can now help prevent PowerShell validation ownership from being
overread as Windows report freshness. The guarded interpretation remains:
selected Windows PowerShell snippets are structurally owned and parse-checked
when `pwsh` exists, while generated Windows report freshness and selected
Windows report artifact publication remain deferred until a future sprint
reviews a Windows-safe generator path, selected manifest promotion, artifact
scope, and matching guard updates together.

### Day 9 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Ownership, claim-boundary, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |

Day 9 changed Python tests, a Python script, and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 10 Maintainer Validation Docs

Day 10 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day10-maintainer-validation-docs.md`
as the maintainer documentation update record.

### Day 10 Implementation Summary

| Surface | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Added `Windows PowerShell Validation Ownership` guidance. |
| `scripts/validate_windows_powershell.py` | Added maintainer-guide claim-boundary markers for `make windows-powershell-validate` and hosted `--require-pwsh` guidance. |

### Day 10 Maintainer Contract

The maintainer guide now names `make windows-powershell-validate` as the
stable local entry point for changes to the Windows workflow, selected report
target metadata, selected report artifact names, Windows report freshness
wording, or Windows support interpretation.

The guide records local exit semantics:

| Exit | Meaning |
| ---: | --- |
| `0` | Local `pwsh` exists and selected snippets parsed after structural checks passed. |
| `2` | Structural checks passed but local `pwsh` is unavailable; blocker evidence only. |
| `1` | Structural, claim-boundary, hosted wiring, fail-closed, or parse failure. |

The hosted lane remains
`python scripts/validate_windows_powershell.py --require-pwsh` on
`windows-2022`, with missing hosted PowerShell treated as failure.

### Day 10 Retained Non-Claims

The maintainer guide explicitly keeps the hosted validation lane separate from
selected report artifact upload, selected report generation, selected
manifest `windows` promotion, and Windows report freshness. Future Windows
report freshness promotion still requires a Windows-safe generator path,
exact selected upload scope, selected-target manifest metadata, and guard
updates in one reviewed change.

### Day 10 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Maintainer command markers, claim boundaries, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |

Day 10 changed maintainer documentation, a Python script, and planning
documentation but no `.c` or `.h` files. The full C quality gate is not
required.

## Day 11 Windows Claim Calibration

Day 11 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day11-windows-claim-calibration.md`
as the user-facing and report-facing documentation calibration record.

### Day 11 Implementation Summary

| Surface | Change |
| --- | --- |
| `README.md` | Added hosted PowerShell validation ownership to the Windows CI support summary and clarified it is workflow validation only. |
| `INSTALL.md` | Added hosted PowerShell validation ownership to the Windows supported-platform row while retaining non-claims. |
| `tests/corpus/README.md` | Added report-facing wording that the hosted Windows PowerShell lane is snippet parsing and structural guard evidence only. |
| `scripts/validate_windows_powershell.py` | Added claim-boundary markers for the new README, INSTALL, and corpus README wording. |

### Day 11 Claim Boundary

The calibrated support statement is now consistent across public and
report-facing docs: Windows has reviewed CMake build/test coverage, reviewed
CMake install/downstream validation, and hosted PowerShell validation
ownership for selected workflow snippets. It still does not claim report
freshness, selected Windows report artifact publication, Windows Makefile or
`pkg-config` execution parity, package-manager support, shared-library
support, dynamic ABI support, runtime-loader behavior, or broad Windows
parity.

### Day 11 Revisit Criteria

Windows report freshness remains deferred until one reviewed change adds a
Windows-safe generated report path, exact selected upload scope,
selected-target manifest metadata including `windows`, support-tier and
claim-scope fields, and matching workflow/validator guard updates.

### Day 11 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | User-facing and report-facing claim markers, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |

Day 11 changed user-facing documentation, report-facing documentation, a
Python script, and planning documentation but no `.c` or `.h` files. The full
C quality gate is not required.

## Day 12 Integrated Windows-Adjacent Validation

Day 12 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day12-integrated-windows-validation.md`
as the integrated validation record before claim audit.

### Day 12 Changed Surfaces

| Surface | Validation relevance |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Hosted Windows PowerShell validation lane and report non-promotion. |
| `scripts/validate_windows_powershell.py` | Owned PowerShell/workflow/report/docs validation command. |
| `tests/test_validate_windows_powershell.py` | Focused ownership, claim-boundary, hosted, fake PowerShell, and unavailable-path tests. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | Windows PowerShell validation wording and retained non-claims. |

### Day 12 Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 - <<'PY' ... ast.parse(...)` | Passed | Validator and focused test parse as Python. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Owned validator guard coverage passes. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard remains valid. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalized report-index tests remain valid. |
| `python3 scripts/normalize_report_index.py --check` | Passed | Normalized report index reports `112` rows. |
| `make docs-check` | Passed | Doxygen/API docs coverage remains valid. |

### Day 12 Hygiene Results

| Check | Result |
| --- | --- |
| `git diff --check` | Passed. |
| Trailing whitespace scan | Passed. |
| Sprint 189 markdown link check | Passed. |
| `docs/api/html` status after `make docs-check` | No repo changes left by generated docs. |

No `.c` or `.h` files were modified during Sprint 189 through Day 12, so the
full C quality gate is not required.

### Day 12 Claim-Audit Readiness

There are no unresolved validation failures before Day 13. The only nonzero
local results are expected outcomes from missing local `pwsh`: exit `2` for
default local validation and the Make target, and exit `1` for hosted
`--require-pwsh` mode.

## Day 13 Claim Audit and Residual Decision

Day 13 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day13-claim-audit-residual-decision.md`
as the final claim-state and residual decision record before closeout.

### Day 13 Gate Decision

Sprint 189 closes source-controlled PowerShell validation ownership. The
closed claim is intentionally narrow: selected Windows workflow PowerShell
snippets now have an owned validator, Make entry point, fake-`pwsh` local test
coverage, hosted fail-closed workflow wiring, documentation, and drift guards.

Hosted pass evidence remains pending until the branch is pushed and PR CI runs
the new `powershell-validation` job. That is an operational evidence item, not
a source-code residual.

### Day 13 Retained Residuals

| Residual | Status |
| --- | --- |
| Local `pwsh` missing on this machine | Retained environment residual; default validator exits `2`. |
| Hosted Windows PowerShell validation run evidence | Pending PR CI after branch push. |
| Windows report freshness | Still formally deferred by Sprint 182; Sprint 190 owns the next decision. |
| Selected Windows report artifact publication | Not implemented and guarded against. |
| Broad Windows parity, Windows Makefile parity, Windows `pkg-config` execution parity | Not claimed. |
| Package-manager, shared-library, dynamic ABI, DLL/import-library, runtime-loader support | Not claimed. |

### Day 13 Fresh Audit Evidence

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Ownership, hosted wiring, claim-boundary, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| Unsupported claim `rg` scan | No matches | Touched Windows/report surfaces do not contain scanned unsupported promotion phrases. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected report workflow guard remains valid. |

Day 13 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 14 Sprint Closeout

Day 14 created
`docs/planning/EPIC_17/SPRINT_189/artifacts/day14-sprint-closeout.md`
as the final Sprint 189 closeout, retrospective input, and PR-ready summary
record.

### Day 14 Final Sprint State

Sprint 189 closes source-controlled PowerShell validation ownership. The
closed scope includes the owned validator, Make entry point, hosted
fail-closed workflow job, focused drift tests, local unavailable semantics,
claim-boundary guards, and maintainer/user/report-facing documentation.

### Day 14 Item Completion

| Item | Disposition |
| --- | --- |
| 189.1 PowerShell Surface Audit | Complete. |
| 189.2 Validation Command Design | Complete. |
| 189.3 Hosted CI Wiring | Complete in source; hosted pass evidence pending PR CI. |
| 189.4 Guard Tests | Complete. |
| 189.5 Documentation Update | Complete. |
| 189.6 Validation | Complete for local/source-controlled checks. |

### Day 14 Retained Residuals

| Residual | Disposition |
| --- | --- |
| Local `pwsh` unavailable | Expected environment residual; default validator exits `2`. |
| Hosted Windows pass evidence | Pending PR CI after branch push. |
| Windows report freshness | Still formally deferred by Sprint 182; Sprint 190 owns the next decision. |
| Selected Windows report artifact publication | Not implemented and guarded against. |

### Day 14 Final Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Owned validator guard coverage passes. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Make target preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed without local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected report workflow guard remains valid. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalized report-index tests remain valid. |
| `python3 scripts/normalize_report_index.py --check` | Passed | Normalized report index reports `112` rows. |
| `make docs-check` | Passed | Doxygen/API docs coverage remains valid. |
| Unsupported claim scan | No matches | No unsupported Windows/report promotion phrases found. |
| Stale marker scan | No open blockers | Hits are explanatory docs/plan wording, not unresolved Sprint 189 work. |
| `git diff --check` | Passed | Patch whitespace is valid. |
| Sprint 189 markdown link check | Passed | Sprint-local links resolve. |

No `.c` or `.h` files were modified in Sprint 189, so the full C quality gate
is not required.

### Day 14 PR-Ready Notes

- Added `make windows-powershell-validate`.
- Added `scripts/validate_windows_powershell.py`.
- Added `tests/test_validate_windows_powershell.py`.
- Added hosted `powershell-validation` job in `.github/workflows/windows-ci.yml`.
- Updated README, INSTALL, maintainer guide, and corpus README with bounded
  Windows PowerShell validation wording.
- Preserved Windows report freshness deferral and selected artifact
  non-publication.
