# Sprint 189 Day 5: Workflow Snippet Coverage

## Purpose

Expand PowerShell validation ownership so selected Windows workflow snippets
and shell assumptions are guarded against drift before hosted CI wiring.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added unowned `shell: pwsh` detection so newly added Windows PowerShell steps cannot bypass the owned validation surface. |
| `tests/test_validate_windows_powershell.py` | Added focused drift coverage for current workflow ownership, shell drift, command-anchor drift, unowned PowerShell steps, forbidden report freshness commands, deferral record validation, manifest deferral validation, and local unavailable semantics. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 5 workflow snippet validation coverage and results. |

## Workflow Ownership Coverage

The validator now owns every current `shell: pwsh` step in
`.github/workflows/windows-ci.yml`. The selected set remains:

| Job | Step | Required command anchors |
| --- | --- | --- |
| `build-and-test` | `Run enforced reviewed CMake configure path (MSVC, x64)` | `cmake -S . -B build`; `Visual Studio 17 2022` |
| `build-and-test` | `Run enforced reviewed CMake build path (Release)` | `cmake --build build`; `Release` |
| `build-and-test` | `Inspect enforced Windows reviewed consumer CTest surface (ctest -N)` | `EXPECTED_WINDOWS_CTEST_COUNT`; `Total Tests:` |
| `build-and-test` | `Run enforced reviewed CMake execution path (ctest)` | `ctest --test-dir build`; `--output-on-failure` |
| `install-and-downstream` | `Run reviewed CMake install/downstream validation proof` | `sparse_lu_ortho.lib`; `sparse.pc`; `metadata-only`; `find_package`; `mismatch` |

If a future Windows workflow change adds another `shell: pwsh` step, the
validator now fails until that step is intentionally added to the selected
ownership list or moved out of the PowerShell validation surface.

## Drift Coverage

| Drift scenario | Expected failure |
| --- | --- |
| Selected step changes from `shell: pwsh` to another shell. | `must declare shell: pwsh` |
| Selected CMake configure command loses `cmake -S . -B build`. | `missing token 'cmake -S . -B build'` |
| A new unowned `shell: pwsh` step appears in Windows CI. | `windows workflow has unowned PowerShell steps` |
| Windows CI gains a selected report freshness command. | `windows workflow must not run or upload selected report freshness` |
| Sprint 182 deferral record is missing or loses marker text. | Deferral-record validation failure. |
| Selected report manifest lists `windows` while deferral is active. | Manifest Windows deferral validation failure. |

## Hosted/Local Distinction

Day 5 does not change the local or hosted exit-code contract:

- local/default mode still reports exit `2` when structural checks pass but
  `pwsh` is unavailable;
- hosted/fail-closed mode with `--require-pwsh` still returns exit `1` when
  `pwsh` is unavailable;
- no selected report generators, report uploads, CMake commands, or CTest
  commands are executed by the validator unless a future hosted parse path has
  actual `pwsh` available for syntax parsing only.

## Retained Non-Claims

Day 5 preserves these non-claims:

- Windows report freshness;
- selected Windows report artifact publication;
- broad Windows platform parity;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library package support;
- dynamic ABI or runtime-loader behavior.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Current workflow ownership and drift checks behave as expected without local `pwsh`. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural checks pass, then local missing `pwsh` is reported as unavailable evidence. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report manifest still keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects Windows selected report freshness commands/uploads. |

## Day 6 Handoff

Day 6 should deepen report artifact and manifest validation by deriving or
checking selected artifact names from `tests/corpus/manifests/selected_report_targets.tsv`
and ensuring the PowerShell validation owner remains aligned with report guard
assumptions.

## Validation Scope

Day 5 changed Python tests, a Python script, and planning documentation. No
`.c` or `.h` files were modified, so `make format && make lint && make test`
is not required.
