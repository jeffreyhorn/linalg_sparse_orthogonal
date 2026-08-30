# Sprint 189 Day 3: Validation Command Design

## Purpose

Define the owned PowerShell validation command contract before implementation.
The design must support local maintainer use, hosted Windows CI execution,
clear unavailable-state reporting, and strict separation from Windows report
freshness promotion.

## Command Shape Decision

Use a Python validation script with a Make target wrapper:

| Layer | Selected name | Purpose |
| --- | --- | --- |
| Script | `python3 scripts/validate_windows_powershell.py` | Own structural workflow/report checks and PowerShell parse validation. |
| Make target | `make windows-powershell-validate` | Provide the stable maintainer entry point used by docs and local validation. |
| Hosted CI invocation | `python3 scripts/validate_windows_powershell.py --require-pwsh` | Fail closed on hosted Windows if `pwsh` is missing or parsing fails. |

This shape matches existing project patterns: Python owns structured manifest
and workflow assertions, while Make exposes stable maintainer commands.

## Command Inputs

| Input | Use |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Extract and validate selected `shell: pwsh` run blocks. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Confirm `windows` remains absent while Sprint 182 deferral is active. |
| `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` | Confirm active deferral marker text exists. |
| `tests/test_selected_comparison_workflow.py` expectations | Reuse forbidden selected freshness command and artifact names where practical. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Later documentation surfaces for the command and retained non-claims. |

## Local Behavior

| Local condition | Exit | Required output |
| --- | ---: | --- |
| `pwsh` exists and all structural/parse checks pass | `0` | `windows-powershell-validate: passed` plus parsed snippet count. |
| `pwsh` does not exist and structural checks pass | `2` | `windows-powershell-validate: UNAVAILABLE: pwsh not found` plus wording that this is not pass evidence. |
| Structural checks fail | `1` | `windows-powershell-validate: FAIL: ...` with the drift reason. |
| `pwsh` exists but parsing fails | `1` | `windows-powershell-validate: FAIL: PowerShell parse failed ...`. |

Local exit `2` is acceptable unavailable evidence for Sprint 189. It must not
be described as a pass, and Day 12/Day 13 must record it separately from
hosted Windows evidence.

## Hosted Windows Behavior

Hosted Windows CI should call:

```sh
python3 scripts/validate_windows_powershell.py --require-pwsh
```

With `--require-pwsh`:

- missing `pwsh` is a failure, not an unavailable local skip;
- parse failures fail the hosted job;
- selected report freshness commands and uploads remain forbidden;
- selected report manifest `workflow_platforms` must keep `windows` absent;
- the command validates ownership only and does not generate reports.

## PowerShell Parse Strategy

The script should parse selected workflow `run:` blocks without executing the
workflow commands:

1. Extract `run` text and adjacent `shell: pwsh` declaration for selected
   Windows steps.
2. Write each selected PowerShell snippet to a temporary file.
3. Invoke `pwsh` in non-interactive/no-profile mode to call
   `[scriptblock]::Create(...)` on the file contents.
4. Treat parse errors as validation failures.
5. Do not run `cmake`, `ctest`, report generators, upload steps, or generated
   artifact commands from the parse validation.

## Required Structural Checks

The script should fail if:

- `.github/workflows/windows-ci.yml` is missing;
- the workflow loses the Sprint 182 deferral comment;
- the `build-and-test` or `install-and-downstream` job is missing;
- selected Windows steps lose `shell: pwsh`;
- selected Windows jobs stop using `windows-2022`;
- selected report freshness commands appear in Windows CI;
- selected report artifact names appear in Windows CI upload context;
- any selected manifest row lists `windows` while the Sprint 182 deferral is
  active;
- the deferral artifact loses its marker text.

## Selected Step Anchors

Day 4 should start with these selected PowerShell step anchors:

| Job | Step anchor | Required tokens |
| --- | --- | --- |
| `build-and-test` | `Run enforced reviewed CMake configure path` | `cmake -S . -B build`, `Visual Studio 17 2022`, `shell: pwsh` |
| `build-and-test` | `Run enforced reviewed CMake build path` | `cmake --build build`, `Release`, `shell: pwsh` |
| `build-and-test` | `Inspect enforced Windows reviewed consumer CTest surface` | `EXPECTED_WINDOWS_CTEST_COUNT`, `Total Tests:`, `shell: pwsh` |
| `build-and-test` | `Run enforced reviewed CMake execution path` | `ctest --test-dir build`, `--output-on-failure`, `shell: pwsh` |
| `install-and-downstream` | `Run reviewed CMake install/downstream validation proof` | `sparse_lu_ortho.lib`, `sparse.pc`, `metadata-only`, `find_package`, `mismatch`, `shell: pwsh` |

## Report Freshness Boundary

The command must not run or promote:

- `make report-index-oracle-freshness`;
- `make report-index-comparison-freshness`;
- `make bench-canonical-report-freshness`;
- `scripts/check_bench_canonical_freshness.py`;
- selected Windows upload artifacts;
- broad report-index freshness;
- Windows generated report publication.

## Documentation Contract

Maintainer docs should eventually describe:

- `make windows-powershell-validate`;
- direct script invocation for hosted CI;
- exit `0` as parsed/guarded validation success;
- exit `2` as local unavailable evidence only;
- exit `1` as validation failure;
- hosted Windows requirement to fail closed;
- retained Windows report freshness non-claims.

## Day 4 Implementation Checklist

Day 4 should:

1. Add `scripts/validate_windows_powershell.py`.
2. Add `make windows-powershell-validate`.
3. Extract selected Windows workflow PowerShell run blocks.
4. Implement structural checks that do not require `pwsh`.
5. Implement parse checks that run only when `pwsh` is available or required.
6. Emit stable `PASS`, `FAIL`, and `UNAVAILABLE` diagnostics.
7. Run existing schema, manifest, and workflow guards after implementation.

## Validation Scope

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
