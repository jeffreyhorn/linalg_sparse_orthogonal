# Sprint 189 Day 7: Local PowerShell Parse Path

## Purpose

Exercise and harden the local PowerShell validation path so the owned Windows
workflow validator has deterministic behavior when `pwsh` is present and when
it is unavailable.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Flushed pass/fail/unavailable diagnostics so redirected maintainer and CI logs preserve validation order. |
| `scripts/validate_windows_powershell.py` | Cleaned workflow job block line slicing while preserving the existing structural scan semantics. |
| `tests/test_validate_windows_powershell.py` | Added fake-`pwsh` coverage for parse success, parse failure, and full `main()` available-path execution. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 7 available/unavailable behavior evidence. |

## Local Availability Result

This local environment still has no `pwsh` executable on `PATH`.

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural checks pass, then local PowerShell is reported unavailable. |
| `make windows-powershell-validate` | Expected exit `2` | The Make target exposes the same local unavailable evidence. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode refuses to treat missing `pwsh` as success. |

The unavailable output remains explicit:

```text
windows-powershell-validate: UNAVAILABLE: pwsh not found; structural checks passed
windows-powershell-validate: local unavailable PowerShell is not pass evidence
```

## Fake PowerShell Coverage

Day 7 adds deterministic tests that do not require PowerShell to be installed:

| Test | Evidence |
| --- | --- |
| `test_parse_with_fake_pwsh_accepts_selected_snippets` | Confirms the validator invokes `pwsh -NoProfile -NonInteractive -Command ...`, provides `SPARSE_PWSH_SNIPPET`, and parses all five selected snippets. |
| `test_parse_with_fake_pwsh_failure_is_actionable` | Confirms a parse subprocess failure raises a `PowerShell parse failed ...` diagnostic tied to the selected workflow step. |
| `test_main_with_fake_pwsh_returns_pass` | Confirms both local default mode and hosted `--require-pwsh` mode return success when a `pwsh` executable is available and snippets parse. |

These tests exercise the validator's process and exit-code control flow without
executing CMake, CTest, report generators, uploads, or generated report
freshness commands.

## Output Stability

The validator now flushes each pass, fail, and unavailable diagnostic. This
keeps local redirected logs and hosted CI logs in execution order even when
stdout and stderr are captured together.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Fake available, fake failure, and local unavailable paths behave as expected. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point reports unavailable evidence locally. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted mode fails closed when `pwsh` is absent. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 8 Handoff

Day 8 can wire the validation command into hosted Windows CI using
`python3 scripts/validate_windows_powershell.py --require-pwsh`. The Day 7
available-path tests verify that hosted mode succeeds when `pwsh` is present
and fails closed when it is not.

## Validation Scope

Day 7 changed Python tests, a Python script, and planning documentation. No
`.c` or `.h` files were modified, so `make format && make lint && make test`
is not required.
