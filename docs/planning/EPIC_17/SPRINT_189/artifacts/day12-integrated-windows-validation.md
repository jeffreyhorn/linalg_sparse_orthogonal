# Sprint 189 Day 12: Integrated Windows-Adjacent Validation

## Purpose

Run the integrated validation set for the changed Windows PowerShell owner,
hosted workflow, selected report metadata guards, documentation claim
boundaries, and changed-surface hygiene before the Sprint 189 claim audit.

## Changed Surfaces Under Validation

| Surface | Validation relevance |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Hosted Windows PowerShell validation lane, selected PowerShell snippet ownership, selected report non-promotion. |
| `scripts/validate_windows_powershell.py` | Owned validator for Windows workflow, report metadata, docs claim boundaries, hosted wiring, and local/hosted `pwsh` behavior. |
| `tests/test_validate_windows_powershell.py` | Focused drift coverage for the owned validator and local/fake PowerShell behavior. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | Public, maintainer, and report-facing Windows/PowerShell claim calibration. |
| `docs/planning/EPIC_17/SPRINT_189/**` | Sprint plan, working notes, and Day 1 through Day 12 evidence artifacts. |

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 - <<'PY' ... ast.parse(...)` | Passed | PowerShell validator and focused test parse as Python. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Workflow ownership, hosted wiring, claim boundaries, fake PowerShell parse paths, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` is unavailable evidence only. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid and keeps Windows deferral. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected comparison workflow guard still rejects Windows selected freshness commands/uploads. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalized report-index semantics and non-claims remain valid. |
| `python3 scripts/normalize_report_index.py --check` | Passed, `112` rows | Current normalized report-index check succeeds without requiring generated freshness promotion. |
| `make docs-check` | Passed | Doxygen/API docs coverage check succeeds for public headers. |

## Local PowerShell Availability

This local environment still has no `pwsh` executable on `PATH`. The
validator therefore returns exit `2` in default local mode after structural,
report, docs, and hosted wiring checks pass:

```text
windows-powershell-validate: UNAVAILABLE: pwsh not found; structural checks passed
windows-powershell-validate: local unavailable PowerShell is not pass evidence
```

That unavailable result remains local environment evidence only. It does not
replace the hosted Windows `--require-pwsh` job, and it does not promote
Windows report freshness.

## Documentation and Generated Output Hygiene

| Check | Result |
| --- | --- |
| `git diff --check` | Passed. |
| Trailing whitespace scan over changed workflow, scripts, tests, docs, and Sprint 189 artifacts | Passed. |
| Sprint 189 markdown link check | Passed. |
| `docs/api/html` status after `make docs-check` | No repo changes left by generated docs. |

## C Gate Decision

No `.c` or `.h` files were modified during Sprint 189 through Day 12. The
full C quality gate, `make format && make lint && make test`, is therefore
not required by the sprint rule.

## Claim-Audit Readiness

The sprint can enter Day 13 claim audit with no unresolved local validation
failures. The only nonzero local results are expected contract outcomes caused
by missing local `pwsh`:

- default local validation and Make target: exit `2`, unavailable evidence;
- hosted/fail-closed mode: exit `1`, because `--require-pwsh` rejects missing
  local `pwsh`.

## Day 13 Handoff

Day 13 should decide the final claim state for Sprint 189: PowerShell
validation ownership is implemented and guarded locally, with hosted
fail-closed validation wired for Windows CI, while Windows report freshness
and selected Windows report artifact publication remain deferred.
