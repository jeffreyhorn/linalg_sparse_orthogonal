# Sprint 189 Retrospective

**Sprint:** 189 - PowerShell Validation Ownership
**Duration:** 14 days (Days 1-14 landed on branch `sprint-189`)
**Status:** Complete with hosted evidence pending PR CI

## Source Artifact Note

Sprint 189 was executed from the Epic 17 project-plan section for Sprint 189
and lives under `docs/planning/EPIC_17/SPRINT_189/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 189 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited Windows PowerShell workflow snippets, report-adjacent surfaces,
      selected report metadata, selected artifact names, local tool
      availability, and retained non-claims.
- [x] Designed an owned local/hosted validation command with explicit local
      unavailable semantics and hosted fail-closed behavior.
- [x] Added `scripts/validate_windows_powershell.py` and
      `make windows-powershell-validate`.
- [x] Added hosted Windows CI wiring through the dedicated
      `powershell-validation` job on `windows-2022`.
- [x] Added guard coverage for selected PowerShell workflow ownership,
      hosted validation wiring, selected report non-promotion,
      manifest-derived artifact blockers, documentation claim boundaries, fake
      `pwsh` parse behavior, parse failures, and local unavailable wording.
- [x] Updated README, INSTALL, maintainer guide, and corpus README to explain
      PowerShell validation ownership without promoting Windows report
      freshness or broad Windows parity.
- [x] Ran the integrated Windows-adjacent validation set, docs checks,
      report/schema checks, hygiene scans, and C/header gate applicability
      review.
- [x] Recorded final claim audit, residuals, Sprint 190 handoff, and PR-ready
      notes.
- [x] Preserved explicit non-claims for Windows report freshness, selected
      Windows report artifact publication, Windows report generator execution,
      Windows Makefile parity, Windows `pkg-config` execution parity,
      package-manager support, shared-library support, dynamic ABI support,
      DLL/import-library support, runtime-loader support, broad Windows
      parity, portable performance, and state-of-the-art evidence.

## What Went Well

1. **The sprint kept validation ownership separate from report freshness.**
   The final implementation validates selected Windows PowerShell workflow
   material and explicitly keeps selected report generation, selected upload
   artifacts, and manifest `windows` platform promotion out of scope.

2. **The command contract is reviewable.** `make windows-powershell-validate`
   gives maintainers one entry point, while
   `python3 scripts/validate_windows_powershell.py --require-pwsh` gives
   hosted Windows CI fail-closed behavior.

3. **Local absence is explicit instead of silent.** On this machine, missing
   local `pwsh` returns exit `2` after structural checks and prints that local
   unavailable PowerShell is not pass evidence.

4. **The available path is tested without requiring PowerShell locally.**
   Fake-`pwsh` tests exercise the selected snippet parse path, subprocess
   failure diagnostics, and full `main()` available behavior in both default
   and `--require-pwsh` modes.

5. **Workflow and docs drift are guarded together.** The validator now checks
   hosted job wiring, selected artifact non-promotion, selected manifest
   references, Sprint 182 deferral markers, and claim-boundary anchors across
   README, INSTALL, maintainer guide, and corpus README.

6. **The public support wording is better calibrated.** Windows now claims
   reviewed CMake build/test, reviewed CMake install/downstream validation,
   and hosted PowerShell validation ownership for selected workflow snippets,
   while report freshness and broader platform claims remain excluded.

## What Didn't Go Well

1. **Hosted pass evidence cannot exist before PR CI.** The source-controlled
   `powershell-validation` job is wired, but observed hosted pass evidence
   must wait until the branch is pushed and CI runs.

2. **The local command still exits nonzero on this machine.** That is the
   intended local-unavailable contract because `pwsh` is not installed, but it
   remains a less convenient maintainer experience than a local parse pass.

3. **Claim-boundary markers are necessarily brittle.** Guarding exact docs
   anchors catches real drift, but future legitimate wording changes must
   update the validator markers deliberately.

4. **Windows report freshness remains unresolved.** Sprint 189 cleaned the
   validation ownership boundary, but Sprint 190 must still decide whether to
   promote one Windows-safe freshness lane or renew the deferral.

5. **The workflow parser is intentionally narrow.** The validator is tailored
   to the current Windows workflow shape rather than a general YAML parser, so
   major workflow restructuring will require validator updates.

## Final Metrics

### Validation

| Metric | Sprint 189 close state |
| --- | --- |
| focused PowerShell validator tests | passed |
| local validator default mode | expected exit `2`, local `pwsh` unavailable |
| Make validator target | expected exit `2`, local `pwsh` unavailable |
| hosted/fail-closed validator mode locally | expected exit `1`, local `pwsh` unavailable |
| corpus schema validation | passed |
| selected report target manifest validation | passed |
| selected comparison workflow guard | passed |
| normalized report-index tests | passed |
| normalized report-index check | passed, `112` rows |
| documentation/API check | passed, `make docs-check` |
| unsupported Windows/report claim scan | no matches |
| stale marker scan | no open blockers |
| final `git diff --check` | passed |
| trailing-whitespace scan | passed |
| Sprint 189 markdown link check | passed |
| generated docs/build status | no generated repo changes staged |
| full C quality gate | not required; no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 189 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Workflow files changed | 1 |
| Makefile targets added | 1 |
| Python scripts added | 1 |
| Python tests added | 1 |
| Public/maintainer/report docs changed | 4 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Selected manifest rows promoted to Windows | 0 |
| Selected Windows report artifacts published | 0 |

### Claim Governance

| Metric | Sprint 189 close state |
| --- | ---: |
| Windows report freshness claims added | 0 |
| selected Windows report artifact publication claims added | 0 |
| Windows report generator execution claims added | 0 |
| broad Windows parity claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI claims added | 0 |
| DLL/import-library support claims added | 0 |
| runtime-loader support claims added | 0 |
| portable performance claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 189 closes this bounded implementation claim:

PowerShell validation ownership for selected Windows workflow material is now
source-controlled through an owned validator, Make entry point, hosted
fail-closed Windows CI job, local unavailable semantics, fake-`pwsh` parse-path
tests, workflow/report/docs drift guards, and calibrated user/maintainer/report
documentation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-powershell-validation-intake.md](./artifacts/day1-powershell-validation-intake.md);
- [day2-powershell-surface-map.md](./artifacts/day2-powershell-surface-map.md);
- [day3-validation-command-design.md](./artifacts/day3-validation-command-design.md);
- [day4-local-command-scaffold.md](./artifacts/day4-local-command-scaffold.md);
- [day5-workflow-snippet-coverage.md](./artifacts/day5-workflow-snippet-coverage.md);
- [day6-report-artifact-guards.md](./artifacts/day6-report-artifact-guards.md);
- [day7-local-pwsh-path.md](./artifacts/day7-local-pwsh-path.md);
- [day8-hosted-windows-lane.md](./artifacts/day8-hosted-windows-lane.md);
- [day9-ownership-guard-tests.md](./artifacts/day9-ownership-guard-tests.md);
- [day10-maintainer-validation-docs.md](./artifacts/day10-maintainer-validation-docs.md);
- [day11-windows-claim-calibration.md](./artifacts/day11-windows-claim-calibration.md);
- [day12-integrated-windows-validation.md](./artifacts/day12-integrated-windows-validation.md);
- [day13-claim-audit-residual-decision.md](./artifacts/day13-claim-audit-residual-decision.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No Windows report freshness lane, selected Windows report artifact
publication, Windows report generator execution, broad Windows parity,
Windows Makefile parity, Windows `pkg-config` execution parity,
package-manager support, shared-library support, dynamic ABI guarantee,
DLL/import-library support, runtime-loader support, portable performance claim,
or state-of-the-art claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Local `pwsh` absent on this machine | Local developer environment | Install `pwsh` and rerun `make windows-powershell-validate` for exit `0`; keep exit `2` as accepted unavailable evidence on machines without PowerShell. |
| Hosted Windows validation pass evidence | Sprint 189 PR CI | Push the branch and observe the `powershell-validation` job passing on `windows-2022`. |
| Windows report freshness | Sprint 190 | Promote exactly one Windows-safe selected freshness lane with manifest/artifact/freshness/docs guards, or renew the formal deferral with stronger blockers and revisit criteria. |
| Selected Windows report artifact publication | Sprint 190 only if promotion is selected | Add exact selected upload scope, `if-no-files-found: error`, selected manifest metadata, and updated guards in the same reviewed change. |

## Next-Sprint Readiness

Sprint 190 can start from an owned PowerShell validation boundary instead of
an environment ambiguity. It should not reinterpret Sprint 189 as report
freshness evidence.

| Future need | Sprint 189 handoff |
| --- | --- |
| Windows report freshness promotion | Use the Sprint 187 gate: exactly one Windows-safe lane, exact manifest metadata, exact artifact scope, freshness guard, docs update, and hosted proof. |
| Renewed Windows deferral | Keep `windows` out of selected target platforms, keep Windows CI free of selected report generation/uploads, and update blocker/revisit criteria. |
| Hosted PowerShell validation evidence | Check PR CI for the `powershell-validation` job and cite only observed hosted pass logs. |
| Docs updates | Run `python3 tests/test_validate_windows_powershell.py` so claim-boundary markers remain synchronized with README, INSTALL, maintainer guide, and corpus README. |
| Workflow changes | Run `make windows-powershell-validate`; on machines without `pwsh`, record exit `2` and rely on hosted `--require-pwsh` CI for parse evidence. |

## Validation Retrospective

Sprint 189 changed workflow, Python script/test, Makefile, public docs,
maintainer docs, report-facing docs, and planning documentation but no C
source or public headers. The selected validation set was therefore:

```sh
python3 tests/test_validate_windows_powershell.py
python3 scripts/validate_windows_powershell.py
make windows-powershell-validate
python3 scripts/validate_windows_powershell.py --require-pwsh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --check
make docs-check
git diff --check
```

The default local validator and Make target are expected to exit `2` on this
machine because `pwsh` is absent. The hosted-mode command is expected to exit
`1` locally for the same reason because `--require-pwsh` fails closed. Any
future `.c` or `.h` change must run:

```sh
make format
make lint
make test
```

## Carry Forward

- Sprint 190 must choose one accepted state: promote one Windows-safe selected
  report freshness lane or renew the formal deferral.
- Do not cite local `pwsh` absence as success evidence.
- Do not cite Sprint 189 PowerShell validation ownership as selected Windows
  report freshness.
- Keep hosted CI evidence separate from source-controlled wiring until the
  `powershell-validation` job has passed on the pushed branch.
