# Sprint 182 Retrospective

**Sprint:** 182 - Windows Report Freshness Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-182`)
**Status:** Complete

## Source Artifact Note

Sprint 182 was executed from the Epic 16 project-plan section for Sprint 182
and lives under `docs/planning/EPIC_16/SPRINT_182/` with its plan, working
notes, daily artifacts, closeout artifact, formal deferral decision record, and
retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 182 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited the inherited Sprint 181 selected-target manifest authority and
      the existing Linux/macOS selected report freshness lanes.
- [x] Audited the Windows workflow, CMake/MSVC toolchain surface, PowerShell
      assumptions, install/downstream validation, and current non-claims.
- [x] Audited selected oracle, comparison, and benchmark freshness commands for
      Windows shell, path, executable, probe-build, artifact, and runtime risks.
- [x] Selected formal Windows report freshness deferral as the Sprint 182
      product decision.
- [x] Added the formal Windows report freshness deferral record with supported
      claims, unsupported claims, blockers, guard contract, and revisit
      criteria.
- [x] Hardened workflow and manifest guards so accidental Windows selected
      freshness commands, selected upload artifact names, or selected
      `workflow_platforms` fail clearly.
- [x] Aligned corpus/report-index docs, README, maintainer guide, and Windows
      workflow comments with the formal deferral.
- [x] Preserved package/install, shared-library, dynamic ABI, runtime-loader,
      package-manager, broad platform, performance, external-library, release,
      and state-of-the-art non-claims.
- [x] Ran focused schema, manifest, workflow, normalizer, package deferral, and
      whitespace validation.
- [x] Confirmed no C/header files or generated build/report/API artifacts were
      included in the changed surface.

## What Went Well

1. **The sprint closed the product decision cleanly.** Sprint 182 did not
   partially promote Windows report freshness. It evaluated the candidate paths
   and chose a formal deferral backed by exact blockers and future revisit
   criteria.

2. **The guard boundary became enforceable.** The workflow guard rejects
   selected report freshness commands and selected upload artifact names in the
   Windows workflow, while manifest tests reject accidental `windows`
   `workflow_platforms` while the deferral is active.

3. **The selected-target manifest stayed honest.** The sprint preserved
   `tests/corpus/manifests/selected_report_targets.tsv` as positive selected
   target authority rather than adding fake deferral rows.

4. **Documentation now matches the implementation.** README, maintainer guide,
   corpus docs, report-index schema docs, and Windows workflow comments all
   describe the same boundary: Windows CMake/install validation is reviewed,
   but Windows generated report freshness is not promoted.

5. **Failure diagnostics improved.** New tests name the missing Windows job,
   missing deferral wording, selected command, selected upload artifact,
   missing required upload file, manifest row, or deferral blocker that needs
   repair.

6. **Validation covered the actual decision.** The final sweep exercised schema
   validation, selected manifest diagnostics, workflow guard diagnostics,
   report-index regression tests, freshness diagnostics, package deferral
   guards, and whitespace hygiene without requiring unrelated C checks.

## What Didn't Go Well

1. **Windows report freshness remains unimplemented.** The sprint resolves the
   product ambiguity, but users still do not have selected generated report
   freshness on Windows.

2. **The strongest future candidate still needs build-system work.** Selected
   comparison freshness has the best artifact shape, but current probes still
   assume Unix compiler/linker behavior, `.a` archives, `-lm`, extensionless
   executables, and Makefile fallback.

3. **Local PowerShell validation was unavailable.** `pwsh` was not installed in
   the local environment, so PowerShell parsing was recorded as a validation
   gap. Hosted Windows execution remains owned by GitHub Actions on
   `windows-2022`.

4. **Some guards remain text based.** String guards are pragmatic for a formal
   deferral, but a future Windows promotion should move toward explicit
   allowlists for exact workflow jobs, commands, and artifacts.

5. **Freshness diagnostics still show local generated rows as stale or
   advisory.** This is acceptable for the deferral path, but maintainers must
   not confuse non-required local diagnostics with Windows freshness evidence.

## Final Metrics

### Validation

| Metric | Sprint 182 close state |
| --- | --- |
| Python syntax for touched schema/normalizer/tests | passed |
| corpus schema validation | passed: `python3 scripts/validate_corpus_schema.py` |
| selected target manifest diagnostics | passed: `python3 tests/test_selected_report_targets_manifest.py` |
| selected comparison workflow guard diagnostics | passed: `python3 tests/test_selected_comparison_workflow.py` |
| report-index regression tests | passed: `python3 tests/test_normalize_report_index.py` |
| corpus/oracle normalizer construction | passed: `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` |
| oracle freshness diagnostics | passed with expected stale local oracle warnings |
| comparison freshness diagnostics | passed with advisory local comparison diagnostics |
| coverage/dead-code/package freshness diagnostics | passed with advisory absent local report diagnostics |
| static package deferral guard | passed |
| package-manager deferral guard | passed |
| local PowerShell parse check | not run: `pwsh` unavailable locally |
| final `git diff --check` | passed |
| C source/header quality gate | not required: no `*.c` or `*.h` files changed |

### Changed Surface

| Metric | Sprint 182 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 0 |
| generated build/report/API artifacts staged | 0 |
| formal decision records added | 1 |
| workflow guard tests changed | 1 |
| selected manifest tests changed | 1 |
| public/maintainer docs changed | 3 |
| corpus/report-index docs changed | 2 |
| workflow files changed | 1 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |

### Claim Governance

| Metric | Sprint 182 close state |
| --- | ---: |
| Windows report freshness promotions added | 0 |
| Windows selected oracle freshness claims added | 0 |
| Windows selected comparison freshness claims added | 0 |
| Windows selected benchmark freshness claims added | 0 |
| Windows selected manifest platform rows added | 0 |
| formal Windows report freshness deferrals added | 1 |
| Windows workflow selected command guards | 3 |
| Windows workflow selected artifact guards | 4 |
| package-manager support claims added | 0 |
| shared-library/dynamic ABI/runtime-loader claims added | 0 |
| broad Windows parity claims added | 0 |
| portable performance or state-of-the-art claims added | 0 |

## Closed Claim

Sprint 182 closes this Epic 16 Windows report freshness claim:

Windows report freshness is formally deferred. The reviewed Windows support
surface remains CMake/MSVC configure, build, `ctest`, and static-first CMake
install/downstream validation. Windows does not prove selected oracle,
comparison, benchmark, or broad generated report freshness.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [windows-report-freshness-deferral-decision.md](./artifacts/windows-report-freshness-deferral-decision.md);
- [day1-windows-freshness-scope-intake.md](./artifacts/day1-windows-freshness-scope-intake.md);
- [day2-windows-workflow-and-toolchain-audit.md](./artifacts/day2-windows-workflow-and-toolchain-audit.md);
- [day3-report-command-compatibility-audit.md](./artifacts/day3-report-command-compatibility-audit.md);
- [day4-artifact-and-data-semantics-audit.md](./artifacts/day4-artifact-and-data-semantics-audit.md);
- [day5-candidate-decision-matrix.md](./artifacts/day5-candidate-decision-matrix.md);
- [day6-decision-record-design.md](./artifacts/day6-decision-record-design.md);
- [day7-implementation-batch-1.md](./artifacts/day7-implementation-batch-1.md);
- [day8-implementation-batch-2.md](./artifacts/day8-implementation-batch-2.md);
- [day9-manifest-and-support-tier-alignment.md](./artifacts/day9-manifest-and-support-tier-alignment.md);
- [day10-documentation-alignment.md](./artifacts/day10-documentation-alignment.md);
- [day11-guard-and-failure-diagnostics-hardening.md](./artifacts/day11-guard-and-failure-diagnostics-hardening.md);
- [day12-validation-sweep.md](./artifacts/day12-validation-sweep.md);
- [day13-decision-reconciliation.md](./artifacts/day13-decision-reconciliation.md);
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md).

No Windows selected report freshness, broad generated report freshness,
Makefile parity, Bash/POSIX report generation support, package-manager
support, shared-library support, dynamic ABI support, runtime-loader behavior,
broad Windows parity, portable performance, release readiness,
external-library parity, or state-of-the-art claim was added.

## Sprint 183 Readiness

Sprint 183 should start from the next Epic 16 project-plan section. If it
revisits Windows report freshness, it should begin with selected comparison
freshness as the most plausible candidate and require:

| Future need | Sprint 182 handoff |
| --- | --- |
| Windows-safe generator path | Avoid Makefile and Bash assumptions; invoke the selected generator through a reviewed Windows-safe command. |
| CMake/MSVC probe support | Replace Unix `cc`, `.a`, `-lm`, extensionless executable, and fallback `make` assumptions. |
| Hosted Windows executable proof | Prove exact Python and probe execution behavior on `windows-2022`. |
| Selected upload scope | Add exact selected artifact paths with `if-no-files-found: error`; avoid broad `build/**` uploads. |
| Manifest promotion | Add Windows workflow file, job, artifact, platform, support tier, claim scope, and non-claims atomically. |
| Guard evolution | Replace deferral-only rejection with an exact allowlist for the promoted Windows path. |
| Documentation alignment | Update README, INSTALL if package wording changes, maintainer guide, workflow comments, corpus docs, and report-index schema docs together. |
| Validation | Run selected manifest tests, workflow guards, report-index tests, freshness checks, package deferral guards if support wording changes, PowerShell checks where available, and `git diff --check`. |
