# Sprint 190 Retrospective

**Sprint:** 190 - Windows Selected Report Freshness Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-190`)
**Status:** Complete with residual narrowed; hosted Windows evidence and
manifest promotion pending

## Source Artifact Note

Sprint 190 was executed from the Epic 17 project-plan section for Sprint 190
and lives under `docs/planning/EPIC_17/SPRINT_190/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 190 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited prior Windows report freshness deferrals, selected target
      manifest metadata, Windows workflow constraints, hosted PowerShell
      validation ownership, and selected comparison candidates.
- [x] Selected `cholesky-spd-tridiag-5` as the smallest credible Windows-safe
      selected report freshness candidate.
- [x] Added a hosted Windows `selected-comparison-freshness` workflow job on
      `windows-2022` with a bounded timeout.
- [x] Wired the selected workflow path through CMake/MSVC and a CMake-probe
      generator command without Makefile or Bash assumptions.
- [x] Added target-specific freshness validation for
      `--selected-target cholesky-spd-tridiag-5`.
- [x] Kept artifact publication narrow to the six selected Cholesky comparison
      files under `build/comparison/cholesky_spd_tridiag_5/`.
- [x] Hardened workflow, manifest, PowerShell, generator, and normalizer guard
      coverage for exact positive behavior and negative drift cases.
- [x] Updated README, INSTALL, maintainer guide, corpus docs, and schema docs
      to state the bounded workflow path without promoting broad Windows
      report freshness.
- [x] Renewed and narrowed `R186-WIN-REPORT-FRESHNESS` in the Epic 16 residual
      queue because hosted Windows evidence and manifest promotion remain
      pending.
- [x] Ran the final focused validation set and recorded explicit local
      unavailable semantics for `pwsh`.
- [x] Preserved explicit non-claims for broad Windows report freshness,
      Windows selected oracle freshness, Windows selected benchmark freshness,
      broad selected comparison freshness, Windows Makefile parity, Windows
      `pkg-config` execution parity, package-manager support, shared-library
      support, dynamic ABI support, runtime-loader behavior, broad Windows
      parity, portable performance, and state-of-the-art evidence.

## What Went Well

1. **Candidate selection became concrete.** The sprint moved
   `R186-WIN-REPORT-FRESHNESS` from a broad product/workflow ambiguity to one
   exact selected Cholesky comparison lane with known row IDs, required files,
   artifact name, workflow job, and freshness command.

2. **The workflow path is bounded and reviewable.** The new Windows job builds
   through CMake/MSVC, runs only `cholesky-spd-tridiag-5`, validates only that
   selected target, and uploads only the required Cholesky comparison bundle.

3. **Manifest promotion stayed disciplined.** The source selected-target
   manifest still omits `windows`, while tests model the future exact
   Cholesky-only metadata shape. That prevents workflow wiring from being
   mistaken for a completed support-tier promotion.

4. **Freshness validation became target-specific.** The normalizer can now
   check a selected generated comparison without requiring unrelated QR, LU,
   partial-SVD, oracle, benchmark, or broad report-index outputs.

5. **PowerShell unavailable semantics remained explicit.** Local absence of
   `pwsh` still exits nonzero after structural checks and is recorded as
   unavailable evidence, not hosted Windows pass evidence.

6. **Docs and tests guard the same claim boundary.** Public docs, maintainer
   docs, corpus docs, schema docs, workflow tests, manifest tests, and
   PowerShell validation all point at the same bounded Windows outcome.

## What Didn't Go Well

1. **Hosted Windows evidence is still outside the local sprint pass.** The job
   is source-controlled, but observed `windows-2022` pass logs require the
   branch to be pushed and CI to run.

2. **The residual could not be fully closed.** Because hosted evidence and
   selected manifest promotion remain pending, `R186-WIN-REPORT-FRESHNESS`
   had to be renewed and narrowed instead of closed.

3. **Local PowerShell validation remains inconvenient.** The Make target and
   default validator still exit `2` locally because `pwsh` is absent on this
   machine.

4. **Claim-boundary checks are intentionally brittle.** Exact docs markers and
   workflow anchors catch real drift, but legitimate wording or workflow
   restructuring must update the tests deliberately.

5. **The workflow cannot prove artifact semantics until CI runs.** Local
   CMake-probe generation verifies the script path, but not the hosted Windows
   artifact upload.

## Final Metrics

### Validation

| Metric | Sprint 190 close state |
| --- | --- |
| selected comparison workflow guard | passed |
| selected report target manifest validation | passed |
| corpus schema validation | passed |
| Windows PowerShell validator tests | passed |
| local Windows PowerShell Make target | expected exit `2`, local `pwsh` unavailable |
| normalized report-index tests | passed |
| external comparison generator tests | passed |
| local CMake-probe selected Cholesky generation | passed |
| target-specific selected Cholesky freshness | passed, six generated rows fresh |
| final `git diff --check` | passed |
| generated report output status | ignored under `build/` |
| full C quality gate | not required; no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 190 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Workflow files changed | 1 |
| Python scripts changed | 3 |
| Python tests changed | 4 |
| Public/maintainer/report docs changed | 5 |
| Residual queue files changed | 1 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Selected manifest rows promoted to Windows | 0 |
| Selected Windows report workflow jobs added | 1 |
| Selected Windows report artifacts configured | 1 |

### Claim Governance

| Metric | Sprint 190 close state |
| --- | ---: |
| bounded Windows selected Cholesky workflow paths added | 1 |
| broad Windows report freshness claims added | 0 |
| Windows selected oracle freshness claims added | 0 |
| Windows selected benchmark freshness claims added | 0 |
| broad selected comparison freshness claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI claims added | 0 |
| runtime-loader claims added | 0 |
| portable performance claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 190 closes this bounded implementation claim:

One hosted Windows selected Cholesky comparison workflow path is now
source-controlled for `cholesky-spd-tridiag-5`, with CMake/MSVC build steps,
CMake-probe generator support, target-specific freshness validation, exact
artifact upload scope, workflow/manifest/PowerShell guard coverage, local
generated evidence, and calibrated public/maintainer/report documentation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-freshness-intake.md](./artifacts/day1-windows-freshness-intake.md);
- [day2-candidate-lane-audit.md](./artifacts/day2-candidate-lane-audit.md);
- [day3-feasibility-probe.md](./artifacts/day3-feasibility-probe.md);
- [day4-decision-record-draft.md](./artifacts/day4-decision-record-draft.md);
- [day5-workflow-scaffold.md](./artifacts/day5-workflow-scaffold.md);
- [day6-manifest-metadata.md](./artifacts/day6-manifest-metadata.md);
- [day7-freshness-guard.md](./artifacts/day7-freshness-guard.md);
- [day8-hosted-integration.md](./artifacts/day8-hosted-integration.md);
- [day9-deterministic-tests.md](./artifacts/day9-deterministic-tests.md);
- [day10-claim-calibration.md](./artifacts/day10-claim-calibration.md);
- [day11-report-evidence.md](./artifacts/day11-report-evidence.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-final-claim-audit.md](./artifacts/day13-final-claim-audit.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No broad Windows report freshness lane, Windows selected oracle freshness,
Windows selected benchmark freshness, broad selected comparison freshness,
Windows Makefile parity, Windows `pkg-config` execution parity,
package-manager support, shared-library support, dynamic ABI guarantee,
runtime-loader support, portable performance claim, or state-of-the-art claim
was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| `R186-WIN-REPORT-FRESHNESS` | Hosted evidence plus selected manifest promotion review | Observe hosted `selected-comparison-freshness` pass on `windows-2022`; inspect artifact `sprint190-windows-selected-comparison-cholesky`; then either promote exactly `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to `windows` metadata or retain the staged boundary with refreshed blockers. |
| Local `pwsh` absent on this machine | Local developer environment | Install `pwsh` and rerun `make windows-powershell-validate` for exit `0`; keep exit `2` as accepted unavailable evidence on machines without PowerShell. |
| Source manifest Windows metadata | Selected report target manifest owner | Add `windows` only to the selected Cholesky row after hosted evidence and claim boundaries are reviewed together. |
| Hosted selected Cholesky artifact evidence | Sprint 190 PR CI | Confirm the Windows workflow uploads exactly the selected Cholesky comparison bundle and that freshness validation passed before upload. |

## Next-Sprint Readiness

Sprint 191 or a review follow-up can start from an implemented bounded
workflow path instead of an unresolved candidate decision.

| Future need | Sprint 190 handoff |
| --- | --- |
| Hosted Windows evidence review | Inspect PR CI for `selected-comparison-freshness` on `windows-2022` and cite only observed pass logs. |
| Manifest promotion | If hosted evidence passes, add `windows` only to `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` with exact workflow file, job, artifact, and row-count alignment. |
| Retained deferral | If hosted evidence is unavailable or fails, keep manifest `windows` metadata absent and carry the narrowed residual. |
| Docs updates | Continue saying one bounded workflow path exists; do not say reviewed Windows selected Cholesky freshness support until hosted evidence and manifest promotion land. |
| Workflow changes | Run selected workflow, manifest, normalizer, generator, and PowerShell validation tests together because their claim boundaries are coupled. |

## Validation Retrospective

Sprint 190 changed workflow, Python scripts/tests, public docs, maintainer
docs, report-facing docs, residual planning docs, and sprint planning
documentation but no C source or public headers. The selected validation set
was therefore:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_validate_windows_powershell.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
make windows-powershell-validate
git diff --check
git diff --name-only -- '*.c' '*.h'
git status --short --ignored build/comparison/cholesky_spd_tridiag_5
```

The Make target is expected to exit `2` on this machine because `pwsh` is
absent after structural checks pass. Any future `.c` or `.h` change must run:

```sh
make format
make lint
make test
```

## Carry Forward

- Treat Sprint 190 as bounded workflow wiring plus narrowed residual, not as
  completed Windows report freshness promotion.
- Review hosted `windows-2022` evidence before adding `windows` to the
  selected manifest row.
- Keep broad Windows report freshness, Windows selected oracle freshness, and
  Windows selected benchmark freshness as non-claims.
- Keep local generated report output under ignored `build/` paths unless a
  future sprint deliberately changes generated evidence policy.
