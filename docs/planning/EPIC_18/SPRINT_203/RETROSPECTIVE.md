# Sprint 203 Retrospective

**Sprint:** 203 - Windows QR Incompatible Comparison Promotion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-203`)
**Status:** Closed for implementation, validation, claim calibration, and
explicit re-deferral of Windows QR incompatible selected comparison freshness
promotion

## Source Artifact Note

Sprint 203 was executed from the Epic 18 project-plan section for Sprint 203
and lives under `docs/planning/EPIC_18/SPRINT_203/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint evaluated whether the existing selected QR incompatible comparison
target, `SRT-COMP-QR-INCOMPATIBLE-LS` / `qr-incompatible-ls`, could be promoted
to Windows selected comparison freshness. The work records intake, MSVC probe
design, local probe evidence, generator and CMake fix review, selected
generator validation, artifact-path and row-filtering hardening, manifest and
workflow re-deferral decisions, workflow guards, normalizer diagnostics,
documentation calibration, integrated validation, review hardening, and final
closeout.

## Definition Of Done Checklist

- [x] Created Sprint 203 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Mapped Sprint 203 items 203.1 through 203.6 to owner surfaces, evidence
      requirements, validation commands, and non-goals.
- [x] Defined the exact Windows/MSVC/CMake probe contract for the selected
      `qr-incompatible-ls` target.
- [x] Regenerated local QR incompatible comparison artifacts and confirmed the
      six selected generated rows remain fresh when generated rows are included
      explicitly.
- [x] Added normalizer regression coverage for Windows-style QR artifact paths,
      near-match rejection, duplicate rows, unexpected rows, stale rows,
      dependency-only rows, and wrong-target diagnostics.
- [x] Added manifest and workflow guard coverage that keeps QR incompatible
      Windows promotion absent until hosted Windows/MSVC proof exists.
- [x] Calibrated README, INSTALL, maintainer guide, corpus README, and
      report-index schema docs so they preserve the QR incompatible Windows
      re-deferral boundary.
- [x] Updated the Epic 18 project-plan status and residual queue with the
      re-deferred disposition and future evidence requirements.
- [x] Ran focused generator, normalizer, manifest, workflow, docs,
      Windows/PowerShell, Python syntax, QR executable, and whitespace checks.
- [x] Confirmed `.github/workflows` and selected target manifest files remain
      unchanged, so no Windows QR incompatible promotion occurred.

## What Went Well

1. **The sprint did not force promotion without evidence.** The plan allowed
   promotion only if MSVC/CMake generation, hosted artifacts, manifest
   metadata, and docs supported it. The closeout kept that bar intact and
   re-deferred the Windows claim.

2. **Local QR evidence became clearer.** The selected generator can regenerate
   the six-file QR incompatible artifact bundle locally, and the normalizer
   reports the six selected rows fresh when generated-row inclusion is
   explicit.

3. **Windows path handling got sharper tests.** Backslash, mixed separator,
   absolute Windows path, and near-match cases now have selected QR
   incompatible coverage.

4. **Workflow and manifest promotion are guarded.** The workflow guard rejects
   accidental Windows `qr-incompatible-ls` commands, equals-form target
   commands, freshness checks, artifact names, QR subfamily tokens, and upload
   paths. The manifest guard keeps the selected target Linux/macOS-only and
   local-only for Windows purposes.

5. **Documentation moved with the decision.** Public, maintainer, corpus, and
   schema docs now describe the Windows QR incompatible boundary as a
   re-deferral, not an implied Windows selected freshness claim.

6. **Residual ownership improved.** E18-RQ-006 now names the exact local
   evidence delivered and the hosted Windows/MSVC proof and artifact review
   still required before future promotion.

## What Didn't Go Well

1. **Hosted Windows proof remained unavailable during branch work.** Local
   evidence and simulations were useful, but they cannot prove hosted
   Windows/MSVC behavior or artifact upload semantics.

2. **Generated-row freshness needed explicit invocation discipline.** The final
   closeout command had to include `--include-generated` so selected generated
   rows were actually loaded and checked.

3. **Several claim surfaces had to be kept synchronized.** README, INSTALL,
   maintainer docs, corpus docs, schema docs, validator markers, project-plan
   status, and residual queue wording all needed aligned non-claim language.

4. **The sprint title was aspirational.** The right outcome was not promotion;
   it was a well-supported re-deferral with better guard infrastructure.

## Final Metrics

### Validation

| Metric | Sprint 203 close state |
| --- | --- |
| local `qr-incompatible-ls` generator | passed on Days 3, 5, 12, and 14 |
| selected generated-row freshness | passed on Days 3, 5, 6, 10, 12, and 14 with generated rows included |
| external comparison runner tests | passed on Days 5, 12, and 14 |
| normalizer selected-target tests | passed on Days 6, 10, 12, 13, and 14 |
| selected report target manifest tests | passed on Days 8, 12, 13, and 14 |
| selected comparison workflow tests | passed on Days 9, 12, 13, and 14 |
| selected performance docs guard | passed on Day 14 as compatibility coverage |
| Windows/PowerShell validation tests | passed on Days 11, 12, 13, and 14 |
| Python syntax check | passed on Days 12, 13, and 14 |
| focused QR corpus executable | passed on Days 12 and 14 |
| focused QR solve executable | passed on Days 12 and 14 |
| workflow and selected manifest promotion diff | empty on Days 13 and 14 |
| final `git diff --check` | passed |
| final full C quality gate | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 203 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Corpus/schema documentation files changed | 2 |
| Windows claim-boundary validator files changed | 1 |
| Python validation or guard test files changed | 3 |
| Epic project-plan files changed | 1 |
| Epic residual queue files changed | 1 |
| CI workflow files changed | 0 |
| Selected target manifest files changed | 0 |
| Production C implementation files changed | 0 |
| Public or internal C header files changed | 0 |
| Makefile targets changed | 0 |
| CMake registration files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| MSVC probe contract recorded | 1 |
| Local QR incompatible evidence delivered | 1 |
| Generator/path implementation broadened | 0 |
| Manifest promotion completed | 0 |
| Manifest promotion explicitly re-deferred | 1 |
| Normalizer and workflow guard coverage completed | 1 |
| Documentation calibration completed | 1 |
| Focused validation completed | 1 |
| Windows QR incompatible selected freshness claims promoted | 0 |
| Broad Windows report freshness claims promoted | 0 |

The count covers Sprint 203 items 203.1 through 203.6.

## Closed Claim

Sprint 203 closes this bounded claim:

The current branch strengthens the proof path for future Windows QR
incompatible selected comparison freshness promotion, but explicitly
re-defers the promotion itself. The existing selected target
`SRT-COMP-QR-INCOMPATIBLE-LS` remains a Linux/macOS selected comparison target
with local-only Windows interpretation. Local `qr-incompatible-ls` generation
passes, selected generated-row freshness passes with explicit
`--include-generated`, normalizer diagnostics cover Windows-style artifact
paths and selected row mismatches, workflow guards reject accidental Windows QR
promotion, manifest guards preserve the non-Windows metadata boundary, and
public/maintainer/corpus documentation records the retained Windows
non-claim.

This claim does not include Windows QR incompatible selected freshness, broad
Windows report freshness, broad QR parity, broad least-squares parity, raw QR
basis identity, Q sign or orientation policy, global rank-threshold policy,
NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, package-manager proof,
shared-library ABI proof, performance superiority, release readiness, platform
parity, or state-of-the-art sparse linear algebra performance.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-qr-intake.md](./artifacts/day1-windows-qr-intake.md);
- [day2-msvc-probe-design.md](./artifacts/day2-msvc-probe-design.md);
- [day3-probe-execution-record.md](./artifacts/day3-probe-execution-record.md);
- [day4-generator-cmake-fix-design.md](./artifacts/day4-generator-cmake-fix-design.md);
- [day5-selected-generator-fixes.md](./artifacts/day5-selected-generator-fixes.md);
- [day6-artifact-path-row-filtering-tests.md](./artifacts/day6-artifact-path-row-filtering-tests.md);
- [day7-manifest-promotion-decision.md](./artifacts/day7-manifest-promotion-decision.md);
- [day8-manifest-workflow-metadata.md](./artifacts/day8-manifest-workflow-metadata.md);
- [day9-workflow-guard-integration.md](./artifacts/day9-workflow-guard-integration.md);
- [day10-normalizer-freshness-diagnostics.md](./artifacts/day10-normalizer-freshness-diagnostics.md);
- [day11-documentation-calibration.md](./artifacts/day11-documentation-calibration.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Windows QR incompatible selected freshness promotion | Future Windows comparison owner | Run `qr-incompatible-ls` under the reviewed hosted Windows/MSVC path, inspect uploaded artifacts, update selected manifest Windows metadata, update workflow upload path, and calibrate docs together. |
| Hosted Windows/MSVC generated evidence | Future CI evidence owner | Capture successful hosted Windows generator output, generated `study.tsv`, manifest metadata, and selected artifact bundle. |
| Hosted Windows QR artifact inspection | Future report freshness owner | Review the actual uploaded artifact contents before treating the lane as selected Windows freshness evidence. |
| Broad Windows report freshness remains unclaimed | Future Windows report owner | Promote only selected rows with exact manifests, workflow guards, and documentation boundaries before broadening any Windows freshness wording. |
| Package-manager and shared-library ABI proof remain unclaimed | Future packaging owner | Add package recipes, install evidence, ABI policy, and explicit support wording before claiming package or ABI support. |
| State-of-the-art sparse linear algebra performance remains unclaimed | Future evidence owner | Provide representative external comparisons, methodology-bound benchmark evidence, and claim-reviewed documentation. |

## Next-Sprint Readiness

Sprint 203 leaves a safer future promotion path for Windows QR incompatible
selected comparison freshness.

| Future need | Sprint 203 handoff |
| --- | --- |
| Hosted Windows proof | Use the Day 2 probe contract and Day 14 residual path as the future execution checklist. |
| Generated-row freshness | Include generated rows explicitly when checking selected generated comparison artifacts. |
| Manifest promotion | Do not add Windows platform metadata for `SRT-COMP-QR-INCOMPATIBLE-LS` until hosted proof and artifact inspection exist. |
| Workflow promotion | Keep `.github/workflows/windows-ci.yml` free of QR incompatible commands and uploads until the selected promotion package is ready. |
| Documentation maintenance | Preserve the re-deferral markers across README, INSTALL, maintainer guide, corpus README, and report-index schema docs. |
| Validator maintenance | Keep normalizer, manifest, workflow, and Windows/PowerShell guard tests together whenever selected comparison metadata changes. |

## Final Assessment

Sprint 203 is complete as a claim-safe Windows QR incompatible comparison
promotion review and re-deferral sprint. It delivers local selected QR
incompatible evidence, sharper generated-row freshness checks, Windows path and
row diagnostics, workflow and manifest guard coverage, calibrated
documentation, and updated project-plan/residual tracking. It does not promote
Windows QR incompatible selected freshness because the required hosted
Windows/MSVC proof and hosted artifact inspection are still absent.
