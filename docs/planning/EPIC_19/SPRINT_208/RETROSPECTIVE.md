# Sprint 208 Retrospective

**Sprint:** 208 - Selected Windows Cholesky Freshness Promotion  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-208`)  
**Status:** Closed with continued selected Windows Cholesky freshness
re-deferral and stronger guard coverage

## Source Artifact Note

Sprint 208 was executed from the Epic 19 project-plan section for Sprint 208
and lives under `docs/planning/EPIC_19/SPRINT_208/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint inspected current hosted Windows evidence for the exact
`cholesky-spd-tridiag-5` selected comparison path, selected continued
re-deferral after finding generated support-tier and non-claim contradictions,
strengthened manifest and PowerShell absence guards, added Cholesky-specific
Windows-path normalizer regressions, calibrated public and maintainer docs, and
closed with integrated validation. It did not promote selected Windows
Cholesky freshness or any broad Windows, package, ABI, performance, release,
external-library parity, or state-of-the-art claim.

## Definition Of Done Checklist

- [x] Created Sprint 208 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Inspected current hosted Windows Cholesky evidence: run `35731703320`,
      job `106758632567`, and artifact `10696020870`.
- [x] Verified the hosted artifact contained the exact six selected Cholesky
      files and passing selected Cholesky rows for the bounded workflow path.
- [x] Defined objective promotion versus re-deferral criteria before changing
      manifest, workflow, guard, or documentation surfaces.
- [x] Selected continued re-deferral because generated rows and summaries still
      record `local_only`, `no hosted CI proof`, and `no Windows report
      freshness`.
- [x] Strengthened selected manifest contract tests for the current
      Linux/macOS-only metadata and exact future Windows promotion
      prerequisites.
- [x] Strengthened Windows PowerShell validation so it owns exact Cholesky
      re-deferral metadata, workflow scope, artifact scope, and broad Windows
      non-claims.
- [x] Added Cholesky-specific normalizer regressions for duplicate and
      unexpected selected rows under Windows-style artifact paths.
- [x] Updated README, INSTALL, maintainer guide, corpus README, schema docs,
      and Epic 19 project-plan status with reviewed bounded workflow evidence
      and retained selected-freshness re-deferral.
- [x] Ran selected manifest, schema, workflow, PowerShell, normalizer,
      freshness, documentation, support-doc, Python syntax, stale-claim, and
      whitespace validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required by the sprint instruction.

## What Went Well

1. **The promotion decision stayed evidence-led.** Hosted Windows evidence was
   current and useful, but the sprint did not turn bounded workflow evidence
   into a selected freshness claim while generated rows still said
   `local_only` and `no Windows report freshness`.

2. **The absence of Windows selected metadata is now guarded more directly.**
   Manifest tests and PowerShell validation now enforce both the current
   re-deferred state and the exact metadata that a future promotion must move
   together.

3. **Normalizer coverage now mirrors the selected Cholesky risk.** Existing
   Windows-path and selected filtering coverage was extended with
   Cholesky-specific duplicate-row and unexpected-row regressions.

4. **Public and maintainer docs now agree on the boundary.** README, INSTALL,
   maintainer guide, corpus docs, schema docs, and project-plan status all say
   Sprint 208 reviewed bounded Windows Cholesky workflow evidence while
   keeping selected Windows freshness re-deferred.

5. **Validation matched the changed surface.** The sprint ran focused
   manifest, workflow, PowerShell, normalizer, corpus, docs, support-doc, and
   freshness validation without requiring unrelated C quality gates.

## What Didn't Go Well

1. **Hosted success was not enough for promotion.** The hosted job and artifact
   were clean, but generated support-tier and non-claim semantics still lag the
   desired selected Windows freshness model.

2. **Prior Sprint 199 evidence created drift risk.** The sprint had to
   distinguish inherited Sprint 199 evidence from the current Sprint 208 run
   before making any status decision.

3. **Windows support wording remains easy to overstate.** Documentation needed
   repeated calibration so "bounded workflow evidence" did not become
   "promoted selected Windows freshness" or broad Windows parity.

4. **Local PowerShell remains unavailable in this environment.** The guard
   suite correctly treats missing local `pwsh` as unavailable evidence while
   preserving hosted `--require-pwsh` ownership, but that distinction must stay
   visible in future reviews.

## Final Metrics

### Validation

| Metric | Sprint 208 close state |
| --- | --- |
| `make windows-powershell-guard` | passed; local missing `pwsh` remained an expected unavailable-evidence path |
| `python3 tests/test_normalize_report_index.py` | passed |
| `python3 tests/test_selected_report_targets_manifest.py` | passed |
| `python3 scripts/validate_corpus_schema.py` | passed |
| `python3 tests/test_selected_comparison_workflow.py` | passed |
| `python3 tests/test_validate_windows_powershell.py` | passed |
| Python syntax checks | passed for updated PowerShell, manifest, and normalizer test modules |
| `make docs-check` | passed |
| `make support-docs-guard` | passed |
| `make report-index-comparison-freshness` | passed with freshness ok for 46 rows |
| stale and overbroad claim search | passed with no matches in current public, maintainer, corpus, schema, or project-plan docs |
| final `git diff --check` | passed |
| final `.c`/`.h` diff check | passed with no output |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 208 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Corpus/schema documentation files changed | 2 |
| Guard scripts changed | 1 |
| Python validation or regression test files changed | 3 |
| CI workflow files changed | 0 |
| Manifest data files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Hosted artifact intake items completed | 1 |
| Manifest promotion decision items completed as re-deferral | 1 |
| Metadata and guard implementation items completed | 1 |
| Normalizer regression coverage items completed | 1 |
| Documentation calibration items completed | 1 |
| Validation and closeout items completed | 1 |
| Selected Windows Cholesky freshness claims promoted | 0 |
| Selected Windows Cholesky freshness residuals retained | 1 |
| Broad Windows, package, ABI, performance, release, external parity, or state-of-the-art claims promoted | 0 |

The count covers Sprint 208 items 208.1 through 208.6.

## Closed Claim

Sprint 208 closes this bounded claim:

The current branch reviewed current hosted Windows evidence for the exact
`cholesky-spd-tridiag-5` selected comparison workflow path, rejected selected
Windows freshness promotion because generated support-tier and non-claim
surfaces still contradict it, strengthened manifest and PowerShell absence
guards, added Cholesky-specific Windows-path normalizer regressions, calibrated
public and maintainer documentation, and validated the selected re-deferral
state.

This claim does not include selected Windows Cholesky freshness promotion,
broad Windows report freshness, Windows selected oracle freshness, Windows
selected benchmark freshness, Windows QR incompatible selected freshness,
unselected Windows comparison freshness, Windows Makefile or `pkg-config`
parity, package-manager support, package-manager platform parity,
shared-library support, dynamic ABI compatibility, runtime-loader behavior,
portable performance, release readiness, external-library parity, or
state-of-the-art sparse linear algebra evidence.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-cholesky-intake.md](./artifacts/day1-windows-cholesky-intake.md);
- [day2-hosted-artifact-inventory.md](./artifacts/day2-hosted-artifact-inventory.md);
- [day3-row-path-traceability.md](./artifacts/day3-row-path-traceability.md);
- [day4-promotion-criteria.md](./artifacts/day4-promotion-criteria.md);
- [day5-promotion-decision.md](./artifacts/day5-promotion-decision.md);
- [day6-manifest-metadata-design.md](./artifacts/day6-manifest-metadata-design.md);
- [day7-manifest-guard-implementation.md](./artifacts/day7-manifest-guard-implementation.md);
- [day8-workflow-powershell-guard-alignment.md](./artifacts/day8-workflow-powershell-guard-alignment.md);
- [day9-normalizer-regression-design.md](./artifacts/day9-normalizer-regression-design.md);
- [day10-normalizer-regression-implementation.md](./artifacts/day10-normalizer-regression-implementation.md);
- [day11-public-docs-calibration.md](./artifacts/day11-public-docs-calibration.md);
- [day12-maintainer-corpus-docs.md](./artifacts/day12-maintainer-corpus-docs.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Selected Windows Cholesky freshness promotion | Future Windows/freshness owner after generated evidence semantics are redesigned | Change generated support tier and non-claim vocabulary for the exact selected Windows Cholesky lane, update selected manifest Windows metadata, calibrate docs, and keep broad Windows non-claims explicit. |
| Broad Windows report freshness | Future Windows/platform owner | Add selected or broad hosted Windows evidence model, generated row semantics, artifact inspection, manifest metadata, docs, and guards for every included target. |
| Windows QR incompatible selected freshness | Future Sprint 209 or later owner | Add hosted Windows/MSVC proof and artifact inspection for `qr-incompatible-ls` before selected QR Windows metadata promotion. |
| Windows Makefile and `pkg-config` parity | Future Windows install owner | Add Windows-native Makefile or `pkg-config` execution evidence, docs, and guard coverage without relying on CMake-only proof. |
| Package-manager and platform parity | Future package/provider owner | Add provider-specific proof and docs; Sprint 208 does not change package support. |
| ABI, release, performance, external parity, and state-of-the-art evidence | Future release/benchmark owner | Add explicit methodology, hosted proof, compatibility policy, external baselines, acceptance criteria, docs, and guards. |

## Next-Sprint Readiness

Sprint 208 leaves selected Windows Cholesky freshness in a precise
continued-re-deferral state with stronger guards.

| Future need | Sprint 208 handoff |
| --- | --- |
| Current Epic 19 status | Start from `docs/planning/EPIC_19/PROJECT_PLAN.md`, which marks Sprint 207 and Sprint 208 closed and Sprints 209-216 pending. |
| Windows selected Cholesky claim changes | Review Day 5, Day 6, Day 7, Day 8, Day 13, and Day 14 artifacts before adding Windows manifest metadata. |
| Generated evidence semantics | Change generated support-tier and non-claim wording before claiming selected Windows freshness. |
| Guard validation | Run `make windows-powershell-guard`, manifest tests, workflow tests, normalizer tests, corpus schema validation, docs/support guards, and report-index freshness. |
| Source or header changes | Run `make format && make lint && make test` before closeout. |
| Retrospective source material | Use `WORKING_NOTES.md` and Day 1-Day 14 artifacts under `SPRINT_208/artifacts/`. |

## Final Assessment

Sprint 208 is complete as a selected Windows Cholesky evidence and
guard-hardening sprint. It deliberately does not promote selected Windows
freshness; it closes the sprint by proving that the current support tier
remains reviewed bounded workflow evidence plus source-controlled
re-deferral, with stronger manifest, workflow/PowerShell, normalizer,
documentation, and validation coverage around every retained non-claim.

The branch is ready for review as planning, documentation, guard, regression,
and Windows selected-freshness governance work for Sprint 208.

