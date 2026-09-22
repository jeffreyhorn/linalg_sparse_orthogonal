# Sprint 209 Retrospective

**Sprint:** 209 - Windows QR Incompatible Promotion Decision  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-209`)  
**Status:** Closed with a bounded Windows/MSVC QR incompatible evidence lane and
continued selected Windows QR freshness re-deferral

## Source Artifact Note

Sprint 209 was executed from the Epic 19 project-plan section for Sprint 209
and lives under `docs/planning/EPIC_19/SPRINT_209/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint designed the exact hosted Windows/MSVC `qr-incompatible-ls` proof,
added a bounded Windows workflow lane to collect that evidence, strengthened
artifact inspection and selected-target guards, calibrated public and
maintainer documentation, and closed with focused plus integrated validation.
It did not promote selected Windows QR incompatible freshness because no hosted
Windows CI QR run artifact was available for inspection. The selected manifest
therefore remains Linux/macOS-only and `local_only`.

## Definition Of Done Checklist

- [x] Created Sprint 209 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Reviewed Sprint 203 and Sprint 208 handoff evidence for the retained
      Windows QR incompatible promotion residual.
- [x] Defined the exact hosted MSVC/CMake probe command, expected row contract,
      and six-file artifact bundle required before selected promotion.
- [x] Inspected current hosted Windows evidence and confirmed no hosted QR
      incompatible artifact existed for promotion.
- [x] Added one bounded Windows workflow lane for
      `qr-incompatible-ls` evidence collection.
- [x] Kept the workflow upload fail-closed and limited to the exact selected QR
      six-file bundle.
- [x] Extended normalizer freshness checks so selected comparison targets must
      have their required generated sidecar files.
- [x] Added normalizer, workflow, manifest, and PowerShell guard regressions
      for QR required files, Windows-style paths, bounded workflow scope, and
      retained re-deferral metadata.
- [x] Updated README, INSTALL, maintainer guide, corpus README, report-index
      schema docs, and Epic 19 project-plan status with Sprint 209 evidence
      collection and retained non-claims.
- [x] Ran focused QR generator, freshness, normalizer, manifest, workflow,
      PowerShell, docs, support-doc, package-deferral, API-docs, comparison
      freshness, Python syntax, stale-claim, and whitespace validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required by the sprint instruction.

## What Went Well

1. **The promotion decision stayed tied to hosted evidence.** Sprint 209 added
   the workflow lane needed to collect Windows/MSVC QR evidence, but did not
   treat lane availability or local generator proof as selected Windows
   freshness.

2. **The selected QR artifact contract is now explicit.** The proof requires
   exactly `project_observations.tsv`, `baseline_observations.tsv`,
   `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv` for
   `qr-incompatible-ls`.

3. **Freshness validation became harder to satisfy accidentally.** The
   normalizer now checks selected comparison sidecar files, and regressions
   cover missing QR artifacts without forcing unrelated selected targets to
   provide QR files.

4. **Workflow ownership is guarded directly.** The Windows workflow tests and
   PowerShell validator now recognize the Sprint 209 QR evidence lane while
   rejecting broad uploads, stale commands, and premature selected manifest
   promotion metadata.

5. **Public, maintainer, corpus, and schema docs now agree.** The main support
   surfaces describe Sprint 209 as QR evidence collection only and preserve the
   re-deferred selected Windows QR claim boundary.

6. **Validation matched the changed surface.** The sprint focused on workflow,
   Python guard/test, documentation, corpus, and planning changes and kept the
   C quality gate conditional on C/header diffs.

## What Didn't Go Well

1. **Hosted Windows QR evidence was still unavailable.** Without a hosted run
   and artifact inspection for the new QR lane, selected Windows QR freshness
   had to remain re-deferred.

2. **The distinction between evidence collection and promotion needed repeated
   reinforcement.** Several surfaces had to say clearly that adding the lane is
   not the same as promoting `SRT-COMP-QR-INCOMPATIBLE-LS`.

3. **Claim-boundary guards are necessarily detailed.** The work touched
   workflow parsing, manifest contracts, generated report semantics, public
   docs, maintainer docs, corpus docs, and schema wording to keep the same
   boundary intact.

4. **Local validation cannot substitute for hosted Windows/MSVC proof.** Local
   QR generation and freshness pass, but the sprint still depends on future
   hosted Windows CI artifact evidence before manifest promotion.

## Final Metrics

### Validation

| Metric | Sprint 209 close state |
| --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | passed |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | passed |
| `python3 tests/test_normalize_report_index.py` | passed |
| `python3 tests/test_selected_report_targets_manifest.py` | passed |
| `python3 tests/test_selected_comparison_workflow.py` | passed |
| `python3 tests/test_validate_windows_powershell.py` | passed |
| `python3 tests/test_run_external_comparison.py` | passed |
| Python syntax checks for touched scripts/tests | passed |
| `make windows-powershell-guard` | passed |
| `make docs-check` | passed |
| `make support-docs-guard` | passed |
| `make package-manager-deferral-guard` | passed |
| `bash scripts/static_package_deferral_check.sh` | passed |
| `make api-docs-freshness` | passed |
| `make report-index-comparison-freshness` | passed with freshness ok for 46 comparison rows |
| stale and overclaim wording scan | passed with no obsolete Sprint 209 pending-range or positive QR/Windows promotion wording |
| final `git diff --check` | passed |
| final `.c`/`.h` diff check | passed with no output |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 209 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Corpus/schema documentation files changed | 2 |
| Guard scripts changed | 2 |
| Python validation or regression test files changed | 4 |
| CI workflow files changed | 1 |
| Manifest data files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| MSVC probe design items completed | 1 |
| Workflow implementation items completed | 1 |
| Artifact inspection test items completed | 1 |
| Manifest decision items completed as re-deferral | 1 |
| Documentation and claim-guard items completed | 1 |
| Validation and closeout items completed | 1 |
| Selected Windows QR incompatible freshness claims promoted | 0 |
| Selected Windows QR incompatible freshness residuals retained | 1 |
| Broad Windows, QR, package, ABI, performance, release, external parity, or state-of-the-art claims promoted | 0 |

The count covers Sprint 209 items 209.1 through 209.6.

## Closed Claim

Sprint 209 closes this bounded claim:

The current branch adds a selected Windows/MSVC QR incompatible
evidence-collection lane for `qr-incompatible-ls`, strengthens selected QR
artifact inspection and guard coverage, records that no hosted Windows QR run
artifact was available for promotion, keeps `SRT-COMP-QR-INCOMPATIBLE-LS`
Linux/macOS-only and `local_only`, calibrates public and maintainer claim
surfaces, and validates the retained re-deferral state.

This claim does not include selected Windows QR incompatible freshness
promotion, broad Windows report freshness, broad QR parity, broad
least-squares parity, raw QR basis identity, Q sign or orientation identity,
global rank-threshold policy, broad rank-deficient solve support, NumPy,
SciPy, LAPACK, SuiteSparse, Eigen, or other external-library parity, Windows
Makefile parity, Windows `pkg-config` execution parity, package-manager
support, package-manager platform parity, shared-library support, dynamic ABI
compatibility, runtime-loader behavior, portable performance, release
readiness, or state-of-the-art sparse linear algebra evidence.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-qr-intake.md](./artifacts/day1-windows-qr-intake.md);
- [day2-msvc-probe-design.md](./artifacts/day2-msvc-probe-design.md);
- [day3-hosted-evidence-inventory.md](./artifacts/day3-hosted-evidence-inventory.md);
- [day4-workflow-implementation-design.md](./artifacts/day4-workflow-implementation-design.md);
- [day5-workflow-implementation.md](./artifacts/day5-workflow-implementation.md);
- [day6-artifact-inspection-tests.md](./artifacts/day6-artifact-inspection-tests.md);
- [day7-manifest-decision-criteria.md](./artifacts/day7-manifest-decision-criteria.md);
- [day8-manifest-decision.md](./artifacts/day8-manifest-decision.md);
- [day9-guard-integration.md](./artifacts/day9-guard-integration.md);
- [day10-public-docs-calibration.md](./artifacts/day10-public-docs-calibration.md);
- [day11-maintainer-corpus-docs.md](./artifacts/day11-maintainer-corpus-docs.md);
- [day12-focused-validation.md](./artifacts/day12-focused-validation.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Selected Windows QR incompatible freshness promotion | Future Windows comparison owner after hosted artifact exists | Run hosted Windows CI for the Sprint 209 QR lane, inspect `sprint209-windows-selected-comparison-qr-incompatible`, confirm exact six-file membership and row freshness, then update selected manifest metadata, support tier, generated non-claims, docs, schema, corpus wording, and guards together. |
| Broad Windows report freshness | Future Windows/platform owner | Add a broader hosted Windows evidence model, generated row semantics, artifact inspection, manifest metadata, docs, and guards for every included target. |
| Broad QR and least-squares parity | Future solver/comparison owner | Define fixtures, tolerances, external baselines, platform matrix, acceptance rules, generated evidence, and claim wording before promotion. |
| Windows Makefile and `pkg-config` parity | Future Windows install owner | Add Windows-native Makefile or `pkg-config` execution evidence, documentation, and guard coverage without relying only on CMake proof. |
| Package-manager and platform parity | Future package/provider owner | Add provider-specific proof, supported platform tiers, installation validation, uninstall behavior, and user-facing documentation. |
| ABI, release, performance, external parity, and state-of-the-art evidence | Future release/benchmark owner | Add explicit methodology, hosted proof, compatibility policy, external baselines, acceptance criteria, docs, and guards. |

## Next-Sprint Readiness

Sprint 209 leaves Windows QR incompatible selected freshness in a precise
continued-re-deferral state with a workflow lane ready to collect hosted proof.

| Future need | Sprint 209 handoff |
| --- | --- |
| Current Epic 19 status | Start from `docs/planning/EPIC_19/PROJECT_PLAN.md`, which marks Sprints 207-209 closed and later Sprint 210-216 work as future. |
| Windows QR claim changes | Review Day 2, Day 7, Day 8, Day 13, and Day 14 artifacts before adding Windows manifest metadata. |
| Hosted QR evidence | Trigger or inspect the Windows CI `selected-qr-incompatible-comparison-freshness` job and review the exact uploaded artifact. |
| Generated evidence semantics | Keep manifest support tier and non-claims synchronized with generated row wording before selected Windows QR promotion. |
| Guard validation | Run QR generator/freshness, normalizer tests, selected manifest tests, workflow tests, PowerShell guard/tests, docs/support guards, package/static deferral guards, API docs freshness, and comparison freshness. |
| Source or header changes | Run `make format && make lint && make test` before closeout. |
| Retrospective source material | Use `WORKING_NOTES.md` and Day 1-Day 14 artifacts under `SPRINT_209/artifacts/`. |

## Final Assessment

Sprint 209 improves readiness for selected Windows QR incompatible promotion
without claiming it prematurely. The repository now has a bounded hosted
workflow path and stronger local guard coverage, but the decisive evidence is
still future hosted Windows/MSVC artifact inspection.
