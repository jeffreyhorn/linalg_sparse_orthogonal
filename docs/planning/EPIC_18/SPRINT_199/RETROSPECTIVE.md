# Sprint 199 Retrospective

**Sprint:** 199 - Selected Windows Cholesky Freshness Promotion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-199`)
**Status:** Closed with selected Windows Cholesky freshness promotion
re-deferred; exact hosted Windows Cholesky workflow evidence reviewed and
retained as guarded workflow evidence

## Source Artifact Note

Sprint 199 was executed from the Epic 18 project-plan section for Sprint 199
and lives under `docs/planning/EPIC_18/SPRINT_199/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint reviewed hosted Windows evidence for the exact
`cholesky-spd-tridiag-5` MSVC/CMake comparison workflow path. That evidence
proves one successful bounded hosted workflow path, but it does not promote
selected Windows freshness because selected target metadata, generated support
tier, generated non-claim wording, and the claim contract still do not promote
Windows together.

## Definition Of Done Checklist

- [x] Created Sprint 199 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Inspected hosted Windows CI run `34269219871` and artifact
      `sprint190-windows-selected-comparison-cholesky` for the exact
      `cholesky-spd-tridiag-5` target.
- [x] Recorded workflow provenance, artifact metadata, expected file set,
      expected row IDs, row statuses, platform/build metadata, and source
      commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`.
- [x] Applied the promotion threshold and explicitly re-deferred manifest
      promotion because the source-controlled metadata and generated
      support/non-claim semantics still remain local-only.
- [x] Hardened selected comparison artifact matching for Windows backslash,
      mixed-separator, relative, and absolute suffix paths.
- [x] Added near-match rejection and wrong-target row-set mismatch diagnostics
      so unrelated comparison artifacts cannot satisfy the selected Cholesky
      freshness gate.
- [x] Added selected-target CLI misuse coverage and unknown-target validation
      coverage.
- [x] Added PowerShell guard tests for generator target drift, selected
      artifact-name drift, and fail-open artifact upload behavior.
- [x] Calibrated README, INSTALL, corpus docs, maintainer guide, and Epic 18
      planning status to the reviewed-but-re-deferred Windows Cholesky
      disposition.
- [x] Ran focused Python, selected freshness, docs, package/static deferral,
      PowerShell guard, workflow guard, and whitespace validation.
- [x] Confirmed generated `build/` and `docs/api/` outputs remain ignored and
      are not source-controlled staging candidates.
- [x] Confirmed no `.c` or `.h` files changed, so `make format && make lint &&
      make test` was not required by the user quality-check rule.

## What Went Well

1. **Hosted evidence was reviewed at row level.** The sprint did not treat a
   green Windows job as promotion by itself. It inspected the artifact, row
   IDs, statuses, platform metadata, source commit, and generated non-claims.

2. **The manifest decision stayed conservative.** The exact Windows Cholesky
   path is real guarded evidence, but the selected manifest still omits
   `windows` until metadata and generated claim semantics are promoted
   together.

3. **Windows path handling now has focused coverage.** Selected comparison
   filtering covers backslash paths, mixed separators, absolute Windows suffix
   paths, and near-match rejection.

4. **Wrong-target evidence now fails loudly.** A generated comparison artifact
   for a different target can no longer silently satisfy
   `--selected-target cholesky-spd-tridiag-5`; it produces a selected
   row-set mismatch with target-specific diagnostics.

5. **Workflow ownership is guarded directly.** PowerShell validation now has
   tests for selected generator target drift, selected artifact-name drift,
   and missing `if-no-files-found: error`.

6. **Public and maintainer wording now agree.** README, INSTALL, corpus docs,
   maintainer guide, and Epic planning all use the same
   reviewed-but-re-deferred vocabulary.

## What Didn't Go Well

1. **The hosted proof could not become a promoted claim in this sprint.**
   Generated rows still carry `support_tier=local_only`, and generated
   summary/non-claim wording still does not promote Windows.

2. **The claim-boundary markers are exact-string sensitive.** Day 11
   documentation cleanup initially broke guarded README and corpus markers
   because line wrapping changed required substrings. The final validator run
   caught and resolved those issues.

3. **Local PowerShell remains environment-dependent.** The local
   `make windows-powershell-validate` wrapper exits `2` without `pwsh`, so
   hosted `--require-pwsh` remains the authoritative parseability pass/fail
   owner.

4. **Windows freshness remains split across several owner surfaces.** The
   workflow, selected manifest, normalizer, generated rows, public docs, and
   maintainer docs must all move together before any future promotion.

## Final Metrics

### Validation

| Metric | Sprint 199 close state |
| --- | --- |
| selected report target manifest tests | passed during Day 13 and Day 14 validation |
| selected comparison workflow tests | passed during Day 13 and Day 14 validation |
| normalizer tests | passed during Day 13 validation |
| external comparison runner tests | passed during Day 13 validation |
| PowerShell validator tests | passed during Day 13 and Day 14 validation |
| selected Cholesky freshness command | passed during Day 13 validation with six selected rows fresh to current `HEAD` |
| selected comparison freshness Make target | passed during Day 13 validation with local-only freshness wording |
| docs check | passed during Day 13 and Day 14 validation |
| package-manager deferral guard | passed during Day 13 validation |
| static package deferral guard | passed during Day 13 validation |
| local `make windows-powershell-validate` | exit `2` during Day 13 because local `pwsh` is unavailable after structural checks passed; unavailable local evidence, not pass evidence |
| generated artifact scan | passed; `build/` and `docs/api/` remain ignored |
| final `git diff --check` | passed |
| final `.c`/`.h` diff check | passed with no `.c` or `.h` files modified |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 199 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Corpus documentation files changed | 1 |
| Python implementation files changed | 1 |
| Python test files changed | 2 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Hosted evidence review items completed | 1 |
| Manifest decision items completed as re-deferred | 1 |
| Normalizer hardening items completed | 1 |
| Workflow guard items completed | 1 |
| Documentation calibration items completed | 1 |
| Local validation items completed | 1 |
| Windows selected freshness promotions | 0 |
| Broad Windows report freshness claims promoted | 0 |

The count covers Sprint 199 items 199.1 through 199.6.

## Closed Claim

Sprint 199 closes this bounded claim:

The branch reviews hosted Windows evidence for the exact
`cholesky-spd-tridiag-5` MSVC/CMake comparison workflow path, keeps selected
Windows freshness promotion re-deferred in the selected target manifest,
hardens selected comparison filtering and diagnostics for Windows artifact
paths and wrong-target evidence, strengthens PowerShell workflow ownership
tests, calibrates public and maintainer documentation to the re-deferred
disposition, and validates the focused local gates for that state.

This claim does not include promoted selected Windows freshness, broad Windows
report freshness, Windows selected oracle freshness, Windows selected benchmark
freshness, QR incompatible Windows comparison freshness, unselected Windows
comparison families, Windows Makefile parity, Windows `pkg-config` execution
parity, package-manager support, shared-library support, dynamic ABI support,
runtime-loader behavior, broad Windows parity, performance superiority,
external-library parity, release readiness, or state-of-the-art status.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-windows-freshness-intake.md](./artifacts/day1-windows-freshness-intake.md);
- [day2-hosted-artifact-inventory.md](./artifacts/day2-hosted-artifact-inventory.md);
- [day3-evidence-semantics.md](./artifacts/day3-evidence-semantics.md);
- [day4-manifest-decision.md](./artifacts/day4-manifest-decision.md);
- [day5-windows-path-normalization-tests.md](./artifacts/day5-windows-path-normalization-tests.md);
- [day6-freshness-diagnostics.md](./artifacts/day6-freshness-diagnostics.md);
- [day7-normalizer-hardening.md](./artifacts/day7-normalizer-hardening.md);
- [day8-workflow-alignment.md](./artifacts/day8-workflow-alignment.md);
- [day9-powershell-guard.md](./artifacts/day9-powershell-guard.md);
- [day10-gate-integration.md](./artifacts/day10-gate-integration.md);
- [day11-public-docs.md](./artifacts/day11-public-docs.md);
- [day12-maintainer-planning-alignment.md](./artifacts/day12-maintainer-planning-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Selected Windows Cholesky freshness promotion remains re-deferred | Selected report target manifest, generator metadata, normalizer/docs owners | Promote `workflow_platforms`, support tier, generated summary/non-claim wording, and claim contract together after final evidence review. |
| Broad Windows report freshness remains unclaimed | Future Windows report freshness owner | Add Windows-safe generation paths, exact selected upload scopes, manifest metadata, and guards; do not infer from the Cholesky path. |
| QR incompatible Windows comparison freshness remains unpromoted | Future target-specific comparison owner | Prove its MSVC project probe, inspect hosted artifacts, and promote selected-target metadata separately. |
| Windows selected oracle freshness remains unclaimed | Future oracle workflow owner | Add selected oracle generation, upload, freshness path, manifest metadata, and docs. |
| Windows selected benchmark freshness remains unclaimed | Future benchmark workflow owner | Add selected benchmark methodology, hosted lane, artifact, manifest metadata, and claim guards. |
| Local PowerShell unavailable checks remain environment residuals | Maintainer environment | Record exit `2` as unavailable local evidence; hosted `--require-pwsh` remains pass/fail ownership. |

## Next-Sprint Readiness

Sprint 199 leaves the Windows selected Cholesky path in a guarded,
reviewed-but-re-deferred state.

| Future need | Sprint 199 handoff |
| --- | --- |
| Windows Cholesky promotion | Start from Day 4, Day 10, Day 12, and Day 14. Promote only when manifest metadata, generated support tier, generated non-claim wording, and claim contract move together. |
| Windows path diagnostics | Day 5-Day 7 tests cover Windows separator matching, near-match rejection, wrong-target row-set mismatch, stale rows, missing rows, and selected-target CLI misuse. |
| Workflow guard maintenance | Day 8-Day 9 confirm the bounded workflow path and add direct PowerShell guard tests for target, artifact, upload path, and fail-closed drift. |
| Public wording | Day 11-Day 12 calibrate README, INSTALL, corpus docs, maintainer guide, and Epic planning status to the re-deferred claim. |
| Closeout validation | Day 13-Day 14 record the focused validation set and generated artifact hygiene. |

## Final Assessment

Sprint 199 is complete as a selected Windows Cholesky freshness review and
guard-hardening sprint. It does not promote selected Windows freshness, but it
does make the re-deferral explicit, tested, documented, and reviewable.

The branch is ready for review as Python normalizer/validator tests,
documentation calibration, planning evidence, and claim-governance work around
the bounded Windows `cholesky-spd-tridiag-5` workflow path.
