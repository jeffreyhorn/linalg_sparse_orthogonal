# Sprint 205 Retrospective

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-205`)  
**Status:** Closed for documentation consolidation, support-truth routing,
diagnostics vocabulary normalization, claim-guard alignment, validation, and
closeout

## Source Artifact Note

Sprint 205 was executed from the Epic 18 project-plan section for Sprint 205
and lives under `docs/planning/EPIC_18/SPRINT_205/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint reduced public documentation friction by adding a compact
problem-shape quick reference, centralizing support truth around
`INSTALL.md#support-readiness-matrix`, normalizing diagnostics vocabulary, and
aligning claim guards with the simplified wording. It did not promote new
solver behavior, API/ABI support, package support, platform support, hosted API
docs, portable performance, release readiness, or state-of-the-art claims.

## Definition Of Done Checklist

- [x] Created Sprint 205 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Audited public documentation surfaces for duplicate caveats, adoption
      friction, and support/readiness routing gaps.
- [x] Audited maintainer, benchmark/report, selected-target, schema, and
      planning-adjacent surfaces for vocabulary and evidence-owner conflicts.
- [x] Designed and implemented a compact problem-shape quick reference in
      `docs/cookbook.md`.
- [x] Added quick-reference discovery routes from README, tutorial, and
      examples.
- [x] Kept `INSTALL.md#support-readiness-matrix` as the single public
      support/readiness authority.
- [x] Shortened repeated caveats only where replacement links pointed to owner
      surfaces.
- [x] Normalized diagnostics vocabulary across selected direct, iterative,
      QR/SVD, eigensolver, benchmark, report, generated-output, and maintainer
      docs.
- [x] Added `tests/test_support_quick_reference_docs.py` and
      `make support-docs-guard`.
- [x] Re-anchored package-manager, static-package, Windows/PowerShell, and API
      routing guards to the simplified Sprint 205 wording.
- [x] Reconciled Epic 18 project-plan, retrospective, and residual-queue status
      surfaces so Sprint 205 is no longer marked pending future work.
- [x] Ran focused documentation, package, static, Windows, selected-performance,
      generated API, whitespace, generated-output, and C/header-inventory
      validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required by the sprint instruction.

## What Went Well

1. **Support truth stayed centralized.** Sprint 205 did not create a new
   support matrix. Public docs now route support/readiness questions to
   `INSTALL.md#support-readiness-matrix`, while maintainer and report docs keep
   their evidence-owner roles.

2. **The quick reference stayed narrow.** The new cookbook table routes users
   from problem shape to workflow owner docs without duplicating the detailed
   solver-selection guide or support matrix.

3. **Simplification did not become promotion.** README, tutorial, cookbook,
   examples, benchmark docs, API docs, and maintainer docs were shortened or
   clarified while preserving package, ABI, Windows, generated API,
   performance, release, and state-of-the-art non-claims.

4. **Diagnostics vocabulary became more consistent.** The sprint separated
   problem-local residuals, run-local convergence fields, QR-local/SVD-local
   diagnostics, selected-target evidence, generated-output diagnostics, and
   skip/defer status vocabulary.

5. **Guard drift was caught during validation.** Day 13 found stale marker
   expectations in package, static, and Windows claim guards. Updating those
   markers made the guards protect current wording rather than obsolete
   sentences.

6. **Epic status surfaces now agree.** Project-plan, Epic retrospective, and
   residual-queue status no longer describe Sprint 205 as pending after the
   branch-local closure evidence exists.

## What Didn't Go Well

1. **The wording compression had guard cost.** Several existing guards were
   tied to old long-form README sentences. The sprint needed extra validation
   work to preserve claim coverage while accepting the new concise wording.

2. **Homebrew proof validation was stateful locally.** A stale temporary
   `sparse-lu-ortho-local` formula from an earlier proof run blocked the
   package-manager guard until it was uninstalled. The proof then passed.

3. **Generated docs and Python caches remain local noise after validation.**
   Doxygen output is correctly ignored, but validation leaves local generated
   state that reviewers must distinguish from PR content.

4. **The support story still spans many files.** The sprint improved routing,
   but README, INSTALL, cookbook, solver selection, API docs, benchmarks,
   examples, maintainer guide, and guards still have to move together when
   support wording changes.

## Final Metrics

### Validation

| Metric | Sprint 205 close state |
| --- | --- |
| `python3 tests/test_support_quick_reference_docs.py` | passed on Days 12, 13, and 14 |
| `make support-docs-guard` | passed on Days 12, 13, and 14 |
| `python3 tests/test_selected_performance_docs.py` | passed on Days 12, 13, and 14 |
| `python3 tests/test_api_docs_routing.py` | passed on Days 12, 13, and 14 |
| `python3 tests/test_api_docs_local_only_guard.py` | passed on Day 13 |
| `make api-docs-freshness` | passed on Day 13 |
| `bash scripts/package_manager_deferral_check.sh` | passed on Days 13 and 14 after stale local proof state was removed |
| `bash scripts/static_package_deferral_check.sh` | passed on Days 13 and 14 |
| `python3 tests/test_validate_windows_powershell.py` | passed on Days 13 and 14 |
| stale Sprint 205 pending wording search | passed on Day 14 |
| generated-output git tracking check | passed on Days 13 and 14; `docs/api/` remains ignored |
| final `git diff --check` | passed |
| final full C quality gate | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 205 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Public documentation files changed | 7 |
| Maintainer/planning documentation files changed | 5 |
| Makefile targets changed | 1 |
| Shell guard files changed | 2 |
| Python validation or guard scripts changed | 2 |
| Python validation or guard test files changed | 3 |
| Epic project/status files changed | 3 |
| CI workflow files changed | 0 |
| Doxyfile files changed | 0 |
| `.gitignore` files changed | 0 |
| Production C implementation files changed | 0 |
| Public or internal C header files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Public and maintainer audit completed | 1 |
| Compact quick reference designed and implemented | 1 |
| Support truth consolidation completed | 1 |
| Diagnostics vocabulary normalization completed | 1 |
| Claim guard alignment completed | 1 |
| Final validation and closeout completed | 1 |
| Package, ABI, Windows, hosted API, portable performance, release, or state-of-the-art claims promoted | 0 |

The count covers Sprint 205 items 205.1 through 205.6.

## Closed Claim

Sprint 205 closes this bounded documentation and guard consolidation claim:

The project now has a compact problem-shape quick reference in
`docs/cookbook.md` with routes from README, tutorial, and examples;
`INSTALL.md#support-readiness-matrix` remains the public support/readiness
truth; selected public and maintainer docs use more consistent diagnostics and
evidence vocabulary; and focused claim guards protect the simplified wording
from broadening support, package, ABI, platform, performance, generated API,
release, or state-of-the-art claims.

This claim does not include package-manager distribution, Homebrew/core
readiness, bottles, Linuxbrew, public taps, binary packages, shared-library
support, dynamic ABI compatibility, Windows Makefile or `pkg-config` parity,
broad Windows selected freshness, hosted generated API documentation, retained
generated-doc artifacts, committed generated HTML, portable performance,
release readiness, external-library parity, or state-of-the-art sparse linear
algebra evidence.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-support-intake.md](./artifacts/day1-support-intake.md);
- [day2-public-doc-audit.md](./artifacts/day2-public-doc-audit.md);
- [day3-maintainer-report-audit.md](./artifacts/day3-maintainer-report-audit.md);
- [day4-quick-reference-design.md](./artifacts/day4-quick-reference-design.md);
- [day5-support-truth-architecture.md](./artifacts/day5-support-truth-architecture.md);
- [day6-quick-reference-implementation.md](./artifacts/day6-quick-reference-implementation.md);
- [day7-support-truth-consolidation.md](./artifacts/day7-support-truth-consolidation.md);
- [day8-example-workflow-routing.md](./artifacts/day8-example-workflow-routing.md);
- [day9-diagnostics-vocabulary-design.md](./artifacts/day9-diagnostics-vocabulary-design.md);
- [day10-diagnostics-vocabulary-implementation.md](./artifacts/day10-diagnostics-vocabulary-implementation.md);
- [day11-claim-guard-design.md](./artifacts/day11-claim-guard-design.md);
- [day12-claim-guard-implementation.md](./artifacts/day12-claim-guard-implementation.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Additional adoption UX work | Future documentation/product owner | Select a concrete user journey beyond the compact quick-reference/support-truth scope; add owner docs, guard coverage, and focused validation. |
| Broader support promotion | Future platform/package/release owner | Add exact evidence for the promoted support surface before changing support matrix wording. |
| Package-manager distribution | Future packaging owner | Add provider recipes, package validation, support tiers, and user-facing install evidence. |
| ABI/shared-library support | Future ABI/package owner | Add ABI policy, shared-library build/install evidence, compatibility tests, and public support wording. |
| Hosted generated API docs | Future generated-docs publication owner | Select hosted publication deliberately; add hosting workflow, retention/freshness evidence, link validation, and claim-boundary docs. |
| Portable performance or state-of-the-art evidence | Future benchmark/research owner | Add external baselines, workloads, methodology, platform metadata, and claim-reviewed hosted evidence. |

## Next-Sprint Readiness

Sprint 205 leaves documentation easier to navigate without increasing the
support surface.

| Future need | Sprint 205 handoff |
| --- | --- |
| First-use workflow routing | Start from `docs/cookbook.md#problem-shape-quick-reference`. |
| Support/readiness questions | Route public users to `INSTALL.md#support-readiness-matrix`. |
| Detailed solver choice | Keep `docs/solver_selection.md` as the detailed owner. |
| API documentation route | Keep `docs/api_reference.md` as the source-controlled API entry point and `docs/api/` as ignored local output. |
| Claim guard maintenance | Update support quick-reference, package, static, Windows, selected-performance, and API routing guards when wording changes. |
| Review hygiene | Treat `docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/` as ignored local validation output. |

## Final Assessment

Sprint 205 achieved its selected goal: it made user documentation easier to
navigate while preserving the project’s conservative support boundaries. The
branch is ready for PR review with no C/header changes and no new support,
package, ABI, Windows, hosted generated API, portable performance, release, or
state-of-the-art claims.
