# Sprint 157 Retrospective

**Sprint:** 157 - Epic 14 Baseline, Evidence Freeze & Claim Targets
**Duration:** 14 days (Days 1-14 landed on branch `sprint-157`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 157 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Froze the post-Epic-13 branch baseline from merged PR #174 on `master`.
- [x] Inventoried source, public API, examples, benchmarks, scripts, generated
      artifacts, tests, CI, documentation, package, ABI, and platform surfaces.
- [x] Separated reviewed evidence, supplemental evidence, ignored generated
      outputs, local-only checks, advisory rows, and explicit non-claims.
- [x] Consolidated Epic 13 residuals and Epic 14 review findings into a bounded
      Epic 14 residual set.
- [x] Selected complete-gap targets for Sprints 158 through 166 and rejected
      broad partial-progress targets.
- [x] Published evidence-contract templates for API docs, hosted generated
      reports, comparison families, Windows package parity, performance
      publication, public-header preservation, package hardening, and final
      claim audit work.
- [x] Published a quality surface map that defines validation by touched
      surface, including the full C gate for `.c` and public `.h` changes.
- [x] Published the claim target register with accepted target claims,
      rejected unsupported claims, evidence owners, and documentation owners.
- [x] Published the risk register, mitigations, stop conditions, and Sprint 158
      generated API reference handoff.
- [x] Reconciled all Day 1 through Day 14 artifacts against the project-plan
      items.
- [x] Ran final documentation hygiene checks. No `.c` or public `.h` files
      changed, so the full C quality gate was not required for Sprint 157
      edits.

## What Went Well

1. **The sprint stayed at the correct altitude.** Sprint 157 established the
   Epic 14 baseline and claim contract without trying to close implementation
   work that belongs to later sprints.

2. **Unsupported claims stayed explicit.** State-of-the-art, broad ecosystem
   parity, portable performance, package-manager distribution, shared-library
   support, dynamic ABI support, broad Windows parity, and runtime-loader
   maturity were all kept as non-claims unless a later sprint funds and proves
   them.

3. **Evidence ownership is now reviewable.** Each selected target has a
   required evidence template, quality gate, documentation owner, and promotion
   boundary before it can become a public claim.

4. **Generated artifact boundaries are clear.** The sprint distinguishes
   checked-in source metadata from ignored local outputs such as API HTML,
   corpus reports, oracle rows, comparison reports, benchmark reports,
   coverage, and report indexes.

5. **The handoff to Sprint 158 is concrete.** The generated API reference
   residual now has starting sources, Day 1 actions, stop conditions, and
   source-header-first constraints.

## What Didn't Go Well

1. **The prompt carried a line-range mismatch.** The requested Sprint 157 path
   and branch were correct, but the supplied line range pointed at a later
   Epic 14 closeout section. The sprint artifacts resolved this by treating
   the Sprint 157 project-plan section as authoritative.

2. **Several major gaps remain intentionally unfixed.** Generated API HTML,
   hosted generated report publication, comparison breadth, Windows package
   parity, performance publication, public-header cleanup, and package boundary
   hardening are now scoped for later sprints, not closed here.

3. **The artifact set is planning-heavy.** Sprint 157 produced a useful
   evidence contract, but reviewers need the Day 8 target selection, Day 11
   claim register, Day 12 risk register, and Day 14 closeout for the shortest
   path through the work.

## Final Metrics

### Baseline Inventory

| Metric | Sprint 157 close state |
| --- | --- |
| starting branch | `sprint-157` |
| starting commit | `5b370dc33c1775205d839f99f0ef8ab8eaf7c3bd` |
| source files in `src/` | 69 |
| checked-in public headers | 18 |
| generated installed header template | `include/sparse_version.h.in` |
| top-level `tests/test_*.c` files | 59 |
| Makefile `TEST_SRCS` entries | 59 |
| local configure-only CTest registrations | 59 |
| reviewed Windows CTest count baseline | 59 |
| selected Epic 14 targets | 9 |
| rejected unsupported claim families | 12 |
| Sprint 157 risk entries | 12 |

### Validation

| Metric | Sprint 157 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required for Sprint 157 edits | no |
| source-list consistency check | passed: `python3 scripts/check_library_sources.py` |
| local CMake registration baseline | passed: `ctest --test-dir build-s157-baseline -N` reported `Total Tests: 59` |
| documentation whitespace scan | passed |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 157 close state |
| --- | ---: |
| daily artifacts under `SPRINT_157/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 0 |
| public headers changed | 0 |
| generated report files committed | 0 |

## Closed Claim

Sprint 157 closes this Epic 14 baseline and evidence-contract claim:

The project now has a reconciled Epic 14 starting baseline that inventories
code, public API, tests, CI, documentation, generated artifacts, package,
platform, ABI, quality, residual, target, claim, and risk surfaces; selects
only complete-gap targets for Sprints 158 through 166; defines the evidence
contracts needed to promote each target; and preserves explicit non-claims for
unsupported state-of-the-art, broad parity, package-manager, shared-library,
dynamic ABI, runtime-loader, Windows Makefile, Windows `pkg-config`, and
portable performance assertions.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-code-public-surface-inventory.md](./artifacts/day2-code-public-surface-inventory.md);
- [day3-test-ci-baseline.md](./artifacts/day3-test-ci-baseline.md);
- [day4-documentation-claim-baseline.md](./artifacts/day4-documentation-claim-baseline.md);
- [day5-generated-artifact-baseline.md](./artifacts/day5-generated-artifact-baseline.md);
- [day6-package-abi-platform-baseline.md](./artifacts/day6-package-abi-platform-baseline.md);
- [day7-residual-consolidation.md](./artifacts/day7-residual-consolidation.md);
- [day8-target-selection.md](./artifacts/day8-target-selection.md);
- [day9-evidence-contract-templates.md](./artifacts/day9-evidence-contract-templates.md);
- [day10-quality-surface-map.md](./artifacts/day10-quality-surface-map.md);
- [day11-claim-target-register.md](./artifacts/day11-claim-target-register.md);
- [day12-risk-register-and-sprint158-handoff.md](./artifacts/day12-risk-register-and-sprint158-handoff.md);
- [day13-baseline-reconciliation.md](./artifacts/day13-baseline-reconciliation.md);
- [day14-sprint-closeout-and-sprint158-handoff.md](./artifacts/day14-sprint-closeout-and-sprint158-handoff.md).

## Sprint 158 Readiness

Sprint 158 should start from the generated API reference handoff recorded on
Days 12 and 14:

| Starting item | Required posture |
| --- | --- |
| Generated API reference target | Decide whether the closure is committed HTML, hosted publication, or an explicit local-only non-publication decision. |
| Source-header-first rule | Keep public headers and `docs/api_reference.md` as the authoritative API sources unless Sprint 158 explicitly changes that contract. |
| Doxygen run | Confirm tool availability, capture warnings, and block publication until warnings are triaged. |
| Public-header coverage | Inventory generated pages against checked-in public headers and the generated version header template. |
| Generated files | Do not treat ignored local HTML as published evidence unless Sprint 158 changes repository or CI publication policy. |
| Quality gate | Run docs hygiene for documentation-only changes and escalate to `make format && make lint && make test` for `.c` or public `.h` edits. |

## Carry-Forward Targets

| Sprint | Target |
| --- | --- |
| 158 | Generated API reference publication decision |
| 159 | Hosted selected generated oracle/comparison freshness |
| 160 | One bounded QR comparison family |
| 161 | One bounded partial-SVD comparison family |
| 162 | Windows package parity decision |
| 163 | Methodology-bound performance publication |
| 164 | Public header/API coherence batch |
| 165 | Static-first package boundary hardening |
| 166 | Final claim recalibration and residual publication |
