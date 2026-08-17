# Sprint 162 Retrospective

**Sprint:** 162 - Windows Package Parity Decision Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-162`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 162 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Audited the Windows CMake install/downstream package proof against the
      Linux/macOS Make install and `pkg-config` execution proof.
- [x] Reviewed static-first package metadata across CMake package files,
      `sparse.pc`, install scripts, package validation tests, CI, README,
      INSTALL, and maintainer guidance.
- [x] Selected the product decision to retain Windows Makefile
      install/uninstall parity and Windows `pkg-config` command execution
      parity as explicit non-claims.
- [x] Designed the retained non-claim guard path with exact files,
      diagnostics, support-tier wording, and validation commands.
- [x] Extended `scripts/static_package_deferral_check.sh` so the static package
      guard checks Windows package non-claim wording and rejects unselected
      Windows package execution in the Windows workflow.
- [x] Aligned Windows CI wording and hosted-output diagnostics so `sparse.pc`
      is metadata-only inspection and downstream evidence is installed CMake
      package proof.
- [x] Updated README, INSTALL, and maintainer guide wording for the selected
      Windows CMake-first package surface.
- [x] Revalidated the static package guard, Make install/`pkg-config` proof,
      and CMake install/export/downstream proof.
- [x] Published cross-platform validation notes, hosted-only Windows checklist,
      claim-to-evidence trace, retained non-claim trace, and Sprint 163
      methodology-bound performance handoff.
- [x] Ran final documentation/script/workflow hygiene checks. No `.c` or
      public `.h` files changed, so the full C quality gate was not required.

## What Went Well

1. **The sprint separated package shape from execution parity.** Windows
   already had a strong CMake install/downstream package proof. The sprint
   avoided treating installed `sparse.pc` metadata as Windows `pkg-config`
   command proof.

2. **The product decision became explicit.** Windows Makefile parity and
   Windows `pkg-config` execution parity are retained non-claims, not vague
   gaps hidden behind CMake package success.

3. **The guard is executable.** `scripts/static_package_deferral_check.sh` now
   fails if the docs or Windows workflow drift away from the CMake-first
   package tier, or if the Windows workflow starts running `pkg-config`,
   `make install`, or `make uninstall` without a new proof-backed decision.

4. **CI wording is clearer without changing CI behavior.** Windows hosted logs
   now label `sparse.pc` validation as metadata-only and label generated,
   maintained, exact-version, and mismatch consumers as installed CMake
   package evidence.

5. **Validation stayed proportionate to the changed surface.** The sprint ran
   static package, Make install/`pkg-config`, and CMake install/export checks
   while correctly skipping the full C quality gate because no C or header
   files changed.

## What Didn't Go Well

1. **The prompt referenced the older Epic 12 path.** The branch and current
   plan correctly place Sprint 162 under Epic 14, but the prompt still pointed
   at the older Epic 12 project-plan path. Working notes recorded the mismatch
   and treated the Epic 14 plan as authoritative.

2. **Windows execution remains hosted-only.** Local validation could run the
   static guard, Make install/`pkg-config`, and CMake install/export checks,
   but the actual Windows MSVC workflow still requires hosted CI.

3. **The selected closure is a retained non-claim, not new parity.** This is
   the right decision for the evidence available, but it leaves Windows
   Makefile and Windows `pkg-config` execution parity unresolved as future
   product choices.

4. **The package boundary still needs future hardening.** Sprint 162 hardened
   the Windows package non-claim boundary, but shared-library support, dynamic
   ABI policy, runtime-loader behavior, package-manager distribution, and
   broader ABI/package work remain future work.

## Final Metrics

### Validation

| Metric | Sprint 162 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required for Sprint 162 edits | no |
| static package deferral guard | passed |
| Make install/`pkg-config` proof | passed: `23` checks, `0` failures |
| CMake install/export proof | passed: `27` checks, `0` failures, `0` skipped |
| Windows CTest count expectation | `59` hosted CMake tests |
| Windows package proof owner | hosted `.github/workflows/windows-ci.yml::install-and-downstream` |
| local `actionlint` availability | unavailable |
| local `pwsh` availability | unavailable |
| documentation/script/workflow whitespace scan | passed |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 162 close state |
| --- | ---: |
| daily artifacts under `SPRINT_162/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 0 |
| public headers changed | 0 |
| workflow files changed | 1 |
| package guard scripts changed | 1 |
| public documentation files changed | 3 |

## Closed Claim

Sprint 162 closes this Windows package parity decision claim:

The project now has an explicit, guarded Windows package support tier: Windows
package validation remains CMake-first and static-first, installed `sparse.pc`
is metadata-only inspection, Windows Makefile install/uninstall parity and
Windows `pkg-config` command execution parity are retained non-claims, and the
static package guard prevents documentation or workflow drift into unselected
Windows package execution.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-windows-package-audit.md](./artifacts/day2-windows-package-audit.md);
- [day3-metadata-boundary.md](./artifacts/day3-metadata-boundary.md);
- [day4-product-decision.md](./artifacts/day4-product-decision.md);
- [day5-proof-or-guard-design.md](./artifacts/day5-proof-or-guard-design.md);
- [day6-implementation-foundation.md](./artifacts/day6-implementation-foundation.md);
- [day7-implementation-completion.md](./artifacts/day7-implementation-completion.md);
- [day8-ci-alignment.md](./artifacts/day8-ci-alignment.md);
- [day9-downstream-evidence.md](./artifacts/day9-downstream-evidence.md);
- [day10-focused-validation.md](./artifacts/day10-focused-validation.md);
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md);
- [day12-cross-platform-validation.md](./artifacts/day12-cross-platform-validation.md);
- [day13-evidence-claim-review.md](./artifacts/day13-evidence-claim-review.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 163 Readiness

Sprint 163 should start from this package boundary:

| Starting item | Required posture |
| --- | --- |
| Performance publication | Keep performance evidence separate from Sprint 162 package proof. |
| Windows package evidence | Cite only CMake install/downstream validation and metadata-only `sparse.pc` inspection. |
| Windows `pkg-config` | Treat as a retained non-claim unless a future sprint selects a provider and downstream proof path. |
| Windows Makefile install/uninstall | Treat as a retained non-claim unless a future sprint selects and validates a Windows Make route. |
| Static package guard | Keep `scripts/static_package_deferral_check.sh` in the validation set for package/docs/workflow edits. |
| Package non-claims | Preserve package-manager, shared-library, dynamic ABI, runtime-loader, static/shared selector, and broad Windows non-claims. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 162 close:

- Windows Makefile install/uninstall parity;
- Windows `pkg-config` provider selection;
- Windows `pkg-config --exists`, `--cflags`, `--libs`, and `--modversion`
  proof;
- downstream Windows compile/link/run from `pkg-config` output;
- package-manager distribution;
- shared-library product support;
- dynamic ABI compatibility policy;
- runtime-loader validation;
- static/shared package selectors;
- broader Windows package parity beyond the reviewed CMake-first surface;
- Sprint 163 methodology-bound performance publication;
- Sprint 165 static-first package and ABI hardening.

Still consciously constrained rather than silently solved:

- no Windows Makefile parity claim;
- no Windows `pkg-config` execution parity claim;
- no package-manager support claim;
- no shared-library support claim;
- no dynamic ABI compatibility claim;
- no runtime-loader behavior claim;
- no broad Windows platform parity claim;
- no performance, superiority, or state-of-the-art claim from package
  validation evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-windows-package-audit.md](./artifacts/day2-windows-package-audit.md)
- [day3-metadata-boundary.md](./artifacts/day3-metadata-boundary.md)
- [day4-product-decision.md](./artifacts/day4-product-decision.md)
- [day5-proof-or-guard-design.md](./artifacts/day5-proof-or-guard-design.md)
- [day6-implementation-foundation.md](./artifacts/day6-implementation-foundation.md)
- [day7-implementation-completion.md](./artifacts/day7-implementation-completion.md)
- [day8-ci-alignment.md](./artifacts/day8-ci-alignment.md)
- [day9-downstream-evidence.md](./artifacts/day9-downstream-evidence.md)
- [day10-focused-validation.md](./artifacts/day10-focused-validation.md)
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md)
- [day12-cross-platform-validation.md](./artifacts/day12-cross-platform-validation.md)
- [day13-evidence-claim-review.md](./artifacts/day13-evidence-claim-review.md)
- [day14-closeout.md](./artifacts/day14-closeout.md)
