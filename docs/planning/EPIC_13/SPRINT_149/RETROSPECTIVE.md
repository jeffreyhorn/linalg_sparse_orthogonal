# Sprint 149 Retrospective

**Sprint:** 149 - Windows Install-Validation Parity Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-149`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 149 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Audited the Windows CMake install/downstream lane against the Linux and
      macOS reviewed static-first package-contract lanes.
- [x] Defined promotion, split, defer, and reject criteria for Windows package
      evidence.
- [x] Selected conditional promotion to a reviewed Windows CMake
      install/downstream validation lane for the maintained static-first
      package surface.
- [x] Preserved Windows Makefile, Windows `pkg-config`, package-manager,
      shared-library, dynamic ABI, runtime-loader, and broad Windows parity as
      explicit non-claims.
- [x] Updated `.github/workflows/windows-ci.yml` job and step wording to the
      reviewed CMake install/downstream lane.
- [x] Added stronger installed CMake package metadata checks for static target
      type, install-prefix includes, installed `.lib` target location, and
      source/build path leak rejection.
- [x] Added stronger installed `sparse.pc` metadata checks without claiming
      Windows `pkg-config` execution parity.
- [x] Added a generated basic installed CMake consumer in the Windows workflow
      while retaining the maintained example, exact-version consumer, and
      mismatch-version rejection proof.
- [x] Updated README, INSTALL, maintainer guidance, and report-family rows to
      reflect the reviewed Windows CMake install/downstream lane and non-claims.
- [x] Ran affected local package/install validation:
      `tests/test_cmake_install.sh`, `tests/test_install.sh`, and
      `scripts/static_package_deferral_check.sh`.
- [x] Ran local documentation/workflow/report hygiene checks.
- [x] Recorded hosted Windows proof as pending PR CI because no PR or branch
      workflow run existed during Day 13/Day 14 closeout.

## What Went Well

1. **The support claim stayed narrow.** Sprint 149 promoted only the Windows
   hosted MSVC CMake install/downstream lane. It did not treat CMake package
   confidence as Windows Makefile, Windows `pkg-config`, package-manager,
   shared-library, dynamic ABI, runtime-loader, or broad platform parity.

2. **The product decision came before workflow edits.** Days 1-4 separated
   evidence intake, criteria, and the conditional promotion decision before the
   workflow was renamed. That made the later CI wording traceable instead of
   cosmetic.

3. **Metadata checks became real assertions.** The Windows workflow now checks
   positive `STATIC IMPORTED` target metadata, install-prefix include metadata,
   installed `.lib` imported-location metadata, source/build path leaks,
   no shared/module imported metadata, static `sparse.pc` description, exact
   version text, and static archive link metadata.

4. **Consumer proof is stronger without adding platform overreach.** The lane
   now includes a generated basic installed CMake consumer, the maintained
   installed CMake example, exact-version consumer build/run proof, and
   mismatch-version rejection. It still does not execute Windows `pkg-config`.

5. **Docs and report rows moved with the evidence.** README, INSTALL,
   maintainer guidance, and the CI report-family row all name the reviewed
   Windows CMake install/downstream lane while preserving the same non-claims.

## What Didn't Go Well

1. **Hosted Windows evidence is still pending.** Local macOS checks prove
   workflow syntax, Unix install scripts, static package deferral, and report
   hygiene, but only PR CI can prove the MSVC install/downstream job.

2. **The sprint was mostly workflow and documentation, not library behavior.**
   That was appropriate for an install-validation parity decision, but it means
   the review should focus on claim accuracy and CI robustness rather than
   numerical algorithm changes.

3. **Fixed header-count checks remain brittle.** The Windows lane intentionally
   checks for 19 installed headers. Future public-header additions must update
   this count deliberately or hosted CI will fail.

4. **Historical artifact language can look broader than final support wording.**
   Early Day 1 intake uses "install-validation parity" and "supplemental"
   because that was the starting state. Day 13 and Day 14 searches distinguish
   historical artifact context from current public/workflow wording.

5. **Windows `pkg-config` remains a separate unresolved product decision.**
   `sparse.pc` metadata is checked as an installed file, but execution,
   variable resolution, cflags/libs resolution, and downstream compile/link/run
   via Windows `pkg-config` remain non-claims.

## Final Metrics

### Validation

| Metric | Sprint 149 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked `.h` changes | no |
| full C quality gate required | no |
| workflow YAML parse | passed |
| `git diff --check` | passed |
| targeted trailing-whitespace scan | passed |
| stale public/workflow supplemental wording search | passed |
| unsupported Windows package/platform claim search | passed; matches are explicit non-claims |
| corpus schema validation | passed |
| CI report index normalization | passed |
| package report index normalization | passed |
| local CMake install proof | passed: 26 checks, 0 failed, 0 skipped |
| local Make/pkg-config install proof | passed: 23 checks, 0 failed |
| static package deferral check | passed |
| hosted Windows CI | pending PR CI; no PR or branch run existed during Day 13/Day 14 closeout |

### Artifact Package

| Metric | Sprint 149 close state |
| --- | ---: |
| daily artifacts under `SPRINT_149/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 0 |
| workflow files changed | 1 |
| public/support docs changed | 3 |
| report manifest rows changed | 1 |

## Closed Claim

Sprint 149 closes this local claim:

The Windows package lane has been narrowed and promoted to reviewed CMake
install/downstream validation for the maintained static-first package surface,
subject to hosted MSVC proof in PR CI.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-install-intake.md](./artifacts/day1-install-intake.md);
- [day2-windows-package-audit.md](./artifacts/day2-windows-package-audit.md);
- [day3-promotion-criteria.md](./artifacts/day3-promotion-criteria.md);
- [day4-product-decision.md](./artifacts/day4-product-decision.md);
- [day5-workflow-design.md](./artifacts/day5-workflow-design.md);
- [day6-workflow-implementation.md](./artifacts/day6-workflow-implementation.md);
- [day7-metadata-check-design.md](./artifacts/day7-metadata-check-design.md);
- [day8-metadata-implementation.md](./artifacts/day8-metadata-implementation.md);
- [day9-consumer-proof-design.md](./artifacts/day9-consumer-proof-design.md);
- [day10-consumer-implementation.md](./artifacts/day10-consumer-implementation.md);
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md);
- [day12-local-validation.md](./artifacts/day12-local-validation.md);
- [day13-integrated-evidence-review.md](./artifacts/day13-integrated-evidence-review.md);
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md).

## Next-Sprint Readiness

Sprint 150 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Windows install/downstream lane | Treat reviewed CMake install/downstream validation as PR-time pending until hosted Windows CI proves it. |
| Windows CTest lane | Preserve the reviewed MSVC CMake CTest surface and expected count unless Sprint 150 intentionally changes test registration. |
| Package non-claims | Keep Windows Makefile, Windows `pkg-config`, package-manager, shared-library, dynamic ABI, runtime-loader, and broad Windows parity out of QR corpus claims. |
| Public headers | If QR work adds headers, update fixed installed-header checks intentionally. |
| QR fixture selection | Select bounded QR families for complete closure: rank-deficient rectangular, underdetermined minimum-norm, and reorder/COLAMD paths. |
| QR oracle semantics | Prefer residual, rank, nullspace, minimum-norm, and subspace-safe comparisons over raw-basis identity claims. |
| QR proof ownership | Add focused QR corpus proof-owner tests instead of expanding the largest monolithic QR file. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 149 close:

- hosted Windows proof for the reviewed CMake install/downstream lane until PR
  CI runs;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` execution parity;
- Windows `pkg-config` downstream compile/link/run parity;
- package-manager installation or resolver behavior;
- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows platform parity beyond hosted MSVC CMake lanes.

Still consciously constrained rather than silently solved:

- no broad Windows ecosystem parity claim;
- no package-manager availability claim;
- no shared-library or dynamic ABI support claim;
- no local claim from absent hosted CI logs;
- no unqualified package parity claim from CMake-only proof;
- no numerical correctness claim from package-lane validation alone.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-install-intake.md](./artifacts/day1-install-intake.md)
- [day2-windows-package-audit.md](./artifacts/day2-windows-package-audit.md)
- [day3-promotion-criteria.md](./artifacts/day3-promotion-criteria.md)
- [day4-product-decision.md](./artifacts/day4-product-decision.md)
- [day5-workflow-design.md](./artifacts/day5-workflow-design.md)
- [day6-workflow-implementation.md](./artifacts/day6-workflow-implementation.md)
- [day7-metadata-check-design.md](./artifacts/day7-metadata-check-design.md)
- [day8-metadata-implementation.md](./artifacts/day8-metadata-implementation.md)
- [day9-consumer-proof-design.md](./artifacts/day9-consumer-proof-design.md)
- [day10-consumer-implementation.md](./artifacts/day10-consumer-implementation.md)
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md)
- [day12-local-validation.md](./artifacts/day12-local-validation.md)
- [day13-integrated-evidence-review.md](./artifacts/day13-integrated-evidence-review.md)
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md)
