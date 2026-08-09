# Sprint 147 Retrospective

**Sprint:** 147 - Epic 13 Baseline, Claim Targets & Evidence Gates
**Duration:** 14 days (Days 1-14 landed on branch `sprint-147`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 147 day-by-day plan, working notes, artifact directory,
      and closeout package.
- [x] Captured the post-Epic-12 baseline across source/test size, build,
      package, CI/platform, corpus, report, documentation, residual, and claim
      surfaces.
- [x] Converted Epic 12 residuals R1-R14 into selected Epic 13 gaps, duplicate
      fences, rejected claims, and deferred non-goals.
- [x] Defined candidate earned claims C1-C9 for Windows, corpus, report,
      package/ABI, external comparison, adoption, and final closeout.
- [x] Published evidence gates for Windows staged tests and Windows
      install-validation parity.
- [x] Published QR and partial-SVD corpus-family evidence gates with
      source-controlled row requirements, proof-owner tests, oracle/report
      rules, and raw-basis/raw-vector non-claims.
- [x] Published generated report freshness policy for required-generated versus
      advisory rows.
- [x] Published ABI/package and external-comparison gates that keep static-first
      package support, shared-library decisions, and comparison wording bounded.
- [x] Published a quality surface map covering code, headers, scripts,
      Make/CMake, CI, package, corpus, report, docs, benchmarks, generated
      artifacts, and external comparison validation.
- [x] Completed the public claim freeze audit across README, INSTALL,
      benchmark docs, maintainer guide, solver-selection, cookbook, tutorial,
      and public headers.
- [x] Published the Sprint 148 Windows staged-test prerequisite checklist and
      Sprint 147 artifact handoff map.
- [x] Ran Sprint 147 documentation validation:
  - `git diff --check -- docs/planning/EPIC_13/SPRINT_147`;
  - trailing-whitespace scans over Sprint 147 artifacts;
  - artifact existence checks;
  - `.c`/`.h` diff check confirming the full C gate was not required.

## What Went Well

1. **The sprint made Epic 13 narrower and more executable.** Day 5 selected a
   complete closure set instead of spreading the epic across every Epic 12
   residual. The selected path prioritizes Windows, QR/partial-SVD corpus
   breadth, generated freshness, package/ABI posture, external comparison,
   adoption cleanup, and final claim reconciliation.

2. **Claims were turned into gates before implementation.** Day 6 converted
   selected gaps into candidate claims with required evidence and rollback
   rules. That gives later sprints a concrete test for whether a claim was
   earned, narrowed, rejected, or deferred.

3. **Windows work was split correctly.** Days 7 and 14 keep staged pthread/POSIX
   test portability separate from Windows install-validation parity. Sprint 148
   can now focus on reviewed CMake test coverage without accidentally implying
   Windows Makefile, `pkg-config`, package-manager, shared-library, or broad
   platform parity.

4. **Corpus and report evidence boundaries stayed clean.** Days 8 and 9
   distinguish source-controlled corpus intent, focused C proof-owner tests,
   generated oracle rows, normalized report indexes, advisory rows, and
   required-generated freshness checks.

5. **The public claim freeze found no immediate overclaim.** Day 13 scanned the
   current public/support surfaces and found that the existing docs already use
   bounded fixture-local, static-first, local-measurement, and tiered platform
   wording.

## What Didn't Go Well

1. **The sprint was planning-heavy by design.** Sprint 147 created gates and
   handoffs, but it did not close any implementation gap directly. Sprint 148
   must now prove that the planning work reduces implementation ambiguity.

2. **Windows remains the first high-risk dependency.** The next sprint requires
   hosted MSVC evidence. Local Unix checks cannot prove the reviewed Windows
   lane, and expected CTest count drift remains a known failure mode.

3. **The claim surface is broad even after selection.** The sprint reduced
   scope, but the remaining selected tracks still touch platform, package,
   numerical corpus, report, comparison, and public documentation surfaces.
   Later sprints must keep the Day 12 quality map active.

4. **Generated evidence is easy to overread.** Sprint 147 recorded the boundary,
   but Sprints 150-152 can still accidentally treat source-controlled metadata
   or advisory generated rows as pass evidence unless the freshness gate is
   followed strictly.

5. **State-of-the-art remains intentionally unearned.** The sprint preserves
   the rejected broad claim. Any future competitive wording still depends on a
   direct, narrow, named external comparison study.

## Final Metrics

### Validation

| Metric | Sprint 147 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked `.h` changes | no |
| full C quality gate required | no |
| documentation whitespace validation | passed |
| trailing-whitespace scan | passed |
| Day 14 artifact existence check | passed |
| branch-specific hosted CI | not applicable before PR; future hosted evidence begins with Sprint 148/PR workflows |

### Artifact Package

| Metric | Sprint 147 close state |
| --- | ---: |
| daily artifacts under `SPRINT_147/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source-controlled generated report files committed | 0 |
| source files changed | 0 |
| public headers changed | 0 |

## Closed Claim

Sprint 147 closes this claim:

Epic 13 now has a selected closure scope and evidence contract for Sprints
148-156. The sprint freezes the post-Epic-12 baseline, selects completeable
gaps, defines candidate earned claims and rejected broad claims, publishes
Windows/corpus/report/package/comparison/quality gates, audits public wording,
and hands Sprint 148 a concrete Windows staged-test prerequisite checklist.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-baseline-intake.md](./artifacts/day1-baseline-intake.md);
- [day2-technical-baseline.md](./artifacts/day2-technical-baseline.md);
- [day3-corpus-report-evidence-baseline.md](./artifacts/day3-corpus-report-evidence-baseline.md);
- [day4-epic12-residual-intake.md](./artifacts/day4-epic12-residual-intake.md);
- [day5-selected-gap-register.md](./artifacts/day5-selected-gap-register.md);
- [day6-claim-target-register.md](./artifacts/day6-claim-target-register.md);
- [day7-windows-evidence-gate.md](./artifacts/day7-windows-evidence-gate.md);
- [day8-corpus-family-evidence-gate.md](./artifacts/day8-corpus-family-evidence-gate.md);
- [day9-generated-report-freshness-gate.md](./artifacts/day9-generated-report-freshness-gate.md);
- [day10-abi-package-evidence-gate.md](./artifacts/day10-abi-package-evidence-gate.md);
- [day11-external-comparison-evidence-gate.md](./artifacts/day11-external-comparison-evidence-gate.md);
- [day12-quality-surface-map.md](./artifacts/day12-quality-surface-map.md);
- [day13-public-claim-freeze-audit.md](./artifacts/day13-public-claim-freeze-audit.md);
- [day14-closeout-and-windows-handoff.md](./artifacts/day14-closeout-and-windows-handoff.md).

## Next-Sprint Readiness

Sprint 148 should begin from the Day 7 and Day 14 Windows handoff:

| Starting item | Required posture |
| --- | --- |
| `test_threads` | Audit pthread usage before choosing direct port, Windows-native equivalent, split proof owner, or retained staged status. |
| `test_sprint4_integration` | Audit pthread-dependent behavior and preserve POSIX proof if a Windows-specific proof owner is added. |
| `test_fuzz` | Audit POSIX temp-file assumptions and deterministic seed behavior before CMake promotion. |
| CMake registration | Change only after the selected test path configures, builds, and executes on MSVC. |
| Windows CTest count | Current baseline is `EXPECTED_WINDOWS_CTEST_COUNT=56`; update only with before/after enumeration and documented reason. |
| Hosted evidence | Record GitHub Actions run ID, commit SHA, job name, conclusion, registered count, and promoted test list before claiming reviewed Windows coverage. |
| Quality gate | Run `make format && make lint && make test` for any `.c` or `.h` change. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 147 close:

- Windows staged pthread/POSIX test portability until Sprint 148 earns or
  rejects promotion.
- Windows install-validation parity until Sprint 149 makes a product decision.
- Broader QR corpus evidence until Sprint 150 lands maintained fixture
  families, proof-owner tests, oracle/report rows, and bounded docs.
- Broader partial-SVD corpus evidence until Sprint 151 lands maintained
  fixture families, generated oracle/freshness proof, and bounded docs.
- Generated report freshness for claim-bearing rows until Sprint 152 selects
  and validates required-generated families.
- Shared-library ABI support until Sprint 153 implements and validates it or
  strengthens the static-first deferral.
- External-library comparison wording until Sprint 154 names dependency,
  version, fixture set, metric, tolerance, platform, and support tier.
- Adoption-surface cleanup until Sprint 155 updates tutorial/header/front-door
  coherence without widening claims.
- Final hosted/local evidence reconciliation and residual publication until
  Sprint 156.

Still consciously constrained rather than silently solved:

- no unqualified state-of-the-art sparse linear algebra claim;
- no broad external-library or ecosystem parity;
- no broad QR, SVD, or partial-SVD correctness beyond reviewed fixtures;
- no raw QR basis or raw singular-vector identity claim;
- no portable performance or backend-superiority claim;
- no generated report pass evidence from source-controlled rows alone;
- no shared-library support or dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager support;
- no Windows Makefile or `pkg-config` parity;
- no broad Windows platform parity.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-baseline-intake.md](./artifacts/day1-baseline-intake.md)
- [day2-technical-baseline.md](./artifacts/day2-technical-baseline.md)
- [day3-corpus-report-evidence-baseline.md](./artifacts/day3-corpus-report-evidence-baseline.md)
- [day4-epic12-residual-intake.md](./artifacts/day4-epic12-residual-intake.md)
- [day5-selected-gap-register.md](./artifacts/day5-selected-gap-register.md)
- [day6-claim-target-register.md](./artifacts/day6-claim-target-register.md)
- [day7-windows-evidence-gate.md](./artifacts/day7-windows-evidence-gate.md)
- [day8-corpus-family-evidence-gate.md](./artifacts/day8-corpus-family-evidence-gate.md)
- [day9-generated-report-freshness-gate.md](./artifacts/day9-generated-report-freshness-gate.md)
- [day10-abi-package-evidence-gate.md](./artifacts/day10-abi-package-evidence-gate.md)
- [day11-external-comparison-evidence-gate.md](./artifacts/day11-external-comparison-evidence-gate.md)
- [day12-quality-surface-map.md](./artifacts/day12-quality-surface-map.md)
- [day13-public-claim-freeze-audit.md](./artifacts/day13-public-claim-freeze-audit.md)
- [day14-closeout-and-windows-handoff.md](./artifacts/day14-closeout-and-windows-handoff.md)
