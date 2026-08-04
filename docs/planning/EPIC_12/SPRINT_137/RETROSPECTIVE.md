# Sprint 137 Retrospective

**Sprint:** 137 - Epic 12 Baseline, Gap Selection & Evidence Contract
**Duration:** 14 days (Days 1-14 landed on branch `sprint-137`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 137 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 12 project plan, review, gap-closure todo, Epic 11
      retrospective, and Sprint 136 residual queue.
- [x] Captured post-Epic-11 source, test, benchmark, example, build, package,
      CI, report, and support-tier baselines.
- [x] Reconciled Epic 11 residuals into Epic 12 candidates, duplicates,
      already-covered context, optional work, and non-claims.
- [x] Assigned active residuals to owner workstreams with dependencies,
      promotion gates, non-goals, and stop conditions.
- [x] Defined complete-closure criteria and a 21-point scoring rubric that
      favors complete gap closure over broad partial progress.
- [x] Selected one primary target for every later Epic 12 sprint:
      corpus/oracle, QR, partial-SVD, report/freshness, runtime/backend,
      package/ABI, Windows platform lane, adoption, and closeout.
- [x] Defined corpus/oracle, report/freshness, package/ABI/platform, public
      claim, and quality evidence templates.
- [x] Audited and froze current public claim boundaries before implementation
      sprints begin.
- [x] Published Sprint 138 readiness handoff, residual register, stop
      conditions, and closeout checklist.
- [x] Ran Sprint 137 documentation-only validation:
  - `git diff --check`;
  - trailing-whitespace scan under `docs/planning/EPIC_12/SPRINT_137`;
  - focused Markdown local link/path validation under `docs/planning/EPIC_12`;
  - changed/untracked `.c` and `.h` scan.
- [x] No `.c` or `.h` files changed, so the full
      `make format && make lint && make test` gate was not required.

## What Went Well

1. **Epic 12 starts from written boundaries instead of inherited momentum.**
   Sprint 137 treated Epic 11 residuals as candidate inputs, then scored,
   selected, deferred, or rejected them using explicit criteria. That keeps
   Sprint 138-146 implementation work tied to complete closure rather than
   broad residual coverage.

2. **The selected gap sequence is dependency ordered.**
   Corpus/oracle work goes first, then QR and partial-SVD residual closure,
   then report normalization, runtime/backend governance, package/ABI
   follow-through, Windows platform promotion, adoption simplification, and
   final closeout.

3. **Claim gates are now concrete enough for implementation sprints.**
   The sprint produced reusable templates for corpus rows, generated fixtures,
   optional-data skips, oracle rows, report freshness, package/ABI decisions,
   downstream proof, platform promotion, public wording, and quality surfaces.

4. **Public claims were frozen before code work begins.**
   Day 12 confirmed the live public docs already fence the highest-risk claim
   families: state-of-the-art status, external parity, shared libraries,
   dynamic ABI, package-manager support, platform parity, portable
   performance, generated reports, coverage, and dead-code interpretation.

5. **Sprint 138 has a clean handoff.**
   The final closeout gives Sprint 138 a single selected implementation target,
   required templates, quality requirements, claim boundaries, and stop
   conditions. Sprint 138 should not need to reopen baseline or gap selection.

## What Didn't Go Well

1. **The sprint was documentation-heavy by design.**
   Sprint 137 created a substantial artifact package before implementation.
   That was appropriate for an intake and evidence-contract sprint, but future
   implementation sprints should keep artifacts closer to code, test, report,
   and docs changes.

2. **The claim scan was noisy when aimed at the full planning tree.**
   Old planning history contains many historical uses of claim-sensitive
   phrases. Day 12 had to narrow the audit to live public and maintainer-facing
   surfaces before making a useful freeze decision.

3. **No implementation gap was closed yet.**
   Sprint 137 intentionally prepared the evidence architecture and selected
   targets. The actual corpus, QR, partial-SVD, report, runtime, package,
   platform, and adoption closures remain Sprint 138-146 work.

4. **Some residuals remain large future product decisions.**
   Shared libraries, dynamic ABI compatibility, package-manager support,
   broad external parity, and unqualified state-of-the-art status remain
   outside the selected Epic 12 implementation path.

## Final Metrics

### Baseline Metrics

| Metric | Sprint 137 baseline |
|---|---:|
| C/header/template files under `src`, `include`, `tests`, `benchmarks`, and `examples` | 191 |
| Total lines across those files | 123,352 |
| implementation `.c` files | 49 |
| private implementation headers | 20 |
| public headers/templates | 19 |
| test `.c` files | 58 |
| test helper headers | 11 |
| benchmark `.c` files | 16 |
| example `.c` files | 15 |
| Makefile main test binaries | 57 |
| CMake direct `add_sparse_test(...)` lines plus conditional gates | 54 |

### Validation

| Metric | Sprint 137 close state |
|---|---|
| tracked `.c`/`.h` changes | 0 |
| `git diff --check` | passed |
| Sprint 137 trailing-whitespace scan | passed |
| focused Epic 12 Markdown local link/path validation | passed |
| changed/untracked `.c` and `.h` scan | passed; no C/header changes |
| full C quality gate | not required; no `.c`/`.h` changes |

### Artifact Package

| Metric | Sprint 137 close state |
|---|---:|
| daily artifacts under `SPRINT_137/artifacts/` | 14 |
| final retrospective files | 1 |
| Sprint 138 readiness artifacts | 2 |
| public docs changed by Sprint 137 | 0 |
| source files changed by Sprint 137 | 0 |

## Selected Epic 12 Targets

| Sprint | Selected target |
| --- | --- |
| 138 | Maintained numerical corpus/oracle contract with one durable deterministic fixture lane and explicit skip/defer semantics. |
| 139 | QR rank-deficient nullspace/subspace residual closure backed by the Sprint 138 corpus lane. |
| 140 | Partial-SVD repeated/clustered-spectrum residual closure with convergence-budget semantics for deterministic fixtures. |
| 141 | Row-meaning-preserving report index normalization plus stale-report checking for maintained evidence families. |
| 142 | Runtime/backend precedence contract plus one normalized local sentinel lane. |
| 143 | Static-first package/ABI product decision with stricter static-first follow-through and optional-mode install/downstream proof. |
| 144 | Windows CMake install/downstream reviewed-parity lane, or explicit rejection with source-level blockers if hosted proof fails. |
| 145 | Adoption front door for earned build/install, solver selection, diagnostics, corpus/report, runtime, package, and platform behavior. |
| 146 | Epic 12 final evidence inventory, claim recalibration, residual queue, and state-of-the-art assessment. |

## Residual Deferred Debt

Most important carry-forward work:

- implement the Sprint 138 maintained corpus/oracle contract;
- close the selected QR rank-deficient nullspace/subspace residual;
- close the selected partial-SVD repeated/clustered-spectrum residual with
  convergence-budget semantics;
- implement row-meaning-preserving report indexes and stale-report checks;
- define runtime/backend precedence and add one normalized local sentinel lane;
- execute the static-first package/ABI follow-through path and optional static
  mode matrix;
- attempt the Windows CMake install/downstream platform lane with hosted proof
  and fallback semantics;
- simplify adoption docs only after evidence-bearing sprints land;
- publish final Epic 12 evidence, residuals, and claim recalibration.

Still consciously constrained rather than silently solved:

- no unqualified state-of-the-art claim;
- no broad ecosystem or external-library parity claim;
- no broad SuiteSparse corpus completeness claim;
- no portable performance, scalability, memory, OpenMP speedup, or backend
  parity claim;
- no generated-report-as-release-proof claim;
- no coverage-completeness claim;
- no dead-code-removal-ready claim;
- no shared-library, dynamic ABI, runtime-loader, or package-manager support
  claim;
- no macOS reviewed install/export parity claim;
- no general Windows parity or POSIX/pthread staged-test promotion claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-artifact-setup.md](./artifacts/day1-scope-artifact-setup.md)
- [day2-source-test-maintainability-baseline.md](./artifacts/day2-source-test-maintainability-baseline.md)
- [day3-build-package-ci-report-baseline.md](./artifacts/day3-build-package-ci-report-baseline.md)
- [day4-epic11-residual-intake.md](./artifacts/day4-epic11-residual-intake.md)
- [day5-residual-owner-nongoal-map.md](./artifacts/day5-residual-owner-nongoal-map.md)
- [day6-gap-selection-criteria.md](./artifacts/day6-gap-selection-criteria.md)
- [day7-gap-selection-decision.md](./artifacts/day7-gap-selection-decision.md)
- [day8-corpus-oracle-evidence-templates.md](./artifacts/day8-corpus-oracle-evidence-templates.md)
- [day9-report-index-freshness-templates.md](./artifacts/day9-report-index-freshness-templates.md)
- [day10-package-abi-platform-claim-templates.md](./artifacts/day10-package-abi-platform-claim-templates.md)
- [day11-quality-surface-map.md](./artifacts/day11-quality-surface-map.md)
- [day12-public-claim-freeze.md](./artifacts/day12-public-claim-freeze.md)
- [day13-handoff-synthesis-sprint138-readiness.md](./artifacts/day13-handoff-synthesis-sprint138-readiness.md)
- [day14-closeout-and-sprint138-readiness.md](./artifacts/day14-closeout-and-sprint138-readiness.md)

## Closeout

Sprint 137 is complete. It closes the Epic 12 intake and evidence-contract
phase with baseline evidence, selected implementation targets, reusable proof
templates, quality rules, public claim gates, a residual register, and a
Sprint 138-ready corpus/oracle handoff. It does not change source code,
generated reports, workflows, package metadata, public docs, or support claims.
