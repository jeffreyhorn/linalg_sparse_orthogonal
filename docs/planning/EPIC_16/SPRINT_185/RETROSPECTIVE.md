# Sprint 185 Retrospective

**Sprint:** 185 - Large Test and Solver Review-Surface Reduction
**Duration:** 14 days (Days 1-14 landed on branch `sprint-185`)
**Status:** Complete

## Source Artifact Note

Sprint 185 was executed from the Epic 16 project-plan section for Sprint 185
and lives under `docs/planning/EPIC_16/SPRINT_185/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 185 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reviewed Sprint 177 large review-surface residual evidence, candidate
      file-size inventory, source-list checks, Make/CMake registration
      conventions, and prior proof-owner extraction precedent.
- [x] Selected exactly one large review surface:
      `tests/test_ldlt_csc.c`.
- [x] Defined behavior-preserving extraction boundaries, no-behavior-change
      rules, focused validation, and registration expectations before moving
      code.
- [x] Extracted LDLT CSC supernode helpers into
      `tests/test_ldlt_csc_supernode_helpers.h`.
- [x] Extracted LDLT CSC KKT, scaled-KKT, and analysis-backed setup fixtures
      into `tests/test_ldlt_csc_fixtures.h`.
- [x] Extracted dense-oracle, symmetric-swap, and native-wrapper comparison
      helpers into `tests/test_ldlt_csc_oracle_helpers.h`.
- [x] Kept `test_ldlt_csc` as the only registered proof-owner binary and
      preserved `main`, `RUN_TEST(...)` ordering, test names, fixture values,
      numerical tolerances, external dense-reference state, and production
      behavior.
- [x] Added `scripts/check_ldlt_csc_helper_guard.sh` and
      `make ldlt-csc-helper-guard` to protect helper-header presence, include
      ownership, and registration boundaries.
- [x] Added maintainer guidance for the new LDLT CSC helper ownership split in
      `docs/maintainer_guide.md`.
- [x] Ran focused selected-cluster validation, selected-cluster guard,
      source-list check, whitespace checks, and the full
      `make format && make lint && make test` gate after C/H changes.
- [x] Confirmed no production source, public API, internal solver API, CMake
      registration, library source manifest, new test binary, generated
      build/report artifact, or unsupported claim was added.

## What Went Well

1. **The sprint selected one high-impact review surface.** `tests/test_ldlt_csc.c`
   had high review cost, clear helper seams, and lower risk than production
   source extraction or broad integration-test movement.

2. **The extraction stayed behavior-preserving.** Helper bodies moved, but the
   existing proof-owner binary, test bodies, `RUN_TEST(...)` order, fixture
   values, tolerances, external dense-reference state, and process-global
   native/wrapper reset behavior stayed intact.

3. **Header-only helper ownership avoided registration churn.** The three new
   helper headers are included by `tests/test_ldlt_csc.c`; no Make/CMake test
   binary, CMake registration, library source manifest, or production source
   change was needed.

4. **The review surface became easier to scan.** `tests/test_ldlt_csc.c` moved
   from 3915 lines at the Day 3 baseline to 3469 lines at closeout, while the
   extracted helper families now have explicit names and comments.

5. **The new guard covers the sensitive drift points.** `make
   ldlt-csc-helper-guard` checks that `test_ldlt_csc` remains registered, the
   helper headers exist, each helper is included exactly once, and helper
   headers remain out of standalone Make/CMake/library registration.

6. **Maintainer guidance is discoverable.** `docs/maintainer_guide.md` now
   records where future LDLT CSC fixtures, dense oracles, native-wrapper
   helpers, and supernode helpers belong.

7. **Validation matched the risk profile.** Focused selected-cluster validation
   ran before the full gate, and Day 13 completed formatting, lint, full
   tests, guard checks, source-list checks, and whitespace validation.

## What Didn't Go Well

1. **The Makefile header-dependency caveat required explicit handling.** The
   focused `build/test_ldlt_csc` target does not track included helper headers,
   so focused validation had to remove the stale binary before rebuilding.

2. **Full lint remains expensive.** `make lint` runs strict warning compile,
   clang-tidy, and cppcheck across a large C surface. It passed, but it remains
   the long-running part of C/H extraction validation.

3. **Some useful candidates remain intentionally deferred.** QR, SVD, graph,
   integration, iterative, and LDLT CSC production-source surfaces still have
   review cost, but Sprint 185 only selected one cluster.

4. **The guard is layout-focused, not behavior proof.** The new guard protects
   helper-header ownership and registration boundaries; behavior preservation
   still depends on focused tests and the full C gate.

5. **Future helper growth remains a human-review risk.** Contributors can still
   add one-off helpers back into `tests/test_ldlt_csc.c`; the maintainer guide
   and guard reduce that risk but do not eliminate review judgment.

## Final Metrics

### Validation

| Metric | Sprint 185 close state |
| --- | --- |
| focused LDLT CSC build | passed: `make build/test_ldlt_csc` after forcing stale binary removal |
| focused LDLT CSC test | passed: `./build/test_ldlt_csc`, 100 tests, 0 failures, 0 skips, 3556 assertions |
| LDLT CSC helper guard | passed: `make ldlt-csc-helper-guard` |
| source-list check | passed: `make source-list-check`, 49 library sources |
| formatting | passed: `make format` |
| lint | passed: `make lint` |
| full test suite | passed: `make test`, ending with `All tests passed.` |
| final `git diff --check` | passed |
| generated build/report artifacts staged | 0 files |

### Changed Surface

| Metric | Sprint 185 close state |
| --- | ---: |
| selected review-surface clusters | 1 |
| selected proof-owner file baseline lines | 3915 |
| selected proof-owner file closeout lines | 3469 |
| selected proof-owner line reduction | 446 |
| helper headers added | 3 |
| guard scripts added | 1 |
| Makefile targets added | 1 |
| maintainer docs changed | 1 |
| production `.c` files changed | 0 |
| public headers changed | 0 |
| internal solver APIs changed | 0 |
| CMake registrations changed | 0 |
| library source manifest changes | 0 |
| new test binaries added | 0 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |

### Final File Sizes

| File | Closeout lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

### Claim Governance

| Metric | Sprint 185 close state |
| --- | ---: |
| solver behavior changes claimed | 0 |
| correctness-expansion claims added | 0 |
| performance claims added | 0 |
| external-library parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI claims added | 0 |
| platform support promotions added | 0 |
| release readiness claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 185 closes this Epic 16 review-surface reduction claim:

The selected LDLT CSC proof-owner file has a reduced review surface, with
supernode helpers, KKT/setup fixtures, and dense-oracle/native-wrapper helpers
extracted into family-local helper headers while preserving the existing
`test_ldlt_csc` proof-owner binary, test order, test names, fixture values,
numerical tolerances, external dense-reference behavior, and build
registration.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-review-surface-intake.md](./artifacts/day1-review-surface-intake.md);
- [day2-candidate-cluster-baseline.md](./artifacts/day2-candidate-cluster-baseline.md);
- [day3-selected-cluster-decision.md](./artifacts/day3-selected-cluster-decision.md);
- [day4-helper-boundary-design.md](./artifacts/day4-helper-boundary-design.md);
- [day5-registration-guardrail-design.md](./artifacts/day5-registration-guardrail-design.md);
- [day6-initial-helper-extraction.md](./artifacts/day6-initial-helper-extraction.md);
- [day7-fixture-setup-extraction.md](./artifacts/day7-fixture-setup-extraction.md);
- [day8-proof-owner-cleanup.md](./artifacts/day8-proof-owner-cleanup.md);
- [day9-drift-guard-update.md](./artifacts/day9-drift-guard-update.md);
- [day10-maintenance-invariants.md](./artifacts/day10-maintenance-invariants.md);
- [day11-contributor-guidance-alignment.md](./artifacts/day11-contributor-guidance-alignment.md);
- [day12-focused-cluster-validation.md](./artifacts/day12-focused-cluster-validation.md);
- [day13-full-quality-gate.md](./artifacts/day13-full-quality-gate.md);
- [day14-review-ready-handoff.md](./artifacts/day14-review-ready-handoff.md).

No solver behavior, correctness expansion, public API, internal solver API,
production implementation, CMake registration, library source manifest, new
test binary, package/platform/ABI support, performance, release readiness, or
state-of-the-art claim was added.

## Sprint 186 Readiness

Sprint 186 should start from the next Epic 16 project-plan section. If it
continues review-surface reduction or adjacent maintainability work, use this
Sprint 185 handoff:

| Future need | Sprint 185 handoff |
| --- | --- |
| Candidate selection | Select exactly one cluster and score review cost separately from behavior/refactor risk. |
| Helper extraction | Prefer family-local helper headers when the existing test binary should remain the proof owner. |
| Registration | Avoid new Make/CMake entries unless a new proof-owner binary or production source is explicitly selected. |
| Guard pattern | Reuse `make ldlt-csc-helper-guard` shape for selected-cluster helper presence, include ownership, and registration-boundary checks. |
| Focused validation | Force stale test binaries out of `build/` before focused rebuilds when only included helper headers changed. |
| Full validation | If any `.c` or `.h` file changes, run `make format && make lint && make test` plus focused tests, guards, source-list checks, and `git diff --check`. |
| Maintainer guidance | Put reusable helper-placement rules in `docs/maintainer_guide.md` and keep sprint artifacts as provenance. |
| Deferred candidates | Treat QR, SVD, graph, integration, iterative, and LDLT CSC production-source surfaces as separate future decisions rather than opportunistic follow-ons. |
