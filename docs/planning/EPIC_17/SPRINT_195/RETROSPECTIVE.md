# Sprint 195 Retrospective

**Sprint:** 195 - Selected Reliability and Failure-Path Proof
**Duration:** 14 days (Days 1-14 landed on branch `sprint-195`)
**Status:** Complete; one selected symbolic Cholesky allocation-failure owner
proved with deterministic cleanup, stale-output, retry, focused-gate, and
claim-boundary evidence

## Source Artifact Note

Sprint 195 was executed from the Epic 17 project-plan section for Sprint 195
and lives under `docs/planning/EPIC_17/SPRINT_195/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 195 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited allocation-heavy and failure-prone candidates and selected
      exactly one bounded reliability owner:
      `sparse_symbolic_cholesky()` in `src/sparse_etree.c`.
- [x] Recorded selected-owner cleanup, publication, stale-output, retry,
      caller-owned input, and unsupported-breadth invariants before changing
      code.
- [x] Reused the existing deterministic allocation-failure harness instead of
      adding a new public or owner-local API.
- [x] Converted the selected non-empty symbolic `sym->col_ptr` allocation to
      wrapper-controlled allocation so deterministic fail-at-count tests can
      reach it.
- [x] Added regression tests for allocation-hook reachability, partial-state
      cleanup, stale-output suppression, repeated cleanup after failure,
      caller-owned input preservation, and retry-after-reset behavior.
- [x] Added `make symbolic-allocation-failure-gate`, a CTest
      `allocation_failure` label for `test_etree`, and a Python registration
      guard to keep focused proof coverage from drifting.
- [x] Updated README, INSTALL, and maintainer documentation with exact
      selected-owner reliability claims and retained non-claims.
- [x] Ran focused source ownership, CMake selector, formatting, documentation,
      and registration checks.
- [x] Ran the full C quality gate because `.c` files changed.
- [x] Confirmed the sprint did not add broad allocation-failure, OS OOM,
      concurrency, platform parity, package/install, generated-tooling,
      performance, release, or state-of-the-art reliability claims.

## What Went Well

1. **The sprint selected one owner and stayed there.** The candidate scoring
   picked `sparse_symbolic_cholesky()` because it had enough allocation and
   cleanup complexity to be useful while still being bounded enough for a
   complete proof.

2. **The invariant-first approach paid off.** Cleanup, publication,
   stale-output, retry, caller-owned input, and non-claim boundaries were
   written before implementation, which kept the tests and documentation from
   drifting into broader reliability claims.

3. **The implementation change was small and meaningful.** Replacing the
   selected direct `malloc` path with `sparse_malloc_array(...)` made the
   non-empty symbolic column-pointer allocation reachable by the existing
   deterministic harness without changing the public API or success semantics.

4. **The regression tests constrain real failure behavior.** The new
   `test_etree` coverage proves failed allocation status, empty failed outputs,
   double cleanup safety, caller-owned matrix preservation, and successful
   retry output against the known 5x5 symbolic oracle.

5. **Focused validation is now reviewable.** `make
   symbolic-allocation-failure-gate` gives reviewers a compact local command,
   while the Python guard checks the Make target, CMake label, selected
   `RUN_TEST(...)` entries, and the required `col_ptr` fail-after case.

6. **Claim boundaries remained explicit.** README, INSTALL, and maintainer
   wording describe only the selected symbolic proof and preserve non-claims
   for unrelated owners, generated tooling, package/install flows, OS OOM,
   concurrency, platform breadth, and state-of-the-art reliability.

## What Didn't Go Well

1. **The proof is necessarily narrow.** Sprint 195 closes one selected owner,
   but `sparse_symbolic_lu()`, `sparse_analyze()`, standalone etree/postorder
   helpers, direct solvers, matrix construction, and other allocation-heavy
   paths remain outside the proof.

2. **Global allocation-hook discipline remains delicate.** The tests reset the
   harness around failure injection, but future tests using the same process
   global hook must continue storing statuses, resetting, and only then
   asserting.

3. **Documentation grew to prevent over-reading.** The selected proof is easy
   to misinterpret as broad reliability evidence, so user and maintainer docs
   need explicit non-claim wording.

4. **The focused gate still runs the full `test_etree` binary.** This keeps
   behavior coverage broad for the owner, but it means the focused gate is not
   a minimal single-test executable.

5. **No sanitizer or hosted allocation-failure lane was added.** Local
   deterministic checks and the full quality gate passed, but Sprint 195 did
   not add ASan, MSan, Windows, or CI-specific reliability proof for this lane.

## Final Metrics

### Validation

| Metric | Sprint 195 close state |
| --- | --- |
| source-list check | passed with 49 library sources |
| focused registration guard | passed |
| focused Make gate | passed |
| `test_etree` focused gate result | 101 tests, 0 failures, 0 skips, 1262 assertions |
| CMake configure for focused selector | passed |
| CMake selected `test_etree` build | passed |
| CTest symbolic selector listing | selected only `test_etree` |
| CTest symbolic selector execution | passed with 1 of 1 tests passing |
| claim-boundary grep | passed for README, INSTALL, maintainer guide, and Sprint 195 artifacts |
| final `make format` | passed |
| final `make lint` | passed strict warning builds, clang-tidy, and cppcheck |
| final `make test` | passed with final `All tests passed.` |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 195 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint closeout artifacts added | 1 |
| Sprint retrospective files added | 1 |
| C implementation files changed | 1 |
| C test files changed | 1 |
| Python guard tests added | 1 |
| Makefile focused targets added | 1 |
| CMake test labels added | 1 |
| User-facing documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Public header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| Package template files changed | 0 |
| CI workflow files changed | 0 |

### Reliability Claim Metrics

| Metric | Sprint 195 close state |
| --- | ---: |
| selected reliability owners proved | 1 |
| deterministic allocation hook reachability tests added | 2 |
| partial-state cleanup test families added | 1 |
| retry-after-reset test families added | 1 |
| selected fail-after cases covered | 9 |
| focused registration guards added | 1 |
| broad allocation-failure coverage claims added | 0 |
| OS OOM claims added | 0 |
| concurrent allocation-hook claims added | 0 |
| platform parity claims added | 0 |
| package/install reliability claims added | 0 |
| generated-tooling reliability claims added | 0 |
| state-of-the-art reliability claims added | 0 |

## Closed Claim

Sprint 195 closes this bounded implementation claim:

One selected reliability owner, `sparse_symbolic_cholesky()`, now has
deterministic local allocation-failure proof for selected symbolic output
allocation behavior. The proof covers allocation failure status, cleanup of
partially initialized symbolic output, stale-output suppression, repeated
cleanup safety, caller-owned fixture preservation, and successful retry after
reset on bounded fixtures. The proof is reproducible through `make
symbolic-allocation-failure-gate`, guarded by a Python registration check,
discoverable through a CTest `allocation_failure` label, documented with exact
claim boundaries, and validated by focused and full quality gates.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-reliability-intake.md](./artifacts/day1-reliability-intake.md);
- [day2-owner-selection-scoring.md](./artifacts/day2-owner-selection-scoring.md);
- [day3-selected-owner-invariant-record.md](./artifacts/day3-selected-owner-invariant-record.md);
- [day4-harness-design.md](./artifacts/day4-harness-design.md);
- [day5-harness-scaffold.md](./artifacts/day5-harness-scaffold.md);
- [day6-selected-owner-harness-integration.md](./artifacts/day6-selected-owner-harness-integration.md);
- [day7-failed-allocation-regression-tests.md](./artifacts/day7-failed-allocation-regression-tests.md);
- [day8-cleanup-stale-output-proof.md](./artifacts/day8-cleanup-stale-output-proof.md);
- [day9-successful-retry-proof.md](./artifacts/day9-successful-retry-proof.md);
- [day10-focused-gate-definition.md](./artifacts/day10-focused-gate-definition.md);
- [day11-claim-boundaries.md](./artifacts/day11-claim-boundaries.md);
- [day12-focused-validation-source-ownership.md](./artifacts/day12-focused-validation-source-ownership.md);
- [day13-full-quality-gate.md](./artifacts/day13-full-quality-gate.md);
- [day14-closeout-review-package.md](./artifacts/day14-closeout-review-package.md).

No broad allocation-failure, symbolic-analysis-wide, direct-solver, matrix
construction, generated-tooling, package/install, OS OOM, concurrent
allocation-hook, platform parity, performance, release, or state-of-the-art
reliability claim was added.

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| `sparse_symbolic_lu()` allocation failures remain unproved | Future symbolic reliability owner | Select `sparse_symbolic_lu()` explicitly, record invariants, wrapper-control selected allocations if needed, add failure/retry tests, focused gate, docs, and full validation. |
| `sparse_analyze()` allocation failures remain unproved | Future analysis reliability owner | Define analyze ownership and publication semantics across symbolic/numeric states before adding deterministic failure coverage. |
| Standalone etree/postorder/colcount helper allocation failures remain out of scope | Future etree reliability owner | Select helper-level ownership, add deterministic failure injection, cleanup assertions, and focused gate evidence. |
| Direct solvers and matrix construction remain outside this proof | Future solver/matrix reliability owner | Repeat the selected-owner proof process for each owner instead of widening Sprint 195 claims. |
| OS OOM and concurrent allocation-hook behavior remain unclaimed | Future allocator/platform owner | Add allocator policy, concurrency semantics, platform evidence, and stress or sanitizer validation before documenting those claims. |
| No hosted CI lane owns the symbolic allocation-failure gate | Future CI owner | Add a reviewed hosted lane or explicitly keep the gate local-only in support/readiness wording. |

## Next-Sprint Readiness

Sprint 196 can start from a completed selected reliability proof rather than a
partial broad reliability initiative.

| Future need | Sprint 195 handoff |
| --- | --- |
| Additional reliability owner | Use the Sprint 195 pattern: candidate scoring, invariant record, harness design, regression tests, focused gate, claim-boundary docs, focused validation, full quality gate. |
| Focused local validation | Run `make symbolic-allocation-failure-gate` for the selected symbolic Cholesky proof. |
| CTest selection | Use `ctest --test-dir <build-dir> -L symbolic` after configuring a CMake build. |
| Coverage drift guard | Run `python3 tests/test_symbolic_allocation_failure_gate_registration.py` whenever changing the focused gate, CMake label, or selected `test_etree` registrations. |
| Claim wording | Keep this proof scoped to selected `sparse_symbolic_cholesky()` output allocation behavior and retain non-claims for broad reliability surfaces. |

## Validation Retrospective

Sprint 195 changed a C implementation file and a C test file, so the full C
quality gate was required and run.

The final Day 13 validation commands were:

```sh
make format
make lint
make test
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
git diff --check
```

Day 14 then rechecked source ownership, focused registration, untracked sprint
artifacts, and item traceability while preparing the review package.
