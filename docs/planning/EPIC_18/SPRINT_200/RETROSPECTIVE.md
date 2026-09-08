# Sprint 200 Retrospective

**Sprint:** 200 - Additional Allocation-Failure Owner Proof
**Duration:** 14 days (Days 1-14 landed on branch `sprint-200`)
**Status:** Closed with selected `sparse_symbolic_lu()` allocation-failure
owner proof completed; broad allocation-failure and state-of-the-art
reliability claims remain unclaimed

## Source Artifact Note

Sprint 200 was executed from the Epic 18 project-plan section for Sprint 200
and lives under `docs/planning/EPIC_18/SPRINT_200/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint selected exactly one additional allocation-failure owner:
`sparse_symbolic_lu()`. It records the owner decision, lifecycle trace,
pre-edit invariants, minimal allocation-hook reachability changes,
failed-allocation regressions, cleanup and retry proof, focused gate,
registration guard, claim documentation, integrated validation, review
hardening, and final closeout.

## Definition Of Done Checklist

- [x] Created Sprint 200 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Ranked allocation-failure owner candidates and selected exactly one new
      owner: `sparse_symbolic_lu()`.
- [x] Documented cleanup, publication, stale-output, retry, caller-owned input,
      hook-reset, and unsupported-breadth invariants before code edits.
- [x] Converted selected symbolic LU owner allocations from direct allocation
      calls to existing private allocation wrappers where deterministic
      failure injection needed reachability.
- [x] Added deterministic regression tests for selected symbolic LU
      allocation failure, free-safe failed outputs, repeated cleanup,
      caller-owned matrix/permutation preservation, and retry-after-reset.
- [x] Added `make symbolic-lu-allocation-failure-gate` and a Python
      registration guard so the selected proof cannot silently drop out.
- [x] Updated README, INSTALL, maintainer guide, Epic 18 project-plan status,
      residual queue, and Epic retrospective wording to the selected-owner
      claim boundary.
- [x] Ran focused selected-owner, broader symbolic, source-list, docs,
      formatting, lint, full-test, and whitespace validation.
- [x] Preserved explicit residuals for broad allocation-failure coverage,
      analysis lifecycle cleanup, direct solvers, matrix construction,
      platform/hosted proof, generated tooling, package/install reliability,
      OS OOM behavior, concurrent allocation-hook behavior, and
      state-of-the-art reliability support.

## What Went Well

1. **The selected-owner boundary stayed narrow.** The sprint selected
   `sparse_symbolic_lu()` and did not absorb `sparse_analyze()`, standalone
   etree helpers, direct solvers, or matrix construction into the earned
   claim.

2. **Invariants were written before implementation.** Day 4 defined the
   cleanup, publication, stale-output, retry, caller-input, hook, and scope
   contracts before the allocation wrapper and test changes landed.

3. **The proof exercises every selected failure site.** The selected failure
   table covers permutation workspaces, symbolic LU workspaces, propagated
   symbolic Cholesky intermediate allocations, and U-output construction.

4. **Retry and cleanup are both explicit.** The tests do not stop at error
   status. They verify failed outputs are free-safe, repeat the cleanup sweep,
   reset the allocation hook, and then prove fresh success on retry.

5. **The focused gate is reviewable.** `make
   symbolic-lu-allocation-failure-gate` runs only the selected symbolic LU
   proof path, while the existing broader symbolic allocation gate remains
   available for regression confidence.

6. **Public and maintainer docs use the same claim vocabulary.** README,
   INSTALL, maintainer guide, and planning surfaces describe selected symbolic
   LU proof without promoting broad reliability.

## What Didn't Go Well

1. **The selected owner still touches composed symbolic internals.**
   `sparse_symbolic_lu()` depends on matrix construction and symbolic
   Cholesky-derived intermediate behavior, so the artifacts had to be precise
   about which propagated failures are inside the selected proof and which
   broader owners remain outside it.

2. **A second symbolic gate was needed for clarity.** Folding selected LU into
   only the existing symbolic allocation gate would have made the review
   surface less explicit, so the sprint added a separate selected symbolic LU
   gate and guard.

3. **Planning status had to be hardened late.** Day 13 found Epic tracking
   rows that still said Sprint 200 was only complete through Day 11. Those
   rows were corrected before closeout.

4. **Broad allocation-failure support remains far from complete.** The sprint
   closes one meaningful owner, but direct solvers, analysis lifecycle,
   matrix construction, and platform allocation behavior still need separate
   selected proofs before any broad reliability claim is defensible.

## Final Metrics

### Validation

| Metric | Sprint 200 close state |
| --- | --- |
| selected symbolic LU registration guard | passed on Days 11, 13, and 14 |
| selected symbolic LU allocation-failure gate | passed on Days 10, 12, 13, and 14 with 3 tests, 0 failures, 0 skipped, and 3054 assertions |
| broader symbolic allocation-failure gate | passed on Day 12 with 104 tests, 0 failures, 0 skipped, and 4316 assertions |
| source-list check | passed on Day 12 with 49 library sources |
| docs check | passed on Days 11, 12, 13, and 14 with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages |
| format | passed on Day 12 |
| lint | passed on Day 12 |
| full test suite | passed on Day 12 with `All tests passed.` |
| final `git diff --check` | passed |
| final `make format && make lint && make test` | passed on Day 12 because `.c` and test code changed during the sprint |

### Changed Surface

| Metric | Sprint 200 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Epic residual/retrospective files changed | 2 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Makefile targets added | 1 |
| Python guard files added | 1 |
| Python guard files changed | 1 |
| C implementation files changed | 1 |
| C test files changed | 1 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Owner-selection items completed | 1 |
| Invariant-record items completed | 1 |
| Harness-integration items completed | 1 |
| Regression-test items completed | 1 |
| Focused-gate items completed | 1 |
| Docs-and-validation items completed | 1 |
| Broad allocation-failure claims promoted | 0 |
| State-of-the-art reliability claims promoted | 0 |

The count covers Sprint 200 items 200.1 through 200.6.

## Closed Claim

Sprint 200 closes this bounded claim:

The current branch adds a selected `sparse_symbolic_lu()` allocation-failure
owner proof. The proof covers deterministic allocation-failure status,
requested-output cleanup, stale-output suppression, caller-owned
matrix/permutation preservation, repeated cleanup after failure, and
retry-after-reset behavior for bounded fixtures. The branch also adds a
focused Make gate, a registration guard, claim-safe documentation, integrated
validation evidence, review-hardening traceability, and final closeout.

This claim does not include broad allocation-failure coverage,
`sparse_analyze()` lifecycle cleanup, standalone etree/postorder/colcount
helper allocation-failure coverage, direct solver allocation-failure coverage,
eigensolver or graph allocation-failure coverage, SVD allocation-failure
coverage, sparse matrix construction/conversion/IO allocation-failure
coverage, package/install reliability, generated-tooling reliability, hosted
CI proof for this owner, platform parity, operating-system OOM behavior,
concurrent allocation-hook behavior, release readiness, or state-of-the-art
reliability support.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-candidate-intake.md](./artifacts/day1-candidate-intake.md);
- [day2-owner-selection.md](./artifacts/day2-owner-selection.md);
- [day3-lifecycle-trace.md](./artifacts/day3-lifecycle-trace.md);
- [day4-invariant-record.md](./artifacts/day4-invariant-record.md);
- [day5-harness-design.md](./artifacts/day5-harness-design.md);
- [day6-harness-integration.md](./artifacts/day6-harness-integration.md);
- [day7-failed-allocation-tests.md](./artifacts/day7-failed-allocation-tests.md);
- [day8-cleanup-proof.md](./artifacts/day8-cleanup-proof.md);
- [day9-retry-proof.md](./artifacts/day9-retry-proof.md);
- [day10-focused-gate.md](./artifacts/day10-focused-gate.md);
- [day11-claim-documentation.md](./artifacts/day11-claim-documentation.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| `sparse_analyze()` lifecycle allocation-failure cleanup remains unclaimed | Future selected analysis-lifecycle owner | Record lifecycle invariants, deterministic failure reachability, stale analysis-state suppression, cleanup, retry, focused gate, and claim docs. |
| Standalone etree/postorder/colcount helper allocation failures remain unclaimed | Future helper-family owner | Split helper families into bounded owners and prove failure status, cleanup, and retry without implying broad symbolic analysis. |
| Direct-solver allocation-failure proof remains unclaimed | Future selected direct-solver owner | Select one factor/solve publication surface, document caller-output semantics, add deterministic allocation failures, cleanup, retry, and a focused gate. |
| Sparse matrix construction/insertion allocation failures remain unclaimed | Future selected matrix-construction owner | Isolate one constructor or insertion path and avoid broad sparse-matrix allocation claims until each owner has evidence. |
| Hosted allocation-failure proof remains unclaimed | Future CI owner | Add an explicit hosted workflow for selected reliability gates and promote docs only after hosted evidence and claim wording move together. |
| Broad reliability/state-of-the-art support remains unclaimed | Future multi-sprint evidence owner | Close multiple selected owner families, platform gates, sanitizer/stress lanes, documentation, and support matrices before promoting broad reliability. |

## Next-Sprint Readiness

Sprint 200 leaves one additional allocation-failure owner in a closed,
selected-proof state.

| Future need | Sprint 200 handoff |
| --- | --- |
| Analysis lifecycle proof | Start from the Day 2 deferred candidate and Day 14 residual table; keep it separate from symbolic LU. |
| Helper-level symbolic proof | Use the Day 4 invariant model but scope each helper family independently. |
| Direct-solver proof | Reuse the stale-output and retry checklist, but define solver-output mutation semantics before tests. |
| Matrix construction proof | Treat `sparse_create()` and `sparse_insert()` as shared owner surfaces rather than symbolic LU implementation details. |
| Reliability gate maintenance | Keep `make symbolic-lu-allocation-failure-gate` and `tests/test_symbolic_lu_allocation_failure_gate_registration.py` synchronized with the selected tests. |
| Claim docs | Continue using INSTALL as the public support truth and maintainer guide as the proof-owner ledger. |

## Final Assessment

Sprint 200 is complete as a selected allocation-failure owner proof sprint. It
adds meaningful local reliability evidence for `sparse_symbolic_lu()` and
keeps the claim boundary narrow, tested, documented, and reviewable.

The branch is ready for review as C implementation/test, focused gate,
registration guard, documentation, and planning evidence work for the selected
symbolic LU allocation-failure owner.
