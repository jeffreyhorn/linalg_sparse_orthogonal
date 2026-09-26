# Sprint 210 Retrospective

**Sprint:** 210 - Additional Allocation-Failure Owner Proof  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-210`)  
**Status:** Closed with selected no-reorder linked-list LDLT
allocation-failure owner proof

## Source Artifact Note

Sprint 210 was executed from the Epic 19 project-plan section for Sprint 210
and lives under `docs/planning/EPIC_19/SPRINT_210/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint selected one additional allocation-failure owner outside the
already closed selected symbolic LU path: no-reorder linked-list LDLT numeric
factorization. It added deterministic private-hook allocation failure coverage,
cleanup and stale-output assertions, caller-input preservation checks,
retry-after-reset proof, focused Make/CTest validation, active registration
guard coverage, user/maintainer documentation, integrated validation, and
closeout evidence.

## Definition Of Done Checklist

- [x] Created Sprint 210 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Ranked Epic 19 candidate allocation-failure owners and selected exactly
      one owner: no-reorder linked-list LDLT numeric factorization.
- [x] Recorded lifecycle invariants for status, cleanup, stale-output,
      caller-input preservation, retry, output publication, and retained
      non-claims before code edits.
- [x] Converted selected linked-list LDLT output/workspace allocations to
      private allocation wrappers so deterministic failure injection can reach
      the selected owner.
- [x] Added 25 deterministic fail-after cases covering output arrays, working
      copy setup, output `L`, permutation output, and dense workspaces.
- [x] Added cleanup, free-safe output, stale-output, caller-input
      preservation, retry-after-reset, and success-output cleanup tests.
- [x] Added `make ldlt-linked-list-allocation-failure-gate`.
- [x] Added CTest labels `ldlt;linked_list;allocation_failure` for
      `test_ldlt`.
- [x] Added a registration guard that requires focused gate wiring, active
      proof-owner `RUN_TEST(...)` registrations, representative fail-after
      cases, and key assertions.
- [x] Updated README, INSTALL, maintainer guide, and Epic 19 project-plan
      status with selected-owner wording and retained non-claims.
- [x] Ran focused gate, registration guard, source-list check, docs checks,
      support docs guard, `make format`, `make lint`, `make test`, and final
      whitespace validation.

## What Went Well

1. **The owner boundary stayed narrow.** Sprint 210 selected linked-list LDLT
   with `SPARSE_REORDER_NONE` instead of attempting to prove all LDLT, CSC
   LDLT, Cholesky, or broad direct-solver allocation behavior.

2. **The deterministic sweep is concrete.** The final proof covers 25 named
   fail-after sites across output arrays, working-copy setup, output `L`,
   permutation output, and dense factorization workspaces.

3. **The lifecycle evidence is multi-dimensional.** The tests cover failure
   status, cleanup, repeated free safety, stale-output clearing, caller-owned
   matrix preservation, hook reset, retry success, and baseline equality for
   representative retry cases.

4. **The focused gate is easy to rerun.** Maintainers now have one command,
   `make ldlt-linked-list-allocation-failure-gate`, plus a standalone
   registration guard for drift detection.

5. **Review hardening found a real guard weakness.** Day 13 replaced raw
   substring checks for `RUN_TEST(...)` with active-line exact registration
   checks, preventing commented-out registrations from satisfying the guard.

6. **Public and maintainer docs agree.** README, INSTALL, and the maintainer
   guide all name the selected linked-list LDLT proof and keep broad
   reliability, package, platform, performance, release, ABI, external parity,
   and state-of-the-art claims out of scope.

## What Didn't Go Well

1. **The selected owner required source changes.** Some relevant linked-list
   LDLT allocations used direct `malloc`/`calloc`, so deterministic failure
   proof required converting selected owner sites to private allocation
   wrappers.

2. **The proof still leaves important direct-solver residuals.** CSC LDLT,
   reordered LDLT, Cholesky numeric paths, and broader direct-solver
   allocation behavior remain unproved.

3. **Matrix insertion and matrix-pool allocation remain separate.** The sprint
   deliberately avoided claiming broad sparse-matrix allocation behavior even
   though linked-list LDLT creates and populates an internal `L` matrix.

4. **Full lint remains expensive.** The integrated `make lint` pass is slow
   because clang-tidy and cppcheck cover the full library and test tree, but
   it was necessary once `.c` files changed.

## Final Metrics

### Validation

| Metric | Sprint 210 close state |
| --- | --- |
| `make ldlt-linked-list-allocation-failure-gate` | passed |
| `python3 tests/test_ldlt_allocation_failure_gate_registration.py` | passed |
| `make source-list-check` | passed with 49 library sources |
| `make docs-check` | passed |
| `make support-docs-guard` | passed |
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| final `git diff --check` | passed |

### Focused LDLT Proof Metrics

| Metric | Sprint 210 close state |
| --- | ---: |
| selected allocation-failure owner claims closed | 1 |
| deterministic fail-after cases covered | 25 |
| focused LDLT tests in `test_ldlt` after sprint | 95 |
| focused LDLT gate failures | 0 |
| focused LDLT gate skips | 0 |
| focused LDLT gate assertions | 7781 |
| active proof-owner `RUN_TEST(...)` registrations guarded | 6 |
| broad allocation-failure claims added | 0 |
| public API/ABI declarations changed | 0 |

### Changed Surface

| Metric | Sprint 210 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Build registration files changed | 2 |
| C implementation files changed | 1 |
| C test files changed | 1 |
| Python registration guard files added | 1 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Owner selection items completed | 1 |
| Lifecycle invariant items completed | 1 |
| Harness extension items completed | 1 |
| Regression test items completed | 1 |
| Gate and documentation items completed | 1 |
| Validation and closeout items completed | 1 |
| Selected linked-list LDLT allocation-failure owner claims promoted | 1 |
| Broad allocation-failure, package, ABI, platform, performance, release, external parity, or state-of-the-art claims promoted | 0 |

The count covers Sprint 210 items 210.1 through 210.6.

## Closed Claim

Sprint 210 closes this bounded claim:

Selected no-reorder linked-list LDLT numeric factorization has focused local
deterministic allocation-failure proof for bounded known fixtures covering 25
injected allocation-failure sites, cleanup, stale-output suppression,
caller-input preservation, free-safe output state, repeated cleanup after
failure, and retry-after-reset behavior.

This claim does not include CSC LDLT allocation-failure proof, reordered LDLT
allocation-failure proof, Cholesky allocation-failure proof beyond existing
selected symbolic lanes, broad direct-solver allocation-failure coverage, QR,
SVD, eigensolver, sparse matrix construction, conversion, IO,
package/install, generated-tooling allocation-failure proof, operating-system
OOM behavior, platform parity, hosted CI proof, package-manager proof,
shared-library ABI proof, performance proof, release readiness,
external-library parity, state-of-the-art reliability support, or concurrent
allocation-hook behavior.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-allocation-proof-intake.md](./artifacts/day1-allocation-proof-intake.md);
- [day2-owner-ranking.md](./artifacts/day2-owner-ranking.md);
- [day3-lifecycle-baseline.md](./artifacts/day3-lifecycle-baseline.md);
- [day4-harness-design.md](./artifacts/day4-harness-design.md);
- [day5-harness-implementation.md](./artifacts/day5-harness-implementation.md);
- [day6-failure-sweep.md](./artifacts/day6-failure-sweep.md);
- [day7-cleanup-proof.md](./artifacts/day7-cleanup-proof.md);
- [day8-stale-output-preservation.md](./artifacts/day8-stale-output-preservation.md);
- [day9-retry-proof.md](./artifacts/day9-retry-proof.md);
- [day10-focused-gate.md](./artifacts/day10-focused-gate.md);
- [day11-documentation-calibration.md](./artifacts/day11-documentation-calibration.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| CSC LDLT allocation-failure proof | Future direct-solver reliability owner | Select CSC LDLT explicitly, trace lifecycle invariants, add deterministic injection coverage for CSC-specific allocations, cleanup, stale-output, preservation, retry, gate, docs, and full validation. |
| Reordered LDLT allocation-failure proof | Future LDLT/reorder owner | Include reorder allocation, permuted matrix lifecycle, telemetry, selected backend behavior, cleanup, stale-output, retry, and guard coverage. |
| Cholesky numeric allocation-failure proof | Future Cholesky owner | Select a numeric Cholesky path distinct from symbolic Cholesky, define output/in-place semantics, add injection coverage, and preserve Cholesky non-claims. |
| Broad direct-solver allocation reliability | Future reliability epic | Define solver-family ownership model and close owners one at a time rather than inferring from linked-list LDLT. |
| Matrix construction/conversion/IO allocation proof | Future sparse-matrix owner | Select one constructor/conversion/IO owner, define caller-visible output semantics, and add deterministic failure plus cleanup/retry tests. |
| QR/SVD/eigensolver workspace allocation proof | Future numerical workspace owner | Select exactly one workspace/output owner, document lifecycle invariants, add failure injection and focused gate coverage. |
| Package, ABI, platform, performance, release, external parity, and state-of-the-art evidence | Future productization/release owners | Add provider- or methodology-specific proof, docs, and guards before promoting any broad support claim. |

## Next-Sprint Readiness

Sprint 210 leaves one additional selected allocation-failure proof closed and
keeps the remaining allocation reliability work explicitly residual.

| Future need | Sprint 210 handoff |
| --- | --- |
| Current Epic 19 status | Start from `docs/planning/EPIC_19/PROJECT_PLAN.md`, which marks Sprint 210 closed and Sprints 211-216 pending. |
| Allocation-failure owner changes | Review Day 2, Day 3, Day 10, Day 13, and Day 14 artifacts before adding or widening another owner. |
| Focused linked-list LDLT validation | Run `make ldlt-linked-list-allocation-failure-gate` and `python3 tests/test_ldlt_allocation_failure_gate_registration.py`. |
| Docs/support validation | Run `make docs-check`, `make support-docs-guard`, and `git diff --check`. |
| Source or header changes | Run `make format && make lint && make test` before closeout. |
| Retrospective source material | Use `WORKING_NOTES.md` and Day 1-Day 14 artifacts under `SPRINT_210/artifacts/`. |

## Final Assessment

Sprint 210 improves allocation-failure reliability evidence by closing one
selected direct-solver owner end to end. The work is deliberately narrow:
linked-list LDLT no-reorder numeric factorization now has deterministic
allocation-failure, cleanup, stale-output, preservation, retry, gate, and
documentation evidence, but broad allocation reliability remains future work.
