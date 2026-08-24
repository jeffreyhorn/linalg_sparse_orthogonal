# Sprint 178 Retrospective

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Duration:** 14 days (Days 1-14 landed on branch `sprint-178`)
**Status:** Complete

## Source Artifact Note

Sprint 178 was executed from the Epic 16 project-plan section for Sprint 178
and lives under `docs/planning/EPIC_16/SPRINT_178/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 178 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reconciled Sprint 177 Gate 1 and Sprint 177 Day 12 handoff before
      implementation.
- [x] Inventoried allocation-heavy candidate surfaces and selected exactly one
      additional subsystem.
- [x] Selected `sparse_matmul()` workspace allocation as the Sprint 178 proof
      target.
- [x] Documented cleanup, no-publication, retry, ownership, and unsupported
      breadth invariants before implementation.
- [x] Reused the private allocation-failure hook without adding public
      test-injection API.
- [x] Added deterministic regression coverage for the selected accumulator,
      nonzero-flag, and touched-column workspace allocations.
- [x] Added stale-output suppression and retry-after-reset assertions for the
      selected failure paths.
- [x] Added error-precedence coverage for `sparse_matmul()` output-parameter
      rejection and stale-output clearing.
- [x] Added `make matmul-allocation-failure-gate`.
- [x] Added `matmul;allocation_failure` CTest labels for `test_matmul`.
- [x] Added a Python registration guard for Makefile, CMake, and test
      registration drift.
- [x] Updated README and maintainer guidance with scoped allocation-failure
      wording.
- [x] Ran focused, CMake/CTest, documentation, and full C quality validation.
- [x] Preserved broad allocation-failure, state-of-the-art, external parity,
      package/install, generated tooling, public hook, and unrelated allocation
      non-claims.

## What Went Well

1. **The sprint closed one additional allocation-failure gap end to end.**
   Sprint 178 moved from a broad Epic 16 residual to a selected,
   deterministic `sparse_matmul()` workspace allocation proof with tests,
   focused gates, docs, and closeout evidence.

2. **The selected scope stayed narrow.** The proof covers only the accumulator,
   nonzero-flag, and touched-column workspace allocations. Adjacent matrix
   shell allocation, product flush, conversions, solvers, package/install, and
   generated tooling remained explicit non-claims.

3. **The existing private hook was enough.** The Sprint 176 fail-at-count hook
   could target the selected workspace allocations without product API changes
   or allocation-helper redesign.

4. **The tests assert user-visible safety, not just error returns.** The
   regression suite checks `SPARSE_ERR_ALLOC`, clears stale output pointers,
   preserves caller-owned stale matrices, retries after reset, and verifies the
   expected product.

5. **Validation is now easy to run locally.** `make
   matmul-allocation-failure-gate` gives maintainers a single focused command,
   while the CTest labels expose both `matmul` and `allocation_failure`
   selectors.

6. **Registration drift is guarded.** The Python guard checks the Make target,
   CMake label, test registration, and selected regression names before the
   focused Make gate runs the executable.

## What Didn't Go Well

1. **The proof still depends on call-count stability.** The selected
   fail-after values are intentionally documented and guarded, but future
   allocation ordering changes in `sparse_matmul()` will require deliberate
   test updates.

2. **The allocation-failure evidence remains intentionally narrow.** Sprint
   178 widened proof by one subsystem, but broad sparse matrix construction,
   conversion, solver, package/install, and generated-tooling allocation
   failure behavior remains residual.

3. **The claim surface remains distributed.** README, maintainer guide, CMake,
   Makefile, tests, and sprint artifacts all carry parts of the same evidence
   boundary.

4. **The full C gate remains expensive.** The required `make format && make
   lint && make test` path is appropriate for source changes, but it continues
   to dominate validation time.

5. **The selected path did not require product cleanup changes.** That is a
   good outcome for safety, but it means future higher-risk allocation work may
   still uncover actual product-code defects.

## Final Metrics

### Validation

| Metric | Sprint 178 close state |
| --- | --- |
| focused matrix multiply allocation gate | passed: `make matmul-allocation-failure-gate` |
| registration guard | passed: `python3 tests/test_matmul_allocation_failure_gate_registration.py` |
| focused CMake build | passed: `test_matmul` and `test_iterative` built in `build-sprint178-day12` |
| CTest `matmul` selector | passed: 1 of 1 test |
| CTest `allocation_failure` selector | passed: 2 of 2 tests |
| docs terminology hygiene | passed: only intentional anti-drift references to `allocator-failure` remain |
| final docs whitespace hygiene | passed: `git diff --check` |
| full C quality gate | passed: `make format && make lint && make test` |

### Changed Surface

| Metric | Sprint 178 close state |
| --- | ---: |
| C source files changed | 1 |
| public header files changed | 0 |
| internal header files changed | 0 |
| Make targets added | 1 |
| CMake labels added | 1 |
| Python registration guards added | 1 |
| focused allocation-failure tests added | 4 |
| public/maintainer docs changed | 2 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |

### Claim Governance

| Metric | Sprint 178 close state |
| --- | ---: |
| selected allocation-failure subsystem claims added | 1 |
| broad allocation-failure claims added | 0 |
| public allocation-failure API claims added | 0 |
| state-of-the-art claims added | 0 |
| external-library parity claims added | 0 |
| package/install allocation claims added | 0 |
| generated-tooling allocation claims added | 0 |

## Closed Claim

Sprint 178 closes this Epic 16 allocation-failure evidence claim:

`sparse_matmul()` workspace allocation has deterministic allocation-failure
cleanup evidence for the selected accumulator, nonzero-flag, and touched-column
workspace allocations. The proof covers stale-output suppression and
retry-after-reset behavior under the private allocation-failure hook.

This does not claim broad allocation-failure cleanup coverage across matrix
shell construction, insertion/product flush, conversions, solver families,
package/install flows, generated tooling, public allocation-failure APIs, or
unrelated allocation paths.

## Follow-Up Risks

1. **Matrix construction and conversion remain residual.** Treat those as a
   separate future selected-subsystem proof if they become the highest-value
   allocation gap.

2. **Solver-family allocation failures remain residual.** Each direct solver,
   decomposition, eigensolver, graph, or reorder family needs its own scoped
   proof before any wording can widen.

3. **Generated tooling allocation behavior remains out of scope.** Do not mix
   generated-report or generated API tooling allocation work with numerical
   kernel allocation-failure proof.

4. **Registration drift still needs guarding as targets grow.** Future focused
   gates should include comparable Make/CMake/test registration checks.

5. **Call-count proof needs maintenance discipline.** If `sparse_matmul()`
   allocation ordering changes, update fail-after constants and artifacts
   together.

## Sprint 179 Readiness

Sprint 179 should begin from the Epic 16 project-plan section
`Sprint 179: Generated API HTML Publication Decision`.

The highest-value next action is to audit Doxygen inputs, ignored outputs,
warnings, page coverage, source-header authority, and current docs navigation
before deciding whether generated API HTML should be hosted, retained as a CI
artifact, committed, or explicitly kept local-only.
