# Sprint 178 Day 1: Sprint Intake And Gate Baseline

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Purpose

Establish the Sprint 178 baseline before allocation-heavy subsystem
selection begins. Day 1 records the source-plan authority, acceptance gate,
handoff inputs, current proof status, protected non-claims, and artifact
layout for the rest of the sprint.

## Source Authority

The active Sprint 178 project-plan section is:

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- section: `Sprint 178: Allocation-Failure Proof Batch 2`

The sprint artifact path is:

- `docs/planning/EPIC_16/SPRINT_178/`

## Starting Snapshot

| Field | Value |
| --- | --- |
| Branch | `sprint-178` |
| Starting commit | `3907e7545a58c462e24eb3d0d4df1ef7a75589bf` |
| Source project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Sprint plan path | `docs/planning/EPIC_16/SPRINT_178/PLAN.md` |
| Working notes path | `docs/planning/EPIC_16/SPRINT_178/WORKING_NOTES.md` |
| Artifact directory | `docs/planning/EPIC_16/SPRINT_178/artifacts/` |

## Recent Prior PR Context

| Commit | Context |
| --- | --- |
| `3907e754` | Merged PR #197 from Sprint 177. |
| `4bca0a10` | Addressed PR #197 review comments by removing prompt line-number wording from Sprint 177 notes. |
| `aad776d9` | Moved Sprint 177 planning artifacts from Epic 15 to Epic 16. |
| `6ca0e39a` | Added Sprint 177 planning artifacts. |
| `8e5a759f` | Merged Epic 16 planning review, todo, and project plan. |

## Acceptance Gate Baseline

Sprint 177 Gate 1 defines the Sprint 178 pass/fail contract:

| Field | Sprint 178 baseline |
| --- | --- |
| Target | Allocation-Failure Proof Batch 2. |
| Required evidence | Deterministic injected allocation failure covers one additional subsystem, proves cleanup on failure, proves no stale public state publication, and proves successful retry after reset. |
| Validation commands | Focused subsystem gate; `make format`; `make lint`; `make test`; CMake/CTest validation if test registration changes. |
| Pass definition | The selected subsystem has a named fail-at-count or equivalent harness, at least one failure case per selected ownership path, cleanup assertions, recovery assertions, and a focused Make/CTest entry or label. |
| Fail definition | Failures are nondeterministic, cleanup cannot be asserted, public state can be partially published, retry behavior is unproven, or docs imply broad allocation-failure coverage. |
| Claim boundary | One additional named subsystem has deterministic allocation-failure cleanup evidence. |
| Protected non-claims | No broad allocation-failure guarantee across all solvers, constructors, package/install flows, generated tooling, or unrelated allocation paths. |

## Current Allocation-Failure Proof

| Surface | Current state |
| --- | --- |
| Private hook owners | `src/sparse_alloc_internal.c`, `src/sparse_alloc_internal.h` |
| Hook controls | `sparse_alloc_test_fail_after(long remaining)` and `sparse_alloc_test_reset()` |
| Hook scope | Private/internal helper layer; not public API. |
| Countdown behavior | A zero countdown fails the next wrapped allocation once, then resets injection; positive countdowns decrement before the single injected failure. |
| Existing proof target | Iterative repeated-run handle prepare/growth cleanup. |
| Existing covered APIs | CG, GMRES, and MINRES repeated-run handle paths. |
| Existing Make gate | `make iterative-allocation-failure-gate` |
| Existing CTest selector | `ctest -L allocation_failure` through the `test_iterative` label. |
| Existing docs boundary | README and maintainer guide describe selected, family-local coverage only. |

## Candidate Surface Starting List

Day 1 does not select the subsystem. Day 2 should compare these candidate
families before Day 3 selection:

| Candidate | Initial owner files | Day 1 note |
| --- | --- | --- |
| Matrix construction/conversion | `src/sparse_matrix.c`, `src/sparse_matrix_build_internal.c`, `src/sparse_csr.c`, public matrix headers | Good user value and public-state relevance; cleanup observability must be checked carefully. |
| Direct solver setup/factorization | `src/sparse_lu.c`, `src/sparse_lu_csr.c`, `src/sparse_ldlt*.c`, `src/sparse_qr.c` | High claim value but potentially larger blast radius; selection must stay narrow. |
| Decomposition workspace owner | QR, LDLT, Cholesky, or SVD workspace/factor owners | Useful cleanup proof if failure sites are deterministic and retry behavior is observable. |

## Protected Non-Claims

Sprint 178 must keep these unsupported claims out of README, maintainer docs,
and sprint artifacts unless evidence is actually added:

- broad allocation-failure safety across all allocation paths;
- broad direct-solver allocation-failure cleanup coverage;
- broad eigensolver allocation-failure cleanup coverage;
- broad matrix construction coverage unless matrix construction is the single
  selected subsystem and fully proven;
- package/install allocation-failure coverage;
- generated-report or generated API tooling allocation-failure coverage;
- state-of-the-art reliability, external parity, shared-library ABI,
  package-manager, dynamic ABI, runtime-loader, release, or broad platform
  claims.

## Day 1 Decisions

- Treat `docs/planning/EPIC_16/PROJECT_PLAN.md` as the Sprint 178 source
  authority.
- Use Sprint 177 Gate 1 as the closeout acceptance contract.
- Use Sprint 177 Day 12 as the implementation handoff.
- Do not select a subsystem on Day 1.
- Keep the existing Sprint 176 iterative allocation-failure lane as the
  reference pattern, not as evidence for new subsystem coverage.
- Require the full C quality gate if C source or header files change later in
  the sprint.

## Day 1 Deliverables

- `docs/planning/EPIC_16/SPRINT_178/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_178/artifacts/day1-sprint-intake.md`
- acceptance-gate summary
- current proof and non-claim baseline
- starting candidate surface list

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 178 scope is tied to the Epic 16 project plan | Complete | Source authority recorded above and in working notes. |
| Gate 1 pass/fail requirements are visible before selection begins | Complete | Acceptance gate baseline table records required evidence, pass/fail, validation, and claim boundary. |
| Broad allocation-failure claims remain rejected | Complete | Protected non-claims section keeps broad coverage out of scope. |
