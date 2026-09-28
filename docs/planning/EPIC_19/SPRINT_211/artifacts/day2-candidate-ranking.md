# Sprint 211 Day 2: Candidate Ranking

## Summary

Day 2 ranks the current large review-surface candidates and selects one primary
cluster for Day 3 boundary tracing. The selected primary candidate is a bounded
`tests/test_ldlt_csc.c` native/parity helper cluster. The fallback is a
selected `tests/test_chol_csc.c` fixture/helper cluster.

This is still a ranking and selection record, not an extraction. Day 3 must
trace the selected cluster and freeze the no-behavior-change boundary before
implementation begins.

## Ranking Criteria

| Criterion | Meaning |
| --- | --- |
| Size payoff | Expected reduction in a large review surface. |
| Reviewer burden | How much the current file structure slows review. |
| Ownership clarity | Whether a cohesive helper/test family can be named and separated. |
| Helper cohesion | Whether moved helpers naturally depend on each other. |
| Behavior-risk control | Higher score means lower risk of changing solver behavior, diagnostics, tolerances, registration order, or process-global state. |
| Existing guard leverage | Existing guard patterns or helper-owner checks reduce sprint risk. |
| Focused validation availability | Focused tests or guards can prove behavior after movement. |
| Sprint fit | Likelihood of complete closure inside Sprint 211. |

## Measurement Record

Line counts were collected on Day 1. Day 2 added static-function and
`RUN_TEST(...)` density signals:

```sh
for f in tests/test_ldlt_csc.c tests/test_ldlt.c tests/test_etree.c \
  tests/test_integration.c tests/test_qr.c tests/test_iterative.c \
  tests/test_graph.c tests/test_svd.c tests/test_chol_csc.c \
  tests/test_chol_csc_supernodal.c tests/test_reorder_nd.c \
  tests/test_eigs.c tests/test_colamd.c; do
    printf '%s\t' "$f"
    rg -c '^static .*\(' "$f" | tr -d '\n'
    printf '\t'
    rg -c 'RUN_TEST\(' "$f" | tr -d '\n'
    printf '\n'
done
```

| File | Lines | Static-function signal | `RUN_TEST` count |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3469 | 119 | 100 |
| `tests/test_ldlt.c` | 3444 | 115 | 95 |
| `tests/test_etree.c` | 3306 | 129 | 107 |
| `tests/test_integration.c` | 3279 | 53 | 58 |
| `tests/test_qr.c` | 3040 | 72 | 79 |
| `tests/test_iterative.c` | 2929 | 94 | 85 |
| `tests/test_graph.c` | 2764 | 68 | 61 |
| `tests/test_svd.c` | 2658 | 90 | 114 |
| `tests/test_chol_csc.c` | 2554 | 111 | 92 |
| `tests/test_chol_csc_supernodal.c` | 2504 | 72 | 62 |
| `tests/test_reorder_nd.c` | 2304 | 47 | 35 |
| `tests/test_eigs.c` | 2155 | 51 | 43 |
| `tests/test_colamd.c` | 2017 | 78 | 70 |

## Ranked Candidates

| Rank | Candidate | Decision | Rationale |
| ---: | --- | --- | --- |
| 1 | `tests/test_ldlt_csc.c` selected native/parity helper cluster | Selected primary | Largest review surface, high direct-solver value, current helper headers, and existing `ldlt-csc-helper-guard` make complete guarded extraction plausible. |
| 2 | `tests/test_chol_csc.c` selected fixture/helper cluster | Fallback | Large and helper dense with focused test binary; lacks the ready helper guard coverage already present for LDLT CSC. |
| 3 | `tests/test_chol_csc_supernodal.c` helper/fixture cluster | Alternate | Existing helper header helps, but dense backend/env-contract behavior has higher process-global risk. |
| 4 | Remaining `tests/test_qr.c` helper cluster | Alternate | Important and still large, but prior QR extraction and QR evidence lanes make boundary selection more delicate. |
| 5 | `tests/test_svd.c` remaining selected helper cluster | Deferred | Sprint 201 already closed one selected SVD helper surface; preserve current selected/shared helper ownership. |
| 6 | `tests/test_ldlt.c` cluster outside allocation proof | Deferred | Large but recently modified by Sprint 210; avoid disturbing the new focused allocation-failure gate. |
| 7 | `tests/test_etree.c` cluster outside symbolic LU proof | Deferred | Large and helper dense, but recent symbolic LU allocation proof must remain isolated. |
| 8 | Python report tooling cluster | Deferred | Useful maintainability target, but CLI/schema/artifact behavior is a different sprint shape. |
| 9 | Graph/reorder test cluster | Deferred | Large but validation can be slow and heuristic behavior needs careful preservation. |
| 10 | Production `src/sparse_ldlt_csc.c` extraction | Deferred | High payoff but source-list, CMake, and behavior risk are too high for this sprint's first-choice path. |

## Selected Primary Candidate

Sprint 211 selects a bounded `tests/test_ldlt_csc.c` native/parity helper
cluster for Day 3 boundary tracing.

Initial hypothesis:

- keep `tests/test_ldlt_csc.c` as the proof-owner binary;
- preserve all `RUN_TEST(...)` registrations and order;
- move only helper or assertion bodies if Day 3 finds a cohesive native/parity
  cluster;
- prefer existing helper owner headers:
  `tests/test_ldlt_csc_fixtures.h`,
  `tests/test_ldlt_csc_oracle_helpers.h`, or
  `tests/test_ldlt_csc_supernode_helpers.h`;
- extend `scripts/check_ldlt_csc_helper_guard.sh` if the selected cluster
  needs stronger ownership, dependency, prerequisite, or registration checks.

## Fallback Candidate

If the LDLT CSC cluster has no clean extraction boundary, Sprint 211 falls back
to a selected `tests/test_chol_csc.c` fixture/helper cluster.

Fallback conditions:

- the selected LDLT CSC native/parity helpers cannot move without broad source
  churn;
- existing LDLT CSC helper headers would get confused ownership;
- focused LDLT CSC validation is too broad for a selected cluster;
- Day 3 finds behavior-preservation invariants that are too hard to prove
  before implementation.

## Deferred Scope Expansions

- Do not reduce multiple large files in Sprint 211.
- Do not move production source unless Day 3 explicitly rejects the test-helper
  path and records a safer production boundary.
- Do not disturb Sprint 201 SVD helper guard or Sprint 210 LDLT allocation
  proof ownership.
- Do not change public APIs, ABI, status codes, diagnostics, numerical
  tolerances, fixture data, skip behavior, or generated artifact formats.
- Do not turn one selected reduction into broad review-surface cleanup,
  performance, package, platform, release, external-library parity, or
  state-of-the-art claims.

## Item 211.1 Evidence

Item 211.1 is complete for Day 2: current large C tests, C implementation
files, helper headers, and Python tooling candidates were ranked by risk,
review burden, ownership clarity, guard leverage, and extraction feasibility.
Day 3 should trace the selected `tests/test_ldlt_csc.c` native/parity helper
cluster and freeze exact behavior-preservation boundaries before code edits.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 2 validation command.
