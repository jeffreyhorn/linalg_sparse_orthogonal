# Sprint 201 Day 2: Candidate Ranking

## Summary

Day 2 ranks large review-surface candidates and recommends a Day 3 shortlist.
The leading recommendation is a selected `tests/test_svd.c` helper cluster,
preferably one that moves cohesive full-SVD/rank/pseudoinverse/low-rank helper
logic into a family-local helper header without changing the `test_svd` proof
owner binary.

This is a recommendation only. Day 3 must still select exactly one cluster and
freeze no-behavior-change boundaries before any code extraction begins.

## Ranking Criteria

| Criterion | Meaning |
| --- | --- |
| Size payoff | Expected reduction in a large review surface. |
| Reviewer burden | How much the current file structure slows review or obscures ownership. |
| Ownership clarity | Whether a cohesive helper/test family can be named and separated. |
| Helper cohesion | Whether moved functions naturally depend on each other rather than many unrelated test families. |
| Behavior-risk exposure | Risk of changing solver behavior, ordering, tolerances, diagnostics, cleanup, or process-global state. |
| Current coverage | Whether existing focused tests already prove behavior before and after movement. |
| Registration risk | Risk of Make/CMake/source-list drift. |
| Sprint fit | Likelihood of complete closure inside Sprint 201. |

## Ranked Candidates

| Rank | Candidate | Decision | Rationale |
| ---: | --- | --- | --- |
| 1 | `tests/test_svd.c` selected full-SVD/helper cluster | Recommended | Large remaining test surface, high RUN_TEST density, existing `tests/test_svd_helpers.h` precedent, and likely header-only extraction without production or CMake source-list churn. |
| 2 | `tests/test_chol_csc.c` selected helper/fixture cluster | Alternate | High helper density and direct-solver value; likely family-local helper extraction, but more intertwined with CSC factor/solve behavior. |
| 3 | `tests/test_chol_csc_supernodal.c` selected helper/fixture cluster | Alternate | Focused supernodal surface with existing helper header and clear sections; backend/env contracts raise process-global cleanup risk. |
| 4 | Remaining `tests/test_qr.c` economy or sparse-mode cluster | Alternate | Still large and already has guard precedent, but Sprint 193 just reduced QR external-reference surface and remaining clusters are behavior-sensitive. |
| 5 | `tests/test_ldlt.c` selected direct-solver helper cluster | Deferred | Large and important, but tolerance, inertia, backend, and solve/refine coverage make no-behavior-change boundaries harder. |
| 6 | `tests/test_ldlt_csc.c` additional helper cluster | Deferred | Largest current test surface, but Sprint 185 already reduced the highest-value LDLT CSC helper areas and further movement may have lower clarity. |
| 7 | `tests/test_etree.c` selected fixture/helper cluster | Deferred | High helper density, but Sprint 200 just added selected symbolic LU reliability proof; avoid disturbing fresh proof ownership unless no better candidate exists. |
| 8 | `tests/test_integration.c` selected lifecycle fixture cluster | Deferred | Broad cross-solver lifecycle tests risk a multi-cluster refactor and harder behavior-preservation review. |
| 9 | `tests/test_iterative.c` selected helper cluster | Deferred | Large, but convergence, allocation-failure, repeated-run handles, and process-global behavior increase extraction sensitivity. |
| 10 | `src/sparse_ldlt_csc.c` production module extraction | Deferred | Only current >2000-line production source, but source-list, CMake, ABI-adjacent review, and behavior risks are higher than needed for this sprint. |
| 11 | `scripts/run_external_comparison.py` or `tests/test_normalize_report_index.py` Python split | Deferred | Large surfaces, but outside the primary C solver/test examples and tied to generated evidence semantics rather than solver reviewability. |

## Recommended SVD Candidate Details

The preferred Day 3 candidate is a selected SVD helper cluster inside
`tests/test_svd.c`. Candidate subclusters include:

| Subcluster | Candidate contents | Day 2 assessment |
| --- | --- | --- |
| Full SVD reconstruction and orthogonality helpers | fixture builders, reconstruction helpers, U/V orthogonality helpers, and full-SVD assertion helpers already adjacent to existing `tests/test_svd_helpers.h` utilities | Strong fit if Day 3 finds a clean set of static helpers that can move without moving `RUN_TEST(...)` registrations. |
| Rank/pseudoinverse/low-rank helper block | rank-deficient fixture helpers, Moore-Penrose helper logic, low-rank dense/sparse comparison helpers | Strong alternate; user-visible numerical behavior is important, but tolerances must be preserved exactly. |
| Partial SVD helper consolidation | interactions between `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, and `tests/test_svd_partial_shared_helpers.h` | Lower fit for Sprint 201 because the existing partial helper header is already large and further movement may become helper-to-helper churn. |

Preferred extraction shape for Day 3 evaluation:

- keep `tests/test_svd.c` as the registered proof-owner binary;
- move only selected static helper definitions or tightly coupled tests into a
  family-local helper header if the existing project pattern supports it;
- preserve all `RUN_TEST(...)` registrations, test names, fixture values,
  random-free deterministic data, tolerances, skip behavior, and diagnostics;
- avoid Make/CMake/source-list changes unless Day 3 selects a compiled helper
  or new proof-owner binary, which is not preferred.

## Alternate Candidate Notes

| Candidate | Why not ranked first | Future handoff |
| --- | --- | --- |
| `tests/test_chol_csc.c` helper cluster | Strong helper density, but factor/solve/dispatch sections are intertwined with CSC direct-solver behavior. | Good fallback if SVD helper boundaries are too diffuse. |
| `tests/test_chol_csc_supernodal.c` helper cluster | Existing helper precedent, but dense backend env-contract tests and process-global state need careful reset invariants. | Select only a fixture/helper-only section if chosen. |
| Remaining QR cluster | Guard precedent is excellent, but another QR extraction immediately after Sprint 193 risks over-focusing one subsystem. | Choose only if SVD/Cholesky candidates fail Day 3 boundary checks. |
| LDLT/LDLT CSC cluster | Large review payoff, but direct-solver tolerance and backend behavior are more sensitive. | Better future sprint after selecting one very specific solver subcluster. |
| Etree cluster | Large and helper-dense, but recent symbolic LU allocation-failure proof should remain undisturbed. | Revisit after Sprint 200 review settles. |
| Integration cluster | High line count, but cross-solver lifecycle tests are intentionally broad. | Split only after a fixture-only boundary is proven. |
| Production source extraction | Potential maintainability value, but behavior and registration risks exceed Day 2 preferred path. | Reserve for a dedicated production-module sprint. |

## Preferred Shortlist For Day 3

| Priority | Cluster | Day 3 decision question |
| ---: | --- | --- |
| 1 | `tests/test_svd.c` selected helper cluster | Can a cohesive SVD helper/test block move to a family-local helper header while preserving `test_svd` as the proof-owner binary? |
| 2 | `tests/test_chol_csc.c` selected helper cluster | Can a bounded fixture/helper section move without touching factor/dispatch semantics? |
| 3 | `tests/test_chol_csc_supernodal.c` selected helper cluster | Can a helper-only section move without changing backend env-contract cleanup or process-global behavior? |
| 4 | Remaining `tests/test_qr.c` economy/sparse-mode cluster | Is there a non-Sprint-193 QR cluster with clear guardable ownership and enough payoff? |

## Deferred Scope Expansions

- Do not reduce multiple large files in Sprint 201.
- Do not change production solver behavior, public API, public headers, ABI,
  status codes, diagnostics, numerical tolerances, random seeds, or expected
  outputs.
- Do not move `RUN_TEST(...)` registrations out of the selected proof-owner
  binary unless Day 3 explicitly changes the proof-owner model.
- Do not select `tests/test_etree.c` without protecting Sprint 200 symbolic LU
  allocation-failure proof ownership.
- Do not turn review-surface reduction into package, platform, performance,
  release, or state-of-the-art claims.

## Item 201.1 Evidence

Item 201.1 is complete for Day 2: current >2000-line source/test surfaces were
ranked by review risk, ownership clarity, and extraction feasibility. Day 3
should choose one cluster from the preferred shortlist and record the
behavior-preservation contract before implementation.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 2 validation command.
