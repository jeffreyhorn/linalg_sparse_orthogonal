# Sprint 193 Day 2: Candidate Ranking

## Summary

Day 2 ranked large source/test review-surface candidates and recommends one
cluster for Day 3 confirmation: the `tests/test_qr.c` external dense-reference
rank/nullspace/threshold block.

The recommendation is intentionally test-only and header-only. It aims to
reduce the largest remaining C test surface while preserving the existing
`test_qr` proof-owner binary and avoiding Make/CMake/library source-list
churn.

## Ranking Criteria

| Criterion | Meaning |
| --- | --- |
| Size payoff | Expected reduction in a large review surface. |
| Helper cohesion | Whether the candidate can move as one understandable ownership boundary. |
| Current coverage | Whether existing tests already prove behavior before and after movement. |
| User-facing importance | How much the surface supports public numerical credibility. |
| Algorithm risk | Risk of changing solver behavior, tolerance behavior, or numerical semantics. |
| Cleanup/global-state risk | Risk around allocations, environment variables, process-global overrides, or early returns. |
| Registration risk | Risk of Make/CMake/source-list drift. |
| Sprint fit | Likelihood of complete closure inside Sprint 193. |

## Ranked Candidates

| Rank | Candidate | Decision | Rationale |
| ---: | --- | --- | --- |
| 1 | `tests/test_qr.c` external dense-reference rank/nullspace/threshold block | Recommended | Largest remaining C test file, cohesive external-reference ownership, strong numerical credibility, and likely header-only movement without registration churn. |
| 2 | `tests/test_svd.c` external dense-reference fixture/helper block | Alternate | Very cohesive and lower risk, but smaller immediate line-reduction payoff unless scope expands into partial SVD helpers. |
| 3 | `tests/test_chol_csc_supernodal.c` dense backend/env-contract block | Alternate | Focused and important, but environment cleanup and backend state make it riskier. |
| 4 | `tests/test_chol_csc.c` external dense-reference block | Alternate | Good boundary candidate, but smaller payoff than QR. |
| 5 | `tests/test_etree.c` selected fixture/helper cluster | Deferred | High helper density, but no single seam is as clearly bounded yet. |
| 6 | `tests/test_integration.c` lifecycle/fixture cluster | Deferred | Broad cross-solver behavior risks a multi-cluster refactor. |
| 7 | `tests/test_iterative.c` CG/GMRES helper cluster | Deferred | Convergence, allocation, and handle lifetime behavior increase extraction risk. |
| 8 | Further `tests/test_ldlt_csc.c` extraction | Deferred | Sprint 185 already closed the highest-value LDLT CSC test-helper reduction. |
| 9 | Production `src/*.c` extraction | Deferred | Production source movement has higher public-behavior and source-list risk than needed for Sprint 193. |

## Recommended Cluster Details

The recommended QR cluster includes the external dense-reference tests for:

- rank-1 nullspace projector;
- rank-deficient duplicate-column nullspace projector;
- rank-deficient dependent-row nullspace projector;
- rank-deficient wide nullspace subspace;
- diagonal rank-threshold family;
- scaled diagonal rank-threshold family;
- perturbed duplicate-column rank-threshold family;
- perturbed dependent-row rank-threshold family.

It also includes the local external-reference reader helpers and the wide
rank-deficient fixture helper that directly support those tests.

## Why This Cluster Fits Sprint 193

| Requirement | Fit |
| --- | --- |
| One cluster | The cluster is one contiguous QR external-reference ownership area. |
| Behavior-preserving | Test names, `RUN_TEST(...)` order, fixtures, tolerances, skip behavior, Python command strings, and diagnostics can remain unchanged. |
| Low registration risk | A header-only helper extraction can preserve the existing `test_qr` binary and avoid Make/CMake source-list updates. |
| Guardable | A QR helper guard can follow the Sprint 185 `ldlt-csc-helper-guard` pattern. |
| Review-surface value | The selected cluster removes a large, specialized proof block from the largest remaining C test file. |
| Focused validation | `make build/test_qr` and `./build/test_qr` directly own the selected behavior before the full C/H quality gate. |

## Rejected Scope Expansions

- Do not move general QR householder, sparse-mode, economy-mode, reorder, or
  refinement tests in Sprint 193.
- Do not change `src/sparse_qr.c`, `include/sparse_qr.h`, public QR APIs, or
  numerical tolerance policy.
- Do not add a new QR test binary unless Day 3 discovers that a header-only
  boundary cannot preserve the proof-owner model.
- Do not merge QR helpers into shared solver helpers; keep the boundary
  family-local.

## Day 3 Recommendation

Day 3 should select the QR external dense-reference cluster if the invariant
contract can preserve:

- fixture keys and Python reference command strings;
- Windows skip behavior and messages;
- expected ranks, nullities, thresholds, perturbations, projector layout, and
  tolerance constants;
- `RUN_TEST(...)` ordering in `tests/test_qr.c`;
- existing `test_qr` registration in Make and CMake;
- helper-header absence from standalone test/library registration.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '50,112p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '1,220p' tests/test_qr.c
sed -n '1,220p' tests/test_svd.c
sed -n '1,220p' tests/test_chol_csc_supernodal.c
sed -n '1,220p' tests/test_chol_csc.c
for f in tests/test_qr.c tests/test_svd.c tests/test_chol_csc_supernodal.c tests/test_chol_csc.c tests/test_etree.c tests/test_integration.c tests/test_iterative.c tests/test_graph.c; do printf '\n%s\n' "$f"; rg -n '^/\* [=-]|^static void test_|RUN_TEST\(' "$f" | head -n 80; done
for f in tests/test_qr.c tests/test_svd.c tests/test_chol_csc_supernodal.c tests/test_chol_csc.c tests/test_etree.c tests/test_integration.c tests/test_iterative.c tests/test_graph.c tests/test_ldlt.c; do printf '%s\tRUN_TEST=' "$f"; rg --count 'RUN_TEST\(' "$f"; done
sed -n '1120,2225p' tests/test_qr.c
sed -n '1,220p' tests/test_qr_helpers.h
sed -n '2860,3975p' tests/test_qr.c
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
