# Sprint 194 Day 8 Direct and Iterative Diagnostics Cleanup

## Objective

Apply the Day 7 diagnostics wording contract to direct and iterative solver
documentation without changing solver behavior, public status codes, result
structs, or tolerance semantics.

## Inputs

- `docs/planning/EPIC_17/SPRINT_194/artifacts/day7-diagnostics-wording-contract.md`
- `README.md`
- `docs/solver_selection.md`
- `docs/cookbook.md`
- `examples/README.md`
- `docs/maintainer_guide.md`
- public direct and iterative headers under `include/`

## Direct-Solver Wording Changes

| Location | Before | After |
| --- | --- | --- |
| `README.md` workflow chooser | LU and Cholesky bullets named `SPARSE_ERR_SINGULAR` and `SPARSE_ERR_NOT_SPD` without saying the diagnostic owner. | LU and Cholesky bullets now state that those statuses arrive through factorization or solve return codes. |
| `docs/solver_selection.md` diagnostics table | "Factorization/solve return code" compressed the direct diagnostic owner. | The table now says "Factorization or solve return code" and keeps residuals problem-local. |
| `docs/solver_selection.md` direct table | LU and Cholesky notes used generic singular/non-SPD wording. | LU now says singular or near-singular pivots report `SPARSE_ERR_SINGULAR`; Cholesky now says non-SPD inputs report `SPARSE_ERR_NOT_SPD` through the factorization return code; LDLT notes document singular pivots and the existing symmetry-check status. |
| `docs/cookbook.md` compressed-input route | Direct calls were summarized as factorization and solve only. | The route now covers factorization, solve, refinement, and condition-estimate return codes and tells readers to inspect the function-specific return code first. |
| `examples/README.md` direct examples | Examples reported factorization/solve return codes and residuals. | Examples now say factorization or solve return codes and problem-local residuals only for the shown system. |
| `examples/README.md` `example_basic_solve` | Described residual computation generally. | Now describes solve return-code handling and problem-local residual computation. |

## Iterative Wording Changes

| Location | Before | After |
| --- | --- | --- |
| `README.md` iterative example | Only printed a convergence branch and a generic error branch. | The snippet now handles `SPARSE_ERR_NOT_CONVERGED` separately as iteration-budget exhaustion and prints the populated result fields for that path. |
| `README.md` API list | Listed iterative APIs without the result-field interpretation. | Added a paragraph naming `sparse_iter_result_t` fields and the documented `SPARSE_OK`/`SPARSE_ERR_NOT_CONVERGED` population rule. |
| `docs/solver_selection.md` diagnostics table | Iterative diagnostics were described as convergence status, residual norm/history, iteration count, stagnation, and breakdown. | The table now names `sparse_iter_result_t`, final relative residual, residual-history count, iteration count, stagnation, and breakdown fields. |
| `docs/solver_selection.md` iterative section | Used generic "solver result fields" wording. | Now names `iterations`, final relative `residual_norm`, `converged`, `stagnated`, `residual_history_count`, and `breakdown`, and states that non-convergence is iteration-budget exhaustion. |
| `docs/cookbook.md` iterative route | Did not explain result-field semantics before preconditioner selection. | Added `sparse_iter_result_t` guidance, approximate-solution wording, and `SPARSE_ERR_NOT_CONVERGED` boundary. |
| `examples/README.md` diagnostics handoff | Iterative examples reported convergence, residual, stagnation, and breakdown fields. | Now names `sparse_iter_result_t`, final relative residual, residual-history count, and local-solve scope. |
| `examples/README.md` `example_iterative` | Compared iteration counts and convergence behavior. | Now frames output as `sparse_iter_result_t` iteration counts, final relative residuals, and convergence fields. |
| `examples/README.md` `example_ic_minres` | Matched preconditioner assumptions to solver assumptions. | Now adds that preconditioners are tuning tools, not convergence guarantees. |
| `docs/maintainer_guide.md` | Treated iterative iteration counts as fixture-local diagnostics. | Now extends that wording to final relative residuals, convergence, stagnation, residual-history counts, breakdown, and `SPARSE_ERR_NOT_CONVERGED` interpretation. |

## Retained Semantics

- No direct solver return code changed.
- No iterative solver return code changed.
- No result-struct field was added, removed, or renamed.
- No convergence tolerance or residual scaling changed.
- No default solver, preconditioner, backend, or lifecycle recommendation was
  promoted beyond existing docs.
- Residual examples remain local confidence checks, not broad correctness,
  backend superiority, or state-of-the-art evidence.
- `SPARSE_ERR_NOT_CONVERGED` remains budget exhaustion for APIs that document
  approximate-solution/result-field population; it is not rewritten as
  singularity or invalid input.

## Cross-Checks

- Direct wording still maps to `sparse_err_t` in public direct headers.
- Iterative wording still maps to `sparse_iter_result_t` in
  `include/sparse_iterative.h`.
- Preconditioner wording remains assumption-based and does not promise
  convergence.
- QR/SVD/eigensolver cleanup is intentionally deferred to Day 9.

## Validation Plan

Day 8 changes are documentation-only, but they touch public docs. Validate with:

```sh
git diff --check
python3 tests/test_selected_performance_docs.py
```

No `.c` or `.h` files were modified, so the full
`make format && make lint && make test` gate is not required for this day.
