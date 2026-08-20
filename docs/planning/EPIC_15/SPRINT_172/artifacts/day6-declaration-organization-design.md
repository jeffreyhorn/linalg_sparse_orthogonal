# Sprint 172 Day 6: Declaration Organization Design

## Purpose

Design a reviewable declaration organization pass for the selected Sprint 172
header, `include/sparse_lu.h`, after the Day 5 contract-comment cleanup.

Day 6 is a design-only day. It does not move declarations or edit public
header code.

## Current Declaration Map

The post-Day 5 declaration-like surface is:

| Order | Declaration | Current role |
| --- | --- | --- |
| 1 | `sparse_lu_opts_t` | Option struct for one-shot LU with optional reordering and progress callback. |
| 2 | `sparse_lu_factor_opts()` | Configurable one-shot factorization entry point. |
| 3 | `sparse_lu_factor()` | Simple one-shot factorization entry point. |
| 4 | `sparse_lu_solve()` | Single right-hand-side solve from factored matrix. |
| 5 | `sparse_lu_solve_block()` | Multiple right-hand-side solve from factored matrix. |
| 6 | `sparse_lu_condest()` | Condition-estimate helper using original matrix and LU factors. |
| 7 | `sparse_lu_solve_transpose()` | Transpose solve helper required by condition estimation. |
| 8 | `sparse_apply_row_perm()` | Advanced exposed phase helper. |
| 9 | `sparse_apply_inv_col_perm()` | Advanced exposed phase helper. |
| 10 | `sparse_forward_sub()` | Advanced exposed phase helper. |
| 11 | `sparse_backward_sub()` | Advanced exposed phase helper. |
| 12 | `sparse_lu_refine()` | Iterative refinement using original matrix and LU factors. |

The current order is already workflow-oriented: options, factorization,
solves, condition/transpose support, advanced phase helpers, and refinement.

## Recommended Grouping Plan

Day 7 should preserve declaration order and improve generated API readability
with concise section headings only. This avoids churn in public API browsing
while still making the header easier to scan.

Recommended groups:

1. **Options**
   - `sparse_lu_opts_t`
2. **Factorization**
   - `sparse_lu_factor_opts()`
   - `sparse_lu_factor()`
3. **Solves**
   - `sparse_lu_solve()`
   - `sparse_lu_solve_block()`
4. **Conditioning And Transpose Solves**
   - `sparse_lu_condest()`
   - `sparse_lu_solve_transpose()`
5. **Advanced Solver Phases**
   - `sparse_apply_row_perm()`
   - `sparse_apply_inv_col_perm()`
   - `sparse_forward_sub()`
   - `sparse_backward_sub()`
6. **Refinement**
   - `sparse_lu_refine()`

## Section-Heading Plan

Day 7 may add or normalize short C block-comment headings before each group.
Headings should be plain ASCII, stable under generated documentation, and
short enough not to dominate the declaration comments.

Recommended heading style:

```c
/* Options */
/* Factorization */
/* Solves */
/* Conditioning and transpose solves */
/* Advanced solver phases */
/* Refinement */
```

The existing long decorative separators before condition estimation, advanced
solver phases, and iterative refinement can be replaced by the shorter heading
style. This is a readability-only documentation change.

## Ordering Rules

Day 7 should follow these ordering rules:

- Keep `sparse_lu_opts_t` before every factorization function that consumes it.
- Keep `sparse_lu_factor_opts()` before `sparse_lu_factor()` because it is the
  more configurable entry point and documents the richer option contract.
- Keep solve functions immediately after factorization functions.
- Keep `sparse_lu_condest()` before `sparse_lu_solve_transpose()` because the
  condition-estimate comment introduces the transpose-solve dependency.
- Keep advanced phase helpers after public solve/condition helpers because
  they are lower-level building blocks.
- Keep `sparse_lu_refine()` last because it depends conceptually on both the
  original matrix and an existing LU solve path.

## Non-Move Exception List

These declarations should not move on Day 7:

- `sparse_lu_opts_t`: must remain before `sparse_lu_factor_opts()` for reader
  and generated-doc dependency order.
- `sparse_lu_factor_opts()` and `sparse_lu_factor()`: should remain adjacent
  to keep one-shot factorization entry points together.
- `sparse_lu_solve()` and `sparse_lu_solve_block()`: should remain adjacent
  to keep single- and multi-RHS solve contracts together.
- `sparse_lu_condest()` and `sparse_lu_solve_transpose()`: should remain
  adjacent because the condition estimator describes transpose-solve usage.
- The four advanced solver phase helpers: should remain together and after
  higher-level solve APIs.
- `sparse_lu_refine()`: should remain after solve-related declarations because
  it is a post-solve improvement path.

## Diff-Review Expectations

Day 7 should make any public header diff easy to audit:

- Acceptable Day 7 changes:
  - add or normalize section headings;
  - move only whole documentation-comment plus declaration blocks if a later
    implementation artifact explicitly records the move;
  - update local wording required by a heading change.
- Disallowed Day 7 changes:
  - function signature changes;
  - typedef, struct layout, enum, macro, include, include-guard, or installed
    header name changes;
  - implementation or test behavior changes;
  - new ABI, package-manager, shared-library, runtime-loader, platform-parity,
    performance, external-library parity, LU CSR parity, or state-of-the-art
    support claims.

If Day 7 edits `include/sparse_lu.h`, it must capture before/after declaration
surfaces and run the full C quality gate: `make format && make lint &&
make test`.

## Generated API And Readability Constraints

Generated API readers should see high-level public entry points before helper
phases. The section headings should improve scanning without implying new
feature support or a new ABI/package boundary.

The final generated-doc order should remain:

1. option configuration;
2. factorization;
3. solves;
4. condition and transpose operations;
5. advanced phases;
6. refinement.

## Completion Status

Day 6 is complete. The declaration organization plan is reviewable before
implementation, public API behavior remains unchanged, and generated
API/readability constraints are explicit.

## Day 7 Handoff

Day 7 should implement only the short section-heading normalization unless a
new issue is found during pre-edit review. The default implementation should
preserve declaration order.
