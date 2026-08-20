# Sprint 172 Day 4: LU Contract Cleanup Design

## Purpose

Day 4 designs the declaration-preserving cleanup for the selected Sprint 172
header family, `include/sparse_lu.h`. This artifact maps the current public
LU declarations, defines normalized contract language, and records what Day 5
may and may not edit.

Day 4 does not edit public headers.

## Evidence Read

| Evidence | Day 4 use |
| --- | --- |
| `include/sparse_lu.h` | Selected public header and current comment/declaration surface. |
| `src/sparse_lu.c` | Source evidence for error returns, cancellation, factored-state checks, allocation paths, and in-place mutation. |
| `tests/test_sparse_lu.c` | Primary linked-list LU solve, singularity, null, transpose, condition-estimate, and refinement proof owner. |
| `tests/test_edge_cases.c` | Bad-state, unfactored solve, repeated solve, factored mutation, and singular edge-case proof owner. |
| `tests/test_lu_csr.c` | Cross-check for LU block solve behavior only; not a Sprint 172 LU CSR cleanup target. |
| `examples/example_basic_solve.c` | First-use LU factor/solve example. |
| `examples/example_condition.c` | Condition-estimate example. |
| `examples/example_colamd.c` | Reordered LU usage example. |
| `examples/example_matrix_market.c` | Matrix Market LU solve usage example. |

## Current Declaration Map

| Header line | Declaration surface | Current role | Day 5 treatment |
| ---: | --- | --- | --- |
| 1-2 | `SPARSE_LU_H` include guard | Installed public header guard. | Preserve exactly. |
| 58 | `sparse_lu_opts_t` | One-shot LU options: pivot, reorder, tolerance, progress callback, callback context. | Preserve layout and field order. Comments may be shortened and normalized. |
| 117 | `sparse_lu_factor_opts(...)` | One-shot LU factorization with optional reorder and callback behavior. | Preserve signature. Clarify lifecycle/error language only. |
| 169 | `sparse_lu_factor(...)` | In-place one-shot LU factorization. | Preserve signature. Clarify copy-before-factor and tolerance wording only. |
| 187 | `sparse_lu_solve(...)` | Solve with existing LU factors. | Preserve signature. Clarify read-only solve and output/aliasing wording only. |
| 212 | `sparse_lu_solve_block(...)` | Multi-RHS solve with caller-owned column-major buffers. | Preserve signature. Fix duplicate error wording if edited. |
| 255-256 | `sparse_lu_condest(...)` | 1-norm condition estimate using original and LU-factored matrices. | Preserve multiline signature. Shorten algorithm prose if needed. |
| 281 | `sparse_lu_solve_transpose(...)` | Solve transposed system with existing factors. | Preserve signature. Clarify workspace/allocation and factored-state errors only. |
| 297 | `sparse_apply_row_perm(...)` | Advanced/testing helper applying row permutation. | Preserve signature. May add concise factored/permutation context. |
| 310 | `sparse_apply_inv_col_perm(...)` | Advanced/testing helper applying inverse column permutation. | Preserve signature. May add concise output overwrite wording. |
| 323 | `sparse_forward_sub(...)` | Advanced/testing helper for unit lower triangular solve. | Preserve signature. May add null/error wording only if source evidence is clear. |
| 336 | `sparse_backward_sub(...)` | Advanced/testing helper for upper triangular solve. | Preserve signature. May add singularity wording from source evidence. |
| 357-358 | `sparse_lu_refine(...)` | Iterative refinement using original matrix and LU factors. | Preserve multiline signature. Normalize null/badarg/allocation wording. |

## Intended Documentation Sections

Day 5 should keep declaration order unchanged. Day 6 can revisit section
headings, but Day 4 does not authorize declaration movement.

| Intended section | Current declarations |
| --- | --- |
| Overview and one-shot lifecycle | file-level comment, `sparse_lu_opts_t`, `sparse_lu_factor_opts`, `sparse_lu_factor` |
| Solve entry points | `sparse_lu_solve`, `sparse_lu_solve_block` |
| Condition and transpose support | `sparse_lu_condest`, `sparse_lu_solve_transpose` |
| Advanced/testing phase helpers | `sparse_apply_row_perm`, `sparse_apply_inv_col_perm`, `sparse_forward_sub`, `sparse_backward_sub` |
| Iterative refinement | `sparse_lu_refine` |

## Normalized Language Plan

| Contract area | Normalized wording direction | Evidence/control |
| --- | --- | --- |
| Ownership and lifecycle | State that LU factorization mutates the caller-owned matrix in place; callers that need original coefficients must factor a copy; repeated stable-pattern workflows belong to `sparse_analysis.h`. | Already stated in header overview and factor comments; source mutates `SparseMatrix`; README also warns about in-place factorization. |
| Factored-state preconditions | State that solve, block solve, transpose solve, condition estimate, and refinement require an LU-factored matrix. | Source uses `sparse_matrix_require_factored_state(...)`; edge-case tests assert `SPARSE_ERR_BADARG` on unfactored matrices. |
| Reordering behavior | Keep the current one-shot reorder contract: reordered paths publish back only on success where documented, and solve unpermutes through matrix-owned reorder metadata. | Source has reordered working-copy path and publish-on-success helper. Do not turn this into a repeated-run guarantee. |
| Error handling | Normalize existing returns without inventing new ones. Include `SPARSE_ERR_NULL`, `SPARSE_ERR_SHAPE`, `SPARSE_ERR_BADARG`, `SPARSE_ERR_SINGULAR`, `SPARSE_ERR_ALLOC`, and `SPARSE_ERR_CANCELLED` only where source/header already supports them. | Source and tests cover these returns unevenly by function; Day 5 should not add unproven returns. |
| Tolerance | Keep factor tolerance as absolute pivot threshold; keep solve singularity wording norm-relative where currently documented; keep refinement tolerance as relative residual threshold. | Header and source already distinguish factor threshold, solve `DROP_TOL * factor_norm`, and refinement residual tolerance. |
| Workspace/allocation | Clarify that public solve/condest/refine paths may allocate temporary workspace where already documented. Do not promise allocation-free solves. | Source allocates temporary vectors/workspaces for solve, block solve, transpose, condest, and refine. |
| Callback behavior | Shorten `progress_cb` prose while preserving phase name, step/total timing, cancellation return, and matrix-state caveats. | Header and source define `phase = "lu_factor"`, top-of-column callback, and `SPARSE_ERR_CANCELLED`. |
| Threading | Normalize existing same-matrix mutation/read-only solve notes. Do not claim global thread safety or mutex support. | Header currently states factor mutates same matrix and solve is read-only on factored matrix with separate buffers. |
| Advanced helpers | Keep helpers clearly labeled advanced/testing support. Do not promote them as the preferred user workflow. | Header section already labels them individual phases for testing and advanced use. |

## Comment Treatment Plan

| Comment block | Day 5 action |
| --- | --- |
| File overview and usage pattern | Retain, but consider shortening the example and keeping the `sparse_analysis.h` handoff concise. |
| `sparse_lu_opts_t` | Rewrite field comments for pivot/reorder/tolerance/callback readability. Preserve field order and designated-init compatibility note. |
| `sparse_lu_factor_opts(...)` | Rewrite to emphasize one-shot lifecycle, fresh/copy matrix requirement, reorder publish-on-success behavior, and invalid enum rejection. |
| `sparse_lu_factor(...)` | Retain core in-place factor explanation; normalize tolerance and error wording; avoid long internal references where possible. |
| `sparse_lu_solve(...)` | Retain concise solve chain; clarify factored-state requirement, `x` overwrite, and aliasing. |
| `sparse_lu_solve_block(...)` | Normalize `nrhs == 0`, non-NULL buffer requirement, and duplicate `SPARSE_ERR_BADARG` wording. |
| Condition-estimate block | Shorten algorithm/design prose if needed; keep original matrix requirement, factored matrix requirement, and allocation error. |
| Transpose solve block | Retain permutation/triangular solve explanation; normalize factored-state, allocation, and singularity wording. |
| Advanced phase helpers | Keep section as advanced/testing; add only concise error/output wording if source evidence is clear. |
| Refinement block | Clarify original matrix plus LU factors, in-place update of `x`, null/badarg/allocation propagation, and tolerance. |

## Day 5 Allowed Edits

Day 5 may edit only:

- comments in `include/sparse_lu.h`;
- the Day 5 artifact under `docs/planning/EPIC_15/SPRINT_172/artifacts/`;
- `WORKING_NOTES.md`.

Day 5 may update user-facing docs only if the header cleanup creates a clear
terminology mismatch. Such edits should be small and should not expand into
tutorial rewrites.

## Day 5 Disallowed Edits

Day 5 must not:

- change any LU function declaration or signature;
- change `sparse_lu_opts_t` field names, order, types, or layout;
- change include guards or includes;
- change public macros, typedefs, enum values, installed header names, or
  exported names;
- edit `src/*.c` or tests to make comment cleanup true;
- edit `include/sparse_lu_csr.h` or other direct-solver headers;
- add generated API HTML;
- add package-manager, shared-library, dynamic ABI, runtime-loader, broad
  platform, portable performance, external-library parity, LU CSR parity, or
  state-of-the-art claims.

## Declaration-Preservation Plan

Before Day 5 edits, capture the selected declaration-like surface:

```sh
rg -n "^sparse_err_t |^typedef struct|^#ifndef|^#define" include/sparse_lu.h \
  > docs/planning/EPIC_15/SPRINT_172/artifacts/day5-lu-declarations-before.txt
```

After Day 5 edits, capture the same surface:

```sh
rg -n "^sparse_err_t |^typedef struct|^#ifndef|^#define" include/sparse_lu.h \
  > docs/planning/EPIC_15/SPRINT_172/artifacts/day5-lu-declarations-after.txt
diff -u \
  docs/planning/EPIC_15/SPRINT_172/artifacts/day5-lu-declarations-before.txt \
  docs/planning/EPIC_15/SPRINT_172/artifacts/day5-lu-declarations-after.txt
```

The before/after diff should be empty. A non-empty diff is a stop condition
unless it is explained as a comment-only false positive, which the current
command is designed to avoid.

Day 5 should also run:

```sh
git diff -- include/sparse_lu.h
git diff --word-diff=porcelain -- include/sparse_lu.h
git diff --check
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|LU CSR parity" include/sparse_lu.h
make format && make lint && make test
```

Because Day 5 is expected to edit a public `.h` file, the full C quality gate
is required even if edits are comment-only.

## Day 6 Declaration-Organization Handoff

Day 4 does not recommend declaration reordering for Day 5. The header already
has a coherent workflow order:

1. options and factor entry points;
2. solve entry points;
3. condition/transpose support;
4. advanced phase helpers;
5. refinement.

Day 6 may consider replacing the decorative section separators with plainer
Doxygen-friendly headings, but should preserve declaration order unless a
separate declaration baseline proves the move is harmless and worth the review
cost.

## Behavior-Preservation Constraints

The cleanup must preserve:

- in-place factorization semantics;
- solve and transpose solve factored-state requirements;
- multi-RHS `nrhs == 0` behavior and non-NULL pointer requirements;
- condition-estimate requirement for original and factored matrices;
- refinement using original matrix residuals and LU-factor solves;
- progress callback phase, step, total, elapsed, user pointer, and
  cancellation behavior;
- existing error returns;
- static-first package and ABI non-claim boundaries.

## Validation Notes

Day 4 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 4.

## Completion Check

- The selected LU header declaration map is recorded.
- Day 5 allowed and disallowed edits are explicit.
- Normalized contract-language targets are tied to current header/source/test
  evidence.
- Declaration-preservation commands are defined before implementation.
- No API behavior or binary/package support claim is changed by design.
