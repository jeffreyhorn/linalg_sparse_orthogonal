# Sprint 172 Day 8: Usage Documentation Design

## Purpose

Design the Day 9 usage-documentation update for the cleaned
`include/sparse_lu.h` contract. Day 8 is design-only and does not edit
README, tutorial, examples, or code.

## Selected Workflow

Day 8 selects one recommended workflow for Day 9 alignment:

**One-shot LU solve on a fresh matrix copy, with optional iterative
refinement.**

The cleaned header now states that LU factorization overwrites the
caller-owned working matrix, so docs should consistently show:

1. keep the original matrix if residuals, condition estimation, or refinement
   need original coefficients;
2. create `SparseMatrix *LU = sparse_copy(A)` before factorization;
3. call `sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, tol)`;
4. call `sparse_lu_solve(LU, b, x)`;
5. optionally call `sparse_lu_refine(A, LU, b, x, max_iters, tol)`;
6. free both `LU` and `A`.

For stable-pattern repeated direct solves, docs should continue to route to
`example_analysis.c` and the explicit `sparse_analysis.h` lifecycle instead of
suggesting repeated re-entry into one-shot LU factorization.

## Reviewed Usage Surfaces

| Surface | Current finding | Day 9 action |
| --- | --- | --- |
| `README.md` | Already shows `sparse_copy(A)`, `sparse_lu_factor(...)`, `sparse_lu_solve(...)`, six-argument `sparse_lu_refine(...)`, one-shot vs repeated-run routing, and non-claim boundaries. | Recheck only; no planned edit. |
| `docs/tutorial.md` | LU section shows the right copy/factor/solve/free shape, but its refinement snippet calls `sparse_lu_refine(A, LU, b, x, 3)` without the required tolerance argument. | Primary Day 9 edit: update the snippet to `sparse_lu_refine(A, LU, b, x, 3, 1e-15)` or another explicit tolerance consistent with nearby examples. |
| `docs/cookbook.md` | First-use ladder already tells callers to copy before mutating one-shot factorization and route stable-pattern reuse to the repeated-run lifecycle. | Recheck only; no planned edit. |
| `docs/api_reference.md` | Correctly says checked-in headers are the source of truth and keeps generated HTML local-only. | Recheck only; no planned edit. |
| `docs/maintainer_guide.md` | Maintains one-shot vs repeated-run boundaries and warns against widening evidence claims. | Recheck only; no planned edit. |
| `examples/example_basic_solve.c` | Demonstrates copy, factor, solve, residual, and cleanup. No refinement call. | Recheck only; no planned edit. |
| `examples/example_condition.c` | Demonstrates copy, factor, condition estimate, solve, and cleanup. | Recheck only; no planned edit. |
| `examples/example_matrix_market.c` | Demonstrates loaded Matrix Market input through the normal one-shot LU workflow. | Recheck only; no planned edit. |
| `examples/example_colamd.c` | Demonstrates fill-in comparison and uses LU copies for comparison factors. | Recheck only; no planned edit. |
| `examples/cmake_example/main.c` | Minimal downstream CMake consumer solve proof. | Recheck only; no planned edit. |

## Day 9 Edit Scope

Planned Day 9 source edits:

- `docs/tutorial.md` only.

Allowed Day 9 changes:

- fix the LU refinement snippet to match the public header signature;
- if needed, add one short sentence reinforcing that the tolerance argument is
  the convergence tolerance for the optional refinement step.

Disallowed Day 9 changes:

- edits to `.c` or `.h` files;
- broad README rewrites;
- generated API HTML output;
- package, install, shared-library, dynamic ABI, runtime-loader, platform,
  performance, external-library parity, LU CSR parity, or state-of-the-art
  claims.

## Claim-Scan Plan For Day 9

After the Day 9 documentation edit, run:

```sh
rg -n "sparse_lu_refine\\([^\\n]*\\)" README.md docs/tutorial.md include/sparse_lu.h examples tests
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|LU CSR parity" docs/tutorial.md
git diff --check
```

If Day 9 unexpectedly edits package/adoption/ABI/platform wording, also run:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
```

## Validation Required For Day 9

If Day 9 only edits documentation, `git diff --check` plus the claim/signature
scans above is sufficient. The full C quality gate is only required if Day 9
modifies `.c` or `.h` files.

## Completion Status

Day 8 is complete. Documentation edits are scoped before implementation, the
selected docs will reinforce the cleaned LU one-shot workflow, and unsupported
adoption/package/ABI/platform claims remain separate.

## Day 9 Handoff

Update `docs/tutorial.md` so the LU refinement snippet uses the six-argument
`sparse_lu_refine(A, LU, b, x, max_iters, tol)` signature. Keep the edit
minimal and run the Day 9 claim/signature scans.
