# Sprint 200 Day 2: Owner Selection

## Purpose

Score Sprint 200 allocation-failure owner candidates, exclude existing proof
lanes, select exactly one owner, and freeze the proof boundary before any code
edits.

## Selected Owner

Sprint 200 selects `sparse_symbolic_lu()` in `src/sparse_etree.c`.

Selected boundary:

- function: `sparse_symbolic_lu(const SparseMatrix *A, const idx_t *perm,
  sparse_symbolic_t *sym_L, sparse_symbolic_t *sym_U)`;
- caller-owned inputs: `A` and optional `perm`;
- caller-visible outputs: optional `sym_L` and optional `sym_U`;
- selected success modes: L+U, L-only, U-only, and valid-permutation calls;
- primary proof binary: `tests/test_etree.c`;
- focused gate approach: extend or specialize the symbolic allocation-failure
  gate with symbolic-LU registration checks.

## Ranking

| Rank | Candidate | Total | Disposition |
| ---: | --- | ---: | --- |
| 1 | `sparse_symbolic_lu()` symbolic owner | 28 | Selected. Explicit prior residual with clear symbolic output publication and direct reuse of Sprint 195 proof patterns. |
| 2 | Helper-level etree/postorder/colcount owner | 22 | Fallback only. Bounded, but less public-output value. |
| 3 | `sparse_analyze()` lifecycle owner | 22 | Deferred. Too composed for one selected proof. |
| 4 | Linked-list LU solve workspace owner | 21 | Deferred. Output mutation complicates stale-output semantics. |
| 5 | LU CSR selected entry point | 21 | Deferred. Valuable but broader implementation surface. |
| 6 | Direct-solver output publication owner | 21 | Deferred. Needs one function selected in a future proof. |
| 7 | Core sparse matrix constructor path | 22 | Rejected for Sprint 200. Too broad for selected-owner wording. |
| 8 | Partial SVD output owner | 20 | Deferred. Multi-output numerical proof cost is high. |

## Existing Owners Excluded

| Existing proof lane | Reason excluded |
| --- | --- |
| Iterative repeated-run handles | Already covered by `make iterative-allocation-failure-gate`. |
| `sparse_matmul()` workspace | Already covered by `make matmul-allocation-failure-gate`. |
| Selected `sparse_symbolic_cholesky()` output allocation | Already covered by `make symbolic-allocation-failure-gate`. |

## Why Symbolic LU

`sparse_symbolic_lu()` is the best Day 2 selection because it closes an
explicit Sprint 195 residual without selecting a broad direct-solver or full
analysis lifecycle. The implementation delays `sym_L` publication until the
pipeline succeeds, initializes `sym_U` before building upper-triangle
structure, and reuses symbolic Cholesky intermediates. Those behaviors create
specific cleanup and stale-output questions that can be proven with the
existing deterministic allocation hook and the established `test_etree`
fixture family.

## Proof Checklist

| Proof area | Required evidence |
| --- | --- |
| Failed allocation | Deterministic fail-after cases return `SPARSE_ERR_ALLOC` or the documented propagated allocation failure. |
| Cleanup | Temporary matrix, permutation workspace, etree workspaces, symbolic Cholesky intermediate, and partial `sym_U` state are cleaned up. |
| Stale-output suppression | Failed `sym_L` and `sym_U` outputs do not look like successful symbolic structures and remain free-safe. |
| Retry | The same fixture succeeds after the hook is reset. |
| Caller-owned input | `A` and valid `perm` remain intact across failed calls. |
| Unsupported breadth | No broad allocation-failure, `sparse_analyze()`, direct-solver, matrix-constructor, OS OOM, concurrent-hook, or platform reliability claim is added. |

## Day 3 Handoff

Day 3 should trace `sparse_symbolic_lu()` allocation and publication points in
detail, especially:

1. `B = sparse_create(n, n)` and insertion cleanup.
2. Optional `seen` and `inv_perm` allocation and permutation validation.
3. `row_cols` allocation and `A^T * A` construction.
4. `parent`, `postorder`, and `cc` workspace allocation.
5. `sparse_symbolic_cholesky()` intermediate ownership.
6. `sym_U` `u_cnt`, `col_ptr`, and `row_idx` allocation.
7. Final `sym_L` publication and L-only/U-only behavior.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 2 validation command.
