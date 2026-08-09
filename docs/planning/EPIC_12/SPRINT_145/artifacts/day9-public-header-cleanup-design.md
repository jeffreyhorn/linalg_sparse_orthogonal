# Sprint 145 Day 9 Public Header Cleanup Design

## Purpose

Select a scoped public-header cleanup batch before editing any API comments.
The cleanup must make first-use contracts easier to read without changing API
behavior, ABI posture, ownership rules, return-code semantics, mutation
contracts, or bounded numerical claims.

## Reviewed Header Surface

| Header | Lines | Day 9 finding | Day 10 disposition |
| --- | ---: | --- | --- |
| `include/sparse_matrix.h` | 617 | High first-use value; top comment and `SPARSE_CSC_THRESHOLD` contain dense maintainer/history/benchmark detail before basic matrix contracts. | Clean selected comments. |
| `include/sparse_iterative.h` | 773 | Highest density; file-level examples, breakdown summary, option comments, repeated-run handle text, and block/matrix-free entries mix first-use and deep diagnostics. | Clean selected comments. |
| `include/sparse_qr.h` | 391 | High adoption value for least-squares/rank-sensitive users; comments are mostly contract-critical but can route corpus/evidence interpretation to docs. | Clean lightly. |
| `include/sparse_svd.h` | 260 | High adoption value for rank/condition/low-rank users; partial-SVD and sparse low-rank comments contain useful but dense evidence/runtime detail. | Clean lightly. |
| `include/sparse_analysis.h` | 499 | Important but less first-use after README/solver-selection now route repeated direct lifecycle. | Defer unless Day 10 has time. |
| `include/sparse_eigs.h` | 651 | Dense and important, but not first-use for Sprint 145 front door; prior sprints already established backend governance. | Defer to avoid over-broad header churn. |
| Other public headers | 70-360 each | Lower first-use friction or more localized contracts. | Defer. |

## Selected Cleanup Targets

Day 10 should edit only these public headers unless validation exposes a
directly related inconsistency:

1. `include/sparse_matrix.h`
2. `include/sparse_iterative.h`
3. `include/sparse_qr.h`
4. `include/sparse_svd.h`

This is the smallest set that covers the adoption path established by Days 6-8:
matrix construction, direct/iterative first solve, QR least-squares/rank
diagnostics, and SVD rank/condition/partial-SVD behavior.

## Comment Routing Rules

Keep in public headers:

- function purpose and shape assumptions;
- ownership and lifetime rules;
- caller allocation requirements and output-buffer shape;
- NULL handling and explicit error-return contracts;
- mutation or non-mutation guarantees;
- option defaults that change behavior;
- result-field semantics;
- public typed backend/option behavior;
- safety constraints such as identity-permutation preconditions;
- explicit non-claims where removing them would make an API comment
  misleading.

Move or shorten in public headers:

- benchmark fixture history and measured crossover narrative;
- sprint-history phrasing;
- report/CI ownership details;
- maintainer-only rationale better owned by `docs/maintainer_guide.md`;
- long tutorial-style examples already covered by README, examples, cookbook,
  or solver-selection docs;
- dense evidence summaries when a short bounded non-claim plus a doc link is
  enough.

Do not move:

- any wording needed to preserve ABI posture or compile-time width contract;
- ownership and cleanup rules for returned matrices, dense arrays, and result
  structs;
- `SPARSE_ERR_*` behavior;
- `NULL` versus explicit diagnostic behavior;
- one-shot mutation versus repeated-run lifecycle distinctions;
- QR rank/nullspace and partial-SVD bounded evidence non-claims unless the
  replacement still names the boundary clearly.

## Header-Specific Plan

| Header | Planned cleanup | Must preserve |
| --- | --- | --- |
| `sparse_matrix.h` | Shorten the file-level compatibility-shell narrative; compress `SPARSE_CSC_THRESHOLD` history into current dispatch/override behavior with a doc pointer; keep matrix lifecycle comments focused. | `SPARSE_IDX_BITS` contract, `sparse_scalar_t` real-only boundary, copy/mutation contracts, silent-zero accessor behavior, Matrix Market errno behavior, and permutation reset contract. |
| `sparse_iterative.h` | Replace large file-level usage examples with a compact first-use summary and examples/doc links; shorten breakdown summary while keeping success/failure semantics; keep result/option field contracts clear. | CG/GMRES/MINRES/BiCGSTAB assumptions, result fields, residual history ownership, progress/cancel behavior, repeated-run handle lifecycle, and one-shot versus repeated-run boundaries. |
| `sparse_qr.h` | Keep QR API contracts but route corpus/evidence interpretation to `docs/solver_selection.md#qr-evidence-boundary`; avoid repeating front-door teaching text. | Identity-permutation precondition, non-mutation, `sparse_qr_t` free/reuse rules, rank threshold semantics, residual output, minimum-norm distinction, and bounded QR non-claim. |
| `sparse_svd.h` | Keep SVD ownership/output-shape/error contracts; shorten partial-SVD evidence paragraph and sparse low-rank environment-variable detail with a doc pointer. | Economy/full output shape, `sparse_svd_free`, partial-SVD `compute_uv && economy` requirement, convergence/fail-closed behavior, dense allocation ownership, sparse low-rank memory behavior, and bounded partial-SVD non-claim. |

## API Contract Preservation Checklist

Before and after Day 10 edits, confirm:

- no declarations, typedefs, enum values, struct fields, macros, or include
  guards changed;
- no public function signature changed;
- no documented default values changed;
- no error code was removed from a function contract;
- no ownership/freeing requirement was removed;
- no input-mutation or non-mutation guarantee was removed;
- no identity-permutation or shape precondition was removed;
- no QR or partial-SVD bounded non-claim was weakened;
- no static/shared, ABI, package, platform, or portable performance claim was
  introduced.

## Required Quality Gates

Because Day 10 will modify `.h` files, the required gate is:

```sh
make format && make lint && make test
```

Additional focused docs/API scans for Day 10:

```sh
git diff --check
git diff --name-only -- '*.c' '*.h'
rg -n "state-of-the-art|external-library parity|broad QR|broad repeated-spectrum|portable performance|platform parity|package guarantee|ABI promise" include/sparse_matrix.h include/sparse_iterative.h include/sparse_qr.h include/sparse_svd.h
rg -n "SPARSE_ERR|NULL|caller|free|not modified|identity permutations|default|progress_cb|backend|repeated-run" include/sparse_matrix.h include/sparse_iterative.h include/sparse_qr.h include/sparse_svd.h
```

## Day 9 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public-header changes are scoped and justified. | Complete | Four high-impact headers selected; lower-priority headers deferred. |
| No API behavior or ABI promise changes are planned accidentally. | Complete | Preservation checklist forbids declaration, field, default, ownership, error, mutation, ABI, and non-claim changes. |
| Required quality gates are known before header edits begin. | Complete | Day 10 must run `make format && make lint && make test` because headers will change. |

## Day 10 Handoff

Implement the selected cleanup batch in the four public headers only. Keep edits
comment-only unless a typo in a comment references a nonexistent symbol. Run
the full required C gate before recording Day 10 as complete.
