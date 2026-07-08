# Sprint 113 Day 14: Closeout and Handoff

## Purpose

Close Sprint 113 with proof-owner truth, validation status, and a
dependency-ordered residual queue for final Epic 10 integration.

## Completed Item Checklist

| Project item | Status | Evidence |
|---|---|---|
| Item 1: Residual Intake and Boundary Refresh | Complete | Day 1 intake and Day 2 owner selection artifacts exclude completed prior-sprint work and order remaining residuals. |
| Item 2: Eigensolver Behavior Owner Proof Batch | Complete | Day 3 proof design and Day 4 grow-m behavior tests cover sizing, retry progress, invalid capacity, and cancellation. |
| Item 3: Eigensolver Source Movement or No-Move Contract | Complete as no-move | Day 5 movement decision and Day 6 no-move contract keep eigensolver internals in place pending broader proof. |
| Item 4: Direct and Iterative Proof-Owner Cleanup | Complete | Day 7 boundary and Day 8 LDLT CSC external dense-reference oracle cleanup centralize local ownership without hiding proof values. |
| Item 5: SVD Proof-Owner Cleanup | Complete | Day 9 boundary and Day 10 partial-SVD `A*v ~= sigma*u` residual helper cleanup centralize the mechanical residual loop. |
| Item 6: Proof-Owner Metrics and Non-Claims | Complete | Day 11 captures before/current metrics, membership drift status, residual queues, and broad-abstraction non-claims. |
| Item 7: Validation and Closeout | Complete | Day 12 validation matrix, Day 13 execution evidence, and this Day 14 handoff close the sprint. |

## Eigensolver Outcome

Sprint 113 selected grow-m sizing and retry behavior as the eigensolver owner.
The completed proof batch added tests in `tests/test_eigs.c` covering:

- public handle grow-m preparation, reuse, and growth;
- default grow-m capacity peak-basis behavior;
- explicit grow-m capacity peak-basis behavior;
- too-small explicit iteration budgets;
- retry progress step accumulation;
- cancellation at retry boundaries.

The movement decision was no-move. The no-move contract keeps the current
eigensolver source layout intact because the sprint proved grow-m behavior but
did not prove broader movement safety for shared Lanczos kernels, Ritz handling,
partial-result publication, or shift-invert conversion.

## Direct/Iterative Outcome

Sprint 113 selected LDLT CSC external dense-reference oracle cleanup from the
direct/iterative residual queue.

The cleanup in `tests/test_ldlt_csc.c` added a local oracle state owner and
centralized cleanup while preserving:

- fixture key, builder, and tolerance at call sites;
- exact RHS construction;
- two-pass indefinite factorization;
- RHS permutation;
- `ldlt_csc_solve`;
- solution unpermutation;
- dense-reference read status;
- max-difference and residual assertions.

## SVD Outcome

Sprint 113 selected partial-SVD vector/residual cleanup from the SVD residual
queue.

The cleanup in `tests/test_svd_partial_helpers.h` added
`partial_svd_max_av_residual` and updated:

- `test_partial_svd_vectors_Av`;
- `test_partial_svd_vectors_wide`.

The helper centralizes only the mechanical `A*v ~= sigma*u` residual loop while
leaving fixture shape, inserted values, rank `k`, options, singular-value
tolerances, residual diagnostic labels, and `1e-6` thresholds visible at the
test sites.

## Final Validation Summary

Focused validation passed:

- `make build/test_eigs && build/test_eigs`: 36 tests, 0 failed, 0 skipped,
  345 assertions.
- `make build/test_ldlt_csc && build/test_ldlt_csc`: 100 tests, 0 failed, 0
  skipped, 3556 assertions.
- `make build/test_svd && build/test_svd`: 98 tests, 0 failed, 0 skipped, 1562
  assertions.

Full required quality gate passed:

```sh
make format && make lint && make test
```

Documentation and hygiene checks passed:

- `git diff --check`;
- trailing-whitespace scan;
- local Markdown link check.

Build/source/API drift checks passed:

- only `tests/test_eigs.c`, `tests/test_ldlt_csc.c`, and
  `tests/test_svd_partial_helpers.h` changed under build/test scope;
- no `Makefile`, `CMakeLists.txt`, `cmake/`, `include/`, or `src/` drift;
- no public API, install-header, helper-target, source-list, or reviewed CTest
  membership drift.

## Residual Deferred Debt

### 1. Eigensolver Source-Movement Proof

Dependency order:

1. Add direct proof for `lanczos_iterate_op` behavior across basic, thick
   restart, and LOBPCG-adjacent dispatch paths.
2. Add repeated/clustered spectrum proof before moving Ritz selection.
3. Add Ritz vector lifting proof before extracting shared vector publication
   helpers.
4. Add partial-result publication proof after `m_cap` exhaustion.
5. Add shift-invert grow-m conversion proof.
6. Revisit source movement only after the above proof isolates one safe owner.

Non-claim: Sprint 113 does not prove a safe eigensolver source split.

### 2. Direct/Iterative Exact-RHS and Oracle Owners

Dependency order:

1. QR sequential RHS setup.
2. CG preconditioner-specific exact-RHS setup.
3. GMRES exact-RHS setup.
4. BiCGSTAB exact-RHS setup.
5. MINRES exact-RHS setup.
6. Broad direct/iterative oracle abstraction only after at least two more
   solver-specific cleanup lanes prove common ownership.

Non-claim: Sprint 113 does not prove a broad cross-solver proof abstraction.

### 3. SVD Proof Owners

Dependency order:

1. Reconstruction helper movement, split by storage contract before any shared
   helper is introduced.
2. U/Vt orthogonality helper movement, split by economy/full leading-dimension
   convention.
3. Moore-Penrose product helper extraction.
4. Dense low-rank proof-loop cleanup.
5. Sparse low-rank proof-loop cleanup.
6. Condition-number proof cleanup.

Non-claim: Sprint 113 does not prove broad SVD reconstruction, orthogonality,
Moore-Penrose, low-rank, or condition-number helper extraction.

## Final Handoff

Sprint 113 leaves Epic 10 integration with:

- bounded behavior proof added for eigensolver grow-m behavior;
- eigensolver source movement explicitly deferred behind proof requirements;
- one direct/iterative oracle cleanup completed;
- one SVD residual cleanup completed;
- metrics and non-claims recorded;
- validation evidence complete and passing;
- public API, install headers, helper targets, Make/CMake source lists, and
  reviewed CTest membership unchanged.
