# Day 7 Direct/Iterative Proof-Owner Boundary Selection

## Purpose

Day 7 selects exactly one bounded direct or iterative proof-owner cleanup target
for Day 8.  The selected cleanup must reduce meaningful setup noise without
hiding solver calls, options, expected values, residual thresholds, convergence
status, dense-reference comparisons, or printed proof evidence.

## Candidate Comparison

| Candidate | Primary file | Duplication / setup noise | Proof values at risk | Validation cost | Decision |
|---|---|---|---|---|---|
| QR sequential RHS setup | `tests/test_qr.c` | Some repeated RHS setup remains, but Sprint 109 already completed the main QR exact-RHS helper path. | Least-squares residuals, QR refinement before/after residuals, and literal RHS values often explain the test. | `test_qr` plus full quality chain if edited. | Defer to avoid duplicating Sprint 109 and hiding QR proof literals. |
| LDLT CSC external dense-reference oracle cleanup | `tests/test_ldlt_csc.c` | One dedicated oracle lane allocates six arrays, handles external-reference skip/fail paths, applies permutation solve/unpermute, compares to dense oracle, and repeats through three call sites. | Dense oracle vector comparison, Windows skip behavior, permutation handling, LDLT CSC solve residual, fixture names, and tolerances. | `test_ldlt_csc` plus full quality chain if edited. | **Select for Day 8.** |
| CG preconditioner-specific exact-RHS setup | `tests/test_iterative.c` | Some setup remains, but Sprint 110 already completed generic CG exact-RHS cleanup. | Preconditioner construction, iteration comparisons, residual norms, and convergence status. | `test_iterative` plus full quality chain if edited. | Defer; do not repeat Sprint 110 generic CG cleanup. |
| GMRES exact-RHS setup | `tests/test_iterative.c` | Several exact-RHS patterns exist, but restart settings and lucky-breakdown behavior are family-specific. | Restart value, convergence status, residual norms, initial guess behavior, and breakdown cases. | `test_iterative` plus full quality chain if edited. | Defer; cleanup should be a later GMRES-only pass. |
| BiCGSTAB exact-RHS setup | `tests/test_iterative.c` and `tests/test_bicgstab.c` | Some exact-RHS setup exists, including Sprint 103 comparison fixtures. | Breakdown behavior, ILU comparison, LU/GMRES references, nonconvergence status, and residual evidence. | `test_bicgstab`, possibly `test_iterative`, plus full quality chain if edited. | Defer; more than one file can become involved. |
| MINRES exact-RHS setup | `tests/test_iterative.c` and `tests/test_minres.c` | Some repeated RHS setup remains, but symmetry and preconditioner assumptions are central to proof. | Symmetry assumptions, KKT fixtures, IC/Jacobi preconditioner behavior, convergence status, and residual norms. | `test_minres`, possibly `test_iterative`, plus full quality chain if edited. | Defer; high proof-context risk. |

## Selected Target

Selected Day 8 target:

```text
tests/test_ldlt_csc.c external dense-reference oracle cleanup
```

The cleanup should stay inside the existing LDLT CSC test file and should
focus only on the external dense-reference lane around:

- `read_ldlt_external_dense_reference_solution`;
- `assert_ldlt_external_dense_reference`;
- `test_s98_external_dense_reference_kkt_5x5`;
- `test_s98_external_dense_reference_kkt_10x10`;
- `test_s102_external_dense_reference_scaled_kkt_10x10`.

## Selection Rationale

This target is the best Day 8 cleanup because:

- it is not the completed Sprint 109 QR exact-RHS cleanup;
- it is not the completed Sprint 110 generic CG exact-RHS cleanup;
- it is a single direct-solver oracle lane, not a cross-solver abstraction;
- repeated allocation and cleanup noise is concentrated in one helper;
- the proof values can remain visible at the oracle boundary:
  - fixture key;
  - fixture builder;
  - tolerance;
  - dense-reference vector comparison;
  - expected `x_true`;
  - permutation and unpermutation flow;
  - relative residual;
  - Windows skip behavior;
  - external helper skip/fail behavior.

## Day 8 Cleanup Boundary

Allowed cleanup:

- introduce a small local fixture/state struct inside `tests/test_ldlt_csc.c`;
- centralize allocation and cleanup for the external dense-reference helper;
- reduce repeated free blocks on skip/fail/success exits;
- keep the existing external reference command and fixture keys;
- keep the existing Windows skip behavior;
- keep each external dense-reference test name and call site.

Disallowed cleanup:

- no shared helper header;
- no new compiled helper target;
- no public API or production header changes;
- no changes to Python reference-helper semantics;
- no cross-direct-solver oracle abstraction;
- no hiding solver call, permutation application, dense reference comparison,
  residual threshold, fixture key, or tolerance.

## Proof-Visibility Rules

Day 8 must keep these values visible at the call site or the immediate oracle
helper boundary:

- fixture key string;
- fixture builder function;
- tolerance;
- `x_true[i] = i + 1` construction;
- `sparse_matvec(A, x_true, b)` exact-RHS construction;
- `s20_two_pass_indefinite_factor` call;
- permutation of `b` through `F1->perm`;
- `ldlt_csc_solve` on the permuted factor;
- unpermutation back to public ordering;
- dense-reference read status and skip/fail handling;
- max `|x - x_ref|` comparison;
- relative residual print and threshold.

## Day 8 Implementation Checklist

1. Measure the current external dense-reference helper block in
   `tests/test_ldlt_csc.c`.
2. Add a local state struct that owns only the helper's temporary arrays and
   factors.
3. Add a single cleanup helper for that local state.
4. Replace repeated free blocks in `assert_ldlt_external_dense_reference`.
5. Keep the solver call and proof assertions in the same helper body.
6. Keep all three external dense-reference call sites unchanged except for any
   formatting required by the cleanup.
7. Record before/after metrics and proof-value visibility in the Day 8
   artifact.

## Focused Validation

If Day 8 edits `tests/test_ldlt_csc.c`, run:

```sh
make build/test_ldlt_csc
build/test_ldlt_csc
```

Because a `.c` file will be modified, Day 8 must also run:

```sh
make format && make lint && make test
git diff --check
```

No CMake test target, Make source list, helper target, public header,
production private header, install/export rule, or reviewed CTest registration
change is expected.

## Explicit Non-Claims

Day 7 does not approve:

- QR sequential RHS cleanup;
- CG preconditioner-specific exact-RHS cleanup;
- GMRES exact-RHS cleanup;
- BiCGSTAB exact-RHS cleanup;
- MINRES exact-RHS cleanup;
- a generic cross-solver exact-RHS helper;
- a generic direct-solver external oracle abstraction;
- production LDLT CSC source movement.

## Completion Criteria

- Exactly one Day 8 cleanup target is selected.
- The selected target is bounded to one direct-solver oracle lane.
- Proof values that carry solver meaning remain visible by contract.
- Day 8 can implement without broad cross-solver abstraction.
