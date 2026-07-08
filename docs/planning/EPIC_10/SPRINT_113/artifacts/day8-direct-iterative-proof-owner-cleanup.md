# Sprint 113 Day 8: Direct/Iterative Proof Owner Cleanup

## Purpose

Implement the selected direct/iterative proof-owner cleanup from Day 7 without
changing public API, build registration, or reviewed CTest scope.

## Selected Owner

Day 8 cleaned up the LDLT CSC external dense-reference oracle in
`tests/test_ldlt_csc.c`.

This was selected because the owner had a clear local boundary:

- the fixture key, fixture builder, and tolerance remain at each call site;
- exact-RHS setup remains visible in the oracle helper;
- permutation and unpermutation proof flow remains visible;
- dense-reference read, max-difference, and residual checks remain visible;
- repeated allocation and cleanup noise was contained in one helper.

## Before/After Metrics

- Before: `tests/test_ldlt_csc.c` had 3896 lines.
- After: `tests/test_ldlt_csc.c` has 3915 lines.
- Net change: +19 lines after replacing repeated cleanup branches with a local
  owner object and explicit error cleanup paths.

## Code Changes

- Added `ldlt_external_dense_reference_state_t` as a local owner for the
  external dense-reference oracle fixture state.
- Added `ldlt_external_dense_reference_state_free` to centralize cleanup of:
  - sparse fixture matrix;
  - first and second LDLT factors;
  - permuted matrix;
  - exact solution, RHS, permuted RHS, permuted solution, solved solution, and
    dense reference vectors.
- Added `ldlt_external_dense_reference_state_alloc` to keep vector allocation
  grouped at the oracle boundary.
- Refactored `assert_ldlt_external_dense_reference` to use the local state owner
  and cleanup helper.
- Replaced the fatal solve macro path with explicit `ASSERT_ERR` plus cleanup so
  allocation ownership remains clear on failure.

## Proof Visibility Preserved

The cleanup intentionally kept the following values and behaviors explicit in
`assert_ldlt_external_dense_reference`:

- `x_true[i] = i + 1`;
- exact RHS construction through `sparse_matvec`;
- two-pass indefinite factorization;
- RHS permutation through `F1->perm`;
- `ldlt_csc_solve`;
- solution unpermutation through `F1->perm`;
- external dense-reference read status;
- `max|x-x_ref|` checks;
- solved-vs-exact checks;
- relative residual calculation and assertion.

## Validation

Focused validation passed:

```sh
make build/test_ldlt_csc && build/test_ldlt_csc
```

Result:

- `test_ldlt_csc`: 100 tests run;
- 0 failed;
- 0 skipped;
- 3556 assertions.

Full required quality chain passed because C code changed:

```sh
make format && make lint && make test
```

Result:

- formatting completed;
- strict warning build completed;
- `clang-tidy` completed;
- `cppcheck` completed;
- full test suite passed.

## Drift Assessment

No scope drift was introduced:

- no public API changes;
- no install-header changes;
- no helper-target changes;
- no Makefile or CMake source-list changes;
- no reviewed CTest registration changes;
- no new external reference corpus dependency.

## Remaining Direct/Iterative Cleanup Queue

The following candidates remain deferred for later sprint days or future sprint
planning:

- QR sequential RHS setup;
- CG preconditioner-specific exact-RHS setup;
- GMRES exact-RHS setup;
- BiCGSTAB exact-RHS setup;
- MINRES exact-RHS setup;
- broad direct/iterative oracle abstraction, which is still intentionally not
  approved without a larger design pass.
