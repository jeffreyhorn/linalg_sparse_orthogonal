# Sprint 178 Day 11: Scoped Claim Documentation

## Scope

Day 11 updates public and maintainer wording for the selected Sprint 178
allocation-failure proof. The new positive claim is intentionally narrow:

- selected subsystem: `sparse_matmul()`;
- selected allocation sites: accumulator, nonzero-flag, and touched-column
  workspaces;
- maintained Make command: `make matmul-allocation-failure-gate`;
- maintained CTest selector: `ctest --test-dir <build-dir> -L matmul`;
- registration guard:
  `python3 tests/test_matmul_allocation_failure_gate_registration.py`.

## Public README Update

The README now distinguishes two selected allocation-failure proofs:

- Sprint 176 iterative repeated-run handles for CG, GMRES, and MINRES
  prepare/growth cleanup;
- Sprint 178 `sparse_matmul()` workspace cleanup, stale-output suppression, and
  retry-after-reset behavior.

The README command list now includes the focused matrix multiply gate so local
maintainers have an exact command for the new proof.

## Maintainer Guide Update

The maintainer guide now records `tests/test_matmul.c` as the evidence owner
for the Sprint 178 lane and lists the regression names guarded by the focused
Make target:

- `test_matmul_acc_allocation_failure_clears_stale_output`;
- `test_matmul_remaining_workspace_allocation_failures_clear_stale_output`;
- `test_matmul_workspace_allocation_failure_recovers`;
- `test_matmul_error_precedence_clears_stale_output`.

It also records the registration guard that keeps Makefile, CMake, and test
registration aligned.

## Protected Non-Claims

Day 11 keeps these surfaces outside the earned claim:

- matrix shell construction;
- insertion and product-flush allocation;
- matrix copy, transpose, CSR/CSC conversion, and build helpers;
- direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines, and
  reorder routines;
- package/install flows;
- generated-report tooling;
- broad allocation-failure cleanup coverage.

## Validation

- `make matmul-allocation-failure-gate`
- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `rg -n "allocator-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178 || true`
- `git diff --check`

## Handoff

Day 12 should run the integrated Sprint 178 validation sweep and reconcile the
evidence table without broadening the selected allocation-failure claim.
