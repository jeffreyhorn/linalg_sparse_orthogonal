# Sprint 53 Day 11: Regression Expansion Batch

## Purpose

Day 11 closes the smallest remaining high-value CSC proof gap after the main
Sprint 53 landing work. The goal is not more generic coverage. The goal is to
prove that the already-documented cheap same-pattern guard and old-factor
preservation behavior also hold on the highest-value indefinite repeated-run
path, not only on the earlier SPD path.

## Main Day 11 Result

Sprint 53 now has direct proof that the public repeated-run LDL^T CSC path:

- rejects obvious `nnz` drift on the above-threshold indefinite KKT workflow
- returns `SPARSE_ERR_BADARG` at the shared refactor boundary
- preserves the previously valid factor state so solves on the original matrix
  still succeed afterward

This closes a real proof gap:

- Sprint 52 already proved cheap `nnz`-drift rejection and old-factor
  preservation on the SPD repeated-run path
- Sprint 53 already proved same-pattern indefinite refactor success
- before Day 11, Sprint 53 did **not** yet prove the bounded failure contract
  on that same high-value indefinite CSC repeated-run path

## Touched Code

### `tests/test_integration.c`

Day 11 adds one focused regression:

- `test_public_lifecycle_ldlt_refactor_rejects_nnz_drift_and_preserves_old_factors_amd`

The test uses the same above-threshold indefinite KKT workload shape as the
Sprint 53 success-path proof:

1. build `kkt-150`
2. analyze with:
   - `SPARSE_FACTOR_LDLT`
   - `SPARSE_REORDER_AMD`
3. factor once through the public repeated-run path
4. copy the original matrix
5. remove one symmetric coupling pair to create obvious `nnz` drift
6. assert:
   - `sparse_refactor_numeric(...) == SPARSE_ERR_BADARG`
7. solve again with the old factors against the original RHS
8. confirm the exact solution is still recovered

That makes the test valuable in two ways at once:

- it proves the cheap gross-structure guard on the high-value indefinite path
- it proves the old-factor preservation contract on the same path

## Why This Was the Right Day 11 Gap

After Days 4-10, the live proof surface already covered:

- one-shot vs explicit-analysis parity on indefinite KKT
- same-pattern indefinite refactor success
- reordered indefinite repeated-run success
- benchmark-side indefinite repeated-run evidence
- LDL^T CSC dispatch wording and layered pipeline interpretation

What remained weaker was the bounded failure side of that same repeated-run
story. The SPD path already had it. The indefinite CSC path did not.

Day 11 fixes exactly that without widening into:

- implementation churn
- broad `test_ldlt_csc.c` growth
- benchmark-framework work
- more documentation reconciliation

## Validation

Because `tests/test_integration.c` changed, Day 11 ran the full required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Day 11 also ran the focused follow-ons justified by the touched surface:

- `./build/test_integration`
- `./build/test_ldlt_csc`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`

Representative direct results:

- `./build/test_integration`
  - `37 / 37`
- `./build/test_ldlt_csc`
  - `96 / 96`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
  - `workflow = ldlt_kkt`
  - `speedup_refactor = 1.26x`
  - `res_public = 2.96e-16`
  - `res_csc = 2.96e-16`

## Operational Result

Sprint 53's proof surface is now better balanced:

1. success-path indefinite repeated-run proof already existed
2. bounded failure-path indefinite repeated-run proof now exists too
3. the measured indefinite benchmark surface stayed unchanged and healthy

That leaves Day 12 to audit compatibility and claimed behavior from a branch
whose LDL^T CSC repeated-run story now has both success and failure proof on
the main high-value path.
