# Sprint 217 Day 9: ILUT Validity And Failure Behavior

**Scope:** [Day 9 plan](../PLAN.md), [Day 2 numerical contract](day2-numerical-behavior-contract.md), and the [Day 8 ILUT gate](day8-norm-finalization-ilut-gate.md).

## Factor-System Acceptance

`test_ilut_west0067` requires successful factorization and solve before reading
`z`, nonnull factors, 67 finite entries, a finite representable 2-norm, and
an independent factor-system backward error below
`max(1e-12, 64*n*DBL_EPSILON)`. The oracle forms `L(Uz)` from stored factors,
uses `r[perm[i]]` when pivoting is active, and checks a bijective permutation.
It now scales factor coefficients and `z` before multiplication, then combines
the factor and RHS scales using binary exponents. This avoids overflowing an
intermediate product on platforms where `long double` has only `double` range.
It rejects nonfinite factors, inputs, permutations, and undefined zero-scale
cases. The threshold has a `1e-12` floor to allow the selected sparse factor
and substitution roundoff while remaining far below an order-one mismatch.

The positive controls solve a known two-by-two factor system both with and
without a row swap. A zeroed solution gives error 1; NaN/Inf solution entries,
a NaN factor, and duplicate permutation entries are rejected. The extreme
one-by-one control has `U[0,0] = z[0] = DBL_MAX`, `r[0] = 0` and gives error
1 without evaluating their overflowing raw product.

For `west0067` on this host: ILU(0) reports singular; ILUT factorization and
solve return `SPARSE_OK`; all 67 entries are finite; `||z||_2 = 1.915e+274`,
`max|z| = 1.914849e+274`, and `sqrt(67)*max|z| = 1.567371e+275 < DBL_MAX`.
The scaled oracle reports `eta = 8.249302e-80 <= 1e-12`. This validates the
computed factor-system solution, **not** `A*z=r` for incomplete factors or
useful preconditioning of this pathological matrix.

## Failure Contract

`sparse_ilu_solve` now returns the existing `SPARSE_ERR_NUMERIC` for a
nonfinite cached factor norm, RHS entry, stored factor entry, substitution
sum, or output. The permuted RHS allocation is freed on every new failure
path. A one-by-one fixture checks a finite successful solve, NaN RHS,
infinite U diagonal, NaN L diagonal, and finite operands whose quotient
overflows; it then restores valid factors and verifies a successful solve.
A permuted failure followed by success exercises the allocated RHS path.
Output is documented as potentially partial on failure; tests do not treat
it as a valid solution after a non-OK status.

No nonfinite selected `west0067` factor or solve result was observed. The
numeric failure path is a negative-control hardening, not a claim that the
original finite `west0067` path was failing.

## Focused Evidence

Commands: `make -s build/test_ilu`, `./build/test_ilu`. Final result: 37 run,
0 failed, 0 skipped, 520 assertions. The relevant output was:

```text
west0067: ILUT returns success
west0067: ILUT solve norm=1.915e+274
west0067: ILUT finite_z=67/67 max_abs_z=1.914849e+274 norm_upper_bound=1.567371e+275 factor_eta=8.249302e-80 limit=1.000000e-12
west0067 ILUT-right-GMRES(30): 500 iters, res=6.228e+16, conv=0
west0067 pivot-ILUT-GMRES: 500 iters, res=2.151e+05, conv=0
```

Both GMRES fixtures require a recognized completion status, positive
iteration count, finite reported and independently measured residuals, and
finite solution entries. Neither converges in 500 iterations; this is not
reported as solver convergence or as a useful ILUT preconditioner. The
selected solver-status fixtures outside ILUT remain assigned to Day 10;
no full-suite green claim is made here.

`make -s format-check`, touched-source `clang-tidy`, touched-source/test
`cppcheck`, and `git diff --check` passed. A repository-wide `make -s lint`
run reached `cppcheck` file 106/109 without a changed-file diagnostic, but
was interrupted after more than 18 minutes; its exit status is **not** a
lint pass. The focused static checks were rerun after the final C edits.
