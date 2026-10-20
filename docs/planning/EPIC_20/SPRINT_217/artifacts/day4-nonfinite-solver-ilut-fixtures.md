# Sprint 217 Day 4: Nonfinite, Solver, And ILUT Fixtures

**Scope:** [Day 4 plan](../PLAN.md) and [Day 2 contract](day2-numerical-behavior-contract.md).
This is before-fix evidence. Production `src/*.c` and public headers are
unchanged; new test cases are expected to fail until later sprint days.

## Fixture Ownership

| Existing binary / source | Added coverage | Registration |
| --- | --- | --- |
| `test_sparse_vector` / `tests/test_sparse_vector.c` | Positive/negative infinity; NaN alone, mixed with finite, and mixed with infinity in both orders | `RUN_TEST` in existing `main` |
| `test_iterative` / `tests/test_iterative.c` | CG and GMRES ordinary finite controls, tiny nonzero RHS false-success checks, GMRES NaN RHS, CG finite-input initial residual overflow | `RUN_TEST` in existing `main` |
| `test_minres` / `tests/test_minres.c` | Ordinary finite control, tiny RHS false-success check, NaN RHS | `RUN_TEST` in existing `main` |
| `test_bicgstab` / `tests/test_bicgstab.c` | Ordinary finite control, tiny RHS false-success check, finite-input initial residual overflow | `RUN_TEST` in existing `main` |
| `test_ilu` / `tests/test_ilu.c` | `west0067` finite-entry count, maximum magnitude, and independent factor-system backward-error diagnostic | Added to existing `test_ilut_west0067` |

The fixed two-entry RHS `{1e-200, -2e-200}` seeds the deterministic
tiny-solve cases; the finite control uses `{2, -3}`. CG, GMRES, and
BiCGSTAB use a 2x2 identity matrix; MINRES uses the existing 2x2 SPD
tridiagonal builder. Successful tiny solves must pass an independent `hypot`
true-residual ratio check. A non-successful tiny solve may return
`SPARSE_ERR_NOT_CONVERGED` or `SPARSE_ERR_NUMERIC`, but it cannot claim
convergence. The initial-residual overflow fixtures use a finite diagonal
entry of `1e308` with a finite initial guess of `2`, making `A*x` overflow;
their target is `SPARSE_ERR_NUMERIC`. NaN caller-supplied RHS targets
`SPARSE_ERR_BADARG`.

The existing `Makefile` lists all five test sources, and `CMakeLists.txt`
already calls `add_sparse_test` for all five binaries. No new build or CTest
registration was needed.

## Focused Pre-Fix Results

Run locally on 2026-10-20, branch `sprint-217`, base commit
`bde1d873c55a5fbda9fd39531f4ca2c05c7ee3f7`. Toolchain: Apple clang
17.0.0 (`clang-1700.6.4.2`), target `x86_64-apple-darwin24.6.0`.

| Command | Exit | Summary |
| --- | ---: | --- |
| `make format-check` | 0 | Repository C formatting check passed |
| `make lint` | 0 | Strict compile, clang-tidy, and cppcheck completed |
| `make build/test_sparse_vector build/test_iterative build/test_minres build/test_bicgstab build/test_ilu` | 0 | All focused binaries compiled |
| `./build/test_sparse_vector` | 1 | 36 run, 7 failed, 0 skipped, 73 assertions |
| `./build/test_iterative` | 1 | 89 run, 4 failed, 0 skipped, 762 assertions |
| `./build/test_minres` | 1 | 45 run, 2 failed, 0 skipped, 710 assertions |
| `./build/test_bicgstab` | 1 | 51 run, 1 failed, 0 skipped, 381 assertions |
| `./build/test_ilu` | 0 | 34 run, 0 failed, 0 skipped, 328 assertions |

Four vector failures are the unchanged Day 3 finite extremes. Three new
`vec_norminf` NaN cases fail: all-NaN returns zero; finite/NaN returns the
finite maximum; infinity/NaN returns infinity. `vec_norm2` infinity and NaN
cases pass on the current implementation. The new ordinary solver controls
pass, so failures below are not caused by a broken matrix fixture.

| New solver case | Pre-fix output | Day 2 target |
| --- | --- | --- |
| CG tiny RHS | `status=0 converged=1 true_rel=1` | No success unless finite true residual meets tolerance |
| GMRES tiny RHS | `status=0 converged=1 true_rel=1` | Same |
| MINRES tiny RHS | `status=0 converged=1 true_rel=1` | Same |
| BiCGSTAB tiny RHS | `status=0 converged=1 true_rel=1` | Same |
| GMRES NaN RHS | `status=13` (`SPARSE_ERR_NOT_CONVERGED`) | `SPARSE_ERR_BADARG` |
| MINRES NaN RHS | `status=13` (`SPARSE_ERR_NOT_CONVERGED`) | `SPARSE_ERR_BADARG` |
| CG overflowed initial residual | `status=13` (`SPARSE_ERR_NOT_CONVERGED`) | `SPARSE_ERR_NUMERIC` |
| BiCGSTAB overflowed initial residual | `status=14` (`SPARSE_ERR_NUMERIC`), test passes | Keep this behavior |

The tiny-RHS false successes are independently measured: `hypot(b-Ax)/hypot(b)
= 1` while the solver reports success. The input is nonzero and finite; the
old 2-norm's square/sum underflow makes its RHS norm zero. This is a
confirmed convergence defect.

## `west0067` Diagnosis

The diagnostic traverses `L` and `U` with `sparse_get`, separately forms
`u=U*z` and `q=L*u`, uses `p_i=r[perm[i]]` (identity when `perm==NULL`),
and reports the normwise factor-solve backward error from the Day 2 formula.
It does not call the triangular solve to form its reference. It rejects
nonfinite inputs or malformed permutations within the diagnostic by returning
NaN rather than silently producing a passing metric. On this macOS host,
`long double` accumulation is wider than `double`; a portable scale-safe
oracle and enforced threshold remain Days 8-10 work.

```text
west0067: ILUT solve norm=inf
west0067: ILUT finite_z=67/67 max_abs_z=1.914849e+274 factor_eta=8.242351e-80
```

All 67 solution entries are finite. The independent bound
`||z||_2 <= sqrt(67)*max|z| < 1.6e275` is far below `DBL_MAX`, so the
printed infinity is definitely an intermediate-overflow error in `vec_norm2`,
not a nonfinite ILUT solution entry or an unrepresentable mathematical norm.
The small factor-system backward error supports a successful triangular
factor solve on this run, but is not yet an enforced test gate and does not
mean the incomplete factors approximate `A` well. The separate ILUT-right
GMRES run remains nonconverged. The existing `znorm > 0` assertion is still
present and must be replaced by the Day 2 acceptance gate in Days 8-10.

## Next Gate

Day 5 reviews the complete fixture matrix. Do not waive the 14 focused test
failures (7 + 4 + 2 + 1) or report `make test` as passing. Full `make test`
was not run on Day 4; the focused red baselines are the
intended evidence before production changes.
