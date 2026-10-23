# Sprint 217 Day 10: Iterative Convergence Hardening

**Scope:** [Day 10 plan](../PLAN.md), [Day 2 numerical contract](day2-numerical-behavior-contract.md), and the Day 3-5 selected-solver fixtures.

## Decision Points

The selected CG, matrix-free CG, GMRES, matrix-free GMRES, MINRES, BiCGSTAB,
matrix-free BiCGSTAB, and block CG paths now validate finite caller RHS and
initial-guess entries before taking the nonempty zero-RHS fast path. NULL,
shape, option, and empty-system ordering remains intact. A nonfinite caller
vector returns `SPARSE_ERR_BADARG` with `converged=0`.

An internal relative-residual helper rejects nonfinite or negative norms,
zero/nonfinite denominators, an overflowing quotient, and a nonzero numerator
whose quotient underflows to zero. Computed invalid data returns
`SPARSE_ERR_NUMERIC`, `converged=0`, and `residual_norm=NAN`; completed
iterations and recorded history count are retained where available.

CG and matrix-free CG guard their initial and updated residuals, dot-product
denominators, recurrence coefficients, and output, then recompute `b-Ax`
before the final status. GMRES guards its initial true residual (including
the zero-iteration path), preconditioned Arnoldi vector and norm, rotation
and estimated residual, solution update, and final true residual. A zero
preconditioned Krylov norm after a nonconverged true residual is a breakdown,
not convergence. MINRES guards its initial/preconditioned norms, Lanczos
norm, rotations, recurrence estimate, and true residual; the latter uses the
scaled vector norm rather than a direct sum of squares. BiCGSTAB's two
variants guard entry, intermediate relative residuals, and the final true
residual. Block CG validates all columns up front and recomputes all true
residuals before publishing aggregate convergence. The independent-column
block adapter no longer hides a numeric column's NaN diagnostic behind a
finite maximum from another column.

## Discrepancy Ledger

| Selected case | Before Day 10 | Final result |
| --- | --- | --- |
| CG computed residual overflow | `NOT_CONVERGED` | `NUMERIC`, NaN diagnostic, no convergence |
| CG nonfinite guess with zero RHS | `SPARSE_OK` | `BADARG`, no convergence |
| GMRES nonfinite RHS | `NOT_CONVERGED` | `BADARG`, no convergence |
| MINRES nonfinite RHS | `NOT_CONVERGED` | `BADARG`, no convergence |
| Block CG invalid second RHS column | `NOT_CONVERGED` | `BADARG`, no convergence |
| GMRES, MINRES, block CG computed overflow | Not explicitly guarded | `NUMERIC`, NaN diagnostic |
| CG/GMRES/MINRES nonfinite preconditioner output | Could enter recurrence unchecked | `NUMERIC`, NaN diagnostic |
| BiCGSTAB nonfinite input or computed ratio | Inconsistent entry/final handling | `BADARG` for input, `NUMERIC` for computed failure |
| Finite zero RHS and selected preconditioned controls | Existing success behavior | Retained by family-local and full tests |

The full-suite `test_minres_stagnation_detected` fixture initially failed:
at iteration 3295 its nonzero `phi_bar=-9.8813129168249309e-324` divided by
`||b||=5.4772255750516612` rounded to zero. The new numeric failure is the
Day 2 contract's intended outcome, not a zero-residual convergence. The
fixture now accepts that status only with `converged=0`, a NaN diagnostic,
and an iteration count below its 5000-step limit; its historical finite
stagnation/nonconvergence expectations remain unchanged.

## Verification

After the final code cleanup, `make -s format-check` and `make -s test` both
passed. The focused binaries also passed: `test_iterative` 93/93 (783
assertions), `test_minres` 46/46 (719), `test_bicgstab` 52/52 (387), and
`test_block_solvers` 17/17 (179). Targeted `clang-tidy` on the three changed
solver sources and targeted `cppcheck` on those sources plus their changed
tests passed. An initial targeted `clang-tidy` run identified five dead
stores after final true-residual verification; these were removed before the
passing rerun. No repository-wide `make lint` result is claimed.

Item 217.3's planned numerical-status discrepancy ledger is closed for the
selected paths. Day 11 still owns the public-header wording that describes
these return statuses and result-field validity.
