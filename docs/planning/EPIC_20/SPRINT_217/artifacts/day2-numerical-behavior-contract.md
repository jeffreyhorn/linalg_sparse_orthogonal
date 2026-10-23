# Sprint 217 Day 2: Numerical Behavior Contract

**Scope:** [Day 2 plan](../PLAN.md), informed by the [Day 1 inventory](day1-norm-consumer-inventory.md).
This is an implementation and test specification, not a claim about the
current public API. No C code was changed on Day 2.

## Public Norm Result Matrix

The return type remains `double`; the signatures in `include/sparse_vector.h`
do not change. Preserve the existing compatibility behavior for NULL and
nonpositive lengths even though NULL with positive length does not denote a
valid mathematical vector.

| Input (`n > 0` unless stated) | `vec_norm2` target | `vec_norminf` target |
| --- | --- | --- |
| `v == NULL`, any `n` | `+0.0` | `+0.0` |
| `n <= 0`, any `v` | `+0.0`; do not dereference `v` | `+0.0`; do not dereference `v` |
| All finite zero entries | `+0.0` | `+0.0` |
| Finite entries, nonzero mathematical norm representable as finite `double` | Finite positive result, no intermediate overflow/underflow to `inf` or zero | Largest absolute entry, finite and nonnegative |
| Finite entries, mathematical 2-norm too large to round to finite `double` | `+inf` from final-result overflow only | Still finite (all components are finite) |
| At least one infinity, no NaN | `+inf` | `+inf` |
| At least one NaN, with or without infinity | `NaN` | `NaN` |

For a nonzero vector of representable finite `double` components, the exact
2-norm is at least the largest component magnitude; the target must not return
zero merely because component squares underflow. If the true result is between
finite representable doubles, normal rounding applies. If the correctly
rounded result overflows the return type, infinity is legitimate. `-0.0`
components do not make a nonzero vector and both zero results are `+0.0`.
No errno, global error state, or new error code is introduced by these
scalar-returning helpers. A NaN scan takes precedence over an infinity
result, independent of element order.

## Arithmetic And Accuracy

Use a scaled sum-of-squares recurrence for finite `vec_norm2` inputs:
`scale = 0`, `ssq = 1`; for each `a = fabs(v[i]) > 0`, set
`ssq = 1 + ssq*(scale/a)^2, scale = a` if `a > scale`, otherwise set
`ssq += (a/scale)^2`. Return `scale*sqrt(ssq)` (or zero when `scale == 0`).
Inspect every element for NaN before returning infinity so the result is
order-independent. This is the standard scaled accumulation pattern, not a
plain sum of squared components. Avoid a `double` intermediate whose overflow
would turn a representable result into infinity. `vec_norminf` needs only a
NaN-propagating absolute maximum, with the same early NULL/length rules.

For independent regression oracles, use analytically known powers of two or
`hypot` on two-element inputs; do not reuse the implementation recurrence in
the expected-value calculation. For finite normal expected values, require
relative error at most `8*DBL_EPSILON`; for subnormal expected values, at most
two ULPs. Include exact zero, singleton, mixed scales, negative entries,
`1e200`, `1e-200`, `DBL_MAX`, smallest subnormal, infinity, NaN, and mixed
NaN/infinity in the matrix. The tolerance is a test acceptance bound, not a
claim of correctly rounded results for every length. A mathematically
unrepresentable 2-norm must be tested separately from an intermediate
overflow bug.

## Selected Iterative-Solver Status Matrix

Applies to CG and matrix-free CG, GMRES and matrix-free GMRES, MINRES,
BiCGSTAB and matrix-free BiCGSTAB, plus block CG where norms are used.
The handle-backed variants inherit the corresponding solver contract.
Use only existing `sparse_err_t` values in
`include/sparse_types.h:125-157`; no public enum change is assumed.

| Case | Required status and result |
| --- | --- |
| `n == 0` or genuinely zero finite RHS | Preserve the existing successful trivial-system behavior; zero the solution for a zero RHS where the selected solver currently does, report `converged=1`, residual `0.0`. |
| Nonfinite caller-supplied RHS or initial guess at entry | `SPARSE_ERR_BADARG`; `converged=0`, no successful residual claim. Do not reclassify NULL, shape, or option errors. |
| Finite nonzero RHS whose norm formerly underflowed to zero | Do not take the zero-RHS branch. Continue solving using a positive scaled norm. |
| Nonfinite initial or updated computed residual norm, Arnoldi/Lanczos norm, or relative ratio (including unrepresentable ratio) | `SPARSE_ERR_NUMERIC`; `converged=0`, `residual_norm=NAN` when a result is supplied. Do not use a NaN comparison or `inf/inf` to decide convergence. |
| Finite initial residual exactly zero | `SPARSE_OK`, `converged=1`, residual `0.0`; avoid dividing by a zero Krylov norm. |
| Zero Krylov/Lanczos norm with nonzero finite true residual | Existing breakdown/nonconvergence status and `breakdown=1` where supported; do not label it successful convergence or a nonfinite-norm error. |
| Finite relative residual above tolerance or iteration limit | Existing `SPARSE_ERR_NOT_CONVERGED`/stagnation or breakdown behavior; `converged=0` unless a separately verified finite true residual meets tolerance. |
| Finite verified true relative residual within tolerance | `SPARSE_OK`, `converged=1`; report that finite relative residual. |

For `n > 0`, validate caller-supplied RHS and initial guess before taking the
zero-RHS fast path; a nonfinite initial guess with zero RHS is therefore
`SPARSE_ERR_BADARG`. Keep the existing `n == 0` no-op ordering and do not
read zero-length vectors. This validation order is an intentional selected-
solver behavior decision, not a description of the current implementation.

The result struct is reset according to existing entry behavior. On a new
numeric failure after work has begun, preserve the number of completed
iterations/history entries if available, but never leave `converged=1` or a
finite-looking residual of zero. A `NAN` diagnostic is deliberate: it
indicates that no meaningful finite relative residual was established. For
block CG, any invalid column makes the aggregate call numeric failure; the
maximum-column diagnostic must not silently ignore NaN. Existing allocation,
preconditioner callback, cancellation, and option-error precedence remains
unchanged. Public solver documentation must describe any newly returned
`SPARSE_ERR_BADARG` or `SPARSE_ERR_NUMERIC` in Days 11-12.

For zero tolerance, only an exactly zero residual can converge. For positive
tolerance, compare finite nonnegative norms without an underflowing
`rnorm/bnorm` manufacturing zero or an overflowing quotient manufacturing a
false success; an unrepresentable relative ratio is a numeric failure under
this selected-solver contract. Explicit finiteness checks precede each
denominator use and final convergence comparison. MINRES's separate direct
sum-of-squares true-residual calculations require the same protection.

## `west0067` ILUT Acceptance Calculation

Fixture and current assertion: `tests/test_ilu.c:1122-1167`. The selected
factorization uses `tol=1e-4`, `max_fill=20`, default `pivot=0`, and an RHS of
all ones. Keep ILU(0)'s expected singular outcome separate from ILUT's solve
check. The current `znorm > 0` assertion is insufficient because `inf > 0`.

1. Require `sparse_ilut_factor` and `sparse_ilu_solve` to return `SPARSE_OK`
   before reading factors or `z`; fail on either status rather than using
   uninitialized output. Require all `z[i]`, `L` entries, `U` entries, and
   selected RHS entries to be finite. A finite `z` is not required to have a
   finite 2-norm if its exact 2-norm exceeds `DBL_MAX`; report that separately.
2. Define the effective row-permuted RHS as `p_i = r[ilut.perm[i]]` when
   `perm != NULL`, otherwise `p_i = r[i]`. Validate `perm` as an in-range
   bijection before indexing. This follows `src/sparse_ilu.c:195-205`, where
   solve applies the permutation. Do not assume pivoting is permanently off.
3. Independently form `u_i = sum_j U(i,j)*z_j` and
   `q_i = sum_j L(i,j)*u_j` by traversing factor rows, not by calling the
   triangular solve or reusing its forward/back-substitution work arrays.
   Account for the stored unit diagonal in `L`. Use an overflow-resistant
   accumulator or scaling strategy for validation; a floating-point overflow
   in the oracle must fail explicitly, never become an infinite denominator
   that passes a ratio check.
4. Compute the normwise factor-solve backward error
   `eta = ||q-p||_inf / (||L||_inf * ||U||_inf * ||z||_inf + ||p||_inf)`.
   Use scale-aware evaluation of products and sums; require all terms and
   `eta` to be valid. The nonzero RHS makes a zero denominator a test error.
   Accept only `eta <= max(1e-12, 64*n*DBL_EPSILON)`. This bound scales with
   the selected system size and double-precision accumulation; the test should
   print `eta` and the threshold on failure.
5. Negative controls must reject an injected NaN or infinity in `z`, a
   non-`SPARSE_OK` solve status, and a zeroed `z` (which yields `eta=1` for
   nonzero `p`). Check both `perm == NULL` and a nonidentity permutation in
   unit-level oracle tests. These controls prove the gate is discriminating.

`L*U` is an incomplete, dropped approximation to `A`; therefore this gate
checks whether `z` solves the computed factor system, **not** whether it is an
accurate solution of `A*z=r`. An `A*z-r` diagnostic may be reported, but it
must not be used as an exact-solve acceptance criterion. The Day 1
`west0067: ILUT solve norm=inf` line remains ambiguous until the finite-entry
and factor-product checks run. If the new gate fails on the current tree,
record the measured failure and repair the selected ILUT path in Days 8-10;
do not relax the gate merely to make the baseline pass.

## Acceptance Cases And Ownership

| Case | Expected outcome | Planned work |
| --- | --- | --- |
| `vec_norm2([1e200, 1e200])` | Finite, approximately `sqrt(2)*1e200` | Days 3-7 |
| `vec_norm2([1e-200, 1e-200])` | Positive, approximately `sqrt(2)*1e-200` | Days 3-7 |
| `vec_norm2([DBL_MAX, DBL_MAX])` | `+inf` because final norm is not representable | Days 3-7 |
| Either norm with NULL or `n <= 0` | `+0.0`, no dereference | Days 3-7 |
| Either norm with `[NAN, INFINITY]` in either order | `NaN` | Days 3-7 |
| Infinity norm with `[NAN, 3.0]` | `NaN`, not `3.0` | Days 3-7 |
| Tiny nonzero RHS in selected iterative solvers | Not classified as zero RHS | Days 8-10 |
| Nonfinite solver input versus computed residual | `SPARSE_ERR_BADARG` versus `SPARSE_ERR_NUMERIC`, no convergence claim | Days 8-10 |
| `west0067` ILUT solution | Status, finite entries, and independent `eta` gate; no exact `Az=r` claim | Days 8-10 |

These decisions close Sprint 217 item 217.1's contract phase. They do not
close implementation, regression, public documentation, or validation items.
