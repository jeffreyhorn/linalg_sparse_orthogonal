# Sprint 217 Day 7: Infinity Norm And Norm Integration

**Scope:** [Day 7 plan](../PLAN.md), [Day 2 numerical contract](day2-numerical-behavior-contract.md),
and the [Day 6 scaled 2-norm](day6-scaled-norm-implementation.md).

## Norm Result

`vec_norminf` now returns NaN as soon as a NaN magnitude is encountered.
The existing maximum scan remains in place for finite values and positive or
negative infinity. The NULL and nonpositive-length `+0.0` fast path is
unchanged. `vec_norm2` is unchanged from Day 6; it still scans past infinity
so a later NaN takes precedence. No public signature or error enum changed.

The complete `test_sparse_vector` suite passes: 40 run, 0 failed, 0 skipped,
85 assertions. It includes all Day 3-5 fixtures plus a Day 7 finite
near-maximum pair `[DBL_MAX/2,-DBL_MAX/2]`, checked against independent
`hypot` at `8*DBL_EPSILON` relative tolerance. The existing ordinary
`[3,4]` case, `1e200` and `1e-200` pairs, smallest-subnormal singleton/pair,
`DBL_MAX` singleton, genuinely overflowing `DBL_MAX` pair, signed zero,
NULL/empty, both infinities, and NaN in either order with infinity all pass.
The subnormal pair is checked within two smallest-subnormal ULPs. These
checks implement every Day 2 public-norm result class; no norm-specific
blocker remains. The tolerance is a regression gate, not a universal
correct-rounding claim.

## Focused Call-Site Smoke

All six dependent binaries rebuilt against the new norm helpers. Results:

| Binary | Run | Failed | Skipped | Relevant result |
| --- | ---: | ---: | ---: | --- |
| `test_iterative` | 90 | 3 | 0 | Tiny-RHS cases pass; nonfinite RHS, CG computed-residual, and CG invalid-guess precedence cases remain red |
| `test_block_solvers` | 16 | 1 | 0 | Nonfinite second-column status case remains red |
| `test_minres` | 45 | 1 | 0 | Tiny-RHS case passes; nonfinite RHS status case remains red |
| `test_bicgstab` | 51 | 0 | 0 | Tiny-RHS case rejects false success; computed-overflow case returns numeric error |
| `test_ilu` | 34 | 0 | 0 | `west0067` still has 67 finite entries, 2-norm `1.915e+274`, and observational factor error `8.242351e-80` |
| `test_integration` | 58 | 0 | 0 | Existing integration residual checks show no regression |

The five remaining red tests are selected-solver contract failures, not
norm-helper failures. The full repository suite cannot be called green yet;
none of these fixtures were skipped or had their expectations relaxed.

`make -s format-check` and `make -s lint` passed. An isolated UBSan rebuild
of `test_sparse_vector` in `build/day6-ubsan` passed all 40 tests with no
runtime diagnostic. This is focused vector validation, not a sanitizer run
over every solver.

## Solver Guard Checklist For Days 8-10

There is no production `vec_norminf` caller. All production helper consumers
use `vec_norm2`. Its new behavior removes arithmetic false zero/overflow
for representable finite norms but makes NaN/infinity classification
reliable; callers must not rely on unordered comparisons to reject them.

| Owner and branch | Concrete guard or follow-up |
| --- | --- |
| `src/sparse_iterative.c:65-85` shared trivial-system helper, used by CG, GMRES, and BiCGSTAB (including matrix-free/handle variants) | Validate nonfinite caller RHS and initial guess before zero-RHS success for `n>0`; reject a nonfinite RHS norm rather than letting `bnorm != 0.0` carry NaN into ratios. Preserve the `n==0` no-read path. |
| `src/sparse_iterative.c:280-377,463-559` CG and matrix-free CG | Validate initial/updated computed `rnorm` and `rnorm/bnorm` before `<= tol`, history, callback, and result writes. The current finite-input overflow fixture returns NOT_CONVERGED instead of NUMERIC. Check zero-residual success separately from zero-denominator breakdown. |
| `src/sparse_iterative.c:633-683` GMRES zero-iteration and initial-residual fast paths | Validate the quotient before `<= tol` and result publication; the nonfinite-RHS fixture currently returns NOT_CONVERGED instead of BADARG. |
| `src/sparse_iterative.c:754-835,879-948` GMRES Arnoldi and final true residual | Reject nonfinite `beta`, `H(j+1,j)`, and relative residual before reciprocal normalization, lucky-breakdown, history, and final convergence. An infinity is not a valid basis scale; NaN makes `<`/`<=` unordered. |
| `src/sparse_iterative.c:1054-1256,1317-1491` matrix-backed and matrix-free BiCGSTAB | Cover initial, half-step, updated, and final true ratios, not only existing coefficient and updated-`rnorm` checks. The final `if (true_rel_res > tol)` does not reject NaN. Preserve already-passing computed-overflow status. |
| `src/sparse_iterative_minres.c:46-103,159-217,258-275` MINRES | Validate input before its separate zero-RHS branch, finite `beta`/`gamma` and ratios before normalization or convergence, and replace/protect both direct square-and-sum true-residual calculations. Its nonfinite-RHS fixture currently returns NOT_CONVERGED. |
| `src/sparse_iterative_block.c:76-83,112-127,157-203` block CG | Validate every RHS/guess column, each norm and quotient before per-column convergence and maximum reporting. A NaN quotient currently cannot increase `max_res`; the second-column fixture returns NOT_CONVERGED instead of BADARG. |

The selected-solver status contract is detailed in the Day 2 artifact. These
locations are implementation obligations for Days 8-10, not evidence that
all listed failure modes have already been reproduced. Nonfinite ratios and
the ILUT numerical gate still need targeted acceptance tests as those paths
are hardened.
