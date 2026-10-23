# Sprint 217 Day 8: Norm Finalization And Initial ILUT Gate

**Scope:** [Day 8 plan](../PLAN.md), [Day 2 contract](day2-numerical-behavior-contract.md),
and the [captured `west0067` solution](day8-west0067-z-values.md).

## Norm Review

The public signatures remain unchanged. `vec_norm2` uses a finite
scale/sum-of-squares recurrence; `vec_norminf` propagates NaN rather than
silently skipping it. Both retain the historical `+0` result for NULL and
nonpositive lengths. The Day 3-7 fixtures cover ordinary and signed-zero
values, normal and smallest-subnormal finite scales, mixed magnitudes,
representable near-`DBL_MAX` magnitudes, true final overflow, both infinity
signs, and NaN in either order with infinity. The complete vector binary
passes 40/40 tests with no skips. Day 7's isolated UBSan vector run also
passed without a runtime diagnostic. No remaining helper precision warning
or implementation blocker was observed on this host; item 217.2's planned
28 hours are complete. This does not establish universal correct rounding
or every selected solver's status contract.

## ILUT Acceptance Change

The old `test_ilut_west0067` allowed a failed ILUT factorization to skip the
solve without failing and used `znorm > 0`, which accepts infinity. It also
read `z` after a nonfatal solve assertion. The Day 8 test now requires:

1. ILU(0)'s expected singular status remains separate. ILUT factorization
   must return `SPARSE_OK` with nonnull `L` and `U`; solve must return
   `SPARSE_OK` before any output entry is inspected.
2. All 67 `z` entries must be finite. A conservative bound
   `sqrt(n)*max|z|` must be finite and below `DBL_MAX`; this selected fixture
   therefore requires a finite, positive reported 2-norm.
3. The independent oracle traverses the computed `U` and `L` factors to
   form `L(Uz)`, checks factor entries and the effective permuted RHS, and
   computes `eta = ||L(Uz)-Pr||_inf / (||L||_inf||U||_inf||z||_inf+||Pr||_inf)`.
   It must be finite and at most `max(1e-12,64*n*DBL_EPSILON)`. The test
   prints the observed `eta` and limit.

Observed locally: factor and solve statuses are success, finite entries
`67/67`, `max|z|=1.914849e+274`, upper bound `1.567371e+275`, 2-norm
`1.915e+274`, and `eta=8.242351e-80 <= 1e-12`. The `inf` printed before
Day 6 was caused by overflow of component squares in the old norm. It was
not a nonfinite solve output or a mathematical norm beyond `double` range.
The 67 raw values are kept in the linked snapshot for reproducibility, not
as fixed cross-platform golden results.

This is a factor-system solve check, **not** a claim that dropped ILUT factors
solve the original `A*z=r` system exactly or precondition GMRES well.
Day 9 still owns negative controls for nonfinite/zeroed output, permutation
handling, and portability/scale review of the oracle. The current oracle
uses `long double` products and fails closed if its denominator or error is
nonfinite; platforms where `long double` has only double range may need a
scaled evaluation before this gate can be treated as portable evidence.

## Focused Result

`test_sparse_vector`: 40 run, 0 failed, 0 skipped, 85 assertions.
`test_ilu`: 34 run, 0 failed, 0 skipped, 334 assertions. The opt-in
`SPARSE_TEST_ILUT_VALUES=1` run also passed and emitted all 67 entries.
`make -s format-check`, `make -s lint`, and the focused binary build passed
after the final test cleanup edit. The initial formatting pass found one
line-wrap violation in the new diagnostic print; it was corrected before
these passing checks. The selected-solver status fixtures from Day 7 remain
outside this change and prevent a full-suite green claim.
