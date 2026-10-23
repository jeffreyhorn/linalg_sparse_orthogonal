# Sprint 217 Day 6: Scaled 2-Norm Implementation

**Scope:** [Day 6 plan](../PLAN.md), [Day 2 contract](day2-numerical-behavior-contract.md),
and [Day 5 design](day5-fixture-closure-scaled-norm-design.md).

## Change

`src/sparse_vector.c` now computes `vec_norm2` with a scaled sum of squares.
It retains `+0.0` for NULL or `n <= 0` and for all-zero vectors. It scans
past infinities so a later NaN still takes precedence; absent NaN, any
infinity gives `+inf`. Finite nonzero components update a finite `scale`
and `ssq` using ratios at most one. The final `scale*sqrt(ssq)` can overflow
only when the returned norm is too large to represent. No public signature,
error code, `vec_norminf` behavior, or solver status path changed today.

## Focused Evidence

On the Day 5 pre-fix baseline, `test_sparse_vector` had 39 tests, 9 failures,
and 0 skips. After this change it has 39 tests, 3 failures, and 0 skips:
only `test_norminf_nan_only`, `test_norminf_nan_with_finite`, and
`test_norminf_nan_with_infinity` remain red. They belong to Day 7 and were
not skipped or weakened. All six previous `vec_norm2` failures now pass,
including `DBL_MAX` singleton, smallest subnormal, `1e200` pairs,
`1e-200` pairs, and mixed scales. The separate `[DBL_MAX,-DBL_MAX]`
control still returns legitimate `+inf`; NULL, zero, signed-zero, and
negative-length controls pass.

Rebuilt dependent binaries show the expected partial integration effect:

| Binary | Run | Failed | Relevant observation |
| --- | ---: | ---: | --- |
| `test_iterative` | 90 | 3 | CG no longer falsely converges on tiny RHS but returns nonconvergence; GMRES tiny RHS converges with independent relative residual `1.45e-16`; nonfinite input and computed-residual status failures remain |
| `test_block_solvers` | 16 | 1 | Invalid second RHS column still misclassified |
| `test_minres` | 45 | 1 | Tiny RHS converges with independent relative residual `1.30e-16`; nonfinite RHS status remains wrong |
| `test_bicgstab` | 51 | 0 | Tiny RHS no longer falsely converges; it returns nonconvergence with true residual 1; computed-overflow fixture returns numeric error |
| `test_ilu` | 34 | 0 | `west0067` solution entries remain finite and its reported 2-norm is now `1.915e+274`, not `inf`; independent factor backward error remains `8.242351e-80` |

The passing tiny-RHS fixtures reject false success but do not prove every
solver can solve every tiny identity system. Days 8-10 own the remaining
selected-solver status and numerical-path hardening. The ILUT backward-error
gate is still observational and must be enforced in that phase. This is not
a full-suite green claim.

## Validation And Handoff

`make -s format-check`, `make -s lint`, and the focused six-binary build
passed. An isolated UBSan build of `test_sparse_vector` in
`build/day6-ubsan` also compiled cleanly. Its run emitted no UBSan runtime
diagnostic and matched the normal vector result: 39 run, the same three
`vec_norminf` failures, and 0 skips. This sanitizer observation does not
turn the intentionally red vector suite into a pass. The normal vector
binary has no `vec_norm2` failure. Day 7 must update `vec_norminf`
for NaN propagation and rerun the complete norm matrix. Day 8 should begin
from the remaining invalid-input/computed-residual status failures rather
than changing the now-passing norm expectations.
