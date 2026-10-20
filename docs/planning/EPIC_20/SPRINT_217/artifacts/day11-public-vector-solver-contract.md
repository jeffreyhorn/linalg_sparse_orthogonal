# Sprint 217 Day 11: Public Vector And Solver Contract

**Scope:** [Day 11 plan](../PLAN.md), [Day 2 numerical contract](day2-numerical-behavior-contract.md), and [Day 10 solver record](day10-iterative-convergence-hardening.md).

## Header-To-Regression Traceability

| Public contract | Header | Regression evidence |
| --- | --- | --- |
| `vec_norm2` uses scaled accumulation; finite representable extremes stay finite, final-result overflow is infinity, and a subnormal nonzero input stays nonzero | `include/sparse_vector.h` | `tests/test_sparse_vector.c`: large, tiny, mixed-scale, `DBL_MAX`, and smallest-subnormal cases |
| Both norms preserve NULL/nonpositive-length zero behavior; NaN takes precedence over infinity | `include/sparse_vector.h` | `tests/test_sparse_vector.c`: NULL/empty, NaN/Inf in both orders, zero cases |
| Nonfinite caller RHS or initial guess is `SPARSE_ERR_BADARG`; computed invalid residual/norm/ratio is `SPARSE_ERR_NUMERIC` with no convergence and NaN diagnostic | `include/sparse_iterative.h` | `tests/test_iterative.c`, `tests/test_minres.c`, `tests/test_bicgstab.c`, `tests/test_block_solvers.c`: Sprint 217 entry, overflow, preconditioner, and matrix-free cases |
| A nonzero recurrence estimate whose relative ratio underflows to zero is not successful convergence | `include/sparse_iterative.h` | `tests/test_stagnation.c`: MINRES stagnation/numeric-failure case |
| ILUT's successful factor-system solve is distinct from quality of `A*z=r`; nonfinite RHS/factors/substitution or overflow returns numeric failure and output may be partial | `include/sparse_ilu.h` | `tests/test_ilu.c`: `west0067` finite/independent factor-error gate, finite-input overflow and nonfinite negative controls |

The iterative contract names only CG, GMRES, MINRES, BiCGSTAB, their covered
matrix-free variants, and block CG. Handle-backed variants inherit the
corresponding single-RHS contract. No independent numerical parity is claimed
for the other block adapters or unrelated solver families. Callback,
allocation, cancellation, and other hard failures do not promise usable
solution or residual fields. The solver documentation keeps zero-length and
finite zero-RHS behavior separate from invalid nonempty inputs.

`vec_dot` and other vector helpers retain their existing signatures and
historical NULL behavior; scaled or nonfinite guarantees are stated only for
the two norm helpers. The ILUT factorization documentation does not claim
that successful factors are a useful preconditioner on `west0067`.

## Local Reference Inspection

`make -s api-docs-validate` passed, including Doxygen coverage, local-only,
and routing guards. After the final callback-status wording correction,
`make -s docs-check` passed again: Doxygen generated 18 public-header
reference pages and 18 source pages, and the API coverage check passed.
The generated
`sparse__vector_8h.html` page shows the scaled-norm/range and NaN wording;
`sparse__iterative_8h.html` shows the selected-solver scope note and
`SPARSE_ERR_NUMERIC` returns on the touched declarations; and
`sparse__ilu_8h.html` shows the partial-output and factor-system caveats.
The scope note initially appeared only in source, because an unattached
Doxygen comment block was discarded; moving it into the `@file` block made
it visible in the regenerated reference. `make -s format-check` and
`git diff --check` passed after the final header edit. No code behavior or
signatures changed on Day 11;
the Day 10 full-suite result remains the most recent test-suite run.
