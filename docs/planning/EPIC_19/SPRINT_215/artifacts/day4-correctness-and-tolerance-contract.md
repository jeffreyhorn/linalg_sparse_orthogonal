# Sprint 215 Day 4: Correctness And Tolerance Contract

## Authority And Scope

This predeclares the numerical gate for the single Day 3 cell: a real-float64,
unregularized SPD solve on the exact `tests/data/symmetric_4.mtx` bytes
(`ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`),
with `x_ref=[1,1,1,1]` and `b=[13,14,17,16]`. The two paths are the project
no-reorder linked-list Cholesky solve and SuiteSparse `v7.12.2` / CHOLMOD
5.3.4 natural-order simplicial `CHOLMOD_A` solve. Their full binary, build,
matrix-conversion and identity requirements remain in
`day3-baseline-selection-decision.md` and `day3-methodology-inputs.md`.

This is a **specification**, not an observed result. Freeze these formulas
and constants before running the pilot. Do not relax them based on output.
The project and external solution vectors must each pass independently;
matching each other, row labels or process exit codes is not enough.

## Eligible Input And Oracle

Verify the Day 3 file hash, format, 4 x 4 shape, seven stored/ten expanded
entries, finite coefficients and identical full matrix/RHS on both sides
before numerical classification. Keep the original expanded `A` and `b`
untouched for all residuals. The coefficient rows are
`[10,1,2,0]`, `[1,12,1,0]`, `[2,1,14,0]`, `[0,0,0,16]`.
Thus `||A||_inf=17`, `||b||_inf=17`, `||x_ref||_inf=1`, and the minimum
strict-diagonal-dominance margin is 7. For any `Ax=y`, selecting a component
of `x` with maximal magnitude gives `||x||_inf <= ||y||_inf/7`, so
`||A^-1||_inf <= 1/7` and the conservative cell-specific condition bound is
`K=||A||_inf/7=17/7`. These quantities are input facts, not fitted pilot
statistics. If an independent input check disagrees, mark the cell
`invalid-input` and stop; do not recompute a favorable tolerance from a
different matrix under the same key.

## Metrics And Fixed Thresholds

Let `n=4`, `u=DBL_EPSILON=2^-52`, and `T=32*n*u` (about `2.84e-14`).
Compute all metrics from the untouched full `A`, original `b`, and each
solver's own float64 solution `x`. Infinity norms use maximum absolute row
sum for `A` and maximum absolute component for vectors. Reject any nonfinite
coefficient, RHS, solution, intermediate product, norm or metric before
comparing thresholds. Preserve raw values at full precision.

| Metric | Formula | Independent pass rule for each side |
| --- | --- | --- |
| Residual | `r=A*x-b`; `R=||r||_inf`; `D=||A||_inf*||x||_inf+||b||_inf` | `R <= T*max(1,||b||_inf) + T*D`. This is an absolute scale floor plus a relative backward-error term. Record `R` and `eta=R/D`; for this cell `D>0`. |
| Oracle solution error | `E=||x-x_ref||_inf`; `X=||x_ref||_inf` | `E <= T*K*max(1,X) + T*K*X`. Record both `E` and `E/X`; for this cell `X=1`. The factor `K=17/7` accounts for input conditioning. |
| Cross-solver difference | `C=||x_project-x_cholmod||_inf` | Record `C`; if both independent oracle gates pass, `C` must not exceed the sum of their oracle-error limits. This is a consistency check, never a substitute for the two independent gates. |

The rule uses an absolute component for scale near zero and a relative
component for proportional error. It deliberately does **not** demand
bitwise equality or an unscaled fixed decimal cutoff. For another
precision, matrix, RHS or problem class, recalculate a new cell-specific
contract *before* measurement; do not reuse `T`, `K`, the oracle or these
pass labels by implication. If a future matrix lacks a reliable oracle,
solution-error acceptance needs a separate predeclared rule, not a skipped
field counted as a pass.

## Status And Factorization Diagnostics

| Path | Required success observations | Failure observations to retain |
| --- | --- | --- |
| Project direct Cholesky | `sparse_load_mm`, `sparse_cholesky_factor_opts` and `sparse_cholesky_solve` each return `SPARSE_OK`; `used_csc_path=0`; options report no reorder and linked-list backend; output is finite. | Exact `sparse_err_t` for each phase, first failing phase and available diagnostics. `SPARSE_ERR_NOT_SPD`, `SPARSE_ERR_SINGULAR`, `SPARSE_ERR_NUMERIC`, allocation/cancellation or any other non-OK status is not a numerical pass on verified eligible input. Pivot/iteration counts are `not_exposed`/`not_applicable`, never fabricated as zero. |
| CHOLMOD direct Cholesky | Initialization succeeds; `cholmod_analyze` returns a factor; `cholmod_factorize` returns true; `factor->minor=n`; `cholmod_solve(CHOLMOD_A,...)` returns a dense solution; `common.status==CHOLMOD_OK` after each applicable phase; effective ordering is natural, `is_super=0`, and type is real double. | Per-phase return/pointer and `common.status`, `factor->minor`, ordering, `is_ll`, `is_super`, xtype/dtype, and first failed phase. `CHOLMOD_NOT_POSDEF`, `CHOLMOD_DSMALL`, negative errors, null result, or `minor<n` are not success even if a process exits zero. `is_ll` is recorded, not required to match the project's factor storage. |

No iterative method is selected: convergence flags, stopping tolerance,
iteration count and warm start are `not_applicable`. CHOLMOD symbolic
analysis and the project's integrated factor call have different internal
phases; their successful solve outputs can be checked, but factor layouts
and analysis counts are not equality metrics. Ordering is pinned natural/no
reorder on both sides. A changed ordering, pivot policy, backend, RHS or
solve system (for example `CHOLMOD_L` instead of `CHOLMOD_A`) creates a
different operation/cell and is `not_comparable` to this row, even if its
residual happens to pass. Future LU/iterative cells must separately pin
pivots, initial guess, convergence/stopping rules and iteration semantics.

## Classification And Timing Firewall

Apply the checks in this order: (1) identity, availability, input and
eligibility; (2) solver options and actual path; (3) phase statuses and
factor diagnostics; (4) finite outputs; (5) residual and oracle thresholds;
(6) cross-solver consistency. Retain the first failing gate and all raw
observations already available. A missing field is not a passing check.

| Situation | Cell disposition | Timing disposition |
| --- | --- | --- |
| Required package, binary identity, RHS, oracle or execution evidence absent | `blocked` / `not-run` with named missing input | `not-evidence`; no speed comparison |
| Wrong bytes/shape/field, failed conversion, or values inconsistent across sides | `invalid-input` and pair `not_comparable` | `not-evidence` |
| Singular, rank-deficient or indefinite verified matrix for this SPD cell | `inapplicable` and pair `not_comparable`; negative-test record only | `not-evidence` |
| Different solver family, matrix, ordering, backend, precision, RHS, pivot/regularization or solve semantics | `not_comparable`; create a new frozen cell if useful | `not-evidence` for this cell |
| Eligible input, but either solver has failure/warning status, nonfinite output, failed factor diagnostic, residual or oracle limit | `correctness-fail`; record which side and metric | `not-evidence` even if a timer emitted a number |
| Both paths pass all required checks with complete provenance | `correctness-pass` for this one cell | Still `not-run` until Day 5 timing methodology and a separate measured run |

If one side passes and the other fails, report the asymmetric statuses, not
"parity" or a runtime ratio. A failing eligible solve is not relabeled
`inapplicable` after its output is seen. Publication remains separately
gated by fixture and CHOLMOD module/license review.

## Negative Fixtures For Later Validation

These are **planned mutation/test cases**, not runs performed on Day 4.
Each must prove that a validator or pilot harness rejects the row for the
specified reason before it can be included in an external report.

| Fixture / mutation | Expected result |
| --- | --- |
| Change one coefficient only on the external side, or change the file while retaining the selected hash | Input/hash mismatch: `invalid-input`, pair `not_comparable`; no numerical or timing pass. |
| Remove `x_ref` or RHS while retaining the selected cell key | `blocked` and `not_comparable` for this oracle-required cell. A later oracle-free cell needs its own metric contract. |
| Inject a direct-solver failure return or a CHOLMOD `minor<n`/non-OK status on the eligible matrix | `correctness-fail` even if a solution vector or zero process exit is present. |
| Inject a `not_converged` result into a future iterative-family report row | `correctness-fail`, no timing evidence. This is a synthetic status fixture; neither selected direct solver exposes iteration/convergence counts. |
| Replace one solution component with NaN or Inf, or make a residual intermediate nonfinite | `correctness-fail` before tolerance evaluation. |
| Use `CHOLMOD_L` triangular solve, AMD ordering, a warm start, regularization or a different pivot rule under the selected key | `not_comparable` semantics/identity mismatch even if a number is produced. |
| Replace the SPD fixture with a singular or indefinite symmetric matrix | `inapplicable` for this cell; a separate negative-domain test may check expected rejection, never a passing SPD comparison. |

## Day 5 Handoff

Day 5 may define timing phases and machine provenance, but cannot convert
this correctness-only specification into a speed claim. Any measured timing
row must link to both independent numerical gates, the exact Day 3 cell
identity and a separate predeclared measurement protocol. No external solver
run, benchmark, tolerance fitting or new selected-report row was made here.
