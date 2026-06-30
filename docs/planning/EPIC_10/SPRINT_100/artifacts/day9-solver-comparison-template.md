# Sprint 100 Day 9 Solver Comparison Evidence Template

## Purpose

Day 9 creates a reusable template for solver-family comparison artifacts. The
template is meant for future direct, iterative, eigensolver, SVD, reorder, and
graph evidence work. It is intentionally claim-aware: it separates correctness,
convergence, timing, unsupported cases, and non-claims.

## Files

| file | role |
|---|---|
| `templates/solver-comparison-evidence-template.md` | reusable blank template for future comparison artifacts |
| `day9-solver-template-pilot-cholesky-csc.md` | pilot-filled example using the existing Cholesky CSC external dense-reference lane |

## Template Design Requirements

Future solver comparison artifacts should include:

- solver family and exact API path
- claim being evaluated
- owner files and validation commands
- fixture identity and fixture class
- matrix source, dimensions, sparsity, symmetry, definiteness, and conditioning
  notes where known
- solver options, reorder/backend/runtime settings, and environment
- RHS/eigenpair/problem construction
- external oracle or deterministic reference behavior
- tolerance model
- correctness metrics
- convergence metrics where applicable
- timing metrics only when explicitly local and non-portable
- unsupported or skipped cases
- non-claims and claim boundaries
- validation command output summary
- follow-up residuals

## Required Separations

| evidence type | must be separate because |
|---|---|
| correctness | numerical equivalence and residual checks are not timing claims |
| convergence | iteration counts and stagnation behavior are not direct-solver correctness |
| timing | local wall time is not portable performance superiority |
| unsupported cases | skipped helpers or platform exclusions must not look like passes |
| non-claims | bounded fixtures must not imply ecosystem parity |

## Existing Lane Patterns Used

| pattern | current owner | template effect |
|---|---|---|
| external-process dense reference | `tests/chol_external_dense_reference.py`, `tests/ldlt_external_dense_reference.py` | oracle command and oracle status are first-class fields |
| deterministic RHS | C harness constructs `x_true = i + 1`, then `b = A*x_true` | RHS construction is required |
| explicit tolerance | Cholesky and LDLT harnesses pass tolerances to assertion helpers | tolerance model is required |
| platform skip | external helpers skip on Windows in the C harness | unsupported environment field is required |
| family-local owner | maintainer guide warns not to reinterpret benchmarks/examples as oracle owners | non-claim section is required |

## Usage Notes

1. Fill the template before widening a comparison lane.
2. Treat benchmarks as timing/reporting surfaces unless the sprint explicitly
   designs oracle ownership for them.
3. For external references, record how the reference is invoked and how skip,
   error, and success states are distinguished.
4. For iterative/eigensolver lanes, record convergence and residual criteria
   separately from any final solution or eigenpair comparison.
5. For reorder/graph lanes, record fill, cut, balance, determinism, or memory
   metrics separately from wall time.
6. Keep a "Non-Claims" section even when the comparison passes.

## Completion Rule

A future comparison claim is not earned unless the filled template names:

- the fixture set;
- the oracle or reference behavior;
- the tolerance or acceptance criteria;
- the validation command;
- unsupported cases;
- the non-claims that remain after the evidence passes.

