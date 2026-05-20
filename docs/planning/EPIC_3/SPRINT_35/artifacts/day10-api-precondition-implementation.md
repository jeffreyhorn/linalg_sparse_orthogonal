# Sprint 35 Day 10: API Precondition Language Implementation

## Scope

Implement the narrow wording fixes identified on Day 9:

- matrix-state assumptions
- matrix-class / preconditioner guidance
- QR least-squares vs minimum-norm selection

The goal is to tighten public guidance without rewriting the already-correct
installed headers.

## Main Result

Day 10 closes the highest-value residual prose gap from Sprint 35.

The tutorial and README now surface the important usage assumptions that were
already present in the installed headers but still too implicit in the
user-facing docs.

## What Changed

### 1. Tutorial prose now explains original-matrix expectations

The Day 9 audit showed that several routines expect an original matrix view
with identity permutations, but the tutorial often only implied that through
example code.

Day 10 now says this directly in the tutorial where it matters:

- QR section
- preconditioning section
- SVD section

This is the main value of the pass. A user can now see the matrix-state rule
without needing to infer it from `sparse_copy()` or by reading the headers.

### 2. Tutorial prose now gives clearer matrix-class guidance

The preconditioning section now states the intended pairing more explicitly:

- IC(0) for SPD workflows
- ILU(0) / ILUT for general or indefinite-system workflows

That also makes the preconditioned-CG assumptions easier to understand
operationally.

### 3. QR routine selection is clearer in the public docs

The user-facing docs now distinguish:

- `sparse_qr_solve()` for square / overdetermined least-squares and a basic
  underdetermined solution
- `sparse_qr_solve_minnorm()` for the underdetermined minimum-2-norm path

This is now reflected both in the tutorial and in the README API summary.

### 4. The headers did not need rewriting

No header ambiguity was uncovered during the pass. The installed headers were
already the precise contract surface; the missing work was in the prose layer.

That is a healthy result:

- headers remain the detailed contract
- tutorial gives the main operational guidance
- README gives short signposts only

## Residual Queue for Day 11

The remaining Sprint 35 work is now support-doc and duplication cleanup, not
core API safety debt:

- `INSTALL.md`
- `examples/README.md`
- `benchmarks/README.md`
- any remaining README duplication around quality/workflow guidance

## Bottom Line

Day 10 keeps the sprint on the right track:

- the important usage assumptions are now visible in public prose
- no stale API naming remains
- no header churn was needed
- Day 11 can focus on support-doc polish instead of still patching core
  README/tutorial safety language
