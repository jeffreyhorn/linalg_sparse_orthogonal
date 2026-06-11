# Sprint 62 Day 3: Direct One-Shot Usability Audit

Date: 2026-06-10
Branch: `sprint-62`


## Purpose

Reduce the broad Sprint 62 direct-usability goal to a ranked live map of
caller-facing pain points by auditing the one-shot direct headers, wrapper
behavior, lifecycle crossover, and the strongest existing regression proofs.

## Audit Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_62/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
- public direct/lifecycle headers:
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_qr.h`
  - `include/sparse_analysis.h`
- implementation seams:
  - `src/sparse_lu.c`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
- caller-story and proof surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_sparse_lu.c`
  - `tests/test_qr.c`

## Ranked Day 3 Conclusions

### 1. The broad “direct usability needs work” claim reduces to four concrete pain-point classes

The live pressure is concentrated in:

1. mutable-matrix surprise on one-shot in-place paths
2. wrapper versus explicit lifecycle ambiguity
3. cancellation/progress semantics that differ by solver family
4. copy-discipline and “fresh original matrix” friction in docs/examples

This is a smaller and more actionable problem than a generic “direct APIs are
hard to use” backlog.

### 2. LU is the strongest first Sprint 62 target

LU is the clearest first hardening seam because it mixes the most behavior into
one public story:

- one-shot in-place mutation
- reorder-before-factor behavior
- cancellation nuances across reorder and elimination boundaries
- `sparse_lu_factor_opts(...)` fast-path crossover into the shared
  `analysis` / `factors` lifecycle for the default-compatible option shape

That makes LU the highest wrapper/lifecycle ambiguity risk and the strongest
first target for Sprint 62.

### 3. Cholesky is the strongest second target, but mainly for mutation surprise and backend clarity

Cholesky is already more explicit than LU about being a copied-matrix one-shot
surface, but it still carries meaningful usability risk:

- the matrix is mutated in place
- the upper triangle is stripped
- the CSC/linked-list backend split sits behind the same public wrapper
- cancellation is family-specific and not bit-identical to the pre-call state

So Cholesky is the strongest second target, but for mutation/backend reasons
rather than the same lifecycle crossover that makes LU first.

### 4. LDL^T is cleaner than the Epic 6 review summary implied

LDL^T still belongs in the Sprint 62 design space, but it is not the best
first landing target:

- the family-local one-shot surface owns a separate `sparse_ldlt_t`
- the input matrix is not mutated
- cancellation leaves the input matrix bit-identical
- the family/header already distinguishes the owned-factor path from the
  shared repeated-run direct lifecycle

The strongest remaining LDL^T gap is coherence follow-through, not the
highest-severity one-shot usability risk.

### 5. QR belongs mostly as a contrast surface, not as the defining Sprint 62 landing target

QR still matters to shared caller expectations because it requires an
unfactored, unreordered matrix with identity permutations. But it does not
share the same repeated-run direct lifecycle convergence problem as LU,
Cholesky, and LDL^T.

That makes QR useful as a comparison surface for docs and expectations, but not
the strongest first code target.

### 6. The strongest current proof burden already lives in integration-level direct regression

The repo already has meaningful direct proof coverage:

- one-shot default wrapper parity
- one-shot versus explicit analysis/factor path parity
- lifecycle mismatch and failure-preservation checks
- cancellation coverage
- one-shot versus lifecycle equivalence for a stable-pattern Cholesky story

So Sprint 62 is not blocked by missing proof infrastructure. The real need is
to sharpen the public/internal direct contract and then add only the smallest
new regression surface needed to freeze it.

## Day 3 Ranked Target Order

1. LU one-shot wrapper and lifecycle coherence
2. Cholesky one-shot mutation and backend clarity
3. LDL^T coherence follow-through
4. QR as a contrast/deferred expectation surface

## Day 3 Exit State

Sprint 62 now has a bounded ranked usability map instead of a generic direct
cleanup claim:

- LU is the strongest first hardening target
- Cholesky is the strongest second target
- LDL^T is important but not first
- QR should stay mostly out of the first landing batch

The next step is to convert this ranking into an explicit lifecycle/wrapper
coherence design and preserved compatibility contract before any code changes
land.
