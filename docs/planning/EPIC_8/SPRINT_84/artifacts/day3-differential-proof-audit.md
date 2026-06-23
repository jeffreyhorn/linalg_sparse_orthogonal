# Sprint 84 Day 3: Differential-Proof Audit

## Purpose

Reduce Sprint 84's broad assurance problem to one ranked live contradiction
map so the sprint can choose one bounded maintained differential lane instead
of another generic “more tests” bucket.

## Main Result

Sprint 84's broad assurance problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - bounded maintained external differential proof on the core direct-family
    SPD lane centered first on Cholesky CSC
- strongest second target:
  - deterministic seeded property expansion beyond the current bounded
    lifecycle/property seams
- strongest third target:
  - failure-path numerical proof on the most fragile cancellation,
    lifecycle-preservation, and residual-accounting seams
- strongest fourth target:
  - iterative and eigensolver external differential follow-through
- strongest support-only but real target:
  - CI/docs/support wording that still reflects the narrower current assurance
    reading

## Strongest Current Contradiction

The strongest current contradiction is not the absence of internal proof:

- `tests/test_chol_csc.c` already owns large SuiteSparse residual checks,
  scalar-vs-batched cross-checks, and path-selection proof
- `tests/test_ldlt.c` already owns residual, refine, lifecycle, and
  cross-backend proof
- `tests/test_iterative.c` already owns true-residual, SuiteSparse, and
  direct-solver comparison proof
- `tests/test_eigs.c` already owns dense cross-checks, SuiteSparse Ritz
  residuals, refinement checks, and SVD-side agreement checks
- `tests/test_fuzz.c` already owns bounded seeded generative lifecycle
  property follow-through

The contradiction is that the highest-value maintained external differential
lane fixed by Sprint 80 still has not landed:

- Sprint 80 froze the first maintained external-oracle lane as a
  CHOLMOD-class SPD Cholesky comparison
- the current tree still proves the core direct-family SPD lanes mainly by
  internal residual, cross-path, and generated-property checks
- benchmark and example surfaces remain intentionally non-oracle surfaces

That fixes the strongest first Sprint 84 move:

- land one bounded maintained external differential lane on the direct SPD
  Cholesky family first
- treat broader solver-family external comparisons as follow-through only if
  that first lane lands cleanly

## Second-Tier Contradictions

### Seeded Property Breadth

The strongest second contradiction is property breadth:

- `tests/test_fuzz.c` already covers LU, Cholesky, QR, SVD, and the large-`n`
  direct lifecycle parity lanes
- deterministic property coverage is still narrower than the current public
  lifecycle and repeated-run assurance surface

This is real Sprint 84 work, but it reads as follow-through after the first
maintained external differential lane is explicit.

### Failure-Path Numerical Proof

The strongest third contradiction is fragile failure-path numerical proof:

- `tests/test_integration.c` already owns cancellation and lifecycle
  preservation semantics across direct, QR, iterative, and eigensolver lanes
- `tests/test_iterative.c` and `tests/test_eigs.c` already pin several true
  residual and refinement invariants
- the most fragile cancellation/error-path/cross-check guarantees are still
  bounded and family-local rather than widened into one clearer assurance
  package

### Iterative / Eigensolver External Depth

The strongest fourth contradiction is iterative/eigs external proof depth:

- these lanes already have stronger internal residual and direct-comparison
  proof than the direct-family external lane has today
- Sprint 80's oracle fence does not justify making them the first maintained
  external comparison center ahead of the bounded direct SPD lane

## Deferred Assurance Claims

Broad assurance-claim widening remains lower-value first work:

- no repo-wide claim that every solver now has maintained external proof
- no benchmark or example drift into correctness ownership
- no broad dependency story for untouched families
- no reopening Sprint 83's capability-surface owner work
- no support-surface churn detached from a real landed proof seam

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 84 move is not generic property expansion
- it is one bounded maintained external differential landing on the direct SPD
  Cholesky lane that Sprint 80 already fenced as first
- seeded property widening follows next
- failure-path numerical proof follows after that where the first lanes expose
  the real fragility
- iterative/eigs external comparisons remain real, but they are explicitly
  later than the first direct-family lane
- CI/docs/support surfaces stay support-only unless implementation truly moves
  the assurance contract

## Exit State

- Sprint 84 now has one ranked live assurance contradiction map grounded in
  the current tree.
- The first implementation center is fixed to one bounded maintained external
  differential lane on the direct-family SPD Cholesky path.
- Later seeded-property, failure-path, iterative/eigs external, and support
  follow-through work is explicitly ordered behind that first lane.
