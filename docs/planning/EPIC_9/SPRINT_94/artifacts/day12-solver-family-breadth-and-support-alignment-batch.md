# Sprint 94 Day 12 - Solver-Family Breadth and Support Alignment Batch

## Scope
- Required center:
  - `docs/maintainer_guide.md`
- Directly forced support-only follow-through:
  - `README.md`
- Explicitly not reopened in this batch:
  - solver-family implementation owners
  - focused capability tests
  - package/install/export surfaces

## Landed Changes
- Tightened `docs/maintainer_guide.md` so the bounded scalar interpretation
  now explicitly reflects:
  - the real shared matrix-shell helper plus storage/build seam
  - the already-real iterative/eigs/QR public scalar seams
  - the current maintained non-claims around later scalar and later
    algorithm-family widening
- Updated the maintainer proof-owner map so the validated Sprint 94 baseline
  now includes:
  - `tests/test_sparse_matrix.c` for the shared matrix-shell width/scalar seam
  - `tests/test_sparse_io.c` for the touched Matrix Market width-aware
    parse-rejection seam
  - the already-real iterative/eigs/QR proof owners
- Tightened `README.md` so the public real-only scalar limitation matches the
  current real landing set without implying broader numeric genericity

## Preserved Invariants
- no solver-family implementation owner was reopened
- no new proof binary or package/install contract was introduced
- no broad complex or mixed-precision claim was added
- no dense/SVD family-wide scalar widening claim was implied

## Validation
- Re-read the touched support surfaces against the Day 11 design and the
  validated Day 10 baseline

## Result
- Sprint 94's final capability landing set is now aligned across code, proof,
  maintainer wording, and public wording
- the sprint remains inside the bounded scalar/index contract with no extra
  solver-family implementation drift
