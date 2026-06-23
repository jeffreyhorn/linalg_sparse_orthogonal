# Sprint 84 Day 4: First Assurance Boundary

## Purpose

Fix the first bounded assurance implementation fence for Sprint 84 so the next
design pass can define one real oracle/property/failure-path contract instead
of another broad proof rewrite.

## Main Result

Sprint 84 now has one explicit first implementation fence:

- required first landing:
  - `tests/test_chol_csc.c`
- support only if the first landing truly forces it:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_framework.h`
  - `tests/test_ldlt.c`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- explicitly deferred from the first landing:
  - `tests/test_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
  - generic seeded-property expansion as a first-batch center
  - broad failure-path numerical-proof widening as a first-batch center
  - iterative/eigs maintained external comparisons
  - benchmark/reporting surfaces as correctness owners
  - package/runtime/dependency-matrix widening

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 84 move is the direct-family SPD external
  differential lane on the Cholesky CSC proof owner
- seeded-property widening remains the strongest second seam, not the first
  implementation center
- failure-path numerical proof remains real, but it is explicitly later than
  the first external differential landing unless that landing forces it
- iterative/eigs external follow-through remains real, but it is explicitly
  later than the first direct-family lane
- proof and support surfaces stay support-only unless the first landing truly
  changes behavior there

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no repo-wide claim that every solver now has maintained external proof
- no benchmark or example drift into oracle ownership
- no broad external dependency story for untouched families
- no seeded-property or failure-path expansion ahead of the first external
  differential contract
- no reopening Sprint 83's capability-surface owner work
- no support-surface churn detached from a real landed assurance seam

## Exit State

- Sprint 84 now has one bounded first assurance landing center.
- Day 5 can design one oracle/property/failure-path architecture contract
  inside that fence.
- Lower-value seeded-property, failure-path, iterative/eigs external, and
  broader support/dependency spillover work is held back until later lanes.
