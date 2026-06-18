# Sprint 79 Day 5 - Differential / Oracle Batch Design

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Define the bounded implementation/proof contract for the first Sprint 79 assurance landing so the sprint can improve one real lifecycle/property seam without widening into broader proof churn or support-surface edits.

## Main Result
Sprint 79 now has one explicit first implementation contract.

Required implementation center:
- `tests/test_integration.c`
- `tests/test_fuzz.c`

Support only if the first batch truly forces it:
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- `docs/maintainer_guide.md`
- `README.md`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

## Ownership Split
Primary public oracle owner:
- `tests/test_integration.c`

Bounded seeded property owner:
- `tests/test_fuzz.c`

Family-local support proof owners only if the public/property seam truly needs them:
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`

Public/support interpretation owners only if wording truly moves:
- `docs/maintainer_guide.md`
- `README.md`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

## Day 6 Batch Goal
The strongest Day 6 batch goal is now fixed:
- strengthen the public callback/cancel and repeated-run lifecycle assurance surface
- prefer one bounded public-oracle improvement first in `tests/test_integration.c`
- add bounded seeded generative/property follow-through in `tests/test_fuzz.c` only where it increases assurance without widening the contract

## Preserved Guarantees
The first batch must preserve:
- current public callback/cancel behavior
- current family/path-local caveat reading
- current reviewed validation scope, including the Windows fuzz exclusion truth
- current benchmark/reporting, install/export, and workflow ownership splits
- current runtime-bounded interpretation of the fuzz/property lane

## Useful Clarification
The first batch should not try to “fix every deferred direct-usability item.”

It should instead:
- land one proof improvement that narrows the residual queue with the highest public assurance payoff
- start with a public oracle improvement
- add property follow-through only where it makes that oracle harder to regress

## Non-Touch Set
The first batch should explicitly not widen into:
- unrelated solver-family proof owners:
  - `tests/test_qr.c`
  - `tests/test_reorder_nd.c`
- benchmark/reporting surfaces
- install/export proof scripts
- workflow YAML surfaces
- broader docs churn
- unrelated implementation or API work

## Exit State
- The first Sprint 79 assurance landing is explicitly designed.
- Ownership, compatibility, and truthfulness fences are fixed in writing.
- Day 6 can now land one bounded lifecycle/property assurance improvement from a precise contract.
