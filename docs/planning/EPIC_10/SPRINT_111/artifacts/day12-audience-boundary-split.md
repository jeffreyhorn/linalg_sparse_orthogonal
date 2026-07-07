# Day 12 Maintainer/User Surface Split

## Purpose

Day 12 cleans remaining adoption-facing surfaces that still interrupted user
workflow guidance with maintainer proof-owner detail. The goal is not to
delete evidence; it is to keep evidence traceable from maintainer-oriented
locations while keeping README/tutorial prose focused on supported workflows,
constraints, and examples.

## Touched Files

- `README.md`
- `docs/tutorial.md`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day12-audience-boundary-split.md`

## Before / After

| Surface | Before | After |
|---|---|---|
| `README.md` direct-solver section | Listed external dense-reference lanes and non-claims in the front-door adoption path. | Points to `docs/maintainer_guide.md` for direct-solver evidence boundaries and keeps README focused on choosing supported workflows. |
| `docs/tutorial.md` getting started | Used "owner surface" wording in the learning path. | Uses user-facing "next level of detail" wording. |
| `docs/tutorial.md` LU section | Explained bounded external dense-reference LU evidence and non-claims inline. | Keeps LU guidance focused on one-shot use, copying, and repeated-run handoff. |
| `docs/tutorial.md` Cholesky section | Ended with external dense-reference evidence wording. | Keeps Cholesky guidance focused on one-shot use, repeated-run handoff, refactor behavior, and preserving the original matrix view. |
| `docs/tutorial.md` LDL^T section | Included Sprint-era KKT evidence detail in the tutorial. | Keeps LDL^T guidance focused on when to use the solver and when to move to the explicit repeated-run lifecycle. |

## Evidence Traceability

The moved evidence remains available through:

- `docs/maintainer_guide.md`
  - "Support Surface Ownership"
  - "Sprint 102 Direct Solver Trust Boundary Snapshot"
- test owners named there:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `tests/test_ldlt_csc.c`
  - `tests/ldlt_external_dense_reference.py`
  - `tests/test_sparse_lu.c`
  - `tests/lu_external_dense_reference.py`
- this Day 12 artifact, which records the specific adoption-surface wording
  moved out of README/tutorial flow.

## Audience Boundary Decisions

- README should stay the project front door and compact workflow chooser.
- Tutorial should stay the longer user learning path after README.
- Maintainer guide should own proof-owner interpretation and evidence-boundary
  detail.
- Benchmark README should continue to own benchmark command and measurement
  interpretation; its maintainer-oriented lane names remain because they are
  live report surfaces and were clarified on Day 11.
- Planning artifacts should preserve sprint provenance and before/after
  cleanup rationale.

## Remaining Audience-Boundary Debt

- README still includes compact reviewed-quality and CI wording because those
  commands are common contributor entry points; future cleanup should avoid
  expanding that section into a maintainer handbook.
- `benchmarks/README.md` remains detailed by design. Its top-level Day 11
  interpretation section now gives users the quick reading contract, while the
  later maintainer-oriented lane detail remains benchmark-local.
- `docs/algorithm.md` is still a mixed technical-history reference, not a
  first-time adoption surface. It should be handled separately if a future
  sprint wants a cleaner user/reference split.

## Validation

Day 12 changed documentation only. Validation:

- `git diff --check`
- trailing-whitespace scan over touched Day 12 docs

## Completion Criteria Status

- User docs now read as adoption material first in the touched sections.
- Maintainer proof records remain available through the maintainer guide and
  named test owners.
- No evidence was deleted without a replacement reference.
- README and tutorial remain coherent after the wording movement.
