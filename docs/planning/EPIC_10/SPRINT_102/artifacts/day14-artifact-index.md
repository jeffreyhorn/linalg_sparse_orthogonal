# Sprint 102 Artifact Index

## Root Files

| file | role |
|---|---|
| `PLAN.md` | 14-day Sprint 102 execution plan |
| `WORKING_NOTES.md` | day-by-day working notes, findings, validation expectations, and closeout state |

## Daily Artifacts

| day | artifact | role |
|---:|---|---|
| 1 | `artifacts/day1-authoritative-inputs.txt` | authoritative Sprint 102 planning inputs |
| 1 | `artifacts/day1-scope-baseline.md` | scope baseline, workstream inventory, validation rules, and claim boundaries |
| 2 | `artifacts/day2-direct-solver-gap-audit.md` | Cholesky, LDLT, LU, QR, SVD, and dispatch oracle-depth audit |
| 3 | `artifacts/day3-fixture-taxonomy.md` | solver fixture classes, expected outcomes, naming rules, and non-claims |
| 4 | `artifacts/day4-oracle-helper-boundary.md` | external dense-reference helper extraction boundary |
| 5 | `artifacts/day5-oracle-helper-extraction.md` | shared external-reference parser extraction evidence |
| 6 | `artifacts/day6-helper-closeout-and-rerank.md` | helper closeout and next oracle-lane rerank |
| 7 | `artifacts/day7-csc-oracle-boundary.md` | LDLT CSC scaled-KKT oracle boundary |
| 8 | `artifacts/day8-csc-oracle-expansion-batch.md` | LDLT CSC scaled-KKT implementation and validation evidence |
| 9 | `artifacts/day9-csc-closeout-and-general-rerank.md` | CSC-family closeout and LU/QR/SVD rerank |
| 10 | `artifacts/day10-general-solver-oracle-boundary.md` | linked-list LU oracle and failure-mode boundary |
| 11 | `artifacts/day11-general-solver-oracle-expansion-batch.md` | linked-list LU external-reference and singular-failure implementation evidence |
| 12 | `artifacts/day12-direct-solver-guidance-update.md` | public and maintainer direct-solver guidance update evidence |
| 13 | `artifacts/day13-validation-and-evidence-reconciliation.md` | full validation results, evidence reconciliation, earned/deferred/non-claim states, and Sprint 103 dependencies |
| 14 | `artifacts/day14-closeout-and-handoff.md` | Sprint 102 closeout, Sprint 103 handoff, residual queue, and retrospective inputs |
| 14 | `artifacts/day14-artifact-index.md` | complete Sprint 102 artifact index |

## Code and Documentation Changes

| surface | role |
|---|---|
| `tests/test_solver_helpers.h` | shared external-reference vector parser used by direct-solver tests |
| `tests/test_chol_csc.c` | Cholesky CSC external-reference lane migrated onto the shared parser |
| `tests/test_ldlt_csc.c` | LDLT CSC external-reference lane migrated and expanded with `ldlt_kkt_scaled_10` |
| `tests/ldlt_external_dense_reference.py` | LDLT external dense-reference helper expanded with the scaled KKT fixture |
| `tests/test_sparse_lu.c` | linked-list LU external-reference and singular expected-failure tests |
| `tests/lu_external_dense_reference.py` | linked-list LU external dense-reference helper |
| `README.md` | bounded direct-solver selection and trust-boundary wording |
| `docs/tutorial.md` | direct-solver trust notes for LU, Cholesky, and LDL^T |
| `docs/maintainer_guide.md` | Sprint 102 direct-solver proof-owner and trust-boundary table |

## Sprint 102 Evidence Flow

| phase | artifacts |
|---|---|
| input and scope baseline | Day 1 |
| solver gap audit and fixture taxonomy | Day 2-3 |
| helper boundary and extraction | Day 4-6 |
| CSC-family oracle expansion | Day 7-9 |
| general LU oracle expansion | Day 10-11 |
| guidance update | Day 12 |
| validation and reconciliation | Day 13 |
| closeout and handoff | Day 14 |

## Primary Handoff Files

Sprint 103 should start with these files:

1. `artifacts/day3-fixture-taxonomy.md`
2. `artifacts/day12-direct-solver-guidance-update.md`
3. `artifacts/day13-validation-and-evidence-reconciliation.md`
4. `artifacts/day14-closeout-and-handoff.md`
5. `artifacts/day14-artifact-index.md`
6. `WORKING_NOTES.md`

## Validation Reference

Final Sprint 102 validation is recorded in:

- `artifacts/day13-validation-and-evidence-reconciliation.md`
- `artifacts/day14-closeout-and-handoff.md`
- `WORKING_NOTES.md`
