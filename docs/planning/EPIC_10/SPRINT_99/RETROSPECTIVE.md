# Sprint 102 Retrospective

**Sprint:** 102 - Direct Solver Robustness & External Oracle Expansion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-102`)
**Status:** Complete

> Note: this retrospective is stored at the requested path
> `docs/planning/EPIC_10/SPRINT_99/RETROSPECTIVE.md`. The sprint artifacts it
> summarizes live under `docs/planning/EPIC_10/SPRINT_102/`.

## Definition Of Done Checklist

- [x] Sprint 102 started from the Epic 10 project-plan scope and Sprint 100/101
      evidence handoff.
- [x] direct solver evidence gaps were audited across Cholesky, LDLT, LU, QR,
      SVD, and direct dispatch.
- [x] fixture taxonomy rules were defined before adding oracle coverage.
- [x] expected-success, expected-failure, unsupported, and non-claim states
      were separated before implementation.
- [x] the external-reference parsing boundary was extracted into
      `tests/test_solver_helpers.h`.
- [x] Cholesky CSC and LDLT CSC external-reference consumers were migrated to
      the shared parser without changing their family-local proof ownership.
- [x] LDLT CSC gained the `ldlt_kkt_scaled_10` external dense-reference lane.
- [x] linked-list LU gained the `lu_nonsym_square_5` external dense-reference
      lane.
- [x] linked-list LU gained deterministic singular expected-failure coverage
      for `lu_singular_square_4`.
- [x] public and maintainer documentation now state direct-solver selection and
      trust boundaries without broad solver-family overclaims.
- [x] earned, deferred, and non-claim states were reconciled against the Day 3
      fixture taxonomy and Sprint 100 evidence template.
- [x] final validation passed:
  - `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10`
  - `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`
  - `python3 tests/lu_external_dense_reference.py lu_singular_square_4`
    as an expected helper failure
  - `make build/test_ldlt_csc build/test_sparse_lu`
  - `./build/test_ldlt_csc`
  - `./build/test_sparse_lu`
  - `make format`
  - `make lint`
  - `make test`
  - `git diff --check`
  - trailing-whitespace scans
- [x] Sprint 103 handoff requirements, residual queue, and closeout artifacts
      were recorded explicitly.

## What Went Well

1. **The sprint started with evidence design instead of fixture sprawl.**
   Day 2 and Day 3 separated solver families, fixture classes, expected
   outcomes, failure modes, and non-claims before new tests landed. That made
   the later LDLT and LU implementation work easier to bound.

2. **The helper extraction stayed appropriately small.**
   `tests/test_solver_helpers.h` now owns only the shared external-reference
   vector parsing contract. Cholesky, LDLT, and LU still own their fixtures,
   solver calls, tolerances, residual checks, and claim boundaries locally.

3. **LDLT CSC gained a higher-value external fixture without widening claims.**
   `ldlt_kkt_scaled_10` adds a scaled indefinite KKT lane with
   `max|x - x_ref| = 8.882e-15` and `rel_residual = 1.692e-17`, while the
   docs keep the claim tied to the named fixture.

4. **LU received both positive and failure-mode oracle evidence.**
   `lu_nonsym_square_5` proves one deterministic nonsymmetric solve against an
   external dense reference, and `lu_singular_square_4` proves deterministic
   singular handling through `SPARSE_ERR_SINGULAR`.

5. **Public wording now matches evidence.**
   README, tutorial, and maintainer-guide updates describe LU, Cholesky,
   LDL^T, and QR selection rules while preserving explicit boundaries around
   external oracle coverage.

6. **The final reconciliation is concrete.**
   Day 13 ties each earned claim to named tests, helpers, fixtures, metrics,
   and validation commands. It also leaves Sprint 103 with explicit next-lane
   prerequisites.

7. **The full code-touch gate passed before closeout.**
   The branch reran `make format && make lint && make test` after the C/header
   changes, so Sprint 102 closes from current validation rather than relying on
   partial focused checks.

## What Didn't Go Well

1. **The oracle coverage remains intentionally narrow.**
   Sprint 102 improved selected direct-solver evidence, but it did not create
   broad external oracle coverage across all solver families.

2. **QR and SVD stayed deferred.**
   Day 2 identified QR and SVD as meaningful remaining gaps, but the sprint
   correctly spent its implementation capacity on LDLT CSC and linked-list LU.

3. **LU CSR did not receive external dense-reference proof.**
   The first LU oracle lane stayed on linked-list LU to keep the scope bounded.
   LU CSR needs a separate fixture, tolerance, and helper decision.

4. **The public matrix shell remains the solver entry center.**
   Sprint 102 did not add direct public CSR/CSC solver APIs. That remains a
   non-claim even though Sprint 101 improved compressed-input construction.

5. **The branch accumulated a dense artifact package.**
   The Day 14 artifact index and closeout help, but maintainers should start
   with the Day 3, Day 13, and Day 14 summaries instead of reading every daily
   artifact sequentially.

## Final Metrics

### Validation

| Metric | Sprint 102 close state |
|---|---:|
| full branch-level gate | `make format`, `make lint`, and `make test` passed |
| focused LDLT CSC binary | `99` tests, `0` failures, `2318` assertions |
| focused linked-list LU binary | `39` tests, `0` failures, `144` assertions |
| LDLT scaled KKT max solution error | `8.882e-15` |
| LDLT scaled KKT relative residual | `1.692e-17` |
| LU nonsymmetric max solution error | `8.882e-16` |
| LU nonsymmetric residual | `3.553e-15` |
| LU singular fixture status | `SPARSE_ERR_SINGULAR` in C; helper `ERROR` status |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scans | passed on touched code, docs, and Sprint 102 planning files |

### Sprint 102 Artifact Package

| Metric | Sprint 102 close state |
|---|---:|
| total artifact files under `SPRINT_102/artifacts/` | `16` |
| baseline/audit/taxonomy artifacts | `3` |
| helper boundary/extraction/closeout artifacts | `3` |
| CSC oracle artifacts | `3` |
| LU/general solver oracle artifacts | `2` |
| guidance, validation, closeout, and index artifacts | `5` |

Notes:

- baseline, audit, and taxonomy artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-baseline.md`
  - `day2-direct-solver-gap-audit.md`
  - `day3-fixture-taxonomy.md`
- helper artifacts:
  - `day4-oracle-helper-boundary.md`
  - `day5-oracle-helper-extraction.md`
  - `day6-helper-closeout-and-rerank.md`
- CSC oracle artifacts:
  - `day7-csc-oracle-boundary.md`
  - `day8-csc-oracle-expansion-batch.md`
  - `day9-csc-closeout-and-general-rerank.md`
- LU/general solver artifacts:
  - `day10-general-solver-oracle-boundary.md`
  - `day11-general-solver-oracle-expansion-batch.md`
- guidance, validation, and closeout artifacts:
  - `day12-direct-solver-guidance-update.md`
  - `day13-validation-and-evidence-reconciliation.md`
  - `day14-closeout-and-handoff.md`
  - `day14-artifact-index.md`

### Landed Product Surface

| Metric | Sprint 102 close state |
|---|---:|
| shared helper header updated | `tests/test_solver_helpers.h` |
| external dense-reference helper added | `tests/lu_external_dense_reference.py` |
| external dense-reference helper expanded | `tests/ldlt_external_dense_reference.py` |
| focused direct-solver test files updated | `tests/test_chol_csc.c`, `tests/test_ldlt_csc.c`, `tests/test_sparse_lu.c` |
| public documentation files updated | `README.md`, `docs/tutorial.md`, `docs/maintainer_guide.md` |
| new external-reference positive fixture lanes | `2` |
| new deterministic expected-failure fixture lanes | `1` |

## Residual Deferred Debt

Most important carry-forward work:

- QR external dense least-squares or rank oracle lane
- SVD external dense singular-value, rank, reconstruction, or pseudoinverse
  oracle lane
- LU CSR external dense-reference coverage
- direct CSC dispatch oracle reuse rules beyond family-backed routing checks
- public comparison wording for Sprint 103 and later, tied to the Sprint 102
  trust-boundary table
- direct solver benchmark or performance sentinel work, if future claims need
  timing evidence

Still consciously constrained rather than silently solved:

- no direct public CSR/CSC solver APIs
- no broad external oracle coverage for every direct solver
- no QR or SVD external dense-reference lane
- no LU CSR external oracle coverage
- no portable performance superiority claim
- no broad SuiteSparse/PETSc/Trilinos parity or replacement claim
- no broad state-of-the-art solver superiority claim

Not carried forward as unresolved Sprint 102 debt:

- direct solver gap audit
- fixture taxonomy
- external-reference parser extraction
- Cholesky CSC and LDLT CSC migration to the shared parser
- LDLT CSC scaled KKT external-reference lane
- linked-list LU nonsymmetric external-reference lane
- linked-list LU singular expected-failure lane
- direct-solver trust-boundary documentation
- final validation and evidence reconciliation
- Sprint 103 handoff requirements

## Key Deliverables

- [PLAN.md](../SPRINT_102/PLAN.md)
- [WORKING_NOTES.md](../SPRINT_102/WORKING_NOTES.md)
- [day2-direct-solver-gap-audit.md](../SPRINT_102/artifacts/day2-direct-solver-gap-audit.md)
- [day3-fixture-taxonomy.md](../SPRINT_102/artifacts/day3-fixture-taxonomy.md)
- [day4-oracle-helper-boundary.md](../SPRINT_102/artifacts/day4-oracle-helper-boundary.md)
- [day5-oracle-helper-extraction.md](../SPRINT_102/artifacts/day5-oracle-helper-extraction.md)
- [day7-csc-oracle-boundary.md](../SPRINT_102/artifacts/day7-csc-oracle-boundary.md)
- [day8-csc-oracle-expansion-batch.md](../SPRINT_102/artifacts/day8-csc-oracle-expansion-batch.md)
- [day10-general-solver-oracle-boundary.md](../SPRINT_102/artifacts/day10-general-solver-oracle-boundary.md)
- [day11-general-solver-oracle-expansion-batch.md](../SPRINT_102/artifacts/day11-general-solver-oracle-expansion-batch.md)
- [day12-direct-solver-guidance-update.md](../SPRINT_102/artifacts/day12-direct-solver-guidance-update.md)
- [day13-validation-and-evidence-reconciliation.md](../SPRINT_102/artifacts/day13-validation-and-evidence-reconciliation.md)
- [day14-closeout-and-handoff.md](../SPRINT_102/artifacts/day14-closeout-and-handoff.md)
- [day14-artifact-index.md](../SPRINT_102/artifacts/day14-artifact-index.md)

## Bottom Line

Sprint 102 achieved its goal:

- direct-solver oracle work now has explicit fixture taxonomy and evidence
  rules
- Cholesky CSC, LDLT CSC, and linked-list LU share external-reference vector
  parsing without losing family-local proof ownership
- LDLT CSC and linked-list LU gained named external dense-reference evidence
- linked-list LU gained deterministic singular expected-failure evidence
- public documentation now matches bounded direct-solver trust levels
- final validation passed before closeout
- Sprint 103 receives QR, SVD, LU CSR, and comparison work as explicit future
  evidence, not as implied Sprint 102 claims
