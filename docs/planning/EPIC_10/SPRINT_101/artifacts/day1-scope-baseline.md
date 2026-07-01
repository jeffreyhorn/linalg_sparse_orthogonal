# Sprint 101 Day 1 Scope Baseline

## Purpose

Day 1 opens Sprint 101 by converting the Epic 10 project-plan section and the
Sprint 100 handoff requirements into a bounded implementation package. The
sprint is an implementation sprint, but Day 1 is intentionally a setup day:
audit, design, implementation, docs, tests, and validation need clear order
before code changes begin.

## Day 1 Deliverables

| deliverable | status | location |
|---|---|---|
| Sprint 101 workstream inventory | complete | this artifact |
| working-notes baseline | complete | `../WORKING_NOTES.md` |
| initial artifacts directory | complete | `../artifacts/` |
| authoritative input list | complete | `day1-authoritative-inputs.txt` |
| validation expectation list | complete | this artifact and working notes |

## Sprint 101 Workstreams

| # | workstream | project-plan item | day ownership | expected output |
|---:|---|---|---|---|
| 1 | Storage surface audit | Item 1 | Days 2-3 | public storage and solver entry audit artifacts |
| 2 | Compressed-first API design | Item 2 | Days 4-5 | bounded API design and implementation boundary |
| 3 | Constructor/import implementation | Item 3 | Days 6-7 | selected implementation batch, tests, post-batch audit |
| 4 | Lifecycle and ownership clarification | Item 4 | Days 8-9 | lifecycle design, ownership/error follow-through, tests |
| 5 | Compatibility path documentation | Item 5 | Days 10-11 | public wording design and docs/examples batch |
| 6 | Regression proof | Item 6 | Day 12 | compressed-first regression proof artifact |
| 7 | Validation and closeout | Item 7 | Days 13-14 | full validation, claim reconciliation, Sprint 102 handoff |

## Landing Order

1. Scope and artifact setup.
2. Public storage and solver entry audits.
3. Compressed-first API design and implementation boundary freeze.
4. First bounded constructor/import implementation batch.
5. Post-batch audit and remaining-scope rerank.
6. Lifecycle and ownership design plus focused follow-through.
7. Compatibility-path docs and example updates.
8. Regression proof expansion.
9. Full validation, product-model reconciliation, and closeout.

This order is deliberate. It prevents Sprint 101 from changing public API,
tests, or public docs before the current product-model cost and compatibility
risk are understood.

## Day-Level Ownership

| day | title | owned scope |
|---:|---|---|
| 1 | Scope Baseline | workstreams, inputs, artifact structure, validation expectations |
| 2 | Storage Audit | public construction/import/export/mutation/publication audit |
| 3 | Solver Entry Audit | solver entry and compressed-first adoption cost audit |
| 4 | API Design | selected CSR/CSC-front-door API design and ownership rules |
| 5 | Boundary Freeze | file-level implementation, test, docs, and validation boundary |
| 6 | Import Batch 1 | first bounded constructor/import implementation batch |
| 7 | Post-Batch Audit | reconcile landed work and rerank remaining candidates |
| 8 | Lifecycle Design | ownership, lifetime, mutation, and repeated-run rule design |
| 9 | Lifecycle Batch | focused lifecycle/ownership/error follow-through |
| 10 | Compatibility Design | compressed-first narrative and mutable-shell wording rules |
| 11 | Docs Batch | public docs and examples follow-through |
| 12 | Regression Proof | focused tests and proof gap closure |
| 13 | Validation | required quality checks and product-model reconciliation |
| 14 | Closeout | artifact index, Sprint 102 handoff, residual queue |

## Initial Risk Register

| risk | why it matters | Day 1 handling |
|---|---|---|
| claiming full shell replacement | Sprint 100 explicitly preserves mutable matrix-shell compatibility | keep full replacement as a non-goal and require compatibility wording |
| designing APIs before audit | compressed-first costs may be docs, ownership, or solver-entry issues rather than new API gaps | require Days 2-3 audits before Day 4 API design |
| implementation without validation scope | constructor/import changes can affect ownership, mutation, and solver behavior | freeze file-level and validation boundaries on Day 5 |
| docs outrunning code | public docs could imply product maturity before behavior is implemented and tested | schedule docs batch after implementation and lifecycle proof |
| CMake/Make/source-list drift | Sprint 101 may touch source, tests, examples, or headers | require source-list and full C quality checks when relevant |
| promotion drift | compressed-first remains a candidate claim until implementation, tests, docs, and validation exist | use Sprint 100 claim map and handoff package as guardrails |

## Validation Expectations

### Docs-Only Days

For days that only change planning or public documentation:

- run `git diff --check`;
- run a trailing-whitespace scan on touched documentation paths;
- do not run the C quality chain unless `.c` or `.h` files changed.

### Code or Header Touch Days

If any `.c` or `.h` file changes:

```sh
make format && make lint && make test
```

Focused tests or commands may be added before the full chain, but they do not
replace the full required chain.

### Test or Source-List Touch Days

If new tests or source files are added:

- confirm Makefile and CMake registration expectations;
- run source-list checks when library sources change;
- check Make/CMake parity if the test surface changes.

### Example or Public Docs Touch Days

If examples or user-facing docs change:

- run documentation hygiene;
- run example build or focused example commands when examples change;
- confirm wording stays inside Sprint 100 claim boundaries.

## Day 1 Exit Criteria

- Sprint 101 work is bounded before audit or implementation begins.
- Every Sprint 101 project-plan item has day-level ownership.
- Sprint 100 claim boundaries are visible in working notes and artifacts.
- Day 2 can start the public storage surface audit without redefining Sprint
  101 scope.
