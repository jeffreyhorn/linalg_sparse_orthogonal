# Sprint 100 Day 7 Residual Queue Conversion

## Purpose

Day 7 converts the Epic 9 residual queue into an Epic 10 claim and risk map.
This artifact is the bridge between inherited carry-forward work and the
Sprint 101-109 dependency model that Day 8 will produce.

## Classification Summary

| class | count | interpretation |
|---|---:|---|
| Epic 9 carry-forward items converted | `8` | all have Epic 10 dispositions |
| inherited non-claims reviewed | `10` | preserved, revisited, or kept out of scope |
| immediate Sprint 100 blockers found | `0` | no residual blocks Day 8 claim modeling |
| unsupported positive claims found | `0` | Day 6 disallowed-claim list remains preventive |

## Carry-Forward Conversion Table

| Epic 9 residual | Epic 10 disposition | candidate owner | risk | evidence required |
|---|---|---|---|---|
| broader LDLT CSC Matrix Market or indefinite corpus comparison | in scope as direct-solver oracle expansion candidate | Sprint 102 | high | fixture taxonomy, oracle behavior, tolerance model, runtime budget, focused LDLT CSC proof |
| iterative solver external comparison architecture | in scope as comparison architecture candidate | Sprint 103 | high | convergence semantics, restart/preconditioner boundaries, residual criteria, external or deterministic reference plan |
| eigensolver/LOBPCG external comparison architecture | in scope as comparison architecture candidate | Sprint 103 | high | eigenpair fixture taxonomy, tolerance/cluster policy, runtime cap, backend/non-claim wording |
| QR/SVD external comparison architecture | in scope as direct/SVD oracle expansion candidate | Sprint 102 and Sprint 103 | high | per-family reference owner, numerical tolerance rules, rank/conditioning fixtures, runtime budget |
| generated reorder/fill report target if repeated captures justify it | stretch candidate, not mandatory unless repeated captures justify ownership | Sprint 105 | medium | evidence of repeated manual use, stable fields, `nnz_L` primary fill field, local timing caveat |
| continued large-source extraction | in scope as maintainability requirement | Sprint 106 plus touched family sprints | high | family-local extraction boundary, source-list parity, focused validation, full C chain if code changes |
| continued giant-test extraction | in scope as maintainability requirement | Sprint 106 plus touched family sprints | high | test owner map, fixture/helper extraction plan, CMake/Make count parity, focused tests |
| lower-level chronology cleanup where useful | in scope only when touched or when it improves proof clarity | Sprint 105, Sprint 106, Sprint 107 | medium | avoid compatibility-breaking renames, preserve useful rationale, docs hygiene for docs-only work |

## Sprint Owner Draft

| sprint | residual ownership |
|---|---|
| Sprint 101 | compressed-first product-model work, but not full shell replacement |
| Sprint 102 | direct-solver oracle work: LDLT CSC, Cholesky, LU, QR, SVD candidates |
| Sprint 103 | iterative, eigensolver, LOBPCG, and SVD comparison architecture |
| Sprint 104 | backend/runtime residuals and performance evidence guardrails |
| Sprint 105 | reorder/fill reporting, graph/ND chronology cleanup, large-matrix evidence |
| Sprint 106 | large-source and giant-test extraction |
| Sprint 107 | solver-selection docs, user-facing claim clarity, chronology cleanup on docs/examples |
| Sprint 108 | package, ABI, platform tier, and Windows/macOS support decisions |
| Sprint 109 | final competitive calibration, unsupported-claim cleanup, residual queue refresh |

## Risk and Evidence Map

| risk level | residual types | required guardrail |
|---|---|---|
| high | external comparison architecture | one-page proof architecture before implementation; fixtures, oracle, tolerance, runtime, and claim language must be named |
| high | source/test extraction | family-local owner map before edits; source-list parity and focused tests required |
| high | package/ABI/platform support widening | explicit support-tier decision before wording changes imply new maturity |
| medium | reorder/fill generated reporting | add target only after repeated use justifies stable generated output |
| medium | chronology cleanup | improve clarity without losing regression rationale or breaking compatibility names |
| medium | benchmark sentinels | keep thresholds local, justified, and non-portable |
| low | docs-only non-claim preservation | document with hygiene checks and no C quality chain unless code changes |

## Validation Expectations by Residual

| residual family | validation expectation |
|---|---|
| direct external comparison | focused direct-family tests plus full C quality chain if `.c` or `.h` changes |
| iterative/eigensolver comparison | focused solver-family tests plus full C quality chain if `.c` or `.h` changes |
| QR/SVD comparison | focused QR/SVD tests plus full C quality chain if `.c` or `.h` changes |
| reorder/fill reporting | focused benchmark/report commands; keep timing language local |
| source extraction | source-list check, focused tests, Make/CMake parity, full C quality chain |
| giant-test extraction | focused tests, Make/CMake test-count parity, full C quality chain |
| chronology cleanup | docs hygiene if docs-only; focused tests if comments/renames touch code/test behavior |
| package/platform | install/export scripts, CMake consumer proof, workflow count checks if touched |

## Inherited Non-Claim Register

| inherited non-claim | Epic 10 disposition | owner |
|---|---|---|
| full compressed-first replacement of the linked-list shell | preserve as non-goal; Sprint 101 should make compressed-first workflows more central without claiming replacement | Sprint 101, Sprint 107 |
| broad complex support | preserve as non-goal unless a future epic replans scalar-family work | Sprint 109 |
| broad mixed-precision maturity | preserve as non-goal unless replanned | Sprint 109 |
| broad backend-neutral acceleration maturity | preserve; Sprint 104 may improve bounded backend/runtime contract without claiming vendor parity | Sprint 104 |
| shared-library-first package contract | revisit explicitly in Sprint 108; current truth remains static-first | Sprint 108 |
| dynamic ABI guarantee | revisit only if Sprint 108 creates shared-library/ABI proof | Sprint 108 |
| symmetric Linux/macOS/Windows reviewed parity | preserve as non-goal; publish tiered platform truth | Sprint 108 |
| Windows Makefile parity or install-validation lane | preserve unless Sprint 108 explicitly adds and validates a lane | Sprint 108 |
| portable timing superiority or universal reorder/fill superiority | preserve as non-goal; benchmark evidence remains local/bounded | Sprint 104, Sprint 105, Sprint 109 |
| every-solver-family external correctness comparison | preserve as non-goal for Epic 10 closeout unless all families earn proof; partial expansion is still valuable | Sprint 102, Sprint 103, Sprint 109 |

## In-Scope Versus Out-of-Scope Summary

### In Scope

- centralize compressed-first workflows without deleting compatibility shell
- expand selected direct solver oracle evidence
- design iterative/eigensolver/SVD comparison architectures
- improve backend/runtime observability and local evidence
- improve reorder/fill reporting only where repeated use justifies it
- reduce large source and giant-test ownership risk
- clean chronology where it improves current proof clarity
- clarify package/platform support tiers

### Out of Scope Unless Replanned

- GPU or distributed sparse solvers
- broad complex or mixed-precision product maturity
- universal vendor backend parity
- shared-library ABI guarantee without Sprint 108 proof
- package-manager ecosystem ownership
- fake symmetric platform parity
- broad portable timing superiority
- every-solver-family external correctness parity

## Day 7 Conclusion

Every Epic 9 residual now has an Epic 10 disposition. The queue is actionable
without becoming a mandate to claim universal state-of-the-art parity. Day 8
should convert these dispositions into a dependency-aware claim model for
Sprints 101-109.

