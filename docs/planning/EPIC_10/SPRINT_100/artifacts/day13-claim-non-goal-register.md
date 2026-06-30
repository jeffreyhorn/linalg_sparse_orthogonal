# Sprint 100 Day 13 Claim and Non-Goal Register

## Purpose

This register is the compact claim-state companion to the Day 13 handoff
package. It exists so later sprint closeouts can quickly decide whether a claim
is already earned, still candidate, blocked/stretch, or a non-goal.

## State Definitions

| state | meaning |
|---|---|
| earned | supported by live implementation, docs/artifacts, and validation evidence |
| candidate | planned or partially supported, but still needs implementation and evidence |
| blocked/stretch | cannot be claimed without an explicit decision or proof expansion |
| non-goal | outside Epic 10 unless replanned |

## Earned

| claim | evidence |
|---|---|
| strongest local reviewed baseline is clean at Sprint 100 Day 2 | `make quality-review-full`; CMake tests `54`; Make/CMake parity `54` vs `54`; CTest `54 / 54` |
| static-first package truth is maintained | Day 3 package baseline; inherited Make install `14 / 14`; CMake install `16 / 16`, `0` skips |
| tiered platform support is the current support model | Day 3 platform draft; Day 11 platform template; Day 12 public audit |
| threshold-free canonical benchmark report exists | Day 5 benchmark baseline; Day 10 benchmark pilot |
| selected Cholesky CSC and LDLT CSC external dense-reference lanes exist | Day 5 comparison baseline; Day 9 Cholesky pilot |
| public docs avoid broad unsupported state-of-the-art claims | Day 12 public claim audit |

## Candidate

| claim | owner | evidence required |
|---|---|---|
| compressed-first product model is primary | Sprint 101 | implementation, lifecycle tests, examples/docs, compatibility-shell wording |
| direct solver oracle evidence is deeper | Sprint 102 | fixture taxonomy, oracle helpers, focused direct-family validation |
| iterative/eigensolver/SVD comparison architecture exists | Sprint 103 | convergence/eigen/rank fixtures, residual criteria, validation commands |
| backend/runtime contract is clearer and observable | Sprint 104 | descriptor/observability tests, benchmark fields, fallback docs |
| local performance sentinels are decision-grade | Sprint 104 | filled sentinel template, baseline, machine-class caveats |
| reorder/fill and graph evidence is clearer | Sprint 105 | named fixtures, fill metric contract, local timing caveats |
| large source and giant-test ownership risk is reduced | Sprint 106 | before/after metrics, extraction artifacts, parity checks, validation |
| user-facing solver-selection path is clearer | Sprint 107 | docs/examples aligned with earned technical evidence |
| platform support tiers are explicit | Sprint 108 | support-tier table, expected counts, staged exclusions |
| final Epic 10 competitive calibration is truthful | Sprint 109 | final validation, claim audit, unsupported-claim cleanup, residual queue |

## Blocked or Stretch

| claim | owner | block |
|---|---|---|
| broader LDLT CSC Matrix Market or indefinite corpus parity | Sprint 102 stretch | fixture taxonomy, runtime budget, oracle behavior not yet defined |
| shared-library or ABI support | Sprint 108 stretch | no shared-library install/export/runtime-loader proof exists |
| Windows Makefile parity | Sprint 108 stretch | Windows reviewed lane is CMake-first only |
| Windows install-validation parity | Sprint 108 stretch | no reviewed Windows install/export script lane exists |
| package-manager ecosystem integration | future work | no recipe ownership or validation exists |

## Non-Goals Unless Replanned

- broad state-of-the-art replacement claim;
- SuiteSparse/PETSc/Trilinos parity or replacement;
- GPU sparse kernels;
- distributed-memory sparse solvers;
- universal vendor backend parity;
- broad complex-number maturity;
- broad mixed-precision maturity;
- full replacement of the mutable linked-list shell;
- symmetric Linux/macOS/Windows reviewed parity;
- portable timing superiority;
- every-solver-family external correctness parity.

## Promotion Rule

A candidate or blocked claim can move to earned only if the owning sprint
records:

- implementation or documentation changes;
- relevant filled Sprint 100 evidence template;
- validation command and result;
- unsupported or skipped cases;
- non-claims that remain after the work.
