# Sprint 84 Retrospective

**Sprint:** 84 — Numerical Assurance & Differential Testing Phase 2  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 84 fixed the assurance baseline, proof split, and
      implementation-day validation contract before landing proof work
- [x] the strongest live assurance contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 83
- [x] Sprint 84 fixed one explicit first implementation fence centered on:
  - `tests/test_chol_csc.c`
- [x] Sprint 84 landed one bounded maintained external differential batch:
  - the direct-family SPD Cholesky CSC lane now has maintained external
    comparison proof on `nos4` and `bcsstk04`
  - the maintained external lane stayed test-owned and family-local
- [x] Sprint 84 landed one bounded deterministic seeded-property batch:
  - the retained large-`n` direct-family lifecycle owner now proves reorder
    agreement, repeated-solve invariance, same-pattern refactor agreement,
    and residual smallness on CSC-backed Cholesky and LDL^T lanes
- [x] Sprint 84 landed one bounded failure-path numerical proof batch:
  - the shared public lifecycle owner now proves preserved-old-factor solve
    behavior and successful later retry-after-failure on linked-list
    Cholesky, CSC Cholesky, and AMD LDL^T lanes
- [x] Sprint 84 used bounded follow-through correctly:
  - `tests/test_chol_csc.c` became the maintained external differential owner
  - `tests/test_fuzz.c` became the bounded seeded-property expansion owner
  - `tests/test_integration.c` became the bounded failure-path lifecycle owner
  - `docs/maintainer_guide.md` was reconciled with the landed assurance split
  - broader README, package, install/export, runtime-package, and
    reviewed-Windows claims were correctly not widened where the tree already
    stayed truthful
- [x] Sprint 84 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 84 closed with one explicit Sprint 85-first handoff queue instead
      of another generic Epic 8 summary

## What Went Well

1. **Sprint 84 chose the right first assurance lane.**
   The sprint did not start by broadening external proof across every solver
   family. It first landed one bounded maintained external differential seam
   on the highest-value direct-family SPD Cholesky CSC path.

2. **The oracle fence stayed disciplined.**
   The maintained external lane stayed:
   - test-owned
   - fixture-backed
   - pure-stdlib from the repo side
   - family-local rather than repo-wide in its claims

3. **Seeded-property widening followed the correct owner.**
   After the Day 6 external lane existed, the strongest residual contradiction
   was deterministic lifecycle/property depth. Day 9 widened that in
   `tests/test_fuzz.c` instead of reopening direct-family implementation code
   or scattering the property model across unrelated owners.

4. **Failure-path proof moved to the correct shared lifecycle surface.**
   The strongest retry-after-failure contradiction lived in the public
   lifecycle owner. Day 11 resolved that in `tests/test_integration.c`
   without forcing family-local churn in `tests/test_chol_csc.c` or
   `tests/test_ldlt.c`.

5. **Support-surface discipline remained strong.**
   Sprint 84 moved `docs/maintainer_guide.md` when the proof-owner story
   actually changed, but it correctly avoided over-claiming in:
   - `README.md`
   - install/export proof
   - package/runtime support claims
   - reviewed-Windows coverage

6. **Sprint 85 now has a cleaner starting point.**
   Sprint 84 reduced the strongest assurance contradictions enough that the
   next Epic 8 move is maintainability and hotspot cleanup, not another round
   of proof-ownership clarification first.

## What Didn't Go Well

1. **Sprint 84 widened assurance depth, not assurance breadth everywhere.**
   That was the correct bounded result, but it leaves real residual work:
   - no maintained external differential adoption on iterative solvers
   - no maintained external differential adoption on eigensolvers
   - no repo-wide external-proof claim

2. **The maintained external differential lane stayed intentionally narrow.**
   That kept the sprint truthful, but it also means the external-oracle story
   still centers on one direct-family lane rather than a wider solver-family
   matrix.

3. **The proof package was intentionally test-owner-heavy.**
   That is a strength for correctness ownership, but it also means Sprint 84
   did not reduce the giant-test and hotspot reasoning cost that Sprint 85 now
   inherits.

4. **Windows reviewed coverage still does not include `test_fuzz`.**
   Sprint 84 correctly refused to over-claim cross-platform reviewed evidence,
   but that also means part of the widened seeded-property surface remains
   outside the reviewed Windows subset.

5. **The reviewed runtime long pole remains large.**
   Sprint 84 closed from a strong measured baseline, but it did not reduce the
   operational drag from `test_reorder_nd`, which remained the dominant
   reviewed runtime anchor.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 84 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `477.50 sec` |
| reviewed `test_reorder_nd` time | `344.21 sec` |
| focused `test_chol_csc` follow-on | `151 / 151` |
| focused `test_ldlt` follow-on | `87 / 87` |
| focused `test_fuzz` follow-on | `28 / 28` |
| focused `test_integration` follow-on | `56 / 56` |
| focused `test_iterative` follow-on | `80 / 80` |
| focused `test_eigs` follow-on | `31 / 31` |

### Sprint 84 artifact package

| Metric | Sprint 84 close state |
|---|---:|
| total artifact files under `SPRINT_84/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-assurance-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-differential-proof-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-final-proof-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-assurance-boundary.md`
  - `day5-oracle-property-failure-path-architecture-design.md`
  - `day6-direct-family-differential-batch.md`
  - `day8-seeded-property-expansion-design.md`
  - `day9-seeded-property-expansion-batch.md`
  - `day10-failure-path-numerical-proof-design.md`
  - `day11-failure-path-numerical-proof-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 84 close state |
|---|---:|
| implementation `src/` files touched | `0` |
| public header files touched | `0` |
| proof-owner test files touched | `3` |
| helper script files touched | `1` |
| benchmark source files touched | `0` |
| support docs requiring follow-through | `1` |

Notes:

- proof-owner test files touched:
  - `tests/test_chol_csc.c`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
- helper script files touched:
  - `tests/chol_external_dense_reference.py`
- support surface intentionally moved:
  - `docs/maintainer_guide.md`
- support surfaces intentionally left untouched after recheck:
  - `README.md`
  - `.github/workflows/windows-ci.yml`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Residual Deferred Debt

Sprint 84 deliberately stopped after the highest-value assurance package. The
main open work it hands forward is:

- large-source and giant-test maintainability work after the widened
  assurance surface
- later iterative/eigensolver maintained external differential adoption only
  where bounded evidence justifies widening
- later reviewed runtime convergence and reordering-scalability work
- later package/platform/runtime maturity only where touched mechanics justify
  broader claims

Still consciously constrained rather than silently “solved”:

- no repo-wide maintained external-differential claim
- no iterative maintained external differential lane
- no eigensolver maintained external differential lane
- no widened reviewed-Windows seeded-property claim
- no package/install/export/runtime-package claim broadening

Not carried forward as unresolved Sprint 84 debt:

- the baseline/proof-surface recheck
- the live differential-proof rerank
- the bounded oracle/property/failure-path architecture contract
- the Day 6 direct-family external differential landing
- the Day 9 deterministic seeded-property expansion
- the Day 11 failure-path lifecycle proof widening
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 85-first handoff queue

## Key Deliverables

1. **One maintained external differential lane landed on the highest-value direct-family SPD path.**
   `tests/test_chol_csc.c` now owns bounded maintained external differential
   proof for forced CSC Cholesky on `nos4` and AMD-reordered forced CSC
   Cholesky on `bcsstk04`.

2. **One bounded large-`n` deterministic seeded-property widening landed.**
   `tests/test_fuzz.c` now proves reorder agreement, repeated-solve
   invariance, same-pattern refactor agreement, and residual smallness on
   retained large-`n` CSC-backed Cholesky and LDL^T lifecycle flows.

3. **One stronger shared failure-path lifecycle proof owner landed.**
   `tests/test_integration.c` now proves successful later retry-after-failure
   behavior on the same public `analysis` / `factors` objects for linked-list
   Cholesky, CSC Cholesky, and AMD LDL^T.

4. **One truthful maintainer-policy assurance update landed.**
   `docs/maintainer_guide.md` now reflects the bounded direct-family external
   differential owner, the seeded-property owner, and the failure-path
   lifecycle owner without implying repo-wide oracle maturity.

5. **Sprint 84 closed from a measured assurance baseline, not just from test-design prose.**
   The branch ended with a full Day 13 validation sweep, focused reviewed
   proof-owner reruns, benchmark/reporting confirmation, and an explicit Day
   14 handoff queue for Sprint 85 and later Epic 8 lanes.

## Bottom Line

Sprint 84 succeeded because it stayed bounded where the repo most needed it.
It did not pretend that one external oracle lane solved assurance everywhere,
but it did remove the strongest live contradictions around direct-family
maintained differential proof, large-`n` deterministic lifecycle properties,
and retry-after-failure lifecycle guarantees. That is enough real
assurance-surface movement to make Sprint 85 the correct next Epic 8 step:
maintainability and hotspot reduction on top of a clearer, better-proved
surface, instead of another round of proof-ownership ambiguity first.
