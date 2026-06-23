# Sprint 85 Retrospective

**Sprint:** 85 — Large-Source Maintainability Phase 5  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 85 fixed the maintainability baseline, proof split, and
      implementation-day validation contract before landing hotspot cleanup
- [x] the strongest live hotspot contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 84
- [x] Sprint 85 fixed one explicit first implementation fence centered on:
  - `src/sparse_iterative.c`
- [x] Sprint 85 landed one bounded iterative-source cleanup batch:
  - one repeated frontend / trivial-case helper seam now owns result reset,
    converged marking, and trivial-system handling inside the iterative owner
  - the first cleanup stayed source-owned and did not force proof-owner churn
- [x] Sprint 85 landed one bounded direct-family hotspot batch:
  - the dense LDL^T primitive and backend-selection seam moved out of the
    Cholesky CSC hotspot and into the LDL^T CSC owner
  - the mixed family-ownership contradiction inside `src/sparse_chol_csc.c`
    was reduced without widening into generic family refactoring
- [x] Sprint 85 landed one bounded giant-test architecture batch:
  - `tests/test_chol_csc.c` now uses more local runner groups for early
    coverage families
  - the long flat `main()` registration block was reduced without
    redistributing proof ownership
- [x] Sprint 85 used bounded follow-through correctly:
  - `tests/test_iterative.c` remained the retained iterative proof owner
  - `tests/test_chol_csc.c` remained the retained direct-family and
    giant-test proof owner
  - `tests/test_ldlt.c` took only the minimal forced internal-header
    follow-through after the Day 9 owner move
  - `docs/maintainer_guide.md` and `README.md` were correctly not widened
    where the sprint did not change their maintained contract
- [x] Sprint 85 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 85 closed with one explicit Sprint 86-first handoff queue instead
      of another generic Epic 8 maintainability summary

## What Went Well

1. **Sprint 85 chose the right first cleanup lane.**
   The sprint did not start by splitting giant tests or reopening Sprint 84
   proof work. It first reduced repeated frontend and trivial-case ownership
   concentration inside `src/sparse_iterative.c`.

2. **The decomposition fence stayed disciplined.**
   The first source cleanup remained:
   - source-owned
   - bounded to one helper seam
   - behavior-preserving
   - free of proof-owner or docs churn

3. **The direct-family cleanup moved the right seam.**
   Day 9 did not perform a generic Cholesky reorganization. It removed the
   dense LDL^T / backend-selection block that was clearly owned by the LDL^T
   CSC family and re-homed it under the correct owner.

4. **The giant-test cleanup stayed local to the real proof owner.**
   Day 11 reduced registration concentration in `tests/test_chol_csc.c`
   without redistributing test logic across files or introducing a new
   proof-owner model.

5. **Support-surface discipline remained strong.**
   Sprint 85 correctly avoided unnecessary churn in:
   - `docs/maintainer_guide.md`
   - `README.md`
   - install/export proof
   - benchmark ownership
   - package/runtime claims

6. **Sprint 86 now starts from cleaner ownership, not just smaller files.**
   The sprint reduced the strongest mixed-responsibility seams enough that the
   next Epic 8 move can be runtime and scalability work instead of another
   decomposition-order argument first.

## What Didn't Go Well

1. **Sprint 85 reduced hotspot cost, not hotspot count everywhere.**
   That was the correct bounded result, but it leaves real residual work:
   - `src/sparse_qr.c` remains a large source hotspot
   - `src/sparse_ldlt.c` remains a large source hotspot
   - `tests/test_qr.c` and `tests/test_integration.c` remain large proof
     owners

2. **The first iterative cleanup stayed intentionally narrow.**
   That kept the batch truthful, but it also means `src/sparse_iterative.c`
   was improved rather than broadly decomposed.

3. **The giant-test package was organization-first, not proof redistribution.**
   That is the right ownership decision, but it means Sprint 85 did not
   materially shrink the absolute size of `tests/test_chol_csc.c`.

4. **Support docs stayed mostly untouched because the sprint was truthful.**
   That is a strength, but it also means the visible sprint output is more in
   ownership quality than in broad support-surface narration.

5. **The reviewed runtime long pole remains large.**
   Sprint 85 closed from a strong measured baseline, but it did not reduce the
   operational drag from `test_reorder_nd`, which remains the dominant
   reviewed runtime anchor and the right Sprint 86 handoff target.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 85 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `404.15 sec` |
| reviewed `test_reorder_nd` time | `283.53 sec` |
| focused `test_iterative` follow-on | `80 / 80` |
| focused `test_chol_csc` follow-on | `151 / 151` |
| focused `test_integration` follow-on | `56 / 56` |
| focused `test_ldlt` follow-on | `87 / 87` |
| focused `test_qr` follow-on | `73 / 73` |

### Sprint 85 artifact package

| Metric | Sprint 85 close state |
|---|---:|
| total artifact files under `SPRINT_85/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-hotspot-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-hotspot-rerank-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-proof-docs-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-maintainability-boundary.md`
  - `day5-decomposition-ownership-architecture-design.md`
  - `day6-iterative-source-cleanup-batch.md`
  - `day8-direct-family-hotspot-design.md`
  - `day9-direct-family-hotspot-batch.md`
  - `day10-giant-test-architecture-design.md`
  - `day11-giant-test-architecture-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 85 close state |
|---|---:|
| implementation `src/` files touched | `3` |
| internal header files touched | `1` |
| public header files touched | `0` |
| proof-owner test files touched | `2` |
| helper script files touched | `0` |
| support docs requiring follow-through | `0` |

Notes:

- implementation `src/` files touched:
  - `src/sparse_iterative.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
- internal header files touched:
  - `src/sparse_ldlt_csc_internal.h`
- proof-owner test files touched:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- support surfaces intentionally left untouched after recheck:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Residual Deferred Debt

Sprint 85 deliberately stopped after the highest-value maintainability package.
The main open work it hands forward is:

- reviewed runtime convergence and reordering-scalability work after the
  reduced hotspot package
- later bounded cleanup on adjacent large source hotspots such as
  `src/sparse_qr.c` and `src/sparse_ldlt.c` only where the refreshed hotspot
  map justifies more extraction
- later bounded cleanup on adjacent large proof-owner hotspots such as
  `tests/test_qr.c` and `tests/test_integration.c`
- later package/platform/runtime maturity only where touched mechanics justify
  broader claims

Still consciously constrained rather than silently “solved”:

- no repo-wide large-source decomposition claim
- no broad proof-owner redistribution across giant tests
- no benchmark or example drift into correctness ownership
- no package/install/export/runtime-package claim broadening
- no Sprint 86 runtime/scalability work pre-landed inside Sprint 85

Not carried forward as unresolved Sprint 85 debt:

- the baseline/proof-surface recheck
- the live hotspot rerank
- the bounded decomposition / ownership architecture contract
- the Day 6 iterative-source cleanup landing
- the Day 9 direct-family owner move
- the Day 11 giant-test registration-layout cleanup
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 86-first handoff queue

## Key Deliverables

1. **One bounded iterative-source cleanup landed on the highest-value first hotspot.**
   `src/sparse_iterative.c` now owns a clearer local helper seam for result
   reset, converged marking, and trivial-system handling instead of repeating
   that frontend boilerplate across multiple entry paths.

2. **One better-owned direct-family dense LDL^T seam landed under the correct family owner.**
   `src/sparse_ldlt_csc.c` and `src/sparse_ldlt_csc_internal.h` now own the
   dense LDL^T primitive and backend-selection seam that had previously been
   embedded inside the Cholesky CSC hotspot.

3. **One bounded giant-test registration cleanup landed without proof-owner churn.**
   `tests/test_chol_csc.c` now uses a more consistent local runner-group
   structure for early coverage families, reducing registration concentration
   while preserving one-file proof ownership.

4. **Sprint 85 closed from a measured maintainability baseline, not just from cleanup prose.**
   The branch ended with a full Day 13 validation sweep, focused reviewed
   proof-owner reruns, benchmark/reporting confirmation, and an explicit Day
   14 handoff queue for Sprint 86 and later Epic 8 lanes.

## Bottom Line

Sprint 85 succeeded because it stayed bounded where the repo most needed it.
It did not pretend to solve every remaining large source or giant test, but it
did remove the strongest live ownership contradictions around the iterative
frontend hotspot, the dense LDL^T block stranded inside the Cholesky CSC
owner, and the registration concentration inside the largest retained
direct-family proof owner. That is enough real maintainability movement to
make Sprint 86 the correct next Epic 8 step: reviewed runtime convergence and
reordering-scalability work on top of cleaner hotspot ownership instead of
another round of decomposition ambiguity first.
