# Sprint 74 Retrospective

**Sprint:** 74 — Capability Surface Modernization Phase 1  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 74 scope, capability hotspot map, and validation baseline were
      fixed before implementation work began
- [x] the strongest live capability ceilings were re-ranked from the repo
      instead of treated as one generic modernization bucket
- [x] the first landing stayed bounded to the width contract and did not widen
      into repo-wide 64-bit conversion, scalar genericity, or algorithm-family
      expansion
- [x] the width lane now has one explicit compile-time owner through
      `SPARSE_IDX_BITS`, `SPARSE_PRIDX`, `SPARSE_SCNIDX`, and
      `sparse_idx_bits()`
- [x] the matrix shell now uses the checked `idx_t`/`size_t` bridge more
      consistently on the highest-value touched paths
- [x] the strongest public real-only callback/result seams now point at one
      explicit scalar owner through `sparse_scalar_t`, `SPARSE_SCALAR_BITS`,
      and `sparse_scalar_bits()`
- [x] focused proof landed in the right owners:
  - `tests/test_sparse_matrix.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- [x] maintained public/policy wording now states the narrower Sprint 74
      capability interpretation directly
- [x] proof-owner alignment was closed without redundant regression work
- [x] the full Sprint 74 branch passed the standard code-day gate, the
      strongest reviewed baseline, and the focused width/scalar, example,
      benchmark, and install follow-ons
- [x] Sprint 74 closed with one explicit first-phase capability package and a
      ranked Sprint 75 carry-forward queue

## What Went Well

1. **Sprint 74 moved the strongest real capability seams instead of only restating them.**
   The branch landed substantive width/scalar ownership work in:
   - `include/sparse_types.h`
   - `src/sparse_types.c`
   - `src/sparse_alloc_internal.h`
   - `src/sparse_alloc_internal.c`
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   and tied that work to focused proof in:
   - `tests/test_sparse_matrix.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`

2. **The first capability lane stayed properly bounded.**
   Sprint 74 did not collapse into:
   - a repo-wide `int64_t` conversion campaign
   - broad scalar genericity
   - fake complex-readiness
   - unsymmetric eigensolver expansion
   - widened reviewed/install/platform claims
   That kept the work aligned with the Sprint 70 and Sprint 74 capability
   fences.

3. **The width contract is materially clearer now.**
   The Day 6 landing replaced a hand-edited-width feel with one deliberate
   compile-time seam:
   - `SPARSE_IDX_BITS`
   - `SPARSE_PRIDX`
   - `SPARSE_SCNIDX`
   - `sparse_idx_bits()`
   and made the checked `idx_t`/`size_t` bridge the clearer owner for
   allocation, byte sizing, memory-usage accounting, and Matrix Market
   formatting on the touched matrix-shell paths.

4. **The strongest public real-only seams now have one explicit owner.**
   Day 9 did not pretend to solve generic scalar support for the whole repo.
   It did the narrower useful thing:
   - established `sparse_scalar_t` as the public dense-scalar owner on the
     strongest touched callback/result seams
   - routed iterative public callback/result and dense-buffer contracts through
     that owner
   - routed eigensolver public callback/result seams through that owner
   - proved the public alias in the right focused owners

5. **Docs and proof ownership stayed aligned with the landed capability work.**
   Sprint 74 followed through in:
   - `README.md`
   - `docs/maintainer_guide.md`
   without reopening:
   - `INSTALL.md`
   - examples
   - public headers that were already truthful
   - broader packaging/platform stories

6. **The validated close state is strong.**
   Sprint 74 ended with:
   - `make format` passed
   - `make lint` passed
   - `make test` passed
   - `make quality-review-full` passed
   - reviewed CMake parity still exact at `53`
   - Makefile/CMake parity still `53 vs 53`
   - reviewed CMake `ctest` still `53 / 53`
   - focused width/scalar proof owners revalidated explicitly
   - representative benchmarks and install/package regressions still clean

## What Didn't Go Well

1. **Sprint 74 only opens the capability path; it does not finish it.**
   The width seam and the strongest public scalar alias seam are now clearer,
   but the branch does not yet deliver:
   - broad scalar-type widening
   - complex support
   - wider algorithm-family breadth
   - a repo-wide 64-bit-ready implementation story

2. **The strongest scalar lane remains intentionally narrow.**
   That is the correct Sprint 74 outcome, but it means Epic 7 still carries
   real deferred work around the deeper real-only contracts in:
   - iterative implementations
   - eigensolver implementations
   - later SVD and broader numeric surfaces

3. **The width contract is clearer than the full width reality.**
   Sprint 74 made the width seam explicit and more coherent on the touched
   path, but it did not convert the rest of the implementation into a fully
   widened-index product.

4. **Runtime asymmetry in the reviewed suite remains visible.**
   The full reviewed path passed, but `test_reorder_nd` still dominated the
   reviewed CMake time even though Sprint 74 itself was not a reorder sprint.
   That remains operational friction for later proof-heavy work.

5. **The branch depended on disciplined non-moves.**
   Sprint 74’s success required not reopening install/package wording, not
   widening the public product claim, and not treating `sparse_scalar_t` as
   proof of generic or complex support. That discipline held, but later
   capability work still needs to preserve it.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 74 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `372.56 sec` |
| reviewed `test_reorder_nd` time | `259.98 sec` |
| install regression | `11 / 11` |
| CMake install regression | `13 / 13` |

### Sprint 74 artifact package

| Metric | Sprint 74 close state |
|---|---:|
| total artifact files under `SPRINT_74/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `6` |
| review/closeout artifacts | `3` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-capability-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-rerun-recheck.md`
  - `day3-capability-ceiling-audit.md`
  - `day4-first-capability-boundary.md`
  - `day7-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-index-scalar-architecture-design.md`
  - `day6-index-width-integration-batch1.md`
  - `day8-scalar-surface-preparation-design.md`
  - `day9-scalar-surface-preparation-batch.md`
  - `day10-docs-packaging-test-alignment-design.md`
  - `day11-docs-packaging-test-alignment-batch.md`
- review/closeout artifacts:
  - `day12-regression-coverage-and-safety-alignment.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed capability package

| Metric | Sprint 74 close state |
|---|---:|
| public headers touched in landed package | `4` |
| implementation `.c` files touched in landed package | `3` |
| internal helper headers touched | `1` |
| focused proof-owner tests touched | `3` |
| maintained public/policy docs touched | `2` |
| representative benchmark families revalidated | `4` |

Notes:

- public headers touched:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- implementation `.c` files touched:
  - `src/sparse_types.c`
  - `src/sparse_alloc_internal.c`
  - `src/sparse_matrix.c`
- internal helper headers touched:
  - `src/sparse_alloc_internal.h`
- focused proof-owner tests touched:
  - `tests/test_sparse_matrix.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- maintained public/policy docs touched:
  - `README.md`
  - `docs/maintainer_guide.md`
- representative benchmark families revalidated:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`

## Residual Deferred Debt

Sprint 74 deliberately stopped after the first bounded capability phase. The
main open work it intentionally hands forward is:

- scalar-breadth modernization beyond the landed public seam
- later algorithm-family widening only where the capability contract and proof
  justify it
- backend/performance maturity only where benchmark-governed ownership seams
  still justify the proof cost
- later permanent-surface cleanup only after the higher-value capability lanes
  move

Still consciously constrained rather than silently “solved”:

- no repo-wide 64-bit conversion claim
- no repo-wide scalar genericity claim
- no fake complex-readiness or broader precision-product claim
- no unsymmetric eigensolver expansion
- no widened reviewed/install/platform claim

Not carried forward as unresolved Sprint 74 debt:

- the capability ceiling rerank
- the Day 6 index-width integration batch
- the Day 9 scalar-surface preparation batch
- the bounded README/maintainer-guide follow-through
- the proof-owner alignment pass
- the full Day 13 validation sweep
- the Day 14 closeout and ranked Sprint 75 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-capability-baseline.md](./artifacts/day1-scope-and-capability-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-rerun-recheck.md](./artifacts/day2-validation-baseline-and-rerun-recheck.md)
- [day3-capability-ceiling-audit.md](./artifacts/day3-capability-ceiling-audit.md)
- [day4-first-capability-boundary.md](./artifacts/day4-first-capability-boundary.md)
- [day5-index-scalar-architecture-design.md](./artifacts/day5-index-scalar-architecture-design.md)
- [day6-index-width-integration-batch1.md](./artifacts/day6-index-width-integration-batch1.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-scalar-surface-preparation-design.md](./artifacts/day8-scalar-surface-preparation-design.md)
- [day9-scalar-surface-preparation-batch.md](./artifacts/day9-scalar-surface-preparation-batch.md)
- [day10-docs-packaging-test-alignment-design.md](./artifacts/day10-docs-packaging-test-alignment-design.md)
- [day11-docs-packaging-test-alignment-batch.md](./artifacts/day11-docs-packaging-test-alignment-batch.md)
- [day12-regression-coverage-and-safety-alignment.md](./artifacts/day12-regression-coverage-and-safety-alignment.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom-Line Closeout

Sprint 74 succeeded because it turned the strongest immediate capability
ceilings into explicit, validated seams without widening into fake breadth.
The branch now has one real compile-time width contract, one clearer public
scalar owner for the strongest touched real-only seams, and one truthful docs
and proof interpretation of what that first capability phase actually means.
