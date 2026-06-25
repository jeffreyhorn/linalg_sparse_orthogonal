# Sprint 91 Retrospective

**Sprint:** 91 — Compressed-First Product Convergence Phase 3  
**Duration:** 14 days  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 91 fixed the implementation-day validation and maintained-surface
      contract before compressed-first product work widened
- [x] the sprint reduced the broad linked-list-first problem to one ranked
      live shell-cost map rather than reopening generic direct-family cleanup
- [x] Sprint 91 froze one explicit first implementation fence centered on:
  - compressed-first construction/import entry
- [x] Sprint 91 froze one explicit compressed-first product contract:
  - linked-list shell retained as mutable compatibility owner
  - CSR/CSC-backed construction/import promoted to first-class public entry
    paths
  - broader publication/lifecycle reinterpretation kept bounded
- [x] Sprint 91 landed one bounded construction/import batch:
  - `sparse_create_from_csr(...)`
  - `sparse_create_from_csc(...)`
  - compatibility `sparse_from_*` wrappers preserved
- [x] Sprint 91 landed one bounded public-story batch:
  - README now teaches compressed-first direct entry as a real peer lane
  - one-shot vs repeated-run direct lifecycle reading is clearer
- [x] Sprint 91 landed one bounded public-workflow proof batch:
  - constructor-built CSR path proven into one-shot LU workflow
  - constructor-built CSC path proven into repeated-run Cholesky lifecycle
- [x] Sprint 91 froze one final proof-owner map before close:
  - `test_csr.c`
  - `test_integration.c`
  - `README.md`
  - retained adjacent direct-family proof owners
- [x] Sprint 91 ran the full final validation sweep and closed from one
      explicit validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - focused touched proof owners and representative examples
  - `make bench-canonical-report`
- [x] Sprint 91 closed with one explicit Sprint 92-first handoff queue instead
      of reopening the compressed-first seam

## What Went Well

1. **Sprint 91 chose the right first product seam.**
   The sprint did not try to remove the linked-list shell wholesale. It
   targeted the highest-value public cost first: compressed inputs already in
   CSR/CSC form now have first-class constructor-style entry paths.

2. **The bounded product contract stayed coherent across code, docs, and proof.**
   Day 6 made the constructor-style entry real, Day 9 taught it in the README,
   and Day 11 proved the public direct-workflow behavior. That kept the sprint
   from ending as an API-only or docs-only half-step.

3. **Compatibility stayed intact while the conceptual center shifted.**
   The legacy `sparse_from_*` APIs were preserved as compatibility wrappers,
   so Sprint 91 improved the public product reading without forcing broad
   caller churn.

4. **The proof-owner split stayed disciplined.**
   Constructor validity remained owned by `test_csr.c`, while the missing
   public direct-workflow behavior landed in `test_integration.c`. That is a
   cleaner ownership outcome than scattering more constructor logic across
   unrelated proof surfaces.

5. **The sprint closed from a strong maintained baseline.**
   Sprint 91 did not stop at `make test`. It also closed from the reviewed
   baseline with exact Makefile/CMake parity, focused touched-proof reruns,
   representative examples, and canonical benchmark/reporting follow-through.

6. **The Sprint 92 handoff is materially clearer now.**
   Epic 9 no longer needs to reopen compressed-first public product
   convergence immediately. Sprint 92 can start from the backend-maturity lane
   instead of relitigating the Sprint 91 product/lifecycle seam.

## What Didn't Go Well

1. **The sprint inherited a slightly awkward opening state.**
   Sprint 91’s durable local record begins with the Day 2 validation baseline
   rather than a separate Day 1 artifact package. That did not block the work,
   but it makes the sprint-local paper trail less uniform than the recent
   closes.

2. **The linked-list shell still remains a real public compatibility owner.**
   Sprint 91 improved the public reading, but it did not remove the shell as a
   mutable construction and one-shot compatibility surface. That remaining
   duality is smaller now, not gone.

3. **The sprint’s public-surface widening stayed deliberately narrow.**
   README moved, but broader header, maintainer, example, and tutorial follow-
   through was intentionally not reopened once the public story became
   truthful enough. That was the right bounded call, but it means some product
   narration still remains distributed.

4. **The strongest reviewed runtime long pole is still outside this sprint’s adopted lane.**
   Sprint 91 closed cleanly, but the reviewed path still retains a substantial
   `test_reorder_nd` tail. The sprint correctly did not dilute its scope by
   reopening that lane, but the long pole remains visible at close.

5. **The sprint improved compressed-first entry more than full product-model convergence.**
   Entry-path and lifecycle clarity materially improved, but publication/export
   and broader shell-to-compute interpretation work remain later bounded Epic 9
   work if refreshed evidence justifies more movement.

## Final Metrics

### Validation and close anchors

| Metric | Sprint 91 close state |
|---|---:|
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `340.76 sec` |
| reviewed `test_reorder_nd` time | `203.14 sec` |
| canonical reporting follow-through | `make bench-canonical-report` passed |

### Focused Sprint 91 proof and example anchors

| Metric | Sprint 91 close state |
|---|---:|
| constructor proof owner | `test_csr` |
| `test_csr` result | `13 / 13` |
| public lifecycle proof owner | `test_integration` |
| `test_integration` result | `58 / 58` |
| retained adjacent direct-family owner | `test_chol_csc` |
| `test_chol_csc` result | `151 / 151` |
| retained adjacent direct-family owner | `test_ldlt_csc` |
| `test_ldlt_csc` result | `96 / 96` |
| `example_analysis` residual | `4.44e-16` |
| `example_basic_solve` residual | `0.00e+00` |

### Sprint 91 artifact package

| Metric | Sprint 91 close state |
|---|---:|
| total artifact files under `SPRINT_91/artifacts/` | `13` |
| validation/alignment/closeout artifacts | `4` |
| audit/design/fence artifacts | `6` |
| implementation/follow-through artifacts | `3` |

Notes:

- validation/alignment/closeout artifacts:
  - `day2-validation-baseline-and-maintained-surface-recheck.md`
  - `day12-final-alignment-and-validation-queue.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`
- audit/design/fence artifacts:
  - `day3-remaining-linked-list-first-cost-audit.md`
  - `day4-first-implementation-boundary.md`
  - `day5-compressed-first-architecture-design.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day8-publication-and-lifecycle-design.md`
  - `day10-proof-follow-through-design.md`
- implementation/follow-through artifacts:
  - `day6-construction-import-batch.md`
  - `day9-publication-lifecycle-batch.md`
  - `day11-proof-follow-through-batch.md`

### Landed change class

| Metric | Sprint 91 close state |
|---|---:|
| public headers touched | `1` |
| source files touched | `1` |
| proof-owner C test files touched | `2` |
| public product docs touched | `1` |
| maintainer docs touched | `0` |
| workflow files touched | `0` |
| install/export proof scripts touched | `0` |

Notes:

- landed product/proof surfaces:
  - `include/sparse_csr.h`
  - `src/sparse_csr.c`
  - `tests/test_csr.c`
  - `tests/test_integration.c`
  - `README.md`
- no package/workflow/install-export surface had to move for Sprint 91 to land

## Residual Deferred Debt

Sprint 91 intentionally improved the public compressed-first reading without
pretending the full linked-list-first product model is now gone.

Most important carry-forward work:

- portable dense/backend maturity
- runtime/threading and reviewed-runtime convergence
- broader capability-envelope widening
- later public narrative/workflow coherence beyond the bounded Sprint 91
  README shift
- remaining large-source and proof-owner maintainability concentration
- broader build/package/workflow convergence

Still consciously constrained rather than silently "solved":

- no claim that the linked-list shell is no longer a public compatibility owner
- no claim that all direct workflows are now fully compressed-first by design
- no claim that publication/export semantics were comprehensively rewritten in
  this sprint
- no claim that runtime concentration or reorder scalability were fixed here

Not carried forward as unresolved Sprint 91 debt:

- constructor-style compressed public entry
- public README adoption-story alignment for the new entry paths
- public direct-workflow proof for constructor-built CSR/CSC inputs
- final proof-owner map freeze
- Day 13 validated close baseline

## Key Deliverables

1. **One real compressed-first public constructor seam landed.**
   Sprint 91 materially changed the product model by adding first-class public
   constructor-style CSR/CSC entry paths instead of leaving compressed inputs
   conceptually behind the linked-list shell.

2. **One clearer public workflow reading landed.**
   The README now teaches compressed-first one-shot direct entry as a real
   peer lane and positions the linked-list shell more clearly as a mutable
   compatibility owner.

3. **One missing public-lifecycle proof gap was closed.**
   Constructor-built CSR/CSC matrices are now explicitly proven entering the
   direct workflows that the public product story teaches.

4. **One exact validated Sprint 91 baseline landed.**
   The sprint closes from a full local validation sweep, reviewed parity, the
   focused touched proof owners, representative examples, and canonical
   reporting.

5. **One explicit Sprint 92-first handoff queue landed.**
   Sprint 91 closed the compressed-first first-move contradiction cleanly
   enough that Epic 9 can now move to backend/kernel maturity without
   reopening the same seam.

## Bottom-Line Closeout

Sprint 91 succeeded because it moved the public product model in a real way
without lying about what remains. Compressed CSR/CSC inputs now have first-
class public constructor-style entry paths, the README now teaches those paths
coherently, and the public direct-workflow lifecycle now proves them. The
sprint stayed bounded, preserved compatibility, and closed from a strong
validated baseline. It did not finish all compressed-first convergence work,
but it removed the highest-value linked-list-first entry-path contradiction and
left Epic 9 ready to move to backend maturity next.
