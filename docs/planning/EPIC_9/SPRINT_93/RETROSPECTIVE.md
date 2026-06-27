# Sprint 93 Retrospective

**Sprint:** 93 — Runtime Scalability, Threading & ND Convergence Phase 2  
**Duration:** 14 days  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 93 fixed the implementation-day validation and maintained-surface
      contract before runtime/threading work widened
- [x] the sprint reduced the broad runtime problem to one ranked live
      contradiction map instead of reopening generic graph/reorder churn
- [x] Sprint 93 froze one explicit threading/runtime contract:
  - algorithmic ND runtime debt first
  - runtime-control debt second
  - proof-topology debt only where later evidence truly needed it
- [x] Sprint 93 froze one explicit first implementation fence centered on:
  - the ND recursive runtime seam
- [x] Sprint 93 froze one explicit ND runtime-reduction contract:
  - reduce recursion-side repeated work
  - preserve ordering semantics and policy reading
  - keep proof and benchmark widening bounded
- [x] Sprint 93 landed one bounded ND runtime-reduction batch:
  - packed side/separator scratch reuse
  - removed separate side-array churn
  - removed the extra separator-emission scan
- [x] Sprint 93 landed one bounded runtime-control cleanup batch:
  - default-policy baseline and compat-override seams separated
  - override staging grouped behind one scoped helper
  - shipped env names and policy precedence preserved
- [x] Sprint 93 landed one bounded runtime-evidence follow-through batch:
  - `bench_reorder` now emits `reorder_path`
  - `bench_reorder` now emits `fixture_slice`
  - `bench_reorder` now emits `nd_base_threshold`
- [x] Sprint 93 froze one final owner map before close:
  - ND runtime owner
  - reviewed runtime proof owners
  - bounded reorder runtime-evidence owner
  - retained canonical reporting owner
- [x] Sprint 93 ran the full final validation sweep and closed from one
      explicit validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - focused reviewed-runtime reruns
  - bounded `bench_reorder --sprint86-slice` reruns
  - `make bench-canonical-report`
- [x] Sprint 93 closed with one explicit Sprint 94-first handoff queue instead
      of reopening the runtime/ND seam

## What Went Well

1. **Sprint 93 chose the right first runtime seam.**
   The sprint did not begin with generic threading work. It targeted the
   strongest reviewed long pole first: recursion-side cost inside the ND path.

2. **The runtime reduction stayed technically disciplined.**
   Day 7 removed repeated side-array allocation and the extra separator scan
   without changing permutation semantics, threshold behavior, or policy
   interpretation.

3. **The runtime-control cleanup sharpened the model without widening claims.**
   Day 10 improved internal control ownership by separating baseline defaults,
   compatibility overrides, and override staging, while preserving shipped env
   names and precedence.

4. **The evidence gap closed in the right place.**
   Sprint 93 did not try to solve residual runtime interpretation by moving
   proof owners around. It made the retained reorder benchmark rows more
   self-describing instead.

5. **The sprint stayed bounded under pressure.**
   It did not widen into broad graph-policy redesign, public runtime-claim
   rewriting, workflow churn, or build/package changes detached from the
   touched runtime seam.

6. **The sprint closed from a strong validated baseline.**
   Sprint 93 closed from the implementation-day queue, the full reviewed path,
   exact Makefile/CMake parity, focused runtime-heavy reruns, bounded
   benchmark-evidence reruns, and canonical benchmark reporting.

## What Didn't Go Well

1. **The reviewed long pole is still real.**
   Sprint 93 reduced the contradiction materially, but `test_reorder_nd`
   remained the slowest reviewed path by a wide margin at close.

2. **The runtime evidence is clearer, not simpler.**
   `bench_reorder` now emits enough context to interpret rows cleanly, but the
   Sprint 86 slice still reads as mixed by matrix and by entry path rather than
   as a single monotone improvement story.

3. **Threading maturity remained intentionally bounded.**
   Sprint 93 improved the runtime/control model, but it did not create a broad
   OpenMP or parallel-runtime product claim. The maintained proof still reads
   truthfully as a serial build lane on this machine.

4. **Proof-topology reduction stayed deferred.**
   That was the right bounded call, but it means the heaviest reviewed proof
   owner still remains concentrated in `test_reorder_nd.c`.

5. **The sprint improved ND/runtime convergence more than broad scalability.**
   Sprint 93 materially reduced the highest-value runtime contradiction, but it
   did not attempt to solve every runtime, graph, or concurrency ceiling in
   one pass.

## Final Metrics

### Validation and close anchors

| Metric | Sprint 93 close state |
|---|---:|
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `286.93 sec` |
| reviewed `test_reorder_nd` time | `169.17 sec` |
| focused `test_reorder_nd` rerun time | `175.541 s` |
| canonical reporting follow-through | `make bench-canonical-report` passed |

### Focused Sprint 93 runtime and evidence anchors

| Metric | Sprint 93 close state |
|---|---:|
| reviewed runtime owner | `test_reorder_nd` |
| `test_reorder_nd` result | `35 / 35`, `1` skip |
| retained adjacent graph owner | `test_graph` |
| `test_graph` result | `61 / 61` |
| retained threading proof owner | `test_threads` |
| `test_threads` result | `8 / 8` |
| retained OpenMP proof owner | `test_omp` |
| `test_omp` result | `12 / 12` |
| `example_analysis` residual | `4.44e-16` |
| `example_basic_solve` residual | `0.00e+00` |
| bounded direct-path ND evidence row | `Pres_Poisson,...,nd,...,5165.8,...,direct,sprint86,160` |
| bounded analyze-path ND evidence row | `Pres_Poisson,...,nd,...,5589.6,...,analyze,sprint86,160` |

### Sprint 93 artifact package

| Metric | Sprint 93 close state |
|---|---:|
| total artifact files under `SPRINT_93/artifacts/` | `15` |
| baseline/validation/closeout artifacts | `5` |
| audit/design/fence artifacts | `7` |
| implementation/follow-through artifacts | `3` |

Notes:

- baseline/validation/closeout artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-and-runtime-baseline.md`
  - `day2-validation-baseline-and-maintained-surface-recheck.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`
- audit/design/fence artifacts:
  - `day3-reviewed-runtime-audit.md`
  - `day4-threading-and-runtime-contract-design.md`
  - `day5-first-implementation-boundary.md`
  - `day6-nd-runtime-reduction-design.md`
  - `day8-post-landing-audit-and-rerank.md`
  - `day9-runtime-control-cleanup-design.md`
  - `day11-proof-surface-and-runtime-evidence-design.md`
- implementation/follow-through artifacts:
  - `day7-nd-runtime-reduction-batch.md`
  - `day10-runtime-control-cleanup-batch.md`
  - `day12-proof-and-runtime-evidence-follow-through-batch.md`

### Landed change class

| Metric | Sprint 93 close state |
|---|---:|
| implementation source owners touched | `1` |
| proof-owner C test files touched | `0` |
| benchmark C owners touched | `1` |
| public product docs touched | `0` |
| benchmark/support docs touched | `2` |
| build-system surfaces touched | `0` |
| workflow files touched | `0` |
| sprint-local planning surfaces touched | substantial |

Notes:

- landed implementation/evidence/support surfaces:
  - `src/sparse_reorder_nd.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- no README, INSTALL, test-owner, build-system, or workflow surface had to
  move for Sprint 93 to land

## Residual Deferred Debt

Sprint 93 intentionally reduced the strongest reviewed runtime contradiction
without pretending the full runtime/threading story is now broad or finished.

Most important carry-forward work:

- capability-envelope widening
- later public narrative and workflow coherence
- remaining large-source and proof-owner maintainability concentration
- build/package/workflow convergence
- broader comparison depth
- final Epic 9 integration and closeout

Still consciously constrained rather than silently "solved":

- no broad threading-scalability claim
- no broad OpenMP maturity claim
- no claim that the ND lane is now uniformly superior across all matrices or
  entry paths
- no claim that the heaviest reviewed proof owner has been decomposed

Not carried forward as unresolved Sprint 93 debt:

- the reviewed runtime contradiction rerank
- the threading/runtime contract freeze
- the Day 7 recursion-side runtime reduction
- the Day 10 runtime-control cleanup
- the Day 12 bounded runtime-evidence follow-through
- the Day 13 validated close baseline

## Key Deliverables

1. **One real ND runtime reduction landed.**
   Sprint 93 removed avoidable recursion-side work from the strongest reviewed
   runtime seam without changing ordering semantics.

2. **One cleaner runtime-control model landed.**
   The touched ND runtime/control path now has smaller internal ownership
   boundaries while preserving shipped env and policy behavior.

3. **One missing runtime-evidence context gap was closed.**
   `bench_reorder` now emits the path, fixture slice, and live ND threshold
   directly in each bounded runtime-evidence row.

4. **One exact validated Sprint 93 baseline landed.**
   The sprint closes from the full local validation queue, reviewed parity,
   focused runtime-heavy reruns, bounded benchmark-evidence reruns, and
   canonical benchmark reporting.

5. **One explicit Sprint 94-first handoff queue landed.**
   Sprint 93 closed the runtime/ND first-move contradiction cleanly enough that
   Epic 9 can now move to capability widening instead of reopening the same
   seam.

## Bottom-Line Closeout

Sprint 93 succeeded because it materially reduced the strongest reviewed
runtime contradiction without lying about what remains. The ND runtime seam is
smaller, the touched control model is cleaner, the bounded benchmark evidence
is more interpretable, and the sprint closes from a strong validated baseline.
It did not solve broad threading maturity or eliminate the reviewed long pole,
but it moved the highest-value runtime contradiction enough that Epic 9 can
proceed to capability widening next.
