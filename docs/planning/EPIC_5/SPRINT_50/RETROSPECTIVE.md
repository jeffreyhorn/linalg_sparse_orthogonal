# Sprint 50 Retrospective

**Sprint:** 50 — Direct-Solver Lifecycle Baseline & API Design  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 50 baseline and scope captured before deeper lifecycle design work
- [x] reviewed validation/truthfulness baseline rechecked and fixed as the design anchor
- [x] direct-solver public surface inventory completed against the live repo
- [x] direct lifecycle precedent inventory completed across analysis, factors, and later handle work
- [x] ranked direct-solver lifecycle gap analysis completed
- [x] first-pass public direct-solver lifecycle API design completed
- [x] post-design audit completed
- [x] final caller-facing direct lifecycle contract completed
- [x] explicit non-goal and compatibility fence completed
- [x] Sprint 51 landing/validation plan completed
- [x] caller-surface audit completed across headers, README, examples, and benchmarks
- [x] summary/handoff draft completed from the final design state
- [x] final sanity sweep completed for terminology, scope, and budget consistency
- [x] Sprint 50 closeout and Sprint 51 handoff completed from the design baseline

## What Went Well

1. **Sprint 50 stayed tightly scoped as a design sprint instead of drifting into premature implementation.**
   The sprint did not pretend to be Sprint 51. It kept the work bounded to:
   - baseline/truthfulness recheck
   - public surface and precedent inventory
   - ranked gap analysis
   - caller-facing lifecycle design
   - compatibility and non-goal fencing
   - landing and validation planning
   That made the handoff cleaner than mixing partial header edits into an
   still-moving contract.

2. **The analysis-centric design anchor was the right choice.**
   Sprint 50 correctly concluded that the public repeated direct-run story
   should center on:
   - `sparse_analysis_t`
   - `sparse_factors_t`
   - analyze once
   - factor / solve
   - refactor / solve many
   - free explicitly
   That is stronger than inventing a broad new generic direct handle when the
   repository already had a better direct-solver precedent.

3. **The sprint kept the one-shot compatibility story honest.**
   The design did not overreach by treating one-shot LU / Cholesky / LDL^T as
   legacy mistakes. It made the final relationship explicit:
   - one-shot direct APIs remain first-class peer entry points
   - one-shot usage remains the simple/default path for one-off solves
   - the analysis/factor/refactor lifecycle is the explicit opt-in path for
     stable-pattern repeated direct runs
   That is a more credible public contract than forcing a migration story the
   repo has not actually landed yet.

4. **The sprint surfaced the real usability gap instead of only talking about API shape.**
   Day 5 reframed the problem correctly:
   - repeated direct workflow already exists
   - it is still under-centered publicly
   - LU / Cholesky one-shot paths still lean heavily on hidden mutable
     matrix-state knowledge
   - docs/examples still over-center the one-shot path
   That is a better foundation for Epic 5 than treating the problem as “just
   add more types.”

5. **The non-goal and compatibility fence was explicit early enough to matter.**
   Sprint 50 fixed the allowed-change boundary before implementation work:
   - no broad public factor-container redesign
   - no demotion/removal of one-shot direct APIs
   - no raw CSC/native storage exposure
   - no structural-pattern verifier redesign
   - no broad benchmark-framework redesign
   That should reduce Sprint 51-52 churn by making the design constraints
   explicit instead of implicit.

6. **The sprint chose the right later-adoption surfaces.**
   The later high-signal adopters were narrowed to the places that actually
   prove the repeated direct-run story:
   - `examples/example_analysis.c`
   - `benchmarks/bench_refactor.c`
   - direct factor/refactor regression binaries
   It correctly avoided broad conversion pressure on small one-shot examples
   and tutorial surfaces that are still better kept one-shot-first.

7. **The design package closed in a coherent state.**
   By Day 14, Sprint 50 could hand forward:
   - preserved truthfulness baseline
   - public surface and precedent inventories
   - lifecycle gap ranking
   - first-pass and final API design
   - post-design audit
   - scope fence
   - landing/validation plan
   - caller-surface audit
   - summary/handoff synthesis
   - final sanity confirmation
   That is a stronger Sprint 51 starting point than a pile of disconnected
   notes.

## What Didn't Go Well

1. **Sprint 50 intentionally did not land code, so the repeated-run direct story is still design-first rather than user-visible.**
   That is the right scope result for this sprint, but it also means the most
   important practical outcomes are still deferred to Sprint 51:
   - header/API edits
   - implementation/wrapper integration
   - direct tests
   - example/benchmark adoption
   - measured validation from the new direct-lifecycle end state

2. **The mutable-`SparseMatrix` one-shot tradeoff remains unresolved by design.**
   Sprint 50 correctly kept this as an explicit accepted compatibility
   boundary, but that also means the most awkward direct-solver lifecycle
   behavior remains present on the public surface rather than actually
   disappearing in this sprint.

3. **A small caller-doc drift queue is already known going into implementation.**
   Sprint 50 found two concrete documentation mismatches that it did not fix
   because they belong to the later adoption pass:
   - `benchmarks/README.md` mislabels `bench_refactor`
   - `examples/README.md` omits `example_analysis`
   Those are small issues, but they show the direct repeated-run story is not
   yet fully aligned across every caller-facing surface.

## Final Metrics

### Validated design baseline

| Metric | Sprint 50 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity contract | `53 vs 53` |
| Sprint 50 `*.c` / `*.h` changes landed | `0` |
| full code validation rerun in Sprint 50 | `not required` |

### Sprint 50 artifact package

| Metric | Sprint 50 close state |
|---|---:|
| total artifact files under `SPRINT_50/artifacts/` | `15` |
| inventory/analysis/design/fence artifacts (Days 3-10) | `8` |
| caller/synthesis/closeout artifacts (Days 11-14) | `4` |

### Direct lifecycle design outputs

| Metric | Sprint 50 close state |
|---|---:|
| direct solver families explicitly covered | `3` |
| primary public header targets named | `4` |
| targeted later follow-on binaries fixed in the landing plan | `8` |
| concrete later caller-doc drift items recorded | `2` |

Notes:

- direct solver families explicitly covered:
  - LU
  - Cholesky
  - LDL^T
- primary public header targets named:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- targeted later follow-on binaries fixed in the landing plan:
  - `./build/example_analysis`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- concrete later caller-doc drift items recorded:
  - `benchmarks/README.md` mislabels `bench_refactor`
  - `examples/README.md` omits `example_analysis`

## Residual Deferred Debt

Sprint 50 was explicitly about direct-solver lifecycle baseline and API
design. The main open work it intentionally hands forward is:

- public header/API integration for the final direct repeated-run contract
- implementation/wrapper integration for the analysis/factor/refactor story
- direct regression coverage for the bounded lifecycle contract
- high-signal adoption in:
  - `examples/example_analysis.c`
  - `benchmarks/bench_refactor.c`
- final docs alignment once the implementation lands
- measured validation from the implemented direct-lifecycle end state

Not carried forward as unresolved Sprint 50 debt:

- missing baseline/truthfulness recheck
- missing direct-solver public surface inventory
- missing lifecycle precedent audit
- missing ranked gap analysis
- missing final caller-facing lifecycle contract
- missing non-goal / compatibility fence
- missing Sprint 51 landing and validation plan

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-direct-solver-public-surface-inventory.md](./artifacts/day3-direct-solver-public-surface-inventory.md)
- [day4-lifecycle-precedent-inventory.md](./artifacts/day4-lifecycle-precedent-inventory.md)
- [day5-direct-solver-lifecycle-gap-analysis.md](./artifacts/day5-direct-solver-lifecycle-gap-analysis.md)
- [day6-public-direct-solver-lifecycle-api-design-batch1.md](./artifacts/day6-public-direct-solver-lifecycle-api-design-batch1.md)
- [day7-post-design-audit.md](./artifacts/day7-post-design-audit.md)
- [day8-public-direct-solver-lifecycle-api-design-batch2.md](./artifacts/day8-public-direct-solver-lifecycle-api-design-batch2.md)
- [day9-non-goal-and-compatibility-fence.md](./artifacts/day9-non-goal-and-compatibility-fence.md)
- [day10-validation-and-landing-plan.md](./artifacts/day10-validation-and-landing-plan.md)
- [day11-caller-surface-audit.md](./artifacts/day11-caller-surface-audit.md)
- [day12-summary-and-handoff-draft.md](./artifacts/day12-summary-and-handoff-draft.md)
- [day13-design-sanity-sweep.md](./artifacts/day13-design-sanity-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 50 achieved its goal:

- Epic 5 now has a coherent direct-solver lifecycle design package instead of a
  vague “state-model improvement” theme
- the repeated-run direct story is explicitly centered on the existing
  analysis/factor/refactor path
- one-shot LU / Cholesky / LDL^T APIs remain first-class and compatibility-safe
- the non-goal and validation fence for Sprint 51-52 is now written down
- the highest-value later adoption, test, benchmark, and doc surfaces are
  explicitly named

Sprint 51 can now start from a bounded, internally consistent contract instead
of reopening basic questions about whether the direct repeated-run public story
should be analysis-centric, whether one-shot direct APIs stay first-class, or
which caller surfaces actually need to prove the lifecycle path.
