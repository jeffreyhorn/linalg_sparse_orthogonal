# Sprint 63 Retrospective

**Sprint:** 63 — Direct-Lifecycle Uniformity & CSC/LU Follow-Through  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 63 baseline and scope captured from the Sprint 62 validated Epic 6 state
- [x] reviewed validation/truthfulness baseline rechecked before lifecycle code landed
- [x] remaining direct-lifecycle and CSC seams reduced to a ranked live-path audit
- [x] lifecycle-uniformity safety contract designed before implementation changes landed
- [x] exact LU and Cholesky/CSC implementation fence fixed before code batches landed
- [x] bounded LU lifecycle follow-through landed with stronger early invalid-option rejection
- [x] bounded Cholesky/CSC follow-through landed with clearer CSC dispatch/publication semantics
- [x] large-`n` CSC-backed repeated-run direct failure-preserve proof landed on the public lifecycle lane
- [x] compatibility/regression sweep tightened the family-local CSC and header truthfulness story
- [x] docs, examples, benchmarks, and maintainer guidance updated to match the landed Sprint 63 behavior
- [x] full validation sweep completed from the final landed Sprint 63 tree
- [x] Sprint 63 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 63 stayed focused on the strongest real lifecycle seams instead of drifting into a general direct-family rewrite.**
   The sprint kept the implementation center of gravity on:
   - LU lifecycle follow-through
   - Cholesky CSC lifecycle follow-through
   - shared repeated-run direct failure-preserve semantics
   - bounded compatibility proof
   It did not widen into:
   - LDL^T symmetry-for-its-own-sake work
   - QR churn
   - configuration-surface work
   - benchmark-governance or packaging/platform work
   That scope discipline is why Sprint 63 closed as one coherent lifecycle
   uniformity sprint rather than as another mixed direct-solver cleanup pass.

2. **The broad “direct-lifecycle uniformity” goal turned into concrete shipped behavior on both the LU and CSC sides.**
   Sprint 63 did not stop at wording cleanup. It landed real direct-solver
   safety and coherence improvements:
   - invalid LU pivot/reorder enums now reject before reorder or factor
     mutation begins
   - invalid Cholesky reorder/backend enums now reject before reorder or
     factor mutation begins
   - the large-`n` CSC-backed repeated-run Cholesky lane now explicitly
     preserves the previous usable factors on same-pattern non-SPD failure
     and obvious nnz-drift rejection
   - the CSC supernodal path now rejects a stored non-positive diagonal before
     deeper supernodal mutation work begins
   Those are meaningful lifecycle and productization improvements, not just
   cleaner comments.

3. **The public repeated-run direct contract is clearer at sprint close than it was at sprint start.**
   Sprint 63 ended with a stronger and more explicit shipped rule:
   - repeated direct reuse preserves symbolic/permutation setup, not stale
     numeric factor contents
   - failed `sparse_refactor_numeric(...)` calls preserve the previous usable
     factor state
   - that rule now extends explicitly to the large-`n` CSC-backed Cholesky
     repeated-run lane
   This is better than the pre-sprint state where the public lifecycle was
   already present but still had uneven proof and interpretation across the
   hardest CSC path.

4. **The sprint improved internal uniformity without faking total direct-family equivalence.**
   Sprint 63 made the real lifecycle behavior more uniform while preserving
   intentional family-local differences:
   - one-shot wrappers remain first-class/default peer entry points
   - reordered LU and reordered Cholesky still preserve the caller matrix on
     cancel/failure through temporary reordered working copies
   - no-reorder linked-list Cholesky cancellation remains intentionally
     non-bit-identical
   - LDL^T remains a cleaner separate-owner surface rather than being widened
     artificially for visual symmetry
   That honesty matters more than a cosmetically uniform but misleading direct
   API story.

5. **The sprint chose the right proof homes for the user-facing and family-local contracts.**
   Sprint 63 spread proof burden intelligently instead of piling everything into
   one test:
   - `tests/test_integration.c` now owns the public large-`n` CSC-backed
     repeated-run old-factor-preservation story
   - `tests/test_chol_csc.c` now owns the family-local CSC supernodal early
     rejection proof
   - the LU and Cholesky header follow-through makes the early invalid-option
     contract visible at the call site
   That proof placement is one of the strongest technical choices in the
   sprint.

6. **The docs/example/benchmark follow-through stayed bounded and clarified proof ownership.**
   Sprint 63 did not turn Day 12 into a broad docs rewrite. Instead it made a
   few high-value clarifications:
   - `README.md` now states the repeated-run direct failure-preserve rule
     directly
   - `docs/tutorial.md` teaches that rule at the one-shot-to-lifecycle handoff
   - `examples/README.md` keeps `example_analysis` as the adoption example, not
     the error-path proof source
   - `benchmarks/README.md` keeps `bench_refactor_csc` as the throughput/proof
     surface, not the error-path contract source
   - `docs/maintainer_guide.md` now owns the Sprint 63 direct-family
     interpretation explicitly
   That is a better documentation state than the pre-sprint mix of public
   contract, benchmark proof, and maintainer interpretation.

7. **Sprint 63 still closed from the strongest reviewed baseline.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the maintained reviewed anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - full reviewed CMake `ctest` `53 / 53`
   - reviewed CMake total real time `348.10 sec`
   That keeps the Sprint 63 handoff credible as productization/lifecycle work,
   not just local implementation cleanup.

8. **Sprint 63 ended with a smaller and more explicit deferred queue.**
   The sprint did not just improve LU and CSC. It also clarified what remains:
   - no-reorder linked-list Cholesky bit-identical cancellation restoration
   - CSC progress-callback parity for Cholesky / LDL^T
   - broader LDL^T or QR wording follow-through only if a new contradiction appears
   - later direct-family docs/examples density cleanup outside the touched
     high-signal surfaces
   That is a much cleaner exit than the original broad “direct-lifecycle
   uniformity” headline.

## What Didn't Go Well

1. **The sprint still required a substantial amount of contract-writing and proof interpretation relative to the size of the implementation edits.**
   That was appropriate for the remaining Epic 6 direct-lifecycle seams, but it
   means Sprint 63 still carries a high design/proof-to-code ratio.

2. **The direct-family lifecycle story is stronger, not totally “solved.”**
   The remaining queue is smaller, but still real:
   - no-reorder linked-list Cholesky cancellation restoration
   - CSC progress-callback parity
   - later LDL^T or QR wording follow-through if needed
   - later docs/examples density cleanup

3. **`tests/test_integration.c` remains one of the most important and dense proof homes in the repo.**
   That was the right place for the large-`n` CSC-backed public lifecycle
   proof, but it also means another high-value direct-family guarantee now
   lives in one of the repo’s already dense integration surfaces.

4. **The reviewed CMake rebuild still emits ordinary benchmark warning noise.**
   The sprint closed cleanly, but the recurring `bench_eigs_reuse.c`
   double-promotion warnings remain part of the background validation story
   instead of being cleaned up here.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 63 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `348.10 sec` |

### Sprint 63 artifact package

| Metric | Sprint 63 close state |
|---|---:|
| total artifact files under `SPRINT_63/artifacts/` | `15` |
| main design/integration artifacts | `8` |
| compatibility/docs/validation/closeout artifacts | `4` |

Notes:

- main design/integration artifacts:
  - `day3-internal-path-audit.md`
  - `day4-lifecycle-uniformity-design-and-safety-contract.md`
  - `day5-header-and-internal-landing-design.md`
  - `day6-lu-lifecycle-follow-through-batch1.md`
  - `day7-cholesky-csc-lifecycle-follow-through-batch1.md`
  - `day8-post-landing-audit-and-residual-rerank.md`
  - `day9-solve-refactor-semantics-design.md`
  - `day10-large-n-csc-cholesky-lifecycle-semantics-batch.md`
- compatibility/docs/validation/closeout artifacts:
  - `day11-compatibility-layer-and-regression-sweep.md`
  - `day12-docs-example-benchmark-follow-through.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Sprint 63 landed lifecycle-uniformity package

| Metric | Sprint 63 close state |
|---|---:|
| direct families materially tightened | `2` |
| new highest-signal regression additions on touched proof homes | `4` |
| targeted Day 13 follow-on commands rerun | `20` |
| explicitly deferred later lifecycle/follow-through lanes | `5` |

Notes:

- direct families materially tightened:
  - `LU`
  - `Cholesky / CSC`
- new highest-signal regression additions on touched proof homes:
  - invalid LU pivot values reject before reorder/factor mutation and preserve
    original matrix state for later retry
  - invalid Cholesky backend values reject before reorder/factor mutation and
    preserve original matrix state for later retry
  - large-`n` CSC-backed Cholesky repeated-run non-SPD refactor failure
    preserves old usable factors
  - family-local CSC supernodal path rejects stored non-positive diagonal
    before deeper mutation work begins
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_sparse_lu`
  - `./build/test_cholesky`
  - `./build/test_chol_csc`
  - `./build/test_ldlt`
  - `./build/test_ldlt_csc`
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- explicitly deferred later lifecycle/follow-through lanes:
  - no-reorder linked-list Cholesky bit-identical cancellation restoration
  - CSC progress-callback parity for Cholesky / LDL^T
  - broader LDL^T wording follow-through only if needed
  - broader QR wording follow-through only if needed
  - later direct-family docs/examples density cleanup outside the bounded Sprint 63 surfaces

## Residual Deferred Debt

Sprint 63 was explicitly about the strongest remaining LU and CSC lifecycle
heterogeneity behind the public direct-lifecycle model. The main open work it
intentionally hands forward is:

- no-reorder linked-list Cholesky bit-identical cancellation restoration
- CSC progress-callback parity for Cholesky / LDL^T
- broader LDL^T or QR wording follow-through only if later contradictions appear
- later direct-family docs/examples density cleanup outside the touched
  high-signal Sprint 63 surfaces
- later productization/performance/assurance work above the now-stabilized
  LU/CSC lifecycle base

Still consciously constrained rather than silently “solved”:

- no direct API redesign
- no fake family-uniform cancellation promise
- no widening into configuration-surface work
- no packaging/platform expansion
- no reopening of the repeated-run workflow fence

Not carried forward as unresolved Sprint 63 debt:

- missing LU lifecycle follow-through
- missing Cholesky CSC lifecycle follow-through
- missing large-`n` CSC-backed repeated-run old-factor-preservation proof
- missing family-local CSC early-rejection regression
- missing bounded public/adoption/maintainer wording follow-through
- missing validated Sprint 63 closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-internal-path-audit.md](./artifacts/day3-internal-path-audit.md)
- [day4-lifecycle-uniformity-design-and-safety-contract.md](./artifacts/day4-lifecycle-uniformity-design-and-safety-contract.md)
- [day5-header-and-internal-landing-design.md](./artifacts/day5-header-and-internal-landing-design.md)
- [day6-lu-lifecycle-follow-through-batch1.md](./artifacts/day6-lu-lifecycle-follow-through-batch1.md)
- [day7-cholesky-csc-lifecycle-follow-through-batch1.md](./artifacts/day7-cholesky-csc-lifecycle-follow-through-batch1.md)
- [day8-post-landing-audit-and-residual-rerank.md](./artifacts/day8-post-landing-audit-and-residual-rerank.md)
- [day9-solve-refactor-semantics-design.md](./artifacts/day9-solve-refactor-semantics-design.md)
- [day10-large-n-csc-cholesky-lifecycle-semantics-batch.md](./artifacts/day10-large-n-csc-cholesky-lifecycle-semantics-batch.md)
- [day11-compatibility-layer-and-regression-sweep.md](./artifacts/day11-compatibility-layer-and-regression-sweep.md)
- [day12-docs-example-benchmark-follow-through.md](./artifacts/day12-docs-example-benchmark-follow-through.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 63 achieved its goal:

- the strongest remaining LU and CSC lifecycle asymmetries were reduced
  materially without widening into a general direct-family redesign
- the public repeated-run direct story is now better matched by both the
  implementation and the proof surface on the hardest large-`n` CSC-backed
  Cholesky lane
- the sprint closed from a fully reviewed validated baseline and left behind a
  smaller, cleaner, and more explicit deferred queue
