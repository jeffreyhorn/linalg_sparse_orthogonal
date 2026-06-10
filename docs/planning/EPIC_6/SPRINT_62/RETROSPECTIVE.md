# Sprint 62 Retrospective

**Sprint:** 62 — Direct-Solver Usability & Lifecycle Coherence  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 62 baseline and scope captured from the Sprint 61 validated Epic 6 state
- [x] reviewed validation/truthfulness baseline rechecked before direct-usability code landed
- [x] direct one-shot usability pain points reduced to a ranked family-specific audit
- [x] lifecycle coherence and safety contract designed before direct-family hardening landed
- [x] bounded LU landing design fixed before implementation changes
- [x] first LU one-shot hardening batch landed with stronger reused-state rejection and lifecycle guidance
- [x] second LU one-shot hardening batch landed with preserved-caller reordered cancel/failure behavior
- [x] post-LU audit narrowed the remaining sprint queue to one bounded Cholesky seam
- [x] Cholesky convergence/preservation design fixed before the second family batch landed
- [x] bounded Cholesky hardening landed with preserved-caller reordered cancel/failure behavior
- [x] compatibility/regression sweep tightened the direct-family story without widening into unrelated families
- [x] docs, tutorial, examples, and maintainer guidance updated to match the landed usability model
- [x] full validation sweep completed from the final landed Sprint 62 tree
- [x] Sprint 62 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 62 stayed focused on the highest-value direct usability seams instead of turning into a broad direct-family rewrite.**
   The sprint kept the implementation center of gravity on:
   - LU one-shot hardening
   - reordered Cholesky preservation hardening
   - integration-level compatibility proof
   - adoption/docs follow-through
   It did not widen into:
   - CSC implementation redesign
   - LDL^T symmetry-for-its-own-sake work
   - QR churn
   - configuration-surface changes
   - packaging/platform work
   That scope discipline is why the sprint shipped a coherent usability package
   instead of a scattered direct-solver cleanup.

2. **The broad Epic 6 “direct usability” critique turned into real shipped behavior changes.**
   Sprint 62 did not stop at naming or documentation cleanup. It materially
   reduced mutable-matrix surprise on the highest-value paths:
   - LU one-shot wrappers now reject reused row/column state up front
   - reordered LU one-shot calls now factor on a temporary reordered working
     copy and only publish back on success
   - reordered Cholesky one-shot calls now follow the same preserved-caller
     rule
   Those are meaningful productization improvements, not just clearer comments.

3. **The public direct-workflow boundary is clearer at sprint close than it was at sprint start.**
   Sprint 62 ended with a more explicit shipped split:
   - one-shot direct wrappers remain first-class/default peer entry points
   - the explicit repeated-run direct lifecycle remains the canonical reuse
     path:
     - `sparse_analyze()`
     - `sparse_factor_numeric()`
     - `sparse_factor_solve()`
     - `sparse_refactor_numeric()`
   That is better than the pre-sprint state where one-shot convenience and
   lifecycle reuse were technically distinct but not consistently presented as
   such.

4. **Compatibility was tightened without faking total family uniformity.**
   The sprint improved coherence without pretending every direct family now has
   identical semantics:
   - reordered LU and reordered Cholesky preserve the caller matrix on
     cancel/failure
   - no-reorder linked-list Cholesky cancellation remains on its existing
     family-local compatibility lane
   - LDL^T remains a cleaner separate-owner surface rather than being widened
     artificially for visual symmetry
   That honesty matters more than a cosmetically uniform but misleading direct
   API story.

5. **`test_integration` absorbed the new proof burden well.**
   The sprint chose the right proof home for the user-facing direct-lifecycle
   story:
   - mistaken second LU one-shot reuse is now covered
   - reordered LU cancel-after-reorder preservation is now covered
   - reordered Cholesky cancel-after-reorder preservation is now covered
   - reordered Cholesky `SPARSE_ERR_NOT_SPD` preservation is now covered
   - factor preservation after rejected one-shot reuse is now covered
   That kept the highest-signal direct-usability proof close to actual caller
   stories instead of scattering it across lower-signal family-local tests.

6. **The sprint still closed from the strongest reviewed baseline.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the maintained reviewed anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - full reviewed CMake `ctest` `53 / 53`
   - reviewed CMake total real time `377.56 sec`
   That keeps the Sprint 62 handoff credible as productization work rather than
   only a local implementation exercise.

7. **Sprint 62 ended with a smaller and more explicit deferred queue.**
   The sprint did not just improve LU and Cholesky. It also clarified what
   remains for Sprint 63 and later:
   - no-reorder linked-list Cholesky cancellation restoration
   - broader LDL^T wording follow-through only if a later contradiction appears
   - QR as a comparison/deferred surface
   - deeper direct-lifecycle uniformity and CSC/LU follow-through in Sprint 63
   That is a much cleaner exit than the original “direct usability” headline.

## What Didn't Go Well

1. **The direct-family usability story still needed substantial design and documentation work relative to the amount of code touched.**
   That was the right tradeoff for a productization sprint, but it means the
   branch still carries a high contract-writing-to-code ratio.

2. **The sprint improved the highest-value cancel/preservation seams, not every direct-family edge case.**
   The remaining queue is smaller, but still real:
   - no-reorder linked-list Cholesky restoration
   - later LDL^T follow-through if needed
   - later CSC/LU/direct-lifecycle uniformity work

3. **`test_integration` continues to accumulate high-value caller-story proof density.**
   That was the right proof home for Sprint 62, but it also means one of the
   repo’s already dense tests became even more central to future direct-family
   validation.

4. **The reviewed CMake rebuild still emits ordinary benchmark warning noise.**
   The sprint closed cleanly, but the recurring `bench_eigs_reuse.c`
   double-promotion warnings remain part of the background validation story
   instead of being cleaned up here.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 62 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `377.56 sec` |

### Sprint 62 artifact package

| Metric | Sprint 62 close state |
|---|---:|
| total artifact files under `SPRINT_62/artifacts/` | `15` |
| main design/integration artifacts | `8` |
| compatibility/docs/validation/closeout artifacts | `4` |

Notes:

- main design/integration artifacts:
  - `day3-direct-one-shot-usability-audit.md`
  - `day4-lifecycle-coherence-design-and-safety-contract.md`
  - `day5-lu-landing-design.md`
  - `day6-one-shot-hardening-batch1.md`
  - `day7-one-shot-hardening-batch2.md`
  - `day8-post-lu-lifecycle-wrapper-audit.md`
  - `day9-cholesky-convergence-design.md`
  - `day10-cholesky-preservation-hardening-batch.md`
- compatibility/docs/validation/closeout artifacts:
  - `day11-compatibility-layer-and-regression-sweep.md`
  - `day12-docs-and-example-adoption-update.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Sprint 62 landed direct-usability package

| Metric | Sprint 62 close state |
|---|---:|
| direct families materially hardened | `2` |
| new high-signal direct compatibility proofs added in `test_integration` | `5` |
| targeted Day 13 follow-on commands rerun | `20` |
| explicitly deferred later direct-usability lanes | `5` |

Notes:

- direct families materially hardened:
  - `LU`
  - `Cholesky`
- new high-signal direct compatibility proofs added in `test_integration`:
  - mistaken second LU one-shot reuse returns `SPARSE_ERR_BADARG`
  - rejected LU one-shot reuse preserves the previously built factor
  - reordered LU cancel-after-reorder preserves the original matrix and allows later retry
  - reordered Cholesky cancel-after-reorder preserves the original matrix and allows later retry
  - reordered Cholesky `SPARSE_ERR_NOT_SPD` preserves the original caller matrix
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
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
- explicitly deferred later direct-usability lanes:
  - no-reorder linked-list Cholesky cancellation restoration
  - broader LDL^T wording/compatibility follow-through if needed
  - QR comparison/deferred surface
  - broader direct-family docs/examples simplification
  - deeper direct-lifecycle uniformity and CSC/LU follow-through in Sprint 63

## Residual Deferred Debt

Sprint 62 was explicitly about the highest-value one-shot direct usability and
lifecycle coherence seams. The main open work it intentionally hands forward
is:

- no-reorder linked-list Cholesky cancellation restoration
- deeper direct-lifecycle uniformity and CSC/LU follow-through
- broader LDL^T wording or compatibility follow-through only if later needed
- QR comparison/deferred follow-through
- broader direct-family docs/examples simplification outside the touched
  high-signal surfaces
- later backend/performance/assurance work from the wider Epic 6 queue

Still consciously constrained rather than silently “solved”:

- no fake convergence between one-shot wrappers and explicit lifecycle reuse
- no silent copy-everywhere behavior to hide matrix mutation
- no CSC implementation redesign in Sprint 62
- no packaging/platform widening
- no reopening of the repeated-run workflow fence

Not carried forward as unresolved Sprint 62 debt:

- missing direct-family usability audit
- missing explicit lifecycle/wrapper safety contract
- missing LU preserved-caller reordered cancel/failure hardening
- missing reordered Cholesky preserved-caller hardening
- missing integration-level direct compatibility proof for the touched seams
- missing direct adoption/docs follow-through
- missing validated Sprint 62 closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-direct-one-shot-usability-audit.md](./artifacts/day3-direct-one-shot-usability-audit.md)
- [day4-lifecycle-coherence-design-and-safety-contract.md](./artifacts/day4-lifecycle-coherence-design-and-safety-contract.md)
- [day5-lu-landing-design.md](./artifacts/day5-lu-landing-design.md)
- [day6-one-shot-hardening-batch1.md](./artifacts/day6-one-shot-hardening-batch1.md)
- [day7-one-shot-hardening-batch2.md](./artifacts/day7-one-shot-hardening-batch2.md)
- [day8-post-lu-lifecycle-wrapper-audit.md](./artifacts/day8-post-lu-lifecycle-wrapper-audit.md)
- [day9-cholesky-convergence-design.md](./artifacts/day9-cholesky-convergence-design.md)
- [day10-cholesky-preservation-hardening-batch.md](./artifacts/day10-cholesky-preservation-hardening-batch.md)
- [day11-compatibility-layer-and-regression-sweep.md](./artifacts/day11-compatibility-layer-and-regression-sweep.md)
- [day12-docs-and-example-adoption-update.md](./artifacts/day12-docs-and-example-adoption-update.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 62 achieved its goal:

- the highest-value direct one-shot usability surprises are materially smaller
- the relationship between one-shot direct solves and the explicit repeated-run
  lifecycle is clearer
- the strongest reordered LU and Cholesky cancel/failure seams now preserve the
  caller matrix instead of stranding it in an intermediate reordered state
- the docs, tutorial, examples, and maintainer story now match the shipped
  behavior more directly
- the sprint closes from a fully validated reviewed baseline with an explicit
  deferred queue for Sprint 63 and later Epic 6 work
