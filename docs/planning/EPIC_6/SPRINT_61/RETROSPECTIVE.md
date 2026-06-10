# Sprint 61 Retrospective

**Sprint:** 61 — Configuration Surface Modernization Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 61 baseline and scope captured from the Sprint 60 frozen Epic 6 contract
- [x] reviewed validation/truthfulness baseline rechecked before configuration code landed
- [x] live `SPARSE_ND_*`, `SPARSE_FM_*`, and adjacent env-var surface reduced to a ranked inventory
- [x] typed-options and precedence contract designed before public/header changes landed
- [x] first typed reorder/ND option batch landed through `sparse_analysis_opts_t.reorder_opts`
- [x] second typed reorder/ND option batch landed and completed the initial public policy slice
- [x] remaining bounded analysis-time control batch landed for the Phase 1 seam
- [x] compatibility/default behavior tightened and explicitly regression-tested
- [x] public docs, public header wording, and maintainer-policy story updated to match the landed model
- [x] full validation sweep completed from the final landed Sprint 61 tree
- [x] Sprint 61 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 61 stayed tightly inside the Phase 1 fence instead of turning into a repo-wide configuration rewrite.**
   The sprint kept the scope bounded to the highest-value analysis/reorder
   controls and did not widen into:
   - public `SPARSE_FM_*` tuning
   - debug/profile migration
   - backend/AUTO policy work
   - packaging/platform work
   - repeated-run workflow redesign
   That discipline is the main reason the sprint produced a coherent shipped
   Phase 1 surface instead of another partial control experiment.

2. **The broad Epic 6 “too env-var driven” critique was converted into a real shipped typed path.**
   Sprint 61 did not stop at inventory and design. It actually landed typed
   configuration on `sparse_analysis_opts_t.reorder_opts` for:
   - `SPARSE_SUPERNODAL_POSTORDER`
   - `SPARSE_ND_ROOT_BISECT`
   - `SPARSE_ND_ROOT_BISECT_MAX_N`
   - `SPARSE_ND_COARSENING`
   - `SPARSE_ND_COARSEST_BISECTION`
   - `SPARSE_ND_SEP_LIFT_STRATEGY`
   - `SPARSE_ND_SEP_LIFT_WEIGHT`
   - `SPARSE_ND_COARSEN_FLOOR_RATIO`
   That is a meaningful productization step, not just documentation or naming
   cleanup.

3. **The precedence model is now much clearer than at sprint start.**
   Sprint 61 ended with one explicit shipped rule:
   1. explicit typed value
   2. legacy compatibility env var when unspecified
   3. internal default
   That rule now exists in:
   - implementation behavior
   - regression coverage
   - public header wording
   - README/maintainer guidance
   That coherence is one of the sprint’s strongest architectural wins.

4. **Compatibility behavior was preserved without pretending the legacy env-var lane disappeared.**
   The sprint kept env-var compatibility where promised, but reclassified it
   cleanly:
   - `SPARSE_ND_SUPERNODAL_POSTORDER` is now compatibility-only
   - `SPARSE_ND_COARSENING_CV_FALLTHROUGH` is explicitly internal-policy-only
   - debug/profile controls remain deferred
   This is much better than the earlier state where the remaining env-var
   surface was harder to interpret from public docs alone.

5. **The graph/reorder proof home absorbed the new control-plane burden cleanly.**
   The highest-risk proof surface for this sprint was always the ND/graph path,
   and it held up well:
   - `test_graph` stayed clean
   - `test_graph_fm_buckets` stayed clean
   - `test_reorder_nd` became the main precedence/default proof home
   - `test_reorder_amd_qg` stayed clean
   That let the sprint add real control-plane coverage without scattering the
   proof burden across unrelated test binaries.

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
   - reviewed CMake total real time `368.17 sec`
   That matters because Sprint 61 is the first real Epic 6 implementation
   sprint.

7. **Sprint 61 ended with a smaller and more intentional deferred queue.**
   The sprint did not just land code; it also clarified what remains for later:
   - `SPARSE_FM_*` rationalization
   - debug/profile control treatment
   - optional compatibility-only alias cleanup
   - broader later configuration/policy work beyond the analysis/reorder seam
   That gives Sprint 62+ a cleaner Phase 2 starting point.

## What Didn't Go Well

1. **The sprint still required substantial design and policy writing around a relatively small code seam.**
   That was appropriate for the first Epic 6 implementation sprint, but it
   means the ratio of contract/docs work to net public-surface expansion is
   still high.

2. **The full env-var story is not solved yet, only the highest-value Phase 1 slice.**
   Sprint 61 landed the right first controls, but the remaining queue is still
   real:
   - `SPARSE_FM_*`
   - debug/profile controls
   - compatibility-only alias cleanup if later justified
   - broader policy rationalization outside the ND/analysis seam

3. **The strongest proof surface for this sprint remains one of the slower local paths.**
   `test_reorder_nd` is now the main precedence/default proof home, which is
   the right technical choice, but it also means the control-plane validation
   tail is relatively expensive compared with smaller unit-style coverage.

4. **The reviewed CMake rebuild still emits ordinary warnings on the benchmark tail.**
   The sprint ended with clean reviewed validation, but the recurring
   `bench_eigs_reuse` warning noise remains part of the background validation
   story rather than being fully cleaned up here.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 61 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `368.17 sec` |

### Sprint 61 artifact package

| Metric | Sprint 61 close state |
|---|---:|
| total artifact files under `SPRINT_61/artifacts/` | `15` |
| main design/integration artifacts | `8` |
| compatibility/docs/validation/closeout artifacts | `4` |

Notes:

- main design/integration artifacts:
  - `day4-typed-options-design-and-precedence-contract.md`
  - `day5-header-and-internal-surface-landing-design.md`
  - `day6-typed-analysis-reorder-option-batch1.md`
  - `day7-typed-analysis-reorder-option-batch2.md`
  - `day8-post-landing-analysis-postorder-audit.md`
  - `day9-analysis-postorder-integration-design.md`
  - `day10-analysis-postorder-integration-batch.md`
  - `day11-compatibility-layer-and-regression-sweep.md`
- compatibility/docs/validation/closeout artifacts:
  - `day12-docs-and-maintainer-story-update.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`
  - `day3-env-var-surface-inventory.md`

### Sprint 61 landed Phase 1 package

| Metric | Sprint 61 close state |
|---|---:|
| public typed analysis/reorder controls landed | `8` |
| internal-policy-only controls formalized | `1` |
| targeted Day 13 follow-on commands rerun | `19` |
| explicitly deferred later control lanes | `4` |

Notes:

- public typed analysis/reorder controls landed:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- internal-policy-only controls formalized:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_graph`
  - `./build/test_graph_fm_buckets`
  - `./build/test_reorder_nd`
  - `./build/test_reorder_amd_qg`
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_analysis`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- explicitly deferred later control lanes:
  - `SPARSE_FM_*`
  - debug/profile controls
  - compatibility-only alias cleanup
  - broader configuration/policy rationalization outside the Sprint 61 seam

## Residual Deferred Debt

Sprint 61 was explicitly about the highest-value Phase 1 analysis/reorder
control seam. The main open work it intentionally hands forward is:

- later FM-family control rationalization
- later debug/profile control treatment
- optional compatibility-only alias cleanup if later justified
- later broader configuration/policy work outside the Sprint 61 analysis/reorder seam
- later direct-solver usability and lifecycle coherence work from the wider Epic 6 queue
- later backend/AUTO policy rationalization from the wider Epic 6 queue

Still consciously constrained rather than silently “solved”:

- no public `SPARSE_FM_*` tuning surface yet
- no debug/profile migration into the public API
- no repo-wide configuration helper layer
- no backend/AUTO rewrite in Sprint 61
- no reopening of the repeated-run workflow fence

Not carried forward as unresolved Sprint 61 debt:

- missing typed analysis/reorder front door
- missing explicit precedence contract
- missing bounded env-var compatibility behavior
- missing graph/reorder-sensitive precedence regression coverage
- missing validated Phase 1 configuration handoff

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-env-var-surface-inventory.md](./artifacts/day3-env-var-surface-inventory.md)
- [day4-typed-options-design-and-precedence-contract.md](./artifacts/day4-typed-options-design-and-precedence-contract.md)
- [day5-header-and-internal-surface-landing-design.md](./artifacts/day5-header-and-internal-surface-landing-design.md)
- [day6-typed-analysis-reorder-option-batch1.md](./artifacts/day6-typed-analysis-reorder-option-batch1.md)
- [day7-typed-analysis-reorder-option-batch2.md](./artifacts/day7-typed-analysis-reorder-option-batch2.md)
- [day8-post-landing-analysis-postorder-audit.md](./artifacts/day8-post-landing-analysis-postorder-audit.md)
- [day9-analysis-postorder-integration-design.md](./artifacts/day9-analysis-postorder-integration-design.md)
- [day10-analysis-postorder-integration-batch.md](./artifacts/day10-analysis-postorder-integration-batch.md)
- [day11-compatibility-layer-and-regression-sweep.md](./artifacts/day11-compatibility-layer-and-regression-sweep.md)
- [day12-docs-and-maintainer-story-update.md](./artifacts/day12-docs-and-maintainer-story-update.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 61 achieved its goal:

- the highest-value analysis/reorder env-var seam now has a real typed public front door
- precedence between typed values, compatibility env vars, and internal defaults is explicit
- the remaining env-var surface is smaller and more intentionally classified
- the graph/reorder proof home now carries explicit precedence/default regression coverage
- the branch closed from a fully validated reviewed baseline with exact preserved parity anchors

Epic 6 can now move into later configuration work and broader productization
work from a branch that has already shipped one bounded, coherent, and tested
Phase 1 modernization step instead of still relying on environment-variable
policy as the only advanced control surface.
