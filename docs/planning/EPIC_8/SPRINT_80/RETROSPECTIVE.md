# Sprint 80 Retrospective

**Sprint:** 80 — Epic 8 Baseline, Competitive Target & External Oracle Contract  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 80 fixed the Epic 8 baseline, proof split, and validation
      contract before any implementation-sprint work began
- [x] the strongest live Epic 8 contradiction map was reranked from the current
      tree rather than inherited blindly from Epic 7
- [x] Sprint 80 reduced the external-comparison question to one bounded
      maintained contract centered on:
  - CHOLMOD-class SPD direct-solver correctness comparison first
  - BLAS/LAPACK-class dense-kernel performance-reference support
- [x] Sprint 80 preserved the benchmark-governance fence:
  - canonical reporting remains threshold-free
  - `bench-fast` remains the bounded runtime lane
  - `wall-check` remains the narrow thresholded regression gate
- [x] Sprint 80 fixed one explicit non-goal and risk fence for the whole epic:
  - no fake state-of-the-art claim inflation
  - no fake platform parity
  - no external dependency sprawl without proof
  - no canonical benchmark timing-gate inflation
- [x] Sprint 80 correctly used explicit no-op outcomes where the tree already
      stayed truthful:
  - `PROJECT_PLAN.md` did not need correction
  - the main support surfaces did not need churn
- [x] Sprint 80 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 80 closed with one explicit Sprint 81-first handoff queue rather
      than another generic Epic 8 summary

## What Went Well

1. **Sprint 80 established a real execution contract instead of starting implementation too early.**
   The branch spent its time on the highest-value setup work:
   - baseline recheck
   - live competitive gap inventory
   - external-oracle contract
   - benchmark/performance contract
   - explicit non-goal and risk fence

2. **The sprint kept the external-oracle lane bounded and credible.**
   It did not widen into “compare against everything” theater. The maintained
   first lane is now clearly:
   - CHOLMOD-class SPD direct-solver correctness comparison
   with:
   - BLAS/LAPACK-class dense calibration as performance-reference support

3. **The benchmark-governance story stayed disciplined.**
   Sprint 80 kept:
   - canonical reporting threshold-free
   - runtime/profiling signals separate from proof
   - tests as the owners of regression/oracle truth
   That prevents backend work in later sprints from inflating its claims.

4. **The sprint used no-op decisions correctly.**
   Day 10 and Day 11 both closed as explicit no-ops rather than forced edits:
   - `docs/planning/EPIC_8/PROJECT_PLAN.md` already stayed truthful
   - `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`,
     `benchmarks/README.md`, and the workflow notes already reconciled cleanly

5. **The final validation baseline is strong and current.**
   Sprint 80 did not stop at design prose. It closed from a full Day 13 sweep
   with exact retained anchors for:
   - reviewed CMake parity
   - Makefile/CMake parity
   - focused proof-owner reruns
   - canonical benchmark reporting
   - install/export proof

6. **Sprint 81 now has a cleaner starting point.**
   The branch ended with one explicit handoff order:
   - Sprint 81 storage/product first
   - Sprint 82 backend second
   - Sprint 83 capability widening third

## What Didn't Go Well

1. **Sprint 80 intentionally delivered no implementation progress on the core product gaps.**
   That was the right bounded choice, but it means the actual state-of-the-art
   gap remains fully open across:
   - linked-list-first storage/product cost
   - builtin scalar dense/backend ceiling
   - bounded capability surface

2. **The external-oracle lane is still contract-only.**
   Sprint 80 clarified what Epic 8 should compare against, but it did not yet
   land:
   - maintained CHOLMOD-backed differential proof
   - backend-aware external calibration code

3. **The validation sweep exposed a transient canonical-report hiccup.**
   The first standalone `make bench-canonical-report` rerun hit a
   non-reproducing `bench_eigs_reuse.csv` write failure. The immediate clean
   rerun succeeded without edits, so the branch still closed honestly, but the
   transient behavior is worth preserving in the record.

4. **The reviewed runtime long pole remains large.**
   `test_reorder_nd` still dominated the reviewed CTest runtime. Sprint 80
   closed cleanly, but it did not reduce that operational drag.

5. **Success depended on resisting low-value churn.**
   Sprint 80 worked because it refused to widen into:
   - fake platform or packaging claim broadening
   - benchmark-threshold policy churn
   - generic review-package rewriting
   That discipline held, but the implementation sprints still carry the real
   hard work.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 80 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `642.39 sec` |
| reviewed `test_reorder_nd` time | `486.38 sec` |
| focused `test_chol_csc` follow-on | `147 / 147` |
| focused `test_ldlt_csc` follow-on | `96 / 96` |
| focused `test_ldlt` follow-on | `84 / 84` |
| focused `test_iterative` follow-on | `80 / 80` |
| focused `test_qr` follow-on | `72 / 72` |
| focused `test_integration` follow-on | `51 / 51` |
| focused `test_reorder_nd` follow-on | `35 / 35` |
| focused `test_fuzz` follow-on | `26 / 26` |
| focused `test_eigs` follow-on | `31 / 31` |
| install regression | `11 / 11` |
| CMake install/export regression | `13 / 13` |

### Sprint 80 artifact package

| Metric | Sprint 80 close state |
|---|---:|
| total artifact files under `SPRINT_80/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| contract/refinement artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-epic8-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-live-competitive-gap-inventory.md`
  - `day4-external-oracle-candidate-audit.md`
  - `day11-support-surface-truth-sweep.md`
- contract/refinement artifacts:
  - `day5-external-oracle-contract.md`
  - `day6-performance-and-benchmark-contract.md`
  - `day7-non-goal-and-risk-fence.md`
  - `day8-review-readiness-refinement.md`
  - `day9-closure-sequence-refinement.md`
  - `day10-project-plan-reconciliation.md`
  - `day12-final-proof-alignment-and-validation-queue.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed closeout package

| Metric | Sprint 80 close state |
|---|---:|
| implementation files touched | `0` |
| proof-owner test files touched | `0` |
| support/policy docs touched in landed batches | `0` |
| review/todo surfaces touched | `2` |
| project-plan corrections required | `0` |

Notes:

- review/todo surfaces touched:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`
- intentionally untouched after rerank/recheck:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - workflow YAML surfaces

## Residual Deferred Debt

Sprint 80 deliberately stopped after the highest-value baseline-and-contract
package. The main open work it hands forward is:

- linked-list-first product/storage modernization
- bounded optional dense-backend acceleration
- capability-surface widening on the highest-value seams
- maintained external differential proof using the bounded contract frozen here
- later maintainability, runtime, package/platform, usability, and final
  comparison work in the Epic 8 order

Still consciously constrained rather than silently “solved”:

- no fake state-of-the-art claim inflation
- no fake platform or install parity claim broadening
- no shared-library maturity claim without proof
- no canonical benchmark timing-gate conversion
- no broad external dependency matrix

Not carried forward as unresolved Sprint 80 debt:

- the baseline recheck
- the competitive gap inventory rerank
- the bounded external-oracle contract
- the benchmark/performance contract recheck
- the explicit non-goal and risk fence
- the review/todo refinement
- the no-op project-plan and support-surface rechecks
- the full Day 13 validation sweep
- the explicit Day 14 handoff queue

## Key Deliverables

1. **One refreshed Epic 8 starting baseline was fixed in writing.**
   Sprint 80 now has:
   - one reviewed parity anchor
   - one proof-owner split
   - one authoritative rerun set

2. **One ranked live contradiction map replaced the generic Epic 8 wish list.**
   The sprint fixed the current order:
   - storage/product ceiling first
   - backend ceiling second
   - capability ceiling third

3. **One bounded external comparison contract landed.**
   The maintained first lane is now:
   - CHOLMOD-class SPD direct-solver correctness comparison
   with:
   - BLAS/LAPACK-class dense calibration as bounded performance-reference
     support

4. **One bounded benchmark/performance-claim fence landed.**
   Sprint 80 preserved:
   - threshold-free canonical benchmark reporting
   - a bounded runtime lane in `bench-fast`
   - a narrow threshold gate in `wall-check`

5. **Sprint 80 closed from a measured baseline, not just from planning prose.**
   The branch ended with a full Day 13 validation sweep plus retained
   benchmark/install/report anchors and an explicit Day 14 handoff queue.

## Bottom-Line Closeout

Sprint 80 succeeded because it did not confuse “start of Epic 8” with “start
coding everything.” It froze the real baseline, fixed the competitive target,
bounded the external-oracle and benchmark claim model, preserved the non-goal
fence, and closed from a validated rerun set. That gives Sprint 81 a truthful,
measurable starting contract instead of another planning reset.
