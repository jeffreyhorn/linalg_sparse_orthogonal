# Sprint 71 Retrospective

**Sprint:** 71 — Public Surface De-chronologization & Reference Cleanup  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 71 scope and docs-only validation baseline were captured before cleanup design or landing work began
- [x] the strongest public-surface history and policy-density seams were re-ranked from the live repo before the first cleanup boundary was fixed
- [x] the first landing fence was kept bounded to `README.md` and `INSTALL.md` before support-surface or header cleanup widened
- [x] the compact front-door and install-contract cleanup batch landed without widening product, benchmark, or platform claims
- [x] the strongest deferred reference surface was narrowed to `include/sparse_cholesky.h` and cleaned as an API-local reference lane
- [x] tutorial/example/benchmark support-surface reconciliation landed without reopening policy ownership or proof-owner surfaces
- [x] the maintainer-guide authority split was rechecked against all touched public/reference/support surfaces
- [x] the full Sprint 71 package was re-read for coherence against the Sprint 70 architecture contract, the Sprint 71 project-plan scope, and live repo wording
- [x] Sprint 71 closed with one explicit cleaned public/reference package and a ranked Sprint 72 carry-forward queue

## What Went Well

1. **Sprint 71 stayed tightly scoped as a permanent-surface cleanup sprint.**
   The branch improved maturity and readability without drifting into:
   - implementation work
   - product-model redesign
   - capability widening
   - platform or packaging claim inflation
   That discipline kept the sprint aligned with the Sprint 70 architecture fence.

2. **The strongest public contradiction centers were actually reduced, not just re-labeled.**
   Sprint 71 landed meaningful cleanup on:
   - `README.md`
   - `INSTALL.md`
   - `include/sparse_cholesky.h`
   - `docs/tutorial.md`
   - `examples/README.md`
   while correctly leaving already-coherent or policy-authority surfaces alone.

3. **The audience split across public docs, support docs, headers, benchmarks, and policy surfaces is now clearer.**
   The branch left a cleaner stable reading:
   - `README.md` = compact front door
   - `INSTALL.md` = operator/install-contract guide
   - `docs/tutorial.md` = step-by-step teaching flow
   - `examples/README.md` = adoption/workflow teaching
   - `benchmarks/README.md` = retained workflow/performance proof
   - `docs/maintainer_guide.md` = policy authority
   - `include/sparse_cholesky.h` = API-local call-site truth

4. **Sprint 71 improved user-facing maturity without weakening truthfulness.**
   The sprint preserved:
   - examples not replacing regression/oracle/property owners
   - benchmarks not replacing test-owned guarantees
   - `make bench-canonical-report` as threshold-free artifact reporting
   - the static-first release/install story
   - Windows as the reviewed CMake-first consumer story rather than a reviewed install-validation lane

5. **The package now reads more like durable product/reference documentation and less like sprint-delivery history.**
   That matters for Epic 7 because later implementation work can now start from:
   - a cleaner front door
   - a cleaner install guide
   - a cleaner high-signal public header
   - a cleaner tutorial/examples/benchmarks support split
   instead of carrying more chronology cleanup pressure into deeper product-model work.

6. **The project plan survived the deeper cleanup without needing correction.**
   Rechecks against [PROJECT_PLAN.md](../PROJECT_PLAN.md) confirmed the Sprint 71 scope still matches the live branch outcome and that Sprint 72 can begin from the intended product-model queue rather than from another docs-only follow-through sprint.

## What Didn't Go Well

1. **Sprint 71 was intentionally non-functional work, so the value is readability and contract quality rather than new capability.**
   That is the right outcome for this sprint, but it means the branch did not yet reduce:
   - direct-workflow mutation/copy burden
   - residual env-var/control debt
   - `int32_t` capability limits
   - backend/performance maturity gaps
   - platform/release asymmetry

2. **The strongest implementation-facing Epic 7 problems are now more exposed, not smaller.**
   Sprint 71 cleared public/reference drag out of the way, which makes the next unresolved seams more obvious:
   - product-model convergence
   - configuration modernization
   - capability modernization led by index width
   - benchmark-governed backend/performance work

3. **Some large permanent surfaces remain intentionally untouched because they were not the highest-value Sprint 71 centers.**
   Sprint 71 did not attempt a broad cleanup wave across:
   - other public headers
   - implementation files
   - proof-owner tests
   - workflow files
   That restraint was correct, but it means some chronology or density pressure still exists on later-ranked surfaces.

4. **Platform/release confidence remains bounded by the same reviewed evidence fence Sprint 70 fixed.**
   Sprint 71 correctly did not claim:
   - broader reviewed install-validation parity
   - wider Windows proof coverage
   - shared-library maturity
   - benchmark timings as pass/fail portability gates
   That honesty is a strength, but it also highlights how much Epic 7 still depends on disciplined contract wording.

## Final Metrics

### Baseline and contract anchors

| Metric | Sprint 71 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` anchor | `53` |
| project-plan correction required | `0` |
| implementation files touched in Sprint 71 closeout package | `0` |

### Sprint 71 artifact package

| Metric | Sprint 71 close state |
|---|---:|
| total artifact files under `SPRINT_71/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `5` |
| review/closeout artifacts | `4` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-public-surface-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-truth-surface-recheck.md`
  - `day3-public-surface-history-audit.md`
  - `day4-first-landing-boundary.md`
  - `day7-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-front-door-and-install-cleanup-design.md`
  - `day6-front-door-and-install-cleanup-batch.md`
  - `day8-public-header-narrative-cleanup-design.md`
  - `day9-public-header-narrative-cleanup-batch.md`
  - `day10-support-surface-reconciliation-design.md`
- review/closeout artifacts:
  - `day11-support-surface-reconciliation-batch.md`
  - `day12-truth-surface-review-and-closeout-queue.md`
  - `day13-package-coherence-review-and-handoff-queue.md`
  - `day14-closeout-and-handoff.md`

### Sprint 71 cleanup package

| Metric | Sprint 71 close state |
|---|---:|
| primary permanent-surface centers materially cleaned | `5` |
| preserved audience/authority surfaces in final package | `7` |
| explicit stable truth readings revalidated at close | `5` |
| ranked carry-forward items fixed for Sprint 72+ | `5` |

Notes:

- primary permanent-surface centers materially cleaned:
  - `README.md`
  - `INSTALL.md`
  - `include/sparse_cholesky.h`
  - `docs/tutorial.md`
  - `examples/README.md`
- preserved audience/authority surfaces in final package:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `include/sparse_cholesky.h`
- explicit stable truth readings revalidated at close:
  - examples do not replace regression/oracle/property owners
  - benchmarks do not replace test-owned guarantees
  - `make bench-canonical-report` remains threshold-free artifact reporting
  - maintained release shape remains static-first
  - Windows remains the reviewed CMake-first consumer story

## Residual Deferred Debt

Sprint 71 was about making the permanent public/reference surfaces cleaner and
more durable, not about reopening deeper Epic 7 implementation work. The main
open work it intentionally hands forward is:

- Sprint 72 product-model convergence from the public direct-workflow seam
- configuration modernization only where remaining env-var/default-policy seams still carry real ownership cost
- capability modernization led by index width, with scalar breadth later
- benchmark-governed backend/performance maturity without widening product or platform claims
- later permanent-surface cleanup only where future implementation work moves ownership again

Still consciously constrained rather than silently “solved”:

- no broad implementation rewrite hidden behind documentation cleanup
- no benchmark/test/platform claim widening without new evidence
- no repo-wide public-header cleanup wave detached from ranked contradiction centers
- no reinterpretation of canonical benchmark reporting as a timing gate
- no broader reviewed install-validation or Windows parity claims than the current evidence supports

Not carried forward as unresolved Sprint 71 debt:

- the public-surface history audit and first cleanup fence
- the front-door and install cleanup batch
- the public-header narrative cleanup on `include/sparse_cholesky.h`
- the tutorial/example support reconciliation batch
- the maintainer-guide authority recheck
- the full truth-surface review
- the package coherence review against Sprint 70 and the Sprint 71 project-plan scope
- the Sprint 71 closeout and ranked Sprint 72 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-public-surface-baseline.md](./artifacts/day1-scope-and-public-surface-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-truth-surface-recheck.md](./artifacts/day2-validation-baseline-and-truth-surface-recheck.md)
- [day3-public-surface-history-audit.md](./artifacts/day3-public-surface-history-audit.md)
- [day4-first-landing-boundary.md](./artifacts/day4-first-landing-boundary.md)
- [day5-front-door-and-install-cleanup-design.md](./artifacts/day5-front-door-and-install-cleanup-design.md)
- [day6-front-door-and-install-cleanup-batch.md](./artifacts/day6-front-door-and-install-cleanup-batch.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-public-header-narrative-cleanup-design.md](./artifacts/day8-public-header-narrative-cleanup-design.md)
- [day9-public-header-narrative-cleanup-batch.md](./artifacts/day9-public-header-narrative-cleanup-batch.md)
- [day10-support-surface-reconciliation-design.md](./artifacts/day10-support-surface-reconciliation-design.md)
- [day11-support-surface-reconciliation-batch.md](./artifacts/day11-support-surface-reconciliation-batch.md)
- [day12-truth-surface-review-and-closeout-queue.md](./artifacts/day12-truth-surface-review-and-closeout-queue.md)
- [day13-package-coherence-review-and-handoff-queue.md](./artifacts/day13-package-coherence-review-and-handoff-queue.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom-Line Closeout

Sprint 71 succeeded because it reduced the strongest remaining chronology and
policy spill from the highest-value permanent public/reference surfaces
without weakening the Sprint 70 architecture contract or the repo’s
truthfulness fence.

The branch now closes with:

- one cleaner front door
- one cleaner install/operator surface
- one cleaner high-signal public header
- one cleaner tutorial/example/benchmark support split
- one revalidated authority/truth-surface package
- one ranked implementation-facing carry-forward queue for Sprint 72 and beyond

That is the right Sprint 71 outcome: Sprint 72 can now start from a cleaner,
more durable public package and spend its energy on product-model convergence
instead of on more permanent-surface repair.
