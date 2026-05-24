# Sprint 40 Retrospective

**Sprint:** 40 — Baseline, Lifecycle Inventory & Epic 4 Architecture Contract  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 40 baseline and scope captured before architecture work
- [x] reviewed-quality baseline recorded before ownership mapping
- [x] reviewed CMake / dead-code / platform support baseline recorded
- [x] hotspot and allocation-density inventory completed
- [x] first lifecycle inventory completed for LU / Cholesky / LDLT
- [x] second lifecycle inventory completed for QR / SVD / analysis / iterative / eigs
- [x] stable lifecycle/state taxonomy written
- [x] lifecycle precondition / mutation / cancellation contract map written
- [x] first future handle-model design note completed
- [x] second handle-model note and staged migration contract completed
- [x] quality-contract ownership map completed
- [x] public migration-risk audit completed
- [x] validation anchor and change-type command matrix completed
- [x] Sprint 40 architecture-contract synthesis completed

## What Went Well

1. **The sprint stayed architecture-first instead of drifting into premature implementation.**
   Sprint 40 kept its scope disciplined: baseline, taxonomy, contract mapping,
   handle-model design, migration strategy, ownership mapping, migration-risk
   ranking, and validation anchoring all landed before any heavy Epic 4 code
   churn started.

2. **The lifecycle problem is now explicit instead of anecdotal.** Days 5-8
   turned the hidden `SparseMatrix` state story into a concrete model:
   - five lifecycle classes
   - four caller-visible contract axes
   - a measured LU/Cholesky vs LDLT/QR/SVD/analysis split

3. **The future handle model now has a real migration contract.** Days 9-10
   moved beyond “explicit handles would be cleaner” and produced:
   - target object families
   - keep/move ownership split
   - staged migration phases
   - earliest safe insertion points
   - explicit high-risk migration edges

4. **Later cleanup work now has stable ownership rules.** Day 11 clarified
   where quality truth should live:
   - `Makefile` for commands
   - scripts for machine behavior
   - workflow YAML for CI matrix truth
   - Sprint 30 docs / executable policy for maintainer authority
   - `README.md` for concise operator summaries

5. **The sprint reduced later migration uncertainty materially.** Days 12-14
   now give Sprint 41+:
   - a ranked public migration-risk map
   - an internal-only vs public-facing refactor boundary
   - a change-type validation matrix
   - an explicit Sprint 41 / Sprint 42 handoff contract

## What Didn't Go Well

1. **Sprint 40 did not end with fresh end-to-end runtime timings.** That was
   the honest outcome for a docs/architecture sprint, but it means the main
   quantitative validation evidence remains the inherited Epic 3 baseline
   rather than a brand-new full code sweep inside Sprint 40 itself.

2. **The architectural target is clearer than the eventual public migration
   story.** The sprint successfully bounded the migration-risk surfaces, but it
   also confirmed that README/tutorial/examples/headers will need careful
   coordination later once explicit handles or workspaces become user-visible.

3. **The strongest inherited operational caveats remain real.** Sprint 40
   correctly preserved rather than “fixing on paper” several constraints:
   - dead-code remains serialized
   - `quality-review-full` remains the strongest local reviewed baseline
   - cross-platform reviewed parity remains intentionally asymmetric

## Final Metrics

### Preserved baseline invariants

| Metric | Sprint 40 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| dead-code benchmark/example compile-db gap | `0` |
| authoritative dead-code execution mode | `serial` |

### Sprint 40 artifact package

| Metric | Sprint 40 close state |
|---|---:|
| total artifact files under `SPRINT_40/artifacts/` | `17` |
| lifecycle inventory artifacts | `2` |
| handle-model / migration-strategy artifacts | `2` |
| ownership / migration-risk / validation / synthesis artifacts | `4` |

### Architectural handoff outputs

| Metric | Sprint 40 close state |
|---|---:|
| lifecycle taxonomy classes | `5` |
| caller-visible lifecycle contract axes | `4` |
| target future object families | `4` |
| staged migration phases | `4` |

## Residual Deferred Debt

Sprint 40 was designed to define architecture, not to consume Epic 4
implementation debt. The main open work it intentionally hands forward is:

- Sprint 41:
  - bounded internal-first groundwork
  - helper/allocation and hotspot-prep work under the new ownership and
    validation contracts
- Sprint 42:
  - first lifecycle-handle landing work
  - especially LU / Cholesky internal payload separation
  - bridge normalization around `sparse_factors_t`
- later public-facing phases:
  - README/tutorial/example/header reconciliation once explicit handles or
    workspaces become user-visible

Not carried forward as unresolved Sprint 40 design debt:

- missing lifecycle taxonomy
- unclear migration sequence
- unclear ownership model for quality truth
- unclear validation baseline for implementation sprints
- unclear boundary between internal-only work and public migration risk

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day7-state-model-taxonomy.md](./artifacts/day7-state-model-taxonomy.md)
- [day8-lifecycle-contract-map.md](./artifacts/day8-lifecycle-contract-map.md)
- [day9-handle-model-design-1.md](./artifacts/day9-handle-model-design-1.md)
- [day10-handle-model-design-2-and-migration-strategy.md](./artifacts/day10-handle-model-design-2-and-migration-strategy.md)
- [day11-quality-contract-ownership-map.md](./artifacts/day11-quality-contract-ownership-map.md)
- [day12-public-migration-risk-audit.md](./artifacts/day12-public-migration-risk-audit.md)
- [day13-validation-anchor-and-command-matrix.md](./artifacts/day13-validation-anchor-and-command-matrix.md)
- [day14-architecture-contract-synthesis.md](./artifacts/day14-architecture-contract-synthesis.md)

## Bottom Line

Sprint 40 achieved its goal:

- Epic 4 now starts from a measured preserved baseline instead of vague
  follow-on work
- the matrix/factor lifecycle problem has a stable taxonomy and contract map
- the future handle model has a staged migration strategy
- quality-truth ownership is explicit
- public migration risk is ranked
- later implementation sprints now have a concrete validation contract

Sprint 41 and Sprint 42 can now begin implementation-heavy Epic 4 work without
reopening the architecture-definition problem first.
