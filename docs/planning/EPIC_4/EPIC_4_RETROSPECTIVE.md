# Epic 4 Retrospective — Sprints 40-49 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × ~124-152 h each = **~1 384 hours nominal**  
**Branch range:** `sprint-40` → `sprint-49`  
**Started:** Sprint 40 (review-driven structural-refinement kickoff after
`reviews/review-codex-2026-05-21.md`)  
**Closed:** Sprint 49 Day 14 (Epic 4 closeout)  
**Goal:** Close the structural-refinement backlog from
`reviews/review-codex-2026-05-21.md` +
`reviews/todo-codex-2026-05-21.md` across lifecycle/state modeling, shared
allocation and overflow safety, graph/ND decomposition, iterative and
eigensolver repeated-run efficiency, auxiliary benchmark/tooling cleanup,
documentation/quality-contract ownership, bounded public lifecycle exposure,
and final Epic 4 integration/validation.

> **Status: Epic 4 complete.** All 10 sprints landed. The epic opened by
> converting the review findings into a measured architecture and validation
> contract (Sprint 40), then closed each major structural queue in order:
> shared allocation/overflow safety (Sprint 41), internal lifecycle/state
> scaffolding (Sprint 42), graph/ND decomposition (Sprints 43-44), reusable
> iterative and eigensolver repeated-run state (Sprints 45-46), auxiliary
> benchmark/example/tooling modernization (Sprint 47), documentation and
> quality-contract simplification (Sprint 48), and bounded public repeated-run
> lifecycle handles plus final compatibility/validation closeout (Sprint 49).
> Epic 4 ends with the strongest local reviewed baseline still intact
> (`make quality-review-full`), reviewed CMake parity still exact (`53` vs
> `53`, `53 / 53` passing), and all original top-level review findings given an
> explicit final disposition.

---

## Summary table

Per-sprint deliverables + nominal hours from `PROJECT_PLAN.md`. Epic 4 sprint
retrospectives and closeout artifacts record measured outcomes and validation
results, but they do not track separate actual-hour totals, so the `actual h`
column is left `n/t`.

| sprint | title | plan assessment | key deliverables | nominal h | actual h |
|---|---|---|---|---:|---:|
| 40 | Baseline, Lifecycle Inventory & Epic 4 Architecture Contract | Met exactly | measured baseline, lifecycle taxonomy, handle-model design, quality-ownership map, validation anchor | 124 | n/t |
| 41 | Shared Allocation, Overflow & Internal Utility Consolidation | Met exactly | shared internal allocation/overflow helper layer, first-wave and broader `src/` migrations, internal-first prep rules | 136 | n/t |
| 42 | Matrix Lifecycle Refactor Phase 1 - Internal Handles & Compatibility Scaffolding | Met with bounded internal-first scope | internal lifecycle seam, shared matrix-state guards, factor-path normalization, cancellation contract cleanup, focused misuse tests | 144 | n/t |
| 43 | Graph / ND Subsystem Decomposition Phase 1 | Met exactly | graph ownership/core extraction, hierarchy/coarsening extraction, coarse-bisection extraction, seam tests | 152 | n/t |
| 44 | Graph / ND Subsystem Decomposition Phase 2 & Large-Test Maintainability Batch | Met exactly | FM extraction, separator extraction, orchestration cleanup, first large-test helper proof point | 148 | n/t |
| 45 | Iterative Solver Workspace Reuse & Repeated-Solve Efficiency | Met exactly | shared iterative workspace layer, CG/GMRES and block-CG migration, repeated-solve benchmark evidence | 140 | n/t |
| 46 | Eigensolver Workspace Reuse & Advanced Repeated-Run Efficiency | Met exactly | shared eigensolver workspace/state layer, Lanczos/thick-restart/LOBPCG migration, repeated-run benchmark evidence | 148 | n/t |
| 47 | Benchmark CLI Modernization, Auxiliary Surface Safety & Example/Tooling Cleanup | Met exactly | shared benchmark CLI parsing helpers, `bench_main` modernization, reorder parity, bounded example/tooling hardening | 136 | n/t |
| 48 | Quality-Contract Simplification, README Reduction & Maintainer Guide | Met exactly | maintainer guide, README reduction, tutorial/header cross-reference cleanup, clearer quality-contract ownership | 124 | n/t |
| 49 | Lifecycle API Exposure, Final Integration, Validation & Epic 4 Closeout | Met exactly | bounded public repeated-run handles, wrapper integration, benchmark/test/docs agreement, final residual review and validation | 132 | n/t |
| **Total** | | | | **1 384** | **n/t** |

Epic 4 tracking stayed outcome- and validation-oriented rather than hour-burn
or velocity-oriented. The planning baseline therefore remains the authoritative
epic hour total.

---

## Cumulative metrics

### Epic-completion metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| nominal planned hours | `1 384` |
| original top-level review findings with explicit final disposition | `7 / 7` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake parity target | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Structural-refinement trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 40-42 | architecture contract, helper consolidation, lifecycle groundwork | baseline and ownership model defined, shared internal safety seam landed, first internal lifecycle/state scaffolding landed |
| 43-44 | graph / ND decomposition | `src/sparse_graph.c` Phase-1 and Phase-2 split into owned modules: `sparse_graph_core.c`, `sparse_graph_coarsen.c`, `sparse_graph_bisect.c`, `sparse_graph_refine.c`, `sparse_graph_separator.c`, plus a narrower orchestration residual |
| 45-46 | repeated-run efficiency | internal reusable iterative and eigensolver workspace/state models landed with direct repeated-run benchmark evidence |
| 47-48 | auxiliary surface + docs/quality ownership cleanup | benchmark CLI/tooling/example safety improved and maintainer-policy / README ownership clarified |
| 49 | final public integration | bounded public repeated-run lifecycle handles landed for iterative and eigensolver workloads, with direct tests and benchmark alignment |

### Final validation anchor (Sprint 49 Day 13)

| metric | final |
|---|---:|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `make quality-review-full` | passed |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `414.75 sec` |

### Public repeated-run close state

| metric | final |
|---|---:|
| public repeated-run handle families exposed | `2` |
| public headers extended directly | `2` |
| repeated-run benchmark drivers aligned to public handle path | `2` |
| direct public-handle regression binaries rerun in Sprint 49 Day 13 | `3` |

Representative final repeated-run evidence from Sprint 49 Day 13:

- iterative:
  - CG `90.1570 ms` one-shot vs `88.5640 ms` reuse, `1.02x`
  - GMRES `95.9320 ms` one-shot vs `87.5930 ms` reuse, `1.10x`
- eigensolver:
  - grow-m `4.0260 ms` one-shot vs `3.5480 ms` reuse, `1.13x`
  - thick-restart `125.2890 ms` one-shot vs `124.6060 ms` reuse, `1.01x`

These runs stayed parity-stable on convergence behavior, iteration counts,
residuals, and eigensolver values.

---

## Stable contracts landed across Epic 4

| sprint | contract | rationale |
|---|---|---|
| 40 | measured baseline + lifecycle taxonomy + validation anchor | later refactors could compare against stable truth instead of inferred intent |
| 41 | shared internal allocation/overflow helper seam | later hotspot refactors stopped re-inventing size/overflow arithmetic locally |
| 42 | internal-first lifecycle/state refactor rule | state/model cleanup could advance without premature public API churn |
| 43-44 | graph subsystem ownership split | graph/ND work moved from monolith management to owned module seams |
| 45-46 | reusable internal workspace/state model with one-shot wrapper preservation | repeated-run efficiency landed without breaking the simple caller path |
| 47 | safer benchmark CLI and auxiliary-surface contract | benchmark/example/tooling surfaces reached the same usability/safety bar as core work |
| 48 | maintainer guide + clearer README/operator boundary | maintainer policy gained a stable home and the README became more operator-facing |
| 49 | bounded public repeated-run lifecycle handles | Epic 4 exposed the repeated-run benefit publicly without publishing raw internal workspace layout |

---

## Residual final limits / follow-up journal

Epic 4 closes without an unfinished top-level remediation sprint, but it does
intentionally leave a few bounded limits in the stable post-epic contract:

1. **Compatibility-facing mutable `SparseMatrix` / in-place factor APIs remain.**
   This is the main explicit accepted tradeoff coming out of the epic.

2. **Public repeated-run exposure is intentionally bounded.**
   Epic 4 exposes handle-based repeated-run support for the main iterative and
   eigensolver paths, not every internal workspace/view or a broader public
   factor-handle redesign.

3. **Examples and tutorial material remain one-shot-first.**
   That is now an explicit documentation choice, not a hidden contradiction in
   the public API story.

4. **Benchmark-framework redesign remains out of scope.**
   Sprint 47 and Sprint 49 modernized the touched benchmark surfaces without
   turning Epic 4 into a larger framework rewrite.

These are future-facing boundaries, not hidden Epic 4 closeout defects.

---

## Lessons (Epic-level)

- **Architecture-first sequencing paid off.** Sprint 40 made the later
  implementation sprints smaller, more attributable, and less likely to reopen
  foundational ownership questions.

- **Internal-first refactoring was the correct bridge to public exposure.**
  Sprint 42, Sprint 45, and Sprint 46 proved the lifecycle/workspace seams
  before Sprint 49 exposed a bounded public handle model.

- **Bounded decomposition beats heroic rewrites.** Sprint 43 and Sprint 44 did
  not try to erase `src/sparse_graph.c` in one move; they split it in phases
  and kept the orchestration residual honest.

- **Behavior-level validation preserved trust through structural churn.**
  Epic 4 changed file ownership and runtime structure heavily while keeping the
  strongest local reviewed baseline and reviewed CMake parity stable the whole
  way through.

- **Documentation ownership matters as much as code ownership in long refactor
  epics.** Sprint 48 made later closeout work easier because maintainer policy,
  user/operator guidance, and local executable truth no longer competed for the
  same space.

- **Benchmark/test/docs agreement is a real closeout criterion.** Sprint 49 was
  not complete when the public handles first compiled; it was complete when the
  benchmark, regression, and README surfaces all agreed on the final caller
  story.

---

## DoD verification

Required cross-epic themes from `PROJECT_PLAN.md`:

- measured Epic 4 baseline, lifecycle inventory, ownership contract, and
  validation anchor: ✓
- shared internal allocation/overflow helper layer plus broad hotspot
  migration: ✓
- internal lifecycle/state scaffolding and compatibility-preserving guard
  normalization: ✓
- graph / ND subsystem Phase-1 and Phase-2 decomposition: ✓
- reusable iterative repeated-run workspace/state support with benchmark
  evidence: ✓
- reusable eigensolver repeated-run workspace/state support with benchmark
  evidence: ✓
- benchmark CLI, auxiliary tooling, and example safety modernization: ✓
- quality-contract simplification, README reduction, and maintainer guide: ✓
- bounded public repeated-run lifecycle exposure for iterative and eigensolver
  workloads: ✓
- final compatibility sweep, residual review, validation closeout, and epic
  handoff: ✓

Final measured close state:

- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- no original top-level Epic 4 review finding remains unintentionally deferred:
  ✓
- no new cleanup sprint was required to explain away unfinished Epic 4 work: ✓

---

## Key references

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [review-codex-2026-05-21.md](./reviews/review-codex-2026-05-21.md)
- [todo-codex-2026-05-21.md](./reviews/todo-codex-2026-05-21.md)
- [SPRINT_40/RETROSPECTIVE.md](./SPRINT_40/RETROSPECTIVE.md)
- [SPRINT_41/day14-closeout-and-handoff.md](./SPRINT_41/artifacts/day14-closeout-and-handoff.md)
- [SPRINT_49/RETROSPECTIVE.md](./SPRINT_49/RETROSPECTIVE.md)
- [SPRINT_49/artifacts/day14-closeout-and-handoff.md](./SPRINT_49/artifacts/day14-closeout-and-handoff.md)

---

## Bottom line

Epic 4 achieved its goal:

- the review-driven structural backlog was closed in planned order rather than
  through ad hoc cleanup
- lifecycle/state, graph, repeated-run, auxiliary-surface, and documentation
  ownership problems now have stable implemented seams
- the public repeated-run story is real, bounded, and validated
- the strongest local reviewed baseline and reviewed CMake parity contract
  remained intact through the whole epic

Epic 5 or later feature work can now start from a cleaner ownership model, a
clearer documentation contract, and a validated public repeated-run surface
instead of reopening whether the Epic 4 structural groundwork was actually
finished.
