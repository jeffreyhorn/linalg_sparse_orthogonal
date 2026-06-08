# Epic 5 Retrospective — Sprints 50-59 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × ~132-160 h each = **~1 452 hours nominal**  
**Branch range:** `sprint-50` → `sprint-59`  
**Started:** Sprint 50 (review-driven lifecycle/productization kickoff after
`reviews/review-codex-2026-05-31.md`)  
**Closed:** Sprint 59 Day 14 (Epic 5 closeout)  
**Goal:** Close the post-Epic-4 lifecycle, CSC, repeated-run, maintainability,
documentation, and quality/platform productization backlog from
`reviews/review-codex-2026-05-31.md` +
`reviews/todo-codex-2026-05-31.md` across explicit direct-solver lifecycle
exposure, deeper analyze/factor/refactor integration, CSC/factor-many
follow-through, repeated-run solver support completion or rationalized
bounding, large-source and giant-test maintainability cleanup, public-surface
simplification, and final Epic 5 integration/validation closeout.

> **Status: Epic 5 complete.** All 10 sprints landed. The direct-solver public
> repeated-run story was frozen around `sparse_analysis_t` /
> `sparse_factors_t` (Sprint 50), then made real through public lifecycle
> implementation and deeper integration (Sprints 51-52). CSC follow-through
> and indefinite factor-many proof closed the largest deferred direct-solver
> backlog (Sprint 53). The steady-state repeated-run solver support boundary
> was made explicit and finished for iterative/eigensolver workflows
> (Sprint 54). The largest remaining implementation hotspots were reduced by
> owned source extraction (Sprints 55-56), the biggest proof surfaces were
> made more maintainable and the lifecycle regression floor was strengthened
> (Sprint 57), public docs/examples/benchmark taxonomy were simplified
> (Sprint 58), and the final quality/platform residual map plus Epic-wide
> caller story were reconciled and validated (Sprint 59). Epic 5 ends with the
> strongest local reviewed baseline still intact (`make quality-review-full`),
> reviewed CMake parity still exact (`53` vs `53`, `53 / 53` passing), one
> coherent measured Epic handoff package, and an explicit bounded future queue
> instead of hidden closeout drift.

---

## Summary table

Per-sprint deliverables + nominal hours from `PROJECT_PLAN.md`. Epic 5 sprint
retrospectives track outcomes, validation, and deferred limits rather than
separate actual-hour totals, so the `actual h` column is left `n/t`.

| sprint | title | plan assessment | key deliverables | nominal h | actual h |
|---|---|---|---|---:|---:|
| 50 | Direct-Solver Lifecycle Baseline & API Design | Met exactly | measured baseline, direct lifecycle contract, explicit non-goal fence, landing/validation plan | 132 | n/t |
| 51 | Public Direct-Solver Lifecycle API Phase 1 | Met exactly | first public lifecycle header batch, LU/Cholesky/LDL^T lifecycle integration, one-shot wrapper preservation, focused regression/docs adoption | 148 | n/t |
| 52 | Analysis/Refactor Integration & Direct-Solver Lifecycle Phase 2 | Met exactly | deeper analysis/factor/refactor integration, tighter refactor contract, factor-many benchmark proof, public repeated-run regression expansion | 156 | n/t |
| 53 | CSC Direct-Solver Completion & Dispatch Follow-Through | Met exactly | analysis-aware indefinite LDL^T CSC completion, tighter CSC dispatch ownership, indefinite factor-many benchmark proof, CSC contract reconciliation | 144 | n/t |
| 54 | Public Repeated-Run Solver Lifecycle Completion | Met exactly | final repeated-run solver support boundary, MINRES public handle support, eigensolver-handle proof tightening, reuse benchmark alignment | 152 | n/t |
| 55 | Large-Source Decomposition Phase 1 | Met exactly | eigensolver backend extraction, MINRES extraction, build-surface alignment, historical comment reduction in touched permanent files | 160 | n/t |
| 56 | Large-Source Decomposition Phase 2 | Met exactly | LDLT CSC supernodal extraction, Cholesky CSC supernodal extraction, partial-SVD extraction, bounded CSC comment reconciliation | 148 | n/t |
| 57 | Giant-Test Refactor & Lifecycle Regression Expansion | Met exactly | giant-test helper seams, lifecycle/free zero-state regression proof, factor-many/one-shot compatibility proof | 144 | n/t |
| 58 | Documentation, Examples & Benchmark Story Simplification | Met exactly | README/tutorial reduction, public-header narrative cleanup, example modernization, benchmark taxonomy cleanup | 136 | n/t |
| 59 | Quality/Platform Follow-Through, Final Integration & Epic 5 Closeout | Met exactly | final quality/platform residual disposition, caller-story reconciliation, Epic 5 summary/handoff, final validated closeout baseline | 132 | n/t |
| **Total** | | | | **1 452** | **n/t** |

Epic 5 tracking stayed outcome- and validation-oriented rather than
hour-burn-oriented. The planning baseline therefore remains the authoritative
epic hour total.

---

## Cumulative metrics

### Epic-completion metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| nominal planned hours | `1 452` |
| closed Epic 5 work bands with explicit final disposition | `8 / 8` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake parity target | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Lifecycle/productization trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 50-52 | direct lifecycle design + implementation/integration | explicit analysis/factors-based repeated-run direct contract landed publicly, wrappers preserved, factor/refactor/solve path strengthened and benchmarked |
| 53 | CSC direct-solver follow-through | deeper analysis-aware indefinite LDL^T CSC completion, tighter LDL^T dispatch ownership, real indefinite factor-many proof |
| 54 | repeated-run solver completion | final public repeated-run boundary fixed: iterative handles `CG`/`GMRES`/`MINRES`, eigensolver handle grow-m/thick-restart/`LOBPCG`, intentional exclusions named |
| 55-56 | large-source decomposition | major production hotspots split into owned source files with narrower orchestration residuals |
| 57 | giant-test refactor + regression expansion | helper-owned proof seams landed, lifecycle/factor-many direct regressions tightened |
| 58-59 | public-surface simplification + final quality/platform closeout | workflow docs/headers/examples/benchmarks simplified, final caller story reconciled, residual platform limits documented truthfully, Epic 5 packaged from measured baseline |

### Final validation anchor (Sprint 59 Day 13)

| metric | final |
|---|---:|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `make quality-review-full` | passed |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `143.38 sec` |

### Public lifecycle close state

| metric | final |
|---|---:|
| public repeated-run direct lifecycle model | `analysis/factors-based` |
| iterative public-handle families | `3` |
| eigensolver public-handle backends | `3` |
| major production hotspots materially reduced in decomposition sprints | `5` |
| permanent giant-test helper seams landed | `3` |

Representative final close state:

- repeated direct solves remain the explicit analysis/factors lifecycle:
  - analyze once
  - factor / solve
  - refactor / solve many
- iterative public handles are limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver handle is limited to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- major production hotspot reductions:
  - `src/sparse_eigs.c`: `3233 -> 1534`
  - `src/sparse_iterative.c`: `2377 -> 1985`
  - `src/sparse_ldlt_csc.c`: `2723 -> 2127`
  - `src/sparse_chol_csc.c`: `2194 -> 1532`
  - `src/sparse_svd.c`: `1728 -> 1319`
- permanent giant-test helper seams:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_svd_partial_helpers.h`
  - `tests/test_iterative_handle_helpers.h`

---

## Stable contracts landed across Epic 5

| sprint | contract | rationale |
|---|---|---|
| 50 | explicit direct-solver lifecycle fence centered on `sparse_analysis_t` / `sparse_factors_t` | later implementation could advance without broad direct-handle redesign churn |
| 51-52 | public repeated-run direct lifecycle through shared analysis/factor/refactor path with one-shot wrapper preservation | repeated-run direct workflow became real while preserving the simple caller path |
| 53 | CSC repeated-run direct contract clarified around truthful LDL^T/Cholesky dispatch | CSC follow-through stopped being a visible weak spot in the public direct story |
| 54 | explicit steady-state repeated-run solver support boundary | iterative/eigensolver public repeated-run support stopped looking accidental or incomplete |
| 55-56 | bounded decomposition-first maintainability rule | large-source cleanup improved ownership without reopening public behavior or API surface |
| 57 | behavior-level giant-test refactor plus lifecycle/factor-many regression expansion | test maintainability improved while strengthening the actual public repeated-run proof floor |
| 58 | workflow-first docs/examples/benchmark simplification | public surfaces now explain the stable product shape more directly than sprint history |
| 59 | final quality/platform residual map + Epic-wide caller-story reconciliation | the epic closes from one truthful documented baseline instead of a partly staged closeout |

---

## Residual final limits / follow-up journal

Epic 5 closes without an unfinished planned remediation sprint, but it does
intentionally leave a few bounded limits in the stable post-epic contract:

1. **Direct repeated-run public exposure remains intentionally centered on analysis/factors rather than a generic universal direct handle.**
   One-shot direct APIs remain first-class/default workflows, and the
   compatibility-facing mutable `SparseMatrix` one-shot story still exists as
   an accepted tradeoff.

2. **Public repeated-run solver exposure is intentionally bounded.**
   Iterative public handles stop at `CG`, `GMRES`, `MINRES`; eigensolver
   handles stop at grow-m, thick-restart, and explicit `LOBPCG`;
   `BiCGSTAB` and block iterative workflows remain one-shot compatibility
   surfaces.

3. **Dead-code execution remains serialized.**
   The quality contract is truthful and useful, but the shared
   `build/deadcode-cmake` / `build/deadcode/` topology still blocks a broader
   concurrency-safe dead-code execution model.

4. **macOS dead-code remains staged and broader Windows reviewed-wrapper/dead-code work remains deferred.**
   Linux remains the enforced reviewed source-of-truth path; Windows closes
   with the reviewed CMake subset as the enforced truth surface.

5. **Some bounded maintainability seams remain future work.**
   Representative named seams:
   - later iterative decomposition:
     - `GMRES`
     - shared block-wrapper scaffolding
   - later CSC decomposition/comment cleanup if still justified
   - deferred giant-test seams:
     - `tests/test_ldlt_csc.c`
     - `tests/test_qr.c`
     - intentionally retained dense `tests/test_integration.c`

6. **Some broader docs-density cleanup remains future work.**
   The highest-signal public workflow drift is closed, but deeper long-form
   chronology/performance-history density still exists in some project docs.

These are future-facing post-Epic-5 boundaries, not hidden closeout defects.

---

## Lessons (Epic-level)

- **Design-first and audit-first sequencing paid off again.** Sprint 50,
  Sprint 53, Sprint 54, Sprint 55, Sprint 56, Sprint 57, Sprint 58, and
  Sprint 59 all benefited from forcing the boundary/design/audit decision
  before implementation. That kept later landings smaller and less likely to
  reopen scope.

- **Bound the public workflow story before expanding it.** Sprint 50 and
  Sprint 54 show the same pattern: explicit support/exclusion boundaries make
  later implementation and docs work much easier to validate and teach.

- **CSC/productization follow-through matters as much as raw feature landing.**
  Sprint 53 and Sprint 59 both closed important “almost there” gaps that would
  otherwise have left the public story feeling less uniform than the actual
  implementation quality justified.

- **Decomposition works best when it follows real ownership seams rather than
  raw line counts.** Sprint 55 and Sprint 56 succeeded because they extracted
  backend-owned or helper-owned slices, not because they chased maximal file
  shrinkage at any cost.

- **Test maintainability should strengthen behavior proof, not dilute it.**
  Sprint 57 improved the biggest proof surfaces by extracting helper ownership
  and adding regression value in the same sprint, rather than treating
  maintainability and confidence as separate concerns.

- **Documentation simplification is most effective when it follows a settled
  product contract.** Sprint 58 and Sprint 59 landed well because the
  lifecycle/support boundary was already stable. They were reconciling and
  simplifying, not inventing the product story late.

- **A good epic closeout makes residual limits explicit instead of pretending
  to erase them.** Sprint 59 was successful largely because it documented the
  real quality/platform and future-work boundaries honestly rather than trying
  to force false parity or fake completeness.

---

## DoD verification

Required cross-epic themes from `PROJECT_PLAN.md`:

- direct-solver lifecycle baseline, explicit public contract, and non-goal
  fence: ✓
- public direct lifecycle implementation plus deeper analysis/factor/refactor
  integration: ✓
- CSC direct-solver completion and dispatch follow-through: ✓
- explicit public repeated-run solver support boundary and completion: ✓
- large-source decomposition across the strongest remaining implementation
  hotspots: ✓
- giant-test maintainability improvement plus lifecycle/factor-many regression
  expansion: ✓
- public docs/examples/benchmark simplification around stable workflow
  categories: ✓
- final quality/platform reconciliation, final integration sweep, and Epic 5
  closeout package: ✓

Final measured close state:

- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- final caller story reconciled across README/tutorial/examples/benchmarks/
  headers/tests: ✓
- no planned Sprint 60 remediation sprint required to explain away unfinished
  Epic 5 work: ✓

---

## Key references

- [PROJECT_PLAN.md](../PROJECT_PLAN.md)
- [review-codex-2026-05-31.md](../reviews/review-codex-2026-05-31.md)
- [todo-codex-2026-05-31.md](../reviews/todo-codex-2026-05-31.md)
- [SPRINT_50/RETROSPECTIVE.md](../SPRINT_50/RETROSPECTIVE.md)
- [SPRINT_53/RETROSPECTIVE.md](../SPRINT_53/RETROSPECTIVE.md)
- [SPRINT_54/RETROSPECTIVE.md](../SPRINT_54/RETROSPECTIVE.md)
- [SPRINT_55/RETROSPECTIVE.md](../SPRINT_55/RETROSPECTIVE.md)
- [SPRINT_56/RETROSPECTIVE.md](../SPRINT_56/RETROSPECTIVE.md)
- [SPRINT_57/RETROSPECTIVE.md](../SPRINT_57/RETROSPECTIVE.md)
- [SPRINT_58/RETROSPECTIVE.md](../SPRINT_58/RETROSPECTIVE.md)
- [SPRINT_59/RETROSPECTIVE.md](../SPRINT_59/RETROSPECTIVE.md)
- [SPRINT_59/artifacts/day11-epic5-summary-and-handoff-batch.md](../SPRINT_59/artifacts/day11-epic5-summary-and-handoff-batch.md)
- [SPRINT_59/artifacts/day14-closeout-and-handoff.md](../SPRINT_59/artifacts/day14-closeout-and-handoff.md)

---

## Bottom line

Epic 5 achieved its goal:

- the direct repeated-run story is now an explicit public lifecycle rather than
  an under-centered internal precedent
- the CSC direct-solver queue no longer stands out as an obvious unfinished
  productization gap
- the repeated-run solver support boundary is explicit, validated, and
  intentionally bounded
- the largest implementation and proof hotspots are materially smaller and more
  cleanly owned
- the highest-signal public docs, examples, headers, and benchmark surfaces are
  simpler and more product-like
- the final quality/platform residual map is explicit and truthful
- the branch closed from a fully validated reviewed baseline with exact
  preserved truthfulness anchors

Future work can now start from a cleaner lifecycle model, a more uniform
product surface, and one measured final Epic 5 handoff package instead of
reopening whether the direct repeated-run story, the remaining CSC follow-ons,
or the quality/platform closeout were actually finished.
