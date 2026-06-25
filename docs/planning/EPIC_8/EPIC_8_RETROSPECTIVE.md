# Epic 8 Retrospective — Sprints 80-89 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × day-cap totals = **1 680 hours nominal**  
**Branch range:** `sprint-80` -> `sprint-89`  
**Started:** Sprint 80 (Epic 8 baseline, competitive target, and external
oracle contract)  
**Closed:** Sprint 89 Day 14 (Epic 8 closeout)  
**Goal:** Close the post-Epic-7 residual queue from a bounded, truthful,
validated baseline across product/storage modernization, dense/backend
ceiling reduction, capability-surface modernization, differential and
property-proof expansion, maintainability hotspot reduction, reviewed runtime
convergence, packaging/ABI/install-export truthfulness, front-door usability
cleanup, and final external-comparison-backed Epic 8 integration.

> **Status: Epic 8 complete.** All 10 sprints landed. The epic opened by
> freezing the live competitive gap, external-oracle contract, benchmark
> governance fence, and non-goal/risk fence (Sprint 80). It then closed the
> main remaining seams in order: compressed-first product/storage
> modernization (Sprint 81), bounded dense/backend ceiling reduction through
> optional accelerated direct-family lanes (Sprint 82), bounded public
> capability-surface widening around shared scalar/index vocabulary
> (Sprint 83), maintained external differential and seeded/failure-path proof
> expansion (Sprint 84), large-source and giant-test maintainability cleanup
> (Sprint 85), reorder/ND runtime reduction and reviewed runtime convergence
> (Sprint 86), static-first packaging/ABI/install-export truth convergence
> (Sprint 87), front-door usability and workflow simplification (Sprint 88),
> and final external-comparison-backed Epic 8 closeout (Sprint 89). Epic 8
> ends with the strongest local reviewed baseline still intact
> (`make quality-review-full`), reviewed CMake parity still exact
> (`53` vs `53`, `53 / 53` passing), maintained install/export proof still
> passing (`13 / 13`, `15 / 15` with `0` skips at close), one real bounded
> external SPD comparison lane, one truthful bounded runtime-reference
> reading, and one explicit post-Epic-8 residual queue instead of hidden
> closeout drift.

---

## Summary table

Per-sprint deliverables + nominal hours from the sprint plans. As with the
earlier epic retrospectives, Epic 8 sprint retrospectives and closeout
artifacts track outcomes, validation, and bounded residuals rather than
separate actual-hour totals, so the `actual h` column remains `n/t`.

| sprint | title | plan assessment | key deliverables | nominal h | actual h |
|---|---|---|---|---:|---:|
| 80 | Epic 8 Baseline, Competitive Target & External Oracle Contract | Met exactly | live competitive-gap inventory, bounded external-oracle contract, benchmark-governance fence, non-goal/risk fence, validated epic baseline | 168 | n/t |
| 81 | Core Product / Storage Modernization Phase 2 | Met exactly | compressed-first copy/transpose/import path, repeated-run direct-workflow convergence, bounded matrix-shell/product-model cleanup | 168 | n/t |
| 82 | Dense Backend & Performance Ceiling Phase 3 | Met exactly | optional bounded Accelerate-backed Cholesky/LDL^T dense lanes, retained builtin defaults, backend-proof follow-through | 168 | n/t |
| 83 | Capability Surface Modernization Phase 2 | Met exactly | shared scalar/index public-owner widening, matrix-shell scalar-surface expansion, QR bounded public-header widening | 168 | n/t |
| 84 | Numerical Assurance & Differential Testing Phase 2 | Met exactly | maintained direct-family external differential lane, seeded lifecycle/property expansion, failure-path lifecycle proof widening | 168 | n/t |
| 85 | Large-Source Maintainability Phase 5 | Met exactly | iterative-source cleanup, LDL^T dense-owner re-home, Cholesky giant-test registration cleanup | 168 | n/t |
| 86 | Reordering Scalability & Reviewed Runtime Convergence | Met exactly | ND base-threshold runtime reduction, ND proof-owner rebalance, bounded reorder runtime measurement slice | 168 | n/t |
| 87 | Packaging, ABI & Cross-Platform Quality Convergence Phase 3 | Met exactly | exact-version CMake export contract, stronger local consumer proof, tighter supplemental macOS package lane | 168 | n/t |
| 88 | Front-Door Usability & Workflow Simplification | Met exactly | README/front-door simplification, examples adoption-map cleanup, INSTALL/support-surface consolidation | 168 | n/t |
| 89 | Final Integration, External Comparison & Epic 8 Closeout | Met exactly | end-state re-audit, bounded external comparison execution, explicit no-op retirement of final fix batch, final Epic 8 closeout | 168 | n/t |
| **Total** | | | | **1 680** | **n/t** |

Epic 8 tracking stayed outcome- and validation-oriented rather than
hour-burn-oriented. The sprint plans therefore remain the authoritative epic
hour baseline.

---

## Cumulative metrics

### Epic-completion metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| nominal planned hours | `1 680` |
| sprint plans present | `10` |
| sprint working-notes files present | `10` |
| sprint retrospectives present | `10` |
| sprint artifact files under `EPIC_8/SPRINT_*/artifacts/` | `150` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake parity target | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Epic 8 execution trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 80 | baseline, competitive target, oracle contract, benchmark-governance and risk fence | Epic 8 started from one explicit contradiction map, one bounded oracle contract, and one anti-claim fence rather than from generic modernization pressure |
| 81-83 | product/storage modernization, backend-ceiling reduction, capability-surface widening | the repo became more compressed-first on high-value touched paths, gained bounded optional dense acceleration, and widened the public scalar/index owner story without overstating the numeric contract |
| 84 | assurance and external differential expansion | the project gained one real maintained external SPD comparison lane plus stronger seeded lifecycle/property and failure-path proof |
| 85-86 | maintainability hotspot cleanup plus reviewed runtime convergence | the strongest large-source and giant-test seams became more reviewable while the ND runtime long pole was materially reduced |
| 87-88 | packaging/install-export truth convergence plus front-door usability cleanup | the static-first package/export story became sharper and the adoption/support audience split became clearer |
| 89 | final external comparison, residual calibration, and epic closeout | Epic 8 closed from a live external-comparison-backed, validated baseline instead of from another hoped-for implementation batch |

### Final validation anchor (Sprint 89 Day 13)

| metric | final |
|---|---:|
| `make quality-review-full` | passed |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `375.43 sec` |
| reviewed `test_reorder_nd` time | `215.72 sec` |
| install regression | `13` passed, `0` failed |
| CMake install/export regression | `15` passed, `0` failed, `0` skipped |
| canonical reporting follow-on | `make bench-canonical-report` passed |

### Final Epic 8 end-state markers

| metric | final |
|---|---:|
| maintained direct-family external SPD comparison fixtures | `2` |
| maintained install/package proof surfaces | `2` |
| bounded runtime-reference slice fixtures | `2` |
| front-door adoption/support owner files materially tightened | `3` |
| explicit no-op final implementation batch | `1` |

Representative final close state:

- the project now has one real maintained bounded external comparison lane:
  - `./build/quality-review-cmake/test_chol_csc`
  - retained external dense reference agreement on `nos4` and `bcsstk04`
- product/storage and workflow reading is less linked-list-first in the
  highest-value touched paths:
  - copy/transpose/import construction is more compressed-first
  - repeated-run Cholesky and LDL^T direct workflows are more consistently
    analysis-backed
- dense/backend acceleration remains intentionally bounded and truthful:
  - builtin dense paths remain the shipped defaults
  - optional Accelerate-backed direct-family dense lanes remain runtime-bounded
    and proof-backed
- public capability seams are wider but still explicitly bounded:
  - `SPARSE_IDX_BITS`
  - `SPARSE_SCALAR_BITS`
  - `sparse_scalar_t`
  - highest-value matrix-shell and QR caller-owned seams now read through the
    shared scalar/index vocabulary
- assurance is materially stronger than at epic start:
  - external SPD agreement exists
  - seeded lifecycle/property coverage widened
  - failure-path retry/oracle coverage widened
- runtime/reporting is more truthful and better calibrated:
  - the strongest ND runtime long pole is materially smaller than before
  - bounded reorder runtime comparison remains fixture-dependent rather than
    overclaimed as a broad superiority result
- packaging/platform truth is sharper and still intentionally bounded:
  - exact-version CMake export contract
  - static-first Make/pkg-config and CMake install/export proof
  - narrower but more truthful macOS and Windows workflow interpretation
- front-door usability is materially cleaner:
  - `README.md`
  - `examples/README.md`
  - `INSTALL.md`
  now read more as distinct adoption, next-step, and operational setup owners

---

## Stable contracts landed across Epic 8

| sprint | contract | rationale |
|---|---|---|
| 80 | explicit competitive gap map, external-oracle contract, benchmark-governance fence, and non-goal/risk fence | later implementation sprints could move quickly without reopening what Epic 8 was actually trying to prove or carefully refusing to claim |
| 81 | compressed-first touched construction/import and repeated-run direct-workflow convergence | product/storage work advanced without pretending the whole public matrix shell had already been redesigned |
| 82 | optional bounded direct-family dense acceleration with builtin-default retention | backend work moved the ceiling without turning the repo into a fake general backend framework |
| 83 | explicit shared scalar/index public-owner widening with retained real-only scalar contract | capability modernization progressed without pretending the repo had already become broadly generic, complex-aware, or mixed-precision |
| 84 | one real maintained external SPD comparison lane plus stronger seeded/failure-path proof | assurance work moved from internal confidence only to bounded external differential evidence |
| 85 | ownership-first large-source and giant-test cleanup | hotspot reduction improved reviewability without subsystem rewrites or shared-harness theater |
| 86 | bounded ND runtime reduction plus branch-local runtime evidence | runtime work reduced the reviewed long pole without inventing a broad timing gate or rewriting the whole graph/reorder family |
| 87 | exact-version static-first packaging/install-export contract plus bounded workflow truth | package/platform surfaces became more truthful without widening reviewed claims beyond evidence |
| 88 | adoption-first public front door plus explicit audience split across adoption/support/maintainer owners | usability improved without flattening truthful distinctions between support surfaces |
| 89 | evidence-backed epic closeout with explicit no-op final fix batch and calibrated residual queue | Epic 8 ends from measured proof and explicit non-claims rather than from generic closeout prose |

---

## Residual final limits / follow-up journal

Epic 8 closes without a planned remediation epic to explain away unfinished
work, but it intentionally leaves a bounded post-epic queue in the stable
contract:

1. **Reviewed reorder/ND runtime concentration remains the clearest operational long pole.**
   The runtime burden is materially smaller than it was before Sprint 86, but
   the reviewed close still carries:
   - reviewed `test_reorder_nd` = `215.72 sec`
   - reviewed total = `375.43 sec`

2. **Broader external comparison depth remains future work.**
   Epic 8 now has one real maintained bounded external SPD lane, but it still
   does not claim broad sparse-solver ecosystem comparison coverage.
   Representative deferred ideas:
   - METIS-class or broader graph/reordering comparisons
   - wider solver-family external comparisons beyond the retained SPD lane

3. **Capability widening remains intentionally bounded rather than exhausted.**
   Epic 8 improved the public scalar/index owner story, but it still does not
   claim:
   - broad complex support
   - broad mixed-precision support
   - family-wide numeric genericity across every public seam

4. **Packaging/platform truth remains intentionally asymmetric.**
   Epic 8 sharpened the static-first install/export story, but it still does
   not claim:
   - shared-library-first product maturity
   - symmetric reviewed install/export parity across Linux, macOS, and Windows
   - broader package/platform guarantees than the maintained proof justifies

5. **Some large-source and giant-test maintainability hotspots remain explicit carry-forward work.**
   Epic 8 materially improved several hotspots, but it still does not claim
   every large owner is fully decomposed. Representative carry-forward seams:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
   - selected large direct-family and iterative owners only where future
     hotspot maps justify more extraction

6. **Runtime evidence remains bounded and fixture-dependent, not a broad best-in-class claim.**
   The final touched runtime slice stayed mixed:
   - `bcsstk14` favors AMD
   - `Pres_Poisson` favors ND
   Epic 8 therefore closes with a calibrated runtime story rather than a
   simple superiority headline.

These are post-Epic-8 boundaries, not hidden closeout defects.

---

## Lessons (Epic-level)

- **Boundary-first sequencing paid off again.** Sprint 80 and the later audit /
  design days kept the epic from diffusing into generic modernization or
  comparison theater.

- **Bounded implementation beats framework theater.** Sprint 81 through
  Sprint 87 worked because each lane chose one real owner seam and one real
  proof boundary instead of pretending the repo was ready for universal
  product, backend, package, or runtime frameworks.

- **Capability and assurance work are strongest when they expose explicit seams.**
  Sprint 83 and Sprint 84 succeeded because they made public scalar/index
  ownership and external differential/lifecycle proof boundaries explicit
  rather than hiding them behind typedefs or audit prose.

- **Truthful non-claims are part of product quality.** Sprint 80, Sprint 87,
  Sprint 88, and Sprint 89 repeatedly improved the repo by narrowing claims,
  not by inflating them. The final Epic 8 close state is stronger precisely
  because it stays bounded where evidence stays bounded.

- **Runtime work is most valuable when it ships measured reduction, not just profiling.**
  Sprint 86 mattered because it actually moved the reviewed long pole down and
  then preserved the remaining cost as explicit carry-forward work.

- **Good no-op decisions are part of engineering discipline.** Multiple Epic 8
  lanes, especially in Sprint 80 and Sprint 89, closed as explicit no-ops
  rather than forced churn. That kept the epic aligned to real contradictions
  instead of generating low-value edits for appearance.

---

## DoD verification

Required cross-epic themes from `PROJECT_PLAN.md`:

- Epic 8 baseline, competitive target, and external-oracle contract: complete
- core product/storage modernization phase 2: complete
- dense backend and performance ceiling phase 3: complete
- capability surface modernization phase 2: complete
- numerical assurance and differential testing phase 2: complete
- large-source maintainability phase 5: complete
- reordering scalability and reviewed runtime convergence: complete
- packaging, ABI, and cross-platform quality convergence phase 3: complete
- front-door usability and workflow simplification: complete
- final integration, external comparison, and Epic 8 closeout: complete

Final measured close state:

- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- install/export proof:
  - `tests/test_install.sh` passing
  - `tests/test_cmake_install.sh` passing
- canonical benchmark/reporting proof:
  - `make bench-canonical-report` passing
- bounded external comparison proof:
  - maintained SPD external comparison lane passing
  - maintained package/install/export comparison lane passing
- final public/product/proof/package/usability story reconciled across:
  - README
  - examples
  - INSTALL
  - maintainer guide
  - tests
  - install/export proof
  - benchmark reporting
- no hidden Sprint 90 remediation sprint is needed to explain away unfinished
  Epic 8 planned work: complete
