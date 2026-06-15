# Epic 6 Retrospective — Sprints 60-69 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × ~144-168 h each = **~1 572 hours nominal**  
**Branch range:** `sprint-60` → `sprint-69`  
**Started:** Sprint 60 (review-driven productization/performance/usability kickoff after
`reviews/review-codex-2026-06-08.md`)  
**Closed:** Sprint 69 Day 14 (Epic 6 closeout)  
**Goal:** Close the remaining usability, configuration, backend-performance,
platform, maintainability, assurance, and public-surface gaps identified in
`reviews/review-codex-2026-06-08.md` +
`reviews/todo-codex-2026-06-08.md` across typed configuration convergence,
direct-solver usability tightening, backend/performance modernization,
benchmark governance, packaging/platform convergence, remaining large-source
and giant-test cleanup, and final cross-surface public product integration.

> **Status: Epic 6 complete.** All 10 sprints landed. The epic opened by
> freezing the post-Epic-5 productization target, non-goal fence, and reviewed
> validation/platform contract (Sprint 60). It then closed the main remaining
> product surface queues in order: typed reorder/analysis configuration
> convergence (Sprint 61), direct one-shot and lifecycle usability tightening
> (Sprint 62), LU/CSC repeated-run direct uniformity follow-through
> (Sprint 63), the first bounded backend-aware dense-kernel seam
> (Sprint 64), canonical benchmark/performance governance plus targeted
> efficiency follow-through (Sprint 65), static-first packaging/platform
> contract convergence (Sprint 66), residual large-source implementation
> cleanup (Sprint 67), giant-test maintainability plus stronger numerical
> assurance (Sprint 68), and final public-surface reconciliation plus Epic 6
> closeout (Sprint 69). Epic 6 ends with the strongest local reviewed baseline
> still intact (`make quality-review-full`), reviewed CMake parity still exact
> (`53` vs `53`, `53 / 53` passing), one coherent validated product surface,
> and an explicit bounded carry-forward queue instead of hidden closeout drift.

---

## Summary table

Per-sprint deliverables + nominal hours from `PROJECT_PLAN.md`. Epic 6 sprint
retrospectives and closeout artifacts track outcomes, validation, and bounded
residuals rather than separate actual-hour totals, so the `actual h` column is
left `n/t`.

| sprint | title | plan assessment | key deliverables | nominal h | actual h |
|---|---|---|---|---:|---:|
| 60 | Epic 6 Baseline, Productization Audit & Architecture Contract | Met exactly | measured baseline, explicit target/non-goal fence, control-placement map, frozen validation/platform contract | 148 | n/t |
| 61 | Configuration Surface Modernization Phase 1 | Met exactly | typed reorder/analysis options, explicit typed/env/default precedence, bounded compatibility preservation, docs/regression support | 160 | n/t |
| 62 | Direct-Solver Usability & Lifecycle Coherence | Met exactly | one-shot LU/Cholesky hardening, preserved-caller reorder behavior, clearer one-shot vs explicit lifecycle story, direct regression/docs adoption | 156 | n/t |
| 63 | Direct-Lifecycle Uniformity & CSC/LU Follow-Through | Met exactly | LU lifecycle safety follow-through, large-`n` CSC-backed Cholesky repeated-run preservation, CSC dispatch/publication proof refresh | 164 | n/t |
| 64 | Performance Backend Architecture Phase 1 | Met exactly | first backend-aware CSC supernodal Cholesky dense-kernel seam, `SPARSE_ERR_BACKEND_CONTRACT`, benchmark path-identification proof | 168 | n/t |
| 65 | Performance Governance, Benchmark Consolidation & Solver Efficiency Follow-Through | Met exactly | canonical benchmark taxonomy, normalized machine-readable output, `make bench-canonical-report`, bounded direct CSC/Cholesky efficiency follow-through | 152 | n/t |
| 66 | Packaging, ABI & Platform Quality Convergence | Met exactly | explicit static-first packaging contract, install/export truth alignment, workflow/platform reconciliation, `VERSION`-driven install/package proof | 160 | n/t |
| 67 | Large-Source Maintainability Phase 3 | Met exactly | graph/reorder ownership extraction, shared ND policy convergence, large-`n` Cholesky analysis-to-CSC handoff cleanup | 156 | n/t |
| 68 | Giant-Test Refactor Phase 2 & Numerical Assurance Expansion | Met exactly | bounded `test_chol_csc` helper extraction, stronger public-path oracle coverage, seeded large-`n` CSC lifecycle property lane | 164 | n/t |
| 69 | Public Product Surface Finalization, Integration & Epic 6 Closeout | Met exactly | README/tutorial productization, examples/benchmarks/tests ownership reconciliation, final integrated validation and Epic 6 handoff | 144 | n/t |
| **Total** | | | | **1 572** | **n/t** |

Epic 6 tracking stayed outcome- and validation-oriented rather than
hour-burn-oriented. The planning baseline therefore remains the authoritative
epic hour total.

---

## Cumulative metrics

### Epic-completion metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| nominal planned hours | `1 572` |
| closed Epic 6 work bands with explicit final disposition | `10 / 10` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake parity target | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Productization/performance trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 60-61 | Epic 6 target definition + typed configuration convergence | explicit control-placement model landed and the highest-value reorder/analysis env-var surfaces moved to typed options with shipped precedence rules |
| 62-63 | direct-solver usability and lifecycle coherence | one-shot LU/Cholesky surprise reduced, reordered preserve-caller behavior tightened, LU/CSC repeated-run lifecycle semantics became more uniform and more explicitly proved |
| 64-65 | backend/performance modernization + benchmark governance | one bounded backend-aware CSC supernodal Cholesky seam landed, canonical maintained benchmark surfaces were fixed, output normalized, and threshold-free reporting added |
| 66 | packaging, ABI, and platform-quality convergence | static-first packaging/release story became explicit, install/export proof tightened around `VERSION`, and platform asymmetries were documented truthfully rather than hidden |
| 67-68 | residual maintainability + assurance expansion | graph/ND/Cholesky residual ownership seams narrowed, giant-test pressure reduced, and stronger oracle/property proof landed on the hardest large-`n` CSC-backed Cholesky public path |
| 69 | final public product integration and epic closeout | README/tutorial/examples/benchmarks/tests story reconciled into one coherent front-door/adoption/proof model and Epic 6 closed from a full integrated validated baseline |

### Final validation anchor (Sprint 69 Day 12)

| metric | final |
|---|---:|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `make quality-review-full` | passed |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `797.77 sec` |

### Final Epic 6 product-surface state

| metric | final |
|---|---:|
| typed reorder/analysis compatibility controls landed publicly in Phase 1 | `8` |
| canonical maintained benchmark binaries | `4` |
| maintained install/package proof surfaces | `2` |
| bounded large-`n` CSC lifecycle seeded property lane | `1` |
| final integrated public ownership split | `examples / benchmarks / tests` |

Representative final close state:

- direct repeated solves remain the explicit `analysis` / `factors` lifecycle:
  - analyze once
  - factor / solve
  - refactor / solve many
- typed reorder/analysis options now own the highest-value advanced policy
  lane, with precedence:
  - explicit typed value
  - bounded compatibility env var when unspecified
  - internal default
- backend-aware acceleration remains intentionally bounded to the first CSC
  supernodal Cholesky dense-kernel lane, with:
  - self-contained builtin default path
  - explicit `SPARSE_ERR_BACKEND_CONTRACT`
- canonical maintained benchmark/reporting surface is intentionally small:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
  - `make bench-canonical-report`
- install/package proof remains intentionally focused on:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- giant-test / oracle / property proof for the hardest direct lane is now
  split intentionally across:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`

---

## Stable contracts landed across Epic 6

| sprint | contract | rationale |
|---|---|---|
| 60 | explicit Epic 6 product target, non-goal fence, validation/platform truth fence | later implementation could move quickly without reopening what the epic was actually trying to ship |
| 61 | typed reorder/analysis option surface with explicit typed/env/default precedence | configuration modernization became a real shipped product surface rather than an audit-only complaint |
| 62-63 | clearer one-shot vs explicit repeated-run direct lifecycle reading plus stronger CSC/LU preservation semantics | direct usability improved without redesigning the established repeated-run direct model |
| 64 | bounded backend-aware dense-kernel seam plus explicit backend-contract error taxonomy | performance modernization advanced without pretending the repo had become a broad pluggable-backend framework |
| 65 | canonical maintained benchmark surface and threshold-free reporting model | performance proof became smaller, more machine-readable, and more maintainable without fake timing-threshold governance |
| 66 | explicit static-first packaging/release contract and truthful platform asymmetry map | productization claims now match install/export/workflow evidence directly |
| 67 | ownership-first maintainability rule for residual source hotspots | large-source cleanup improved orchestration clarity without reopening unrelated families or public APIs |
| 68 | bounded helper extraction plus stronger oracle/property assurance on the hardest large-`n` CSC-backed Cholesky lane | giant-test cleanup and confidence expansion reinforced each other instead of competing for sprint scope |
| 69 | final public front-door/adoption/proof ownership split | README/tutorial/examples/benchmarks/tests now explain one coherent product story instead of partially overlapping ones |

---

## Residual final limits / follow-up journal

Epic 6 closes without an unfinished planned remediation sprint, but it does
intentionally leave a few bounded limits in the stable post-epic contract:

1. **The remaining direct-family usability queue is intentionally narrow, not erased.**
   Representative named seams:
   - CSC progress-callback parity for Cholesky / LDL^T
   - no-reorder linked-list Cholesky bit-identical cancellation restoration

2. **The strongest remaining validation/runtime concentration is still `test_reorder_nd`.**
   The reviewed baseline remains truthful and strong, but `test_reorder_nd`
   is still the clearest cost center on future reviewed sweeps.

3. **Some bounded giant-test maintainability seams remain future work.**
   Representative named seams:
   - `tests/test_reorder_nd.c`
   - `tests/test_ldlt_csc.c`
   Sprint 68 reduced the highest-value first seam but did not pretend the
   remaining giant proof files were all equally urgent.

4. **Public header/reference cleanup was intentionally deferred once the final product story no longer required it.**
   Representative deferred reference surfaces:
   - `include/sparse_cholesky.h`
   - `include/sparse_analysis.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`

5. **Packaging and platform truth remains intentionally bounded.**
   The repo now has a clearer static-first product surface, but it still does
   not claim:
   - broad shared-library or dynamic-ABI guarantees
   - reviewed Windows proof for the bounded `test_fuzz` lifecycle property lane
   - macOS/Windows dead-code parity with Linux

6. **Backend and benchmark governance remain intentionally scoped.**
   Epic 6 landed a real first backend-aware seam and a real canonical
   benchmark surface, but it did not turn into:
   - a broad backend framework
   - universal benchmark-threshold CI gating
   - wider performance governance beyond the maintained canonical surfaces

These are future-facing post-Epic-6 boundaries, not hidden closeout defects.

---

## Lessons (Epic-level)

- **Audit-first and boundary-first sequencing paid off repeatedly.** Sprint 60,
  Sprint 61, Sprint 62, Sprint 64, Sprint 65, Sprint 66, Sprint 67, and
  Sprint 68 all benefited from fixing the ranking, design, and non-widening
  fence before code moved.

- **Typed configuration work succeeds when it lands with explicit precedence and compatibility ownership.** Sprint 61 worked because it did not treat
  “remove env vars” as the goal; it replaced the highest-value policy lane with
  a better owned model and made the remaining compatibility path explicit.

- **Product maturity improvements are often about making existing capability more truthful and coherent, not only about adding new algorithm surface.**
  Sprint 62, Sprint 63, Sprint 66, and Sprint 69 all produced high-value
  outcomes by tightening interpretation, preservation rules, ownership, and
  documentation around already-important workflows.

- **Bounded modernization beats fake framework generalization.** Sprint 64 and
  Sprint 65 landed better because they chose one real backend-aware lane and
  one real canonical benchmark surface instead of pretending the repository was
  ready for a universal backend or performance-governance framework.

- **Assurance expansion is most useful when it targets the hardest already-important lane.** Sprint 68 improved confidence because it strengthened the
  large-`n` CSC-backed Cholesky public path with oracle and property proof,
  not because it added generic randomized volume.

- **A good closeout sprint proves that no extra cleanup batch is needed.**
  Sprint 69 was successful partly because it verified that no forced header,
  policy, or implementation follow-through remained necessary once the final
  public product story had been reconciled.

---

## DoD verification

Required cross-epic themes from `PROJECT_PLAN.md`:

- measured Epic 6 baseline, explicit product target, architecture contract,
  and validation/platform fence: ✓
- typed configuration modernization with explicit precedence and bounded
  compatibility preservation: ✓
- direct-solver usability and lifecycle coherence tightening: ✓
- direct-lifecycle uniformity plus LU/CSC follow-through: ✓
- first bounded backend/performance architecture landing: ✓
- canonical benchmark/performance-governance surface plus targeted efficiency
  follow-through: ✓
- packaging, ABI, install/export, and platform-quality convergence: ✓
- remaining large-source maintainability follow-through: ✓
- giant-test refactor plus numerical assurance expansion: ✓
- final public product-surface integration, full validation, and Epic 6
  closeout package: ✓

Final measured close state:

- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- final public story reconciled across README/tutorial/examples/benchmarks/
  maintainer guide/tests/install/package proof: ✓
- no planned Sprint 70 remediation sprint is required to explain away
  unfinished Epic 6 work: ✓

---

## Key references

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [review-codex-2026-06-08.md](./reviews/review-codex-2026-06-08.md)
- [todo-codex-2026-06-08.md](./reviews/todo-codex-2026-06-08.md)
- [SPRINT_60/RETROSPECTIVE.md](./SPRINT_60/RETROSPECTIVE.md)
- [SPRINT_61/RETROSPECTIVE.md](./SPRINT_61/RETROSPECTIVE.md)
- [SPRINT_62/RETROSPECTIVE.md](./SPRINT_62/RETROSPECTIVE.md)
- [SPRINT_63/RETROSPECTIVE.md](./SPRINT_63/RETROSPECTIVE.md)
- [SPRINT_64/RETROSPECTIVE.md](./SPRINT_64/RETROSPECTIVE.md)
- [SPRINT_65/RETROSPECTIVE.md](./SPRINT_65/RETROSPECTIVE.md)
- [SPRINT_66/RETROSPECTIVE.md](./SPRINT_66/RETROSPECTIVE.md)
- [SPRINT_67/RETROSPECTIVE.md](./SPRINT_67/RETROSPECTIVE.md)
- [SPRINT_68/RETROSPECTIVE.md](./SPRINT_68/RETROSPECTIVE.md)
- [SPRINT_69/RETROSPECTIVE.md](./SPRINT_69/RETROSPECTIVE.md)
- [SPRINT_69/artifacts/day12-full-validation-sweep.md](./SPRINT_69/artifacts/day12-full-validation-sweep.md)
- [SPRINT_69/artifacts/day13-epic6-summary-and-residual-finalization.md](./SPRINT_69/artifacts/day13-epic6-summary-and-residual-finalization.md)
- [SPRINT_69/artifacts/day14-closeout-and-epic6-final-handoff.md](./SPRINT_69/artifacts/day14-closeout-and-epic6-final-handoff.md)

---

## Bottom line

Epic 6 achieved its goal:

- the repo now has a clearer advanced-configuration story instead of a looser
  env-var-first tuning surface
- the one-shot and explicit repeated-run direct stories are more coherent and
  better proved on the hardest LU/CSC lanes
- the first backend-aware performance seam is real, bounded, and validated
- the benchmark surface is smaller, more canonical, and more machine-readable
- the packaging/install/platform story is more explicit and more truthful
- the strongest remaining implementation and giant-test seams are smaller and
  more intentionally owned
- the highest-signal public docs/examples/benchmarks/tests now explain one
  coherent final product story
- the branch closed from a fully validated reviewed baseline with exact
  preserved truthfulness anchors

Future work can now start from a cleaner public product model, a stronger
configuration/performance/packaging baseline, and one measured final Epic 6
handoff package instead of reopening whether the remaining usability,
benchmark-governance, platform, assurance, or closeout surfaces were actually
finished.
