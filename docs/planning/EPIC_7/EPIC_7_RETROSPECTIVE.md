# Epic 7 Retrospective — Sprints 70-79 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × planned totals = **1 549 hours nominal**  
**Branch range:** `sprint-70` -> `sprint-79`  
**Started:** Sprint 70 (Epic 7 baseline, state-of-the-art target, and
architecture-contract freeze)  
**Closed:** Sprint 79 Day 14 (Epic 7 closeout)  
**Goal:** Close the post-Epic-6 residual queue from a bounded, truthful,
validated baseline across public-surface cleanup, product-model convergence,
configuration modernization, capability modernization, backend/performance
architecture, benchmark governance, packaging/platform convergence,
maintainability follow-through, numerical assurance expansion, and final
cross-surface Epic 7 integration.

> **Status: Epic 7 complete.** All 10 sprints landed. The epic opened by
> freezing the post-Epic-6 target state, validation/platform truth fence, and
> ranked execution order (Sprint 70). It then closed the main remaining seams
> in order: public-surface de-chronologization and reference cleanup
> (Sprint 71), first-phase matrix/product-model convergence (Sprint 72),
> residual configuration-surface modernization (Sprint 73), first-phase
> capability-surface modernization centered on width/scalar contracts
> (Sprint 74), second backend/performance architecture work in the CSC
> supernodal Cholesky lane (Sprint 75), benchmark-governance and longitudinal
> reporting convergence (Sprint 76), packaging/ABI/cross-platform contract
> convergence (Sprint 77), large-source and giant-test maintainability
> follow-through (Sprint 78), and final assurance expansion plus integrated
> Epic 7 closeout (Sprint 79). Epic 7 ends with the strongest local reviewed
> baseline still intact (`make quality-review-full`), reviewed CMake parity
> still exact (`53` vs `53`, `53 / 53` passing), install/export proof still
> passing (`11 / 11`, `13 / 13`), one coherent validated end-of-epic product
> story, and an explicit bounded post-Epic-7 queue instead of hidden closeout
> drift.

---

## Summary table

Per-sprint deliverables + nominal hours from the sprint plans. Like the earlier
epic retrospectives, Epic 7 sprint retrospectives and closeout artifacts track
outcomes, validation, and bounded residuals rather than separate actual-hour
totals, so the `actual h` column remains `n/t`.

| sprint | title | plan assessment | key deliverables | nominal h | actual h |
|---|---|---|---|---:|---:|
| 70 | Epic 7 Baseline, State-of-the-Art Target & Architecture Contract | Met exactly | explicit Epic 7 target, architecture contract, ranked execution fence, validation/platform/package truth baseline | 152 | n/t |
| 71 | Public Surface De-chronologization & Reference Cleanup | Met exactly | README/INSTALL/tutorial/examples/benchmarks/maintainer-guide cleanup, ownership wording tightening, preserved support-surface truthfulness | 158 | n/t |
| 72 | Core Matrix/Product Model Convergence Phase 1 | Met exactly | direct-workflow matrix-shell ownership cleanup, `sparse_reset_perms()` behavior tightening, first product-model contract follow-through | 158 | n/t |
| 73 | Configuration Surface Modernization Phase 2 | Met exactly | FM/graph policy convergence, debug/profile precedence cleanup, maintainer/proof-owner follow-through | 154 | n/t |
| 74 | Capability Surface Modernization Phase 1 | Met exactly | bounded index-width contract, `sparse_scalar_t` public seam, capability docs/proof follow-through | 150 | n/t |
| 75 | Performance Backend Architecture Phase 2 | Met exactly | batched panel-solver dense-kernel seam, public CSC callback/runtime follow-through, benchmark-side measurability refresh | 154 | n/t |
| 76 | Benchmark Governance, Profiling & Longitudinal Reporting | Met exactly | stronger canonical report bundle, longitudinal metadata, support-surface benchmark-governance reconciliation | 153 | n/t |
| 77 | Packaging, ABI & Cross-Platform Quality Convergence Phase 2 | Met exactly | clearer install/export contract, workflow-level macOS/Windows proof clarification, validated install/package close state | 156 | n/t |
| 78 | Large-Source Maintainability Phase 4 & Giant-Test Architecture | Met exactly | LDLT CSC source decomposition, Cholesky CSC giant-test architecture cleanup, chronology/comment debt reduction | 158 | n/t |
| 79 | Numerical Assurance Expansion, Final Integration & Epic 7 Closeout | Met exactly | public LDLT lifecycle oracle/property expansion, final support-surface reconciliation, install-path dependency fix, final Epic 7 closeout | 156 | n/t |
| **Total** | | | | **1 549** | **n/t** |

Epic 7 tracking stayed outcome- and validation-oriented rather than
hour-burn-oriented. The sprint plans therefore remain the authoritative epic
hour baseline.

---

## Cumulative metrics

### Epic-completion metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| nominal planned hours | `1 549` |
| sprint plans present | `10` |
| sprint working-notes files present | `10` |
| sprint retrospectives present | `10` |
| sprint artifact files under `EPIC_7/SPRINT_*/artifacts/` | `150` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake parity target | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |

### Epic 7 execution trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 70 | target-state freeze, architecture contract, non-goal fence | Epic 7 started from one explicit state-of-the-art target, one ranked execution order, and one validation/platform/package truth fence |
| 71-72 | public-surface cleanup plus first product-model convergence | support surfaces became less chronological and more ownership-driven, while direct-workflow matrix-shell semantics became clearer and safer |
| 73-74 | residual configuration modernization plus first capability modernization | the densest env-var/policy contradictions moved behind clearer internal owners, and width/scalar contracts became explicit bounded modernization seams |
| 75-76 | backend/performance architecture plus benchmark governance/reporting | the CSC supernodal Cholesky lane gained a clearer dense-kernel seam and the canonical benchmark surface gained stronger threshold-free longitudinal reporting |
| 77 | packaging, ABI, install/export, and cross-platform quality convergence | the static-first packaging story, install/export surface, and macOS/Windows workflow-proof interpretation became more explicit and more truthful |
| 78 | large-source and giant-test maintainability | the strongest remaining mixed-role implementation and proof hotspots were reduced without widening into broad subsystem or shared-harness redesign |
| 79 | final assurance expansion, final integration, and closeout | the public LDLT repeated-run oracle/property lane strengthened, final support-surface truthfulness was reconciled, and Epic 7 closed from a fresh validated baseline |

### Final validation anchor (Sprint 79 Day 13)

| metric | final |
|---|---:|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `make quality-review-full` | passed |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `451.58 sec` |
| reviewed `test_reorder_nd` time | `315.52 sec` |
| install regression | `11 / 11` |
| CMake install/export regression | `13 / 13` |

### Final Epic 7 end-state markers

| metric | final |
|---|---:|
| canonical maintained benchmark/reporting bundle | `4 CSVs + index.tsv + manifest.txt` |
| maintained install/package proof surfaces | `2` |
| bounded large-`n` lifecycle seeded property lanes | `2` |
| public scalar/type modernization seams landed | `2` |
| public final close baseline owner sprints | `70-79 integrated` |

Representative final close state:

- public support/reference surfaces now read more as durable product and proof
  ownership, less as sprint chronology:
  - `README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- direct one-shot vs repeated-run ownership is clearer:
  - copied factored matrix shells are bounded compatibility surfaces
  - repeated-run direct workflows remain the explicit `analysis` / `factors`
    lifecycle
  - stale solve-ready shell assumptions are reduced
- typed/default/internal policy ownership is stronger:
  - residual FM/graph policy surfaces parse once and lower into clearer
    internal state
  - developer-only debug/profile controls are more explicitly bounded
- capability ceilings now have explicit bounded seams instead of implicit
  typedef-only stories:
  - `SPARSE_IDX_BITS`
  - `SPARSE_SCALAR_BITS`
  - `sparse_idx_bits()`
  - `sparse_scalar_bits()`
- backend-aware acceleration remains intentionally bounded and measurable:
  - CSC supernodal Cholesky dense-kernel descriptor owns the panel-solver seam
  - benchmark output names the current supernodal panel-solver path
- benchmark governance remains intentionally small and threshold-free:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
  - `make bench-canonical-report`
- packaging/platform proof remains intentionally explicit and bounded:
  - local install proof via `tests/test_install.sh`
  - CMake export / `find_package(Sparse)` proof via
    `tests/test_cmake_install.sh`
  - workflow-level macOS and Windows proof described without inflated parity
    claims
- final high-value public assurance now includes:
  - repeated-run LDLT same-pattern oracle coverage in `tests/test_integration.c`
  - large-`n` CSC-backed Cholesky and LDLT lifecycle property coverage in
    `tests/test_fuzz.c`

---

## Stable contracts landed across Epic 7

| sprint | contract | rationale |
|---|---|---|
| 70 | explicit Epic 7 target state, architecture contract, validation/platform/package truth fence | later implementation sprints could move quickly without reopening what Epic 7 was actually trying to ship |
| 71 | support-surface de-chronologization and durable ownership wording | public references became more stable as long-lived product/proof surfaces instead of sprint-history summaries |
| 72 | clearer direct-workflow matrix-shell ownership and reset semantics | product-model convergence advanced without a broad `SparseMatrix` rewrite |
| 73 | single-owner lowering for the strongest residual configuration/policy seams | residual env-var modernization became a real ownership cleanup instead of another audit-only queue |
| 74 | explicit bounded width/scalar capability seams | capability modernization progressed without pretending the repo had already become broadly generic or complex-aware |
| 75 | measurable backend-aware CSC supernodal dense-kernel seam | backend work advanced without turning the repo into a fake general backend framework |
| 76 | stronger threshold-free canonical reporting bundle with longitudinal metadata | benchmark governance improved while preserving the no-threshold interpretation |
| 77 | explicit static-first install/export contract plus bounded platform-proof interpretation | packaging/platform surfaces became more truthful without widening reviewed claims beyond evidence |
| 78 | ownership-first large-source and giant-test maintainability rule | hotspot cleanup improved reviewability without subsystem redesign or shared harness churn |
| 79 | bounded final public oracle/property expansion plus explicit residual queue closeout | Epic 7 ends from measured proof and explicit residuals rather than from generic closeout language |

---

## Residual final limits / follow-up journal

Epic 7 closes without a planned remediation epic to explain away unfinished
work, but it intentionally leaves a bounded post-epic queue in the stable
contract:

1. **Residual direct-family lifecycle/callback parity remains intentionally narrow, not erased.**
   Representative named seams:
   - broader direct-family callback parity beyond the bounded Sprint 79 LDLT
     repeated-run oracle/property lane
   - later Cholesky cancellation-restoration follow-through where evidence and
     payoff justify it

2. **Platform-confidence-limited property expansion remains future work.**
   The repo now has stronger seeded lifecycle/property proof, but it still does
   not claim broad reviewed platform parity for every property lane.

3. **Some family-local oracle/differential broadening remains intentionally deferred.**
   Representative support surfaces:
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt.c`
   - `tests/test_ldlt_csc.c`
   Sprint 79 strengthened the highest-value public seam first rather than
   turning closeout into a proof-expansion wave.

4. **The strongest remaining reviewed runtime concentration is still `test_reorder_nd`.**
   The final validated baseline is strong, but `test_reorder_nd` remains the
   clearest operational long pole in reviewed sweeps.

5. **Some bounded maintainability follow-through remains outside Epic 7’s closeout scope.**
   Representative named seams inherited from the sprint carry-forward queues:
   - `src/sparse_iterative.c`
   - `tests/test_ldlt_csc.c`
   - `src/sparse_chol_csc.c`
   - `tests/test_qr.c`

6. **Packaging/platform truth remains intentionally bounded.**
   Epic 7 tightened the static-first package/install/export story, but it still
   does not claim:
   - broad shared-library or dynamic-ABI guarantees
   - fake Windows/macOS proof symmetry
   - wider platform support than the maintained evidence justifies

These are post-Epic-7 boundaries, not hidden closeout defects.

---

## Lessons (Epic-level)

- **Boundary-first sequencing paid off again.** Sprint 70 and the later audit /
  design days kept the epic from diffusing into generic modernization work.

- **Public-surface truthfulness and implementation modernization reinforced each other.**
  Sprint 71 through Sprint 77 repeatedly showed that better ownership wording,
  better lifecycle semantics, and better bounded policy surfaces are often more
  valuable than trying to widen capability claims prematurely.

- **Bounded modernization beats framework theater.** Sprint 73, Sprint 74,
  Sprint 75, Sprint 76, and Sprint 77 all landed better because they chose one
  real ownership seam and one real proof boundary instead of pretending the repo
  was ready for universal configuration, backend, reporting, or platform
  frameworks.

- **Capability and assurance work are strongest when they expose explicit seams.**
  Sprint 74 and Sprint 79 succeeded because they made width/scalar and
  lifecycle/property boundaries explicit instead of hiding them behind internal
  convention or audit prose.

- **Closeout work is highest-value when it can still improve the tree.**
  Sprint 79 did not just summarize the final state; it found and fixed a real
  clean-install dependency bug during validation. That is the right model for a
  final sprint.

- **Good no-op decisions are part of engineering discipline.**
  Several Epic 7 sprints closed support or summary lanes as explicit no-ops
  rather than forced edits. That kept the epic aligned to real contradictions
  instead of generating low-value churn.

---

## DoD verification

Required cross-epic themes from `PROJECT_PLAN.md`:

- Epic 7 baseline, target state, and architecture contract: complete
- public-surface de-chronologization and reference cleanup: complete
- product-model convergence phase 1: complete
- configuration-surface modernization phase 2: complete
- capability-surface modernization phase 1: complete
- performance backend architecture phase 2: complete
- benchmark governance, profiling, and longitudinal reporting: complete
- packaging, ABI, and cross-platform quality convergence phase 2: complete
- large-source maintainability and giant-test architecture follow-through:
  complete
- numerical assurance expansion, final integration, and Epic 7 closeout:
  complete

Final measured close state:

- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- install/export proof:
  - `tests/test_install.sh` passing
  - `tests/test_cmake_install.sh` passing
- canonical benchmark/reporting proof:
  - `make bench-canonical-report` passing
- final public/product/proof/package story reconciled across:
  - README
  - INSTALL
  - tutorial
  - examples
  - benchmarks
  - maintainer guide
  - tests
  - install/export proof
- no hidden Sprint 80 remediation sprint is needed to explain away unfinished
  Epic 7 planned work: complete
