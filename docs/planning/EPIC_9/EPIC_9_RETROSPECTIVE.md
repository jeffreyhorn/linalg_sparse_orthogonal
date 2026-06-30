# Epic 9 Retrospective - Sprints 90-99 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprint project-plan estimates = **1,666 hours nominal**
**Day-plan nominal total:** **1,670 hours** after Sprint 90 and Sprint 93 were
rounded to 14-day day-cap plans
**Branch range:** `sprint-90` -> `sprint-99`
**Started:** Sprint 90 (Epic 9 baseline, state-of-the-art target, and product
contract)
**Closed:** Sprint 99 Day 14 (Sprint 99 and Epic 9 closeout)
**Goal:** Close the post-Epic-8 residual queue from a bounded, truthful,
validated baseline across product-model convergence, dense/backend maturity,
runtime scalability, capability-surface modernization, public narrative
coherence, large-source and giant-test maintainability, build/package/platform
convergence, assurance and external comparison architecture, and final Epic 9
competitive calibration.

> **Status: Epic 9 complete.** All 10 sprints landed. The epic opened by
> freezing the post-Epic-8 target state, comparison/measurement contract,
> product-model contradiction map, and non-goal fence (Sprint 90). It then
> closed the main remaining seams in order: compressed-first construction and
> lifecycle convergence (Sprint 91), portable dense/backend maturity (Sprint
> 92), runtime scalability and nested-dissection convergence (Sprint 93),
> scalar/index capability-surface modernization (Sprint 94), public narrative
> and workflow coherence (Sprint 95), large-source and giant-test
> maintainability (Sprint 96), build/package/cross-platform convergence
> (Sprint 97), assurance and external comparison expansion (Sprint 98), and
> final evidence-backed Epic 9 closeout (Sprint 99). Epic 9 ends with the
> strongest local reviewed baseline intact (`make quality-review-full`),
> reviewed Make/CMake parity exact (`54` vs `54`, `54 / 54` passing), static
> install/export proof passing (`14 / 14`), CMake install/export proof passing
> (`16 / 16`, `0` skips), bounded benchmark/reporting evidence regenerated,
> and one explicit post-Epic-9 residual queue instead of hidden closeout drift.

---

## Summary Table

Per-sprint deliverables + nominal hours from the project plan and sprint
day-plans. Like the earlier epic retrospectives, Epic 9 sprint retrospectives
and closeout artifacts track outcomes, validation, and bounded residuals
rather than separate actual-hour totals, so the `actual h` column remains
`n/t`.

| sprint | title | plan assessment | key deliverables | project-plan h | day-plan h | actual h |
|---|---|---|---|---:|---:|---:|
| 90 | Epic 9 Baseline, State-of-the-Art Target & Product Contract | Met exactly | live Epic 9 contradiction map, product target, comparison/measurement contract, non-goal fence, review/todo/plan package | 166 | 168 | n/t |
| 91 | Compressed-First Product Convergence Phase 3 | Met exactly | CSR/CSC construction/import entry paths, compressed-first product contract, lifecycle/public-surface follow-through, validated closeout | 168 | 168 | n/t |
| 92 | Portable Dense Backend & Kernel Maturity Phase 4 | Met exactly | shared dense-backend descriptor, builtin fallback truth, LDLT backend adoption, observability and benchmark follow-through | 168 | 168 | n/t |
| 93 | Runtime Scalability, Threading & ND Convergence Phase 2 | Met exactly | ND recursive runtime reduction, runtime-control cleanup, bounded runtime evidence, reviewed runtime baseline | 166 | 168 | n/t |
| 94 | Capability Surface Modernization Phase 3 | Met exactly | scalar/index capability contract, matrix-shell scalar widening, index/ABI maturity follow-through, bounded proof/docs/package alignment | 168 | 168 | n/t |
| 95 | Public Narrative, Docs & Workflow Coherence | Met exactly | README/tutorial/example/header/support-surface cleanup, proof-owner naming cleanup, explicit residual queue | 164 | 164 | n/t |
| 96 | Large-Source & Giant-Test Maintainability Phase 6 | Met exactly | LDLT dense/backend extraction, iterative block-wrapper extraction, Cholesky CSC giant-test split, full validation closeout | 168 | 168 | n/t |
| 97 | Build, Packaging & Cross-Platform Product Convergence Phase 4 | Met exactly | source-list manifest/checker, reviewed source-list gate, static-first package decision, install/export proof strengthening, workflow/platform truth calibration | 166 | 166 | n/t |
| 98 | Assurance, External Comparison & Coverage Architecture Phase 3 | Met exactly | LDLT CSC external dense-reference lane, bounded reorder/fill artifact, benchmark-governance guardrails, assurance topology snapshot | 168 | 168 | n/t |
| 99 | Final Integration, Competitive Calibration & Epic 9 Closeout | Met exactly | end-state contradiction re-audit, final comparison and surface validation, residual queue, Sprint/Epic retrospectives, post-Epic-9 handoff | 164 | 164 | n/t |
| **Total** | | | | **1,666** | **1,670** | **n/t** |

Epic 9 tracking stayed outcome- and validation-oriented rather than
hour-burn-oriented. The project-plan estimates remain the authoritative epic
scope baseline; the sprint day-plans remain the day-by-day execution baseline.

---

## Cumulative Metrics

### Epic-Completion Metrics

| metric | final |
|---|---:|
| planned sprints completed | `10 / 10` |
| project-plan nominal hours | `1,666` |
| day-plan nominal hours | `1,670` |
| sprint plans present | `10` |
| sprint working-notes files present | `10` |
| sprint retrospectives present | `10` |
| sprint artifact files under `EPIC_9/SPRINT_*/artifacts/` | `148` |
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed Make/CMake parity target | `54 vs 54` |
| full reviewed CMake `ctest` | `54 / 54` |
| final unsupported broad claims found | `0` |
| post-Epic-9 carry-forward items | `8` |

### Epic 9 Execution Trajectory

| sprint band | planned focus | landed outcome |
|---|---|---|
| 90 | target-state freeze, product contract, comparison/measurement contract, non-goal fence | Epic 9 started from one explicit contradiction map and one bounded state-of-the-art target instead of inheriting a generic modernization story |
| 91-92 | compressed-first product convergence and portable dense/backend maturity | compressed-first construction/import became first-class on high-value paths, and dense/backend maturity moved behind a shared bounded backend seam with builtin fallback truth preserved |
| 93-94 | runtime scalability and capability-surface modernization | the ND runtime long pole was reduced, runtime-control interpretation tightened, and scalar/index capability seams became more explicit without broad complex or mixed-precision claims |
| 95 | public narrative, docs, workflow, and proof-owner naming coherence | permanent public/support surfaces became less sprint-historical and more product-owner oriented |
| 96 | large-source and giant-test maintainability | selected large implementation and proof-owner seams were reduced without broad subsystem rewrites |
| 97 | build, package, source-list, and cross-platform convergence | Make/CMake source-list drift gained a reviewed guard, static-first package proof sharpened, and platform asymmetry became explicit |
| 98 | assurance, external comparison, coverage, and benchmark architecture | the project gained a maintained LDLT CSC external dense-reference lane and bounded reorder/fill calibration while preserving benchmark non-claims |
| 99 | final integration, competitive calibration, validation, and closeout | Epic 9 closed from a final contradiction re-audit, no-op final fix decision, explicit residual queue, `make quality-review-full`, and surface-validation sweep |

### Final Validation Anchor (Sprint 99 Days 10-11)

| metric | final |
|---|---:|
| `make quality-review-full` | passed |
| Makefile reviewed path | `format-check + lint + test + deadcode-check` passed |
| reviewed CMake `ctest -N` | `54` |
| Makefile test count | `54` |
| Make/CMake test-count parity | passed |
| full reviewed CMake `ctest` | `54 / 54` |
| Make install/export proof | `14` passed, `0` failed |
| CMake install/export proof | `16` passed, `0` failed, `0` skipped |
| examples built | `12` |
| representative examples executed | `4` |
| bounded reorder/fill command | `make bench-reorder-sprint86` passed |
| canonical benchmark report | `make bench-canonical-report` passed |
| final diff hygiene | `git diff --check` passed |
| final trailing-whitespace scan | passed on Sprint 99 and Epic 9 closeout docs |

### Final Epic 9 End-State Markers

| metric | final |
|---|---:|
| maintained external direct-family comparison lanes | `2` |
| maintained install/package proof surfaces | `2` |
| canonical benchmark report files regenerated | `6` |
| post-Epic-9 residual queue items | `8` |
| deliberate non-claims preserved | `10` |
| unsupported positive broad claims to remove | `0` |

Representative final close state:

- product/storage reading is less linked-list-first on the highest-value
  touched paths:
  - CSR/CSC construction/import entry points are first-class public paths
  - the linked-list shell remains the mutable compatibility owner
  - direct repeated-run lifecycle ownership is clearer
- backend maturity is stronger but still bounded:
  - builtin dense paths remain authoritative fallback truth
  - portable backend seams improved without becoming a broad backend framework
  - LDLT backend adoption and observability are stronger than at epic start
- runtime/reporting is more calibrated:
  - ND runtime work reduced reviewed runtime pressure
  - bounded reorder/fill evidence names fixtures and `nnz_L` as the
    claim-bearing fill field
  - timing remains local context rather than a portable product claim
- capability surfaces are clearer:
  - scalar/index ownership and public helper seams improved
  - broad complex and mixed-precision maturity remain explicit non-claims
- public and support surfaces are more product-oriented:
  - README/tutorial/examples/install/benchmark/maintainer surfaces were
    reduced, renamed, or consolidated where history had obscured ownership
- maintainability improved on selected owners:
  - LDLT dense/backend source ownership moved out of the densest path
  - iterative block-wrapper ownership became clearer
  - Cholesky CSC proof ownership split into more reviewable owners
- package/platform proof is sharper and intentionally bounded:
  - static-first install/export proof is maintained
  - CMake consumer proof is maintained
  - Windows remains CMake-first reviewed subset, not Makefile or install
    parity
- assurance is stronger but not universal:
  - Cholesky CSC and LDLT CSC have maintained external dense-reference solve
    checks on named fixtures
  - every-solver-family external comparison remains future architecture work

---

## Stable Contracts Landed Across Epic 9

| sprint | contract | rationale |
|---|---|---|
| 90 | explicit Epic 9 target state, product-model contradiction map, comparison/measurement contract, and non-goal fence | later implementation sprints could move without reopening what Epic 9 was actually trying to prove or refusing to claim |
| 91 | compressed-first construction/import as first-class public entry with linked-list shell retained as compatibility owner | product convergence advanced without pretending the whole matrix shell had been replaced |
| 92 | shared bounded dense/backend seam with builtin fallback truth | backend maturity improved without turning optional acceleration into a universal platform claim |
| 93 | algorithmic runtime reduction first, runtime-control cleanup second, proof topology only where evidence required it | runtime work reduced cost without inventing a broad timing gate |
| 94 | bounded scalar/index capability widening with explicit real-only and non-claim fences | capability modernization progressed without claiming broad complex, mixed-precision, or family-wide generic numeric maturity |
| 95 | permanent support surfaces should read as product/proof ownership, not sprint chronology | public narrative became more stable and less historical |
| 96 | family-local extraction over broad refactor campaigns | maintainability improved without unsafe subsystem rewrites |
| 97 | reviewable source-list guard and static-first package proof | build/package convergence reduced drift without hiding source membership or claiming shared-library maturity |
| 98 | named external correctness lanes plus bounded reorder/fill artifact | assurance expanded without turning local comparison evidence into broad ecosystem or timing claims |
| 99 | evidence-backed closeout with explicit no-op final fix decision and residual queue | Epic 9 ends from validated proof and explicit non-claims rather than generic closeout prose |

---

## Residual Final Limits / Follow-Up Journal

Epic 9 closes without a planned remediation epic to explain away unfinished
work, but it intentionally leaves a bounded post-epic queue in the stable
contract:

1. **The linked-list shell remains the public mutable compatibility owner.**
   Compressed-first entry paths improved materially, but Epic 9 does not claim
   full compressed-first replacement of the shell.

2. **External comparison depth remains deliberately narrow.**
   Maintained lanes now include named Cholesky CSC and LDLT CSC external
   dense-reference solve checks, but the following remain future architecture:
   - broader LDLT CSC Matrix Market or indefinite corpus comparison
   - iterative solver external comparison
   - eigensolver/LOBPCG external comparison
   - QR/SVD external comparison
   - broader ecosystem comparison

3. **Benchmark evidence remains local and bounded.**
   `bench-reorder-sprint86` and `bench-canonical-report` are useful
   calibration/reporting surfaces, but Epic 9 does not claim:
   - portable timing superiority
   - universal reorder/fill superiority
   - benchmark supremacy

4. **Large-source and giant-test concentration remain active debt.**
   Sprint 96 improved selected owners, but representative remaining seams still
   require future family-local extraction:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_integration.c`
   - large direct, QR, eigensolver, SVD, iterative, matrix, and graph owners

5. **Capability widening remains intentionally bounded.**
   Epic 9 improved scalar/index ownership and selected surfaces, but does not
   claim:
   - broad complex support
   - broad mixed-precision maturity
   - broad backend-neutral acceleration maturity

6. **Packaging/platform truth remains intentionally asymmetric.**
   Epic 9 sharpened static-first package proof and CMake consumer proof, but it
   still does not claim:
   - shared-library-first product maturity
   - dynamic ABI guarantees
   - package-manager integration
   - Windows Makefile parity
   - Windows install-validation
   - symmetric Linux/macOS/Windows reviewed parity

7. **Chronology cleanup is complete only on the highest-value public/support surfaces.**
   Public and maintainer surfaces are cleaner, but lower-level tests,
   implementation comments, and historical planning artifacts still contain
   chronology where it is either archival or lower priority.

These are post-Epic-9 boundaries, not hidden closeout defects.

---

## Lessons (Epic-Level)

- **Contradiction maps scale better than broad themes.** Sprint 90 and Sprint
  99 were useful because they named the durable contradictions and classified
  them instead of writing generic modernization language.

- **Closeout quality depends on non-claims.** Epic 9 is stronger because it
  explicitly refuses broad complex/mixed-precision maturity, shared-library
  maturity, fake platform parity, dynamic ABI, and timing superiority claims.

- **Bounded implementation beats framework theater.** The most successful
  sprints chose one real owner seam and one proof boundary: compressed-first
  construction/import, shared dense backend seams, ND runtime reduction,
  source-list checking, LDLT CSC external reference, and final validation.

- **Package proof is strongest when it proves negative shape too.** The final
  static-first scripts assert that no shared-library artifacts are installed,
  which makes the product claim sharper than a vague install smoke test.

- **Benchmark/reporting surfaces need claim-bearing fields.** Epic 9 preserved
  `nnz_L` as the bounded fill field and kept runtime as local context; this
  prevented benchmark output from turning into unsupported product claims.

- **Make/CMake parity checks reduce drift without fake platform symmetry.**
  Local Make/CMake test-count parity is valuable, while Windows remains a
  smaller reviewed CMake-first subset with explicit exclusions and expected
  counts.

- **No-op final fix decisions are real engineering output.** Sprint 99 avoided
  low-value closeout churn by recording why residual work and non-claims did
  not justify a final implementation batch.

- **Documentation-heavy closeout still needs validation.** Sprint 99 changed
  planning documentation, but it closed only after `make quality-review-full`
  and surface-validation commands had already proven the relevant code/build
  surfaces.

---

## DoD Verification

Required Epic 9 themes from `PROJECT_PLAN.md`:

- Epic 9 baseline, state-of-the-art target, and product contract: complete
  - Sprint 90 fixed the live contradiction map, target state, comparison
    contract, and non-goal fence.
- Compressed-first product convergence: complete within bounded scope
  - Sprint 91 promoted CSR/CSC construction/import paths while retaining the
    linked-list shell as mutable compatibility owner.
- Portable dense backend and kernel maturity: complete within bounded scope
  - Sprint 92 widened backend seams and LDLT adoption while preserving builtin
    fallback truth.
- Runtime scalability, threading, and ND convergence: complete within bounded
  scope
  - Sprint 93 reduced ND runtime pressure and tightened runtime-control
    interpretation without creating broad timing claims.
- Capability surface modernization: complete within bounded scope
  - Sprint 94 widened scalar/index seams and proof/docs support while keeping
    broad complex and mixed-precision maturity out of the claim set.
- Public narrative, docs, and workflow coherence: complete
  - Sprint 95 cleaned public/support surfaces and proof-owner names while
    preserving historical planning artifacts as historical.
- Large-source and giant-test maintainability: complete within bounded scope
  - Sprint 96 reduced selected source and proof-owner hotspots and carried
    remaining large owners forward explicitly.
- Build, packaging, and cross-platform product convergence: complete within
  bounded scope
  - Sprint 97 added source-list guardrails, sharpened static-first install
    proof, and preserved asymmetric platform truth.
- Assurance, external comparison, and coverage architecture: complete within
  bounded scope
  - Sprint 98 added LDLT CSC external dense-reference proof and bounded
    reorder/fill calibration without broadening benchmark or coverage claims.
- Final integration, competitive calibration, and Epic 9 closeout: complete
  - Sprint 99 produced the final evidence package, residual queue, validation
    sweep, Sprint retrospective, Epic retrospective, and post-Epic-9 handoff.

Final validation and closeout artifacts:

- [Sprint 99 closeout evidence package](./SPRINT_99/artifacts/day12-closeout-evidence-package.md)
- [Sprint 99 final residual queue](./SPRINT_99/artifacts/day9-final-residual-queue.md)
- [Sprint 99 reviewed validation](./SPRINT_99/artifacts/day10-reviewed-validation.md)
- [Sprint 99 surface validation](./SPRINT_99/artifacts/day11-surface-validation.md)
- [Sprint 99 retrospective](./SPRINT_99/RETROSPECTIVE.md)
- [Post-Epic-9 handoff](./POST_EPIC_9_HANDOFF.md)

---

## Bottom-Line Closeout

Epic 9 achieved its real target: a bounded, state-of-the-art sparse linear
algebra product surface that is more compressed-first on high-value paths,
more mature on selected backend/direct-family seams, clearer in scalar/index
and package/platform contracts, stronger in named external comparison lanes,
and better documented at public/support boundaries.

Epic 9 did not claim universal maturity. The final closeout preserves the
linked-list compatibility shell, static-first package contract, asymmetric
platform proof, local benchmark interpretation, bounded external comparison
lanes, and explicit post-Epic-9 residual queue.

That is the correct close state: stronger evidence, clearer product claims,
and no hidden unsupported promises.
