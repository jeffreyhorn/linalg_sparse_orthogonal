# Epic 3 Retrospective — Sprints 30-39 (linalg_sparse_orthogonal)

**Epic budget:** 10 sprints × ~120-148 h each = **~1 308 hours nominal**  
**Branch range:** `sprint-30` → `sprint-39`  
**Started:** Sprint 30 (review-driven quality-hardening kickoff after
`reviews/review-codex-2026-05-15.md`)  
**Closed:** Sprint 39 Day 14 (2026-05-22)  
**Goal:** Close the review-driven quality-hardening backlog from
`reviews/review-codex-2026-05-15.md` +
`reviews/todo-codex-2026-05-15.md` across compile hygiene, benchmark/tooling
sync, test-suite truthfulness, dead-code detection, build-quality enforcement,
public docs/API consistency, cross-platform parity, maintainability cleanup,
and final Epic closeout.

> **Status: Epic 3 complete.** All 10 sprints landed. Repository-wide
> full-tree warnings dropped **123 → 0** (Sprint 30 baseline to Sprint 32
> close). The test-side dormant-scaffold and warning queue closed to zero
> (Sprint 32). The dead-code workflow shipped (`make deadcode`,
> `make deadcode-report`, `make deadcode-check`; Sprint 33), then the
> benchmark/example compile-db gap closed **7 → 0** and the internal dead-code
> cleanup queue closed **1 → 0** (Sprints 33 and 38). Reviewed-quality wrappers
> and reviewed CMake parity became named maintained commands (Sprint 34),
> `make quality-review-full` became the strongest local reviewed baseline
> (Sprint 38), and the final cross-platform enforced/staged/excluded contract
> was made explicit (Sprint 36) and closed out truthfully (Sprint 39). Epic 3
> ends with a measured final validation baseline, a concise quality-readiness
> checklist, and no new deferred cleanup sprint.

---

## Summary table

Per-sprint deliverables + nominal hours from `PROJECT_PLAN.md`. Epic 3 sprint
retrospectives record measured outcomes and validation results, but they do not
track separate actual-hour totals, so the `actual h` column is left `n/t`.

| sprint | title | key deliverables | nominal h | actual h |
|---|---|---|---:|---:|
| 30 | Warning Baseline & Core Compile-Hygiene Triage | Apple Clang full-tree warning baseline, `src/` warning cleanup, compile-hygiene playbook, repeatable warning workflow, auxiliary warning queues | 120 | n/t |
| 31 | Benchmark Tooling Sync & Portability Cleanup | `bench_main` sync, benchmark/example portability cleanup, designated initializers in tooling surface, `make tooling-build` compile-only gate | 120 | n/t |
| 32 | Test-Suite Truthfulness & Dormant Scaffold Resolution | dormant `test_reorder_nd.c` cleanup, opt-in/skip test policy, test warning queue closed to zero, active suite truthfulness formalized | 128 | n/t |
| 33 | Dead-Code Detection Infrastructure & First Cleanup Pass | `make deadcode`, `make deadcode-report`, `make deadcode-check`, dead-code policy, first internal cleanup, public-surface keep audit | 124 | n/t |
| 34 | Build-Quality Enforcement Phase 1 | reviewed Makefile wrappers, reviewed CMake parity wrappers, Linux CI enforcement pass, failure UX, regression-prevention initializer cleanup | 136 | n/t |
| 35 | Public Docs, Header Examples & API-Usage Consistency | header/example audit, README/tutorial/API wording cleanup, designated-initializer doc rule, support-doc reconciliation | 128 | n/t |
| 36 | Cross-Platform Quality Parity | explicit Linux/macOS/Windows enforced/staged/excluded model, macOS workflow alignment, Windows reviewed CMake subset alignment, parity report | 144 | n/t |
| 37 | Auxiliary-Code Cleanup & Maintainability Refactor | narrow shared test/benchmark helpers, Makefile/report maintainability cleanup, workflow-doc compression, operator-surface normalization | 148 | n/t |
| 38 | Coverage, Regression-Proofing & Quality-Gate Expansion | coverage-honesty cleanup, dead-code compile-db gap closure, dead-code closeout wording, `make quality-review-full`, readiness checklist | 136 | n/t |
| 39 | Epic 3 Stabilization, Final Audit & Closeout | final warning/dead-code/platform audits, standards closeout, scaffolding cleanup, Epic summary, final validation, final handoff | 124 | n/t |
| **Total** | | | **1 308** | **n/t** |

Epic 3 retrospectives measured outcomes, validation, and residual limits rather
than hour burn. The planning baseline therefore remains the authoritative epic
hour total.

---

## Cumulative metrics

### Repository-wide warning trajectory (Sprints 30 → 32)

| sprint | full-tree warnings | `src` | `tests` | `benchmarks` | `examples` | landed change |
|---|---:|---:|---:|---:|---:|---|
| 30 Day 1 baseline | 123 | 11 | 98 | 13 | 1 | baseline captured |
| 30 close | 112 | 0 | 98 | 13 | 1 | core-library warning cluster removed |
| 31 close | 98 | 0 | 98 | 0 | 0 | benchmark/example warning queue closed |
| **32 close** | **0** | **0** | **0** | **0** | **0** | **test-suite warning queue + dormant scaffold cleanup closed** |

Sprint 32 is the decisive warning-close point for Epic 3:

- `-Wmissing-field-initializers`: `72 → 0`
- `-Wdouble-promotion`: `45 → 0`
- `-Wunused-function`: `3 → 0`
- `-Wimplicit-function-declaration`: `2 → 0`
- `-Wswitch`: `1 → 0`

### Active-suite / reviewed CMake truthfulness trajectory

| sprint | reviewed/CTest suite truth | landed change |
|---|---:|---|
| 32 close | `53` registered tests | active/opt-in/historical split formalized; dormant `RUN_TEST(...)` scaffold removed |
| 33 close | `53` registered tests | dead-code tooling added without regressing active-suite truth |
| 34 close | `53 / 53` reviewed CMake parity passing | reviewed CMake parity wrappers named and enforced locally |
| 36 close | `53 / 53` cross-platform reviewed CMake baseline preserved | strongest shared reviewed baseline clarified across Linux/macOS/Windows |
| **39 close** | **`53 / 53` passing** | **final closeout still grounded in the same auditable suite size** |

### Dead-code maturation trajectory (Sprints 33 → 39)

| sprint | `coverage-gap` | internal queue | public keep/review | secondary signals | noise | landed change |
|---|---:|---:|---:|---:|---:|---|
| 33 first classified report | 7 | 1 | 4 | 35 | 6 | initial reporting buckets land |
| 33 close | 7 | 0 | 4 | 35 | 6 | first definitely-unused internal helper removed |
| 38 close | 0 | 0 | 4 | 35 | 6 | benchmark/example compile-db gap closed |
| **39 close** | **0** | **0** | **4** | **35** | **6** | **residual buckets explicitly dispositioned as justified/supporting/noise context** |

Important closeout state:

- the dead-code workflow is now useful and truthful
- it is still a completeness/reporting tool, not a zero-findings claim
- authoritative execution remains serialized because of the shared
  `build/deadcode-cmake` / `build/deadcode/` topology

### Quality-baseline / enforcement trajectory (Sprints 34 → 39)

| sprint | landed quality contract change |
|---|---|
| 34 | reviewed Makefile wrappers + reviewed CMake parity wrappers + Linux CI phase-1 enforcement |
| 35 | public docs/examples/headers reconciled to current API behavior |
| 36 | explicit Linux/macOS/Windows enforced/staged/excluded contract |
| 37 | maintainability cleanup without changing reviewed/dead-code/operator behavior |
| 38 | `make quality-review-full` + README readiness checklist + dead-code compile-db gap closure |
| **39** | final measured baseline + Epic summary + no-new-deferred-queue closeout |

### Final Epic 3 measured close state (Sprint 39 Day 13)

| metric | final |
|---|---:|
| `make format` wall time | `4.35 s` |
| `make lint` wall time | `374.79 s` |
| `make test` wall time | `87.33 s` |
| `make quality-review-full` wall time | `543.79 s` |
| reviewed CMake `ctest -N` | `53` |
| reviewed CMake full `ctest` | `53 / 53` |
| full reviewed CMake `ctest` real time | `143.93 s` |
| serial `make deadcode-report` | `0.27 s` |
| serial `make deadcode-check` | `0.47 s` |

---

## Stable quality contracts landed across Epic 3

| sprint | contract | rationale |
|---|---|---|
| 30 | Sprint 30 warning workflow + Apple Clang CMake full-tree inventory as repository-wide warning authority | separates routine local quality from authoritative full-tree warning proof |
| 31 | `make tooling-build` compile-only benchmark/example gate, wired into `make lint` | catches tooling drift without expensive runtime benches |
| 32 | live `RUN_TEST_SLOW(...)`, `RUN_TEST_EXPERIMENTAL(...)`, `SKIP_TEST(...)` truth surface | keeps active, opt-in, and historical tests honest |
| 33 | `make deadcode`, `make deadcode-report`, `make deadcode-check` | conservative dead-code evidence/reporting flow with explicit buckets |
| 34 | `make quality-review-compile`, `make quality-review`, `make quality-review-cmake-compile`, `make quality-review-cmake` | named reviewed-quality wrapper model without breaking legacy command meaning |
| 35 | public non-default examples should use designated initializers; README/tutorial/header ownership split | stabilizes the public usage contract |
| 36 | explicit Linux/macOS/Windows enforced/staged/excluded CI contract | replaces Linux-implied quality expectations with a truthful cross-platform model |
| 37 | quality-target ownership model + permanent-file comment cleanup boundaries | improves maintainability without changing behavior |
| 38 | `make quality-review-full` as strongest local reviewed baseline; README readiness checklist | creates one concise local pre-feature/pre-PR baseline |
| 39 | final Epic 3 summary + closeout package | hands the repo back to normal feature work from one measured validated baseline |

---

## Residual final limits / follow-up journal

Epic 3 closes without a new planned cleanup sprint, but it does intentionally
leave a few bounded limits in the stable post-epic contract:

1. **Repository-wide warning proof still uses the Sprint 30 authority model.**
   Future work should not collapse that into `make quality-review-full` alone.

2. **Dead-code execution remains serialized.**
   The workflow is truthful and useful, but the shared
   `build/deadcode-cmake` / `build/deadcode/` paths still block
   concurrency-safe use.

3. **macOS dead-code remains staged.**
   The Apple Clang reviewed path is enforced; the dead-code flow is still not a
   maintained macOS enforcement surface.

4. **Windows local Makefile reviewed-wrapper parity remains staged.**
   Windows closes with an enforced reviewed CMake subset rather than full local
   Makefile parity.

5. **Windows dead-code remains excluded.**
   That remains an honest closeout boundary, not a hidden backlog item.

6. **Coverage remains calibrated to operating reality, not aspiration.**
   The enforced threshold remains `80%` line coverage on `src/` in the Linux
   coverage path until a later feature effort justifies broader instrumentation
   or synthetic-fault-injection scaffolding.

These are post-Epic-3 contract boundaries, not newly surfaced Sprint 39 debt.

---

## Lessons (Epic-level)

- **Audit-first sequencing prevented fake cleanup work.** Sprints 30, 32, 33,
  35, 36, 38, and 39 all benefited from auditing the real queue before
  implementation. That consistently kept later code/doc batches smaller,
  attributable, and truthful.

- **Evidence hierarchy matters in quality-hardening epics.** Epic 3 succeeded
  because it kept different claims separate:
  - repository-wide warning proof
  - routine local reviewed baseline
  - dead-code completeness/reporting
  - cross-platform enforced vs staged vs excluded surfaces

- **Preserve command meaning; layer stronger wrappers above it.** Sprint 34’s
  wrapper design was one of the most important epic-level calls. Adding
  `quality-review-*` and later `quality-review-full` avoided breaking existing
  maintainer habits while still making the quality baseline explicit.

- **Truthful staged/excluded boundaries are better than fake parity.** Sprint 36
  and Sprint 39 closed cross-platform work by naming the real Windows/macOS
  limits instead of pretending all platforms support the same local contract.

- **Historical engineering evidence belongs in `docs/planning/`.** Sprint 32,
  Sprint 35, Sprint 37, and Sprint 39 converged on the same boundary:
  permanent operator-facing files should hold stable contract language, while
  day-by-day rationale, measurements, and retired experiments belong in sprint
  artifacts.

- **Small shared layers beat broad new frameworks in cleanup work.** Sprint 33
  chose reporting buckets over raw tool noise, Sprint 37 chose narrow helper
  headers over broad test/benchmark frameworks, and Sprint 39 chose comment
  cleanup over new standards files. Those scope calls kept behavior legible.

- **Validation logs in-tree make closeout materially easier.** Sprint 38 and
  Sprint 39 show the value of storing measured end-state logs/artifacts in-tree:
  Day 14 closeout no longer depends on reconstructing timing or pass/fail state
  from memory.

---

## DoD verification

Required cross-epic gates:

- repository-wide warning debt reduced from `123` to `0` on the authoritative
  Apple Clang CMake full-tree path: ✓
- benchmark/example compile drift closed and compile-only tooling gate added:
  ✓
- dormant test scaffold removed and active/opt-in/historical test truthfulness
  formalized: ✓
- dead-code targets implemented, documented, and kept truthful through final
  closeout: ✓
- dead-code benchmark/example compile-db gap reduced from `7` to `0`: ✓
- reviewed Makefile wrappers, reviewed CMake parity wrappers, and Linux
  enforcement path landed: ✓
- public docs, header examples, tutorial wording, and support docs reconciled
  to current API behavior: ✓
- cross-platform enforced/staged/excluded model documented and validated: ✓
- strongest local reviewed baseline command + readiness checklist landed: ✓
- final Epic 3 summary, measured validation baseline, handoff, and
  retrospective written: ✓

Final measured close state:

- direct maintained gates: passing
- `make quality-review-full`: passing
- reviewed CMake parity: `53 / 53`
- serial `make deadcode-report`: passing
- serial `make deadcode-check`: passing
- no new deferred cleanup sprint required: ✓

---

## Key references

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [review-codex-2026-05-15.md](./reviews/review-codex-2026-05-15.md)
- [todo-codex-2026-05-15.md](./reviews/todo-codex-2026-05-15.md)
- [SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md](./SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md)
- [SPRINT_39/HANDOFF.md](./SPRINT_39/HANDOFF.md)
- [SPRINT_39/RETROSPECTIVE.md](./SPRINT_39/RETROSPECTIVE.md)
- [SPRINT_39/artifacts/day12-epic3-summary-report.md](./SPRINT_39/artifacts/day12-epic3-summary-report.md)
- [SPRINT_39/artifacts/day13-full-validation-sweep.md](./SPRINT_39/artifacts/day13-full-validation-sweep.md)

## Bottom line

Epic 3 achieved its engineering goal:

- the repo moved from a review-discovered quality backlog to a measured stable
  baseline
- full-tree warning debt closed to zero
- dead-code tooling, reporting, and disposition became real and reviewable
- reviewed local and CMake parity commands became named maintained surfaces
- public docs and examples now teach current stable behavior
- cross-platform limits are explicit and truthful rather than implied
- and the final closeout package now hands the repo back to normal feature work
  without another cleanup sprint

The next feature branch should treat Epic 3 as complete and inherit its quality
contracts directly rather than continuing the closeout program.
