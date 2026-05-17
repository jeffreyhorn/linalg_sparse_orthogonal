# Sprint 30 Retrospective — Warning Baseline, Core Compile Hygiene & Cleanup Queueing

**Sprint budget:** 14 working days (~120 hours estimated, per `PLAN.md`)
**Branch:** `sprint-30`
**Calendar elapsed:** 2026-05-15 → 2026-05-16 (intensive condensed run; the day budget tracks engineering effort, not wall-clock days)

> **Status:** Day 14 final. Sprint 30 closed all seven
> `PROJECT_PLAN.md` items without expanding into feature work.
> The sprint established the first authoritative Apple Clang CMake
> full-tree warning baseline, removed the entire library-proper
> warning cluster (`src`: 11 -> 0), documented a compile-hygiene
> policy, automated the warning reproduction workflow, and converted
> the remaining warning debt into a concrete queued backlog for
> Sprint 31 and later. The repository is not yet full-tree
> warning-clean: the final measured state remains `112` warnings
> (`tests`: 98, `benchmarks`: 13, `examples`: 1), but the warning
> debt is now measured, reproducible, and explicitly prioritized.

## Goal recap

> Establish a measured warning baseline for the full tree, fix the
> highest-signal core-library warnings first, and create a repeatable
> compile-quality workflow that later sprints can enforce.

(See [PLAN.md](./PLAN.md) for the day-by-day breakdown; see [COMPILE_HYGIENE_PLAYBOOK.md](./COMPILE_HYGIENE_PLAYBOOK.md), [REBUILD_WORKFLOW.md](./REBUILD_WORKFLOW.md), and [WORKING_NOTES.md](./WORKING_NOTES.md) for the supporting detail.)

## Definition of Done checklist

Against the seven Sprint 30 `PROJECT_PLAN.md` items:

| # | Item | Target | Landed | Verdict |
|---|------|--------|--------|---------|
| 1 | Warning baseline inventory | Capture clean warning baselines and classify them by source area and warning class | Days 1-2. Clean Apple Clang CMake and Makefile captures landed, with counts by area, class, and file in Sprint 30 artifacts | ✅ Complete |
| 2 | Core-library `INFINITY` / `NAN` cleanup | Remove the known library-proper compile-hygiene warnings in `src/sparse_lu.c`, `src/sparse_ldlt.c`, `src/sparse_qr.c`, and `src/sparse_svd.c` without changing semantics | Days 3-5. Audit Day 3 (`a97b6bf`), implementation batches Day 4 (`e4be19c`) and Day 5 (`ebeb6b2`) removed the full `src/` warning cluster | ✅ Complete |
| 3 | Compile-hygiene playbook | Define immediate-fix versus backlog-acceptable warning classes for later Epic 3 work | Day 6 (`448912f`) produced the playbook and Day 12/14 aligned it to the measured sprint-close state | ✅ Complete |
| 4 | Rebuild loop automation | Make warning reproduction and validation easy to rerun locally | Day 7 (`2dd997b`) added the repeatable `warning-workflow` entry point and stable artifact capture locations | ✅ Complete |
| 5 | Cross-compiler reproduction pass | Distinguish portable warning debt from path-specific noise using Apple Clang CMake, Makefile, and a stricter compile-only pass | Days 8-9 (`179d66e`, `df86ef4`) confirmed all residual debt is CMake full-tree debt, not shared with the Makefile `all` path; stricter pass found one real new `src/` issue and it was fixed same day | ✅ Complete |
| 6 | Initial non-library triage | Turn the residual warning mass in `tests/`, `benchmarks`, and `examples` into a real follow-up queue | Days 10-11 (`0d742d8`, `f6e6957`) converted the auxiliary warning debt into explicit test and tooling cleanup queues | ✅ Complete |
| 7 | Validation & sprint notes | Re-run full validation and record before/after warning counts in durable sprint notes | Days 12-14 (`18612b2`, `b777935`, `c382d9c`) reconciled the counts, finalized the validation checklist, added handoff inputs, and closed the sprint | ✅ Complete |

Headline gates from `PROJECT_PLAN.md` Sprint 30:

| gate | result |
|---|---|
| Authoritative Apple Clang CMake baseline captured | ✅ PASS (`123` warnings on Day 1 baseline; reproducible through workflow artifacts) |
| Core-library warnings removed | ✅ PASS (`src`: `11 -> 0`) |
| Makefile `all` cross-check remains clean | ✅ PASS (`0 -> 0`) |
| Warning workflow automation added | ✅ PASS (`make warning-workflow WARNING_WORKFLOW_LABEL=<label>`) |
| Stricter compile sweep reconciled | ✅ PASS (one `src/sparse_types.c` missing-prototype issue found and fixed on Day 9) |
| Remaining non-library debt classified into actionable queues | ✅ PASS (Days 10-11 artifacts + Day 14 handoff) |
| Final validation clean | ✅ PASS (`make format`, `make lint`, `make test`; workflow `ctest` `52/52`) |

## Final metrics

### Authoritative Apple Clang CMake full-tree path

| Metric | Day 1 baseline | Day 13 final | Delta |
|---|---:|---:|---:|
| Full-tree warnings | 123 | 112 | -11 |
| `src` warnings | 11 | 0 | -11 |
| `tests` warnings | 98 | 98 | 0 |
| `benchmarks` warnings | 13 | 13 | 0 |
| `examples` warnings | 1 | 1 | 0 |

### Warning classes

| Warning class | Day 1 baseline | Day 13 final | Delta |
|---|---:|---:|---:|
| `-Wmissing-field-initializers` | 72 | 72 | 0 |
| `-Wdouble-promotion` | 45 | 34 | -11 |
| `-Wunused-function` | 3 | 3 | 0 |
| `-Wimplicit-function-declaration` | 2 | 2 | 0 |
| `-Wswitch` | 1 | 1 | 0 |

### Secondary Makefile `all` path

| Metric | Day 1 baseline | Day 13 final |
|---|---:|---:|
| Warnings | 0 | 0 |

### Validation state at sprint close

| gate | result |
|---|---|
| `make warning-workflow WARNING_WORKFLOW_LABEL=day13-final` | ✅ PASS |
| Workflow `ctest` | ✅ PASS (`52/52` in `162.13 sec`) |
| `make format` | ✅ PASS |
| `make lint` | ✅ PASS |
| `make test` | ✅ PASS |

## Performance and quality highlights

### The sprint removed all library-proper warning debt without changing public behavior

The whole Sprint 30 reduction came from the library proper. Days 4-5
replaced implementation-side `INFINITY` uses with `HUGE_VAL` in
`src/sparse_lu.c:478`, `src/sparse_ldlt.c:1406`,
`src/sparse_qr.c:1080`, and `src/sparse_svd.c:1687`.
This was a narrow compile-hygiene fix, not a numerical-method change.
The warning-class delta proves it: `-Wdouble-promotion` dropped `45 ->
34`, and every removed warning came from `src/`.

### The Makefile path stayed useful, but only as a narrow cross-check

Sprint 30 confirmed that the repository has two materially different
warning surfaces:

1. Apple Clang CMake is the authoritative full-tree inventory.
2. Makefile `all` is a library-only build and therefore a secondary
   hygiene signal, not a repository-wide cleanliness claim.

That distinction mattered repeatedly. Without it, the repository could
have been mischaracterized as warning-clean from the Makefile path
alone.

### The stricter pass found one real defect in repository hygiene

Day 9 proved the value of a stricter compile-only sweep. Adding
`-Wstrict-prototypes` and `-Wmissing-prototypes` did not explode into
noise; it revealed one genuine `src/` issue in
`src/sparse_types.c:5`, where `sparse_set_errno_` lacked a prior visible
declaration. Fixing it
the same day let the strict-tree pass fall back to the same class/area
profile as the normal Day 8 baseline.

### The sprint ended with a usable backlog, not just a lower count

Days 10-11 changed the shape of the remaining work. The residual `112`
warnings are now queued concretely:

- `tests`: initializer drift, dormant `-Wunused-function` scaffolding,
  and mechanical `-Wdouble-promotion`
- `benchmarks`: stale reorder CLI drift, `snprintf` portability debt,
  and designated-initializer cleanup
- `examples`: public-facing positional initializer cleanup

That handoff matters more than the raw count reduction because Sprint 31
can now start from a measured, ranked queue instead of re-discovering
the warning inventory.

## What went well

- **The sprint stayed scoped to hardening work.** No feature expansion leaked into the branch. Every artifact serves baseline capture, cleanup, workflow, policy, or queue definition.
- **The library cleanup was measurably narrow.** The warning reduction is easy to explain: `src` `11 -> 0`; full-tree `123 -> 112`; class delta only in `-Wdouble-promotion`.
- **Workflow automation paid off during the sprint itself.** The Day 7 `warning-workflow` target and serialized warning capture made later comparisons reproducible instead of noisy.
- **The stricter compile sweep was worth doing early.** It found one real issue while the sprint context was still warm and closed that gap immediately.
- **The backlog is now sequenced, not vague.** Days 10-11 and the Day 14 handoff turn the remaining debt into a practical Sprint 31 / Sprint 32 plan.

## What didn’t go well

- **The Makefile path could easily mislead reviewers.** Because it omits tests, benchmarks, and examples, it stayed at `0` warnings throughout the sprint and needed repeated explanation to avoid false cleanliness claims.
- **Auxiliary warning volume remained high by design.** Sprint 30 was the right place to do baseline, core cleanup, and workflow work first, but that means the repository still closes the sprint with `112` full-tree warnings.
- **Git index-lock collisions happened during earlier day-end bookkeeping.** The issue was transient, but it reinforced that Git write steps should stay serialized rather than parallelized.

## Sprint 31 inputs

Concrete handoff items for Sprint 31 and the next Epic 3 cleanup steps:

1. **Benchmark tooling sync is the highest-signal next move.**
   [HANDOFF.md](./HANDOFF.md)
   ranks `benchmarks/bench_main.c` and `benchmarks/bench_convergence.c`
   first because they combine correctness drift and portability debt.
2. **Designated-initializer cleanup should follow immediately after.**
   The benchmark/example queue is explicit and mechanical enough to be a
   focused Sprint 31 batch.
3. **`tests/test_reorder_nd.c` is the first test-honesty target.**
   Sprint 32 should treat it as both a warning cleanup and a dormant
   scaffolding audit, not merely a count-reduction exercise.
4. **Sprint 30’s policy and workflow should remain the source of truth.**
   Future cleanup claims should continue to cite Apple Clang CMake as the
   authoritative full-tree path and use the warning workflow artifacts as
   evidence.

## Day-by-day capsule

| Day | Theme | Hours | Outcome |
|---:|---|---:|---|
| 1 | Baseline build capture | 8 | Clean CMake and Makefile warning baselines captured; raw counts recorded |
| 2 | Warning taxonomy & ownership map | 8 | Warning stream classified by area and class; core vs auxiliary debt separated |
| 3 | Core-library hotspot audit | 10 | Core warning cluster audited; `HUGE_VAL` idiom chosen before edits |
| 4 | Core fixes batch 1 | 10 | `src/sparse_lu.c` and `src/sparse_ldlt.c` warning sites fixed |
| 5 | Core fixes batch 2 | 8 | `src/sparse_qr.c` and `src/sparse_svd.c` warning sites fixed; `src` warnings reached `0` |
| 6 | Compile-hygiene playbook | 8 | Area-specific warning policy documented |
| 7 | Rebuild workflow automation | 10 | `warning-workflow` target and capture script added |
| 8 | Cross-path warning analysis | 8 | Apple Clang CMake vs Makefile differences measured and documented |
| 9 | Stricter compile sweep | 10 | `src/sparse_types.c` prototype hygiene issue found and fixed |
| 10 | Test warning triage | 8 | `tests` warning debt classified into concrete first-fix queues |
| 11 | Tooling warning triage | 8 | `benchmarks` / `examples` warning debt classified into concrete first-fix queues |
| 12 | Baseline reconciliation | 8 | Before/after counts reconciled; validation checklist finalized |
| 13 | Final validation pass | 8 | Workflow rerun matched reconciled state; final warning counts confirmed |
| 14 | Sprint closeout & handoff | 8 | Retrospective, handoff, and closeout documentation completed |

**Total: 120 hours** (matched the Sprint 30 plan estimate exactly).

## Day-budget vs estimate

| Item | Estimate | Actual | Notes |
|---|---:|---:|---|
| 1. Warning baseline inventory | 16 hrs | 16 hrs | Days 1-2 |
| 2. Core-library `INFINITY` / `NAN` cleanup | 24 hrs | 28 hrs | Days 3-5; audit-first approach took slightly longer but fully removed `src` warnings |
| 3. Compile-hygiene playbook | 8 hrs | 8 hrs | Day 6 |
| 4. Rebuild loop automation | 16 hrs | 18 hrs | Days 7-8; extra time went into stabilizing serialized warning capture |
| 5. Cross-compiler reproduction pass | 20 hrs | 18 hrs | Days 8-9; stricter pass stayed narrow and efficient |
| 6. Initial non-library triage | 16 hrs | 16 hrs | Days 10-11 |
| 7. Validation & sprint notes | 20 hrs | 16 hrs | Days 12-14; validation and closeout stayed within plan |

Total estimate 120 hrs; total actual 120 hrs. Two items ran a little
over their per-item estimates, but the cross-compiler pass and the final
validation/notes package came in under enough to keep the sprint on its
overall budget.

## DoD verification

Final quality gates at sprint close:

| gate | result |
|---|---|
| `make format` | ✅ clean |
| `make lint` | ✅ clean |
| `make test` | ✅ all tests pass |
| `make warning-workflow WARNING_WORKFLOW_LABEL=day13-final` | ✅ clean reproduction of the sprint-close warning state |
| workflow `ctest` | ✅ `52/52` passed in `162.13 sec` |

## Acknowledgements

Sprint 30 did not make the repository fully warning-clean, but it did
the harder prerequisite work correctly: it replaced anecdotal warning
complaints with a reproducible baseline, eliminated the entire
library-proper warning cluster, established a written compile-hygiene
policy, and left the remaining debt in a form that later sprints can
actually execute against. That is the right opening move for Epic 3.
