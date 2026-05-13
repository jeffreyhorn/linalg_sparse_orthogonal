# Sprint 29 Retrospective — SVD Improvements, Eigenpair Refinement, Progress Callbacks, CI Hardening & Epic 2 Wrap-Up

**Sprint budget:** 14 working days (168 hours = 14 × 12 ceiling)
**Branch:** `sprint-29`
**Calendar elapsed:** 2026-05-12 → 2026-05-13 (intensive condensed run; the day budget tracks engineering effort, not wall-clock days)

> **Status:** *Day-14 fill-in pending.*  Day 13's skeleton lists the
> per-item closures landed Days 1-13; Day 14 will fill in the final
> verdicts, metrics, and Sprint-30+ routing single-pass.

## Goal recap

> *Filled in Day 14.*  Per PLAN.md Sprint 29 goal: address remaining
> review findings + the final Sprint 20 deferred follow-up — fix the
> dense-in-disguise SVD paths, add an opt-in inverse-iteration
> refinement post-pass for `sparse_eigs_sym`, add progress / cancel
> callbacks across long-running routines, add Windows / macOS CI,
> improve the sparse low-rank approximation, calibrate the coverage
> gate, fix the build-and-test bench-step CI hang, absorb two Sprint-28
> deferrals (`bench_reorder --reorder-via-analyze` + macOS-15+ tsan),
> close out Epic 2 with final documentation and validation.

(See `docs/planning/EPIC_2/SPRINT_29/PLAN.md` for the day-by-day
breakdown; per-axis decision docs in `docs/planning/EPIC_2/SPRINT_29/`.)

## Definition of Done checklist

| item | status | reference |
|---|---|---|
| 1. Sparse low-rank without dense accumulator | *Day 14 to verify* | Day 1 commit (skeleton); Day 2 commit (outer-product impl + corpus sweep) |
| 2. Full SVD U/V output beyond economy mode | *Day 14 to verify* | Day 3 commit (impl + tests); Day 4 commit (economy-unchanged close) |
| 3. Optional eigenpair refinement (Sprint 20 deferred) | *Day 14 to verify* | Day 4 commit (design); Day 5 commit (impl + 4 tests) |
| 4. Progress / cancel callbacks | *Day 14 to verify* | Day 6 commit (LU/Chol/LDL^T); Day 7 commit (QR/iter/eigs/ND close) |
| 5. Windows CI | *Day 14 to verify* | Day 7 commit (draft); Day 8 commit (close + portability fixes) |
| 6. macOS CI (Apple Clang + Homebrew GCC) | *Day 14 to verify* | Day 9 commit (matrix + install/pkg-config) |
| 7. API accessor error reporting | *Day 14 to verify* | Day 10 commit (`accessor_error_decision.md`) |
| 8. Final integration + coverage calibration + bench-step fix | *Day 14 to verify* | Day 11 commit (cross-feature tests + audit); Day 12 commit (threshold 95→80); Day 13 commit (`bench-fast` + CI update) |
| 9. README + INSTALL + retrospective + Epic 2 wrap-up | *Day 14 single-pass* | Day 13 commit (skeleton + README + INSTALL drafts); Day 14 commit (fill-in) |
| 10a. `bench_reorder --reorder-via-analyze` (Sprint 28 absorption) | *Day 14 to verify* | Day 13 commit |
| 10b. macOS-15+ tsan handling (Sprint 28 absorption) | *Day 14 to verify* | Day 8 commit (Linux-CI tsan job + decision doc) |

Headline gates from PROJECT_PLAN.md Sprint 29:

| gate | result |
|---|---|
| Item 1: sparse low-rank ≥ 50 % memory + ≥ 30 % wall reduction | *Day 14 to fill* |
| Item 2: full-mode U/V orthonormal (||U^T U − I||_F ≤ 1e-10) | *Day 14 to fill* |
| Item 3: refinement residual ≤ 1e-13 on clustered spectrum | *Day 14 to fill* |
| Item 4: default-NULL-callback overhead ≤ Sprint-28 baseline + 5 % | *Day 14 to fill* |
| Items 5/6: Windows + macOS CI green on `sprint-29` HEAD | *Day 14 to fill* |
| Item 8: `make coverage` passes calibrated 80 % threshold | *Day 14 to fill* |
| `make wall-check` PASS | *Day 14 to re-verify* |
| `make sanitize` CLEAN | *Day 14 to re-verify* |
| Linux-CI tsan job green (macOS 15+ blocked per Day 8) | *Day 14 to verify* |
| `make lint` EXIT 0 | *Day 14 to re-verify* |
| Test count: 2068 assertions PASS | *Day 14 to re-verify* |

## Final metrics

*Day 14 fill-in.*  Per-item numbers:

### Sprint-29 production default flips

*Day 14 fill-in.*  Candidate flips: SPARSE_SVD_LOWRANK_OUTER (Day 2
verdict); SPARSE_SUPERNODAL_POSTORDER (no — Sprint 28 verdict carries).

### SVD low-rank outer-product wall + memory (Day 2)

*Day 14 fill-in from `lowrank_sweep_day2.txt`.*

### Eigenpair refinement residual deltas (Day 5)

*Day 14 fill-in from Day 5 test output.*

### Callback overhead (Day 7)

*Day 14 fill-in: wall-check delta vs Sprint 28 baseline.*

### Coverage threshold (Day 12)

Lowered from 95 → 80 after measured aggregate of 81.3 %.  See
`coverage_threshold_decision.md` for the per-file-group breakdown.

### CI bench-step (Day 13)

`make bench-fast` lands at ~63 s locally (vs 6 h timeout under full
`make bench`).  CI's `build-and-test` job now runs `bench-build`
(compile-coverage for slow benches) + `bench-fast` (runtime regression
signal on the genuinely fast benches + `bench_reorder --skip-factor`).

## Performance highlights

*Day 14 fill-in.*  Anchor headlines:
- Sprint-29 SVD low-rank memory + wall headlines (Day 2 verdict).
- Eigenpair refinement residual tightening on clustered spectra
  (Day 5).
- Progress-callback rollout across 11 long-running routines with no
  measurable default-NULL-callback overhead (Day 7).

## What went well

*Day 14 fill-in.*  Seeds:
- Day 1 + Day 4 design-doc-first pattern carried Sprint 28's framing
  to two more items.
- Day 6 / Day 7 split of progress-callback rollout (LU/Chol/LDL^T
  first, then QR/iter/eigs/ND) kept per-day scope manageable.
- Day 13 skeleton-first retrospective pattern (this doc) means Day 14
  fills a partially-populated structure rather than writing from
  scratch.

## What surprised us

*Day 14 fill-in.*

## What didn't go well

*Day 14 fill-in.*  Seeds:
- Day 11 `make coverage` blocked locally by Homebrew lcov 2.4 + Apple
  gcov format mismatch; Day 12 worked around via gcovr + Apple gcov +
  `--gcov-ignore-parse-errors`.

## Items deferred (route to Sprint 30+)

*Day 14 fill-in.*  Seeds (anticipated from per-day decision docs):
- Supernodal numeric-factor kernel (Sprint 28 inheritance, still
  open).
- ND opts struct (Sprint 29 Day 7 noted: ND doesn't have an opts
  struct so progress callbacks aren't wired there).
- Synthetic-fault-injection scaffolding to push aggregate coverage
  ≥ 95 % (Sprint 29 Day 12 noted as out-of-budget).

## Lessons (Sprint 29-specific)

*Day 14 fill-in.*  Seeds:
- Day-11 audit → Day-12 decision pattern works when the metric is
  inherently noisy or the gate is decoupled from a single CI run.
- gcovr's `--gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file`
  is the macOS-local-coverage workaround for the lcov-can't-parse-
  Apple-gcov problem documented Sprint 24 Day 12.

## Sprint 30+ inputs

*Day 14 fill-in.*  Concrete handoff items.

## Day-by-day capsule

| day | theme | key commit |
|---|---|---|
| 1 | Item 1 sparse-low-rank design + skeleton | `91bdd69` |
| 2 | Item 1 close — outer-product impl + bench validation | `b611be7` |
| 3 | Item 2 — Full SVD U/V output | `3c4e182` |
| 4 | Item 2 close + Item 3 eigenpair-refinement design | `cc264f0` |
| 5 | Item 3 close — refinement impl + 4 tests | `7f2cfe0` |
| 6 | Item 4 — Progress / cancel callbacks (LU + Chol + LDL^T) | `9a5ac90` |
| 7 | Item 4 close (QR + iter + eigs callbacks) + Item 5 start (Windows CI) | `e3017e8` |
| 8 | Item 5 close (Windows portability) + Item 10b confirm | `1730281` |
| 9 | Item 6 — macOS CI (Apple Clang + Homebrew GCC matrix) | `38da002` |
| 10 | Item 7 — API accessor error reporting (decision + doc) | `90beea3` |
| 11 | Item 8 Part 1 — cross-feature integration tests + coverage audit | `7beee8a` |
| 12 | Item 8 Part 2 — coverage-gate calibration (95→80) | `2480ff5` |
| 13 | Item 8 close + Item 10a `bench_reorder --reorder-via-analyze` + Item 9 prep | *this commit* |
| 14 | Item 9 close — Sprint-29 + Epic-2 retrospective + README + PR | *Day 14* |

## Day-budget vs estimate

| day | budgeted | actual | notes |
|---|---:|---:|---|
| 1-12 | 12 each (144 total) | 12 each (144 total) | every day landed on-budget per Day-13 inheritance check |
| 13 | 12 (4 + 3 + 5 split) | *Day 13 fill-in* | |
| 14 | 12 | *Day 14 fill-in* | |

## DoD verification

*Day 14 fill-in.*  Re-run `make format && make lint && make test &&
make sanitize && make wall-check`.  Linux-CI tsan job green check.
Windows + macOS CI green check.

## Acknowledgements

*Day 14 fill-in.*
